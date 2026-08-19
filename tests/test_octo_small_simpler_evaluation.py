import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _statistics_file(tmp_path: Path) -> Path:
    from octo_small_bridge.normalization import BridgeV2NormalizationStatistics

    path = tmp_path / "normalization.json"
    BridgeV2NormalizationStatistics(
        state_q01=np.asarray([-1, -2, -3, -4, -5, -6, 0, 0], dtype=np.float32),
        state_q99=np.asarray([1, 2, 3, 4, 5, 6, 0, 1], dtype=np.float32),
        action_q01=np.asarray([-1, -2, -3, -4, -5, -6, 0], dtype=np.float32),
        action_q99=np.asarray([1, 2, 3, 4, 5, 6, 1], dtype=np.float32),
        metadata_sha256="a" * 64,
        retained_episodes=2,
        retained_frames=8,
    ).save(path)
    return path


class _Tokenizer:
    def __call__(self, instructions, **kwargs):
        assert instructions == ["Put Spoon on Towel"]
        assert kwargs == {
            "padding": "max_length",
            "truncation": True,
            "max_length": 16,
            "return_tensors": "np",
        }
        return {
            "input_ids": np.arange(16, dtype=np.int64)[None],
            "attention_mask": np.ones((1, 16), dtype=np.int64),
        }


def _v2_statistics_file(tmp_path: Path) -> Path:
    return _statistics_file(tmp_path)


def test_octo_bridge_policy_uses_v2_quantiles_and_binary_gripper(tmp_path):
    from octo_small_bridge.simpler_evaluation import (
        OctoBridgeSimplerPolicy,
        load_bridge_statistics,
    )

    statistics = load_bridge_statistics(_v2_statistics_file(tmp_path))
    normalized_actions = np.asarray(
        [[[2.2, -2.2, 0, 0, 0, 0, 0.50001]] * 8], dtype=np.float32
    )
    policy = OctoBridgeSimplerPolicy(
        model=object(),
        tokenizer=_Tokenizer(),
        statistics=statistics,
        device="cpu",
        precision="fp32",
        sampler=lambda prepared, generator: normalized_actions,
    )

    prepared = policy.prepare_observation(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.asarray([4, -8, 0, 0, 0, 0, 99, 0.5], dtype=np.float32),
        "Put Spoon on Towel",
    )
    actions = policy.predict_actions(prepared, generator=None)

    np.testing.assert_array_equal(
        prepared["proprio"][0, 0],
        np.asarray([2.2, -2.2, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        actions[0, 0, :6],
        np.asarray([2.2, -4.4, 0, 0, 0, 0], dtype=np.float32),
    )
    assert actions[0, 0, 6] == 1.0
    assert policy.gripper_threshold == 0.5


def test_old_octo_bridge_checkpoint_is_rejected_before_model_loading(tmp_path):
    from octo_small_bridge.simpler_evaluation import load_octo_bridge_policy
    from simpler_bridge.evaluation import SimplerEvaluationError

    base_model = _self_contained_base_model(tmp_path)
    checkpoint = tmp_path / "old-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"old")
    calls = []

    with pytest.raises(SimplerEvaluationError, match="checkpoint_manifest"):
        load_octo_bridge_policy(
            checkpoint,
            base_model=base_model,
            statistics=_v2_statistics_file(tmp_path),
            device="cpu",
            precision="fp32",
            model_loader=lambda *args, **kwargs: calls.append("model"),
            weight_loader=lambda *args, **kwargs: calls.append("weights"),
            torch_module=SimpleNamespace(
                device=lambda value: SimpleNamespace(type="cpu")
            ),
        )

    assert calls == []


def test_bridge_statistics_and_octo_adapter_preserve_gripper_semantics(tmp_path):
    from octo_small_bridge.simpler_evaluation import (
        OctoBridgeSimplerPolicy,
        load_bridge_statistics,
    )

    statistics = load_bridge_statistics(_statistics_file(tmp_path))
    proprio = statistics.normalize_proprio(
        np.asarray([1, 2, 3, 4, 5, 6, 0, 1], dtype=np.float32)
    )
    np.testing.assert_allclose(
        proprio,
        np.asarray([1, 1, 1, 1, 1, 1, 0, 1], dtype=np.float32),
    )

    normalized_actions = np.zeros((1, 8, 7), dtype=np.float32)
    normalized_actions[..., 6] = 0.75
    policy = OctoBridgeSimplerPolicy(
        model=object(),
        tokenizer=_Tokenizer(),
        statistics=statistics,
        device="cpu",
        precision="fp32",
        generator_factory=lambda device, seed: f"{device}-{seed}",
        sampler=lambda prepared, generator: normalized_actions,
    )
    prepared = policy.prepare_observation(
        np.zeros((12, 20, 3), dtype=np.uint8),
        np.asarray([1, 2, 3, 4, 5, 6, 0, 1], dtype=np.float32),
        "Put Spoon on Towel",
    )

    assert policy.make_generator(4) == "cpu-4"
    assert set(prepared) == {
        "image_primary",
        "proprio",
        "language_input_ids",
        "language_attention_mask",
    }
    assert prepared["image_primary"].shape == (1, 1, 3, 256, 256)
    assert prepared["proprio"].shape == (1, 1, 8)
    assert "image_wrist" not in prepared
    assert policy.describe_observation(prepared) == {
        "model_image_shape": [1, 1, 3, 256, 256],
        "model_proprio_shape": [1, 1, 8],
        "observation_tokenizers": ["primary"],
    }

    actions = policy.predict_actions(prepared, generator="cpu-4")

    np.testing.assert_allclose(
        actions[..., :6],
        np.zeros((1, 8, 6), dtype=np.float32),
    )
    np.testing.assert_allclose(actions[..., 6], 1.0)
    assert policy.protocol_metadata()["precision"] == "fp32"


def test_bridge_statistics_reject_invalid_shapes_and_reversed_quantiles(tmp_path):
    from simpler_bridge.evaluation import SimplerEvaluationError
    from octo_small_bridge.simpler_evaluation import load_bridge_statistics

    path = _statistics_file(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["action_q01"][0] = 2
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(SimplerEvaluationError, match="must not exceed"):
        load_bridge_statistics(path)

    value["action_q01"][0] = -1
    value["state_q01"] = [0, 1]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SimplerEvaluationError, match=r"shape \(8,\)"):
        load_bridge_statistics(path)


def test_bridge_statistics_keep_constant_dimensions_at_zero(tmp_path):
    from octo_small_bridge.simpler_evaluation import load_bridge_statistics

    path = _statistics_file(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["state_q01"][0] = 1
    value["state_q99"][0] = 1
    path.write_text(json.dumps(value), encoding="utf-8")

    statistics = load_bridge_statistics(path)

    normalized = statistics.normalize_proprio(
        np.asarray([9, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    )
    assert normalized[0] == 0.0


def _self_contained_base_model(tmp_path: Path) -> Path:
    root = tmp_path / "base-model"
    text = root / "text_encoder"
    text.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"base")
    (root / "model_config.json").write_text(
        json.dumps(
            {
                "action_dim": 7,
                "action_horizon": 8,
                "proprio_dim": 8,
                "language_tokens": 16,
                "diffusion_steps": 20,
            }
        ),
        encoding="utf-8",
    )
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (text / name).write_text("{}", encoding="utf-8")
    (text / "spiece.model").write_bytes(b"tokenizer")
    return root


def test_policy_loader_uses_primary_only_and_strict_checkpoint_weights(tmp_path):
    from octo_small_bridge.checkpoint_contract import BridgeCheckpointContract
    from octo_small_bridge.simpler_evaluation import load_octo_bridge_policy

    base_model = _self_contained_base_model(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "step-00020000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"fine-tuned")
    statistics_path = _statistics_file(tmp_path)
    BridgeCheckpointContract(statistics_path, "d" * 64).write(checkpoint)
    calls = {}

    class Model:
        config = SimpleNamespace(
            action_dim=7,
            action_horizon=8,
            proprio_dim=8,
            language_tokens=16,
            diffusion_steps=20,
        )

        def to(self, device):
            calls["device"] = str(device)
            return self

        def eval(self):
            calls["eval"] = True
            return self

    model = Model()

    def model_loader(path, **kwargs):
        calls["model_loader"] = (Path(path), kwargs)
        return model, _Tokenizer()

    def weight_loader(loaded_model, path, **kwargs):
        calls["weight_loader"] = (loaded_model, Path(path), kwargs)

    torch_module = SimpleNamespace(
        device=lambda value: SimpleNamespace(type="cuda", __str__=lambda self: value),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
        ),
    )

    spec, policy = load_octo_bridge_policy(
        checkpoint,
        base_model=base_model,
        device="cuda:0",
        precision="bf16",
        model_loader=model_loader,
        weight_loader=weight_loader,
        torch_module=torch_module,
    )

    assert spec.requested_path == checkpoint.resolve()
    assert calls["model_loader"] == (
        base_model.resolve(),
        {"device": "cpu", "observation_tokenizers": ("primary",)},
    )
    assert calls["weight_loader"] == (
        model,
        checkpoint.resolve() / "model.safetensors",
        {"strict": True, "device": "cpu"},
    )
    assert calls["eval"] is True
    assert policy.statistics.path == checkpoint.resolve() / "normalization.json"


def test_octo_simpler_runtime_contract_is_python310_and_model_specific():
    from octo_small_bridge.simpler_evaluation import (
        OCTO_SIMPLER_RUNTIME_PACKAGE_VERSIONS,
        validate_runtime_contract,
    )

    versions = dict(OCTO_SIMPLER_RUNTIME_PACKAGE_VERSIONS)
    assert versions["torch"] == "2.4.1"
    assert versions["transformers"] == "4.44.2"
    assert versions["sapien"] == "2.2.2"
    assert versions["setuptools"] == "75.8.0"
    assert validate_runtime_contract(
        version_info=(3, 10),
        package_versions=versions,
        device="cpu",
    ) == versions

    cuda_versions = {
        **versions,
        "torch": "2.4.1+cu121",
        "torchvision": "0.19.1+cu121",
    }
    assert validate_runtime_contract(
        version_info=(3, 10),
        package_versions=cuda_versions,
        device="cpu",
    ) == cuda_versions

    incompatible_versions = {**cuda_versions, "torch": "2.4.1.post1+cu121"}
    with pytest.raises(Exception, match="requires torch==2.4.1"):
        validate_runtime_contract(
            version_info=(3, 10),
            package_versions=incompatible_versions,
            device="cpu",
        )

    try:
        validate_runtime_contract(
            version_info=(3, 12),
            package_versions=versions,
            device="cpu",
        )
    except Exception as error:
        assert "Python 3.10 or 3.11" in str(error)
    else:
        raise AssertionError("Python 3.12 must be rejected")


def test_octo_simpler_cli_requires_all_model_paths_and_defaults_to_full_protocol():
    from octo_small_bridge.evaluate_simpler import build_parser

    parser = build_parser()
    arguments = parser.parse_args(
        [
            "--checkpoint",
            "/models/step-00020000",
            "--base-model",
            "/models/octo-small-pytorch",
            "--statistics",
            "/data/bridge/meta/stats.json",
        ]
    )

    assert arguments.tasks == "all"
    assert arguments.output_dir == Path("outputs/octo_small_bridge_simpler_eval")
    assert arguments.device == "cuda:0"
    assert arguments.sim_device == "cuda:0"
    assert arguments.precision == "bf16"
    assert arguments.action_horizon == 1
    self_contained = parser.parse_args(
        [
            "--checkpoint",
            "/models/step-00020000",
            "--base-model",
            "/models/octo-small-pytorch",
        ]
    )
    assert self_contained.statistics is None
    assert parser.parse_args(
        [
            "--checkpoint",
            "/models/step-00020000",
            "--base-model",
            "/models/octo-small-pytorch",
            "--statistics",
            "/data/bridge/meta/stats.json",
            "--sim-device",
            "cuda:12",
        ]
    ).sim_device == "cuda:12"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                "/models/step-00020000",
                "--base-model",
                "/models/octo-small-pytorch",
                "--statistics",
                "/data/bridge/meta/stats.json",
                "--action-horizon",
                "8",
            ]
        )
    assert arguments.smoke_test is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "/models/step-00020000"])


@pytest.mark.parametrize(
    "sim_device",
    ("cuda:-1", "cuda", "3", "/dev/nvidia3", "cpu", "cuda:+1", "cuda: 1"),
)
def test_octo_simpler_cli_rejects_non_logical_cuda_sim_devices(sim_device):
    from octo_small_bridge.evaluate_simpler import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--checkpoint",
                "/models/step-00020000",
                "--base-model",
                "/models/octo-small-pytorch",
                "--statistics",
                "/data/bridge/meta/stats.json",
                "--sim-device",
                sim_device,
            ]
        )


def test_octo_simpler_cli_applies_smoke_protocol_and_model_metadata(tmp_path, monkeypatch):
    from octo_small_bridge import evaluate_simpler

    captured = {}
    checkpoint = SimpleNamespace(
        as_dict=lambda: {"requested_path": "/models/step-00020000"}
    )
    statistics = SimpleNamespace(
        as_dict=lambda: {"path": "/data/bridge/meta/stats.json", "sha256": "abc"}
    )
    policy = SimpleNamespace(statistics=statistics, protocol_metadata=lambda: {})
    monkeypatch.setattr(
        evaluate_simpler,
        "validate_simpler_source",
        lambda path: {"simpler_env_commit": "06accaca9353"},
    )
    monkeypatch.setattr(
        evaluate_simpler,
        "validate_runtime_contract",
        lambda device: {"torch": "2.4.1"},
    )
    monkeypatch.setattr(
        evaluate_simpler,
        "load_octo_bridge_policy",
        lambda *args, **kwargs: (checkpoint, policy),
    )

    def evaluate(settings, **kwargs):
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return {"status": "complete"}

    monkeypatch.setattr(evaluate_simpler, "evaluate_simpler_policy", evaluate)

    status = evaluate_simpler.main(
        [
            "--checkpoint",
            "/models/step-00020000",
            "--base-model",
            "/models/octo-small-pytorch",
            "--statistics",
            "/data/bridge/meta/stats.json",
            "--tasks",
            "eggplant,spoon",
            "--output-dir",
            str(tmp_path / "results"),
            "--smoke-test",
        ]
    )

    assert status == 0
    assert [task.key for task in captured["settings"].tasks] == ["eggplant", "spoon"]
    assert captured["settings"].policy_seeds == (0,)
    assert captured["settings"].object_episode_ids == (0,)
    assert captured["settings"].max_steps == 8
    assert captured["kwargs"]["checkpoint"] == {
        "requested_path": "/models/step-00020000",
        "statistics": {"path": "/data/bridge/meta/stats.json", "sha256": "abc"},
    }
    assert captured["kwargs"]["route"] == "octo-small-bridge-simpler-widowx-eval"


@pytest.mark.parametrize("preflight_only", (False, True), ids=("evaluation", "preflight"))
def test_octo_cli_forwards_sim_device_to_environment_builder(
    tmp_path, monkeypatch, preflight_only
):
    from octo_small_bridge import evaluate_simpler

    captured = {}
    checkpoint = SimpleNamespace(as_dict=lambda: {})
    policy = SimpleNamespace(
        statistics=SimpleNamespace(as_dict=lambda: {}),
        protocol_metadata=lambda: {},
    )
    monkeypatch.setattr(evaluate_simpler, "validate_simpler_source", lambda path: {})
    monkeypatch.setattr(
        evaluate_simpler, "validate_runtime_contract", lambda device: {}
    )
    monkeypatch.setattr(
        evaluate_simpler,
        "load_octo_bridge_policy",
        lambda *args, **kwargs: (checkpoint, policy),
    )
    monkeypatch.setattr(
        evaluate_simpler,
        "create_simpler_environment",
        lambda task, *, sim_device: captured.setdefault(
            "builder", (task.key, sim_device)
        ),
    )

    def run(settings, **kwargs):
        captured["settings"] = settings
        kwargs["environment_factory"](settings.tasks[0])
        return {"status": "complete"}

    monkeypatch.setattr(evaluate_simpler, "evaluate_simpler_policy", run)
    monkeypatch.setattr(evaluate_simpler, "run_simpler_preflight", run)
    arguments = [
        "--checkpoint",
        "/models/step-00020000",
        "--base-model",
        "/models/octo-small-pytorch",
        "--statistics",
        "/data/bridge/meta/stats.json",
        "--tasks",
        "spoon",
        "--output-dir",
        str(tmp_path / "output"),
        "--sim-device",
        "cuda:7",
    ]
    if preflight_only:
        arguments.append("--preflight-only")

    status = evaluate_simpler.main(arguments)

    assert status == 0
    assert captured["settings"].sim_device == "cuda:7"
    assert captured["builder"] == ("spoon", "cuda:7")


def test_octo_simpler_cli_writes_contract_failures(tmp_path, monkeypatch):
    from octo_small_bridge import evaluate_simpler
    from simpler_bridge.evaluation import SimplerEvaluationError

    monkeypatch.setattr(
        evaluate_simpler,
        "validate_simpler_source",
        lambda path: (_ for _ in ()).throw(SimplerEvaluationError("source mismatch")),
    )
    output_dir = tmp_path / "output"

    status = evaluate_simpler.main(
        [
            "--checkpoint",
            "/models/step-00020000",
            "--base-model",
            "/models/octo-small-pytorch",
            "--statistics",
            "/data/bridge/meta/stats.json",
            "--output-dir",
            str(output_dir),
        ]
    )

    failure = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert status == 2
    assert failure["exit_code"] == 2
    assert failure["error"] == "SimplerEvaluationError: source mismatch"
