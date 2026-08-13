import importlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


EXPECTED_EVALUATION_SEEDS = (3471197683, 1232873419, 1448008435)


def _evaluation():
    return importlib.import_module("qwen3_vl_groot.libero_evaluation")


def _write_checkpoint(
    root: Path,
    *,
    backbone_family: str = "qwen3_vl",
) -> tuple[Path, Path]:
    if backbone_family == "qwen3_vl":
        architecture = "Qwen3VLForConditionalGeneration"
        text_config = {"num_hidden_layers": 36, "hidden_size": 2560}
        model_config = {}
    elif backbone_family == "qwen3_5":
        architecture = "Qwen3_5ForConditionalGeneration"
        text_config = {
            "num_hidden_layers": 24,
            "hidden_size": 1024,
            "layer_types": [
                layer_type
                for _ in range(6)
                for layer_type in (
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                )
            ],
        }
        model_config = {"backbone_family": "qwen3_5"}
    else:
        raise ValueError(backbone_family)
    base_model = root / "base-model"
    base_model.mkdir(parents=True)
    (base_model / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "text_config": text_config,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = root / "checkpoints" / "step-00020000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"compact-qwen-weights")
    (checkpoint / "normalization.json").write_text(
        json.dumps(
            {
                "state_q01": [0.0] * 8,
                "state_q99": [1.0] * 8,
                "action_q01": [0.0] * 7,
                "action_q99": [1.0] * 7,
                "epsilon": 1.0e-6,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "policy_config.json").write_text(
        json.dumps(
            {
                "format": "qwen3-vl-groot-bridge-compact-v1",
                "base_model": str(base_model),
                "global_step": 20_000,
                "config": {
                    "paths": {"lerobot": str(root / "LIBERO_lerobot")},
                    "data": {
                        "state_dim": 8,
                        "action_dim": 7,
                        "action_horizon": 8,
                        "train_crop_size": 115,
                        "output_image_size": 256,
                    },
                    "model": model_config,
                    "train": {},
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, base_model


def _observation(index: int = 0) -> dict[str, np.ndarray]:
    image = np.empty((128, 128, 3), dtype=np.uint8)
    image[:64] = 7 + index
    image[64:] = 19 + index
    return {
        "agentview_image": image,
        "robot0_eef_pos": np.asarray([index + 1, 2, 3], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.25, 0.75], dtype=np.float32),
    }


def test_resolve_qwen_checkpoint_requires_a_concrete_compatible_step(tmp_path):
    evaluation = _evaluation()
    checkpoint, base_model = _write_checkpoint(tmp_path)

    resolved = evaluation.resolve_qwen_checkpoint(checkpoint)

    assert resolved.requested_path == checkpoint.resolve()
    assert resolved.base_model_path == base_model.resolve()
    assert resolved.global_step == 20_000
    assert resolved.weights_sha256 == (
        "6b4d9b5e71081686fa2a44af4f65d52d23b6a8b921b402050f938cbdcb2972d0"
    )
    with pytest.raises(evaluation.EvaluationError, match="step-XXXXXXXX"):
        evaluation.resolve_qwen_checkpoint(checkpoint.parent.parent)


def test_resolve_qwen_checkpoint_honors_model_override_and_rejects_bad_contract(tmp_path):
    evaluation = _evaluation()
    checkpoint, _ = _write_checkpoint(tmp_path)
    override = tmp_path / "override"
    override.mkdir()
    (override / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config": {"num_hidden_layers": 36, "hidden_size": 2560},
            }
        ),
        encoding="utf-8",
    )

    resolved = evaluation.resolve_qwen_checkpoint(checkpoint, model_path=override)
    assert resolved.base_model_path == override.resolve()

    manifest_path = checkpoint / "policy_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["data"]["action_horizon"] = 16
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(evaluation.EvaluationError, match="action_horizon=8"):
        evaluation.resolve_qwen_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "config_text",
    [
        "{",
        "[]",
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config": {
                    "num_hidden_layers": 36,
                    "hidden_size": "not-an-integer",
                },
            }
        ),
    ],
)
def test_resolve_qwen_checkpoint_wraps_malformed_base_model_metadata(
    tmp_path,
    config_text,
):
    evaluation = _evaluation()
    checkpoint, base_model = _write_checkpoint(tmp_path)
    (base_model / "config.json").write_text(config_text, encoding="utf-8")

    with pytest.raises(evaluation.EvaluationError):
        evaluation.resolve_qwen_checkpoint(checkpoint)


def test_resolve_qwen35_checkpoint_and_reject_cross_family_override(tmp_path):
    evaluation = _evaluation()
    checkpoint, base_model = _write_checkpoint(tmp_path, backbone_family="qwen3_5")

    resolved = evaluation.resolve_qwen_checkpoint(checkpoint)

    assert resolved.base_model_path == base_model.resolve()
    assert resolved.config["model"]["backbone_family"] == "qwen3_5"

    override = tmp_path / "qwen3-vl-override"
    override.mkdir()
    (override / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config": {"num_hidden_layers": 36, "hidden_size": 2560},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(evaluation.EvaluationError, match="backbone family qwen3_5"):
        evaluation.resolve_qwen_checkpoint(checkpoint, model_path=override)


def test_qwen_observation_and_action_conversion_match_libero_training_contract():
    evaluation = _evaluation()

    images, states = evaluation.build_qwen_observation_batch(
        [_observation()],
        data_config={"train_crop_size": 115, "output_image_size": 256},
    )

    assert len(images) == 1
    assert images[0].shape == (256, 256, 3)
    np.testing.assert_array_equal(images[0][0, 0], [19, 19, 19])
    np.testing.assert_array_equal(images[0][-1, 0], [7, 7, 7])
    np.testing.assert_array_equal(states, [[1, 2, 3, 0, 0, 0, 0, 0.25]])

    predicted = np.zeros((1, 8, 7), dtype=np.float32)
    predicted[..., :6] = 0.125
    predicted[..., 6] = np.linspace(0.0, 1.0, 8)
    environment_actions = evaluation.qwen_actions_to_environment(predicted)
    np.testing.assert_allclose(environment_actions[..., :6], 0.125)
    np.testing.assert_allclose(
        environment_actions[0, :, 6],
        [1.0, 5 / 7, 3 / 7, 1 / 7, -1 / 7, -3 / 7, -5 / 7, -1.0],
        atol=1.0e-6,
    )


def test_qwen_policy_splits_large_environment_batches_and_preserves_order(tmp_path):
    evaluation = _evaluation()
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    checkpoint = evaluation.resolve_qwen_checkpoint(checkpoint_path)

    class FakeBridgePolicy:
        def __init__(self):
            self.batch_sizes = []
            self.generators = []

        def predict_actions(
            self,
            images,
            states,
            instructions,
            denoising_steps=4,
            *,
            generator=None,
        ):
            self.batch_sizes.append(len(images))
            self.generators.append(generator)
            actions = np.zeros((len(images), 8, 7), dtype=np.float32)
            actions[..., 0] = np.asarray(states)[:, 0, None]
            actions[..., 6] = 0.25
            return actions

    bridge = FakeBridgePolicy()
    policy = evaluation.QwenLiberoPolicy(
        checkpoint=checkpoint,
        policy=bridge,
        device="cpu",
        denoising_steps=6,
        policy_batch_size=2,
    )
    generator = object()

    actions = policy.predict_action_chunk(
        [_observation(index) for index in range(5)],
        "put the book away",
        generator=generator,
    )

    assert bridge.batch_sizes == [2, 2, 1]
    assert bridge.generators == [generator, generator, generator]
    np.testing.assert_array_equal(actions[:, 0, 0], [1, 2, 3, 4, 5])
    np.testing.assert_allclose(actions[..., 6], 0.5)


def test_qwen_policy_rejects_invalid_action_shapes(tmp_path):
    evaluation = _evaluation()
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    checkpoint = evaluation.resolve_qwen_checkpoint(checkpoint_path)

    class BrokenBridgePolicy:
        def predict_actions(self, *args, **kwargs):
            return np.zeros((1, 7, 7), dtype=np.float32)

    policy = evaluation.QwenLiberoPolicy(
        checkpoint=checkpoint,
        policy=BrokenBridgePolicy(),
        device="cpu",
        denoising_steps=4,
        policy_batch_size=1,
    )

    with pytest.raises(evaluation.EvaluationError, match=r"expected \(1, 8, 7\)"):
        policy.predict_action_chunk([_observation()], "put the book away", generator=object())


def test_qwen_policy_wraps_incompatible_compact_lora_errors(tmp_path, monkeypatch):
    evaluation = _evaluation()
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    checkpoint = evaluation.resolve_qwen_checkpoint(checkpoint_path)
    inference = importlib.import_module("qwen3_vl_groot.inference")

    def reject_checkpoint(*args, **kwargs):
        raise RuntimeError("compact parameter mismatch")

    monkeypatch.setattr(inference.BridgePolicy, "from_pretrained", reject_checkpoint)

    with pytest.raises(evaluation.EvaluationError, match="compact parameter mismatch"):
        evaluation.QwenLiberoPolicy.from_checkpoint(
            checkpoint,
            device="cpu",
            denoising_steps=4,
            policy_batch_size=1,
        )


def test_qwen_evaluation_cli_requires_checkpoint_and_keeps_protocol_defaults():
    evaluate = importlib.import_module("qwen3_vl_groot.evaluate")
    parser = evaluate.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    arguments = parser.parse_args(
        [
            "--checkpoint",
            "/tmp/checkpoints/step-00020000",
            "--model-path",
            "/tmp/Qwen3-VL-4B-Instruct",
        ]
    )

    assert arguments.checkpoint == "/tmp/checkpoints/step-00020000"
    assert arguments.model_path == "/tmp/Qwen3-VL-4B-Instruct"
    assert arguments.episodes == 150
    assert arguments.num_envs == 50
    assert arguments.max_steps == 960
    assert arguments.denoising_steps == 4
    assert arguments.policy_batch_size == 4


def test_qwen_evaluation_repeats_fixed_states_and_writes_qwen_report(tmp_path, monkeypatch):
    evaluation = _evaluation()
    checkpoint_path, _ = _write_checkpoint(tmp_path)
    checkpoint = evaluation.resolve_qwen_checkpoint(checkpoint_path)
    task = SimpleNamespace(
        task_id=5,
        name="STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        language="pick up the book and place it in the back compartment of the caddy",
        bddl_file=tmp_path / "task.bddl",
        init_states_file=tmp_path / "task.init",
        init_states_sha256="c" * 64,
        init_states_git_blob="d" * 40,
        init_states=np.zeros((2, 45), dtype=np.float64),
    )
    settings = evaluation.QwenEvaluationSettings(
        checkpoint=checkpoint.requested_path,
        output_dir=tmp_path / "results",
        episodes=6,
        num_envs=2,
        max_steps=1,
        settle_steps=0,
        device="cpu",
        policy_batch_size=1,
    )

    class FakeEnvironment:
        def __init__(self):
            self.batch_size = 0
            self.seeds = []

        def reset(self):
            return None

        def seed(self, seed):
            self.seeds.append(seed)

        def set_init_state(self, init_states):
            self.batch_size = len(init_states)
            return np.asarray(
                [_observation(index) for index in range(self.batch_size)],
                dtype=object,
            )

        def step(self, actions):
            observations = np.asarray(
                [_observation(index) for index in range(self.batch_size)],
                dtype=object,
            )
            return (
                observations,
                np.zeros(self.batch_size),
                np.zeros(self.batch_size, dtype=bool),
                np.asarray([{} for _ in range(self.batch_size)], dtype=object),
            )

        def check_success(self):
            return np.ones(self.batch_size, dtype=bool)

        def close(self):
            return None

    class FakePolicy:
        def __init__(self):
            self.generator_seeds = []

        def make_generator(self, seed):
            self.generator_seeds.append(seed)
            return seed

        def predict_action_chunk(self, observations, language, *, generator):
            assert language == task.language
            assert generator in EXPECTED_EVALUATION_SEEDS
            return np.zeros((len(observations), 8, 7), dtype=np.float32)

    environment = FakeEnvironment()
    policy = FakePolicy()
    monkeypatch.setattr(evaluation, "validate_simulation_dependencies", lambda: {})
    monkeypatch.setattr(evaluation, "resolve_qwen_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        evaluation,
        "configure_libero",
        lambda **kwargs: (tmp_path, tmp_path / "config.yaml", evaluation.LIBERO_COMMIT),
    )
    monkeypatch.setattr(evaluation, "resolve_libero_task", lambda *args, **kwargs: task)
    monkeypatch.setattr(
        evaluation.QwenLiberoPolicy,
        "from_checkpoint",
        classmethod(lambda cls, *args, **kwargs: policy),
    )
    monkeypatch.setattr(
        evaluation,
        "make_vector_environment_with_backoff",
        lambda *args, **kwargs: (environment, settings.num_envs),
    )

    report = evaluation.evaluate_qwen_checkpoint(settings)
    episode_rows = [
        json.loads(line)
        for line in (settings.output_dir / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert multiprocessing.get_start_method() == "spawn"
    assert tuple(policy.generator_seeds) == EXPECTED_EVALUATION_SEEDS
    assert tuple(environment.seeds) == EXPECTED_EVALUATION_SEEDS
    assert [
        (row["episode_id"], row["init_state_id"], row["seed"])
        for row in episode_rows
    ] == [
        (0, 0, 3471197683),
        (1, 1, 3471197683),
        (2, 0, 1232873419),
        (3, 1, 1232873419),
        (4, 0, 1448008435),
        (5, 1, 1448008435),
    ]
    assert report["route"] == "qwen3-vl-groot-libero-checkpoint-eval"
    assert report["checkpoint"]["weights_sha256"] == checkpoint.weights_sha256
    assert report["summary"]["completed_episodes"] == 6
