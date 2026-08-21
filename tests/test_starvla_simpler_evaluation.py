import numpy as np
import pytest
from types import SimpleNamespace


class _FakeClient:
    def __init__(self):
        self.seeds = []
        self.inferences = []

    def metadata(self):
        return {
            "protocol_version": 2,
            "model": "Qwen3VL-GR00T-Bridge-RT-1",
            "device": "cuda:3",
            "native_action_chunk_size": 16,
            "available_unnorm_keys": ["oxe_bridge"],
            "runtime": {"python": "3.12.12", "torch": "2.10.0+cu128"},
            "startup_preflight": {"action_shape": [1, 16, 7], "finite": True},
        }

    def reset_rng(self, seed):
        self.seeds.append(seed)
        return seed

    def infer(self, image, instruction):
        self.inferences.append((np.asarray(image).copy(), instruction))
        actions = np.zeros((1, 16, 7), dtype=np.float32)
        actions[..., 6] = 0.75
        return actions


def test_remote_policy_resizes_rgb_resets_seed_and_reports_native_chunk():
    from starvla_bridge.simpler_evaluation import StarVLARemotePolicy

    client = _FakeClient()
    policy = StarVLARemotePolicy(client)
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    generator = policy.make_generator(2)
    prepared = policy.prepare_observation(
        image,
        np.arange(8, dtype=np.float32),
        "Put Spoon on Towel",
    )
    actions = policy.predict_actions(prepared, generator=generator)

    assert generator == 2
    assert client.seeds == [2]
    assert prepared["image"].shape == (224, 224, 3)
    assert client.inferences[0][1] == "Put Spoon on Towel"
    assert actions.shape == (1, 16, 7)
    protocol = policy.protocol_metadata()
    assert protocol["native_action_chunk_size"] == 16
    assert protocol["model_runtime"]["python"] == "3.12.12"
    assert protocol["startup_preflight"]["finite"] is True
    assert protocol["terminate_episode"] == 0
    assert policy.metadata["model"] == "Qwen3VL-GR00T-Bridge-RT-1"
    assert policy.model_device == "cuda:3"
    assert policy.gripper_threshold == 0.5


def test_remote_policy_rejects_server_metadata_for_wrong_checkpoint_contract():
    from starvla_bridge.simpler_evaluation import StarVLARemotePolicy

    client = _FakeClient()
    client.metadata = lambda: {
        "protocol_version": 2,
        "native_action_chunk_size": 8,
        "available_unnorm_keys": ["oxe_bridge"],
    }

    with pytest.raises(Exception, match="native_action_chunk_size"):
        StarVLARemotePolicy(client)


def test_remote_policy_rejects_protocol_version_one():
    from starvla_bridge.simpler_evaluation import StarVLARemotePolicy

    client = _FakeClient()
    client.metadata = lambda: {
        "protocol_version": 1,
        "native_action_chunk_size": 16,
        "available_unnorm_keys": ["oxe_bridge"],
    }

    with pytest.raises(Exception, match=r"protocol_version must be 2.*found 1"):
        StarVLARemotePolicy(client)


def test_starvla_first_action_uses_world_euler_axis_angle_and_binary_gripper():
    from simpler_bridge.evaluation import bridge_actions_to_simpler

    action = np.asarray(
        [0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.51],
        dtype=np.float32,
    )

    converted = bridge_actions_to_simpler(action, gripper_threshold=0.5)

    np.testing.assert_allclose(converted[:3], action[:3], rtol=0, atol=1.0e-7)
    np.testing.assert_allclose(
        converted[3:6], [0.0, 0.0, np.pi / 2], rtol=0, atol=1.0e-6
    )
    assert converted[6] == 1.0
    action[6] = 0.5
    assert bridge_actions_to_simpler(action, gripper_threshold=0.5)[6] == -1.0


def test_starvla_cli_defaults_to_full_stepwise_protocol():
    from starvla_bridge.evaluate_simpler import build_parser

    parser = build_parser()
    arguments = parser.parse_args(
        [
            "--socket",
            "/tmp/policy.sock",
            "--auth-key-hex",
            "001122",
        ]
    )

    assert arguments.tasks == "all"
    assert arguments.action_horizon == 1
    assert arguments.sim_device == "cuda:0"
    assert arguments.output_dir.name == "starvla_simpler_eval"
    assert arguments.smoke_test is False
    assert arguments.shard_index == 0
    assert arguments.shard_count == 1
    assert arguments.rng_scope == "per_policy_seed_stream"
    assert parser.parse_args(
        [
            "--socket",
            "/tmp/policy.sock",
            "--auth-key-hex",
            "001122",
            "--sim-device",
            "cuda:12",
        ]
    ).sim_device == "cuda:12"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--socket",
                "/tmp/policy.sock",
                "--auth-key-hex",
                "001122",
                "--action-horizon",
                "8",
            ]
        )


@pytest.mark.parametrize(
    "sim_device",
    ("cuda:-1", "cuda", "3", "/dev/nvidia3", "cpu", "cuda:+1", "cuda: 1"),
)
def test_starvla_cli_rejects_non_logical_cuda_sim_devices(sim_device):
    from starvla_bridge.evaluate_simpler import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--socket",
                "/tmp/policy.sock",
                "--auth-key-hex",
                "001122",
                "--sim-device",
                sim_device,
            ]
        )


@pytest.mark.parametrize("preflight_only", (False, True), ids=("evaluation", "preflight"))
def test_starvla_cli_keeps_remote_model_device_and_forwards_sim_device(
    tmp_path, monkeypatch, preflight_only
):
    from starvla_bridge import evaluate_simpler

    captured = {}

    class Client:
        def __init__(self, socket, *, authkey):
            captured["connection"] = (socket, authkey)

        def shutdown(self):
            captured["shutdown"] = True

    policy = SimpleNamespace(
        metadata={"device": "cuda:5"},
        model_device="cuda:5",
        protocol_metadata=lambda: {},
    )
    monkeypatch.setattr(evaluate_simpler, "StarVLAIPCClient", Client)
    monkeypatch.setattr(evaluate_simpler, "StarVLARemotePolicy", lambda client: policy)
    monkeypatch.setattr(evaluate_simpler, "validate_simpler_source", lambda path: {})
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
        "--socket",
        str(tmp_path / "policy.sock"),
        "--auth-key-hex",
        "001122",
        "--tasks",
        "spoon",
        "--output-dir",
        str(tmp_path / "output"),
        "--sim-device",
        "cuda:7",
        "--shard-index",
        "1",
        "--shard-count",
        "3",
        "--rng-scope",
        "per_episode",
    ]
    if preflight_only:
        arguments.append("--preflight-only")

    status = evaluate_simpler.main(arguments)

    assert status == 0
    assert captured["settings"].device == "remote-pyenv:cuda:5"
    assert captured["settings"].sim_device == "cuda:7"
    assert captured["settings"].shard_index == 1
    assert captured["settings"].shard_count == 3
    assert captured["settings"].rng_scope == "per_episode"
    assert captured["builder"] == ("spoon", "cuda:7")
    assert captured["shutdown"] is True
