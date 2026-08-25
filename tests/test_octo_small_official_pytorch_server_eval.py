from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _StartupPolicy:
    def __init__(self):
        self.calls = []
        self.statistics = SimpleNamespace(as_dict=lambda: {"sha256": "abc"})

    def make_generator(self, seed):
        self.calls.append(("seed", seed))
        return {"seed": seed}

    def begin_episode(self, instruction):
        self.calls.append(("begin", instruction))

    def prepare_observation(self, image, instruction):
        history = 1 + sum(call[0] == "prepare" for call in self.calls)
        prepared = {"image_primary": np.zeros((1, history, 3, 256, 256), dtype=np.float32)}
        self.calls.append(("prepare", image, instruction, prepared))
        return prepared

    def predict_actions(self, prepared, *, generator):
        self.calls.append(("predict", prepared, generator))
        return np.zeros((1, 4, 7), dtype=np.float32)

    def describe_observation(self, prepared):
        return {
            "model_image_shape": list(prepared["image_primary"].shape),
            "image_history_length": prepared["image_primary"].shape[1],
            "image_history_horizon": 2,
            "use_proprio": False,
        }

    def protocol_metadata(self):
        return {
            "native_action_chunk_size": 4,
            "model_action_gripper": "continuous_model_prediction",
            "image_history_horizon": 2,
            "use_proprio": False,
            "rng": {
                "backend": "torch.Generator",
                "jax_seed_bitwise_equivalent": False,
            },
        }


def test_server_parser_and_startup_preflight_use_official_contract():
    from octo_small_official_pytorch.server import build_parser, run_startup_preflight

    arguments = build_parser().parse_args(
        [
            "--socket",
            "/tmp/octo.sock",
            "--auth-key-hex",
            "abcd",
            "--checkpoint",
            "/models/official",
        ]
    )
    assert arguments.checkpoint == Path("/models/official")
    assert arguments.device == "cuda:0"
    assert arguments.precision == "bf16"
    assert not hasattr(arguments, "base_model")
    assert not hasattr(arguments, "statistics")

    policy = _StartupPolicy()
    report = run_startup_preflight(policy)

    assert report == {
        "action_shape": [1, 4, 7],
        "finite": True,
        "first_inference_history_length": 1,
        "second_inference_history_length": 2,
        "image_history_horizon": 2,
        "use_proprio": False,
    }
    assert policy.calls[0] == ("seed", 0)
    assert policy.calls[1] == ("begin", "Pick up the object.")


def test_server_metadata_records_actual_model_runtime_and_torch_rng(monkeypatch):
    from octo_small_official_pytorch import server

    monkeypatch.setattr(
        server,
        "runtime_versions",
        lambda: {
            "python": "3.12.11",
            "torch": "2.7.1",
            "transformers": "4.52.4",
        },
    )
    checkpoint = SimpleNamespace(as_dict=lambda: {"format": "official"})
    policy = _StartupPolicy()

    metadata = server.server_metadata(
        checkpoint,
        policy,
        device="cuda:4",
        precision="bf16",
        startup_preflight={"action_shape": [1, 4, 7]},
    )

    assert metadata["native_action_chunk_size"] == 4
    assert metadata["image_history_horizon"] == 2
    assert metadata["use_proprio"] is False
    assert metadata["model_runtime"]["torch"] == "2.7.1"
    assert metadata["rng"] == {
        "backend": "torch.Generator",
        "jax_seed_bitwise_equivalent": False,
    }


def test_simpler_client_defaults_to_official_route_and_horizon_four_ensemble():
    from octo_small_official_pytorch.evaluate_simpler import build_parser

    parser = build_parser()
    arguments = parser.parse_args(["--socket", "/tmp/octo.sock", "--auth-key-hex", "abcd"])

    assert arguments.tasks == "all"
    assert arguments.output_dir == Path("outputs/octo_small_official_pytorch_simpler_eval")
    assert arguments.action_horizon == 1
    assert arguments.action_postprocessing == "octo_temporal_ensemble_v1"
    assert arguments.sim_device == "cuda:0"
