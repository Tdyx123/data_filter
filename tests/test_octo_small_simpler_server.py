from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _StartupPolicy:
    def __init__(self):
        self.calls = []
        self.statistics = SimpleNamespace(
            as_dict=lambda: {"path": "/models/checkpoint/normalization.json", "sha256": "abc"}
        )

    def make_generator(self, seed):
        self.calls.append(("seed", seed))
        return {"seed": seed}

    def prepare_observation(self, image, proprio, instruction):
        self.calls.append(("prepare", image, proprio, instruction))
        return {"prepared": True}

    def predict_actions(self, prepared, *, generator):
        self.calls.append(("predict", prepared, generator))
        return np.zeros((1, 8, 7), dtype=np.float32)

    def protocol_metadata(self):
        return {
            "native_action_chunk_size": 8,
            "precision": "bf16",
            "statistics": self.statistics.as_dict(),
        }


def test_server_parser_owns_model_arguments():
    from octo_small_bridge.server import build_parser

    arguments = build_parser().parse_args(
        [
            "--socket",
            "/tmp/octo.sock",
            "--auth-key-hex",
            "abcd",
            "--checkpoint",
            "/models/checkpoint",
            "--base-model",
            "/models/octo-small",
        ]
    )

    assert arguments.socket == Path("/tmp/octo.sock")
    assert arguments.checkpoint == Path("/models/checkpoint")
    assert arguments.base_model == Path("/models/octo-small")
    assert arguments.statistics is None
    assert arguments.device == "cuda:0"
    assert arguments.precision == "bf16"


def test_server_startup_preflight_exercises_raw_observation_boundary():
    from octo_small_bridge.server import run_startup_preflight

    policy = _StartupPolicy()
    report = run_startup_preflight(policy)

    assert report == {"action_shape": [1, 8, 7], "finite": True}
    assert policy.calls[0] == ("seed", 0)
    _, image, proprio, instruction = policy.calls[1]
    assert image.shape == (256, 256, 3)
    assert image.dtype == np.uint8
    assert proprio.shape == (8,)
    assert proprio.dtype == np.float32
    assert instruction == "Pick up the object."
    assert policy.calls[2] == ("predict", {"prepared": True}, {"seed": 0})


def test_server_main_loads_model_without_runtime_version_validation(monkeypatch):
    from octo_small_bridge import server

    captured = {}
    checkpoint = SimpleNamespace(
        as_dict=lambda: {
            "requested_path": "/models/checkpoint",
            "weights_path": "/models/checkpoint/model.safetensors",
        }
    )
    policy = _StartupPolicy()

    def load(checkpoint_path, **kwargs):
        captured["load"] = (checkpoint_path, kwargs)
        return checkpoint, policy

    def serve(**kwargs):
        captured["serve"] = kwargs

    monkeypatch.setattr(server, "load_octo_bridge_policy", load)
    monkeypatch.setattr(server, "serve_policy", serve)

    status = server.main(
        [
            "--socket",
            "/tmp/octo.sock",
            "--auth-key-hex",
            "abcd",
            "--checkpoint",
            "/models/checkpoint",
            "--base-model",
            "/models/octo-small",
            "--statistics",
            "/models/checkpoint/normalization.json",
            "--device",
            "cuda:3",
            "--precision",
            "fp32",
        ]
    )

    assert status == 0
    assert captured["load"] == (
        Path("/models/checkpoint"),
        {
            "base_model": Path("/models/octo-small"),
            "statistics": Path("/models/checkpoint/normalization.json"),
            "device": "cuda:3",
            "precision": "fp32",
        },
    )
    metadata = captured["serve"]["metadata"]
    assert metadata["model"] == "Octo-small Bridge checkpoint"
    assert metadata["device"] == "cuda:3"
    assert metadata["precision"] == "fp32"
    assert metadata["native_action_chunk_size"] == 8
    assert metadata["action_dim"] == 7
    assert metadata["checkpoint"] == {
        "requested_path": "/models/checkpoint",
        "weights_path": "/models/checkpoint/model.safetensors",
        "statistics": {
            "path": "/models/checkpoint/normalization.json",
            "sha256": "abc",
        },
    }
    assert metadata["startup_preflight"] == {
        "action_shape": [1, 8, 7],
        "finite": True,
    }
    assert captured["serve"]["socket_path"] == Path("/tmp/octo.sock")
    assert captured["serve"]["authkey"] == bytes.fromhex("abcd")
    assert captured["serve"]["policy"] is policy
