from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _StartupPolicy:
    def __init__(self):
        self.calls = []

    def make_generator(self, seed):
        self.calls.append(("seed", seed))
        return seed

    def prepare_observation(self, image, proprio, instruction):
        self.calls.append(("prepare", image, proprio, instruction))
        return {"prepared": True}

    def predict_actions(self, prepared, *, generator):
        self.calls.append(("predict", prepared, generator))
        return np.zeros((1, 8, 7), dtype=np.float32)

    def protocol_metadata(self):
        return {
            "inference_strategy": "deterministic-causal-query-oft",
            "native_action_chunk_size": 8,
        }


def test_oft_server_parser_owns_model_arguments_without_denoising():
    from qwen_vl_oft.server import build_parser

    arguments = build_parser().parse_args(
        [
            "--socket",
            "/tmp/qwen-oft.sock",
            "--auth-key-hex",
            "abcd",
            "--checkpoint",
            "/models/checkpoint",
        ]
    )

    assert arguments.socket == Path("/tmp/qwen-oft.sock")
    assert arguments.checkpoint == Path("/models/checkpoint")
    assert arguments.model_path is None
    assert arguments.device == "cuda:0"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--socket",
                "/tmp/qwen-oft.sock",
                "--auth-key-hex",
                "abcd",
                "--checkpoint",
                "/models/checkpoint",
                "--denoising-steps",
                "4",
            ]
        )


def test_oft_server_startup_preflight_exercises_raw_observation_boundary():
    from qwen_vl_oft.server import run_startup_preflight

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
    assert policy.calls[2] == ("predict", {"prepared": True}, 0)


def test_oft_server_main_publishes_format_specific_metadata(monkeypatch):
    from qwen_vl_oft import server

    captured = {}
    checkpoint = SimpleNamespace(
        config={"data": {"train_crop_size": 230, "output_image_size": 256}},
        as_dict=lambda: {
            "requested_path": "/models/checkpoint",
            "weights_path": "/models/checkpoint/adapter_model.safetensors",
        },
    )
    policy = _StartupPolicy()

    def load(checkpoint_path, **kwargs):
        captured["load"] = (checkpoint_path, kwargs)
        return checkpoint, policy

    def serve(**kwargs):
        captured["serve"] = kwargs

    monkeypatch.setattr(server, "load_oft_simpler_policy", load)
    monkeypatch.setattr(server, "serve_policy", serve)

    status = server.main(
        [
            "--socket",
            "/tmp/qwen-oft.sock",
            "--auth-key-hex",
            "abcd",
            "--checkpoint",
            "/models/checkpoint",
            "--model-path",
            "/models/Qwen3-VL-4B-Instruct",
            "--device",
            "cuda:3",
        ]
    )

    assert status == 0
    assert captured["load"] == (
        Path("/models/checkpoint"),
        {
            "model_path": Path("/models/Qwen3-VL-4B-Instruct"),
            "device": "cuda:3",
        },
    )
    assert captured["serve"]["metadata"] == {
        "model": "Qwen-VL OFT Bridge checkpoint",
        "native_action_chunk_size": 8,
        "action_dim": 7,
        "device": "cuda:3",
        "checkpoint": {
            "requested_path": "/models/checkpoint",
            "weights_path": "/models/checkpoint/adapter_model.safetensors",
        },
        "model_image_shape": [256, 256, 3],
        "train_crop_size": 230,
        "protocol": {
            "inference_strategy": "deterministic-causal-query-oft",
            "native_action_chunk_size": 8,
        },
        "startup_preflight": {"action_shape": [1, 8, 7], "finite": True},
    }
    assert captured["serve"]["socket_path"] == Path("/tmp/qwen-oft.sock")
    assert captured["serve"]["authkey"] == bytes.fromhex("abcd")
    assert captured["serve"]["policy"] is policy
