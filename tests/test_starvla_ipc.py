import time
from pathlib import Path
from types import SimpleNamespace
from threading import Thread

import numpy as np
import pytest
import torch


class _FakePolicy:
    def __init__(self):
        self.calls = []

    def predict_actions(self, image, instruction):
        self.calls.append((np.asarray(image).copy(), instruction))
        actions = np.zeros((1, 16, 7), dtype=np.float32)
        actions[0, 0, 0] = 0.125
        actions[..., 6] = 0.75
        return actions


def _wait_for_socket(path):
    deadline = time.monotonic() + 5.0
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def test_unix_ipc_routes_metadata_seed_inference_and_shutdown(tmp_path):
    from starvla_bridge.ipc import StarVLAIPCClient, serve_policy

    socket_path = tmp_path / "policy.sock"
    authkey = b"test-auth-key"
    policy = _FakePolicy()
    seeded = []
    thread = Thread(
        target=serve_policy,
        kwargs={
            "socket_path": socket_path,
            "authkey": authkey,
            "policy": policy,
            "metadata": {"native_action_chunk_size": 16},
            "seed_callback": seeded.append,
        },
        daemon=True,
    )
    thread.start()
    _wait_for_socket(socket_path)
    client = StarVLAIPCClient(socket_path, authkey=authkey)

    assert client.metadata()["native_action_chunk_size"] == 16
    assert client.reset_rng(4) == 4
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    actions = client.infer(image, "Put Spoon on Towel")

    assert actions.shape == (1, 16, 7)
    assert actions[0, 0, 0] == pytest.approx(0.125)
    assert policy.calls[0][1] == "Put Spoon on Towel"
    assert seeded == [4]
    client.shutdown()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not socket_path.exists()


def test_seed_everything_restarts_numpy_and_torch_random_streams():
    from starvla_bridge.ipc import seed_everything

    seed_everything(7)
    first_numpy = np.random.standard_normal(4)
    first_torch = torch.randn(4)
    seed_everything(7)

    np.testing.assert_array_equal(np.random.standard_normal(4), first_numpy)
    torch.testing.assert_close(torch.randn(4), first_torch, rtol=0, atol=0)
    seed_everything(8)
    assert not np.array_equal(np.random.standard_normal(4), first_numpy)


def test_ipc_rejects_nonfinite_or_wrong_shape_policy_actions(tmp_path):
    from starvla_bridge.ipc import StarVLAIPCClient, StarVLAIPCError, serve_policy

    class BrokenPolicy(_FakePolicy):
        def predict_actions(self, image, instruction):
            return np.full((1, 8, 7), np.nan, dtype=np.float32)

    socket_path = tmp_path / "broken.sock"
    thread = Thread(
        target=serve_policy,
        kwargs={
            "socket_path": socket_path,
            "authkey": b"broken",
            "policy": BrokenPolicy(),
            "metadata": {},
        },
        daemon=True,
    )
    thread.start()
    _wait_for_socket(socket_path)
    client = StarVLAIPCClient(socket_path, authkey=b"broken")

    with pytest.raises(StarVLAIPCError, match=r"expected \(1, 16, 7\)"):
        client.infer(np.zeros((224, 224, 3), dtype=np.uint8), "instruction")

    client.shutdown()
    thread.join(timeout=5)


def test_model_server_cli_and_metadata_expose_fixed_checkpoint_contract():
    from starvla_bridge.server import build_parser, build_server_metadata

    arguments = build_parser().parse_args(
        [
            "--socket",
            "/tmp/policy.sock",
            "--auth-key-hex",
            "001122",
            "--model-dir",
            "/models/starvla",
            "--base-model",
            "/models/qwen",
        ]
    )
    assert arguments.device == "cuda:0"
    assert arguments.socket == Path("/tmp/policy.sock")
    metadata = build_server_metadata(
        SimpleNamespace(
            model_dir=Path("/models/starvla"),
            checkpoint_path=Path("/models/starvla/checkpoints/model.pt"),
            base_model=Path("/models/qwen"),
            action_horizon=16,
            action_dim=7,
        ),
        SimpleNamespace(
            tensor_count=962,
            parameter_bytes=9_976_489_486,
            dtypes=("torch.bfloat16",),
        ),
    )
    assert metadata["native_action_chunk_size"] == 16
    assert metadata["checkpoint_tensor_count"] == 962
    assert metadata["available_unnorm_keys"] == ["oxe_bridge"]
