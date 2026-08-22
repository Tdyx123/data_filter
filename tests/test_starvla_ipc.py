import os
import subprocess
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


class _RecordingConnection:
    def __init__(self, response):
        self.response = response
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        return self.response

    def close(self):
        pass


def _wait_for_socket(path):
    deadline = time.monotonic() + 5.0
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def _literal_array_payload(array):
    contiguous = np.ascontiguousarray(array)
    return {
        "data": contiguous.tobytes(order="C"),
        "dtype": contiguous.dtype.name,
        "shape": list(contiguous.shape),
    }


def _assert_builtin_wire_value(value):
    assert not isinstance(value, (np.ndarray, np.generic))
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str
            _assert_builtin_wire_value(item)
    elif isinstance(value, list):
        for item in value:
            _assert_builtin_wire_value(item)
    else:
        assert type(value) in {str, int, float, bool, bytes, type(None)}


def test_infer_route_decodes_image_bytes_and_encodes_action_bytes():
    from starvla_bridge.ipc import _route_request

    policy = _FakePolicy()
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    response, shutting_down = _route_request(
        {
            "type": "infer",
            "image": _literal_array_payload(image),
            "instruction": "Put Spoon on Towel",
        },
        policy=policy,
        metadata={},
        seed_callback=lambda seed: None,
    )

    assert shutting_down is False
    assert response["ok"] is True
    payload = response["data"]["actions"]
    assert payload["dtype"] == "float32"
    assert payload["shape"] == [1, 16, 7]
    assert isinstance(payload["data"], bytes)
    assert len(payload["data"]) == 1 * 16 * 7 * 4
    _assert_builtin_wire_value(response)
    np.testing.assert_array_equal(policy.calls[0][0], image)


def test_client_infer_sends_builtin_image_payload_and_decodes_action_bytes():
    from starvla_bridge.ipc import StarVLAIPCClient

    expected = np.arange(1 * 16 * 7, dtype=np.float32).reshape(1, 16, 7)
    connection = _RecordingConnection(
        {"ok": True, "data": {"actions": _literal_array_payload(expected)}}
    )
    client = StarVLAIPCClient.__new__(StarVLAIPCClient)
    client._connection = connection
    client._closed = False
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)

    actions = client.infer(image, "Put Spoon on Towel")

    request = connection.sent[0]
    assert request["image"]["dtype"] == "uint8"
    assert request["image"]["shape"] == [224, 224, 3]
    assert isinstance(request["image"]["data"], bytes)
    assert len(request["image"]["data"]) == 224 * 224 * 3
    _assert_builtin_wire_value(request)
    np.testing.assert_array_equal(actions, expected)


@pytest.mark.parametrize(
    ("payload", "error_pattern"),
    [
        (
            {"data": bytes(448), "dtype": "float64", "shape": [1, 16, 7]},
            "dtype.*float32",
        ),
        (
            {"data": bytes(448), "dtype": "float32", "shape": [1, 8, 7]},
            r"shape.*\[1, 16, 7\]",
        ),
        (
            {"data": bytes(447), "dtype": "float32", "shape": [1, 16, 7]},
            "byte length.*448",
        ),
        (
            {"data": memoryview(bytes(448)), "dtype": "float32", "shape": [1, 16, 7]},
            "data must be bytes",
        ),
    ],
)
def test_array_payload_decoder_rejects_invalid_contract(payload, error_pattern):
    from starvla_bridge.ipc import StarVLAIPCError, _decode_array_payload

    with pytest.raises(StarVLAIPCError, match=error_pattern):
        _decode_array_payload(
            payload,
            name="actions",
            expected_dtype=np.dtype("float32"),
            expected_shape=(1, 16, 7),
        )


def test_array_payload_decoder_reports_non_string_keys_as_contract_error():
    from starvla_bridge.ipc import StarVLAIPCError, _decode_array_payload

    payload = _literal_array_payload(np.zeros((1, 16, 7), dtype=np.float32))
    payload[1] = None

    with pytest.raises(StarVLAIPCError, match="payload keys"):
        _decode_array_payload(
            payload,
            name="actions",
            expected_dtype=np.dtype("float32"),
            expected_shape=(1, 16, 7),
        )


def test_client_infer_rejects_missing_action_payload_with_ipc_error():
    from starvla_bridge.ipc import StarVLAIPCClient, StarVLAIPCError

    client = StarVLAIPCClient.__new__(StarVLAIPCClient)
    client._connection = _RecordingConnection({"ok": True, "data": {}})
    client._closed = False

    with pytest.raises(StarVLAIPCError, match="actions payload"):
        client.infer(
            np.zeros((224, 224, 3), dtype=np.uint8),
            "Put Spoon on Towel",
        )


@pytest.mark.parametrize("side", ["server", "client"])
def test_infer_rejects_nonfinite_actions_on_both_sides(side):
    from starvla_bridge.ipc import (
        StarVLAIPCClient,
        StarVLAIPCError,
        _route_request,
    )

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    actions = np.full((1, 16, 7), np.nan, dtype=np.float32)
    if side == "server":
        policy = _FakePolicy()
        policy.predict_actions = lambda image, instruction: actions
        with pytest.raises(StarVLAIPCError, match="NaN or infinite"):
            _route_request(
                {
                    "type": "infer",
                    "image": _literal_array_payload(image),
                    "instruction": "Put Spoon on Towel",
                },
                policy=policy,
                metadata={},
                seed_callback=lambda seed: None,
            )
        return

    client = StarVLAIPCClient.__new__(StarVLAIPCClient)
    client._connection = _RecordingConnection(
        {"ok": True, "data": {"actions": _literal_array_payload(actions)}}
    )
    client._closed = False
    with pytest.raises(StarVLAIPCError, match="NaN or infinite"):
        client.infer(image, "Put Spoon on Towel")


def test_cross_numpy_versions_round_trip_action_bytes():
    project_root = Path(__file__).resolve().parents[1]
    model_python = Path(
        "/home/dwb/.pyenv/versions/miniconda3-3.12-25.11.1-1/bin/python"
    )
    simulator_python = project_root / ".venv-octo-simpler/bin/python"
    if not model_python.is_file() or not simulator_python.is_file():
        pytest.skip("StarVLA model and SimplerEnv interpreters are not both available")
    environment = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    encode_script = """
import pickle
import sys
import numpy as np
from starvla_bridge.ipc import _encode_array_payload

actions = np.arange(112, dtype=np.float32).reshape(1, 16, 7)
sys.stdout.buffer.write(pickle.dumps(_encode_array_payload(actions), protocol=5))
"""
    encoded = subprocess.run(
        [str(model_python), "-c", encode_script],
        check=True,
        capture_output=True,
        env=environment,
    ).stdout
    decode_script = """
import pickle
import sys
import numpy as np
from starvla_bridge.ipc import _decode_array_payload

payload = pickle.loads(sys.stdin.buffer.read())
actions = _decode_array_payload(
    payload,
    name="actions",
    expected_dtype=np.dtype("float32"),
    expected_shape=(1, 16, 7),
)
assert actions.shape == (1, 16, 7)
assert actions.dtype == np.float32
assert actions[0, 15, 6] == 111.0
assert actions.flags.owndata
print(np.__version__)
"""
    decoded = subprocess.run(
        [str(simulator_python), "-c", decode_script],
        input=encoded,
        check=True,
        capture_output=True,
        env=environment,
    )

    assert decoded.stdout.decode().strip() == "1.24.3"


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

    metadata = client.metadata()
    assert metadata["protocol_version"] == 2
    assert metadata["native_action_chunk_size"] == 16
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


def test_seed_everything_accepts_full_parallel_episode_seed():
    from starvla_bridge.ipc import seed_everything

    episode_seed = 8_361_816_881_672_972_874
    numpy_seed = episode_seed % (2**32)

    seed_everything(episode_seed)
    actual_numpy = np.random.standard_normal(4)
    actual_torch = torch.randn(4)

    expected_numpy = np.random.RandomState(numpy_seed).standard_normal(4)
    expected_torch_generator = torch.Generator().manual_seed(episode_seed)
    expected_torch = torch.randn(4, generator=expected_torch_generator)
    np.testing.assert_array_equal(actual_numpy, expected_numpy)
    torch.testing.assert_close(actual_torch, expected_torch, rtol=0, atol=0)


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
        device="cuda:3",
    )
    assert metadata["native_action_chunk_size"] == 16
    assert metadata["checkpoint_tensor_count"] == 962
    assert metadata["available_unnorm_keys"] == ["oxe_bridge"]
    assert metadata["device"] == "cuda:3"
