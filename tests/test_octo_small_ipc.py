import threading
import time
from multiprocessing.context import AuthenticationError

import numpy as np
import pytest


class _FakePolicy:
    def __init__(self, actions=None):
        self.actions = (
            np.arange(56, dtype=np.float32).reshape(1, 8, 7)
            if actions is None
            else actions
        )
        self.seeds = []
        self.observations = []

    def make_generator(self, seed):
        self.seeds.append(seed)
        return {"seed": seed}

    def prepare_observation(self, image, proprio, instruction):
        prepared = {
            "image": np.asarray(image),
            "proprio": np.asarray(proprio),
            "instruction": instruction,
        }
        self.observations.append(prepared)
        return prepared

    def predict_actions(self, prepared, *, generator):
        assert prepared is self.observations[-1]
        assert generator == {"seed": self.seeds[-1]}
        return self.actions


def _wait_for_socket(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"socket was not created: {path}")
        time.sleep(0.01)


def _start_server(tmp_path, policy, *, authkey=b"octo-test-key"):
    from octo_small_bridge.ipc import serve_policy

    socket_path = tmp_path / "octo.sock"
    thread = threading.Thread(
        target=serve_policy,
        kwargs={
            "socket_path": socket_path,
            "authkey": authkey,
            "policy": policy,
            "metadata": {"model": "octo-small", "native_action_chunk_size": 8},
        },
        daemon=True,
    )
    thread.start()
    _wait_for_socket(socket_path)
    return socket_path, thread


def test_authenticated_client_round_trips_metadata_rng_observation_and_actions(tmp_path):
    from octo_small_bridge.ipc import IPC_PROTOCOL_VERSION, OctoIPCClient

    policy = _FakePolicy()
    socket_path, thread = _start_server(tmp_path, policy)
    client = OctoIPCClient(socket_path, authkey=b"octo-test-key")

    metadata = client.metadata()
    assert metadata == {
        "protocol_version": IPC_PROTOCOL_VERSION,
        "model": "octo-small",
        "native_action_chunk_size": 8,
    }
    assert client.reset_rng(4) == 4
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    proprio = np.arange(8, dtype=np.float32)
    actions = client.infer(image, proprio, "Put Spoon on Towel")

    np.testing.assert_array_equal(actions, policy.actions)
    assert policy.seeds == [4]
    np.testing.assert_array_equal(policy.observations[0]["image"], image)
    np.testing.assert_array_equal(policy.observations[0]["proprio"], proprio)
    assert policy.observations[0]["instruction"] == "Put Spoon on Towel"

    client.shutdown()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not socket_path.exists()


@pytest.mark.parametrize(
    ("image", "proprio", "instruction", "message"),
    [
        (
            np.zeros((32, 32, 3), dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            "instruction",
            "uint8 HWC RGB",
        ),
        (
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros(7, dtype=np.float32),
            "instruction",
            "float32 proprio",
        ),
        (
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros(8, dtype=np.float32),
            "  ",
            "instruction must be non-empty",
        ),
    ],
    ids=("image", "proprio", "instruction"),
)
def test_client_rejects_invalid_observation_before_sending(
    tmp_path, image, proprio, instruction, message
):
    from octo_small_bridge.ipc import OctoIPCClient, OctoIPCError

    socket_path, thread = _start_server(tmp_path, _FakePolicy())
    client = OctoIPCClient(socket_path, authkey=b"octo-test-key")

    with pytest.raises(OctoIPCError, match=message):
        client.infer(image, proprio, instruction)

    client.shutdown()
    thread.join(timeout=5)


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (np.zeros((1, 7, 7), dtype=np.float32), "expected \\(1, 8, 7\\)"),
        (
            np.full((1, 8, 7), np.nan, dtype=np.float32),
            "NaN or infinite",
        ),
    ],
    ids=("shape", "finite"),
)
def test_server_rejects_invalid_policy_actions(tmp_path, actions, message):
    from octo_small_bridge.ipc import OctoIPCClient, OctoIPCError

    socket_path, thread = _start_server(tmp_path, _FakePolicy(actions))
    client = OctoIPCClient(socket_path, authkey=b"octo-test-key")
    client.reset_rng(0)

    with pytest.raises(OctoIPCError, match=message):
        client.infer(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros(8, dtype=np.float32),
            "instruction",
        )

    client.shutdown()
    thread.join(timeout=5)


def test_server_requires_rng_reset_before_inference(tmp_path):
    from octo_small_bridge.ipc import OctoIPCClient, OctoIPCError

    socket_path, thread = _start_server(tmp_path, _FakePolicy())
    client = OctoIPCClient(socket_path, authkey=b"octo-test-key")

    with pytest.raises(OctoIPCError, match="reset_rng"):
        client.infer(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros(8, dtype=np.float32),
            "instruction",
        )

    client.shutdown()
    thread.join(timeout=5)


def test_server_rejects_wrong_authentication_key(tmp_path):
    from octo_small_bridge.ipc import OctoIPCClient

    socket_path, thread = _start_server(tmp_path, _FakePolicy())

    with pytest.raises(AuthenticationError):
        OctoIPCClient(socket_path, authkey=b"wrong-key")

    client = OctoIPCClient(socket_path, authkey=b"octo-test-key")
    client.shutdown()
    thread.join(timeout=5)
