import threading
import time

import numpy as np
import pytest


class _Policy:
    def __init__(self):
        self.episodes = []
        self.prepared = []

    def make_generator(self, seed):
        return {"seed": seed}

    def begin_episode(self, instruction):
        self.episodes.append(instruction)

    def prepare_observation(self, image, instruction):
        value = {"image": image, "instruction": instruction}
        self.prepared.append(value)
        return value

    def predict_actions(self, prepared, *, generator):
        assert prepared is self.prepared[-1]
        assert generator == {"seed": 3}
        return np.zeros((1, 4, 7), dtype=np.float32)


def test_ipc_session_supports_metadata_rng_episode_infer_and_shutdown_without_proprio():
    from octo_small_official_pytorch.ipc import _PolicySession, _encode_array_payload

    policy = _Policy()
    session = _PolicySession(policy=policy, metadata={"model": "official"})
    metadata, stop = session.route({"type": "metadata"})
    assert metadata["data"]["protocol_name"] == "octo-small-official-pytorch-ipc-v1"
    assert stop is False
    session.route({"type": "reset_rng", "seed": 3})
    session.route({"type": "begin_episode", "instruction": "instruction"})
    image = np.zeros((12, 13, 3), dtype=np.uint8)
    response, stop = session.route(
        {
            "type": "infer",
            "image": _encode_array_payload(image),
            "instruction": "instruction",
        }
    )

    assert response["ok"] is True
    assert policy.episodes == ["instruction"]
    assert set(policy.prepared[0]) == {"image", "instruction"}
    assert stop is False
    assert session.route({"type": "shutdown"})[1] is True


def test_ipc_infer_requires_rng_and_begin_episode():
    from octo_small_official_pytorch.ipc import (
        OfficialIPCError,
        _PolicySession,
        _encode_array_payload,
    )

    session = _PolicySession(policy=_Policy(), metadata={})
    request = {
        "type": "infer",
        "image": _encode_array_payload(np.zeros((4, 5, 3), dtype=np.uint8)),
        "instruction": "instruction",
    }
    with pytest.raises(OfficialIPCError, match="reset_rng"):
        session.route(request)
    session.route({"type": "reset_rng", "seed": 3})
    with pytest.raises(OfficialIPCError, match="begin_episode"):
        session.route(request)


def test_ipc_rejects_proprio_field_and_wrong_action_shape():
    from octo_small_official_pytorch.ipc import (
        OfficialIPCError,
        _PolicySession,
        _encode_array_payload,
    )

    class Wrong(_Policy):
        def predict_actions(self, prepared, *, generator):
            return np.zeros((1, 8, 7), dtype=np.float32)

    session = _PolicySession(policy=Wrong(), metadata={})
    session.route({"type": "reset_rng", "seed": 3})
    session.route({"type": "begin_episode", "instruction": "instruction"})
    request = {
        "type": "infer",
        "image": _encode_array_payload(np.zeros((4, 5, 3), dtype=np.uint8)),
        "instruction": "instruction",
    }
    with pytest.raises(OfficialIPCError, match=r"expected \(1, 4, 7\)"):
        session.route(request)
    with pytest.raises(OfficialIPCError, match="proprio"):
        session.route({**request, "proprio": {}})


def test_authenticated_ipc_client_roundtrip_has_no_proprio(tmp_path):
    from octo_small_official_pytorch.ipc import OfficialIPCClient, serve_policy

    policy = _Policy()
    socket_path = tmp_path / "official.sock"
    thread = threading.Thread(
        target=serve_policy,
        kwargs={
            "socket_path": socket_path,
            "authkey": b"official-test",
            "policy": policy,
            "metadata": {"native_action_chunk_size": 4},
        },
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not socket_path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError("IPC socket was not created")
        time.sleep(0.01)
    client = OfficialIPCClient(socket_path, authkey=b"official-test")
    assert client.metadata()["native_action_chunk_size"] == 4
    assert client.reset_rng(3) == 3
    client.begin_episode("instruction")
    actions = client.infer(np.zeros((8, 9, 3), dtype=np.uint8), "instruction")

    assert actions.shape == (1, 4, 7)
    assert set(policy.prepared[0]) == {"image", "instruction"}
    client.shutdown()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not socket_path.exists()
