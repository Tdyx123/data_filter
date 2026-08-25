from __future__ import annotations

import time
from multiprocessing.connection import Client, Listener
from multiprocessing.context import AuthenticationError
from pathlib import Path
from typing import Any, Mapping

import numpy as np


IPC_PROTOCOL_VERSION = 1
IPC_PROTOCOL_NAME = "octo-small-official-pytorch-ipc-v1"
ACTION_SHAPE = (1, 4, 7)


class OfficialIPCError(RuntimeError):
    """Raised when the official-parity local policy service rejects a request."""


def _encode_array_payload(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "data": contiguous.tobytes(order="C"),
        "dtype": contiguous.dtype.name,
        "shape": list(contiguous.shape),
    }


def _decode_array_payload(
    payload: Any,
    *,
    name: str,
    expected_dtype: np.dtype[Any],
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if not isinstance(payload, Mapping) or set(payload) != {"data", "dtype", "shape"}:
        raise OfficialIPCError(f"{name} payload is invalid")
    if payload["dtype"] != expected_dtype.name:
        raise OfficialIPCError(f"{name} dtype must be {expected_dtype.name}")
    shape = payload["shape"]
    if type(shape) is not list or any(type(item) is not int or item <= 0 for item in shape):
        raise OfficialIPCError(f"{name} shape must contain positive integers")
    if expected_shape is not None and shape != list(expected_shape):
        raise OfficialIPCError(f"{name} shape must be {list(expected_shape)}, found {shape}")
    data = payload["data"]
    if type(data) is not bytes:
        raise OfficialIPCError(f"{name} data must be bytes")
    item_count = int(np.prod(shape))
    if len(data) != item_count * expected_dtype.itemsize:
        raise OfficialIPCError(f"{name} byte length does not match its shape")
    return np.frombuffer(data, dtype=expected_dtype).reshape(shape).copy()


def _validate_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[0] <= 0
        or image.shape[1] <= 0
        or image.shape[2] != 3
    ):
        raise OfficialIPCError("infer expected a uint8 HWC RGB image with positive dimensions")
    return np.ascontiguousarray(image)


def _validate_instruction(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OfficialIPCError("instruction must be non-empty")
    return value.strip()


class _PolicySession:
    def __init__(self, *, policy: Any, metadata: Mapping[str, Any]) -> None:
        self.policy = policy
        self.metadata = {
            **dict(metadata),
            "protocol_version": IPC_PROTOCOL_VERSION,
            "protocol_name": IPC_PROTOCOL_NAME,
        }
        self.generator: Any | None = None
        self.instruction: str | None = None

    def route(self, request: Any) -> tuple[dict[str, Any], bool]:
        if not isinstance(request, Mapping):
            raise OfficialIPCError("IPC request must be a mapping")
        request_type = request.get("type")
        if request_type == "metadata":
            return {"ok": True, "data": dict(self.metadata)}, False
        if request_type == "reset_rng":
            seed = int(request["seed"])
            self.generator = self.policy.make_generator(seed)
            return {"ok": True, "data": {"seed": seed}}, False
        if request_type == "begin_episode":
            instruction = _validate_instruction(request.get("instruction"))
            self.policy.begin_episode(instruction)
            self.instruction = instruction
            return {"ok": True, "data": {"instruction": instruction}}, False
        if request_type == "infer":
            if "proprio" in request:
                raise OfficialIPCError("Official IPC infer must not contain proprio")
            if self.generator is None:
                raise OfficialIPCError("reset_rng must be called before infer")
            if self.instruction is None:
                raise OfficialIPCError("begin_episode must be called before infer")
            instruction = _validate_instruction(request.get("instruction"))
            if instruction != self.instruction:
                raise OfficialIPCError("infer instruction does not match begin_episode")
            image = _validate_image(
                _decode_array_payload(
                    request.get("image"),
                    name="image",
                    expected_dtype=np.dtype("uint8"),
                )
            )
            prepared = self.policy.prepare_observation(image, instruction)
            actions = np.asarray(
                self.policy.predict_actions(prepared, generator=self.generator),
                dtype=np.float32,
            )
            if actions.shape != ACTION_SHAPE:
                raise OfficialIPCError(f"policy returned {actions.shape}; expected {ACTION_SHAPE}")
            if not np.all(np.isfinite(actions)):
                raise OfficialIPCError("policy returned NaN or infinite actions")
            return {
                "ok": True,
                "data": {"actions": _encode_array_payload(actions)},
            }, False
        if request_type == "shutdown":
            return {"ok": True, "data": {"shutdown": True}}, True
        raise OfficialIPCError(f"Unsupported IPC request type: {request_type!r}")


def serve_policy(
    *,
    socket_path: str | Path,
    authkey: bytes,
    policy: Any,
    metadata: Mapping[str, Any],
) -> None:
    target = Path(socket_path).expanduser().resolve()
    if not authkey:
        raise OfficialIPCError("IPC authkey must be non-empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise OfficialIPCError(f"IPC socket already exists: {target}")
    session = _PolicySession(policy=policy, metadata=metadata)
    listener = Listener(str(target), family="AF_UNIX", authkey=authkey)
    target.chmod(0o600)
    shutting_down = False
    try:
        while not shutting_down:
            try:
                connection = listener.accept()
            except AuthenticationError:
                continue
            try:
                while not shutting_down:
                    try:
                        request = connection.recv()
                    except EOFError:
                        break
                    try:
                        response, shutting_down = session.route(request)
                    except Exception as error:
                        response = {
                            "ok": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    connection.send(response)
            finally:
                connection.close()
    finally:
        listener.close()
        target.unlink(missing_ok=True)


class OfficialIPCClient:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        authkey: bytes,
        connect_timeout: float = 30.0,
    ) -> None:
        self.socket_path = Path(socket_path).expanduser().resolve()
        if not authkey:
            raise OfficialIPCError("IPC authkey must be non-empty")
        deadline = time.monotonic() + float(connect_timeout)
        while True:
            try:
                self._connection = Client(str(self.socket_path), family="AF_UNIX", authkey=authkey)
                break
            except (FileNotFoundError, ConnectionRefusedError) as error:
                if time.monotonic() >= deadline:
                    raise OfficialIPCError(f"Timed out connecting to {self.socket_path}") from error
                time.sleep(0.05)
        self._closed = False

    def _request(self, request: Mapping[str, Any]) -> Any:
        if self._closed:
            raise OfficialIPCError("Policy client is closed")
        try:
            self._connection.send(dict(request))
            response = self._connection.recv()
        except (EOFError, OSError) as error:
            raise OfficialIPCError(f"Policy server connection failed: {error}") from error
        if not isinstance(response, Mapping) or not response.get("ok"):
            message = (
                response.get("error", "invalid response")
                if isinstance(response, Mapping)
                else "invalid response"
            )
            raise OfficialIPCError(str(message))
        return response.get("data")

    def metadata(self) -> dict[str, Any]:
        data = self._request({"type": "metadata"})
        if not isinstance(data, Mapping):
            raise OfficialIPCError("Policy metadata must be a mapping")
        return dict(data)

    def reset_rng(self, seed: int) -> int:
        data = self._request({"type": "reset_rng", "seed": int(seed)})
        return int(data["seed"])

    def begin_episode(self, instruction: str) -> None:
        self._request({"type": "begin_episode", "instruction": _validate_instruction(instruction)})

    def infer(self, image: np.ndarray, instruction: str) -> np.ndarray:
        data = self._request(
            {
                "type": "infer",
                "image": _encode_array_payload(_validate_image(image)),
                "instruction": _validate_instruction(instruction),
            }
        )
        if not isinstance(data, Mapping) or "actions" not in data:
            raise OfficialIPCError("Policy response is missing actions")
        actions = _decode_array_payload(
            data["actions"],
            name="actions",
            expected_dtype=np.dtype("float32"),
            expected_shape=ACTION_SHAPE,
        )
        if not np.all(np.isfinite(actions)):
            raise OfficialIPCError("Policy returned NaN or infinite actions")
        return actions

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def shutdown(self) -> None:
        if not self._closed:
            self._request({"type": "shutdown"})
            self.close()
