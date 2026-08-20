"""Authenticated local Unix IPC for split Octo-small/SimplerEnv evaluation."""

from __future__ import annotations

import time
from multiprocessing.connection import Client, Listener
from multiprocessing.context import AuthenticationError
from pathlib import Path
from typing import Any, Mapping

import numpy as np


IPC_PROTOCOL_VERSION = 1
ACTION_SHAPE = (1, 8, 7)
PROPRIO_SHAPE = (8,)


class OctoIPCError(RuntimeError):
    """Raised when the local Octo policy service rejects or loses a request."""


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
    if not isinstance(payload, Mapping):
        raise OctoIPCError(f"{name} payload must be a mapping")
    expected_keys = {"data", "dtype", "shape"}
    if set(payload) != expected_keys:
        raise OctoIPCError(
            f"{name} payload keys must be {sorted(expected_keys)}, "
            f"found {list(payload.keys())!r}"
        )
    dtype = payload["dtype"]
    if type(dtype) is not str or dtype != expected_dtype.name:
        raise OctoIPCError(
            f"{name} dtype must be {expected_dtype.name}, found {dtype!r}"
        )
    shape = payload["shape"]
    if type(shape) is not list or any(
        type(dimension) is not int or dimension <= 0 for dimension in shape
    ):
        raise OctoIPCError(f"{name} shape must contain positive integers")
    if expected_shape is not None and shape != list(expected_shape):
        raise OctoIPCError(
            f"{name} shape must be {list(expected_shape)}, found {shape!r}"
        )
    data = payload["data"]
    if type(data) is not bytes:
        raise OctoIPCError(
            f"{name} data must be bytes, found {type(data).__name__}"
        )
    expected_items = 1
    for dimension in shape:
        expected_items *= dimension
    expected_nbytes = expected_items * expected_dtype.itemsize
    if len(data) != expected_nbytes:
        raise OctoIPCError(
            f"{name} byte length must be {expected_nbytes}, found {len(data)}"
        )
    return np.frombuffer(data, dtype=expected_dtype).reshape(shape).copy(order="C")


def _validate_image(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[0] <= 0
        or array.shape[1] <= 0
        or array.shape[2] != 3
    ):
        raise OctoIPCError(
            "infer expected a uint8 HWC RGB image with positive height and width; "
            f"found {array.shape} {array.dtype}"
        )
    return np.ascontiguousarray(array)


def _validate_proprio(proprio: Any) -> np.ndarray:
    array = np.asarray(proprio)
    if array.dtype != np.float32 or array.shape != PROPRIO_SHAPE:
        raise OctoIPCError(
            "infer expected float32 proprio with shape (8,); "
            f"found {array.shape} {array.dtype}"
        )
    return np.ascontiguousarray(array)


def _validate_instruction(instruction: Any) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise OctoIPCError("infer instruction must be non-empty")
    return instruction.strip()


class _PolicySession:
    def __init__(self, *, policy: Any, metadata: Mapping[str, Any]):
        self.policy = policy
        self.metadata = {
            **dict(metadata),
            "protocol_version": IPC_PROTOCOL_VERSION,
        }
        self.generator: Any | None = None

    def route(self, request: Any) -> tuple[dict[str, Any], bool]:
        if not isinstance(request, Mapping):
            raise OctoIPCError("IPC request must be a mapping")
        request_type = request.get("type")
        if request_type == "metadata":
            return {"ok": True, "data": dict(self.metadata)}, False
        if request_type == "reset_rng":
            seed = int(request["seed"])
            self.generator = self.policy.make_generator(seed)
            return {"ok": True, "data": {"seed": seed}}, False
        if request_type == "infer":
            if self.generator is None:
                raise OctoIPCError("reset_rng must be called before infer")
            image = _decode_array_payload(
                request.get("image"),
                name="image",
                expected_dtype=np.dtype("uint8"),
            )
            image = _validate_image(image)
            proprio = _decode_array_payload(
                request.get("proprio"),
                name="proprio",
                expected_dtype=np.dtype("float32"),
                expected_shape=PROPRIO_SHAPE,
            )
            instruction = _validate_instruction(request.get("instruction"))
            prepared = self.policy.prepare_observation(image, proprio, instruction)
            actions = np.asarray(
                self.policy.predict_actions(prepared, generator=self.generator),
                dtype=np.float32,
            )
            if actions.shape != ACTION_SHAPE:
                raise OctoIPCError(
                    f"policy returned actions with shape {actions.shape}; "
                    f"expected {ACTION_SHAPE}"
                )
            if not np.all(np.isfinite(actions)):
                raise OctoIPCError("policy returned NaN or infinite actions")
            return {
                "ok": True,
                "data": {"actions": _encode_array_payload(actions)},
            }, False
        if request_type == "shutdown":
            return {"ok": True, "data": {"shutdown": True}}, True
        raise OctoIPCError(f"Unsupported IPC request type: {request_type!r}")


def serve_policy(
    *,
    socket_path: str | Path,
    authkey: bytes,
    policy: Any,
    metadata: Mapping[str, Any],
) -> None:
    target = Path(socket_path).expanduser().resolve()
    if not authkey:
        raise OctoIPCError("IPC authkey must be non-empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise OctoIPCError(f"IPC socket already exists: {target}")
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


class OctoIPCClient:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        authkey: bytes,
        connect_timeout: float = 30.0,
    ) -> None:
        self.socket_path = Path(socket_path).expanduser().resolve()
        if not authkey:
            raise OctoIPCError("IPC authkey must be non-empty")
        deadline = time.monotonic() + float(connect_timeout)
        while True:
            try:
                self._connection = Client(
                    str(self.socket_path),
                    family="AF_UNIX",
                    authkey=authkey,
                )
                break
            except (FileNotFoundError, ConnectionRefusedError) as error:
                if time.monotonic() >= deadline:
                    raise OctoIPCError(
                        f"Timed out connecting to policy server at {self.socket_path}"
                    ) from error
                time.sleep(0.05)
        self._closed = False

    def _request(self, payload: Mapping[str, Any]) -> Any:
        if self._closed:
            raise OctoIPCError("Policy client is closed")
        try:
            self._connection.send(dict(payload))
            response = self._connection.recv()
        except (EOFError, OSError) as error:
            raise OctoIPCError(f"Policy server connection failed: {error}") from error
        if not isinstance(response, Mapping):
            raise OctoIPCError("Policy server returned a non-mapping response")
        if not response.get("ok"):
            raise OctoIPCError(str(response.get("error", "Unknown policy server error")))
        return response.get("data")

    def metadata(self) -> dict[str, Any]:
        data = self._request({"type": "metadata"})
        if not isinstance(data, Mapping):
            raise OctoIPCError("Policy metadata must be a mapping")
        return dict(data)

    def reset_rng(self, seed: int) -> int:
        data = self._request({"type": "reset_rng", "seed": int(seed)})
        if not isinstance(data, Mapping) or "seed" not in data:
            raise OctoIPCError("Policy reset response is missing the seed")
        return int(data["seed"])

    def infer(
        self,
        image: np.ndarray,
        proprio: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        image_array = _validate_image(image)
        proprio_array = _validate_proprio(proprio)
        text = _validate_instruction(instruction)
        data = self._request(
            {
                "type": "infer",
                "image": _encode_array_payload(image_array),
                "proprio": _encode_array_payload(proprio_array),
                "instruction": text,
            }
        )
        if not isinstance(data, Mapping) or "actions" not in data:
            raise OctoIPCError("Policy response is missing the actions payload")
        actions = _decode_array_payload(
            data["actions"],
            name="actions",
            expected_dtype=np.dtype("float32"),
            expected_shape=ACTION_SHAPE,
        )
        if not np.all(np.isfinite(actions)):
            raise OctoIPCError("policy server returned NaN or infinite actions")
        return actions

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def shutdown(self) -> None:
        if not self._closed:
            self._request({"type": "shutdown"})
            self.close()
