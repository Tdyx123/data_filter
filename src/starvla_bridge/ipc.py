"""Authenticated local Unix IPC for the split StarVLA/SimplerEnv runtime."""

from __future__ import annotations

import random
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


IPC_PROTOCOL_VERSION = 2


class StarVLAIPCError(RuntimeError):
    """Raised when the local policy service rejects or loses a request."""


def seed_everything(seed: int) -> None:
    import torch

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _encode_array_payload(array: Any) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(np.asarray(array))
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
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(payload, Mapping):
        raise StarVLAIPCError(f"{name} payload must be a mapping")
    expected_keys = {"data", "dtype", "shape"}
    if set(payload) != expected_keys:
        raise StarVLAIPCError(
            f"{name} payload keys must be {sorted(expected_keys)}, "
            f"found {list(payload.keys())!r}"
        )
    dtype = payload["dtype"]
    if type(dtype) is not str or dtype != expected_dtype.name:
        raise StarVLAIPCError(
            f"{name} dtype must be {expected_dtype.name}, found {dtype!r}"
        )
    shape = payload["shape"]
    expected_shape_list = list(expected_shape)
    if (
        type(shape) is not list
        or any(type(dimension) is not int for dimension in shape)
        or shape != expected_shape_list
    ):
        raise StarVLAIPCError(
            f"{name} shape must be {expected_shape_list}, found {shape!r}"
        )
    data = payload["data"]
    if type(data) is not bytes:
        raise StarVLAIPCError(
            f"{name} data must be bytes, found {type(data).__name__}"
        )
    expected_nbytes = expected_dtype.itemsize
    for dimension in expected_shape:
        expected_nbytes *= dimension
    if len(data) != expected_nbytes:
        raise StarVLAIPCError(
            f"{name} byte length must be {expected_nbytes}, found {len(data)}"
        )
    return np.frombuffer(data, dtype=expected_dtype).reshape(expected_shape).copy(order="C")


def _validate_image_array(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.shape != (224, 224, 3) or array.dtype != np.uint8:
        raise StarVLAIPCError(
            f"infer expected a uint8 image with shape (224, 224, 3), found "
            f"{array.shape} {array.dtype}"
        )
    return np.ascontiguousarray(array)


def _validate_instruction(instruction: Any) -> str:
    text = str(instruction).strip()
    if not text:
        raise StarVLAIPCError("infer instruction must be non-empty")
    return text


def _route_request(
    request: Any,
    *,
    policy: Any,
    metadata: Mapping[str, Any],
    seed_callback: Callable[[int], Any],
) -> tuple[dict[str, Any], bool]:
    if not isinstance(request, Mapping):
        raise StarVLAIPCError("IPC request must be a mapping")
    request_type = str(request.get("type", ""))
    if request_type == "metadata":
        return {"ok": True, "data": dict(metadata)}, False
    if request_type == "reset_rng":
        seed = int(request["seed"])
        seed_callback(seed)
        return {"ok": True, "data": {"seed": seed}}, False
    if request_type == "infer":
        image = _decode_array_payload(
            request.get("image"),
            name="image",
            expected_dtype=np.dtype("uint8"),
            expected_shape=(224, 224, 3),
        )
        instruction = _validate_instruction(request.get("instruction"))
        actions = np.asarray(policy.predict_actions(image, instruction), dtype=np.float32)
        if actions.shape != (1, 16, 7):
            raise StarVLAIPCError(
                f"policy returned actions with shape {actions.shape}; expected (1, 16, 7)"
            )
        if not np.all(np.isfinite(actions)):
            raise StarVLAIPCError("policy returned NaN or infinite actions")
        return {
            "ok": True,
            "data": {"actions": _encode_array_payload(actions)},
        }, False
    if request_type == "shutdown":
        return {"ok": True, "data": {"shutdown": True}}, True
    raise StarVLAIPCError(f"Unsupported IPC request type: {request_type!r}")


def serve_policy(
    *,
    socket_path: str | Path,
    authkey: bytes,
    policy: Any,
    metadata: Mapping[str, Any],
    seed_callback: Callable[[int], Any] = seed_everything,
) -> None:
    target = Path(socket_path).expanduser().resolve()
    if not authkey:
        raise StarVLAIPCError("IPC authkey must be non-empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise StarVLAIPCError(f"IPC socket already exists: {target}")
    complete_metadata = {"protocol_version": IPC_PROTOCOL_VERSION, **dict(metadata)}
    listener = Listener(str(target), family="AF_UNIX", authkey=authkey)
    target.chmod(0o600)
    shutting_down = False
    try:
        while not shutting_down:
            connection = listener.accept()
            try:
                while not shutting_down:
                    try:
                        request = connection.recv()
                    except EOFError:
                        break
                    try:
                        response, shutting_down = _route_request(
                            request,
                            policy=policy,
                            metadata=complete_metadata,
                            seed_callback=seed_callback,
                        )
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


class StarVLAIPCClient:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        authkey: bytes,
        connect_timeout: float = 30.0,
    ):
        self.socket_path = Path(socket_path).expanduser().resolve()
        if not authkey:
            raise StarVLAIPCError("IPC authkey must be non-empty")
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
                    raise StarVLAIPCError(
                        f"Timed out connecting to policy server at {self.socket_path}"
                    ) from error
                time.sleep(0.05)
        self._closed = False

    def _request(self, payload: Mapping[str, Any]) -> Any:
        if self._closed:
            raise StarVLAIPCError("Policy client is closed")
        try:
            self._connection.send(dict(payload))
            response = self._connection.recv()
        except (EOFError, OSError) as error:
            raise StarVLAIPCError(f"Policy server connection failed: {error}") from error
        if not isinstance(response, Mapping):
            raise StarVLAIPCError("Policy server returned a non-mapping response")
        if not response.get("ok"):
            raise StarVLAIPCError(str(response.get("error", "Unknown policy server error")))
        return response.get("data")

    def metadata(self) -> dict[str, Any]:
        data = self._request({"type": "metadata"})
        if not isinstance(data, Mapping):
            raise StarVLAIPCError("Policy metadata must be a mapping")
        return dict(data)

    def reset_rng(self, seed: int) -> int:
        data = self._request({"type": "reset_rng", "seed": int(seed)})
        return int(data["seed"])

    def infer(self, image: np.ndarray, instruction: str) -> np.ndarray:
        image_array = _validate_image_array(image)
        text = _validate_instruction(instruction)
        data = self._request(
            {
                "type": "infer",
                "image": _encode_array_payload(image_array),
                "instruction": text,
            }
        )
        if not isinstance(data, Mapping) or "actions" not in data:
            raise StarVLAIPCError("Policy server response is missing the actions payload")
        actions = _decode_array_payload(
            data["actions"],
            name="actions",
            expected_dtype=np.dtype("float32"),
            expected_shape=(1, 16, 7),
        )
        if not np.all(np.isfinite(actions)):
            raise StarVLAIPCError("policy server returned NaN or infinite actions")
        return actions

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def shutdown(self) -> None:
        if not self._closed:
            self._request({"type": "shutdown"})
            self.close()
