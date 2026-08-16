"""Memory-efficient strict loading for released StarVLA state dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class StarVLACheckpointError(RuntimeError):
    """Raised when a StarVLA checkpoint cannot be loaded exactly."""


@dataclass(frozen=True)
class CheckpointLoadReport:
    path: Path
    tensor_count: int
    parameter_bytes: int
    dtypes: tuple[str, ...]


def load_strict_checkpoint(
    model: Any,
    path: str | Path,
    *,
    expected_tensor_count: int,
    expected_dtype: Any | None = None,
    torch_module: Any | None = None,
) -> CheckpointLoadReport:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise StarVLACheckpointError(f"StarVLA checkpoint does not exist: {checkpoint}")
    if torch_module is None:
        import torch as torch_module
    try:
        state = torch_module.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as error:
        raise StarVLACheckpointError(f"Could not mmap StarVLA checkpoint: {error}") from error
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not torch_module.is_tensor(tensor)
        for name, tensor in state.items()
    ):
        raise StarVLACheckpointError("StarVLA checkpoint must be a string-to-tensor mapping")
    if len(state) != int(expected_tensor_count):
        raise StarVLACheckpointError(
            f"StarVLA checkpoint contains {len(state)} tensors; "
            f"expected {expected_tensor_count}"
        )
    if expected_dtype is not None:
        wrong_dtypes = {
            str(tensor.dtype)
            for tensor in state.values()
            if tensor.dtype != expected_dtype
        }
        if wrong_dtypes:
            raise StarVLACheckpointError(
                f"StarVLA checkpoint tensors must have dtype {expected_dtype}; "
                f"found {sorted(wrong_dtypes)}"
            )
    parameter_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    dtypes = tuple(sorted({str(tensor.dtype) for tensor in state.values()}))
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except Exception as error:
        raise StarVLACheckpointError(f"StarVLA strict checkpoint load failed: {error}") from error
    return CheckpointLoadReport(
        path=checkpoint,
        tensor_count=len(state),
        parameter_bytes=int(parameter_bytes),
        dtypes=dtypes,
    )
