"""Strict model-process runtime contract for the released StarVLA checkpoint."""

from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Mapping
from typing import Any


MODEL_PYTHON_VERSION = (3, 12, 12)
MODEL_PACKAGE_VERSIONS = {
    "torch": "2.10.0+cu128",
    "transformers": "5.2.0",
    "numpy": "2.2.0",
    "diffusers": "0.38.0",
    "accelerate": "1.13.0",
}


class StarVLARuntimeError(RuntimeError):
    """Raised before loading when the dedicated model runtime is incompatible."""


def _discover_versions(torch_module: Any) -> dict[str, str]:
    discovered: dict[str, str] = {"torch": str(torch_module.__version__)}
    for package in MODEL_PACKAGE_VERSIONS:
        if package == "torch":
            continue
        try:
            discovered[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise StarVLARuntimeError(
                f"StarVLA model runtime requires {package}=="
                f"{MODEL_PACKAGE_VERSIONS[package]}; found not installed"
            ) from error
    return discovered


def validate_model_runtime(
    *,
    device: str,
    python_version: tuple[int, int, int] | None = None,
    package_versions: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
) -> dict[str, str]:
    """Validate exact package versions plus the mandatory CUDA/BF16 capability."""

    current_python = tuple(python_version or sys.version_info[:3])
    if current_python != MODEL_PYTHON_VERSION:
        expected = ".".join(str(part) for part in MODEL_PYTHON_VERSION)
        actual = ".".join(str(part) for part in current_python)
        raise StarVLARuntimeError(
            f"StarVLA model runtime requires Python {expected}; found {actual}"
        )

    if torch_module is None:
        import torch as torch_module
    discovered = package_versions or _discover_versions(torch_module)
    normalized = {str(name): str(version) for name, version in discovered.items()}
    for package, expected in MODEL_PACKAGE_VERSIONS.items():
        actual = normalized.get(package)
        if actual != expected:
            raise StarVLARuntimeError(
                f"StarVLA model runtime requires {package}=={expected}; "
                f"found {actual or 'not installed'}"
            )

    if not str(device).startswith("cuda"):
        raise StarVLARuntimeError("Released StarVLA checkpoint requires a CUDA device")
    if not torch_module.cuda.is_available():
        raise StarVLARuntimeError(f"CUDA device is unavailable: {device}")
    if not torch_module.cuda.is_bf16_supported():
        raise StarVLARuntimeError(f"CUDA device does not support BF16: {device}")

    return {
        "python": ".".join(str(part) for part in current_python),
        **normalized,
    }
