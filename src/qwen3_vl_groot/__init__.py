"""Qwen3-VL + GR00T-style Bridge policy."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qwen3-vl-groot-bridge")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]

