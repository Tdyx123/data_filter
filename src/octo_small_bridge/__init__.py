"""Octo-small training and evaluation support for BridgeData V2."""

from typing import Any

__all__ = [
    "BridgeDistributedBatchSampler",
    "BridgeFrameDataset",
    "BridgeFrameRef",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .data import BridgeDistributedBatchSampler, BridgeFrameDataset, BridgeFrameRef

        return {
            "BridgeDistributedBatchSampler": BridgeDistributedBatchSampler,
            "BridgeFrameDataset": BridgeFrameDataset,
            "BridgeFrameRef": BridgeFrameRef,
        }[name]
    raise AttributeError(name)
