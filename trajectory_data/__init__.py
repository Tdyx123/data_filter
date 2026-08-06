"""Shared dataset adapters for trajectory-level robotics data."""

from .runtime import configure_native_thread_pools as _configure_native_thread_pools

_configure_native_thread_pools()

from .core import (  # noqa: E402
    DatasetAdapter,
    DatasetValidationError,
    EpisodeData,
    EpisodeRecord,
    TrajectorySegment,
    aligned_chunk_windows,
    create_dataset,
    register_dataset_adapter,
    segment_episode,
)
from .lerobot import LeRobotDatasetAdapter  # noqa: E402

register_dataset_adapter("lerobot", LeRobotDatasetAdapter)

__all__ = [
    "DatasetAdapter",
    "DatasetValidationError",
    "EpisodeData",
    "EpisodeRecord",
    "LeRobotDatasetAdapter",
    "TrajectorySegment",
    "aligned_chunk_windows",
    "create_dataset",
    "register_dataset_adapter",
    "segment_episode",
]
