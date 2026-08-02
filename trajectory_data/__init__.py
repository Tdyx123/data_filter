"""Shared dataset adapters for trajectory-level robotics data."""

from .core import (
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
from .lerobot import LeRobotDatasetAdapter

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
