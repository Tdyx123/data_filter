"""Model-free Trajectory Data Utility Score (TDUS).

The package deliberately keeps dataset-specific code behind ``DatasetAdapter``.
The scoring modules operate only on :class:`tdus.dataset.TrajectorySegment` and
NumPy embeddings, so adding another robotics dataset does not require changing
the TDUS algorithms.
"""

from .dataset import (
    DatasetAdapter,
    LeRobotDatasetAdapter,
    TrajectorySegment,
    create_dataset,
    register_dataset_adapter,
)


def select_top_k(*args, **kwargs):
    """Lazily import :func:`tdus.selector.select_top_k`."""

    from .selector import select_top_k as implementation

    return implementation(*args, **kwargs)


def select_budget(*args, **kwargs):
    """Lazily import :func:`tdus.selector.select_budget`."""

    from .selector import select_budget as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "DatasetAdapter",
    "LeRobotDatasetAdapter",
    "TrajectorySegment",
    "create_dataset",
    "register_dataset_adapter",
    "select_budget",
    "select_top_k",
]

__version__ = "0.1.0"
