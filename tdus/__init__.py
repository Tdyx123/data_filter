"""Model-free Trajectory Data Utility Score (TDUS).

The package deliberately keeps dataset-specific code behind ``DatasetAdapter``.
The scoring modules operate only on :class:`tdus.dataset.TrajectorySegment` and
NumPy embeddings, so adding another robotics dataset does not require changing
the TDUS algorithms.
"""

import os as _os


def _configure_native_thread_pools() -> None:
    """Prevent each spawned data worker from creating a full CPU thread pool."""

    variables = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    configured = _os.environ.get("TDUS_NUM_THREADS")
    if configured is not None:
        try:
            limit = min(max(int(configured), 1), 64)
        except ValueError:
            limit = 1
        for variable in variables:
            _os.environ[variable] = str(limit)
        return

    for variable in variables:
        _os.environ.setdefault(variable, "1")
    try:
        openblas_threads = int(_os.environ["OPENBLAS_NUM_THREADS"])
    except ValueError:
        openblas_threads = 1
    if openblas_threads < 1 or openblas_threads > 64:
        _os.environ["OPENBLAS_NUM_THREADS"] = str(min(max(openblas_threads, 1), 64))


_configure_native_thread_pools()

from .dataset import (  # noqa: E402 - thread limits must precede NumPy import
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
