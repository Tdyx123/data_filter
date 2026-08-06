"""Model-free Trajectory Data Utility Score (TDUS).

Dataset-specific code lives in the independent :mod:`trajectory_data` package.
TDUS scoring modules operate only on dataset-neutral segments and NumPy
embeddings.
"""

import os as _os


def _forward_legacy_thread_limit() -> None:
    """Map the legacy TDUS override to the shared trajectory-data setting."""

    if "TRAJECTORY_DATA_NUM_THREADS" in _os.environ:
        return
    configured = _os.environ.get("TDUS_NUM_THREADS")
    if configured is None:
        return
    try:
        limit = min(max(int(configured), 1), 64)
    except ValueError:
        limit = 1
    _os.environ["TRAJECTORY_DATA_NUM_THREADS"] = str(limit)


_forward_legacy_thread_limit()

# This must follow the legacy mapping and runs the shared startup bootstrap.
import trajectory_data as _trajectory_data  # noqa: E402, F401


def select_top_k(*args, **kwargs):
    """Lazily import :func:`tdus.selector.select_top_k`."""

    from .selector import select_top_k as implementation

    return implementation(*args, **kwargs)


def select_budget(*args, **kwargs):
    """Lazily import :func:`tdus.selector.select_budget`."""

    from .selector import select_budget as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "select_budget",
    "select_top_k",
]

__version__ = "0.1.0"
