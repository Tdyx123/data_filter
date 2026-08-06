"""Process-wide runtime safeguards for trajectory data consumers."""

from __future__ import annotations

import os


NATIVE_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
THREAD_LIMIT_VARIABLE = "TRAJECTORY_DATA_NUM_THREADS"
MIN_NATIVE_THREADS = 1
MAX_NATIVE_THREADS = 64


def configure_native_thread_pools() -> int:
    """Limit native math pools before NumPy or SciPy can initialize them."""

    configured = os.environ.get(THREAD_LIMIT_VARIABLE, "1")
    try:
        limit = int(configured)
    except ValueError as error:
        raise ValueError(
            f"{THREAD_LIMIT_VARIABLE} must be an integer in "
            f"[{MIN_NATIVE_THREADS}, {MAX_NATIVE_THREADS}]"
        ) from error
    if not MIN_NATIVE_THREADS <= limit <= MAX_NATIVE_THREADS:
        raise ValueError(
            f"{THREAD_LIMIT_VARIABLE} must be an integer in "
            f"[{MIN_NATIVE_THREADS}, {MAX_NATIVE_THREADS}]"
        )
    value = str(limit)
    for variable in NATIVE_THREAD_VARIABLES:
        os.environ[variable] = value
    return limit


__all__ = [
    "MAX_NATIVE_THREADS",
    "MIN_NATIVE_THREADS",
    "NATIVE_THREAD_VARIABLES",
    "THREAD_LIMIT_VARIABLE",
    "configure_native_thread_pools",
]
