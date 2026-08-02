"""Model-free Trajectory Data Utility Score (TDUS).

Dataset-specific code lives in the independent :mod:`trajectory_data` package.
TDUS scoring modules operate only on dataset-neutral segments and NumPy
embeddings.
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
