"""Shared robust scalar normalization."""

from __future__ import annotations

import numpy as np


def robust_unit_interval(
    values: np.ndarray,
    *,
    low: float = 0.01,
    high: float = 0.99,
    epsilon: float = 1.0e-8,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Robustly map finite scalar values to ``[0, 1]``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contain NaN or infinity")
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= low <= high <= 1")
    lower, upper = np.quantile(array, [low, high])
    scaled = np.clip(
        (array - lower) / max(float(upper - lower), epsilon),
        0.0,
        1.0,
    )
    return scaled.astype(np.float32), (float(lower), float(upper))
