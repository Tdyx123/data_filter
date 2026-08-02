"""Fixed SQCN component aggregation."""

from __future__ import annotations

import numpy as np


def compute_sqcn(
    quality: np.ndarray,
    coverage: np.ndarray,
    novelty: np.ndarray,
) -> np.ndarray:
    """Combine Q/C/N using the fixed ``0.8/0.1/0.1`` definition."""

    components = [
        np.asarray(quality, dtype=np.float64),
        np.asarray(coverage, dtype=np.float64),
        np.asarray(novelty, dtype=np.float64),
    ]
    if len({component.shape for component in components}) != 1:
        raise ValueError("SQCN components must have matching shapes")
    if any(
        not np.all(np.isfinite(component))
        or np.any(component < 0.0)
        or np.any(component > 1.0)
        for component in components
    ):
        raise ValueError("SQCN components must contain finite values in [0, 1]")
    result = 0.8 * components[0] + 0.1 * components[1] + 0.1 * components[2]
    return np.clip(result, 0.0, 1.0).astype(np.float32)
