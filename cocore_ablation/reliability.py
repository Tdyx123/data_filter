"""Reliability fusion variants used by Cocore ablation graphs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def fuse_reliability(
    support: np.ndarray,
    progress: np.ndarray,
    metrics: Sequence[str],
    *,
    min_reliability: float,
) -> np.ndarray:
    support_values = np.asarray(support, dtype=np.float32)
    progress_values = np.asarray(progress, dtype=np.float32)
    if (
        support_values.ndim != 1
        or support_values.shape != progress_values.shape
        or not np.all(np.isfinite(support_values))
        or not np.all(np.isfinite(progress_values))
    ):
        raise ValueError("reliability components must be matching finite vectors")
    enabled = tuple(metrics)
    if len(enabled) != len(set(enabled)) or set(enabled) - {"support", "progress"}:
        raise ValueError("reliability metrics must be a unique support/progress subset")
    reliability = np.ones_like(support_values, dtype=np.float32)
    if "support" in enabled:
        reliability *= np.sqrt(np.maximum(support_values, 0.0))
    if "progress" in enabled:
        reliability *= np.sqrt(np.maximum(progress_values, 0.0))
    return np.clip(reliability, float(min_reliability), 1.0).astype(np.float32)
