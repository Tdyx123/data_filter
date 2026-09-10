"""Reliability fusion for subsets of the support_old/action_jump baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cocore.action_variation import fuse_reliability as fuse_cocore_reliability
from .config import _validate_metrics


def fuse_reliability(
    support_old: np.ndarray,
    action_jump: np.ndarray,
    metrics: Sequence[str],
    *,
    min_reliability: float,
) -> np.ndarray:
    enabled = _validate_metrics(metrics)
    neutral = np.ones_like(support_old, dtype=np.float32)
    # Validate components even when reliability is disabled.
    fused = fuse_cocore_reliability(
        neutral, neutral, neutral, neutral,
        enabled or ["support_old", "action_jump"],
        support_old=support_old,
        action_jump=action_jump,
        min_reliability=min_reliability,
    )
    return fused if enabled else np.ones_like(fused)
