"""Action-variation importance and Cocore reliability fusion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np


ACTION_DIFFERENCE_WEIGHT = 2.0
ACTION_VARIANCE_WEIGHT = 1.0
ACTION_VARIANCE_FUTURE_WINDOW = 5
ACTION_VARIATION_TOP_K = 3
DEFAULT_RELIABILITY_METRICS = (
    "support",
    "progress",
    "action_variation",
    "visual_action_consistency",
)

RELIABILITY_METRICS = (
    *DEFAULT_RELIABILITY_METRICS,
    "non_dwell",
    "eef_jerk",
    "local_path_efficiency",
)


def compute_step_action_variation(actions: np.ndarray) -> np.ndarray:
    """Compute fixed-contract AVI values for every step in one episode."""

    values = np.asarray(actions, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("action variation requires finite [time, dimension] actions")

    differences = np.diff(values, axis=0, prepend=values[:1])
    difference_scores = np.linalg.norm(differences, axis=1)
    future_variance = np.zeros(len(values), dtype=np.float32)
    for index in range(len(values)):
        future = values[index + 1 : min(len(values), index + 1 + ACTION_VARIANCE_FUTURE_WINDOW)]
        if len(future) >= 2:
            future_variance[index] = float(np.var(future, axis=0).mean())
    return (
        ACTION_DIFFERENCE_WEIGHT * difference_scores + ACTION_VARIANCE_WEIGHT * future_variance
    ).astype(np.float32)


def top_k_mean(
    values: np.ndarray,
    *,
    k: int = ACTION_VARIATION_TOP_K,
) -> float:
    """Return the mean of the largest ``k`` values, using all values when shorter."""

    scores = np.asarray(values, dtype=np.float32)
    if scores.ndim != 1 or len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("action variation Top-K input must be a non-empty finite vector")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("action variation Top-K must be a positive integer")
    count = min(k, len(scores))
    return float(np.partition(scores, len(scores) - count)[-count:].mean())


def normalize_action_variation(
    values: np.ndarray,
    *,
    quantile_low: float,
    quantile_high: float,
    epsilon: float,
) -> np.ndarray:
    """Robustly scale clip AVI values onto ``[0, 1]``."""

    scores = np.asarray(values, dtype=np.float32)
    if scores.ndim != 1 or len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("action variation scores must be a non-empty finite vector")
    low = float(quantile_low)
    high = float(quantile_high)
    tolerance = float(epsilon)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("action variation quantiles must satisfy 0 <= low < high <= 1")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("action variation epsilon must be finite and positive")
    lower = float(np.quantile(scores, low))
    upper = float(np.quantile(scores, high))
    denominator = max(upper - lower, tolerance)
    return np.clip((scores - lower) / denominator, 0.0, 1.0).astype(np.float32)


def normalize_reliability_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    """Validate a non-empty metric subset and return canonical ordering."""

    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
        raise ValueError("reliability metrics must be a sequence of names")
    values = tuple(metrics)
    if not values:
        raise ValueError("reliability metrics cannot be empty")
    if any(not isinstance(metric, str) for metric in values):
        raise ValueError("reliability metrics must contain only names")
    if len(values) != len(set(values)):
        raise ValueError("reliability metrics cannot contain duplicates")
    unknown = sorted(set(values) - set(RELIABILITY_METRICS))
    if unknown:
        raise ValueError(f"reliability metrics contain unknown names: {unknown}")
    enabled = set(values)
    return tuple(metric for metric in RELIABILITY_METRICS if metric in enabled)


def fuse_reliability(
    support: np.ndarray,
    progress: np.ndarray,
    action_variation: np.ndarray,
    visual_action_consistency: np.ndarray,
    metrics: Sequence[str],
    *,
    min_reliability: float,
    non_dwell: np.ndarray | None = None,
    local_path_efficiency: np.ndarray | None = None,
    eef_jerk: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse the selected Cocore reliability components by geometric mean."""

    components = {
        "support": np.asarray(support, dtype=np.float32),
        "progress": np.asarray(progress, dtype=np.float32),
        "action_variation": np.asarray(action_variation, dtype=np.float32),
        "visual_action_consistency": np.asarray(
            visual_action_consistency,
            dtype=np.float32,
        ),
    }
    if non_dwell is not None:
        components["non_dwell"] = np.asarray(non_dwell, dtype=np.float32)
    if "non_dwell" in metrics and non_dwell is None:
        raise ValueError("non_dwell reliability requires dwell configuration and values")
    if eef_jerk is not None:
        components["eef_jerk"] = np.asarray(eef_jerk, dtype=np.float32)
    if "eef_jerk" in metrics and eef_jerk is None:
        raise ValueError("eef_jerk reliability requires computed values")
    shape = components["support"].shape
    if (
        len(shape) != 1
        or any(values.shape != shape for values in components.values())
        or any(not np.all(np.isfinite(values)) for values in components.values())
        or any(np.any((values < 0.0) | (values > 1.0)) for values in components.values())
    ):
        raise ValueError("reliability components must be matching finite [0, 1] vectors")
    enabled = normalize_reliability_metrics(metrics)
    minimum = float(min_reliability)
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum reliability must be finite and in [0, 1]")
    if local_path_efficiency is not None:
        path = np.asarray(local_path_efficiency, dtype=np.float64)
        if path.shape != shape or np.any(np.isinf(path)) or np.any((path < 0) | (path > 1)):
            raise ValueError("local_path_efficiency must be a matching [0, 1] or NaN vector")
        components["local_path_efficiency"] = path
    elif "local_path_efficiency" in enabled:
        raise ValueError("local_path_efficiency requires configuration and values")
    count = np.zeros(shape, dtype=np.int64)
    product = np.ones(shape, dtype=np.float64)
    for metric in enabled:
        values = components[metric]
        valid = ~np.isnan(values)
        product *= np.where(valid, values, 1.0)
        count += valid
    reliability = np.power(product, 1.0 / np.maximum(count, 1))
    return np.clip(reliability, minimum, 1.0).astype(np.float32)


def action_variation_contract(
    *,
    quantile_low: Real,
    quantile_high: Real,
    epsilon: Real,
) -> dict[str, object]:
    """Return the serialized fixed AVI contract used by cache fingerprints."""

    return {
        "input": "robust_scaled_full_episode_actions",
        "difference": "l2_norm_current_minus_previous",
        "difference_weight": ACTION_DIFFERENCE_WEIGHT,
        "first_step_difference": 0.0,
        "future_window": ACTION_VARIANCE_FUTURE_WINDOW,
        "future_variance": "mean_dimension_population_variance",
        "future_variance_weight": ACTION_VARIANCE_WEIGHT,
        "future_boundary": "truncate_available_less_than_two_is_zero",
        "clip_aggregation": "top_k_mean",
        "top_k": ACTION_VARIATION_TOP_K,
        "normalization": "clip_quantile_scale_to_zero_one",
        "quantile_low": float(quantile_low),
        "quantile_high": float(quantile_high),
        "epsilon": float(epsilon),
    }
