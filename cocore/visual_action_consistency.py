"""Visual-action consistency importance for Cocore."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from cocore.action_variation import ACTION_VARIATION_TOP_K


def compute_step_visual_action_consistency(
    visual_features: np.ndarray,
    actions: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Compute FrameSkip-style VAC values for one complete episode."""

    visual = np.asarray(visual_features, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    tolerance = float(epsilon)
    if (
        visual.ndim != 2
        or action.ndim != 2
        or visual.shape[0] < 2
        or action.shape[0] != visual.shape[0]
        or visual.shape[1] == 0
        or action.shape[1] == 0
        or not np.all(np.isfinite(visual))
        or not np.all(np.isfinite(action))
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError(
            "visual-action consistency requires matching finite [time, dimension] inputs "
            "with at least two steps and a positive epsilon"
        )

    visual_difference = np.linalg.norm(np.diff(visual, axis=0), axis=1)
    action_difference = np.linalg.norm(np.diff(action, axis=0), axis=1)
    ratios = visual_difference / (action_difference + tolerance)
    if not np.all(np.isfinite(ratios)):
        raise ValueError("visual-action consistency produced a non-finite ratio")
    return np.concatenate([ratios[:1], ratios]).astype(np.float32)


def normalize_visual_action_consistency(
    values: np.ndarray,
    *,
    quantile_low: float,
    quantile_high: float,
    epsilon: float,
) -> np.ndarray:
    """Robustly scale candidate VAC values onto ``[0, 1]``."""

    scores = np.asarray(values, dtype=np.float32)
    low = float(quantile_low)
    high = float(quantile_high)
    tolerance = float(epsilon)
    if scores.ndim != 1 or len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("visual-action consistency scores must be a non-empty finite vector")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("visual-action consistency quantiles must satisfy 0 <= low < high <= 1")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("visual-action consistency epsilon must be finite and positive")
    lower = float(np.quantile(scores, low))
    upper = float(np.quantile(scores, high))
    denominator = max(upper - lower, tolerance)
    return np.clip((scores - lower) / denominator, 0.0, 1.0).astype(np.float32)


def visual_action_consistency_contract(
    *,
    quantile_low: Real,
    quantile_high: Real,
    epsilon: Real,
) -> dict[str, object]:
    """Return the serialized fixed VAC contract used by fingerprints."""

    return {
        "formula": "l2(v_t-v_t_minus_1)/(l2(a_t-a_t_minus_1)+epsilon)",
        "visual_input": "configured_encoder_full_episode_frame_features",
        "action_input": "robust_scaled_full_episode_actions",
        "visual_difference": "l2_norm_current_minus_previous",
        "action_difference": "l2_norm_current_minus_previous",
        "ratio": "visual_difference_over_action_difference_plus_epsilon",
        "first_step": "copy_first_valid_ratio",
        "clip_aggregation": "top_k_mean",
        "top_k": ACTION_VARIATION_TOP_K,
        "normalization": "clip_quantile_scale_to_zero_one",
        "quantile_low": float(quantile_low),
        "quantile_high": float(quantile_high),
        "epsilon": float(epsilon),
    }
