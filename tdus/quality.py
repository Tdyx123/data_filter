"""Within-segment quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .dataset import TrajectorySegment
from .encoder import NumericNormalizers


def robust_unit_interval(
    values: np.ndarray,
    *,
    low: float = 0.01,
    high: float = 0.99,
    epsilon: float = 1.0e-8,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Robustly map a scalar sample metric to [0, 1]."""

    values = np.asarray(values, dtype=np.float64)
    lower, upper = np.quantile(values, [low, high])
    scaled = np.clip((values - lower) / max(float(upper - lower), epsilon), 0.0, 1.0)
    return scaled.astype(np.float32), (float(lower), float(upper))


@dataclass(frozen=True)
class RawQuality:
    action_delta: float
    state_transition: float
    motion_efficiency: float


def raw_quality(
    segment: TrajectorySegment,
    normalizers: NumericNormalizers,
    *,
    visual_state: np.ndarray | None = None,
) -> RawQuality:
    """Compute unnormalized quality ingredients for a single segment."""

    if segment.length < 2:
        return RawQuality(0.0, 0.0, 0.0)
    actions = normalizers.action(segment.actions)
    action_delta = float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean())

    state = normalizers.state_sequence(segment)
    if state is None:
        state = visual_state
    if state is None or len(state) < 2:
        state_transition = 0.0
        motion_efficiency = 0.0
    else:
        state = np.asarray(state, dtype=np.float32)
        state_transition = float(
            np.linalg.norm(np.diff(state, axis=0), axis=1).mean()
        )
        motion_efficiency = float(
            np.linalg.norm(state[-1] - state[0]) / max(segment.length - 1, 1)
        )
    return RawQuality(action_delta, state_transition, motion_efficiency)


def score_quality(
    raw_values: Sequence[RawQuality],
    config: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray | tuple[float, float]]]:
    """Normalize quality ingredients and return the weighted quality score."""

    if not raw_values:
        return np.empty((0,), dtype=np.float32), {}
    low = float(config.get("quantile_low", 0.01))
    high = float(config.get("quantile_high", 0.99))
    epsilon = float(config.get("epsilon", 1.0e-8))
    action = np.asarray([value.action_delta for value in raw_values])
    transition = np.asarray([value.state_transition for value in raw_values])
    efficiency = np.asarray([value.motion_efficiency for value in raw_values])
    action_scaled, action_bounds = robust_unit_interval(
        action, low=low, high=high, epsilon=epsilon
    )
    transition_scaled, transition_bounds = robust_unit_interval(
        transition, low=low, high=high, epsilon=epsilon
    )
    efficiency_scaled, efficiency_bounds = robust_unit_interval(
        efficiency, low=low, high=high, epsilon=epsilon
    )
    smooth = 1.0 - action_scaled
    quality = (
        float(config.get("action_smooth_weight", 0.4)) * smooth
        + float(config.get("state_transition_weight", 0.3)) * transition_scaled
        + float(config.get("motion_efficiency_weight", 0.3)) * efficiency_scaled
    )
    details: dict[str, np.ndarray | tuple[float, float]] = {
        "action_smooth": smooth,
        "state_transition": transition_scaled,
        "motion_efficiency": efficiency_scaled,
        "action_delta_bounds": action_bounds,
        "state_transition_bounds": transition_bounds,
        "motion_efficiency_bounds": efficiency_bounds,
    }
    return np.clip(quality, 0.0, 1.0).astype(np.float32), details
