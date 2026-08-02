"""Within-fragment SQCN quality metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .scaling import robust_unit_interval


@dataclass(frozen=True)
class RawQuality:
    action_delta: float
    state_transition: float
    motion_efficiency: float


def raw_quality(actions: np.ndarray, state: np.ndarray) -> RawQuality:
    """Compute quality ingredients from normalized action/state sequences."""

    action_values = np.asarray(actions, dtype=np.float32)
    state_values = np.asarray(state, dtype=np.float32)
    if action_values.ndim != 2 or state_values.ndim != 2:
        raise ValueError("actions and state must have shape [time, dim]")
    if len(action_values) != len(state_values):
        raise ValueError("actions and state must have the same time dimension")
    if not np.all(np.isfinite(action_values)) or not np.all(np.isfinite(state_values)):
        raise ValueError("actions/state contains NaN or infinity")
    if len(action_values) < 2:
        return RawQuality(0.0, 0.0, 0.0)
    action_delta = float(
        np.linalg.norm(np.diff(action_values, axis=0), axis=1).mean()
    )
    state_transition = float(
        np.linalg.norm(np.diff(state_values, axis=0), axis=1).mean()
    )
    motion_efficiency = float(
        np.linalg.norm(state_values[-1] - state_values[0])
        / max(len(state_values) - 1, 1)
    )
    return RawQuality(action_delta, state_transition, motion_efficiency)


def score_quality(
    raw_values: Sequence[RawQuality],
    *,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    epsilon: float = 1.0e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray | tuple[float, float]]]:
    """Normalize and combine quality ingredients using fixed 0.4/0.3/0.3 weights."""

    if not raw_values:
        return np.empty((0,), dtype=np.float32), {}
    action = np.asarray([value.action_delta for value in raw_values])
    transition = np.asarray([value.state_transition for value in raw_values])
    efficiency = np.asarray([value.motion_efficiency for value in raw_values])
    action_scaled, action_bounds = robust_unit_interval(
        action,
        low=quantile_low,
        high=quantile_high,
        epsilon=epsilon,
    )
    transition_scaled, transition_bounds = robust_unit_interval(
        transition,
        low=quantile_low,
        high=quantile_high,
        epsilon=epsilon,
    )
    efficiency_scaled, efficiency_bounds = robust_unit_interval(
        efficiency,
        low=quantile_low,
        high=quantile_high,
        epsilon=epsilon,
    )
    smooth = 1.0 - action_scaled
    quality = 0.4 * smooth + 0.3 * transition_scaled + 0.3 * efficiency_scaled
    details: dict[str, np.ndarray | tuple[float, float]] = {
        "action_smooth": smooth,
        "state_transition": transition_scaled,
        "motion_efficiency": efficiency_scaled,
        "action_delta_bounds": action_bounds,
        "state_transition_bounds": transition_bounds,
        "motion_efficiency_bounds": efficiency_bounds,
    }
    return np.clip(quality, 0.0, 1.0).astype(np.float32), details
