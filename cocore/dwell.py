"""Time-weighted low-change residency, measured in original state units."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

DWELL_FIELDS = ("dwell_ratio", "non_dwell")
DWELL_CACHE_FIELDS = ("dwell_state_sequences", "dwell_timestamps", *DWELL_FIELDS)


def resolve_dwell_config(value: object) -> dict[str, Any] | None:
    """Validate explicit thresholds and supply only the continuous-mode default."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("dwell must be a mapping")
    result = dict(value)
    allowed = {
        "position_speed_threshold",
        "gripper_speed_threshold",
        "angular_speed_threshold",
        "gripper_mode",
    }
    if result.keys() - allowed:
        raise ValueError("dwell contains unsupported fields")
    mode = result.setdefault("gripper_mode", "continuous")
    if mode not in ("continuous", "binary"):
        raise ValueError("dwell gripper_mode must be continuous or binary")
    required = {"position_speed_threshold", "angular_speed_threshold"}
    if mode == "continuous":
        required.add("gripper_speed_threshold")
    for key in required | (result.keys() - {"gripper_mode"}):
        threshold = result.get(key)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not np.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError(f"dwell {key} must be explicitly set, finite and positive")
        result[key] = float(threshold)
    return result


def dwell_contract(config: Mapping[str, Any]) -> dict[str, Any] | None:
    settings = resolve_dwell_config(config.get("dwell"))
    if settings is None:
        return None
    return {
        "version": 1,
        "profile": config["prototypes"]["profile"],
        "weighting": "elapsed_time",
        "state_key": "observation.state",
        "position_axes": [0, 1, 2],
        "orientation_axes": [3, 4, 5],
        "gripper_axis": 7,
        **settings,
    }


def _quaternions(orientation: np.ndarray, profile: str) -> np.ndarray:
    """Return unit wxyz quaternions for rotvec or extrinsic XYZ Euler angles."""
    if profile == "libero":
        angle = np.linalg.norm(orientation, axis=1)
        scale = 0.5 * np.sinc(angle / (2 * np.pi))
        return np.column_stack((np.cos(angle / 2), orientation * scale[:, None]))
    if profile != "bridge_v2":
        raise ValueError("dwell profile must be libero or bridge_v2")
    cr, cp, cy = np.cos(orientation / 2).T
    sr, sp, sy = np.sin(orientation / 2).T
    return np.column_stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def compute_dwell_ratio(
    state: ArrayLike, timestamps: ArrayLike, *, profile: str, config: Mapping[str, Any]
) -> float:
    """Compute residency using only the L-1 internal adjacent frame pairs."""
    settings = resolve_dwell_config(config)
    if settings is None:
        raise ValueError("dwell configuration is required")
    values = np.asarray(state, dtype=np.float64)
    time = np.asarray(timestamps, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 8
        or not np.all(np.isfinite(values))
        or time.shape != (len(values),)
        or not np.all(np.isfinite(time))
    ):
        raise ValueError("dwell requires finite [L>=2, dim>=8] states and matching timestamps")
    dt = np.diff(time)
    if not np.all(dt > 0) or not np.isfinite(dt.sum()):
        raise ValueError("dwell timestamps must be strictly increasing with finite duration")
    position_speed = np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1) / dt
    quaternion = _quaternions(values[:, 3:6], profile)
    # q and -q represent the same rotation. Chord lengths give a stable
    # shortest angle near both zero and pi without arccos cancellation.
    a, b = quaternion[:-1], quaternion[1:]
    b = b * np.where(np.sum(a * b, axis=1) < 0, -1.0, 1.0)[:, None]
    angle = 4 * np.arctan2(np.linalg.norm(a - b, axis=1), np.linalg.norm(a + b, axis=1))
    low = (position_speed < settings["position_speed_threshold"]) & (
        angle / dt < settings["angular_speed_threshold"]
    )
    if settings["gripper_mode"] == "binary":
        low &= values[1:, 7] == values[:-1, 7]
    else:
        low &= np.abs(np.diff(values[:, 7])) / dt < settings["gripper_speed_threshold"]
    return float(np.sum(dt[low]) / np.sum(dt))


def dwell_summary(
    all_rows: Sequence[Mapping[str, Any]], selected_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, float | None]]:
    """Arithmetic means of per-clip time-weighted ratios (not pooled durations)."""
    return {
        field: {
            "all_mean": float(np.mean([row[field] for row in all_rows])) if all_rows else None,
            "selected_mean": float(np.mean([row[field] for row in selected_rows]))
            if selected_rows
            else None,
        }
        for field in DWELL_FIELDS
    }
