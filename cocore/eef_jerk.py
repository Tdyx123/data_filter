"""Translational end-effector jerk in m/s³ and its optional reliability score."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

JERK_FIELDS = ("eef_jerk_raw", "eef_jerk", "eef_jerk_valid", "eef_jerk_reason")
JERK_CACHE_FIELDS = ("eef_jerk_positions", "eef_jerk_timestamps", *JERK_FIELDS)


def jerk_contract(config):
    if "eef_jerk" not in config["reliability_metrics"]:
        return None
    return {
        "version": 1,
        "profile": config["prototypes"]["profile"],
        "state_key": "observation.state",
        "position_axes": [0, 1, 2],
        "coordinate_frame": "fixed",
        "position_unit": "m",
        "time_unit": "s",
        "raw_unit": "m/s^3",
        "estimator": "clip_internal_third_difference",
        "aggregation": "mean_l2_norm",
        "dt": "median",
        "rtol": 1e-4,
        "atol": 1e-8,
        "smoothing": "none",
        "invalid_policy": "exclude_from_graph",
        "normalization": "reverse_quantiles_of_jerk_valid_scanned_candidates",
        **{key: config["encoding"][key] for key in ("quantile_low", "quantile_high", "epsilon")},
    }


def compute_eef_jerk(positions, timestamps):
    """Return (mean jerk, reason); malformed/nonfinite inputs remain hard errors."""
    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or t.shape != (len(p),)
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(t))
    ):
        raise ValueError("eef_jerk requires finite [L, 3] positions and matching timestamps")
    if len(p) < 4:
        return np.nan, "too_short"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        intervals = np.diff(t)
        if not np.all(np.isfinite(intervals)):
            return np.nan, "overflow"
        if np.any(intervals <= 0):
            return np.nan, "non_increasing_time"
        dt = np.median(intervals)
        if not np.allclose(intervals, dt, rtol=1e-4, atol=1e-8):
            return np.nan, "non_uniform_time"
        denominator = dt**3
        if not np.isfinite(denominator) or denominator <= 0:
            return np.nan, "overflow"
        jerk = np.diff(p, n=3, axis=0) / denominator
        # hypot avoids intermediate squared-norm overflow for finite large vectors.
        value = np.hypot.reduce(jerk, axis=1).mean()
    if not np.isfinite(value):
        return np.nan, "overflow"
    return float(value), ""


def normalize_eef_jerk(raw, *, quantile_low, quantile_high, epsilon):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 1 or np.any(np.isinf(raw)) or np.any(raw < 0):
        raise ValueError("eef_jerk raw scores must be nonnegative or NaN")
    if not 0 <= quantile_low < quantile_high <= 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("eef_jerk normalization parameters are invalid")
    valid = np.isfinite(raw)
    scores = np.full(raw.shape, np.nan, dtype=np.float32)
    if valid.any():
        lower, upper = np.quantile(raw[valid], [quantile_low, quantile_high])
        scores[valid] = (
            1
            if upper - lower <= epsilon
            else 1 - np.clip((raw[valid] - lower) / (upper - lower), 0, 1)
        )
    return scores


def jerk_arrays(positions, timestamps, *, quantile_low, quantile_high, epsilon):
    results = [compute_eef_jerk(p, t) for p, t in zip(positions, timestamps, strict=True)]
    raw = np.asarray([result[0] for result in results], dtype=np.float64)
    return {
        "eef_jerk_positions": np.asarray(positions, dtype=np.float64),
        "eef_jerk_timestamps": np.asarray(timestamps, dtype=np.float64),
        "eef_jerk_raw": raw,
        "eef_jerk": normalize_eef_jerk(
            raw, quantile_low=quantile_low, quantile_high=quantile_high, epsilon=epsilon
        ),
        "eef_jerk_valid": np.isfinite(raw),
        "eef_jerk_reason": np.asarray([result[1] for result in results], dtype="U24"),
    }


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_jerk_cache(root, encoded):
    checksums = {}
    for field in JERK_CACHE_FIELDS:
        path = root / f"{field}.npy"
        np.save(path, getattr(encoded, field))
        checksums[field] = _sha256(path)
    return checksums


def validate_jerk_cache(root, clips, config):
    contract = jerk_contract(config)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("eef_jerk") != contract:
        raise ValueError("eef_jerk cache contract mismatch")
    if contract is None:
        if any((root / f"{field}.npy").exists() for field in JERK_CACHE_FIELDS):
            raise ValueError("unexpected eef_jerk cache without configuration")
        return {}
    checksums = manifest.get("eef_jerk_checksums")
    if not isinstance(checksums, dict):
        raise ValueError("eef_jerk cache checksums are missing")
    arrays = {}
    for field in JERK_CACHE_FIELDS:
        path = root / f"{field}.npy"
        try:
            if _sha256(path) != checksums.get(field):
                raise ValueError("checksum mismatch")
            arrays[field] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"eef_jerk cache {field} is missing or invalid") from error
    positions, times = arrays["eef_jerk_positions"], arrays["eef_jerk_timestamps"]
    if (
        positions.ndim != 3
        or positions.shape[0] != len(clips)
        or positions.shape[2] != 3
        or times.shape != positions.shape[:2]
        or any(arrays[f].shape != (len(clips),) for f in JERK_FIELDS)
        or positions.dtype != np.float64
        or times.dtype != np.float64
        or arrays["eef_jerk_raw"].dtype != np.float64
        or arrays["eef_jerk"].dtype != np.float32
        or arrays["eef_jerk_valid"].dtype != np.bool_
        or arrays["eef_jerk_reason"].dtype.kind != "U"
    ):
        raise ValueError("eef_jerk cache shape or dtype mismatch")
    overlap = {}
    for clip, p, t in zip(clips, positions, times, strict=True):
        if len(p) != clip.length:
            raise ValueError("eef_jerk cache clip length mismatch")
        for offset in range(len(p)):
            key = (clip.episode_id, clip.start_step + offset)
            current = (p[offset], t[offset])
            previous = overlap.get(key)
            if previous is not None and (
                not np.array_equal(previous[0], current[0]) or previous[1] != current[1]
            ):
                raise ValueError("eef_jerk cache has inconsistent overlap")
            overlap[key] = current
    expected = jerk_arrays(
        positions, times, **{k: contract[k] for k in ("quantile_low", "quantile_high", "epsilon")}
    )
    for field in JERK_FIELDS:
        a, b = arrays[field], expected[field]
        matches = (
            np.array_equal(a, b)
            if b.dtype.kind in "bU"
            else np.allclose(a, b, rtol=1e-7, atol=1e-8, equal_nan=True)
        )
        if not matches:
            raise ValueError(f"eef_jerk cache {field} does not match replay")
    return arrays


def jerk_summary(raw, reasons, selected_rows):
    values = np.asarray(raw)
    valid = np.isfinite(values)
    return {
        "unit": "m/s^3",
        "valid_clips": int(valid.sum()),
        "invalid_clips": int((~valid).sum()),
        "invalid_reasons": dict(sorted(Counter(str(r) for r in reasons if r).items())),
        "valid_pool_mean": float(values[valid].mean()) if valid.any() else None,
        "selected_mean": float(np.mean([row["eef_jerk_raw"] for row in selected_rows]))
        if selected_rows
        else None,
    }
