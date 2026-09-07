"""Clip-internal translational velocity spectrum and optional reliability score."""

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.signal import periodogram

HF_NUMERIC_FIELDS = (
    "high_frequency_ratio",
    "high_frequency_rms",
    "total_fluctuation_rms",
    "low_high_frequency_jitter",
    "high_frequency_resolution_hz",
)
HF_FIELDS = (*HF_NUMERIC_FIELDS, "high_frequency_valid", "high_frequency_reason")
HF_CACHE_FIELDS = ("high_frequency_positions", "high_frequency_timestamps", *HF_FIELDS)


def resolve_hf_config(value):
    if value is None:
        return None
    required = {"cutoff_hz", "noise_floor_rms", "max_frequency_resolution_hz"}
    if (
        not isinstance(value, Mapping)
        or not required <= value.keys()
        or value.keys() - (required | {"epsilon"})
    ):
        raise ValueError(
            "high_frequency_jitter requires cutoff_hz, noise_floor_rms and max_frequency_resolution_hz"
        )
    result = {"epsilon": 1e-12, **value}
    for key, number in result.items():
        if (
            isinstance(number, bool)
            or not isinstance(number, Real)
            or not np.isfinite(number)
            or number <= 0
        ):
            raise ValueError(f"high_frequency_jitter {key} must be finite and positive")
        result[key] = float(number)
    return result


def hf_contract(config):
    settings = resolve_hf_config(config.get("high_frequency_jitter"))
    if settings is None:
        return None
    return {
        "version": 1,
        "profile": config["prototypes"]["profile"],
        "state_key": "observation.state",
        "position_axes": [0, 1, 2],
        "coordinate_frame": "fixed",
        "input": "clip_internal_position_difference_over_median_dt",
        "strength_unit": "original_position_unit/s",
        "time_unit": "s",
        "dt": "median",
        "rtol": 1e-3,
        "atol": 1e-8,
        "window": "periodic_hann",
        "detrend": "constant",
        "scaling": "density",
        "spectrum": "one_sided_sum_of_axis_psds",
        "zero_padding": False,
        "integration": "sum_bins_times_fs_over_N",
        "exclude_dc": True,
        "high_band": "f_strictly_greater_than_cutoff",
        "score": "one_minus_ratio",
        "invalid_policy": "nan_excluded_from_per_clip_geometric_mean",
        "no_valid_metrics": "neutral_one",
        "minimum_velocity_samples": 3,
        "invalid_reason_order": [
            "too_short",
            "coarse_resolution",
            "missing_band",
            "low_fluctuation",
        ],
        **settings,
    }


def compute_high_frequency_jitter(positions, timestamps, *, config):
    settings = resolve_hf_config(config)
    if settings is None:
        raise ValueError("high_frequency_jitter configuration is required")
    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or t.shape != (len(p),)
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(t))
    ):
        raise ValueError("high_frequency_jitter requires finite [L, 3] positions and timestamps")
    result = dict.fromkeys(HF_NUMERIC_FIELDS, float("nan"))
    result.update(high_frequency_valid=False, high_frequency_reason="too_short")
    if len(p) < 2:
        return result
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            intervals = np.diff(t)
            if np.any(intervals <= 0):
                raise ValueError("high_frequency_jitter timestamps must be strictly increasing")
            dt = float(np.median(intervals))
            if not np.allclose(intervals, dt, rtol=1e-3, atol=1e-8):
                raise ValueError("high_frequency_jitter timestamps must be uniform")
            fs = 1.0 / dt
            if not np.isfinite(fs) or not 0 < settings["cutoff_hz"] < fs / 2:
                raise ValueError("high_frequency_jitter cutoff_hz must be below Nyquist")
            n = len(p) - 1
            resolution = fs / n
            result["high_frequency_resolution_hz"] = resolution
            velocity = np.diff(p, axis=0) / dt
            frequencies, psd = periodogram(
                velocity,
                fs=fs,
                window="hann",
                detrend="constant",
                return_onesided=True,
                scaling="density",
                axis=0,
                nfft=n,
            )
            power = psd.sum(axis=1)
            high = frequencies > settings["cutoff_hz"]
            low = (frequencies > 0) & ~high
            total = float(power[frequencies > 0].sum() * resolution)
            high_energy = float(power[high].sum() * resolution)
            denominator = total + settings["epsilon"]
            if not np.all(np.isfinite([total, high_energy, denominator])):
                raise ValueError("high_frequency_jitter spectrum overflow")
            ratio = float(np.clip(high_energy / denominator, 0, 1))
            rms = float(np.sqrt(total))
            result.update(
                high_frequency_ratio=ratio,
                high_frequency_rms=float(np.sqrt(high_energy)),
                total_fluctuation_rms=rms,
            )
        except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
            raise ValueError("high_frequency_jitter numerical overflow") from error
    reason = (
        "too_short"
        if n < 3
        else "coarse_resolution"
        if resolution > settings["max_frequency_resolution_hz"]
        else "missing_band"
        if not high.any() or not low.any()
        else "low_fluctuation"
        if rms <= settings["noise_floor_rms"]
        else ""
    )
    result.update(high_frequency_valid=not reason, high_frequency_reason=reason)
    if not reason:
        result["low_high_frequency_jitter"] = 1.0 - ratio
    return result


def hf_arrays(positions, timestamps, *, config, clips):
    results = []
    for p, t, clip in zip(positions, timestamps, clips, strict=True):
        try:
            results.append(compute_high_frequency_jitter(p, t, config=config))
        except ValueError as error:
            raise ValueError(f"high_frequency_jitter clip {clip.sample_id}: {error}") from error
    arrays = {
        "high_frequency_positions": np.asarray(positions, dtype=np.float64),
        "high_frequency_timestamps": np.asarray(timestamps, dtype=np.float64),
    }
    for field in HF_FIELDS:
        dtype = (
            np.bool_
            if field == "high_frequency_valid"
            else "U32"
            if field == "high_frequency_reason"
            else np.float64
        )
        arrays[field] = np.asarray([row[field] for row in results], dtype=dtype)
    return arrays


def _checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_hf_cache(root, encoded):
    checksums = {}
    for field in HF_CACHE_FIELDS:
        path = root / f"{field}.npy"
        np.save(path, getattr(encoded, field))
        checksums[field] = _checksum(path)
    return checksums


def validate_hf_cache(root, clips, config):
    contract = hf_contract(config)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("high_frequency_jitter") != contract:
        raise ValueError("high_frequency_jitter cache contract mismatch")
    if contract is None:
        if any((root / f"{field}.npy").exists() for field in HF_CACHE_FIELDS):
            raise ValueError("unexpected high_frequency_jitter cache")
        return {}
    checksums = manifest.get("high_frequency_checksums")
    if not isinstance(checksums, dict):
        raise ValueError("high_frequency_jitter checksums missing")
    arrays = {}
    for field in HF_CACHE_FIELDS:
        path = root / f"{field}.npy"
        try:
            if _checksum(path) != checksums.get(field):
                raise ValueError("checksum mismatch")
            arrays[field] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"high_frequency_jitter cache {field} missing or invalid") from error
    p, t = arrays["high_frequency_positions"], arrays["high_frequency_timestamps"]
    if (
        p.ndim != 3
        or p.shape[0] != len(clips)
        or p.shape[2] != 3
        or t.shape != p.shape[:2]
        or p.dtype != np.float64
        or t.dtype != np.float64
        or any(arrays[f].shape != (len(clips),) for f in HF_FIELDS)
        or any(arrays[f].dtype != np.float64 for f in HF_NUMERIC_FIELDS)
        or arrays["high_frequency_valid"].dtype != np.bool_
        or arrays["high_frequency_reason"].dtype.kind != "U"
    ):
        raise ValueError("high_frequency_jitter cache shape or dtype mismatch")
    overlap = {}
    for clip, positions, timestamps in zip(clips, p, t, strict=True):
        if len(positions) != clip.length:
            raise ValueError("high_frequency_jitter clip length mismatch")
        for offset, (position, timestamp) in enumerate(zip(positions, timestamps, strict=True)):
            key = (clip.episode_id, clip.start_step + offset)
            previous = overlap.get(key)
            if previous is not None and (
                not np.array_equal(position, previous[0]) or timestamp != previous[1]
            ):
                raise ValueError("high_frequency_jitter inconsistent overlapping inputs")
            overlap[key] = (position, timestamp)
    expected = hf_arrays(p, t, config=config["high_frequency_jitter"], clips=clips)
    validate_hf_fields(arrays, expected)
    return arrays


def validate_hf_fields(actual, expected):
    for field in HF_FIELDS:
        if (
            field not in actual
            or np.asarray(actual[field]).shape != np.asarray(expected[field]).shape
        ):
            raise ValueError(f"high_frequency_jitter {field} shape mismatch")
        a, b = np.asarray(actual[field]), np.asarray(expected[field])
        matches = (
            np.array_equal(a, b)
            if field not in HF_NUMERIC_FIELDS
            else np.allclose(a, b, rtol=1e-7, atol=1e-8, equal_nan=True)
        )
        if not matches:
            raise ValueError(f"high_frequency_jitter {field} does not match replay")


def hf_row(arrays, index):
    return {
        **{
            f: float(arrays[f][index]) if np.isfinite(arrays[f][index]) else None
            for f in HF_NUMERIC_FIELDS
        },
        "high_frequency_valid": bool(arrays["high_frequency_valid"][index]),
        "high_frequency_reason": str(arrays["high_frequency_reason"][index]),
    }


def hf_summary(scanned, graph_rows, selected_rows):
    groups = {
        "scanned": [hf_row(scanned, i) for i in range(len(scanned["high_frequency_valid"]))],
        "graph": graph_rows,
        "selected": selected_rows,
    }
    result = {"strength_unit": "original_position_unit/s"}
    for name, rows in groups.items():
        valid = [r for r in rows if r["high_frequency_valid"]]
        result[name] = {
            "valid_count": len(valid),
            "invalid_count": len(rows) - len(valid),
            "invalid_reasons": dict(
                sorted(
                    Counter(
                        r["high_frequency_reason"] for r in rows if not r["high_frequency_valid"]
                    ).items()
                )
            ),
            **{
                f"{f}_mean": float(np.mean([r[f] for r in valid])) if valid else None
                for f in HF_NUMERIC_FIELDS
            },
        }
    return result
