"""Adjacent clip-internal translational reversals and optional reliability score."""

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

BACKTRACKING_NUMERIC_FIELDS = ("local_backtracking_rate", "low_local_backtracking")
_COUNT_FIELDS = ("local_backtracking_valid_count", "local_backtracking_count")
BACKTRACKING_FIELDS = (
    *BACKTRACKING_NUMERIC_FIELDS,
    *_COUNT_FIELDS,
    "local_backtracking_valid",
    "local_backtracking_reason",
)
BACKTRACKING_CACHE_FIELDS = (
    "local_backtracking_positions",
    "local_backtracking_timestamps",
    *BACKTRACKING_FIELDS,
)


def resolve_backtracking_config(value):
    if value is None:
        return None
    if (
        not isinstance(value, Mapping)
        or "epsilon_p" not in value
        or value.keys() - {"epsilon_p", "eta"}
    ):
        raise ValueError("local_backtracking requires epsilon_p and optional eta")
    settings = {"eta": 0.5, **value}
    for key, number in settings.items():
        if (
            isinstance(number, bool)
            or not isinstance(number, Real)
            or not np.isfinite(number)
            or (number <= 0 if key == "epsilon_p" else not 0 <= number < 1)
        ):
            raise ValueError(f"local_backtracking {key} must be finite and in range")
        settings[key] = float(number)
    return settings


def backtracking_contract(full_config):
    settings = resolve_backtracking_config(full_config.get("local_backtracking"))
    if settings is None:
        return None
    return {
        "version": 1,
        "profile": full_config["prototypes"]["profile"],
        "state_key": "observation.state",
        "position_axes": [0, 1, 2],
        "coordinate_frame": "fixed",
        "input": "raw_clip_internal_positions_float64",
        "displacement": "p[t+1]-p[t]",
        "position_unit": "original_position_unit",
        "time_unit": "s",
        "dt": "median",
        "rtol": 1e-4,
        "atol": 1e-8,
        "timing_policy": "strictly_increasing_uniform",
        "valid_comparison": "both_adjacent_displacement_norms_strictly_greater_than_epsilon_p",
        "cosine": "clip(dot(d[t],d[t+1])/(norm(d[t])*norm(d[t+1])),-1,1)",
        "numerical_method": "hypot_norm_direct_dot_with_unit_direction_fallback_for_unsafe_products",
        "backtracking": "cosine_strictly_less_than_negative_eta",
        "rate": "backtracking_count/valid_comparison_count",
        "score": "one_minus_rate",
        "stationary_policy": "exclude_adjacent_pairs_without_bridging",
        "minimum_position_samples": 3,
        "invalid_policy": "nan_excluded_from_per_clip_geometric_mean",
        "invalid_counts": 0,
        "no_valid_metrics": "neutral_one",
        "malformed_input_policy": "error_on_shape_or_nonfinite",
        "invalid_reason_order": [
            "too_short",
            "non_increasing_time",
            "non_uniform_time",
            "overflow",
            "no_valid_comparisons",
        ],
        **settings,
    }


def compute_local_backtracking(positions, timestamps, *, config):
    """Use stable norms, preserving direct dots unless their products are unsafe."""
    settings = resolve_backtracking_config(config)
    if settings is None:
        raise ValueError("local_backtracking configuration is required")
    try:
        p = np.asarray(positions, dtype=np.float64)
        t = np.asarray(timestamps, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(
            "local_backtracking requires finite [L, 3] positions and timestamps"
        ) from error
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or t.shape != (len(p),)
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(t))
    ):
        raise ValueError("local_backtracking requires finite [L, 3] positions and timestamps")
    result = dict.fromkeys(BACKTRACKING_NUMERIC_FIELDS, float("nan"))
    result.update(
        local_backtracking_valid_count=0,
        local_backtracking_count=0,
        local_backtracking_valid=False,
        local_backtracking_reason="too_short",
    )
    if len(p) < 3:
        return result
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            intervals = np.diff(t)
            if np.any(intervals <= 0):
                result["local_backtracking_reason"] = "non_increasing_time"
                return result
            dt = float(np.median(intervals))
            if not np.allclose(intervals, dt, rtol=1e-4, atol=1e-8):
                result["local_backtracking_reason"] = "non_uniform_time"
                return result
            displacement = np.diff(p, axis=0)
            norms = np.hypot.reduce(displacement, axis=1)
            moving = norms > settings["epsilon_p"]
            pairs = moving[:-1] & moving[1:]
            valid_count = int(np.count_nonzero(pairs))
            if not valid_count:
                result["local_backtracking_reason"] = "no_valid_comparisons"
                return result
            before = displacement[:-1][pairs]
            after = displacement[1:][pairs]
            # Preserve exact cancellations in ordinary dot products. Unit-vector
            # rounding can otherwise turn an orthogonal pair into a reversal.
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                products = before * after
                numerator = np.sum(products, axis=1)
                denominator = norms[:-1][pairs] * norms[1:][pairs]
                cosine = numerator / denominator
            tiny = np.finfo(np.float64).tiny
            unsafe = (
                ~np.isfinite(numerator)
                | ~np.isfinite(denominator)
                | ~np.isfinite(cosine)
                | (denominator < tiny)
                | np.any((before != 0) & (after != 0) & (np.abs(products) < tiny), axis=1)
                | ((numerator != 0) & (np.abs(cosine) < tiny))
            )
            if np.any(unsafe):
                unit_before = before[unsafe] / norms[:-1][pairs][unsafe, None]
                unit_after = after[unsafe] / norms[1:][pairs][unsafe, None]
                cosine[unsafe] = np.sum(unit_before * unit_after, axis=1)
            cosine = np.clip(cosine, -1.0, 1.0)
            if not np.all(np.isfinite(cosine)):
                raise FloatingPointError("nonfinite cosine")
            count = int(np.count_nonzero(cosine < -settings["eta"]))
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            result["local_backtracking_reason"] = "overflow"
            return result
    rate = count / valid_count
    result.update(
        local_backtracking_rate=rate,
        low_local_backtracking=1.0 - rate,
        local_backtracking_valid_count=valid_count,
        local_backtracking_count=count,
        local_backtracking_valid=True,
        local_backtracking_reason="",
    )
    return result


def backtracking_arrays(positions, timestamps, *, config, clips):
    rows = []
    for p, t, clip in zip(positions, timestamps, clips, strict=True):
        try:
            rows.append(compute_local_backtracking(p, t, config=config))
        except ValueError as error:
            raise ValueError(f"local_backtracking clip {clip.sample_id}: {error}") from error
    arrays = {
        "local_backtracking_positions": np.asarray(positions, dtype=np.float64),
        "local_backtracking_timestamps": np.asarray(timestamps, dtype=np.float64),
    }
    for field in BACKTRACKING_FIELDS:
        dtype = (
            np.int64
            if field in _COUNT_FIELDS
            else np.bool_
            if field == "local_backtracking_valid"
            else "U32"
            if field == "local_backtracking_reason"
            else np.float64
        )
        arrays[field] = np.asarray([row[field] for row in rows], dtype=dtype)
    return arrays


def _checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_backtracking_cache(root, encoded):
    root = Path(root)
    checksums = {}
    for field in BACKTRACKING_CACHE_FIELDS:
        path = root / f"{field}.npy"
        np.save(path, getattr(encoded, field))
        checksums[field] = _checksum(path)
    return checksums


def validate_backtracking_cache(root, clips, full_config):
    root = Path(root)
    contract = backtracking_contract(full_config)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("local_backtracking") != contract:
        raise ValueError("local_backtracking cache contract mismatch")
    if contract is None:
        if any((root / f"{field}.npy").exists() for field in BACKTRACKING_CACHE_FIELDS):
            raise ValueError("unexpected local_backtracking cache")
        return {}
    checksums = manifest.get("local_backtracking_checksums")
    if not isinstance(checksums, dict):
        raise ValueError("local_backtracking checksums missing")
    arrays = {}
    for field in BACKTRACKING_CACHE_FIELDS:
        path = root / f"{field}.npy"
        try:
            if _checksum(path) != checksums.get(field):
                raise ValueError("checksum mismatch")
            arrays[field] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"local_backtracking cache {field} missing or invalid") from error
    p, t = arrays["local_backtracking_positions"], arrays["local_backtracking_timestamps"]
    if (
        p.ndim != 3
        or p.shape[0] != len(clips)
        or p.shape[2] != 3
        or t.shape != p.shape[:2]
        or p.dtype != np.float64
        or t.dtype != np.float64
        or any(arrays[f].shape != (len(clips),) for f in BACKTRACKING_FIELDS)
        or any(arrays[f].dtype != np.float64 for f in BACKTRACKING_NUMERIC_FIELDS)
        or any(arrays[f].dtype != np.int64 for f in _COUNT_FIELDS)
        or arrays["local_backtracking_valid"].dtype != np.bool_
        or arrays["local_backtracking_reason"].dtype != np.dtype("U32")
    ):
        raise ValueError("local_backtracking cache shape or dtype mismatch")
    overlap = {}
    for clip, positions, timestamps in zip(clips, p, t, strict=True):
        if len(positions) != clip.length:
            raise ValueError("local_backtracking clip length mismatch")
        for offset, (position, timestamp) in enumerate(zip(positions, timestamps, strict=True)):
            key = (clip.episode_id, clip.start_step + offset)
            previous = overlap.get(key)
            if previous is not None and (
                not np.array_equal(position, previous[0]) or timestamp != previous[1]
            ):
                raise ValueError("local_backtracking inconsistent overlapping inputs")
            overlap[key] = (position, timestamp)
    expected = backtracking_arrays(p, t, config=full_config["local_backtracking"], clips=clips)
    validate_backtracking_fields(arrays, expected)
    return arrays


def validate_backtracking_fields(actual, expected):
    for field in BACKTRACKING_FIELDS:
        if (
            field not in actual
            or np.asarray(actual[field]).shape != np.asarray(expected[field]).shape
        ):
            raise ValueError(f"local_backtracking {field} shape mismatch")
        a, b = np.asarray(actual[field]), np.asarray(expected[field])
        if field in _COUNT_FIELDS and (
            a.dtype.kind not in "iu"
            or any(
                isinstance(value, (bool, np.bool_))
                for value in np.asarray(actual[field], dtype=object).flat
            )
        ):
            raise ValueError(f"local_backtracking {field} must contain integer counts")
        if field == "local_backtracking_valid" and a.dtype != np.bool_:
            raise ValueError(f"local_backtracking {field} must contain booleans")
        if field in BACKTRACKING_NUMERIC_FIELDS:
            if not isinstance(expected[field], np.ndarray) and b.ndim == 0:
                value = actual[field]
                if expected[field] is None:
                    matches = value is None
                elif (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                ):
                    raise ValueError(f"local_backtracking {field} must be a finite real number")
                else:
                    matches = np.isclose(value, expected[field], rtol=1e-7, atol=1e-8)
            else:
                if a.dtype.kind not in "fiu":
                    raise ValueError(f"local_backtracking {field} must contain numeric values")
                matches = np.allclose(
                    a.astype(np.float64),
                    b.astype(np.float64),
                    rtol=1e-7,
                    atol=1e-8,
                    equal_nan=True,
                )
        else:
            matches = np.array_equal(a, b)
        if not matches:
            raise ValueError(f"local_backtracking {field} does not match replay")


def backtracking_row(arrays, index):
    return {
        **{
            field: float(arrays[field][index]) if np.isfinite(arrays[field][index]) else None
            for field in BACKTRACKING_NUMERIC_FIELDS
        },
        **{field: int(arrays[field][index]) for field in _COUNT_FIELDS},
        "local_backtracking_valid": bool(arrays["local_backtracking_valid"][index]),
        "local_backtracking_reason": str(arrays["local_backtracking_reason"][index]),
    }


def backtracking_summary(scanned, graph_rows, selected_rows):
    groups = {
        "scanned": [
            backtracking_row(scanned, i) for i in range(len(scanned["local_backtracking_valid"]))
        ],
        "graph": graph_rows,
        "selected": selected_rows,
    }
    result = {}
    for name, rows in groups.items():
        valid = [row for row in rows if row["local_backtracking_valid"]]
        result[name] = {
            "valid_count": len(valid),
            "invalid_count": len(rows) - len(valid),
            "invalid_reasons": dict(
                sorted(
                    Counter(
                        row["local_backtracking_reason"]
                        for row in rows
                        if not row["local_backtracking_valid"]
                    ).items()
                )
            ),
            "rate_mean": float(np.mean([row["local_backtracking_rate"] for row in valid]))
            if valid
            else None,
            "valid_comparison_count": sum(row["local_backtracking_valid_count"] for row in rows),
            "backtracking_count": sum(row["local_backtracking_count"] for row in rows),
        }
    return result
