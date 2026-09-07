"""Clip-internal commanded-versus-observed translation error, in metres."""

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from numbers import Real

import numpy as np

from cocore.eef_jerk import normalize_eef_jerk


EXECUTION_METRIC = "low_action_execution_deviation"
EXECUTION_PREFIX = "action_execution_deviation"
EXECUTION_NUMERIC_FIELDS = ("action_execution_deviation_raw", EXECUTION_METRIC)
EXECUTION_FIELDS = (
    *EXECUTION_NUMERIC_FIELDS,
    "action_execution_deviation_valid",
    "action_execution_deviation_reason",
)
EXECUTION_CACHE_FIELDS = (
    "action_execution_deviation_positions",
    "action_execution_deviation_actions",
    "action_execution_deviation_timestamps",
    *EXECUTION_FIELDS,
)


def resolve_execution_config(value):
    if value is None:
        return None
    required = {"action_source", "action_semantics", "action_scale", "alignment_confirmed"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{EXECUTION_PREFIX} requires exactly {sorted(required)}")
    if (
        value["action_source"] != "original_command"
        or value["action_semantics"] != "delta_from_observed_position"
        or value["alignment_confirmed"] is not True
    ):
        raise ValueError(
            f"{EXECUTION_PREFIX} requires original_command, delta_from_observed_position "
            "and alignment_confirmed: true; state-difference labels are not commands"
        )
    scale = value["action_scale"]
    if (
        not isinstance(scale, (list, tuple))
        or len(scale) != 3
        or any(
            isinstance(number, bool)
            or not isinstance(number, Real)
            or not np.isfinite(number)
            or number <= 0
            for number in scale
        )
    ):
        raise ValueError(f"{EXECUTION_PREFIX} action_scale requires three finite positive numbers")
    return {**value, "action_scale": [float(number) for number in scale]}


def execution_contract(config):
    settings = resolve_execution_config(config.get(EXECUTION_PREFIX))
    if settings is None:
        return None
    return {
        "version": 1,
        "profile": config["prototypes"]["profile"],
        "state_key": "observation.state",
        "position_axes": [0, 1, 2],
        "action_axes": [0, 1, 2],
        "input": "raw_positions_and_commands_float64",
        "position_unit": "m",
        "raw_unit": "m",
        "time_unit": "s",
        "coordinate_frame": "same_fixed_frame",
        "alignment": "action_t_to_position_t_plus_one",
        "clipping": "already_applied_upstream",
        "timing_policy": "strictly_increasing_not_necessarily_uniform",
        "aggregation": "mean_l2_observed_delta_minus_scaled_command",
        "boundary": "clip_internal_length_minus_one_intervals",
        "normalization": "reverse_quantiles_of_valid_scanned_candidates",
        "constant_pool": "neutral_one_not_evidence_of_zero_error",
        "invalid_policy": "exclude_from_graph_when_selected_otherwise_diagnostic",
        **{key: config["encoding"][key] for key in ("quantile_low", "quantile_high", "epsilon")},
        **settings,
    }


def execution_inputs(positions, actions, timestamps):
    """Validate only the three translation command columns supplied by the caller."""
    try:
        p, a, t = (
            np.asarray(value, dtype=np.float64) for value in (positions, actions, timestamps)
        )
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(
            f"{EXECUTION_PREFIX} requires numeric positions/actions/timestamps"
        ) from error
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or a.shape != p.shape
        or t.shape != (len(p),)
        or not all(np.all(np.isfinite(value)) for value in (p, a, t))
    ):
        raise ValueError(
            f"{EXECUTION_PREFIX} requires finite [L, 3] positions/actions and [L] times"
        )
    return p, a, t


def compute_action_execution_deviation(positions, actions, timestamps, *, config):
    """Return (mean error in metres, reason); the last command has no observed endpoint."""
    settings = resolve_execution_config(config)
    if settings is None:
        raise ValueError(f"{EXECUTION_PREFIX} configuration is required")
    p, a, t = execution_inputs(positions, actions, timestamps)
    if len(p) < 2:
        return np.nan, "too_short"
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            if np.any(np.diff(t) <= 0):
                return np.nan, "non_increasing_time"
            error = np.diff(p, axis=0) - a[:-1] * settings["action_scale"]
            value = np.hypot.reduce(error, axis=1).mean()
            if not np.isfinite(value):
                return np.nan, "overflow"
        except FloatingPointError:
            return np.nan, "overflow"
    return float(value), ""


def normalize_execution_deviation(raw, *, quantile_low, quantile_high, epsilon):
    """Use the same reverse-quantile convention as translational Jerk."""
    return normalize_eef_jerk(
        raw, quantile_low=quantile_low, quantile_high=quantile_high, epsilon=epsilon
    )


def execution_arrays(
    positions, actions, timestamps, *, config, quantile_low, quantile_high, epsilon
):
    rows = [
        compute_action_execution_deviation(p, a, t, config=config)
        for p, a, t in zip(positions, actions, timestamps, strict=True)
    ]
    raw = np.asarray([row[0] for row in rows], dtype=np.float64)
    return {
        "action_execution_deviation_positions": np.asarray(positions, dtype=np.float64),
        "action_execution_deviation_actions": np.asarray(actions, dtype=np.float64),
        "action_execution_deviation_timestamps": np.asarray(timestamps, dtype=np.float64),
        "action_execution_deviation_raw": raw,
        EXECUTION_METRIC: normalize_execution_deviation(
            raw, quantile_low=quantile_low, quantile_high=quantile_high, epsilon=epsilon
        ),
        "action_execution_deviation_valid": np.isfinite(raw),
        "action_execution_deviation_reason": np.asarray([row[1] for row in rows], dtype="U24"),
    }


def _checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_execution_cache(root, encoded):
    checksums = {}
    for field in EXECUTION_CACHE_FIELDS:
        path = root / f"{field}.npy"
        np.save(path, getattr(encoded, field))
        checksums[field] = _checksum(path)
    return checksums


def validate_execution_cache(root, clips, config):
    contract = execution_contract(config)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get(EXECUTION_PREFIX) != contract:
        raise ValueError(f"{EXECUTION_PREFIX} cache contract mismatch")
    if contract is None:
        if manifest.get(f"{EXECUTION_PREFIX}_checksums") or any(
            (root / f"{field}.npy").exists() for field in EXECUTION_CACHE_FIELDS
        ):
            raise ValueError(f"unexpected {EXECUTION_PREFIX} cache")
        return {}
    checksums = manifest.get(f"{EXECUTION_PREFIX}_checksums")
    if not isinstance(checksums, dict) or set(checksums) != set(EXECUTION_CACHE_FIELDS):
        raise ValueError(f"{EXECUTION_PREFIX} checksums missing or invalid")
    arrays = {}
    for field in EXECUTION_CACHE_FIELDS:
        path = root / f"{field}.npy"
        try:
            if _checksum(path) != checksums[field]:
                raise ValueError("checksum mismatch")
            arrays[field] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"{EXECUTION_PREFIX} cache {field} missing or invalid") from error
    p, a, t = (
        arrays[f"{EXECUTION_PREFIX}_{name}"] for name in ("positions", "actions", "timestamps")
    )
    if (
        p.ndim != 3
        or p.shape[0] != len(clips)
        or p.shape[2] != 3
        or a.shape != p.shape
        or t.shape != p.shape[:2]
        or any(value.dtype != np.float64 for value in (p, a, t))
        or any(arrays[field].shape != (len(clips),) for field in EXECUTION_FIELDS)
        or arrays[f"{EXECUTION_PREFIX}_raw"].dtype != np.float64
        or arrays[EXECUTION_METRIC].dtype != np.float32
        or arrays[f"{EXECUTION_PREFIX}_valid"].dtype != np.bool_
        or arrays[f"{EXECUTION_PREFIX}_reason"].dtype != np.dtype("U24")
    ):
        raise ValueError(f"{EXECUTION_PREFIX} cache shape or dtype mismatch")
    overlap = {}
    for clip, positions, actions, times in zip(clips, p, a, t, strict=True):
        if len(positions) != clip.length:
            raise ValueError(f"{EXECUTION_PREFIX} clip length mismatch")
        for offset in range(clip.length):
            key = (clip.episode_id, clip.start_step + offset)
            current = (positions[offset], actions[offset], times[offset])
            previous = overlap.get(key)
            if previous is not None and any(
                not np.array_equal(before, after)
                for before, after in zip(previous, current, strict=True)
            ):
                raise ValueError(f"{EXECUTION_PREFIX} inconsistent overlapping inputs")
            overlap[key] = current
    expected = execution_arrays(
        p,
        a,
        t,
        config=config[EXECUTION_PREFIX],
        **{key: contract[key] for key in ("quantile_low", "quantile_high", "epsilon")},
    )
    validate_execution_fields(arrays, expected)
    return arrays


def validate_execution_fields(actual, expected):
    """Compare node arrays or JSON/Parquet rows, preserving null and boolean types."""
    for field in EXECUTION_FIELDS:
        if field not in actual:
            raise ValueError(f"{EXECUTION_PREFIX} missing {field}")
        a, b = np.asarray(actual[field]), np.asarray(expected[field])
        if a.shape != b.shape:
            raise ValueError(f"{EXECUTION_PREFIX} {field} shape mismatch")
        if field in EXECUTION_NUMERIC_FIELDS:
            if expected[field] is None:
                matches = actual[field] is None
            else:
                if a.dtype.kind not in "fiu":
                    raise ValueError(f"{EXECUTION_PREFIX} {field} must be numeric")
                matches = np.allclose(a, b, rtol=1e-7, atol=1e-12, equal_nan=True)
        else:
            if field.endswith("_valid") and a.dtype != np.bool_:
                raise ValueError(f"{EXECUTION_PREFIX} validity must contain booleans")
            matches = np.array_equal(a, b)
        if not matches:
            raise ValueError(f"{EXECUTION_PREFIX} {field} does not match replay")


def execution_row(arrays, index):
    return {
        **{
            field: float(arrays[field][index]) if np.isfinite(arrays[field][index]) else None
            for field in EXECUTION_NUMERIC_FIELDS
        },
        f"{EXECUTION_PREFIX}_valid": bool(arrays[f"{EXECUTION_PREFIX}_valid"][index]),
        f"{EXECUTION_PREFIX}_reason": str(arrays[f"{EXECUTION_PREFIX}_reason"][index]),
    }


def execution_summary(scanned, graph_rows, selected_rows):
    groups = {
        "scanned": [
            execution_row(scanned, i) for i in range(len(scanned[f"{EXECUTION_PREFIX}_valid"]))
        ],
        "graph": graph_rows,
        "selected": selected_rows,
    }
    result = {"unit": "m"}
    for name, rows in groups.items():
        raw = [row[f"{EXECUTION_PREFIX}_raw"] for row in rows if row[f"{EXECUTION_PREFIX}_valid"]]
        result[name] = {
            "valid_count": len(raw),
            "invalid_count": len(rows) - len(raw),
            "invalid_reasons": dict(
                sorted(
                    Counter(
                        row[f"{EXECUTION_PREFIX}_reason"]
                        for row in rows
                        if not row[f"{EXECUTION_PREFIX}_valid"]
                    ).items()
                )
            ),
            "raw_mean": float(np.mean(raw)) if raw else None,
        }
    return result
