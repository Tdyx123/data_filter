"""Reference-scaled, clip-internal abnormal continuous action jump rate."""

from collections.abc import Mapping
import hashlib
import json
from numbers import Integral, Real

import numpy as np
import pyarrow.parquet as pq

JUMP_FIELDS = ("action_jump_rate", "action_jump")
JUMP_CACHE_FIELDS = (
    "action_jump_actions",
    "action_jump_episode_ids",
    "action_jump_offsets",
    "action_jump_dimensions",
    "action_jump_scale",
    "action_jump_threshold",
    "action_jump_pair_count",
    *JUMP_FIELDS,
)


def resolve_jump_config(value):
    if value is None:
        return None
    defaults = {"threshold_quantile": 0.99, "threshold": None, "epsilon": 1e-8}
    if not isinstance(value, Mapping) or value.keys() - defaults.keys():
        raise ValueError("action_jump configuration contains unsupported fields")
    result = {**defaults, **value}
    for name, number in result.items():
        if name == "threshold" and number is None:
            continue
        if isinstance(number, bool) or not isinstance(number, Real) or not np.isfinite(number):
            raise ValueError(f"action_jump {name} must be finite and numeric")
        result[name] = float(number)
    if not 0 < result["threshold_quantile"] < 1:
        raise ValueError("action_jump threshold_quantile must satisfy 0 < q < 1")
    if result["epsilon"] <= 0 or (result["threshold"] is not None and result["threshold"] < 0):
        raise ValueError("action_jump requires positive epsilon and nonnegative threshold")
    return result


def jump_contract(config):
    settings = resolve_jump_config(config.get("action_jump"))
    if settings is None:
        return None
    return {
        "version": 1,
        "profile": config["prototypes"]["profile"],
        "gripper_action_index": config["quality"]["gripper_action_index"],
        "input": "raw_continuous_actions_without_clipping",
        "reference": "all_loaded_episode_frames",
        "scale": "population_std_ddof_zero",
        "jump": "rms_standardized_adjacent_difference",
        "threshold_method": "explicit_or_reference_quantile_linear",
        "comparison": "strict_greater_than",
        "aggregation": "clip_internal_pairs_over_length_minus_one",
        "episode_boundary": "never_compare",
        "score": "one_minus_rate",
        **settings,
    }


def _actions(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("action_jump requires finite nonempty [time, dimension] actions")
    return values


def _jumps(values, scale, epsilon):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            result = np.sqrt(
                np.mean((np.diff(values, axis=0) / np.maximum(scale, epsilon)) ** 2, axis=1)
            )
        except FloatingPointError as error:
            raise ValueError("action_jump numerical overflow") from error
    if not np.all(np.isfinite(result)):
        raise ValueError("action_jump numerical overflow")
    return result


def compute_action_jump_rate(actions, scale, *, threshold, epsilon):
    values = _actions(actions)
    settings = resolve_jump_config({"threshold": threshold, "epsilon": epsilon})
    scale = np.asarray(scale, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("action_jump requires at least two frames")
    if scale.shape != (values.shape[1],) or not np.all(np.isfinite(scale)) or np.any(scale < 0):
        raise ValueError("action_jump scale must be a finite nonnegative dimension vector")
    if settings["threshold"] is None:
        raise ValueError("action_jump rate requires a calibrated threshold")
    return float(np.mean(_jumps(values, scale, settings["epsilon"]) > settings["threshold"]))


def _calibrate_and_score(episodes, episode_ids, clips, settings):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            scale = np.std(np.concatenate(episodes), axis=0, ddof=0)
        except FloatingPointError as error:
            raise ValueError("action_jump scale overflow") from error
    if not np.all(np.isfinite(scale)):
        raise ValueError("action_jump scale overflow")
    jumps = [_jumps(values, scale, settings["epsilon"]) for values in episodes]
    pooled = np.concatenate(jumps)
    if not len(pooled):
        raise ValueError("action_jump reference contains no adjacent comparisons")
    threshold = settings["threshold"]
    if threshold is None:
        threshold = float(np.quantile(pooled, settings["threshold_quantile"], method="linear"))
    by_id = dict(zip(episode_ids, jumps, strict=True))
    rates = []
    for clip in clips:
        if (
            clip.episode_id not in by_id
            or clip.start_step < 0
            or clip.length < 2
            or clip.end_step - clip.start_step + 1 != clip.length
            or clip.end_step > len(by_id[clip.episode_id])
        ):
            raise ValueError("action_jump clip bounds are invalid")
        rates.append(np.mean(by_id[clip.episode_id][clip.start_step : clip.end_step] > threshold))
    rates = np.asarray(rates, dtype=np.float64)
    return {
        "action_jump_scale": scale,
        "action_jump_threshold": np.asarray(threshold, dtype=np.float64),
        "action_jump_pair_count": np.asarray(len(pooled), dtype=np.int64),
        "action_jump_rate": rates,
        "action_jump": 1.0 - rates,
    }


def build_jump_arrays(episodes, episode_ids, clips, *, config, gripper_action_index):
    settings = resolve_jump_config(config)
    if settings is None:
        raise ValueError("action_jump configuration is required")
    values = [_actions(episode) for episode in episodes]
    if not values or len(values) != len(episode_ids) or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("action_jump reference episodes are missing or duplicated")
    width = values[0].shape[1]
    if any(value.shape[1] != width for value in values):
        raise ValueError("action_jump action dimensions differ between episodes")
    if (
        isinstance(gripper_action_index, bool)
        or not isinstance(gripper_action_index, Integral)
        or not -width <= gripper_action_index < width
        or width < 2
    ):
        raise ValueError("action_jump requires a valid gripper index and continuous dimensions")
    dimensions = np.delete(np.arange(width, dtype=np.int64), gripper_action_index % width)
    continuous = [value[:, dimensions] for value in values]
    return {
        "action_jump_actions": np.concatenate(continuous),
        "action_jump_episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "action_jump_offsets": np.r_[0, np.cumsum([len(value) for value in values])].astype(
            np.int64
        ),
        "action_jump_dimensions": dimensions,
        **_calibrate_and_score(continuous, episode_ids, clips, settings),
    }


def _checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_jump_cache(root, encoded):
    checksums = {}
    for field in JUMP_CACHE_FIELDS:
        path = root / f"{field}.npy"
        np.save(path, getattr(encoded, field))
        checksums[field] = _checksum(path)
    return checksums


def validate_jump_cache(root, clips, config):
    contract = jump_contract(config)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("action_jump") != contract:
        raise ValueError("action_jump cache contract mismatch")
    if contract is None:
        if any((root / f"{field}.npy").exists() for field in JUMP_CACHE_FIELDS):
            raise ValueError("unexpected action_jump cache")
        return {}
    checksums = manifest.get("action_jump_checksums")
    if not isinstance(checksums, dict):
        raise ValueError("action_jump checksums missing")
    arrays = {}
    for field in JUMP_CACHE_FIELDS:
        path = root / f"{field}.npy"
        try:
            if _checksum(path) != checksums.get(field):
                raise ValueError("checksum mismatch")
            arrays[field] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"action_jump cache {field} missing or invalid") from error
    integer_fields = {
        "action_jump_episode_ids",
        "action_jump_offsets",
        "action_jump_dimensions",
        "action_jump_pair_count",
    }
    for field, array in arrays.items():
        if array.dtype != (np.int64 if field in integer_fields else np.float64) or not np.all(
            np.isfinite(array)
        ):
            raise ValueError(f"action_jump cache {field} invalid dtype or values")
    actions = _actions(arrays["action_jump_actions"])
    ids, offsets, dims = (
        arrays[f"action_jump_{name}"] for name in ("episode_ids", "offsets", "dimensions")
    )
    records = pq.read_table(root.parent / "scan" / "episodes.parquet").to_pylist()
    if (
        ids.ndim != 1
        or not np.array_equal(ids, [row["episode_id"] for row in records])
        or len(set(ids)) != len(ids)
        or not np.array_equal(offsets, np.r_[0, np.cumsum([row["length"] for row in records])])
        or offsets[-1] != len(actions)
    ):
        raise ValueError("action_jump reference episode boundaries mismatch")
    width = actions.shape[1] + 1
    gripper = contract["gripper_action_index"]
    if not -width <= gripper < width or not np.array_equal(
        dims, np.delete(np.arange(width), gripper % width)
    ):
        raise ValueError("action_jump continuous dimensions mismatch")
    episodes = [actions[start:end] for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
    expected = _calibrate_and_score(
        episodes, ids, clips, resolve_jump_config(config["action_jump"])
    )
    for field, values in expected.items():
        if arrays[field].shape != values.shape or not np.array_equal(arrays[field], values):
            raise ValueError(f"action_jump cache {field} does not match replay")
    return arrays


def validate_jump_fields(actual, expected):
    for field in JUMP_FIELDS:
        if field not in actual or not np.array_equal(actual[field], expected[field]):
            raise ValueError(f"action_jump {field} does not match expected values")
