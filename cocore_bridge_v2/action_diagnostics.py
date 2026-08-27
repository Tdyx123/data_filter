"""Read-only BridgeData V2 action-classification diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from cocore.prototypes import (
    STATE_KEY,
    TRAJECTORY_WINDOW_LENGTH,
    atomic_action_count,
    cluster_count_for_training_count,
    create_action_catalog,
    maximum_retained_parents,
    motion_primitive_contract,
    resolve_motion_primitive_profile,
    trajectory_window_starts,
)
from libero_motion_primitives import classify_motion_primitive
from trajectory_data import DatasetAdapter


_PROFILE = "bridge_v2"
_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
_AXES = (
    ("x", 0, "translation"),
    ("y", 1, "translation"),
    ("z", 2, "translation"),
    ("roll", 3, "roll"),
    ("pitch", 4, "tilt"),
    ("yaw", 5, "rotation"),
    ("gripper", 7, "gripper"),
)


def summarize_bridge_action_counts(counts: Mapping[str, int]) -> dict[str, int | float | str]:
    """Summarize Bridge retention, fallback, and estimated leaf counts."""

    normalized: Counter[str] = Counter()
    for label, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Bridge action diagnostic counts must be non-negative integers")
        normalized[str(label)] += int(count)
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Bridge action diagnostics require at least one trajectory window")
    catalog = create_action_catalog(normalized, total, profile=_PROFILE)
    profile = resolve_motion_primitive_profile(_PROFILE)
    cutoff = max(
        profile.min_action_count,
        math.ceil(profile.min_action_frequency * total),
    )
    retained_counts = {
        category.label: category.raw_count
        for category in catalog.action_categories
        if category.label != "stop" and category.retained
    }

    exact_count = 0
    parent_fallback_count = 0
    no_parent_fallback_count = 0
    raw_stop_count = normalized.get("stop", 0)
    retained_atomic_occurrences = 0
    raw_atomic_occurrences = 0
    retained_atomic_ratio_mass = 0.0
    for label, count in normalized.items():
        if label == "stop" or count == 0:
            continue
        raw_atoms = atomic_action_count(label)
        raw_atomic_occurrences += count * raw_atoms
        if label in retained_counts:
            exact_count += count
            retained_atomic_occurrences += count * raw_atoms
            retained_atomic_ratio_mass += count
            continue
        parents = maximum_retained_parents(label, retained_counts)
        if parents == ("stop",):
            no_parent_fallback_count += count
            continue
        parent_fallback_count += count
        parent_atoms = atomic_action_count(parents[0])
        retained_atomic_occurrences += count * parent_atoms
        retained_atomic_ratio_mass += count * (parent_atoms / raw_atoms)

    estimated_leaf_count = sum(
        cluster_count_for_training_count(category.training_count)
        for category in catalog.action_categories
        if category.action_id is not None and category.training_count > 0
    )
    stop_or_no_parent = raw_stop_count + no_parent_fallback_count
    return {
        "profile": _PROFILE,
        "window_count": total,
        "unique_compound_labels": len(normalized),
        "retention_cutoff": cutoff,
        "retained_non_stop_action_buckets": len(retained_counts),
        "estimated_leaf_prototypes": estimated_leaf_count,
        "exact_non_stop_coverage": exact_count / total,
        "parent_fallback_rate": parent_fallback_count / total,
        "no_parent_fallback_rate": no_parent_fallback_count / total,
        "fallback_rate": (parent_fallback_count + no_parent_fallback_count) / total,
        "raw_stop_rate": raw_stop_count / total,
        "stop_or_no_parent_fallback_rate": stop_or_no_parent / total,
        "atomic_action_retention_quality": retained_atomic_ratio_mass / total,
        "atomic_occurrence_retention_quality": (
            retained_atomic_occurrences / raw_atomic_occurrences
            if raw_atomic_occurrences
            else 1.0
        ),
    }


def _axis_statistics(axis_deltas: Mapping[str, list[np.ndarray]]) -> dict[str, object]:
    contract = motion_primitive_contract(_PROFILE)
    thresholds = contract["primitive_thresholds"]
    assert isinstance(thresholds, Mapping)
    result: dict[str, object] = {}
    for name, _, family in _AXES:
        values = np.concatenate(axis_deltas[name]).astype(np.float64, copy=False)
        absolute = np.abs(values)
        threshold = float(thresholds[family])
        result[name] = {
            "threshold": threshold,
            "activation_rate": float(np.mean(absolute > threshold)),
            "positive_activation_rate": float(np.mean(values > threshold)),
            "negative_activation_rate": float(np.mean(values < -threshold)),
            "absolute_delta_quantiles": {
                format(quantile, ".2f"): float(np.quantile(absolute, quantile))
                for quantile in _QUANTILES
            },
        }
    return result


def analyze_bridge_action_windows(
    adapter: DatasetAdapter,
    *,
    max_episodes: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Scan Bridge states only and report classification behavior without writes."""

    profile = resolve_motion_primitive_profile(_PROFILE)
    records = list(adapter.episodes())
    if max_episodes is not None:
        records = records[:max_episodes]
    expected = {record.episode_id: record for record in records}
    if len(expected) != len(records):
        raise ValueError("Bridge action diagnostics found duplicate episode ids")

    counts: Counter[str] = Counter()
    axis_deltas: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in _AXES}
    seen: set[int] = set()
    for episode in adapter.iter_episodes(
        num_workers=num_workers,
        max_episodes=max_episodes,
        load_images=False,
    ):
        record = expected.get(episode.episode_id)
        if record is None or episode.episode_id in seen:
            raise ValueError(
                f"Bridge action diagnostics found unexpected episode {episode.episode_id}"
            )
        seen.add(episode.episode_id)
        try:
            states = np.asarray(episode.observations.get(STATE_KEY), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"episode {episode.episode_id}: observation.state must be finite [time, dim>=8]"
            ) from error
        if (
            states.ndim != 2
            or states.shape[0] != record.length
            or states.shape[1] < 8
            or not np.all(np.isfinite(states))
        ):
            raise ValueError(
                f"episode {episode.episode_id}: observation.state must be finite [time, dim>=8]"
            )
        starts = np.asarray(trajectory_window_starts(len(states)), dtype=np.int64)
        if len(starts) == 0:
            continue
        future = starts + TRAJECTORY_WINDOW_LENGTH - 1
        deltas = states[future] - states[starts]
        for axis in profile.primitive_config.cyclic_axes:
            outside = (deltas[:, axis] < -math.pi) | (deltas[:, axis] >= math.pi)
            deltas[outside, axis] = (
                (deltas[outside, axis] + math.pi) % (2.0 * math.pi) - math.pi
            )
        for name, axis, _ in _AXES:
            axis_deltas[name].append(deltas[:, axis].copy())
        counts.update(
            classify_motion_primitive(states[start], states[end], profile.primitive_config)
            for start, end in zip(starts, future, strict=True)
        )

    if seen != set(expected):
        raise ValueError("Bridge action diagnostics did not receive every indexed episode")
    summary = summarize_bridge_action_counts(counts)
    return {
        **summary,
        "episode_count": len(records),
        "motion_primitive": motion_primitive_contract(_PROFILE),
        "axis_statistics": _axis_statistics(axis_deltas),
    }


def validate_reference_acceptance(report: Mapping[str, object]) -> tuple[str, ...]:
    """Return deviations from the fixed full BridgeData V2 acceptance contract."""

    checks = (
        (report.get("window_count") == 384_946, "window_count must equal 384946"),
        (
            report.get("unique_compound_labels") == 2_089,
            "unique_compound_labels must equal 2089",
        ),
        (
            report.get("retained_non_stop_action_buckets") == 179,
            "retained_non_stop_action_buckets must equal 179",
        ),
        (
            report.get("estimated_leaf_prototypes") == 2_058,
            "estimated_leaf_prototypes must equal 2058",
        ),
        (
            abs(float(report.get("exact_non_stop_coverage", math.nan)) - 0.6560) <= 0.0005,
            "exact_non_stop_coverage must be within 0.05 percentage points of 65.60%",
        ),
        (
            abs(float(report.get("raw_stop_rate", math.nan)) - 0.0304) <= 0.0005,
            "raw_stop_rate must be within 0.05 percentage points of 3.04%",
        ),
        (
            float(report.get("stop_or_no_parent_fallback_rate", math.inf)) <= 0.035,
            "stop_or_no_parent_fallback_rate must not exceed 3.5%",
        ),
        (
            float(report.get("atomic_action_retention_quality", -math.inf)) >= 0.868,
            "atomic_action_retention_quality must be at least 86.8%",
        ),
    )
    return tuple(message for passed, message in checks if not passed)
