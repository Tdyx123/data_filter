"""Configuration defaults and validation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "dataset": {
        "type": "lerobot",
        "name": "libero90",
        "path": "/data/dwb/datasets/LIBERO_lerobot/libero90",
        "use_images": True,
        "empty_task_policy": "error",
        "feature_keys": {
            "action": "action",
            "timestamp": "timestamp",
            "frame_index": "frame_index",
            "episode_index": "episode_index",
            "vector_observations": ["observation.state"],
            "image_observations": ["observation.images.image"],
        },
    },
    "clip": {"length": 15, "stride": 15},
    "visual": {
        "encoder": "clip",
        "model": "/data/dwb/models/clip-vit-base-patch32",
        "local_files_only": True,
        "batch_size": 64,
        "device": "auto",
    },
    "normalization": {"epsilon": 1.0e-6},
    "relation": {
        "projection_dim": 32,
        "output_dim": 256,
        "lags": [0, 1, 2, 4],
    },
    "quality": {
        "knn": 10,
        "gripper_progress_weight": 0.5,
        "visual_progress_weight": 0.25,
        "noop_threshold": 1.0e-4,
        "gripper_action_index": -1,
        "min_reliability": 0.05,
    },
    "prototypes": {
        "method": "kmeans",
        "count": 64,
        "batch_size": 4096,
        "max_iter": 100,
        "top_r": 3,
        "temperature": 0.1,
    },
    "graph": {
        "knn": 32,
        "similarity_threshold": 0.8,
        "cooccurrence_max_gap": 4,
    },
    "objective": {
        "node_weight": 4.0,
        "transition_weight": 1.2,
        "cooccurrence_weight": 0.5,
        "sequence_weight": 2.0,
        "redundancy_weight": 2.0,
    },
    "selection": {
        "ratio": 0.1,
        "budget": None,
        "quota_mode": "proportional",
        "minimum_per_task": 1,
        "engine": "sparse",
        "branches": 8,
        "seed_candidates": 128,
        "seed_similarity_threshold": 0.9,
        "transition_seed_threshold": 0.0,
        "seed_pairs_per_transition": 4,
        "global_candidates": 256,
        "residual_candidates": 128,
        "random_candidates": 128,
        "local_search": {
            "enabled": True,
            "max_selected_candidates": 128,
            "max_unselected_candidates": 256,
            "max_rounds": 2,
        },
    },
    "runtime": {"num_workers": 0, "max_episodes": None, "resume": True},
    "output": {"directory": "outputs/relcore/libero90"},
}


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_config(config: Mapping[str, Any]) -> dict[str, Any]:
    configured_quality = config.get("quality")
    if isinstance(configured_quality, Mapping) and "reliability_metrics" in configured_quality:
        raise ValueError(
            "quality.reliability_metrics was removed; use --reliability-metrics instead"
        )
    configured_objective = config.get("objective")
    if (
        isinstance(configured_objective, Mapping)
        and "prototype_gain_metrics" in configured_objective
    ):
        raise ValueError(
            "objective.prototype_gain_metrics is not configurable; "
            "use --prototype-gain-metrics instead"
        )
    resolved = _merge(DEFAULT_CONFIG, config)
    if resolved["dataset"].get("empty_task_policy") not in {"error", "exclude"}:
        raise ValueError("dataset.empty_task_policy must be error or exclude")
    if (int(resolved["clip"]["length"]), int(resolved["clip"]["stride"])) != (15, 15):
        raise ValueError("relcore requires SQCN-compatible 15-frame windows with stride 15")
    lags = tuple(int(lag) for lag in resolved["relation"]["lags"])
    if not lags or min(lags) < 0 or max(lags) >= int(resolved["clip"]["length"]) - 1:
        raise ValueError("relation lags must fit both frame and delta sequences")
    if (
        int(resolved["relation"]["projection_dim"]) <= 0
        or int(resolved["relation"]["output_dim"]) <= 0
    ):
        raise ValueError("relation dimensions must be positive")
    if resolved["visual"]["encoder"] == "clip":
        if resolved["visual"].get("local_files_only") is not True:
            raise ValueError("production CLIP must use local_files_only=true")
        expected_model = Path("/data/dwb/models/clip-vit-base-patch32")
        if Path(str(resolved["visual"].get("model", ""))).expanduser() != expected_model:
            raise ValueError(f"production CLIP must use the fixed local model {expected_model}")
    elif resolved["visual"]["encoder"] != "dummy":
        raise ValueError("visual.encoder must be clip or dummy")
    if int(resolved["visual"].get("batch_size", 1)) <= 0:
        raise ValueError("visual.batch_size must be positive")
    quality = resolved["quality"]
    if int(quality["knn"]) <= 0 or float(quality["noop_threshold"]) < 0:
        raise ValueError("quality knn/noop_threshold configuration is invalid")
    if (
        float(quality["gripper_progress_weight"]) < 0
        or float(quality["visual_progress_weight"]) < 0
    ):
        raise ValueError("quality progress weights cannot be negative")
    if not 0.0 < float(quality["min_reliability"]) <= 1.0:
        raise ValueError("quality.min_reliability must be in (0, 1]")
    prototype_count = int(resolved["prototypes"]["count"])
    prototype_method = str(resolved["prototypes"].get("method", ""))
    if prototype_method not in {"kmeans", "motion_primitives"}:
        raise ValueError("prototypes.method must be kmeans or motion_primitives")
    top_r = int(resolved["prototypes"]["top_r"])
    if prototype_count <= 0 or not 0 < top_r <= prototype_count:
        raise ValueError("prototype count/top_r configuration is invalid")
    if (
        int(resolved["prototypes"]["batch_size"]) <= 0
        or int(resolved["prototypes"]["max_iter"]) <= 0
    ):
        raise ValueError("prototype batch_size/max_iter must be positive")
    if float(resolved["prototypes"]["temperature"]) <= 0:
        raise ValueError("prototype temperature must be positive")
    similarity_threshold = float(resolved["graph"]["similarity_threshold"])
    if int(resolved["graph"]["knn"]) <= 0 or not 0.0 <= similarity_threshold < 1.0:
        raise ValueError("graph knn/similarity_threshold configuration is invalid")
    if int(resolved["graph"]["cooccurrence_max_gap"]) < 2:
        raise ValueError("cooccurrence_max_gap must be at least two")
    if any(float(value) < 0 for value in resolved["objective"].values()):
        raise ValueError("objective weights cannot be negative")
    if resolved["selection"]["engine"] not in {"exact", "sparse"}:
        raise ValueError("selection.engine must be exact or sparse")
    quota_mode = resolved["selection"]["quota_mode"]
    if quota_mode not in {"proportional", "none"}:
        raise ValueError("selection.quota_mode must be proportional or none")
    if int(resolved["selection"]["minimum_per_task"]) < 0:
        raise ValueError("selection.minimum_per_task cannot be negative")
    if quota_mode == "none" and int(resolved["selection"]["minimum_per_task"]) != 0:
        raise ValueError("selection.quota_mode=none requires minimum_per_task=0")
    if not 0.0 <= float(resolved["selection"]["seed_similarity_threshold"]) <= 1.0:
        raise ValueError("selection.seed_similarity_threshold must be in [0, 1]")
    if float(resolved["selection"]["transition_seed_threshold"]) < 0:
        raise ValueError("selection.transition_seed_threshold cannot be negative")
    ratio = float(resolved["selection"]["ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("selection.ratio must be in (0, 1]")
    budget = resolved["selection"].get("budget")
    if budget is not None and int(budget) <= 0:
        raise ValueError("selection.budget must be positive when configured")
    integer_selection_keys = (
        "branches",
        "seed_candidates",
        "seed_pairs_per_transition",
        "global_candidates",
        "residual_candidates",
        "random_candidates",
    )
    if any(int(resolved["selection"][key]) <= 0 for key in integer_selection_keys):
        raise ValueError("selection candidate and branch counts must be positive")
    local = resolved["selection"]["local_search"]
    if not 0 <= int(local["max_rounds"]) <= 2:
        raise ValueError("local search supports at most two rounds")
    if int(local["max_selected_candidates"]) <= 0 or int(local["max_unselected_candidates"]) <= 0:
        raise ValueError("local search candidate counts must be positive")
    if int(resolved["runtime"].get("num_workers", 0)) < 0:
        raise ValueError("runtime.num_workers cannot be negative")
    max_episodes = resolved["runtime"].get("max_episodes")
    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("runtime.max_episodes must be positive when configured")
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("relcore config root must be a mapping")
    return resolve_config(payload)
