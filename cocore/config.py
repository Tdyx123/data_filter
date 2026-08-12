"""Cocore defaults, validation, and RelCore-stage translation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import yaml

from relcore.config import DEFAULT_CONFIG as RELCORE_DEFAULT_CONFIG
from relcore.config import resolve_config as resolve_relcore_config


_SHARED_SECTIONS = (
    "dataset",
    "clip",
    "visual",
    "normalization",
    "relation",
    "quality",
    "prototypes",
    "graph",
    "runtime",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    **{section: copy.deepcopy(RELCORE_DEFAULT_CONFIG[section]) for section in _SHARED_SECTIONS},
    "reliability_metrics": ["support", "progress"],
    "objective": {"cooccurrence_weight": 1.0},
    "selection": {
        "ratio": 0.1,
        "budget": None,
        "max_refreshes": 100,
    },
    "output": {"directory": "outputs/cocore/libero90"},
}
DEFAULT_CONFIG["prototypes"]["method"] = "motion_primitives"


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_config(config: Mapping[str, Any]) -> dict[str, Any]:
    configured_selection = config.get("selection")
    if isinstance(configured_selection, Mapping):
        for name in (
            "global_candidates",
            "prototype_candidates",
            "similarity_candidates",
            "random_candidates",
        ):
            if name in configured_selection:
                raise ValueError(
                    f"selection.{name} was removed; lazy heap selection uses max_refreshes"
                )
    configured_method = config.get("prototypes", {}).get("method") if isinstance(
        config.get("prototypes"), Mapping
    ) else None
    if configured_method not in {None, "motion_primitives"}:
        raise ValueError("cocore prototypes.method must be motion_primitives")
    configured_metrics = config.get("reliability_metrics")
    if configured_metrics is not None and list(configured_metrics) != ["support", "progress"]:
        raise ValueError("cocore reliability_metrics are fixed to support,progress")
    resolved = _merge(DEFAULT_CONFIG, config)
    resolved["prototypes"]["method"] = "motion_primitives"
    resolved["reliability_metrics"] = ["support", "progress"]
    weight = float(resolved["objective"]["cooccurrence_weight"])
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("objective.cooccurrence_weight must be finite and non-negative")
    resolved["objective"]["cooccurrence_weight"] = weight
    ratio = float(resolved["selection"]["ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("selection.ratio must be in (0, 1]")
    resolved["selection"]["ratio"] = ratio
    budget = resolved["selection"].get("budget")
    if budget is not None and int(budget) <= 0:
        raise ValueError("selection.budget must be positive or null")
    resolved["selection"]["budget"] = None if budget is None else int(budget)
    max_refreshes = resolved["selection"]["max_refreshes"]
    if isinstance(max_refreshes, bool) or not isinstance(max_refreshes, Integral):
        raise ValueError("selection.max_refreshes must be a positive integer")
    if int(max_refreshes) <= 0:
        raise ValueError("selection.max_refreshes must be a positive integer")
    resolved["selection"]["max_refreshes"] = int(max_refreshes)
    # Reuse RelCore's strict validation for all shared encoding and graph fields.
    to_relcore_config(resolved)
    return resolved


def to_relcore_config(resolved: Mapping[str, Any]) -> dict[str, Any]:
    translated = copy.deepcopy(RELCORE_DEFAULT_CONFIG)
    translated["seed"] = int(resolved.get("seed", 42))
    for section in _SHARED_SECTIONS:
        translated[section] = copy.deepcopy(resolved[section])
    translated["prototypes"]["method"] = "motion_primitives"
    translated["selection"]["ratio"] = float(resolved["selection"]["ratio"])
    translated["selection"]["budget"] = resolved["selection"].get("budget")
    translated["selection"]["quota_mode"] = "none"
    translated["selection"]["minimum_per_task"] = 0
    translated["output"] = copy.deepcopy(resolved["output"])
    return resolve_relcore_config(translated)


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("configuration root must be a mapping")
    return resolve_config(payload)
