"""Cocore defaults, validation, and RelCore-stage translation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml

from relcore.config import DEFAULT_CONFIG as RELCORE_DEFAULT_CONFIG
from relcore.config import resolve_config as resolve_relcore_config


_SHARED_SECTIONS = (
    "dataset",
    "visual",
    "quality",
    "prototypes",
    "graph",
    "runtime",
)

SELECTION_METHODS = ("lazy_heap", "random_multibranch")

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    **{section: copy.deepcopy(RELCORE_DEFAULT_CONFIG[section]) for section in _SHARED_SECTIONS},
    "encoding": {
        "visual_dim": 128,
        "pca_fit_max_samples": None,
        "quantile_low": 0.01,
        "quantile_high": 0.99,
        "epsilon": 1.0e-8,
    },
    "reliability_metrics": ["support", "progress"],
    "objective": {},
    "selection": {
        "method": "lazy_heap",
        "ratio": 0.1,
        "budget": None,
        "max_refreshes": 100,
    },
    "output": {"directory": "outputs/cocore/libero90"},
}
DEFAULT_CONFIG["prototypes"]["method"] = "motion_primitives"
DEFAULT_CONFIG["prototypes"]["profile"] = "libero"
DEFAULT_CONFIG["prototypes"]["tol"] = 1.0e-4
DEFAULT_CONFIG["prototypes"]["num_threads"] = 4
DEFAULT_CONFIG["prototypes"]["use_stop_bucket"] = True
for _obsolete_prototype_field in ("count", "top_r", "temperature"):
    DEFAULT_CONFIG["prototypes"].pop(_obsolete_prototype_field, None)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if "clip" in config:
        raise ValueError(
            "cocore clip configuration was removed; candidates use fixed near-uniform "
            "15-frame windows"
        )
    if "relation" in config:
        raise ValueError("cocore relation encoding configuration was removed; use encoding instead")
    if "normalization" in config:
        raise ValueError("cocore normalization configuration was removed; use encoding instead")
    configured_encoding = config.get("encoding")
    if configured_encoding is not None and not isinstance(configured_encoding, Mapping):
        raise ValueError("cocore encoding must be a mapping")
    configured_objective = config.get("objective")
    if isinstance(configured_objective, Mapping) and "cooccurrence_weight" in configured_objective:
        raise ValueError("objective.cooccurrence_weight was removed; use objective.relation_weight")
    if not isinstance(configured_objective, Mapping) or "relation" not in configured_objective:
        raise ValueError("objective.relation is required")
    if "relation_weight" not in configured_objective:
        raise ValueError("objective.relation_weight is required")
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
    configured_prototypes = config.get("prototypes")
    if configured_prototypes is not None and not isinstance(configured_prototypes, Mapping):
        raise ValueError("cocore prototypes must be a mapping")
    configured_method = (
        configured_prototypes.get("method") if isinstance(configured_prototypes, Mapping) else None
    )
    if isinstance(configured_prototypes, Mapping):
        obsolete = {"count", "top_r", "temperature"} & configured_prototypes.keys()
        if obsolete:
            names = ", ".join(sorted(obsolete))
            raise ValueError(
                f"cocore prototypes {names} were removed; action retention, visual K, "
                "and temperature are fixed by the schema-5 algorithm"
            )
        unsupported = configured_prototypes.keys() - {
            "method",
            "profile",
            "batch_size",
            "max_iter",
            "tol",
            "num_threads",
            "use_stop_bucket",
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"cocore prototypes contains unsupported fields: {names}")
    if configured_method not in {None, "motion_primitives"}:
        raise ValueError("cocore prototypes.method must be motion_primitives")
    configured_metrics = config.get("reliability_metrics")
    if configured_metrics is not None and list(configured_metrics) != ["support", "progress"]:
        raise ValueError("cocore reliability_metrics are fixed to support,progress")
    resolved = _merge(DEFAULT_CONFIG, config)
    resolved["prototypes"]["method"] = "motion_primitives"
    profile = resolved["prototypes"].get("profile")
    if not isinstance(profile, str) or profile not in {"libero", "bridge_v2"}:
        raise ValueError("cocore prototypes.profile must be libero or bridge_v2")
    resolved["prototypes"]["profile"] = profile
    use_stop_bucket = resolved["prototypes"].get("use_stop_bucket")
    if not isinstance(use_stop_bucket, bool):
        raise ValueError("cocore prototypes.use_stop_bucket must be a boolean")
    resolved["prototypes"]["use_stop_bucket"] = use_stop_bucket
    num_threads = resolved["prototypes"].get("num_threads")
    if (
        isinstance(num_threads, bool)
        or not isinstance(num_threads, Integral)
        or num_threads <= 0
    ):
        raise ValueError("cocore prototypes.num_threads must be a positive integer")
    resolved["prototypes"]["num_threads"] = int(num_threads)
    tolerance = resolved["prototypes"].get("tol")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise ValueError("cocore prototypes.tol must be a finite positive number")
    resolved["prototypes"]["tol"] = float(tolerance)
    resolved["reliability_metrics"] = ["support", "progress"]
    relation = str(resolved["objective"]["relation"])
    if relation not in {"sequence", "cooccurrence"}:
        raise ValueError("objective.relation must be sequence or cooccurrence")
    resolved["objective"]["relation"] = relation
    weight = float(resolved["objective"]["relation_weight"])
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("objective.relation_weight must be finite and non-negative")
    resolved["objective"]["relation_weight"] = weight
    encoding = resolved["encoding"]
    visual_dim = encoding.get("visual_dim")
    if isinstance(visual_dim, bool) or not isinstance(visual_dim, Integral) or visual_dim != 128:
        raise ValueError("cocore encoding.visual_dim must be 128")
    maximum = encoding.get("pca_fit_max_samples")
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, Integral) or maximum <= 0
    ):
        raise ValueError("cocore encoding.pca_fit_max_samples must be a positive integer or null")
    encoding["pca_fit_max_samples"] = None if maximum is None else int(maximum)
    low_value = encoding.get("quantile_low")
    high_value = encoding.get("quantile_high")
    epsilon_value = encoding.get("epsilon")
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in (low_value, high_value, epsilon_value)
    ):
        raise ValueError("cocore encoding quantiles and epsilon must be numeric")
    low = float(low_value)
    high = float(high_value)
    epsilon = float(epsilon_value)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("cocore encoding quantiles must satisfy 0 <= low < high <= 1")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("cocore encoding.epsilon must be finite and positive")
    encoding.update(
        {
            "visual_dim": 128,
            "quantile_low": low,
            "quantile_high": high,
            "epsilon": epsilon,
        }
    )
    ratio = float(resolved["selection"]["ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("selection.ratio must be in (0, 1]")
    resolved["selection"]["ratio"] = ratio
    selection_method = str(resolved["selection"].get("method", "lazy_heap"))
    if selection_method not in SELECTION_METHODS:
        raise ValueError(
            f"selection.method must be one of {', '.join(SELECTION_METHODS)}"
        )
    resolved["selection"]["method"] = selection_method
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
        if section == "prototypes":
            translated[section].update(
                {
                    key: copy.deepcopy(value)
                    for key, value in resolved[section].items()
                    if key not in {"num_threads", "profile", "tol", "use_stop_bucket"}
                }
            )
        else:
            translated[section] = copy.deepcopy(resolved[section])
    translated["clip"] = {"length": 15, "stride": 15}
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
