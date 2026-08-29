"""Strict configuration for the isolated Cocore LIBERO ablation runner."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from cocore.config import DEFAULT_CONFIG as COCORE_DEFAULT_CONFIG
from cocore.config import resolve_config as resolve_cocore_config


DEFAULT_CONFIG: dict[str, Any] = copy.deepcopy(COCORE_DEFAULT_CONFIG)
DEFAULT_CONFIG["reliability_metrics"] = ["support", "progress"]
DEFAULT_CONFIG["prototypes"].update(
    {
        "representation": "action_visual",
        "use_assignment_confidence": True,
    }
)
DEFAULT_CONFIG["objective"] = {
    "relation": "sequence",
    "relation_weight": 1.0,
    "redundancy_weight": 1.0,
}
DEFAULT_CONFIG["selection"].update(
    {
        "strategy": "random_multibranch",
        "use_coverage_seed": True,
    }
)
DEFAULT_CONFIG["upstream"] = {"directory": "outputs/cocore/libero90"}
DEFAULT_CONFIG["output"] = {"directory": "outputs/cocore_ablation/libero90"}

_METRIC_ORDER = ("support", "progress")
_TOP_LEVEL_KEYS = frozenset(DEFAULT_CONFIG)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_metrics(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("reliability_metrics must be a list")
    metrics = list(value)
    if any(not isinstance(metric, str) for metric in metrics):
        raise ValueError("reliability_metrics must contain only metric names")
    if len(metrics) != len(set(metrics)):
        raise ValueError("reliability_metrics cannot contain duplicates")
    unknown = sorted(set(metrics) - set(_METRIC_ORDER))
    if unknown:
        raise ValueError(f"reliability_metrics contains unknown metrics: {unknown}")
    enabled = set(metrics)
    return [metric for metric in _METRIC_ORDER if metric in enabled]


def _reject_unknown_fields(
    config: Mapping[str, Any], section: str, allowed: set[str]
) -> None:
    value = config.get(section)
    if value is None or not isinstance(value, Mapping):
        return
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{section} contains unsupported fields: {names}")


def resolve_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("configuration root must be a mapping")
    unknown_top_level = set(config) - _TOP_LEVEL_KEYS
    if unknown_top_level:
        names = ", ".join(sorted(unknown_top_level))
        raise ValueError(f"cocore_ablation contains unsupported sections: {names}")
    for section in ("prototypes", "objective", "selection", "upstream"):
        _reject_unknown_fields(config, section, set(DEFAULT_CONFIG[section]))
    resolved = _merge(DEFAULT_CONFIG, config)

    metrics = _validate_metrics(resolved.get("reliability_metrics"))
    prototypes = resolved["prototypes"]
    representation = prototypes.get("representation")
    if representation not in {"action_visual", "action_only"}:
        raise ValueError(
            "prototypes.representation must be action_visual or action_only"
        )
    confidence = prototypes.get("use_assignment_confidence")
    if not isinstance(confidence, bool):
        raise ValueError("prototypes.use_assignment_confidence must be a boolean")
    if prototypes.get("profile") != "libero":
        raise ValueError("cocore_ablation supports only the LIBERO profile")

    objective = resolved["objective"]
    redundancy_value = objective.get("redundancy_weight")
    if isinstance(redundancy_value, bool):
        raise ValueError("objective.redundancy_weight must be finite and non-negative")
    try:
        redundancy_weight = float(redundancy_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "objective.redundancy_weight must be finite and non-negative"
        ) from error
    if not math.isfinite(redundancy_weight) or redundancy_weight < 0.0:
        raise ValueError("objective.redundancy_weight must be finite and non-negative")

    selection = resolved["selection"]
    strategy = selection.get("strategy")
    if strategy not in {"random_multibranch", "random"}:
        raise ValueError("selection.strategy must be random_multibranch or random")
    use_coverage_seed = selection.get("use_coverage_seed")
    if not isinstance(use_coverage_seed, bool):
        raise ValueError("selection.use_coverage_seed must be a boolean")

    upstream = resolved.get("upstream")
    if not isinstance(upstream, Mapping):
        raise ValueError("upstream must be a mapping")
    upstream_directory = upstream.get("directory")
    if not isinstance(upstream_directory, str) or not upstream_directory.strip():
        raise ValueError("upstream.directory must be a non-empty path")

    cocore_payload = copy.deepcopy(resolved)
    cocore_payload.pop("upstream", None)
    cocore_payload["reliability_metrics"] = ["support", "progress"]
    for name in ("representation", "use_assignment_confidence"):
        cocore_payload["prototypes"].pop(name, None)
    cocore_payload["objective"].pop("redundancy_weight", None)
    for name in ("strategy", "use_coverage_seed"):
        cocore_payload["selection"].pop(name, None)
    validated_base = resolve_cocore_config(cocore_payload)

    validated_base["reliability_metrics"] = metrics
    validated_base["prototypes"].update(
        {
            "representation": representation,
            "use_assignment_confidence": confidence,
        }
    )
    validated_base["objective"]["redundancy_weight"] = redundancy_weight
    validated_base["selection"].update(
        {
            "strategy": strategy,
            "use_coverage_seed": use_coverage_seed,
        }
    )
    validated_base["upstream"] = {"directory": upstream_directory}
    validated_base["output"] = copy.deepcopy(resolved["output"])
    return validated_base


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("configuration root must be a mapping")
    return resolve_config(payload)


def to_cocore_config(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Strip ablation-only fields and point Cocore at the shared upstream cache."""

    payload = copy.deepcopy(dict(resolved))
    upstream = payload.pop("upstream")
    payload["reliability_metrics"] = ["support", "progress"]
    payload["prototypes"].pop("representation", None)
    payload["prototypes"].pop("use_assignment_confidence", None)
    payload["objective"].pop("redundancy_weight", None)
    payload["selection"].pop("strategy", None)
    payload["selection"].pop("use_coverage_seed", None)
    payload["output"] = {"directory": str(upstream["directory"])}
    return resolve_cocore_config(payload)
