from __future__ import annotations

import copy
import math

import pytest

from cocore_ablation.config import DEFAULT_CONFIG, load_config, resolve_config


def test_ablation_defaults_preserve_full_cocore_behavior() -> None:
    config = resolve_config(copy.deepcopy(DEFAULT_CONFIG))

    assert config["reliability_metrics"] == ["support", "progress"]
    assert config["prototypes"]["representation"] == "action_visual"
    assert config["prototypes"]["use_assignment_confidence"] is True
    assert config["objective"]["redundancy_weight"] == 1.0
    assert config["selection"]["use_coverage_seed"] is True
    assert config["selection"]["strategy"] == "random_multibranch"
    assert config["upstream"]["directory"] == "outputs/cocore/libero90"
    assert config["output"]["directory"] == "outputs/cocore_ablation/libero90"


@pytest.mark.parametrize(
    "metrics",
    [
        [],
        ["support"],
        ["progress"],
        ["support", "progress"],
    ],
)
def test_ablation_accepts_every_reliability_subset(metrics: list[str]) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["reliability_metrics"] = metrics

    assert resolve_config(config)["reliability_metrics"] == metrics


@pytest.mark.parametrize(
    "metrics",
    [["support", "support"], ["smoothness"], "support"],
)
def test_ablation_rejects_invalid_reliability_metrics(metrics: object) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["reliability_metrics"] = metrics

    with pytest.raises(ValueError, match="reliability_metrics"):
        resolve_config(config)


@pytest.mark.parametrize("weight", [-1.0, math.inf, math.nan])
def test_ablation_rejects_invalid_redundancy_weight(weight: float) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["objective"]["redundancy_weight"] = weight

    with pytest.raises(ValueError, match="redundancy_weight"):
        resolve_config(config)


def test_ablation_rejects_bridge_profile() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["prototypes"]["profile"] = "bridge_v2"

    with pytest.raises(ValueError, match="LIBERO"):
        resolve_config(config)


def test_shipped_libero_config_resolves_to_full_ablation_defaults() -> None:
    config = load_config("cocore_ablation/config_libero90.yaml")

    assert config["prototypes"]["profile"] == "libero"
    assert config["prototypes"]["representation"] == "action_visual"
    assert config["objective"] == {
        "relation": "sequence",
        "relation_weight": 1.0,
        "redundancy_weight": 1.0,
    }
    assert config["selection"]["strategy"] == "random_multibranch"


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("prototypes", "visual_clusters"),
        ("objective", "diversity_weight"),
        ("selection", "branches"),
        ("upstream", "force"),
    ],
)
def test_ablation_rejects_unknown_public_fields(section: str, field: str) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config[section][field] = 1

    with pytest.raises(ValueError, match=field):
        resolve_config(config)
