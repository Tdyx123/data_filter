from __future__ import annotations

from pathlib import Path

import pytest

from cocore import cli
from cocore.config import load_config, resolve_config
from cocore.pipeline import selection_directory_name


def test_config_fixes_motion_primitives_and_support_progress_reliability() -> None:
    resolved = resolve_config({})

    assert resolved["prototypes"]["method"] == "motion_primitives"
    assert resolved["reliability_metrics"] == ["support", "progress"]
    assert resolved["objective"]["cooccurrence_weight"] == 1.0
    assert resolved["selection"]["max_refreshes"] == 100


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_config_rejects_invalid_cooccurrence_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="cooccurrence_weight"):
        resolve_config({"objective": {"cooccurrence_weight": weight}})


@pytest.mark.parametrize("max_refreshes", [0, -1, 1.5, True])
def test_config_rejects_non_positive_or_non_integer_max_refreshes(max_refreshes) -> None:
    with pytest.raises(ValueError, match="selection.max_refreshes"):
        resolve_config({"selection": {"max_refreshes": max_refreshes}})


@pytest.mark.parametrize(
    "obsolete",
    [
        "global_candidates",
        "prototype_candidates",
        "similarity_candidates",
        "random_candidates",
    ],
)
def test_config_rejects_removed_candidate_pool_options(obsolete: str) -> None:
    with pytest.raises(ValueError, match=f"selection.{obsolete}"):
        resolve_config({"selection": {obsolete: 1}})


def test_selection_directory_always_encodes_weight_and_ratio() -> None:
    assert selection_directory_name(1.5, 0.1) == "select-w1p5-top10pct"
    assert selection_directory_name(0.0, 0.125) == "select-w0-top12p5pct"


def test_run_cli_accepts_weight_and_ratio_but_rejects_relcore_switches() -> None:
    arguments = cli.build_parser().parse_args(
        ["run", "--cooccurrence-weight", "1.5", "--selection-ratio", "0.2"]
    )

    assert arguments.cooccurrence_weight == 1.5
    assert arguments.selection_ratio == 0.2
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--prototype-method", "kmeans"])


def test_main_applies_cli_overrides_to_run_pipeline(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _: {})

    def fake_run_pipeline(config, **kwargs):
        received["config"] = config
        received.update(kwargs)
        return Path("outputs/cocore/test/select-w2-top25pct")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(
        [
            "run",
            "--config",
            "unused.yaml",
            "--cooccurrence-weight",
            "2",
            "--selection-ratio",
            "0.25",
        ]
    )

    assert received["config"]["objective"]["cooccurrence_weight"] == 2.0
    assert received["config"]["selection"]["ratio"] == 0.25
    assert received["config"]["selection"]["budget"] is None
    assert capsys.readouterr().out.strip().endswith("select-w2-top25pct")


@pytest.mark.parametrize("path", ["cocore/config_libero90.yaml", "cocore/config_debug.yaml"])
def test_shipped_configs_resolve_to_fixed_cocore_contract(path: str) -> None:
    config = load_config(path)

    assert config["prototypes"]["method"] == "motion_primitives"
    assert config["reliability_metrics"] == ["support", "progress"]
    assert config["selection"]["max_refreshes"] == 100
    assert not {
        "global_candidates",
        "prototype_candidates",
        "similarity_candidates",
        "random_candidates",
    } & config["selection"].keys()
    assert config["output"]["directory"].startswith("outputs/cocore/")
