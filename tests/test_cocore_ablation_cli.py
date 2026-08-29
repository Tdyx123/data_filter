from __future__ import annotations

from pathlib import Path

import pytest

from cocore_ablation import cli


def test_cli_exposes_all_ablation_overrides() -> None:
    args = cli.build_parser().parse_args(
        [
            "run",
            "--reliability-metrics",
            "none",
            "--prototype-representation",
            "action_only",
            "--no-assignment-confidence",
            "--no-use-stop-bucket",
            "--relation",
            "cooccurrence",
            "--relation-weight",
            "0",
            "--redundancy-weight",
            "0",
            "--no-coverage-seed",
            "--selection-strategy",
            "random",
            "--selection-ratio",
            "0.2",
        ]
    )

    assert args.reliability_metrics == []
    assert args.prototype_representation == "action_only"
    assert args.no_assignment_confidence is True
    assert args.no_use_stop_bucket is True
    assert args.relation == "cooccurrence"
    assert args.relation_weight == 0.0
    assert args.redundancy_weight == 0.0
    assert args.no_coverage_seed is True
    assert args.selection_strategy == "random"
    assert args.selection_ratio == 0.2


@pytest.mark.parametrize("value", ["smoothness", "support,support", "progress,support"])
def test_cli_rejects_noncanonical_reliability_metric_lists(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--reliability-metrics", value])


def test_cli_applies_overrides_before_running(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: {
            "reliability_metrics": ["support", "progress"],
            "prototypes": {
                "representation": "action_visual",
                "use_assignment_confidence": True,
                "use_stop_bucket": True,
            },
            "objective": {
                "relation": "sequence",
                "relation_weight": 1.0,
                "redundancy_weight": 1.0,
            },
            "selection": {
                "strategy": "random_multibranch",
                "use_coverage_seed": True,
                "ratio": 0.1,
                "budget": None,
            },
        },
    )

    def fake_run(config, **kwargs):
        received["config"] = config
        received.update(kwargs)
        return Path("outputs/cocore_ablation/result")

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    cli.main(
        [
            "run",
            "--config",
            "unused.yaml",
            "--prototype-representation",
            "action_only",
            "--redundancy-weight",
            "0.5",
            "--no-coverage-seed",
            "--force",
        ]
    )

    config = received["config"]
    assert config["prototypes"]["representation"] == "action_only"
    assert config["objective"]["redundancy_weight"] == 0.5
    assert config["selection"]["use_coverage_seed"] is False
    assert received["force"] is True
    assert capsys.readouterr().out.strip() == (
        "cocore_ablation_output=outputs/cocore_ablation/result"
    )
