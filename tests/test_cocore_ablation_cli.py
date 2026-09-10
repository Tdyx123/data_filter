from __future__ import annotations

from pathlib import Path

import pytest

from cocore_ablation import cli


def test_cli_exposes_all_ablation_overrides() -> None:
    args = cli.build_parser().parse_args(
        [
            "run",
            "--subfolder-name",
            "no-reliability",
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

    assert args.subfolder_name == "no-reliability"
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


def test_cli_requires_subfolder_name_for_run() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run"])


@pytest.mark.parametrize("command", ["build-graph", "select"])
def test_cli_does_not_expose_individual_pipeline_stages(command: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command])


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "/absolute", "nested/name", r"nested\name"],
)
def test_cli_rejects_invalid_subfolder_names(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--subfolder-name", value])


def test_cli_keeps_validate_as_a_public_command() -> None:
    args = cli.build_parser().parse_args(
        ["validate", "--output-dir", "outputs/experiment/select"]
    )

    assert args.command == "validate"


@pytest.mark.parametrize("value", ["smoothness", "support_old,support_old", "action_jump,support_old"])
def test_cli_rejects_noncanonical_reliability_metric_lists(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "run",
                "--subfolder-name",
                "invalid-metrics",
                "--reliability-metrics",
                value,
            ]
        )


def test_cli_applies_overrides_before_running(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: {
            "reliability_metrics": ["support_old", "action_jump"],
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
            "--subfolder-name",
            "action-only",
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
    assert received["subfolder_name"] == "action-only"
    assert capsys.readouterr().out.strip() == (
        "cocore_ablation_output=outputs/cocore_ablation/result"
    )


@pytest.mark.parametrize("command", ["run", "validate"])
def test_cli_accepts_reference_baseline_flags(command):
    location = ["--subfolder-name", "full-model"] if command == "run" else ["--output-dir", "result"]
    args = cli.build_parser().parse_args([
        command, *location, "--reliability-metrics", "support_old", "action_jump",
        "--support-k", "10", "--selection-ratio", "0.20",
        "--relation", "sequence", "--relation-weight", "1.0",
    ])
    config = {}
    cli._apply_overrides(config, args)
    assert config["reliability_metrics"] == ["support_old", "action_jump"]
    assert config["quality"]["knn"] == 10
    assert config["selection"]["ratio"] == 0.20


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_support_k(value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--subfolder-name", "full", "--support-k", value])
