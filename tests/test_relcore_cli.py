from __future__ import annotations

import copy
from pathlib import Path

import pytest

from relcore import cli
from relcore import pipeline


@pytest.mark.parametrize("command", ["select", "run"])
def test_selection_commands_accept_selection_ratio(command: str) -> None:
    arguments = cli.build_parser().parse_args([command, "--selection-ratio", "0.20"])

    assert arguments.selection_ratio == pytest.approx(0.20)


@pytest.mark.parametrize("command", ["scan", "encode", "build-graph"])
def test_nonselection_commands_reject_selection_ratio(command: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "--selection-ratio", "0.20"])


@pytest.mark.parametrize("value", ["not-a-number", "0", "-0.10", "1.10"])
def test_selection_ratio_rejects_invalid_values(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--selection-ratio", value])


@pytest.mark.parametrize("command", ["select", "run"])
@pytest.mark.parametrize("quota_mode", ["proportional", "none"])
def test_selection_commands_accept_quota_mode(command: str, quota_mode: str) -> None:
    arguments = cli.build_parser().parse_args([command, "--quota-mode", quota_mode])

    assert arguments.quota_mode == quota_mode


@pytest.mark.parametrize("command", ["scan", "encode", "build-graph"])
def test_nonselection_commands_reject_quota_mode(command: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "--quota-mode", "none"])


def test_quota_mode_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--quota-mode", "balanced"])


@pytest.mark.parametrize("command", ["build-graph", "select", "run"])
def test_graph_commands_accept_reliability_metric_names(command: str) -> None:
    arguments = cli.build_parser().parse_args(
        [command, "--reliability-metrics", "non_noop,progress"]
    )

    assert arguments.reliability_metrics == ("progress", "non_noop")


@pytest.mark.parametrize("command", ["scan", "encode"])
def test_pregraph_commands_reject_reliability_metrics(command: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "--reliability-metrics", "progress"])


@pytest.mark.parametrize(
    "value",
    ["", "support,support", "support,unknown"],
)
def test_reliability_metrics_reject_invalid_values(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--reliability-metrics", value])


@pytest.mark.parametrize("command", ["select", "run"])
def test_selection_commands_accept_prototype_gain_metric_names(command: str) -> None:
    arguments = cli.build_parser().parse_args(
        [command, "--prototype-gain-metrics", "sequence,transition"]
    )

    assert arguments.prototype_gain_metrics == ("transition", "sequence")


@pytest.mark.parametrize("command", ["scan", "encode", "build-graph"])
def test_preselection_commands_reject_prototype_gain_metrics(command: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [command, "--prototype-gain-metrics", "transition"]
        )


@pytest.mark.parametrize(
    "value",
    ["", "transition,transition", "transition,unknown"],
)
def test_prototype_gain_metrics_reject_invalid_values(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--prototype-gain-metrics", value])


@pytest.mark.parametrize("command", ["build-graph", "run"])
def test_graph_building_commands_accept_prototype_method(command: str) -> None:
    arguments = cli.build_parser().parse_args([command, "--prototype-method", "motion_primitives"])

    assert arguments.prototype_method == "motion_primitives"


@pytest.mark.parametrize("command", ["scan", "encode", "select"])
def test_commands_that_do_not_choose_graph_method_reject_prototype_method(command: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "--prototype-method", "motion_primitives"])


def test_graph_directory_name_uses_reliability_mask() -> None:
    assert pipeline.graph_directory_name(["progress", "non_noop"]) == "graph-5"


def test_motion_primitive_graph_directory_is_isolated_from_kmeans() -> None:
    assert (
        pipeline.graph_directory_name(["progress", "non_noop"], "motion_primitives")
        == "graph-5-motion-primitives"
    )


@pytest.mark.parametrize(
    ("metrics", "gain_metrics", "ratio", "expected"),
    [
        (
            ["support", "progress", "smoothness", "non_noop"],
            ["transition", "cooccurrence", "sequence"],
            None,
            "select-r15-g7",
        ),
        (["progress", "non_noop"], ["transition", "sequence"], 0.20, "select-r5-g5-top20pct"),
        (["non_noop"], ["cooccurrence"], 0.125, "select-r1-g2-top12p5pct"),
        (["support"], ["sequence"], 1.0, "select-r8-g1-top100pct"),
    ],
)
def test_selection_directory_name_uses_canonical_percent_tag(
    metrics: list[str],
    gain_metrics: list[str],
    ratio: float | None,
    expected: str,
) -> None:
    assert (
        pipeline.selection_directory_name(
            metrics,
            ratio,
            prototype_gain_metrics=gain_metrics,
        )
        == expected
    )


def test_motion_primitive_selection_directory_is_isolated_from_kmeans() -> None:
    assert (
        pipeline.selection_directory_name(
            ["support", "progress", "smoothness", "non_noop"],
            0.125,
            "motion_primitives",
            prototype_gain_metrics=["transition", "sequence"],
        )
        == "select-r15-g5-motion-primitives-top12p5pct"
    )


@pytest.mark.parametrize(
    ("quota_mode", "expected"),
    [
        ("proportional", "select-r15-g7-quota-proportional"),
        ("none", "select-r15-g7-quota-none"),
    ],
)
def test_explicit_quota_mode_isolates_selection_directory(
    quota_mode: str,
    expected: str,
) -> None:
    assert (
        pipeline.selection_directory_name(
            ["support", "progress", "smoothness", "non_noop"],
            prototype_gain_metrics=["transition", "cooccurrence", "sequence"],
            quota_mode=quota_mode,
        )
        == expected
    )


def test_quota_mode_suffix_follows_method_and_ratio_scopes() -> None:
    assert (
        pipeline.selection_directory_name(
            ["support", "progress", "smoothness", "non_noop"],
            0.20,
            "motion_primitives",
            prototype_gain_metrics=["transition", "cooccurrence", "sequence"],
            quota_mode="none",
        )
        == "select-r15-g7-motion-primitives-top20pct-quota-none"
    )


def test_main_run_prototype_method_overrides_loaded_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "runtime": {"max_episodes": None},
        "selection": {"ratio": 0.10, "budget": None},
        "prototypes": {"method": "kmeans"},
    }
    received: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def fake_run_pipeline(
        resolved: dict[str, object],
        *,
        output_dir: str | None,
        force: bool,
        selection_output_ratio: float | None,
        selection_output_quota_mode: str | None,
        reliability_metrics: tuple[str, ...],
        prototype_gain_metrics: tuple[str, ...],
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        return Path("outputs/relcore/test/select-r15-g7-motion-primitives")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(["run", "--config", "unused.yaml", "--prototype-method", "motion_primitives"])

    assert received["config"] == {
        "runtime": {"max_episodes": None},
        "selection": {"ratio": 0.10, "budget": None},
        "prototypes": {"method": "motion_primitives"},
    }


def test_main_select_ratio_scopes_output_and_reports_selection_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {
        "runtime": {"max_episodes": None},
        "selection": {"ratio": 0.10, "budget": 100},
    }
    received: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def fake_selection_stage(
        resolved: dict[str, object],
        *,
        output_dir: str | None,
        force: bool,
        selection_output_ratio: float | None,
        selection_output_quota_mode: str | None,
        reliability_metrics: tuple[str, ...],
        prototype_gain_metrics: tuple[str, ...],
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        received["output_dir"] = output_dir
        received["force"] = force
        received["selection_output_ratio"] = selection_output_ratio
        received["selection_output_quota_mode"] = selection_output_quota_mode
        received["reliability_metrics"] = reliability_metrics
        received["prototype_gain_metrics"] = prototype_gain_metrics
        return Path("outputs/relcore/test/select-r5-g5-top25pct")

    monkeypatch.setattr(cli, "select_stage", fake_selection_stage)

    cli.main(
        [
            "select",
            "--config",
            "unused.yaml",
            "--selection-ratio",
            "0.25",
            "--reliability-metrics",
            "progress,non_noop",
            "--prototype-gain-metrics",
            "sequence,transition",
        ]
    )

    assert received["config"] == {
        "runtime": {"max_episodes": None},
        "selection": {"ratio": 0.25, "budget": None},
    }
    assert received["selection_output_ratio"] == pytest.approx(0.25)
    assert received["selection_output_quota_mode"] is None
    assert received["reliability_metrics"] == ("progress", "non_noop")
    assert received["prototype_gain_metrics"] == ("transition", "sequence")
    assert capsys.readouterr().out.strip() == (
        "relcore_output=outputs/relcore/test/select-r5-g5-top25pct"
    )


def test_main_run_ratio_reports_metric_and_ratio_scoped_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {
        "runtime": {"max_episodes": None},
        "selection": {"ratio": 0.10, "budget": 100},
    }
    received: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def fake_run_pipeline(
        resolved: dict[str, object],
        *,
        output_dir: str | None,
        force: bool,
        selection_output_ratio: float | None,
        selection_output_quota_mode: str | None,
        reliability_metrics: tuple[str, ...],
        prototype_gain_metrics: tuple[str, ...],
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        received["selection_output_ratio"] = selection_output_ratio
        received["selection_output_quota_mode"] = selection_output_quota_mode
        received["reliability_metrics"] = reliability_metrics
        received["prototype_gain_metrics"] = prototype_gain_metrics
        return Path("outputs/relcore/test/select-r15-g7-top25pct")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(
        [
            "run",
            "--config",
            "unused.yaml",
            "--selection-ratio",
            "0.25",
        ]
    )

    assert received["config"] == {
        "runtime": {"max_episodes": None},
        "selection": {"ratio": 0.25, "budget": None},
    }
    assert received["selection_output_ratio"] == pytest.approx(0.25)
    assert received["selection_output_quota_mode"] is None
    assert received["reliability_metrics"] == (
        "support",
        "progress",
        "smoothness",
        "non_noop",
    )
    assert received["prototype_gain_metrics"] == (
        "transition",
        "cooccurrence",
        "sequence",
    )
    assert capsys.readouterr().out.strip() == (
        "relcore_output=outputs/relcore/test/select-r15-g7-top25pct"
    )


def test_main_select_without_ratio_uses_default_metric_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {
        "runtime": {"max_episodes": None},
        "selection": {"ratio": 0.10, "budget": 100},
    }
    received: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def fake_selection_stage(
        resolved: dict[str, object],
        *,
        output_dir: str | None,
        force: bool,
        selection_output_ratio: float | None,
        selection_output_quota_mode: str | None,
        reliability_metrics: tuple[str, ...],
        prototype_gain_metrics: tuple[str, ...],
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        received["selection_output_ratio"] = selection_output_ratio
        received["selection_output_quota_mode"] = selection_output_quota_mode
        received["reliability_metrics"] = reliability_metrics
        received["prototype_gain_metrics"] = prototype_gain_metrics
        return Path("outputs/relcore/test/select-r15-g7")

    monkeypatch.setattr(cli, "select_stage", fake_selection_stage)

    cli.main(["select", "--config", "unused.yaml"])

    assert received["config"] == config
    assert received["selection_output_ratio"] is None
    assert received["selection_output_quota_mode"] is None
    assert received["reliability_metrics"] == (
        "support",
        "progress",
        "smoothness",
        "non_noop",
    )
    assert received["prototype_gain_metrics"] == (
        "transition",
        "cooccurrence",
        "sequence",
    )
    assert capsys.readouterr().out.strip() == "relcore_output=outputs/relcore/test/select-r15-g7"


def test_main_without_selection_ratio_preserves_configured_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "runtime": {"max_episodes": None},
        "selection": {"ratio": 0.10, "budget": 100},
    }
    received: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def fake_run_pipeline(
        resolved: dict[str, object],
        *,
        output_dir: str | None,
        force: bool,
        selection_output_ratio: float | None,
        selection_output_quota_mode: str | None,
        reliability_metrics: tuple[str, ...],
        prototype_gain_metrics: tuple[str, ...],
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        received["selection_output_ratio"] = selection_output_ratio
        received["selection_output_quota_mode"] = selection_output_quota_mode
        received["reliability_metrics"] = reliability_metrics
        received["prototype_gain_metrics"] = prototype_gain_metrics
        return Path("outputs/relcore/test/select-r15-g7")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(["run", "--config", "unused.yaml"])

    assert received["config"] == config
    assert received["selection_output_ratio"] is None
    assert received["selection_output_quota_mode"] is None


def test_main_select_none_quota_mode_disables_hard_task_quotas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "runtime": {"max_episodes": None},
        "selection": {
            "ratio": 0.10,
            "budget": None,
            "quota_mode": "proportional",
            "minimum_per_task": 1,
        },
    }
    received: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def fake_selection_stage(
        resolved: dict[str, object],
        *,
        output_dir: str | None,
        force: bool,
        selection_output_ratio: float | None,
        selection_output_quota_mode: str | None,
        reliability_metrics: tuple[str, ...],
        prototype_gain_metrics: tuple[str, ...],
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        received["selection_output_quota_mode"] = selection_output_quota_mode
        return Path("outputs/relcore/test/select-r15-g7-quota-none")

    monkeypatch.setattr(cli, "select_stage", fake_selection_stage)

    cli.main(["select", "--config", "unused.yaml", "--quota-mode", "none"])

    assert received["config"] == {
        "runtime": {"max_episodes": None},
        "selection": {
            "ratio": 0.10,
            "budget": None,
            "quota_mode": "none",
            "minimum_per_task": 0,
        },
    }
    assert received["selection_output_quota_mode"] == "none"


def test_main_run_proportional_quota_mode_preserves_configured_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "runtime": {"max_episodes": None},
        "selection": {
            "ratio": 0.10,
            "budget": None,
            "quota_mode": "none",
            "minimum_per_task": 0,
        },
    }
    received: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def fake_run_pipeline(
        resolved: dict[str, object],
        *,
        output_dir: str | None,
        force: bool,
        selection_output_ratio: float | None,
        selection_output_quota_mode: str | None,
        reliability_metrics: tuple[str, ...],
        prototype_gain_metrics: tuple[str, ...],
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        received["selection_output_quota_mode"] = selection_output_quota_mode
        return Path("outputs/relcore/test/select-r15-g7-quota-proportional")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(["run", "--config", "unused.yaml", "--quota-mode", "proportional"])

    assert received["config"] == {
        "runtime": {"max_episodes": None},
        "selection": {
            "ratio": 0.10,
            "budget": None,
            "quota_mode": "proportional",
            "minimum_per_task": 0,
        },
    }
    assert received["selection_output_quota_mode"] == "proportional"
