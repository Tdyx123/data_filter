from __future__ import annotations

import copy
from pathlib import Path

import pytest

from relcore import cli
from relcore.pipeline import selection_directory_name


@pytest.mark.parametrize("command", ["select", "run"])
def test_selection_commands_accept_selection_ratio(command: str) -> None:
    arguments = cli.build_parser().parse_args(
        [command, "--selection-ratio", "0.20"]
    )

    assert arguments.selection_ratio == pytest.approx(0.20)


@pytest.mark.parametrize("command", ["scan", "encode", "build-graph"])
def test_nonselection_commands_reject_selection_ratio(command: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "--selection-ratio", "0.20"])


@pytest.mark.parametrize("value", ["not-a-number", "0", "-0.10", "1.10"])
def test_selection_ratio_rejects_invalid_values(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--selection-ratio", value])


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.20, "select-top20pct"),
        (0.125, "select-top12p5pct"),
        (1.0, "select-top100pct"),
    ],
)
def test_selection_directory_name_uses_canonical_percent_tag(
    ratio: float,
    expected: str,
) -> None:
    assert selection_directory_name(ratio) == expected


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
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        received["output_dir"] = output_dir
        received["force"] = force
        received["selection_output_ratio"] = selection_output_ratio
        return Path("outputs/relcore/test")

    monkeypatch.setattr(cli, "select_stage", fake_selection_stage)

    cli.main(
        [
            "select",
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
    assert capsys.readouterr().out.strip() == (
        "relcore_output=outputs/relcore/test/select-top25pct"
    )


def test_main_run_ratio_keeps_legacy_output_layout(
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
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        return Path("outputs/relcore/test")

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
    assert capsys.readouterr().out.strip() == "relcore_output=outputs/relcore/test"


def test_main_select_without_ratio_keeps_legacy_output_layout(
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
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        received["selection_output_ratio"] = selection_output_ratio
        return Path("outputs/relcore/test")

    monkeypatch.setattr(cli, "select_stage", fake_selection_stage)

    cli.main(["select", "--config", "unused.yaml"])

    assert received["config"] == config
    assert received["selection_output_ratio"] is None
    assert capsys.readouterr().out.strip() == "relcore_output=outputs/relcore/test"


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
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        return Path("outputs/relcore/test")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(["run", "--config", "unused.yaml"])

    assert received["config"] == config
