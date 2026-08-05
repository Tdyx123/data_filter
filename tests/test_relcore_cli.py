from __future__ import annotations

import copy
from pathlib import Path

import pytest

from relcore import cli


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
    ("command", "stage_name"),
    [("select", "select_stage"), ("run", "run_pipeline")],
)
def test_main_selection_ratio_overrides_configured_ratio_and_budget(
    command: str,
    stage_name: str,
    monkeypatch: pytest.MonkeyPatch,
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
    ) -> Path:
        received["config"] = copy.deepcopy(resolved)
        received["output_dir"] = output_dir
        received["force"] = force
        return Path("outputs/relcore/test")

    monkeypatch.setattr(cli, stage_name, fake_selection_stage)

    cli.main(
        [
            command,
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
