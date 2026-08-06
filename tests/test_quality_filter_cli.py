from __future__ import annotations

from pathlib import Path

from quality_filter import cli
from quality_filter.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_libero90_config_uses_local_clip_and_quality_only_defaults() -> None:
    config = load_config(PROJECT_ROOT / "quality_filter" / "config_libero90.yaml")

    assert config["dataset"]["name"] == "libero90"
    assert config["clip"] == {"length": 15, "stride": 15}
    assert config["encoder"]["local_files_only"] is True
    assert config["encoder"]["visual_dim"] == 128
    assert config["filter"] == {"percent": 10.0, "seed": None}
    assert config["output"]["directory"] == "outputs/quality_filter/libero90"


def test_cli_exposes_stages_and_forwards_run_overrides(monkeypatch, capsys) -> None:
    config = {
        "runtime": {"max_episodes": None},
        "filter": {"percent": 10.0, "seed": None},
    }
    received: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def fake_run_pipeline(
        value,
        *,
        output_dir,
        percent,
        seed,
        force,
    ) -> Path:
        received.update(
            config=value,
            output_dir=output_dir,
            percent=percent,
            seed=seed,
            force=force,
        )
        return Path("/tmp/quality-filter")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(
        [
            "run",
            "--config",
            "custom.yaml",
            "--output-dir",
            "/tmp/quality-filter",
            "--max-episodes",
            "3",
            "--percent",
            "20",
            "--seed",
            "77",
            "--force",
        ]
    )

    assert received == {
        "config": {
            "runtime": {"max_episodes": 3},
            "filter": {"percent": 10.0, "seed": None},
        },
        "output_dir": "/tmp/quality-filter",
        "percent": 20.0,
        "seed": 77,
        "force": True,
    }
    assert capsys.readouterr().out.strip() == "quality_filter_output=/tmp/quality-filter"


def test_cli_rejects_invalid_percent_and_nonpositive_episode_limit() -> None:
    parser = cli.build_parser()
    for arguments in (
        ["filter", "--percent", "0"],
        ["run", "--percent", "101"],
        ["quality", "--max-episodes", "0"],
    ):
        try:
            parser.parse_args(arguments)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError(f"arguments should be rejected: {arguments}")
