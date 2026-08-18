from __future__ import annotations

from pathlib import Path

import pytest

import cocore
from cocore import cli
from cocore.config import load_config, resolve_config
from cocore.pipeline import selection_directory_name


def _objective(relation: str = "cooccurrence", weight: float = 1.0) -> dict[str, object]:
    return {"objective": {"relation": relation, "relation_weight": weight}}


def test_package_version_matches_optional_stop_release() -> None:
    assert cocore.__version__ == "0.14.1"


def test_config_requires_explicit_relation_and_weight() -> None:
    with pytest.raises(ValueError, match="objective.relation is required"):
        resolve_config({})

    with pytest.raises(ValueError, match="objective.relation_weight is required"):
        resolve_config({"objective": {"relation": "cooccurrence"}})


def test_config_rejects_clip_overrides_with_uniform_window_contract() -> None:
    with pytest.raises(ValueError, match="near-uniform 15-frame"):
        resolve_config({**_objective(), "clip": {"length": 15, "stride": 15}})


@pytest.mark.parametrize("relation", ["sequence", "cooccurrence"])
def test_config_accepts_supported_relations(relation: str) -> None:
    resolved = resolve_config(_objective(relation))

    assert resolved["encoding"] == {
        "visual_dim": 128,
        "pca_fit_max_samples": None,
        "quantile_low": 0.01,
        "quantile_high": 0.99,
        "epsilon": 1.0e-8,
    }
    assert resolved["prototypes"]["method"] == "motion_primitives"
    assert resolved["prototypes"]["tol"] == 1.0e-4
    assert resolved["prototypes"]["num_threads"] == 4
    assert resolved["prototypes"]["use_stop_bucket"] is True
    assert resolved["reliability_metrics"] == ["support", "progress"]
    assert resolved["objective"] == {"relation": relation, "relation_weight": 1.0}
    assert resolved["selection"]["max_refreshes"] == 100


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_config_rejects_invalid_relation_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="objective.relation_weight"):
        resolve_config(_objective(weight=weight))


def test_config_rejects_unknown_relation() -> None:
    with pytest.raises(ValueError, match="objective.relation must be sequence or cooccurrence"):
        resolve_config(_objective("transition"))


def test_config_rejects_removed_cooccurrence_weight_with_migration_message() -> None:
    with pytest.raises(ValueError, match="use objective.relation_weight"):
        resolve_config(
            {
                "objective": {
                    "relation": "cooccurrence",
                    "relation_weight": 1.0,
                    "cooccurrence_weight": 1.0,
                }
            }
        )


def test_config_rejects_clip_section_because_cocore_uses_fixed_windows() -> None:
    with pytest.raises(ValueError, match="clip.*fixed.*15"):
        resolve_config(
            {
                **_objective(),
                "clip": {"length": 15, "stride": 15},
            }
        )


@pytest.mark.parametrize("obsolete", ["relation", "normalization"])
def test_config_rejects_removed_encoding_sections_with_migration_message(
    obsolete: str,
) -> None:
    with pytest.raises(ValueError, match=f"{obsolete}.*encoding"):
        resolve_config(
            {
                **_objective(),
                obsolete: {"epsilon": 1.0e-6},
            }
        )


@pytest.mark.parametrize(
    "encoding",
    [
        {"visual_dim": 64},
        {"visual_dim": 128.5},
        {"visual_dim": "128"},
        {"pca_fit_max_samples": 0},
        {"pca_fit_max_samples": True},
        {"quantile_low": -0.1},
        {"quantile_low": 0.5, "quantile_high": 0.5},
        {"quantile_high": 1.1},
        {"epsilon": 0.0},
        {"epsilon": True},
        {"epsilon": float("nan")},
    ],
)
def test_config_rejects_invalid_quality_style_encoding(encoding: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="encoding"):
        resolve_config({**_objective(), "encoding": encoding})


def test_config_rejects_non_mapping_encoding() -> None:
    with pytest.raises(ValueError, match="encoding.*mapping"):
        resolve_config({**_objective(), "encoding": 128})


@pytest.mark.parametrize("obsolete", ["count", "top_r", "temperature"])
def test_config_rejects_flat_prototype_controls_replaced_by_action_formulas(
    obsolete: str,
) -> None:
    with pytest.raises(ValueError, match="fixed by the schema-5 algorithm"):
        resolve_config(
            {
                **_objective(),
                "prototypes": {"method": "motion_primitives", obsolete: 3},
            }
        )


def test_config_rejects_unknown_prototype_controls() -> None:
    with pytest.raises(ValueError, match="prototypes.*unsupported_field"):
        resolve_config(
            {
                **_objective(),
                "prototypes": {
                    "method": "motion_primitives",
                    "unsupported_field": 3,
                },
            }
        )


def test_config_accepts_disabled_stop_bucket() -> None:
    resolved = resolve_config(
        {
            **_objective(),
            "prototypes": {"use_stop_bucket": False},
        }
    )

    assert resolved["prototypes"]["use_stop_bucket"] is False


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_config_rejects_non_boolean_stop_bucket_control(value: object) -> None:
    with pytest.raises(ValueError, match="prototypes.use_stop_bucket"):
        resolve_config(
            {
                **_objective(),
                "prototypes": {"use_stop_bucket": value},
            }
        )


@pytest.mark.parametrize("num_threads", [1, 4, 8])
def test_config_accepts_positive_prototype_thread_counts(num_threads: int) -> None:
    resolved = resolve_config(
        {
            **_objective(),
            "prototypes": {"num_threads": num_threads},
        }
    )

    assert resolved["prototypes"]["num_threads"] == num_threads


@pytest.mark.parametrize("num_threads", [0, -1, 1.5, True, "4"])
def test_config_rejects_non_positive_or_non_integer_prototype_thread_counts(
    num_threads: object,
) -> None:
    with pytest.raises(ValueError, match="prototypes.num_threads"):
        resolve_config(
            {
                **_objective(),
                "prototypes": {"num_threads": num_threads},
            }
        )


@pytest.mark.parametrize("tol", [0.0, -1.0e-4, float("nan"), float("inf"), True, "1e-4"])
def test_config_rejects_invalid_prototype_convergence_tolerance(tol: object) -> None:
    with pytest.raises(ValueError, match="prototypes.tol"):
        resolve_config(
            {
                **_objective(),
                "prototypes": {"tol": tol},
            }
        )


@pytest.mark.parametrize("max_refreshes", [0, -1, 1.5, True])
def test_config_rejects_non_positive_or_non_integer_max_refreshes(max_refreshes) -> None:
    with pytest.raises(ValueError, match="selection.max_refreshes"):
        resolve_config({**_objective(), "selection": {"max_refreshes": max_refreshes}})


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
        resolve_config({**_objective(), "selection": {obsolete: 1}})


def test_selection_directory_encodes_relation_weight_and_ratio() -> None:
    assert selection_directory_name("sequence", 1.5, 0.1) == "select-sequence-w1p5-top10pct"
    assert (
        selection_directory_name("cooccurrence", 0.0, 0.125) == "select-cooccurrence-w0-top12p5pct"
    )


def test_run_cli_accepts_relation_weight_and_ratio_but_rejects_old_weight() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "run",
            "--relation",
            "sequence",
            "--relation-weight",
            "1.5",
            "--selection-ratio",
            "0.2",
        ]
    )

    assert arguments.relation == "sequence"
    assert arguments.relation_weight == 1.5
    assert arguments.selection_ratio == 0.2
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--cooccurrence-weight", "1.5"])


@pytest.mark.parametrize("command", ["build-graph", "select", "run", "validate"])
def test_graph_commands_accept_stop_bucket_disable_flag(command: str) -> None:
    arguments = [command, "--no-use-stop-bucket"]
    if command == "validate":
        arguments += ["--output-dir", "result"]

    parsed = cli.build_parser().parse_args(arguments)

    assert parsed.no_use_stop_bucket is True


@pytest.mark.parametrize("command", ["scan", "encode"])
def test_non_graph_commands_reject_stop_bucket_disable_flag(command: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "--no-use-stop-bucket"])


def test_main_preserves_disabled_stop_bucket_without_cli_override(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}
    configured = {
        **_objective(),
        "prototypes": {"use_stop_bucket": False},
    }
    monkeypatch.setattr(cli, "load_config", lambda _: configured)

    def fake_run_pipeline(config, **kwargs):
        received["config"] = config
        return Path("outputs/cocore/test/select-cooccurrence-w1-top10pct")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(["run", "--config", "unused.yaml"])

    assert received["config"]["prototypes"]["use_stop_bucket"] is False
    capsys.readouterr()


def test_main_stop_bucket_disable_flag_overrides_enabled_yaml(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}
    configured = {
        **_objective(),
        "prototypes": {"use_stop_bucket": True},
    }
    monkeypatch.setattr(cli, "load_config", lambda _: configured)

    def fake_run_pipeline(config, **kwargs):
        received["config"] = config
        return Path("outputs/cocore/test/select-cooccurrence-w1-top10pct")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(["run", "--config", "unused.yaml", "--no-use-stop-bucket"])

    assert received["config"]["prototypes"]["use_stop_bucket"] is False
    capsys.readouterr()


@pytest.mark.parametrize("explicit_config", [False, True])
def test_validate_stop_bucket_disable_flag_overrides_replay_config(
    explicit_config: bool,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output = tmp_path / "select-cooccurrence-w1-top10pct"
    loaded_paths: list[object] = []
    received: dict[str, object] = {}

    def fake_load_config(path):
        loaded_paths.append(path)
        return {
            **_objective(),
            "prototypes": {"use_stop_bucket": True},
        }

    def fake_validate_output(output_dir, *, config):
        received["output_dir"] = output_dir
        received["config"] = config
        return {"status": "valid", "selected_clips": 1}

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "validate_output", fake_validate_output)
    arguments = [
        "validate",
        "--output-dir",
        str(output),
        "--no-use-stop-bucket",
    ]
    if explicit_config:
        arguments += ["--config", "custom.yaml"]

    cli.main(arguments)

    assert loaded_paths == [
        "custom.yaml" if explicit_config else output / "resolved_config.yaml"
    ]
    assert received["output_dir"] == str(output)
    assert received["config"]["prototypes"]["use_stop_bucket"] is False
    assert capsys.readouterr().out.strip() == '{"selected_clips": 1, "status": "valid"}'


def test_main_applies_cli_overrides_to_run_pipeline(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_config", lambda _: _objective())

    def fake_run_pipeline(config, **kwargs):
        received["config"] = config
        received.update(kwargs)
        return Path("outputs/cocore/test/select-sequence-w2-top25pct")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(
        [
            "run",
            "--config",
            "unused.yaml",
            "--relation",
            "sequence",
            "--relation-weight",
            "2",
            "--selection-ratio",
            "0.25",
        ]
    )

    assert received["config"]["objective"] == {
        "relation": "sequence",
        "relation_weight": 2.0,
    }
    assert received["config"]["selection"]["ratio"] == 0.25
    assert received["config"]["selection"]["budget"] is None
    assert capsys.readouterr().out.strip().endswith("select-sequence-w2-top25pct")


def test_build_graph_cli_reports_schema_seven_graph_directory(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_config", lambda _: _objective())
    monkeypatch.setattr(
        cli,
        "graph_stage",
        lambda config, **kwargs: (
            Path("outputs/cocore/test"),
            None,
            None,
            type("Graph", (), {"sample_ids": (1, 2)})(),
            "graph-fingerprint",
        ),
    )

    cli.main(["build-graph", "--config", "unused.yaml"])

    assert capsys.readouterr().out.strip() == (
        "cocore_output=outputs/cocore/test/graph-17-motion-hard-nearest-pca nodes=2"
    )


@pytest.mark.parametrize("path", ["cocore/config_libero90.yaml", "cocore/config_debug.yaml"])
def test_shipped_configs_resolve_to_fixed_cocore_contract(path: str) -> None:
    config = load_config(path)

    assert "clip" not in config
    assert "relation" not in config
    assert "normalization" not in config
    assert config["encoding"] == {
        "visual_dim": 128,
        "pca_fit_max_samples": None,
        "quantile_low": 0.01,
        "quantile_high": 0.99,
        "epsilon": 1.0e-8,
    }
    assert config["prototypes"]["method"] == "motion_primitives"
    assert set(config["prototypes"]) == {
        "method",
        "batch_size",
        "max_iter",
        "tol",
        "num_threads",
        "use_stop_bucket",
    }
    assert config["prototypes"]["num_threads"] == (
        1 if path.endswith("config_debug.yaml") else 4
    )
    assert config["reliability_metrics"] == ["support", "progress"]
    assert config["objective"] == {
        "relation": "cooccurrence",
        "relation_weight": 1.0,
    }
    assert config["selection"]["max_refreshes"] == 100
    assert (
        not {
            "global_candidates",
            "prototype_candidates",
            "similarity_candidates",
            "random_candidates",
        }
        & config["selection"].keys()
    )
    assert config["output"]["directory"].startswith("outputs/cocore/")
    if path.endswith("config_debug.yaml"):
        assert config["runtime"]["max_episodes"] == 20
