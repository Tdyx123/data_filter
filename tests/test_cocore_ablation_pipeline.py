from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from cocore.pipeline import run_pipeline as run_cocore_pipeline
from cocore_ablation.pipeline import run_pipeline, select_stage, validate_output
from tests.test_cocore_pipeline import (
    CocorePipelineAdapter,
    CocoreVisualEncoder,
    _config as cocore_test_config,
)
from trajectory_data import register_dataset_adapter


def _config(tmp_path: Path) -> dict[str, object]:
    config = copy.deepcopy(cocore_test_config(tmp_path, relation="sequence"))
    config["upstream"] = {"directory": str(tmp_path / "upstream-cocore")}
    config["reliability_metrics"] = ["support", "progress"]
    config["prototypes"].update(  # type: ignore[union-attr]
        {
            "profile": "libero",
            "representation": "action_visual",
            "use_assignment_confidence": True,
            "use_stop_bucket": True,
        }
    )
    config["objective"].update({"redundancy_weight": 1.0})  # type: ignore[union-attr]
    config["selection"].update(  # type: ignore[union-attr]
        {"strategy": "random_multibranch", "use_coverage_seed": True}
    )
    config["output"] = {"directory": str(tmp_path / "ablation-output")}
    return config


def _run(config: dict[str, object], *, subfolder_name: str = "experiment", **kwargs):
    return run_pipeline(config, subfolder_name=subfolder_name, **kwargs)


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "/absolute", "nested/name", r"nested\name"],
)
def test_run_pipeline_rejects_invalid_subfolder_names(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError, match="subfolder_name"):
        run_pipeline(_config(tmp_path), subfolder_name=value)


def test_run_pipeline_reuses_upstream_encode_and_publishes_named_sibling_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)

    result = run_pipeline(
        config,
        subfolder_name="full-model",
        visual_encoder=CocoreVisualEncoder(),
    )
    timing_lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "step=ablation.graph" in line
    ]
    assert len(timing_lines) == 1
    assert float(timing_lines[0].split(" seconds=", 1)[1].split(" ", 1)[0]) > 0.0

    upstream = tmp_path / "upstream-cocore"
    assert (upstream / "scan" / "manifest.json").is_file()
    assert (upstream / "encode" / "manifest.json").is_file()
    assert not (upstream / "graph-18-motion-hard-nearest-pca").exists()
    experiment_root = tmp_path / "ablation-output" / "full-model"
    graph_root = experiment_root / "graph"
    assert result == experiment_root / "select"
    assert graph_root.parent == result.parent
    assert (graph_root / "nodes.npz").is_file()
    assert (result / "selected_manifest.jsonl").is_file()
    assert (result / "resolved_config.yaml").is_file()
    stored = yaml.safe_load((result / "resolved_config.yaml").read_text())
    assert stored["output"]["directory"] == str(experiment_root)

    report = json.loads((result / "selection_report.json").read_text())
    assert report["producer"] == "cocore_ablation"
    assert report["schema_version"] == 1
    assert report["reliability_metrics"] == ["support", "progress"]
    assert report["prototype_representation"] == "action_visual"
    assert report["use_assignment_confidence"] is True
    assert report["initial_set_size"] > 0
    assert report["objective"]["total"] == pytest.approx(
        report["objective"]["weighted_relation"]
        - report["objective"]["weighted_redundancy"]
    )
    assert validate_output(result)["status"] == "valid"


def test_different_subfolders_isolate_graph_and_select_but_reuse_upstream_encode(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    first = _run(
        config,
        subfolder_name="full-model",
        visual_encoder=CocoreVisualEncoder(),
    )
    upstream_manifest = tmp_path / "upstream-cocore" / "encode" / "manifest.json"
    upstream_mtime = upstream_manifest.stat().st_mtime_ns

    second_config = copy.deepcopy(config)
    second_config["objective"]["relation_weight"] = 0.0  # type: ignore[index]
    second = _run(second_config, subfolder_name="no-relation")

    assert first == tmp_path / "ablation-output" / "full-model" / "select"
    assert second == tmp_path / "ablation-output" / "no-relation" / "select"
    assert (first.parent / "graph" / "manifest.json").is_file()
    assert (second.parent / "graph" / "manifest.json").is_file()
    assert upstream_manifest.stat().st_mtime_ns == upstream_mtime


def test_graph_change_reuses_upstream_but_builds_new_graph_and_selection(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    first = _run(
        config,
        subfolder_name="full-model",
        visual_encoder=CocoreVisualEncoder(),
    )
    upstream_manifest = tmp_path / "upstream-cocore" / "encode" / "manifest.json"
    upstream_mtime = upstream_manifest.stat().st_mtime_ns

    changed = copy.deepcopy(config)
    changed["prototypes"]["representation"] = "action_only"  # type: ignore[index]
    second = _run(changed, subfolder_name="action-only")

    assert second.parent != first.parent
    assert second != first
    assert upstream_manifest.stat().st_mtime_ns == upstream_mtime


def test_force_rebuilds_named_graph_and_selection_without_rebuilding_upstream(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    first = _run(config, visual_encoder=CocoreVisualEncoder())
    graph_manifest = first.parent / "graph" / "manifest.json"
    select_manifest = first / "manifest.json"
    upstream_manifest = tmp_path / "upstream-cocore" / "encode" / "manifest.json"
    graph_mtime = graph_manifest.stat().st_mtime_ns
    select_mtime = select_manifest.stat().st_mtime_ns
    upstream_mtime = upstream_manifest.stat().st_mtime_ns
    rebuilt = _run(config, force=True)

    assert rebuilt == first
    assert graph_manifest.stat().st_mtime_ns > graph_mtime
    assert select_manifest.stat().st_mtime_ns > select_mtime
    assert upstream_manifest.stat().st_mtime_ns == upstream_mtime


def test_same_subfolder_rejects_incompatible_selection_without_force(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    _run(config, visual_encoder=CocoreVisualEncoder())
    changed = copy.deepcopy(config)
    changed["objective"]["relation_weight"] = 0.0  # type: ignore[index]

    with pytest.raises(FileExistsError, match="select"):
        _run(changed)


def test_random_without_coverage_is_strict_seeded_random_baseline(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["selection"].update(  # type: ignore[union-attr]
        {"strategy": "random", "use_coverage_seed": False}
    )

    result = _run(config, visual_encoder=CocoreVisualEncoder())

    report = json.loads((result / "selection_report.json").read_text())
    rows = [json.loads(line) for line in (result / "selected_manifest.jsonl").read_text().splitlines()]
    assert report["initial_set_size"] == 0
    assert report["selection_strategy"] == "random"
    assert {row["selection_phase"] for row in rows} == {"random"}
    assert validate_output(result)["status"] == "valid"


def test_graph_variants_change_reliability_and_action_catalog(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["reliability_metrics"] = []
    config["prototypes"].update(  # type: ignore[union-attr]
        {"representation": "action_only", "use_assignment_confidence": False}
    )

    result = _run(config, visual_encoder=CocoreVisualEncoder())

    graph_root = result.parent / "graph"
    nodes = np.load(graph_root / "nodes.npz")
    np.testing.assert_array_equal(nodes["reliability"], np.ones(len(nodes["reliability"])))
    catalog = json.loads((graph_root / "prototype_catalog.json").read_text())
    assert catalog["schema_version"] == 1
    assert catalog["representation"] == "action_only"
    assert catalog["use_assignment_confidence"] is False
    assert all(category["actual_centers"] in {0, 1} for category in catalog["action_categories"])


def test_validator_rejects_tampered_selection_order(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    result = _run(_config(tmp_path), visual_encoder=CocoreVisualEncoder())
    path = result / "selected_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["sample_id"], rows[1]["sample_id"] = rows[1]["sample_id"], rows[0]["sample_id"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="selection order"):
        validate_output(result)


def test_validator_rejects_tampered_run_fingerprint(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    result = _run(_config(tmp_path), visual_encoder=CocoreVisualEncoder())
    path = result / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["fingerprint"] = "0" * 64
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact fingerprint"):
        validate_output(result)


def test_select_stage_writes_a_complete_validatable_artifact(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)

    result = select_stage(config, visual_encoder=CocoreVisualEncoder())

    assert (result / "resolved_config.yaml").is_file()
    assert validate_output(result)["status"] == "valid"


def test_validator_rejects_tampered_replay_metadata(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    result = _run(_config(tmp_path), visual_encoder=CocoreVisualEncoder())

    selected_path = result / "selected_manifest.jsonl"
    original_selected = selected_path.read_text()
    selected = [json.loads(line) for line in original_selected.splitlines()]
    selected[0]["selection_score_delta"] = 999999.0
    selected[0]["selection_phase"] = "tampered"
    selected_path.write_text("".join(json.dumps(row) + "\n" for row in selected))
    with pytest.raises(ValueError, match="selected manifest"):
        validate_output(result)
    selected_path.write_text(original_selected)

    report_path = result / "selection_report.json"
    original_report = report_path.read_text()
    report = json.loads(original_report)
    report["coverage"]["achieved"] = [0.0] * len(report["coverage"]["achieved"])
    report["branch_search"]["rounds"] = 999999
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="selection report"):
        validate_output(result)
    report_path.write_text(original_report)

    all_path = result / "all_clips.parquet"
    all_rows = pq.read_table(all_path).to_pylist()
    all_rows[0]["selected"] = not all_rows[0]["selected"]
    pq.write_table(pa.Table.from_pylist(all_rows), all_path)
    with pytest.raises(ValueError, match="all-clips"):
        validate_output(result)


def test_validator_rejects_tampered_prototype_artifacts(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    result = _run(_config(tmp_path), visual_encoder=CocoreVisualEncoder())
    graph_root = result.parent / "graph"

    centers_path = graph_root / "prototype_centers.npy"
    original_centers = np.load(centers_path, allow_pickle=False)
    np.save(centers_path, np.full_like(original_centers, 123.0))
    with pytest.raises(ValueError, match="prototype centers"):
        validate_output(result)
    np.save(centers_path, original_centers)

    catalog_path = graph_root / "prototype_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["representation"] = "tampered"
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="prototype catalog"):
        validate_output(result)


def test_validator_accepts_absolute_path_for_relative_configured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)
    config["output"] = {"directory": "ablation-output"}

    result = _run(config, visual_encoder=CocoreVisualEncoder())

    assert validate_output(result.resolve())["status"] == "valid"


def test_validator_accepts_original_base_output_config_for_named_result(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = _run(config, visual_encoder=CocoreVisualEncoder())

    assert validate_output(result, config=config)["status"] == "valid"


def test_run_pipeline_persists_the_canonicalized_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)
    config["output"] = {"directory": "./ablation-output"}

    result = _run(config, visual_encoder=CocoreVisualEncoder())

    assert validate_output(result)["status"] == "valid"


def test_full_ablation_defaults_match_production_cocore_exactly(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    production_config = cocore_test_config(tmp_path, relation="sequence")
    production_config["reliability_metrics"] = ["support", "progress"]
    production_root = tmp_path / "production-cocore"
    production = run_cocore_pipeline(
        production_config,
        output_dir=production_root,
        visual_encoder=CocoreVisualEncoder(),
    )
    ablation_config = _config(tmp_path)
    ablation_config["upstream"] = {"directory": str(production_root)}

    ablation = _run(ablation_config)

    production_graph = production_root / "graph-18-motion-hard-nearest-pca"
    production_nodes = np.load(production_graph / "nodes.npz")
    ablation_graph = ablation.parent / "graph"
    ablation_nodes = np.load(ablation_graph / "nodes.npz")
    for name in (
        "task_indices",
        "prototype_indices",
        "prototype_weights",
        "support",
        "progress",
    ):
        np.testing.assert_array_equal(ablation_nodes[name], production_nodes[name])
    np.testing.assert_allclose(
        ablation_nodes["reliability"],
        production_nodes["reliability"],
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    production_catalog = json.loads(
        (production_graph / "prototype_catalog.json").read_text()
    )
    ablation_catalog = json.loads(
        (ablation_graph / "prototype_catalog.json").read_text()
    )
    for name in (
        "method",
        "profile",
        "use_stop_bucket",
        "total_raw_actions",
        "action_categories",
        "leaf_prototypes",
    ):
        assert ablation_catalog[name] == production_catalog[name]
    for name, value in production_catalog["constants"].items():
        assert ablation_catalog["constants"][name] == value
    production_ids = [
        json.loads(line)["sample_id"]
        for line in (production / "selected_manifest.jsonl").read_text().splitlines()
    ]
    ablation_ids = [
        json.loads(line)["sample_id"]
        for line in (ablation / "selected_manifest.jsonl").read_text().splitlines()
    ]
    assert ablation_ids == production_ids
