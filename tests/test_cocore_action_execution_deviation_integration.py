"""Roundtrip, eligibility and replay contracts for execution deviation."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from cocore.pipeline import encode_stage, run_pipeline, validate_output
from test_cocore_action_execution_deviation import SETTINGS
from test_cocore_pipeline import (
    MixedStopCocorePipelineAdapter,
    CocoreVisualEncoder,
    FailingCocoreVisualEncoder,
    _config,
    cocore_pipeline,
    register_dataset_adapter,
)


PREFIX = "action_execution_deviation"
METRIC = "low_action_execution_deviation"


class ExecutionAdapter(MixedStopCocorePipelineAdapter):
    irregular = False
    all_invalid = False

    def iter_episodes(self, **kwargs):
        for episode in super().iter_episodes(**kwargs):
            steps = np.arange(episode.length, dtype=np.float64)
            episode.observations["observation.state"] = np.zeros((episode.length, 8))
            if episode.episode_id != 2:
                episode.observations["observation.state"][:, 0] = steps * 0.125
            episode.actions = np.zeros((episode.length, 7))
            episode.actions[:, 0] = 0.125 if episode.episode_id == 0 else 0.0625
            if self.irregular and episode.episode_id != 2:
                episode.timestamps[80] = episode.timestamps[79]
                episode.timestamps[200] += 0.01
            if self.all_invalid:
                episode.timestamps[:] = 0
            yield episode


def config_for(tmp_path, *, profile="libero", fused=True, adapter=ExecutionAdapter):
    register_dataset_adapter("execution_synthetic", adapter)
    config = _config(tmp_path, relation="sequence")
    config["dataset"]["type"] = "execution_synthetic"
    config["prototypes"]["profile"] = profile
    config["selection"]["budget"] = 30
    config[PREFIX] = dict(SETTINGS)
    config["reliability_metrics"] = ["support", METRIC] if fused else ["support"]
    return config


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
@pytest.mark.parametrize("fused", [False, True])
def test_roundtrip_and_resume(tmp_path, profile, fused):
    config = config_for(tmp_path, profile=profile, fused=fused)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    for row in rows:
        expected = 0 if row["episode_id"] == 0 else 0.0625
        assert row[f"{PREFIX}_raw"] == expected
        assert row[f"{PREFIX}_valid"] is True
        assert row[f"{PREFIX}_reason"] == ""
        assert row[METRIC] == (1 if expected == 0 else 0)
        if not fused:
            assert row["reliability"] == pytest.approx(max(row["support"], 0.05))
    report = json.loads((result / "selection_report.json").read_text())
    assert report[f"{PREFIX}_summary"]["scanned"]["invalid_count"] == 0
    assert validate_output(result, config=config)["status"] == "valid"
    assert run_pipeline(config, visual_encoder=FailingCocoreVisualEncoder()) == result


class IrregularExecutionAdapter(ExecutionAdapter):
    irregular = True


@pytest.mark.parametrize("with_jerk", [False, True])
def test_intersects_validity_with_prototypes_and_does_not_bridge_gaps(tmp_path, with_jerk):
    config = config_for(tmp_path, adapter=IrregularExecutionAdapter)
    config["selection"] = {"ratio": 0.5, "budget": None}
    config["prototypes"]["use_stop_bucket"] = False
    if with_jerk:
        config["reliability_metrics"].append("eef_jerk")
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    root = result.parent
    graph = root / cocore_pipeline.GRAPH_DIRECTORY
    valid = np.load(root / "encode" / f"{PREFIX}_valid.npy")
    prototype = np.load(graph / "prototype_eligible_mask.npy")
    included = prototype & valid
    if with_jerk:
        jerk_valid = np.load(root / "encode" / "eef_jerk_valid.npy")
        assert np.any(valid & ~jerk_valid)
        included &= jerk_valid
    source = np.load(graph / "source_clip_indices.npy")
    np.testing.assert_array_equal(source, np.flatnonzero(included))
    with np.load(graph / "sequence_edges.npz") as edges:
        assert len(edges["source"]) > 0
        assert np.all(source[edges["target"]] - source[edges["source"]] == 1)
    exclusions = json.loads((result / "excluded_clips.json").read_text())
    assert len(exclusions) == int((~included).sum())
    assert any(f"{PREFIX}:non_increasing_time" in row["reasons"] for row in exclusions)
    if with_jerk:
        assert any(len(row["reasons"]) >= 2 for row in exclusions)
    assert validate_output(result, config=config)["status"] == "valid"
    report = json.loads((result / "selection_report.json").read_text())
    assert report["excluded_clips"] == len(exclusions)
    assert report["excluded_unlabeled_clips"] == int((~prototype).sum())
    assert report["selected_clips"] == int(np.floor(len(source) * 0.5 + 0.5))


class InvalidExecutionAdapter(ExecutionAdapter):
    all_invalid = True


def test_all_invalid_diagnostic_is_null_but_fusion_refuses_empty_pool(tmp_path):
    config = config_for(tmp_path, adapter=InvalidExecutionAdapter, fused=False)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert all(row[f"{PREFIX}_raw"] is None and row[METRIC] is None for row in rows)
    assert not (result / "excluded_clips.json").exists()
    assert validate_output(result, config=config)["status"] == "valid"
    config["reliability_metrics"].append(METRIC)
    with pytest.raises(ValueError, match="computable.*execution_deviation"):
        run_pipeline(config, visual_encoder=CocoreVisualEncoder(), force=True)


@pytest.mark.parametrize(
    "target", ["cache_raw", "cache_overlap", "node", "report", "selected", "excluded"]
)
def test_tampering_is_rejected_even_with_updated_checksum(tmp_path, target):
    config = config_for(tmp_path, adapter=IrregularExecutionAdapter)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    if target.startswith("cache_"):
        field = f"{PREFIX}_raw" if target == "cache_raw" else f"{PREFIX}_actions"
        path = result.parent / "encode" / f"{field}.npy"
        values = np.load(path)
        if target == "cache_overlap":
            values[1, 0, 0] += 1
        else:
            values[0] += 1
        np.save(path, values)
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest[f"{PREFIX}_checksums"][field] = cocore_pipeline.file_sha256(path)
        manifest_path.write_text(json.dumps(manifest))
    elif target == "node":
        path = result.parent / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz"
        with np.load(path) as archive:
            values = dict(archive)
        values[METRIC][0] = 0.4
        np.savez(path, **values)
    elif target == "report":
        path = result / "selection_report.json"
        values = json.loads(path.read_text())
        values[f"{PREFIX}_summary"]["scanned"]["raw_mean"] = 123
        path.write_text(json.dumps(values))
    elif target == "selected":
        path = result / "selected_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0][f"{PREFIX}_valid"] = False
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        path = result / "excluded_clips.json"
        path.write_text("[]")
    with pytest.raises(ValueError, match="execution_deviation|exclusion"):
        validate_output(result, config=config)


def test_scale_change_invalidates_encode_cache(tmp_path):
    config = config_for(tmp_path)
    encode_stage(config, visual_encoder=CocoreVisualEncoder())
    config[PREFIX] = {**SETTINGS, "action_scale": [2, 1, 1]}
    with pytest.raises(FileExistsError, match="incompatible"):
        encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())


def test_normalization_includes_valid_scanned_candidates_excluded_by_stop_filter(tmp_path):
    class StopErrorAdapter(ExecutionAdapter):
        def iter_episodes(self, **kwargs):
            for episode in super().iter_episodes(**kwargs):
                if episode.episode_id == 2:
                    episode.actions[:, 0] = 0.25
                yield episode

    config = config_for(tmp_path, adapter=StopErrorAdapter)
    config["prototypes"]["use_stop_bucket"] = False
    config["encoding"].update(quantile_low=0, quantile_high=1)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert {row["episode_id"] for row in rows} == {0, 1}
    assert all(row[METRIC] == 0.75 for row in rows if row["episode_id"] == 1)
    assert validate_output(result, config=config)["status"] == "valid"


@pytest.mark.parametrize("bad_input", ["missing_state", "short_action", "nonfinite_action"])
def test_invalid_raw_inputs_fail_before_visual_encoding(tmp_path, bad_input):
    class MalformedExecutionAdapter(ExecutionAdapter):
        def iter_episodes(self, **kwargs):
            for episode in super().iter_episodes(**kwargs):
                if bad_input == "missing_state":
                    del episode.observations["observation.state"]
                elif bad_input == "short_action":
                    episode.actions = episode.actions[:, :2]
                else:
                    episode.actions[0, 0] = np.nan
                yield episode

    config = config_for(tmp_path, adapter=MalformedExecutionAdapter)
    with pytest.raises(ValueError, match="action_execution_deviation clip ep"):
        encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())


def test_execution_diagnostics_with_jerk_filtering(tmp_path):
    config = config_for(tmp_path, adapter=IrregularExecutionAdapter, fused=False)
    config["reliability_metrics"].append("eef_jerk")
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    report = json.loads((result / "selection_report.json").read_text())
    assert "excluded_action_execution_deviation_clips" not in report
    assert report[f"{PREFIX}_summary"]["scanned"]["invalid_count"] > 0
    assert validate_output(result, config=config)["status"] == "valid"
