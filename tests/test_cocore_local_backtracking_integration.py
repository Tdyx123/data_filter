"""End-to-end contracts for optional local backtracking diagnostics and fusion."""

import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cocore.action_variation import fuse_reliability
from cocore.config import resolve_config
from cocore.pipeline import encode_stage, run_pipeline, validate_output
from test_cocore_pipeline import (
    CocorePipelineAdapter,
    CocoreVisualEncoder,
    FailingCocoreVisualEncoder,
    HfCompatibleIrregularJerkAdapter,
    _config,
    cocore_pipeline,
    register_dataset_adapter,
)


def test_backtracking_requires_explicit_configuration():
    base = {"objective": {"relation": "sequence", "relation_weight": 1}}
    with pytest.raises(ValueError, match="local_backtracking"):
        resolve_config({**base, "reliability_metrics": ["low_local_backtracking"]})
    resolved = resolve_config({**base, "local_backtracking": {"epsilon_p": 0.01}})
    assert resolved["local_backtracking"] == {"epsilon_p": 0.01, "eta": 0.5}
    assert "low_local_backtracking" not in resolved["reliability_metrics"]


def test_backtracking_fusion_ignores_unavailable_component():
    values = np.full(3, 0.25)
    result = fuse_reliability(
        values,
        values,
        values,
        values,
        ["support", "low_local_backtracking"],
        min_reliability=0.05,
        low_local_backtracking=np.array([1, np.nan, 0]),
    )
    np.testing.assert_allclose(result, [0.5, 0.25, 0.05])
    neutral = fuse_reliability(
        values,
        values,
        values,
        values,
        ["low_local_backtracking"],
        min_reliability=0.05,
        low_local_backtracking=np.full(3, np.nan),
    )
    np.testing.assert_array_equal(neutral, np.ones(3))


def _backtracking_config(tmp_path):
    config = _config(tmp_path)
    config["local_backtracking"] = {"epsilon_p": 0.001}
    return config


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
@pytest.mark.parametrize("fused", [False, True])
def test_backtracking_pipeline_roundtrip(tmp_path, profile, fused):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _backtracking_config(tmp_path)
    config["prototypes"]["profile"] = profile
    if fused:
        config["reliability_metrics"] = ["low_local_backtracking"]
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert rows and all(row["local_backtracking_rate"] == 0 for row in rows)
    length = 15 if profile == "libero" else 7
    assert all(row["local_backtracking_valid_count"] == length - 2 for row in rows)
    assert all(row["local_backtracking_count"] == 0 for row in rows)
    if fused:
        assert all(row["reliability"] == 1 for row in rows)
    report = json.loads((result / "selection_report.json").read_text())
    summary = report["local_backtracking_summary"]["graph"]
    assert summary["valid_count"] == len(rows)
    assert summary["valid_comparison_count"] == len(rows) * (length - 2)
    assert summary["rate_mean"] == 0
    assert validate_output(result, config=config)["status"] == "valid"


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_backtracking_reversals_reduce_reliability(tmp_path, profile):
    class ReversingAdapter(CocorePipelineAdapter):
        def iter_episodes(self, **kwargs):
            for episode in super().iter_episodes(**kwargs):
                episode.observations["observation.state"][:, 0] += (
                    0.08 * (np.arange(episode.length) % 2)
                )
                yield episode

    register_dataset_adapter("cocore_pipeline_synthetic", ReversingAdapter)
    config = _backtracking_config(tmp_path)
    config["prototypes"]["profile"] = profile
    config["reliability_metrics"] = ["low_local_backtracking"]
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert rows and all(row["local_backtracking_rate"] == 1 for row in rows)
    assert all(row["low_local_backtracking"] == 0 for row in rows)
    assert all(row["reliability"] == pytest.approx(0.05) for row in rows)
    assert validate_output(result, config=config)["status"] == "valid"


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_backtracking_diagnostics_preserve_selection(tmp_path, profile):
    from cocore.local_backtracking import BACKTRACKING_FIELDS

    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["prototypes"]["profile"] = profile
    baseline = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    before = pq.read_table(baseline / "all_clips.parquet").to_pylist()
    selected_before = (baseline / "selected_manifest.jsonl").read_text().splitlines()
    config["output"]["directory"] = str(tmp_path / "backtracking_diagnostics")
    config["local_backtracking"] = {"epsilon_p": 100}
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    after = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert [
        {k: v for k, v in row.items() if k not in BACKTRACKING_FIELDS} for row in after
    ] == before
    assert all(row["local_backtracking_rate"] is None for row in after)
    assert all(row["local_backtracking_reason"] == "no_valid_comparisons" for row in after)
    selected_after = (result / "selected_manifest.jsonl").read_text().splitlines()
    assert [json.loads(r)["sample_id"] for r in selected_after] == [
        json.loads(r)["sample_id"] for r in selected_before
    ]
    assert validate_output(result, config=config)["status"] == "valid"


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_backtracking_jerk_mapping_and_other_metrics(tmp_path, profile):
    from cocore.local_backtracking import BACKTRACKING_FIELDS

    register_dataset_adapter("backtracking_irregular", HfCompatibleIrregularJerkAdapter)
    config = _backtracking_config(tmp_path)
    config["dataset"]["type"] = "backtracking_irregular"
    config["prototypes"]["profile"] = profile
    config["reliability_metrics"] = [
        "eef_jerk",
        "low_local_backtracking",
        "local_path_efficiency",
        "low_high_frequency_jitter",
        "action_jump",
    ]
    config["local_path_efficiency"] = {"delta_path": 100}
    config["high_frequency_jitter"] = dict(
        cutoff_hz=2,
        noise_floor_rms=1e-5,
        max_frequency_resolution_hz=2,
    )
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    root = result.parent
    indices = np.load(root / cocore_pipeline.GRAPH_DIRECTORY / "source_clip_indices.npy")
    with np.load(root / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz") as nodes:
        assert nodes["local_backtracking_valid"].all()
        for field in BACKTRACKING_FIELDS:
            np.testing.assert_array_equal(
                nodes[field], np.load(root / "encode" / f"{field}.npy")[indices]
            )
        np.testing.assert_allclose(
            nodes["reliability"],
            np.clip(
                (
                    nodes["eef_jerk"]
                    * nodes["low_local_backtracking"]
                    * nodes["low_high_frequency_jitter"]
                    * nodes["action_jump"]
                )
                ** 0.25,
                0.05,
                1,
            ),
            rtol=1e-6,
        )
    report = json.loads((result / "selection_report.json").read_text())
    assert report["excluded_jerk_clips"] > 0
    assert report["local_backtracking_summary"]["scanned"]["invalid_count"] > 0
    assert validate_output(result, config=config)["status"] == "valid"


@pytest.mark.parametrize("parameter", ["epsilon_p", "eta"])
def test_backtracking_cache_reuse_and_parameters(tmp_path, parameter):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _backtracking_config(tmp_path)
    config["local_backtracking"]["eta"] = 0.5
    _, _, encoded = encode_stage(config, visual_encoder=CocoreVisualEncoder())
    resumed = encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())[2]
    assert resumed.fingerprint == encoded.fingerprint
    changed = copy.deepcopy(config)
    changed["local_backtracking"][parameter] *= 0.9
    with pytest.raises(FileExistsError):
        encode_stage(changed, visual_encoder=FailingCocoreVisualEncoder())


@pytest.mark.parametrize(
    "target",
    [
        "input",
        "overlap",
        "timestamp",
        "result",
        "count",
        "count_dtype",
        "validity",
        "reason",
        "node",
        "row",
        "selected",
        "summary",
        "contract",
        "missing",
    ],
)
def test_backtracking_tamper_detection(tmp_path, target):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _backtracking_config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    encode = result.parent / "encode"
    fields = {
        "input": "local_backtracking_positions",
        "overlap": "local_backtracking_positions",
        "timestamp": "local_backtracking_timestamps",
        "result": "local_backtracking_rate",
        "count": "local_backtracking_valid_count",
        "count_dtype": "local_backtracking_count",
        "validity": "local_backtracking_valid",
        "reason": "local_backtracking_reason",
    }
    if target in fields:
        field = fields[target]
        path = encode / f"{field}.npy"
        values = np.load(path)
        if target in {"input", "overlap"}:
            values[0, -1, 0] += 0.01
        elif target == "timestamp":
            values[0, -1] += 0.01
        elif target == "count_dtype":
            values = values.astype(np.float64)
        elif target == "validity":
            values[0] = not values[0]
        elif target == "reason":
            values[0] = "fake"
        else:
            values[0] += 1
        np.save(path, values)
        if target != "input":
            path_manifest = encode / "manifest.json"
            manifest = json.loads(path_manifest.read_text())
            manifest["local_backtracking_checksums"][field] = cocore_pipeline.file_sha256(path)
            path_manifest.write_text(json.dumps(manifest))
    elif target == "missing":
        (encode / "local_backtracking_count.npy").unlink()
    elif target == "node":
        path = result.parent / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz"
        with np.load(path) as nodes:
            arrays = dict(nodes)
        arrays["local_backtracking_count"][0] += 1
        np.savez(path, **arrays)
    elif target == "row":
        path = result / "all_clips.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["local_backtracking_valid_count"] += 1
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif target == "selected":
        path = result / "selected_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["local_backtracking_reason"] = "fake"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        path = result / ("selection_report.json" if target == "summary" else "run_manifest.json")
        data = json.loads(path.read_text())
        if target == "summary":
            data["local_backtracking_summary"]["scanned"]["valid_count"] += 1
        else:
            data["local_backtracking"]["eta"] = 0.9
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="local_backtracking"):
        validate_output(result, config=config)


@pytest.mark.parametrize("problem", ["missing", "nonfinite", "shape"])
def test_backtracking_raw_input_error_identifies_clip(tmp_path, problem):
    class BadPositionAdapter(CocorePipelineAdapter):
        def iter_episodes(self, **kwargs):
            for episode in super().iter_episodes(**kwargs):
                if problem == "missing":
                    del episode.observations["observation.state"]
                elif problem == "nonfinite":
                    episode.observations["observation.state"][0, 0] = np.nan
                else:
                    episode.observations["observation.state"] = episode.observations[
                        "observation.state"
                    ][:, :2]
                yield episode

    register_dataset_adapter("cocore_pipeline_synthetic", BadPositionAdapter)
    with pytest.raises(ValueError, match="local_backtracking clip ep000000_fragment_"):
        encode_stage(_backtracking_config(tmp_path), visual_encoder=CocoreVisualEncoder())


@pytest.mark.parametrize("invalid_value", ["NaN", float("nan")])
def test_backtracking_selected_invalid_score_must_be_json_null(tmp_path, invalid_value):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _backtracking_config(tmp_path)
    config["local_backtracking"]["epsilon_p"] = 100
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result / "selected_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["local_backtracking_rate"] is None
    rows[0]["local_backtracking_rate"] = invalid_value
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="local_backtracking"):
        validate_output(result, config=config)
