from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from cocore import pipeline as cocore_pipeline
from cocore.pipeline import encode_stage, graph_stage, run_pipeline, validate_output
from relcore.schemas import ClipRecord
from trajectory_data import (
    DatasetAdapter,
    EpisodeData,
    EpisodeRecord,
    register_dataset_adapter,
)


class CocorePipelineAdapter(DatasetAdapter):
    load_images_calls: ClassVar[list[bool]] = []
    motion_sign: ClassVar[float] = 1.0

    def __init__(self, _: Mapping[str, object]) -> None:
        self._records = (
            EpisodeRecord(0, 207, 0, "task zero"),
            EpisodeRecord(1, 207, 1, "task one"),
        )

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ("observation.images.image",)

    def episodes(self) -> Sequence[EpisodeRecord]:
        return self._records

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        del num_workers
        self.load_images_calls.append(load_images)
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            states = np.zeros((record.length, 8), dtype=np.float32)
            states[:, 0] = steps * 0.04 * self.motion_sign
            observations = {"observation.state": states}
            if load_images:
                pixels = np.mod(steps + record.episode_id * 37, 255).astype(np.uint8)
                observations["observation.images.image"] = np.broadcast_to(
                    pixels[:, None, None, None], (record.length, 2, 2, 3)
                ).copy()
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations=observations,
                actions=np.stack([steps / 206.0, np.zeros_like(steps)], axis=1),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "cocore-pipeline-adapter-v1"


class CocoreVisualEncoder:
    output_dim = 3

    def encode(self, images: np.ndarray) -> np.ndarray:
        values = images[:, 0, 0, 0].astype(np.float32)
        return np.stack([values + 1.0, values + 2.0, values + 4.0], axis=1)


class FailingCocoreVisualEncoder:
    output_dim = 3

    def encode(self, images: np.ndarray) -> np.ndarray:
        raise AssertionError(f"visual encoder should not run for {len(images)} cached frames")


class InterruptingCocoreVisualEncoder(CocoreVisualEncoder):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, images: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("injected cocore CLIP interruption")
        return super().encode(images)


class ReverseOrderCocorePipelineAdapter(CocorePipelineAdapter):
    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__(config)
        self._records = tuple(reversed(self._records))


class ShortEpisodeCocorePipelineAdapter(CocorePipelineAdapter):
    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__(config)
        self._records = (*self._records, EpisodeRecord(2, 5, 2, "short task"))


def _config(tmp_path: Path, relation: str = "cooccurrence") -> dict[str, object]:
    return {
        "seed": 7,
        "dataset": {
            "type": "cocore_pipeline_synthetic",
            "name": "synthetic",
            "path": str(tmp_path / "dataset"),
            "use_images": True,
        },
        "visual": {"encoder": "dummy"},
        "encoding": {
            "visual_dim": 128,
            "pca_fit_max_samples": None,
            "quantile_low": 0.01,
            "quantile_high": 0.99,
            "epsilon": 1.0e-8,
        },
        "quality": {"knn": 2},
        "prototypes": {"method": "motion_primitives", "batch_size": 64, "max_iter": 2},
        "graph": {"knn": 2, "similarity_threshold": 0.8, "cooccurrence_max_gap": 4},
        "objective": {"relation": relation, "relation_weight": 1.0},
        "selection": {
            "ratio": 0.5,
            "budget": 6,
            "max_refreshes": 2,
        },
        "runtime": {"num_workers": 0, "max_episodes": None, "resume": True},
        "output": {"directory": str(tmp_path / "cocore-output")},
    }


def test_encode_stage_publishes_normalized_visual_half_artifact(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    CocorePipelineAdapter.load_images_calls.clear()
    root = tmp_path / "cocore-encode"

    result_root, _, encoded = encode_stage(
        _config(tmp_path),
        output_dir=root,
        visual_encoder=CocoreVisualEncoder(),
    )

    assert result_root == root
    assert CocorePipelineAdapter.load_images_calls == [False, False, True]
    embeddings = np.load(root / "encode" / "embeddings.npy")
    assert embeddings.shape == (28, 159)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1.0e-6)
    stored = np.load(root / "encode" / "visual_half_embeddings.npy")
    np.testing.assert_array_equal(stored, encoded.visual_half_embeddings)
    assert stored.shape == (28, 2, 3)
    np.testing.assert_allclose(np.linalg.norm(stored, axis=2), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(
        stored[0, 0],
        np.asarray([4.5, 5.5, 7.5], dtype=np.float32)
        / np.linalg.norm(np.asarray([4.5, 5.5, 7.5], dtype=np.float32)),
        atol=1.0e-7,
    )
    manifest = json.loads((root / "encode" / "manifest.json").read_text())
    index = json.loads((root / "encode" / "frame_embeddings_index.json").read_text())
    assert manifest["producer"] == "cocore"
    assert manifest["encoding"] == "quality_fusion"
    assert manifest["visual_embedding_dim"] == 128
    assert manifest["embedding_dim"] == 159
    assert manifest["counts"] == {
        "candidate_fragments": 28,
        "reference_fragments": 18,
        "overlap_fragments": 6,
        "pca_union_fragments": 40,
        "encoded_episodes": 2,
        "encoded_frames": 414,
    }
    assert manifest["visual_half_embedding_dim"] == 3
    assert manifest["clip_length"] == 15
    assert manifest["clip_stride"] == 15
    assert manifest["clip_anchors"] == [0, 7, 14]
    assert manifest["visual_half_windows"] == [[0, 8], [7, 15]]
    assert manifest["visual_half_encoding"] == "l2_normalized_eight_frame_mean"
    assert [entry["episode_id"] for entry in index["episodes"]] == [0, 1]
    assert [entry["frames"] for entry in index["episodes"]] == [207, 207]
    for entry in index["episodes"]:
        frame_path = root / "encode" / entry["path"]
        frames = np.load(frame_path, allow_pickle=False)
        assert frames.shape == (entry["frames"], 3)
        assert frames.dtype == np.float32
    assert (root / "encode" / "visual_pca.npz").is_file()
    assert (root / "encode" / "numeric_normalizers.npz").is_file()
    assert not (root / "encode" / "raw_relations.npy").exists()
    assert not (root / "encode" / "projection_matrices.npz").exists()
    assert not (root / "encode" / "relation_pca.npz").exists()

    encode_stage(
        _config(tmp_path),
        output_dir=root,
        visual_encoder=FailingCocoreVisualEncoder(),
    )
    assert CocorePipelineAdapter.load_images_calls == [False, False, True]


def test_encode_stage_reports_new_build_but_not_cache_hit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    root = tmp_path / "timed-encode"

    encode_stage(
        _config(tmp_path),
        output_dir=root,
        visual_encoder=CocoreVisualEncoder(),
    )

    first = capsys.readouterr()
    assert first.out == ""
    assert [
        line.split(" step=", 1)[1].split(" ", 1)[0]
        for line in first.err.splitlines()
        if line.startswith("cocore_timing")
    ] == [
        "scan",
        "encode.numeric_normalization",
        "encode.visual_cache",
        "encode.pca_fusion",
        "encode",
    ]

    encode_stage(
        _config(tmp_path),
        output_dir=root,
        visual_encoder=FailingCocoreVisualEncoder(),
    )

    cached = capsys.readouterr()
    assert cached.out == ""
    assert "cocore_timing" not in cached.err


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("shape", "shape or dtype"),
        ("nonfinite", "NaN or infinity"),
        ("zero_norm", "non-positive norm"),
        ("frame_boundary", "boundary exceeds frame cache"),
    ],
)
def test_visual_half_cache_validation_rejects_malformed_semantics(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    encode_root = tmp_path / "encode"
    frame_root = encode_root / "frame_embeddings"
    frame_root.mkdir(parents=True)
    frames = np.tile(np.asarray([[1.0, 2.0, 4.0]], dtype=np.float32), (15, 1))
    expected = frames[:8].mean(axis=0)
    expected /= np.linalg.norm(expected)
    visual_halves = np.stack([expected, expected])[None, :].astype(np.float32)
    if corruption == "shape":
        visual_halves = np.empty((0, 2, 3), dtype=np.float32)
    elif corruption == "nonfinite":
        visual_halves[0, 0, 0] = np.nan
    elif corruption == "zero_norm":
        visual_halves[0, 0] = 0.0
    if corruption == "frame_boundary":
        frames = frames[:-1]
    np.save(encode_root / "visual_half_embeddings.npy", visual_halves)
    np.save(frame_root / "ep000000.npy", frames)
    clips = [
        ClipRecord(
            sample_id="ep000000_chunk_000000_000014",
            episode_id=0,
            task_index=0,
            task_name="clip",
            start_step=0,
            end_step=14,
            length=15,
            previous_sample_id=None,
            next_sample_id=None,
        )
    ]

    with pytest.raises(ValueError, match=message):
        cocore_pipeline._validate_visual_half_embedding_cache(encode_root, clips)


def test_encode_stage_caches_every_indexed_episode_including_short_episodes(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_short", ShortEpisodeCocorePipelineAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_pipeline_short"

    root, _, encoded = encode_stage(config, visual_encoder=CocoreVisualEncoder())

    assert len(encoded.clips) == 28
    index = json.loads((root / "encode" / "frame_embeddings_index.json").read_text())
    assert [(entry["episode_id"], entry["frames"]) for entry in index["episodes"]] == [
        (0, 207),
        (1, 207),
        (2, 5),
    ]
    result = run_pipeline(config, visual_encoder=FailingCocoreVisualEncoder())
    assert validate_output(result, config=config) == {"status": "valid", "selected_clips": 6}


def test_interrupted_encode_does_not_publish_partial_frame_cache(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    root = tmp_path / "interrupted-cocore"

    with pytest.raises(RuntimeError, match="injected cocore CLIP interruption"):
        encode_stage(
            _config(tmp_path),
            output_dir=root,
            visual_encoder=InterruptingCocoreVisualEncoder(),
        )

    assert not (root / "encode").exists()


@pytest.mark.parametrize("relation", ["cooccurrence", "sequence"])
def test_run_pipeline_publishes_relation_outputs_and_validate_recomputes_them(
    tmp_path: Path,
    relation: str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path, relation)
    root = tmp_path / "overridden-cocore-output"

    result = run_pipeline(config, output_dir=root, visual_encoder=CocoreVisualEncoder())

    assert result == root / f"select-{relation}-w1-top50pct"
    assert (root / "scan" / "manifest.json").is_file()
    assert (root / "encode" / "manifest.json").is_file()
    assert (root / "encode" / "visual_half_embeddings.npy").is_file()
    assert (root / "graph-14-motion-hard-nearest" / "prototype_catalog.json").is_file()
    assert (root / "graph-14-motion-hard-nearest" / "prototype_centers.npy").is_file()
    assert (root / "graph-14-motion-hard-nearest" / "half_action_labels.npy").is_file()
    for directory in ("scan", "encode", "graph-14-motion-hard-nearest"):
        manifest = json.loads((root / directory / "manifest.json").read_text())
        assert manifest["producer"] == "cocore"
        assert manifest["cocore_version"] == "0.9.0"
    catalog = json.loads(
        (root / "graph-14-motion-hard-nearest" / "prototype_catalog.json").read_text()
    )
    assert catalog["schema_version"] == 5
    assert catalog["strategy"] == "trajectory_retained_action_then_half_visual_nearest"
    assert catalog["total_raw_actions"] == 400
    assert catalog["constants"] == {
        "state_threshold": 0.03,
        "min_action_count": 400,
        "min_action_frequency": 0.005,
        "max_visual_centers": 16,
        "visual_half_windows": [[0, 8], [7, 15]],
        "cluster_count": "min(16, floor(log2(training_count)) - 2)",
        "retention_weight": "0.5 + 0.5 * retained_atomic_ratio",
        "distance_quantiles": [0.1, 0.9],
        "distance_weight_range": [1.0, 0.3],
        "duplicate_merge": "max + 0.5 * min",
    }
    assert catalog["leaf_prototypes"]
    half_action_labels = np.load(
        root / "graph-14-motion-hard-nearest" / "half_action_labels.npy"
    )
    assert half_action_labels.shape == (28, 2)
    assert half_action_labels.dtype.kind == "U"
    assert set(half_action_labels.flat) == {"move forward"}
    centers = np.load(root / "graph-14-motion-hard-nearest" / "prototype_centers.npy")
    assert centers.shape[1] == 3
    nodes = np.load(root / "graph-14-motion-hard-nearest" / "nodes.npz")
    assert "prototype_action_weights" not in nodes.files
    assert "prototype_distance_weights" not in nodes.files
    assert np.all(nodes["prototype_weights"].sum(axis=1) > 0.0)
    assert np.any(nodes["prototype_weights"].sum(axis=1) > 1.0)
    np.testing.assert_allclose(
        nodes["reliability"],
        np.maximum(nodes["support"] ** 0.5 * nodes["progress"] ** 0.5, 0.05),
        rtol=1.0e-6,
    )
    selected = [
        json.loads(line) for line in (result / "selected_manifest.jsonl").read_text().splitlines()
    ]
    all_rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    report = json.loads((result / "selection_report.json").read_text())
    assert len(selected) == 6
    assert len(all_rows) == 28
    assert {row["selection_phase"] for row in selected} == {"coverage_seed", "heap"}
    assert all(
        {"selection_step", "selection_score_delta", "heap_refreshes"} <= row.keys()
        for row in selected
    )
    assert all(
        {
            "support",
            "progress",
            "reliability",
            "prototype_labels",
            "prototype_action_labels",
            "prototype_cluster_ids",
            "primary_action_label",
            "half_action_labels",
        }
        <= row.keys()
        for row in all_rows
    )
    assert all("::" in label for row in all_rows for label in row["prototype_labels"])
    assert all("prototype_half_indices" not in row for row in all_rows)
    assert all("prototype_half_indices" not in row for row in selected)
    assert report["relation_type"] == relation
    assert report["relation_weight"] == 1.0
    assert all("prototype_action_weights" not in row for row in all_rows)
    assert all("prototype_distance_weights" not in row for row in all_rows)
    assert all("prototype_action_weights" not in row for row in selected)
    assert all("prototype_distance_weights" not in row for row in selected)
    assert report["prototype_schema_version"] == 5
    assert report["prototype_strategy"] == "trajectory_retained_action_then_half_visual_nearest"
    assert report["objective"]["total"] == (
        report["objective"]["weighted_relation"] - report["objective"]["redundancy"]
    )
    assert report["objective"]["weighted_relation"] == report["objective"]["relation"]
    assert report["coverage"]["target"] == report["coverage"]["achieved"]
    assert report["algorithm"] == {
        "type": "lazy_max_heap",
        "max_refreshes": 2,
    }
    assert report["heap"] == {
        "initial_size": len(all_rows) - report["initial_set_size"],
        "total_refreshes": sum(row["heap_refreshes"] for row in selected),
        "capped_selections": sum(row["heap_refreshes"] == 2 for row in selected),
        "max_refreshes_observed": max(row["heap_refreshes"] for row in selected),
    }
    run_manifest = json.loads((result / "run_manifest.json").read_text())
    assert run_manifest["producer"] == "cocore"
    assert run_manifest["cocore_version"] == "0.9.0"
    assert run_manifest["relation_type"] == relation
    assert run_manifest["relation_weight"] == 1.0
    assert run_manifest["prototype_schema_version"] == 5
    assert run_manifest["prototype_strategy"] == (
        "trajectory_retained_action_then_half_visual_nearest"
    )
    assert run_manifest["stage_directories"]["graph"] == "graph-14-motion-hard-nearest"
    assert run_manifest["algorithm"] == report["algorithm"]
    select_manifest = json.loads((result / "manifest.json").read_text())
    assert select_manifest["cocore_version"] == "0.9.0"
    assert select_manifest["relation_type"] == relation
    assert select_manifest["relation_weight"] == 1.0
    assert select_manifest["prototype_schema_version"] == 5
    assert select_manifest["prototype_strategy"] == (
        "trajectory_retained_action_then_half_visual_nearest"
    )
    resolved = yaml.safe_load((result / "resolved_config.yaml").read_text())
    assert resolved["output"]["directory"] == str(root)
    assert resolved["objective"] == {"relation": relation, "relation_weight": 1.0}
    assert validate_output(result, config=config) == {"status": "valid", "selected_clips": 6}

    select_manifest["cocore_version"] = "0.7.0"
    (result / "manifest.json").write_text(json.dumps(select_manifest))
    with pytest.raises(ValueError, match="selection manifest Cocore version"):
        validate_output(result, config=config)
    select_manifest["cocore_version"] = "0.9.0"
    (result / "manifest.json").write_text(json.dumps(select_manifest))

    report["relation_type"] = "sequence" if relation == "cooccurrence" else "cooccurrence"
    (result / "selection_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="report relation type"):
        validate_output(result, config=config)

    report["relation_type"] = relation
    report["heap"]["total_refreshes"] += 1
    (result / "selection_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="heap total refreshes"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_relation_weight(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    report = json.loads((result / "selection_report.json").read_text())
    report["relation_weight"] = 2.0
    (result / "selection_report.json").write_text(json.dumps(report))

    with pytest.raises(ValueError, match="report relation weight"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_relation_objective(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    report = json.loads((result / "selection_report.json").read_text())
    report["objective"]["relation"] += 1.0
    (result / "selection_report.json").write_text(json.dumps(report))

    with pytest.raises(ValueError, match="objective relation mismatch"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_absolute_leaf_weights_by_replay(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    nodes_path = result.parent / "graph-14-motion-hard-nearest" / "nodes.npz"
    with np.load(nodes_path) as stored:
        nodes = {name: stored[name].copy() for name in stored.files}
    nodes["prototype_weights"][0] *= np.float32(0.9)
    np.savez(nodes_path, **nodes)

    with pytest.raises(ValueError, match="prototype replay"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_hierarchical_catalog(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    catalog_path = result.parent / "graph-14-motion-hard-nearest" / "prototype_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["leaf_prototypes"][0]["label"] = "wrong"
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="hierarchical prototype label"):
        validate_output(result, config=config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("requested_centers", 999, "requested center"),
        ("actual_centers", 999, "leaf count"),
        ("training_count", -1, "training count"),
        ("nearest_distance_q10", -1.0, "distance quantiles"),
    ],
)
def test_validate_recomputes_hierarchical_catalog_diagnostics(
    tmp_path: Path,
    field: str,
    value: int | float,
    message: str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    catalog_path = result.parent / "graph-14-motion-hard-nearest" / "prototype_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    category = next(
        category
        for category in catalog["action_categories"]
        if category["retained"] and category["training_count"] > 0
    )
    category[field] = value
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match=message):
        validate_output(result, config=config)


def test_validate_rejects_tampered_hierarchical_output_row(tmp_path: Path) -> None:
    import pyarrow as pa

    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    all_path = result / "all_clips.parquet"
    rows = pq.read_table(all_path).to_pylist()
    rows[0]["half_action_labels"] = ["stop", "stop"]
    pq.write_table(pa.Table.from_pylist(rows), all_path)

    with pytest.raises(ValueError, match="hierarchical prototype row"):
        validate_output(result, config=config)


def test_validate_rejects_missing_hierarchical_centers(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    (result.parent / "graph-14-motion-hard-nearest" / "prototype_centers.npy").unlink()

    with pytest.raises(ValueError, match="invalid stage artifacts: graph"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_visual_half_embeddings(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "encode" / "visual_half_embeddings.npy"
    values = np.load(path)
    values[0, 0] = np.roll(values[0, 0], 1)
    np.save(path, values)

    with pytest.raises(ValueError, match="visual half embeddings do not match frame cache"):
        validate_output(result, config=config)


def test_validate_rejects_missing_visual_half_embeddings_as_encode_artifact(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    (result.parent / "encode" / "visual_half_embeddings.npy").unlink()

    with pytest.raises(ValueError, match="invalid stage artifacts: encode"):
        validate_output(result, config=config)


@pytest.mark.parametrize("corruption", ["missing", "contents", "dtype", "shape"])
def test_validate_rejects_corrupt_episode_frame_cache(
    tmp_path: Path,
    corruption: str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    encode_root = result.parent / "encode"
    index = json.loads((encode_root / "frame_embeddings_index.json").read_text())
    path = encode_root / index["episodes"][0]["path"]
    values = np.load(path, allow_pickle=False)
    if corruption == "missing":
        path.unlink()
    elif corruption == "contents":
        values[0, 0] += np.float32(1.0)
        np.save(path, values)
    elif corruption == "dtype":
        np.save(path, values.astype(np.float64))
    else:
        np.save(path, values[:-1])

    with pytest.raises(ValueError, match="frame embedding"):
        validate_output(result, config=config)


def test_encode_cache_rejects_corrupt_episode_frame_cache_without_force(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    root, _, _ = encode_stage(config, visual_encoder=CocoreVisualEncoder())
    index = json.loads((root / "encode" / "frame_embeddings_index.json").read_text())
    (root / "encode" / index["episodes"][0]["path"]).unlink()

    with pytest.raises(FileExistsError, match="frame embedding.*--force"):
        encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())


def test_encode_stage_rejects_schema_four_artifact_names_without_force(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    root, _, _ = encode_stage(config, visual_encoder=CocoreVisualEncoder())
    encode_root = root / "encode"
    halves = np.load(encode_root / "visual_half_embeddings.npy", allow_pickle=False)
    np.save(encode_root / "visual_clip_embeddings.npy", halves[:, 0])
    (encode_root / "visual_half_embeddings.npy").unlink()

    with pytest.raises(FileExistsError, match="stage cache is incompatible"):
        encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())


def test_graph_build_rejects_semantically_tampered_visual_half_cache(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    root, _, _ = encode_stage(config, visual_encoder=CocoreVisualEncoder())
    path = root / "encode" / "visual_half_embeddings.npy"
    values = np.load(path, allow_pickle=False)
    values[0, 0] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    np.save(path, values)

    with pytest.raises(FileExistsError, match="visual half embedding.*--force"):
        graph_stage(config, visual_encoder=FailingCocoreVisualEncoder())
    assert not (root / "graph-14-motion-hard-nearest").exists()


def test_graph_stage_reports_aggregate_and_prototype_step_timings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)

    graph_stage(_config(tmp_path), visual_encoder=CocoreVisualEncoder())

    steps = [
        line.split(" step=", 1)[1].split(" ", 1)[0]
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("cocore_timing")
    ]
    assert steps[-8:] == [
        "graph.reliability",
        "graph.prototypes.action_scan",
        "graph.prototypes.kmeans",
        "graph.prototypes.center_statistics",
        "graph.prototypes.candidate_assignment",
        "graph.prototypes",
        "graph.sparse_graph",
        "graph",
    ]


def test_validate_rejects_tampered_visual_prototype_center(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-14-motion-hard-nearest" / "prototype_centers.npy"
    centers = np.load(path)
    centers[0] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    np.save(path, centers)

    with pytest.raises(ValueError, match="prototype (center|replay)"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_half_action_labels(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-14-motion-hard-nearest" / "half_action_labels.npy"
    labels = np.load(path)
    labels[0, 0] = "stop"
    np.save(path, labels)

    with pytest.raises(ValueError, match="half action"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_selected_hierarchical_row(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    selected_path = result / "selected_manifest.jsonl"
    rows = [json.loads(line) for line in selected_path.read_text().splitlines()]
    rows[0]["primary_action_label"] = "wrong"
    selected_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="selected hierarchical prototype row"):
        validate_output(result, config=config)


@pytest.mark.parametrize(
    "field",
    ["prototype_action_weights", "prototype_distance_weights"],
)
def test_validate_rejects_obsolete_schema_three_field_in_selected_row(
    tmp_path: Path,
    field: str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    selected_path = result / "selected_manifest.jsonl"
    rows = [json.loads(line) for line in selected_path.read_text().splitlines()]
    rows[0][field] = [1.0]
    selected_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="obsolete schema-3 prototype field"):
        validate_output(result, config=config)


def test_validate_rejects_schema_four_manifest_explicitly(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    run_path = result / "run_manifest.json"
    manifest = json.loads(run_path.read_text())
    manifest["prototype_schema_version"] = 4
    run_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="prototype schema version is incompatible"):
        validate_output(result, config=config)


def test_validate_rejects_schema_four_graph_catalog_explicitly(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    catalog_path = result.parent / "graph-14-motion-hard-nearest" / "prototype_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["schema_version"] = 4
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="catalog schema"):
        validate_output(result, config=config)


@pytest.mark.parametrize(
    ("stage", "directory"),
    [
        ("scan", "scan"),
        ("encode", "encode"),
        ("graph", "graph-14-motion-hard-nearest"),
    ],
)
def test_validate_rejects_tampered_stage_manifest_contract(
    tmp_path: Path,
    stage: str,
    directory: str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    manifest_path = result.parent / directory / "manifest.json"
    original = json.loads(manifest_path.read_text())

    for field, value in (
        ("producer", "not-cocore"),
        ("cocore_version", "0.7.0"),
        ("cocore_stage", "wrong-stage"),
    ):
        tampered = {**original, field: value}
        manifest_path.write_text(json.dumps(tampered))
        with pytest.raises(ValueError, match=f"stage manifest metadata.*{stage}"):
            validate_output(result, config=config)
        manifest_path.write_text(json.dumps(original))


def test_validate_replays_current_adapter_state(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())

    CocorePipelineAdapter.motion_sign = -1.0
    try:
        with pytest.raises(ValueError, match="prototype replay"):
            validate_output(result, config=config)
    finally:
        CocorePipelineAdapter.motion_sign = 1.0


def test_validate_rejects_wrong_relation_directory(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    renamed = result.parent / "select-sequence-w1-top50pct"
    result.rename(renamed)

    with pytest.raises(ValueError, match="selection directory"):
        validate_output(renamed, config=config)


def test_sequence_and_cooccurrence_outputs_can_coexist(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    root = tmp_path / "cocore-output"

    cooccurrence = run_pipeline(
        _config(tmp_path, "cooccurrence"),
        output_dir=root,
        visual_encoder=CocoreVisualEncoder(),
    )
    sequence = run_pipeline(
        _config(tmp_path, "sequence"),
        output_dir=root,
        visual_encoder=CocoreVisualEncoder(),
    )

    assert cooccurrence == root / "select-cooccurrence-w1-top50pct"
    assert sequence == root / "select-sequence-w1-top50pct"
    assert cooccurrence.is_dir()
    assert sequence.is_dir()
    assert validate_output(cooccurrence, config=_config(tmp_path, "cooccurrence")) == {
        "status": "valid",
        "selected_clips": 6,
    }
    assert validate_output(sequence, config=_config(tmp_path, "sequence")) == {
        "status": "valid",
        "selected_clips": 6,
    }


def test_validate_aligns_sorted_output_rows_by_sample_id(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_reverse_synthetic", ReverseOrderCocorePipelineAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_pipeline_reverse_synthetic"

    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())

    assert validate_output(result, config=config) == {"status": "valid", "selected_clips": 6}
