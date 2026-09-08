from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from cocore import pipeline as cocore_pipeline
from cocore import prototypes as cocore_prototypes
from cocore.config import to_relcore_config
from cocore.pipeline import encode_stage, graph_stage, run_pipeline, scan_stage, validate_output
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
            EpisodeRecord(0, 605, 0, "task zero"),
            EpisodeRecord(1, 605, 1, "task one"),
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
                actions=np.stack([steps / 604.0, np.zeros_like(steps)], axis=1),
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
        self._records = (*self._records, EpisodeRecord(2, 1, 2, "short task"))


class MixedStopCocorePipelineAdapter(CocorePipelineAdapter):
    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__(config)
        self._records = (*self._records, EpisodeRecord(2, 30, 2, "stop task"))

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        for episode in super().iter_episodes(
            num_workers=num_workers,
            max_episodes=max_episodes,
            load_images=load_images,
        ):
            if episode.episode_id == 2:
                episode.observations["observation.state"][:] = 0.0
            yield episode


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
        "prototypes": {
            "method": "motion_primitives",
            "batch_size": 64,
            "max_iter": 2,
            "tol": 1.0e-4,
            "num_threads": 4,
        },
        "graph": {"knn": 2, "similarity_threshold": 0.8, "cooccurrence_max_gap": 4},
        "objective": {"relation": relation, "relation_weight": 1.0},
        "selection": {
            "ratio": 0.5,
            "budget": 10,
        },
        "runtime": {"num_workers": 0, "max_episodes": None, "resume": True},
        "output": {"directory": str(tmp_path / "cocore-output")},
    }


def test_libero_config_translation_retains_fifteen_frame_clip_geometry(
    tmp_path: Path,
) -> None:
    translated = to_relcore_config(_config(tmp_path))

    assert translated["clip"] == {"length": 15, "stride": 15}


def test_bridge_profile_rejects_legacy_fifteen_frame_scan_cache(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    root = tmp_path / "profile-specific-scan"

    _, _, libero_clips, libero_fingerprint = scan_stage(config, output_dir=root)
    assert {clip.length for clip in libero_clips} == {15}

    config["prototypes"]["profile"] = "bridge_v2"  # type: ignore[index]
    with pytest.raises(FileExistsError, match="--force"):
        scan_stage(config, output_dir=root)

    _, _, bridge_clips, bridge_fingerprint = scan_stage(
        config,
        output_dir=root,
        force=True,
    )
    assert len(bridge_clips) == 174
    assert {clip.length for clip in bridge_clips} == {7}
    assert bridge_fingerprint != libero_fingerprint
    manifest = json.loads((root / "scan" / "manifest.json").read_text())
    assert manifest["clip_length"] == 7


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
    assert CocorePipelineAdapter.load_images_calls == [False, True]
    embeddings = np.load(root / "encode" / "embeddings.npy")
    assert embeddings.shape == (82, 158)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1.0e-6)
    stored = np.load(root / "encode" / "visual_half_embeddings.npy")
    np.testing.assert_array_equal(stored, encoded.visual_half_embeddings)
    assert stored.shape == (82, 2, 3)
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
    assert manifest["embedding_dim"] == 158
    assert manifest["counts"] == {
        "candidate_fragments": 82,
        "pca_fit_fragments": 82,
        "encoded_episodes": 2,
        "encoded_frames": 1210,
    }
    assert manifest["visual_half_embedding_dim"] == 3
    assert manifest["clip_length"] == 15
    assert manifest["window_policy"] == "near_uniform_full_coverage"
    assert "clip_stride" not in manifest
    assert manifest["clip_anchors"] == [0, 7, 14]
    assert manifest["visual_half_windows"] == [[0, 8], [7, 15]]
    assert manifest["visual_half_encoding"] == "l2_normalized_eight_frame_mean"
    assert [entry["episode_id"] for entry in index["episodes"]] == [0, 1]
    assert [entry["frames"] for entry in index["episodes"]] == [605, 605]
    for entry in index["episodes"]:
        frame_path = root / "encode" / entry["path"]
        frames = np.load(frame_path, allow_pickle=False)
        assert frames.shape == (entry["frames"], 3)
        assert frames.dtype == np.float32
    assert (root / "encode" / "visual_pca.npz").is_file()
    assert (root / "encode" / "numeric_normalizers.npz").is_file()
    assert not (root / "scan" / "normalization.npz").exists()
    assert not (root / "encode" / "raw_relations.npy").exists()
    assert not (root / "encode" / "projection_matrices.npz").exists()
    assert not (root / "encode" / "relation_pca.npz").exists()

    encode_stage(
        _config(tmp_path),
        output_dir=root,
        visual_encoder=FailingCocoreVisualEncoder(),
    )
    assert CocorePipelineAdapter.load_images_calls == [False, True]


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

    assert len(encoded.clips) == 82
    index = json.loads((root / "encode" / "frame_embeddings_index.json").read_text())
    assert [(entry["episode_id"], entry["frames"]) for entry in index["episodes"]] == [
        (0, 605),
        (1, 605),
        (2, 1),
    ]
    result = run_pipeline(config, visual_encoder=FailingCocoreVisualEncoder())
    assert validate_output(result, config=config) == {"status": "valid", "selected_clips": 10}


def test_interrupted_encode_does_not_publish_partial_frame_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    root = tmp_path / "interrupted-cocore"

    with pytest.raises(RuntimeError, match="injected cocore CLIP interruption"):
        encode_stage(
            _config(tmp_path),
            output_dir=root,
            visual_encoder=InterruptingCocoreVisualEncoder(),
        )

    assert not (root / "encode").exists()
    captured = capsys.readouterr()
    assert "cocore_timing step=encode.numeric_normalization " in captured.err
    assert "cocore_timing step=encode.visual_cache " not in captured.err
    assert "cocore_timing step=encode.pca_fusion " not in captured.err
    assert "cocore_timing step=encode " not in captured.err


@pytest.mark.parametrize("relation", ["cooccurrence", "sequence"])
def test_run_pipeline_publishes_relation_outputs_and_validate_recomputes_them(
    tmp_path: Path,
    relation: str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path, relation)
    root = tmp_path / "overridden-cocore-output"

    result = run_pipeline(config, output_dir=root, visual_encoder=CocoreVisualEncoder())

    assert result == root / f"select-{relation}-w1-top50pct-random-multibranch"
    assert (root / "scan" / "manifest.json").is_file()
    assert (root / "encode" / "manifest.json").is_file()
    assert (root / "encode" / "visual_half_embeddings.npy").is_file()
    assert (root / "encode" / "action_variation_raw.npy").is_file()
    assert (root / "encode" / "action_variation.npy").is_file()
    assert (root / "encode" / "visual_action_consistency_raw.npy").is_file()
    assert (root / "encode" / "visual_action_consistency.npy").is_file()
    assert (root / "graph-18-motion-hard-nearest-pca" / "prototype_catalog.json").is_file()
    assert (root / "graph-18-motion-hard-nearest-pca" / "prototype_centers.npy").is_file()
    assert (root / "graph-18-motion-hard-nearest-pca" / "half_action_labels.npy").is_file()
    assert (root / "graph-18-motion-hard-nearest-pca" / "source_clip_indices.npy").is_file()
    for directory in ("scan", "encode", "graph-18-motion-hard-nearest-pca"):
        manifest = json.loads((root / directory / "manifest.json").read_text())
        assert manifest["producer"] == "cocore"
        assert manifest["cocore_version"] == "0.19.0"
    scan_manifest = json.loads((root / "scan" / "manifest.json").read_text())
    assert scan_manifest["window_policy"] == "near_uniform_full_coverage"
    assert scan_manifest["clip_length"] == 15
    catalog = json.loads(
        (root / "graph-18-motion-hard-nearest-pca" / "prototype_catalog.json").read_text()
    )
    assert catalog["schema_version"] == 10
    assert catalog["profile"] == "libero"
    assert catalog["use_stop_bucket"] is True
    assert catalog["strategy"] == (
        "trajectory_sampled_optional_stop_retained_action_then_cropped_pca_half_visual_"
        "hybrid_kmeans_nearest"
    )
    assert catalog["total_raw_actions"] == 400
    assert catalog["constants"] == {
        "primitive_thresholds": {
            "translation": 0.03,
            "roll": None,
            "tilt": 0.03,
            "rotation": 0.03,
            "gripper": 0.03,
        },
        "roll_axis": None,
        "roll_labels": None,
        "cyclic_axes": [],
        "min_action_count": 400,
        "min_action_frequency": 0.005,
        "retention_threshold": "max(min_action_count, ceil(min_action_frequency * W))",
        "max_visual_centers": 30,
        "full_kmeans_max_training_count": 65536,
        "full_kmeans_openmp_threads": 1,
        "minibatch_kmeans_openmp_threads": 4,
        "kmeans_n_init": 1,
        "large_bucket_parallelism": "serial",
        "trajectory_window_length": 8,
        "trajectory_window_policy": "full_coverage_max_gap_3_tail_rebalanced",
        "visual_half_windows": [[0, 8], [7, 15]],
        "visual_projection": "frame @ visual_pca.components[:, :frame_embedding_dim].T",
        "visual_projection_centering": "none",
        "visual_projection_padding": "right_zero_to_128",
        "visual_half_encoding": "l2_normalized_mean_of_eight_projected_frames",
        "cluster_count": (
            "min(training_count, min(30, max(10, floor(4 * log2(training_count) - 30))))"
        ),
        "retention_weight": "0.5 + 0.5 * retained_atomic_ratio",
        "distance_quantiles": [0.1, 0.9],
        "distance_weight_range": [1.0, 0.3],
        "duplicate_merge": "max + 0.5 * min",
    }
    assert catalog["leaf_prototypes"]
    half_action_labels = np.load(
        root / "graph-18-motion-hard-nearest-pca" / "half_action_labels.npy"
    )
    assert half_action_labels.shape == (82, 2)
    assert half_action_labels.dtype.kind == "U"
    assert set(half_action_labels.flat) == {"move forward"}
    centers = np.load(root / "graph-18-motion-hard-nearest-pca" / "prototype_centers.npy")
    assert centers.shape[1] == 128
    nodes = np.load(root / "graph-18-motion-hard-nearest-pca" / "nodes.npz")
    source_clip_indices = np.load(
        root / "graph-18-motion-hard-nearest-pca" / "source_clip_indices.npy"
    )
    np.testing.assert_array_equal(source_clip_indices, np.arange(82, dtype=np.int64))
    sequence_edges = np.load(root / "graph-18-motion-hard-nearest-pca" / "sequence_edges.npz")
    assert len(sequence_edges["source"]) == 80
    graph_manifest = json.loads(
        (root / "graph-18-motion-hard-nearest-pca" / "manifest.json").read_text()
    )
    action_variation_contract = {
        "input": "robust_scaled_full_episode_actions",
        "difference": "l2_norm_current_minus_previous",
        "difference_weight": 2.0,
        "first_step_difference": 0.0,
        "future_window": 5,
        "future_variance": "mean_dimension_population_variance",
        "future_variance_weight": 1.0,
        "future_boundary": "truncate_available_less_than_two_is_zero",
        "clip_aggregation": "top_k_mean",
        "top_k": 3,
        "normalization": "clip_quantile_scale_to_zero_one",
        "quantile_low": 0.01,
        "quantile_high": 0.99,
        "epsilon": 1.0e-8,
    }
    visual_action_consistency_contract = {
        "formula": "l2(v_t-v_t_minus_1)/(l2(a_t-a_t_minus_1)+epsilon)",
        "visual_input": "configured_encoder_full_episode_frame_features",
        "action_input": "robust_scaled_full_episode_actions",
        "visual_difference": "l2_norm_current_minus_previous",
        "action_difference": "l2_norm_current_minus_previous",
        "ratio": "visual_difference_over_action_difference_plus_epsilon",
        "first_step": "copy_first_valid_ratio",
        "clip_aggregation": "top_k_mean",
        "top_k": 3,
        "normalization": "clip_quantile_scale_to_zero_one",
        "quantile_low": 0.01,
        "quantile_high": 0.99,
        "epsilon": 1.0e-8,
    }
    encode_manifest = json.loads((root / "encode" / "manifest.json").read_text())
    assert encode_manifest["action_variation"] == action_variation_contract
    assert encode_manifest["visual_action_consistency"] == visual_action_consistency_contract
    assert graph_manifest["action_variation"] == action_variation_contract
    assert graph_manifest["visual_action_consistency"] == visual_action_consistency_contract
    assert graph_manifest["prototype_profile"] == "libero"
    assert graph_manifest["motion_primitive"] == {
        "profile": "libero",
        "primitive_thresholds": {
            "translation": 0.03,
            "roll": None,
            "tilt": 0.03,
            "rotation": 0.03,
            "gripper": 0.03,
        },
        "roll_axis": None,
        "roll_labels": None,
        "cyclic_axes": [],
        "min_action_count": 400,
        "min_action_frequency": 0.005,
        "retention_threshold": "max(min_action_count, ceil(min_action_frequency * W))",
    }
    assert graph_manifest["sequence_adjacency"] == "ordered_candidates"
    assert graph_manifest["prototype_visual_dim"] == 128
    assert graph_manifest["prototype_visual_projection"] == (
        "frame @ visual_pca.components[:, :frame_embedding_dim].T"
    )
    assert graph_manifest["prototype_visual_normalization"] == (
        "l2_normalized_eight_frame_mean_after_projection"
    )
    assert graph_manifest["trajectory_window_length"] == 8
    assert graph_manifest["trajectory_horizon"] == 7
    assert "trajectory_window_max_gap" not in graph_manifest
    assert "trajectory_window_policy" not in graph_manifest
    assert "prototype_action_weights" not in nodes.files
    assert "prototype_distance_weights" not in nodes.files
    assert {
        "action_variation_raw",
        "action_variation",
        "visual_action_consistency_raw",
        "visual_action_consistency",
    } <= set(nodes.files)
    assert np.all(nodes["prototype_weights"].sum(axis=1) > 0.0)
    assert np.any(nodes["prototype_weights"].sum(axis=1) > 1.0)
    np.testing.assert_allclose(
        nodes["reliability"],
        np.maximum(
            (
                nodes["support"]
                * nodes["progress"]
                * nodes["action_variation"]
                * nodes["visual_action_consistency"]
                * nodes["action_jump"]
            )
            ** 0.2,
            0.05,
        ),
        rtol=1.0e-6,
    )
    selected = [
        json.loads(line) for line in (result / "selected_manifest.jsonl").read_text().splitlines()
    ]
    all_rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    report = json.loads((result / "selection_report.json").read_text())
    assert len(selected) == 10
    assert len(all_rows) == 82
    assert {row["selection_phase"] for row in selected} == {
        "coverage_seed",
        "branch_final",
    }
    assert all(
        {"selection_step", "selection_score_delta"} <= row.keys() and "heap_refreshes" not in row
        for row in selected
    )
    assert all("heap_refreshes" not in row for row in all_rows)
    assert all(
        {
            "support",
            "progress",
            "action_variation_raw",
            "action_variation",
            "visual_action_consistency_raw",
            "visual_action_consistency",
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
    assert report["prototype_schema_version"] == 10
    assert report["prototype_profile"] == "libero"
    assert report["use_stop_bucket"] is True
    assert report["prototype_strategy"] == (
        "trajectory_sampled_optional_stop_retained_action_then_cropped_pca_half_visual_"
        "hybrid_kmeans_nearest"
    )
    assert report["objective"]["total"] == (
        report["objective"]["weighted_relation"] - report["objective"]["redundancy"]
    )
    assert report["objective"]["weighted_relation"] == report["objective"]["relation"]
    assert report["coverage"]["target"] == report["coverage"]["achieved"]
    assert report["algorithm"]["type"] == "random_multibranch"
    assert report["selection_schema_version"] == 3
    assert report["action_variation"] == action_variation_contract
    assert report["visual_action_consistency"] == visual_action_consistency_contract
    assert "heap" not in report
    assert "branch_search" in report
    run_manifest = json.loads((result / "run_manifest.json").read_text())
    assert run_manifest["producer"] == "cocore"
    assert run_manifest["cocore_version"] == "0.19.0"
    assert run_manifest["relation_type"] == relation
    assert run_manifest["relation_weight"] == 1.0
    assert run_manifest["prototype_schema_version"] == 10
    assert run_manifest["selection_schema_version"] == 3
    assert run_manifest["action_variation"] == action_variation_contract
    assert run_manifest["visual_action_consistency"] == visual_action_consistency_contract
    assert run_manifest["prototype_profile"] == "libero"
    assert run_manifest["use_stop_bucket"] is True
    assert run_manifest["prototype_strategy"] == (
        "trajectory_sampled_optional_stop_retained_action_then_cropped_pca_half_visual_"
        "hybrid_kmeans_nearest"
    )
    assert run_manifest["stage_directories"]["graph"] == "graph-18-motion-hard-nearest-pca"
    assert run_manifest["algorithm"] == report["algorithm"]
    assert run_manifest["window_policy"] == "near_uniform_full_coverage"
    assert run_manifest["clip_length"] == 15
    assert run_manifest["clip_anchors"] == [0, 7, 14]
    assert run_manifest["visual_half_windows"] == [[0, 8], [7, 15]]
    assert run_manifest["visual_half_encoding"] == "l2_normalized_eight_frame_mean"
    assert run_manifest["trajectory_window_length"] == 8
    assert run_manifest["trajectory_horizon"] == 7
    assert "trajectory_window_max_gap" not in run_manifest
    assert "trajectory_window_policy" not in run_manifest
    assert run_manifest["sequence_adjacency"] == "ordered_candidates"
    select_manifest = json.loads((result / "manifest.json").read_text())
    assert select_manifest["cocore_version"] == "0.19.0"
    assert select_manifest["relation_type"] == relation
    assert select_manifest["relation_weight"] == 1.0
    assert select_manifest["prototype_schema_version"] == 10
    assert select_manifest["selection_schema_version"] == 3
    assert select_manifest["action_variation"] == action_variation_contract
    assert select_manifest["visual_action_consistency"] == visual_action_consistency_contract
    assert select_manifest["prototype_profile"] == "libero"
    assert select_manifest["use_stop_bucket"] is True
    assert select_manifest["prototype_strategy"] == (
        "trajectory_sampled_optional_stop_retained_action_then_cropped_pca_half_visual_"
        "hybrid_kmeans_nearest"
    )
    resolved = yaml.safe_load((result / "resolved_config.yaml").read_text())
    assert resolved["output"]["directory"] == str(root)
    assert resolved["objective"] == {"relation": relation, "relation_weight": 1.0}
    assert validate_output(result, config=config) == {"status": "valid", "selected_clips": 10}

    select_manifest["cocore_version"] = "0.7.0"
    (result / "manifest.json").write_text(json.dumps(select_manifest))
    with pytest.raises(ValueError, match="selection manifest Cocore version"):
        validate_output(result, config=config)
    select_manifest["cocore_version"] = "0.19.0"
    (result / "manifest.json").write_text(json.dumps(select_manifest))

    report["relation_type"] = "sequence" if relation == "cooccurrence" else "cooccurrence"
    (result / "selection_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="report relation type"):
        validate_output(result, config=config)

    report["relation_type"] = relation
    report["heap"] = {}
    (result / "selection_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="heap metadata"):
        validate_output(result, config=config)


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_support_only_bridge_profile_changes_graph_and_artifact_contract(
    tmp_path: Path,
    profile: str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    dual_config = _config(tmp_path, "sequence")
    dual_config["reliability_metrics"] = [
        "support", "progress", "action_variation", "visual_action_consistency"
    ]
    dual_config["prototypes"]["profile"] = profile  # type: ignore[index]
    support_config = copy.deepcopy(dual_config)
    support_config["reliability_metrics"] = ["support"]
    root = tmp_path / "support-only-output"

    _, _, _, _, dual_fingerprint = graph_stage(
        dual_config,
        output_dir=root,
        visual_encoder=CocoreVisualEncoder(),
    )

    with pytest.raises(FileExistsError, match="--force"):
        graph_stage(
            support_config,
            output_dir=root,
            visual_encoder=FailingCocoreVisualEncoder(),
        )

    _, _, _, _, support_fingerprint = graph_stage(
        support_config,
        output_dir=root,
        force=True,
        visual_encoder=FailingCocoreVisualEncoder(),
    )

    assert support_fingerprint != dual_fingerprint
    graph_root = root / "graph-18-motion-hard-nearest-pca"
    nodes = np.load(graph_root / "nodes.npz")
    embeddings = np.load(root / "encode" / "embeddings.npy")
    effective_k = min(int(support_config["quality"]["knn"]), len(embeddings) - 1)
    scaled_support = nodes["support"] * (effective_k + 1)
    np.testing.assert_allclose(scaled_support, np.round(scaled_support), atol=1e-6)
    assert np.all(scaled_support >= 1)
    np.testing.assert_allclose(
        nodes["reliability"],
        np.maximum(nodes["support"], 0.05),
        rtol=1.0e-6,
    )

    result = run_pipeline(
        support_config,
        output_dir=root,
        visual_encoder=FailingCocoreVisualEncoder(),
    )
    graph_manifest = json.loads((graph_root / "manifest.json").read_text())
    select_manifest = json.loads((result / "manifest.json").read_text())
    run_manifest = json.loads((result / "run_manifest.json").read_text())
    report = json.loads((result / "selection_report.json").read_text())
    resolved = yaml.safe_load((result / "resolved_config.yaml").read_text())
    for metadata in (graph_manifest, select_manifest, run_manifest, report, resolved):
        assert metadata["reliability_metrics"] == ["support"]

    assert validate_output(result, config=support_config) == {
        "status": "valid",
        "selected_clips": 10,
    }
    with pytest.raises(ValueError, match="reliability metrics"):
        validate_output(result, config=dual_config)

    original_graph_manifest = (graph_root / "manifest.json").read_text()
    graph_manifest.pop("support_mode")
    (graph_root / "manifest.json").write_text(json.dumps(graph_manifest))
    with pytest.raises(ValueError, match="support"):
        validate_output(result, config=support_config)
    (graph_root / "manifest.json").write_text(original_graph_manifest)

    nodes.close()
    nodes_path = graph_root / "nodes.npz"
    with np.load(nodes_path) as stored_nodes:
        tampered_nodes = {name: stored_nodes[name] for name in stored_nodes.files}
    tampered_nodes["reliability"] = np.zeros_like(tampered_nodes["reliability"])
    np.savez(nodes_path, **tampered_nodes)

    with pytest.raises(ValueError, match="graph node reliability"):
        validate_output(result, config=support_config)


def test_support_formula_invalidates_graph_and_downstream_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    original_hash = cocore_pipeline.stable_hash

    def legacy_hash(payload):
        if isinstance(payload, dict) and payload.get("stage") == "graph":
            payload = {key: value for key, value in payload.items() if key != "support_mode"}
        return original_hash(payload)

    # Reproduce the pre-migration graph fingerprint, which had no formula identifier.
    with monkeypatch.context() as context:
        context.setattr(cocore_pipeline, "stable_hash", legacy_hash)
        result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    old_run = json.loads((result / "run_manifest.json").read_text())
    with pytest.raises(FileExistsError, match="--force"):
        run_pipeline(config, visual_encoder=FailingCocoreVisualEncoder())
    result = run_pipeline(config, force=True, visual_encoder=FailingCocoreVisualEncoder())
    new_run = json.loads((result / "run_manifest.json").read_text())
    for stage in ("scan", "encode"):
        assert new_run["stage_fingerprints"][stage] == old_run["stage_fingerprints"][stage]
    for stage in ("graph", "select"):
        assert new_run["stage_fingerprints"][stage] != old_run["stage_fingerprints"][stage]
    assert validate_output(result, config=config)["status"] == "valid"


def test_support_k_is_saved_validated_and_invalidates_graph_cache(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    old_run = json.loads((result / "run_manifest.json").read_text())
    changed = copy.deepcopy(config)
    changed["quality"]["knn"] = 4
    with pytest.raises(ValueError, match="support.*k"):
        validate_output(result, config=changed)
    with pytest.raises(FileExistsError, match="--force"):
        run_pipeline(changed, visual_encoder=FailingCocoreVisualEncoder())
    result = run_pipeline(changed, force=True, visual_encoder=FailingCocoreVisualEncoder())
    new_run = json.loads((result / "run_manifest.json").read_text())
    for stage in ("scan", "encode"):
        assert old_run["stage_fingerprints"][stage] == new_run["stage_fingerprints"][stage]
    for stage in ("graph", "select"):
        assert old_run["stage_fingerprints"][stage] != new_run["stage_fingerprints"][stage]
    stored = yaml.safe_load((result / "resolved_config.yaml").read_text())
    assert stored["quality"]["knn"] == 4
    assert validate_output(result, config=changed)["status"] == "valid"
    assert validate_output(result)["status"] == "valid"


def test_random_multibranch_pipeline_publishes_and_replays_branch_search(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path, "sequence")
    root = tmp_path / "random-multibranch-output"

    result = run_pipeline(config, output_dir=root, visual_encoder=CocoreVisualEncoder())
    first = capsys.readouterr()

    assert result == root / "select-sequence-w1-top50pct-random-multibranch"
    selected = [
        json.loads(line) for line in (result / "selected_manifest.jsonl").read_text().splitlines()
    ]
    report = json.loads((result / "selection_report.json").read_text())
    algorithm = {
        "type": "random_multibranch",
        "branches": 8,
        "children_per_branch": 4,
        "batch_size": 10,
        "first_recombination_round": 20,
        "recombination_interval": 10,
        "commit_size": 100,
        "retained_size": 100,
        "seed": 7,
        "recombination_ranking": {
            "committed": "branch_frequency_then_seeded_random",
            "retained": "branch_frequency_then_reliability_then_sample_id",
        },
        "similarity_penalty": {
            "backend": "faiss",
            "index": "IndexFlatIP",
            "metric": "cosine_on_l2_normalized_embeddings",
            "query": "range_search",
            "main_reference": "all_fixed",
            "scope": "new_active_to_all_main_and_previous_active",
            "pairs": "incremental_cross_pairs_only",
            "accumulation": "parent_plus_child_delta",
            "recombination": "reset_then_replay_retained",
            "final_objective": "winner_accumulated_incremental_redundancy",
        },
        "sequence_relation": {
            "scope": "new_active_to_all_main_and_previous_active",
            "pairs": "incremental_cross_edges_only",
            "accumulation": "parent_plus_child_delta",
            "recombination": "reset_then_replay_retained",
            "final_objective": "winner_accumulated_incremental_sequence",
        },
    }
    assert report["algorithm"] == algorithm
    assert "heap" not in report
    branch_search = report["branch_search"]
    assert {key: value for key, value in branch_search.items() if key != "timings"} == {
        "rounds": 1,
        "evaluated_branches": 8,
        "recombinations": 0,
        "committed_clips": 0,
        "final_active_clips": len(selected) - report["initial_set_size"],
    }
    timings = branch_search["timings"]
    assert len(timings["rounds"]) == 1
    assert timings["rounds"][0]["round"] == 1
    assert np.isfinite(timings["rounds"][0]["seconds"])
    assert timings["rounds"][0]["seconds"] >= 0.0
    assert timings["recombinations"] == []
    assert timings["average_round_seconds"] == timings["rounds"][0]["seconds"]
    assert timings["average_recombination_seconds"] is None
    random_timing_steps = [
        line.split(" step=", 1)[1].split(" ", 1)[0]
        for line in first.err.splitlines()
        if " step=select.random_multibranch" in line
    ]
    assert random_timing_steps == [
        "select.random_multibranch",
        "select.random_multibranch.round_average",
    ]
    assert {row["selection_phase"] for row in selected} == {
        "coverage_seed",
        "branch_final",
    }
    assert all("heap_refreshes" not in row for row in selected)
    select_manifest = json.loads((result / "manifest.json").read_text())
    run_manifest = json.loads((result / "run_manifest.json").read_text())
    assert select_manifest["algorithm"] == algorithm
    assert run_manifest["algorithm"] == algorithm
    assert report["selection_schema_version"] == 3
    assert select_manifest["selection_schema_version"] == 3
    assert run_manifest["selection_schema_version"] == 3
    assert validate_output(result, config=config) == {
        "status": "valid",
        "selected_clips": 10,
    }

    cached = run_pipeline(
        config,
        output_dir=root,
        visual_encoder=FailingCocoreVisualEncoder(),
    )
    cached_output = capsys.readouterr()
    assert cached == result
    assert cached_output.out == ""
    assert "cocore_timing" not in cached_output.err

    run_path = result / "run_manifest.json"
    run_manifest["selection_schema_version"] = 0
    run_path.write_text(json.dumps(run_manifest))
    with pytest.raises(ValueError, match="selection schema version"):
        validate_output(result, config=config)
    run_manifest["selection_schema_version"] = 3
    run_path.write_text(json.dumps(run_manifest))

    run_manifest["algorithm"] = {"type": "lazy_max_heap", "max_refreshes": 2}
    run_path.write_text(json.dumps(run_manifest))
    with pytest.raises(ValueError, match="algorithm"):
        validate_output(result, config=config)
    run_manifest["algorithm"] = algorithm
    run_path.write_text(json.dumps(run_manifest))

    selected_path = result / "selected_manifest.jsonl"
    selected[0]["heap_refreshes"] = None
    selected_path.write_text("\n".join(json.dumps(row) for row in selected) + "\n")
    with pytest.raises(ValueError, match="heap metadata"):
        validate_output(result, config=config)
    del selected[0]["heap_refreshes"]
    selected_path.write_text("\n".join(json.dumps(row) for row in selected) + "\n")

    all_path = result / "all_clips.parquet"
    all_rows = pq.read_table(all_path).to_pylist()
    all_rows[0]["heap_refreshes"] = None
    pq.write_table(pa.Table.from_pylist(all_rows), all_path)
    with pytest.raises(ValueError, match="heap metadata"):
        validate_output(result, config=config)
    for row in all_rows:
        row.pop("heap_refreshes", None)
    pq.write_table(pa.Table.from_pylist(all_rows), all_path)

    report["branch_search"]["rounds"] += 1
    (result / "selection_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="branch search"):
        validate_output(result, config=config)


def test_validate_rejects_invalid_random_multibranch_timings(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path, "sequence")
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    report_path = result / "selection_report.json"
    report = json.loads(report_path.read_text())
    valid = report["branch_search"]["timings"]

    corruptions: list[dict[str, object]] = []
    wrong_round = copy.deepcopy(valid)
    wrong_round["rounds"][0]["round"] = 2
    corruptions.append(wrong_round)
    boolean_seconds = copy.deepcopy(valid)
    boolean_seconds["rounds"][0]["seconds"] = True
    corruptions.append(boolean_seconds)
    negative_seconds = copy.deepcopy(valid)
    negative_seconds["rounds"][0]["seconds"] = -0.1
    corruptions.append(negative_seconds)
    nonfinite_seconds = copy.deepcopy(valid)
    nonfinite_seconds["rounds"][0]["seconds"] = float("nan")
    corruptions.append(nonfinite_seconds)
    wrong_average = copy.deepcopy(valid)
    wrong_average["average_round_seconds"] = float(valid["average_round_seconds"]) + 1.0
    corruptions.append(wrong_average)
    unexpected_recombination = copy.deepcopy(valid)
    unexpected_recombination["recombinations"] = [{"round": 20, "seconds": 0.1}]
    corruptions.append(unexpected_recombination)
    wrong_empty_average = copy.deepcopy(valid)
    wrong_empty_average["average_recombination_seconds"] = 0.0
    corruptions.append(wrong_empty_average)

    for corrupted in corruptions:
        report["branch_search"]["timings"] = corrupted
        report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="branch search timing"):
            validate_output(result, config=config)


def test_random_multibranch_reports_recombination_detail_and_average_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path, "sequence")
    config["selection"]["budget"] = 13
    monkeypatch.setattr("cocore.random_multibranch.BATCH_SIZE", 1)
    monkeypatch.setattr("cocore.random_multibranch.FIRST_RECOMBINATION_ROUND", 2)
    monkeypatch.setattr("cocore.random_multibranch.RECOMBINATION_INTERVAL", 10)
    monkeypatch.setattr("cocore.random_multibranch.COMMIT_SIZE", 1)
    monkeypatch.setattr("cocore.random_multibranch.RETAINED_SIZE", 1)

    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    captured = capsys.readouterr()
    timings = json.loads((result / "selection_report.json").read_text())["branch_search"]["timings"]

    assert len(timings["recombinations"]) == 1
    assert timings["recombinations"][0]["round"] == 2
    assert timings["recombinations"][0]["seconds"] >= 0.0
    assert timings["average_recombination_seconds"] == timings["recombinations"][0]["seconds"]
    random_timing_steps = [
        line.split(" step=", 1)[1].split(" ", 1)[0]
        for line in captured.err.splitlines()
        if " step=select.random_multibranch" in line
    ]
    assert random_timing_steps == [
        "select.random_multibranch",
        "select.random_multibranch.round_average",
        "select.random_multibranch.recombination_average",
    ]


def test_selection_schema_rebuilds_legacy_select_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path, "sequence")
    root = tmp_path / "random-multibranch-schema-upgrade"
    monkeypatch.setattr(
        cocore_pipeline,
        "SELECTION_SCHEMA_VERSION",
        0,
    )
    legacy_result = run_pipeline(
        config,
        output_dir=root,
        visual_encoder=CocoreVisualEncoder(),
    )
    legacy_run = json.loads((legacy_result / "run_manifest.json").read_text())
    legacy_fingerprint = json.loads((legacy_result / "manifest.json").read_text())["fingerprint"]
    assert "selection_schema_version" not in legacy_run

    monkeypatch.setattr(
        cocore_pipeline,
        "SELECTION_SCHEMA_VERSION",
        3,
    )
    upgraded_result = run_pipeline(
        config,
        output_dir=root,
        visual_encoder=FailingCocoreVisualEncoder(),
    )
    upgraded_run = json.loads((upgraded_result / "run_manifest.json").read_text())
    upgraded_fingerprint = json.loads((upgraded_result / "manifest.json").read_text())[
        "fingerprint"
    ]

    assert upgraded_result == legacy_result
    assert upgraded_fingerprint != legacy_fingerprint
    assert upgraded_run["stage_fingerprints"]["select"] == upgraded_fingerprint
    assert {
        stage: upgraded_run["stage_fingerprints"][stage] for stage in ("scan", "encode", "graph")
    } == {stage: legacy_run["stage_fingerprints"][stage] for stage in ("scan", "encode", "graph")}
    upgraded_report = json.loads((upgraded_result / "selection_report.json").read_text())
    upgraded_manifest = json.loads((upgraded_result / "manifest.json").read_text())
    assert upgraded_report["selection_schema_version"] == 3
    assert upgraded_manifest["selection_schema_version"] == 3
    assert upgraded_run["selection_schema_version"] == 3


def test_disabled_stop_bucket_excludes_unlabeled_candidates_from_graph_and_selection(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_mixed_stop", MixedStopCocorePipelineAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_pipeline_mixed_stop"
    config["prototypes"]["use_stop_bucket"] = False
    config["selection"]["budget"] = None
    config["selection"]["ratio"] = 0.5

    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    graph_root = result.parent / "graph-18-motion-hard-nearest-pca"

    source_indices = np.load(graph_root / "source_clip_indices.npy", allow_pickle=False)
    np.testing.assert_array_equal(source_indices, np.arange(82, dtype=np.int64))
    with np.load(graph_root / "nodes.npz") as nodes:
        assert len(nodes["prototype_indices"]) == 82
        assert np.all(np.any(nodes["prototype_indices"] >= 0, axis=1))
    assert np.load(graph_root / "half_action_labels.npy", allow_pickle=False).shape == (82, 2)

    catalog = json.loads((graph_root / "prototype_catalog.json").read_text())
    stop = next(
        category for category in catalog["action_categories"] if category["label"] == "stop"
    )
    assert catalog["use_stop_bucket"] is False
    assert stop["action_id"] is None
    assert stop["training_count"] == 0
    assert all(leaf["action_label"] != "stop" for leaf in catalog["leaf_prototypes"])

    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    report = json.loads((result / "selection_report.json").read_text())
    assert len(rows) == 82
    assert sum(row["selected"] for row in rows) == 41
    assert report["number_of_clips"] == 82
    assert report["number_of_scanned_clips"] == 84
    assert report["eligible_clips"] == 82
    assert report["excluded_unlabeled_clips"] == 2
    assert report["use_stop_bucket"] is False
    assert validate_output(result, config=config) == {"status": "valid", "selected_clips": 41}


def test_validate_rejects_tampered_source_clip_indices(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_mixed_stop", MixedStopCocorePipelineAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_pipeline_mixed_stop"
    config["prototypes"]["use_stop_bucket"] = False
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-18-motion-hard-nearest-pca" / "source_clip_indices.npy"
    source_indices = np.load(path, allow_pickle=False)
    source_indices[-1] = 82
    np.save(path, source_indices)

    with pytest.raises(ValueError, match="source clip indices do not match prototype replay"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_graph_candidate_counts(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-18-motion-hard-nearest-pca" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["excluded_unlabeled_nodes"] = 1
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="graph manifest candidate counts"):
        validate_output(result, config=config)


def test_disabled_stop_bucket_rejects_budget_above_eligible_candidate_count(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_mixed_stop", MixedStopCocorePipelineAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_pipeline_mixed_stop"
    config["prototypes"]["use_stop_bucket"] = False
    config["selection"]["budget"] = 83

    with pytest.raises(
        ValueError, match="selection budget must be within eligible candidate count"
    ):
        run_pipeline(config, visual_encoder=CocoreVisualEncoder())


def test_run_pipeline_reports_all_completed_timings_and_cached_run_is_silent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)

    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())

    first = capsys.readouterr()
    assert first.out == ""
    timing_lines = [line for line in first.err.splitlines() if line.startswith("cocore_timing")]
    expected_steps = [
        "scan",
        "encode.numeric_normalization",
        "encode.visual_cache",
        "encode.pca_fusion",
        "encode",
        "graph.reliability",
        "graph.prototypes.action_scan",
        "graph.prototypes.training_data",
        "graph.prototypes.kmeans",
        "graph.prototypes.center_statistics",
        "graph.prototypes.candidate_assignment",
        "graph.prototypes",
        "graph.sparse_graph",
        "graph",
        "select.context",
        "select.coverage_seed",
        "select.random_multibranch",
        "select.random_multibranch.round_average",
        "select.export",
        "select",
    ]
    assert len(timing_lines) == len(expected_steps)
    for line, expected_step in zip(timing_lines, expected_steps, strict=True):
        match = re.fullmatch(
            r"cocore_timing step=([^ ]+) seconds=([0-9]+\.[0-9]{6}) status=completed",
            line,
        )
        assert match is not None
        assert match.group(1) == expected_step
        assert float(match.group(2)) >= 0.0

    cached = run_pipeline(config, visual_encoder=FailingCocoreVisualEncoder())

    second = capsys.readouterr()
    assert cached == result
    assert second.out == ""
    assert "cocore_timing" not in second.err


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
    nodes_path = result.parent / "graph-18-motion-hard-nearest-pca" / "nodes.npz"
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
    catalog_path = result.parent / "graph-18-motion-hard-nearest-pca" / "prototype_catalog.json"
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
    catalog_path = result.parent / "graph-18-motion-hard-nearest-pca" / "prototype_catalog.json"
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
    (result.parent / "graph-18-motion-hard-nearest-pca" / "prototype_centers.npy").unlink()

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


def test_validate_rejects_tampered_action_variation_cache(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "encode" / "action_variation.npy"
    values = np.load(path)
    values[0] = 1.0 - values[0]
    np.save(path, values)

    with pytest.raises(ValueError, match="action variation cache"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_graph_action_variation(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-18-motion-hard-nearest-pca" / "nodes.npz"
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["action_variation"] = arrays["action_variation"].copy()
    arrays["action_variation"][0] = 1.0 - arrays["action_variation"][0]
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="graph node action variation"):
        validate_output(result, config=config)


def test_validate_rejects_missing_graph_action_variation_field(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-18-motion-hard-nearest-pca" / "nodes.npz"
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files if name != "action_variation"}
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="graph node reliability arrays"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_all_clips_action_variation(tmp_path: Path) -> None:
    import pyarrow as pa

    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result / "all_clips.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["action_variation"] = 1.0 - rows[0]["action_variation"]
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(ValueError, match="hierarchical prototype row"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_selected_action_variation(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result / "selected_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["action_variation"] = 1.0 - rows[0]["action_variation"]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="selected action variation"):
        validate_output(result, config=config)


@pytest.mark.parametrize(
    "filename",
    ["visual_action_consistency_raw.npy", "visual_action_consistency.npy"],
)
def test_validate_rejects_tampered_visual_action_consistency_cache(
    tmp_path: Path,
    filename: str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "encode" / filename
    values = np.load(path)
    values[0] = values[0] * np.float32(2.0) + np.float32(1.0)
    np.save(path, values)

    with pytest.raises(ValueError, match="visual-action consistency cache"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_graph_visual_action_consistency(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-18-motion-hard-nearest-pca" / "nodes.npz"
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["visual_action_consistency"] = arrays["visual_action_consistency"].copy()
    arrays["visual_action_consistency"][0] = np.float32(
        0.25 if arrays["visual_action_consistency"][0] > 0.5 else 0.75
    )
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="graph node visual-action consistency"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_all_clips_visual_action_consistency(
    tmp_path: Path,
) -> None:
    import pyarrow as pa

    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result / "all_clips.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["visual_action_consistency"] += 0.125
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(ValueError, match="hierarchical prototype row"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_selected_visual_action_consistency(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result / "selected_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["visual_action_consistency"] += 0.125
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="visual-action consistency"):
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
    assert not (root / "graph-18-motion-hard-nearest-pca").exists()


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
    assert steps[-9:] == [
        "graph.reliability",
        "graph.prototypes.action_scan",
        "graph.prototypes.training_data",
        "graph.prototypes.kmeans",
        "graph.prototypes.center_statistics",
        "graph.prototypes.candidate_assignment",
        "graph.prototypes",
        "graph.sparse_graph",
        "graph",
    ]


def test_graph_and_validation_honor_single_prototype_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["prototypes"]["num_threads"] = 1  # type: ignore[index]
    thread_counts: list[int] = []
    tolerances: list[float] = []
    real_builder = cocore_pipeline.build_hierarchical_motion_prototypes

    def recording_builder(*args: object, **kwargs: object):
        thread_counts.append(int(kwargs["num_threads"]))
        tolerances.append(float(kwargs["tol"]))
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(
        cocore_pipeline,
        "build_hierarchical_motion_prototypes",
        recording_builder,
    )

    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())

    assert thread_counts == [1]
    assert tolerances == [1.0e-4]
    thread_counts.clear()
    tolerances.clear()

    assert validate_output(result, config=config) == {"status": "valid", "selected_clips": 10}
    assert thread_counts == [1]
    assert tolerances == [1.0e-4]


def test_graph_fingerprint_ignores_prototype_thread_count(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["prototypes"]["num_threads"] = 1  # type: ignore[index]

    first = graph_stage(config, visual_encoder=CocoreVisualEncoder())
    config["prototypes"]["num_threads"] = 4  # type: ignore[index]
    second = graph_stage(config, visual_encoder=FailingCocoreVisualEncoder())

    assert first[0] == second[0]
    assert first[4] == second[4]


def test_graph_fingerprint_includes_prototype_convergence_tolerance(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    graph_stage(config, visual_encoder=CocoreVisualEncoder())
    config["prototypes"]["tol"] = 2.0e-4  # type: ignore[index]

    with pytest.raises(FileExistsError, match="--force"):
        graph_stage(config, visual_encoder=FailingCocoreVisualEncoder())


def test_graph_fingerprint_includes_motion_primitive_profile(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    first = graph_stage(config, visual_encoder=CocoreVisualEncoder())
    config["prototypes"]["profile"] = "bridge_v2"  # type: ignore[index]

    with pytest.raises(FileExistsError, match="--force"):
        graph_stage(config, visual_encoder=FailingCocoreVisualEncoder())

    second = graph_stage(
        config,
        force=True,
        visual_encoder=CocoreVisualEncoder(),
    )
    assert first[4] != second[4]


def test_bridge_action_contract_rebuilds_only_graph_and_select_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["prototypes"]["profile"] = "bridge_v2"  # type: ignore[index]
    monkeypatch.setattr(cocore_prototypes, "MIN_ACTION_FREQUENCY", 0.001)

    legacy_result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    root = legacy_result.parent
    stage_directories = {
        "scan": root / "scan",
        "encode": root / "encode",
        "graph": root / "graph-18-motion-hard-nearest-pca",
        "select": legacy_result,
    }
    legacy_fingerprints = {
        stage: json.loads((directory / "manifest.json").read_text())["fingerprint"]
        for stage, directory in stage_directories.items()
    }
    monkeypatch.setattr(cocore_prototypes, "MIN_ACTION_FREQUENCY", 0.005)

    with pytest.raises(FileExistsError, match="--force"):
        run_pipeline(config, visual_encoder=FailingCocoreVisualEncoder())

    rebuilt_result = run_pipeline(
        config,
        force=True,
        visual_encoder=FailingCocoreVisualEncoder(),
    )
    rebuilt_fingerprints = {
        stage: json.loads((directory / "manifest.json").read_text())["fingerprint"]
        for stage, directory in stage_directories.items()
    }

    assert rebuilt_result == legacy_result
    assert rebuilt_fingerprints["scan"] == legacy_fingerprints["scan"]
    assert rebuilt_fingerprints["encode"] == legacy_fingerprints["encode"]
    assert rebuilt_fingerprints["graph"] != legacy_fingerprints["graph"]
    assert rebuilt_fingerprints["select"] != legacy_fingerprints["select"]
    assert validate_output(rebuilt_result, config=config) == {
        "status": "valid",
        "selected_clips": 10,
    }


def test_validate_rejects_tampered_visual_prototype_center(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-18-motion-hard-nearest-pca" / "prototype_centers.npy"
    centers = np.load(path)
    centers[0] = np.zeros(128, dtype=np.float32)
    centers[0, 0] = 1.0
    np.save(path, centers)

    with pytest.raises(ValueError, match="prototype (center|replay)"):
        validate_output(result, config=config)


def test_validate_rejects_incompatible_cropped_pca_width(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "encode" / "visual_pca.npz"
    with np.load(path, allow_pickle=False) as stored:
        payload = {name: stored[name].copy() for name in stored.files}
    payload["components"] = payload["components"][:, :-1]
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="twice the frame dimension"):
        validate_output(result, config=config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prototype_visual_dim", 64),
        ("prototype_visual_projection", "centered PCA transform"),
        ("prototype_visual_normalization", "per-frame L2"),
    ],
)
def test_validate_rejects_tampered_graph_projection_contract(
    tmp_path: Path,
    field: str,
    value: int | str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-18-motion-hard-nearest-pca" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="graph manifest prototype schema"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_half_action_labels(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    path = result.parent / "graph-18-motion-hard-nearest-pca" / "half_action_labels.npy"
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


def test_validate_rejects_schema_nine_manifest_explicitly(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    run_path = result / "run_manifest.json"
    manifest = json.loads(run_path.read_text())
    manifest["prototype_schema_version"] = 9
    run_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="prototype schema version is incompatible"):
        validate_output(result, config=config)


def test_validate_rejects_schema_nine_graph_catalog_explicitly(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    catalog_path = result.parent / "graph-18-motion-hard-nearest-pca" / "prototype_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["schema_version"] = 9
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="catalog schema"):
        validate_output(result, config=config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("primitive_thresholds", {"translation": 9.0}),
        ("cyclic_axes", [5]),
    ],
)
def test_validate_rejects_tampered_motion_primitive_catalog_contract(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    catalog_path = result.parent / "graph-18-motion-hard-nearest-pca" / "prototype_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["constants"][field] = value
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="catalog schema"):
        validate_output(result, config=config)


def test_validate_rejects_motion_primitive_profile_config_mismatch(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    mismatched = copy.deepcopy(config)
    mismatched["prototypes"]["profile"] = "bridge_v2"  # type: ignore[index]

    with pytest.raises(ValueError, match="motion primitive configuration|action_jump cache contract mismatch"):
        validate_output(result, config=mismatched)


@pytest.mark.parametrize(
    ("stage", "directory"),
    [
        ("scan", "scan"),
        ("encode", "encode"),
        ("graph", "graph-18-motion-hard-nearest-pca"),
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


@pytest.mark.parametrize(
    ("directory", "field", "value", "message"),
    [
        ("scan", "window_policy", "legacy_stride", "window policy"),
        ("encode", "window_policy", "legacy_stride", "window policy"),
        (
            "graph-18-motion-hard-nearest-pca",
            "sequence_adjacency",
            "contiguous",
            "graph manifest prototype schema",
        ),
    ],
)
def test_validate_rejects_tampered_window_and_sequence_policies(
    tmp_path: Path,
    directory: str,
    field: str,
    value: str,
    message: str,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    manifest_path = result.parent / directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        validate_output(result, config=config)


def test_validate_rejects_scan_clips_that_do_not_match_uniform_replay(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    clips_path = result.parent / "scan" / "clips.parquet"
    rows = pq.read_table(clips_path).to_pylist()
    rows[1]["start_step"] += 1
    rows[1]["end_step"] += 1
    pq.write_table(pa.Table.from_pylist(rows), clips_path)

    with pytest.raises(ValueError, match="near-uniform candidate windows"):
        validate_output(result, config=config)


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

    assert cooccurrence == root / "select-cooccurrence-w1-top50pct-random-multibranch"
    assert sequence == root / "select-sequence-w1-top50pct-random-multibranch"
    assert cooccurrence.is_dir()
    assert sequence.is_dir()
    assert validate_output(cooccurrence, config=_config(tmp_path, "cooccurrence")) == {
        "status": "valid",
        "selected_clips": 10,
    }
    assert validate_output(sequence, config=_config(tmp_path, "sequence")) == {
        "status": "valid",
        "selected_clips": 10,
    }


def test_validate_aligns_sorted_output_rows_by_sample_id(tmp_path: Path) -> None:
    register_dataset_adapter("cocore_pipeline_reverse_synthetic", ReverseOrderCocorePipelineAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_pipeline_reverse_synthetic"

    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())

    assert validate_output(result, config=config) == {"status": "valid", "selected_clips": 10}


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_dwell_pipeline_cache_fusion_and_replay(tmp_path: Path, profile: str) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["prototypes"]["profile"] = profile
    config["dwell"] = dict(
        position_speed_threshold=0.5, gripper_speed_threshold=0.1, angular_speed_threshold=0.1
    )
    config["reliability_metrics"] = ["non_dwell"]
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    root = result.parent
    ratio = np.load(root / "encode" / "dwell_ratio.npy")
    np.testing.assert_allclose(ratio, 1.0)
    with np.load(root / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz") as nodes:
        np.testing.assert_allclose(nodes["non_dwell"], 0.0)
        np.testing.assert_allclose(nodes["reliability"], 0.05)
    validate_output(result, config=config)
    encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())
    changed = copy.deepcopy(config)
    changed["dwell"]["position_speed_threshold"] = 0.3
    with pytest.raises(FileExistsError):
        encode_stage(changed, visual_encoder=FailingCocoreVisualEncoder())
    path = root / "encode" / "dwell_ratio.npy"
    ratio[0] = 0.0
    np.save(path, ratio)
    with pytest.raises(ValueError, match="dwell"):
        validate_output(result, config=config)


@pytest.mark.parametrize(
    "target", ["raw_state", "timestamps", "graph", "all_rows", "selected_rows", "report"]
)
def test_dwell_diagnostics_and_tamper_detection(tmp_path: Path, target: str) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["dwell"] = dict(
        position_speed_threshold=0.5, gripper_speed_threshold=0.1, angular_speed_threshold=0.1
    )
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert all(row["dwell_ratio"] == 1 and row["non_dwell"] == 0 for row in rows)
    report_path = result / "selection_report.json"
    report = json.loads(report_path.read_text())
    assert report["dwell_summary"]["dwell_ratio"]["all_mean"] == 1
    assert report["dwell_summary"]["non_dwell"]["selected_mean"] == 0
    assert "non_dwell" not in report["reliability_metrics"]
    validate_output(result, config=config)
    if target in ("raw_state", "timestamps"):
        field = "dwell_state_sequences" if target == "raw_state" else "dwell_timestamps"
        path = result.parent / "encode" / f"{field}.npy"
        values = np.load(path)
        values.flat[0] += 0.01
        np.save(path, values)
    elif target == "graph":
        path = result.parent / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz"
        with np.load(path) as nodes:
            values = dict(nodes)
        values["non_dwell"][0] = 1
        np.savez(path, **values)
    elif target == "all_rows":
        rows[0]["dwell_ratio"] = 0
        pq.write_table(pa.Table.from_pylist(rows), result / "all_clips.parquet")
    elif target == "selected_rows":
        path = result / "selected_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["non_dwell"] = 1
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        report["dwell_summary"]["dwell_ratio"]["all_mean"] = 0
        report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_output(result, config=config)


def test_dwell_cache_replay_recomputes_after_checksum_update(tmp_path):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["dwell"] = dict(
        position_speed_threshold=0.5, angular_speed_threshold=0.1, gripper_mode="binary"
    )
    root, _, artifact = encode_stage(config, visual_encoder=CocoreVisualEncoder())
    path = root / "encode" / "dwell_ratio.npy"
    ratio = np.load(path)
    ratio[0] = 0.5
    np.save(path, ratio)
    manifest_path = root / "encode" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dwell_checksums"]["dwell_ratio"] = cocore_pipeline.file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dwell cache does not match"):
        cocore_pipeline._validate_dwell_cache(
            root / "encode", artifact.clips, cocore_pipeline.resolve_config(config)
        )


def test_dwell_diagnostics_preserve_default_scores_and_selection(tmp_path):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    baseline = run_pipeline(
        config, output_dir=tmp_path / "baseline", visual_encoder=CocoreVisualEncoder()
    )
    config["dwell"] = dict(
        position_speed_threshold=0.5, angular_speed_threshold=0.1, gripper_mode="binary"
    )
    diagnostic = run_pipeline(
        config, output_dir=tmp_path / "diagnostic", visual_encoder=CocoreVisualEncoder()
    )
    baseline_rows = pq.read_table(baseline / "all_clips.parquet").to_pylist()
    diagnostic_rows = pq.read_table(diagnostic / "all_clips.parquet").to_pylist()
    for before, after in zip(baseline_rows, diagnostic_rows, strict=True):
        assert "dwell_ratio" not in before
        assert {k: v for k, v in after.items() if k not in ("dwell_ratio", "non_dwell")} == before


class IrregularJerkAdapter(CocorePipelineAdapter):
    def iter_episodes(self, **kwargs):
        for episode in super().iter_episodes(**kwargs):
            episode.timestamps[25] += 0.02
            yield episode


def test_eef_jerk_filter_cache_and_replay(tmp_path):
    register_dataset_adapter("cocore_jerk_irregular", IrregularJerkAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_jerk_irregular"
    config["reliability_metrics"] = ["support", "eef_jerk"]
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    root = result.parent
    valid = np.load(root / "encode/eef_jerk_valid.npy")
    assert valid.any() and not valid.all()
    indices = np.load(root / cocore_pipeline.GRAPH_DIRECTORY / "source_clip_indices.npy")
    np.testing.assert_array_equal(indices, np.flatnonzero(valid))
    report = json.loads((result / "selection_report.json").read_text())
    assert report["excluded_unlabeled_clips"] == 0
    assert report["excluded_jerk_clips"] == int((~valid).sum())
    assert report["excluded_clips"] == int((~valid).sum())
    assert report["eef_jerk_summary"]["invalid_clips"] == int((~valid).sum())
    excluded = json.loads((result / "excluded_clips.json").read_text())
    assert len(excluded) == int((~valid).sum())
    assert all(r["eef_jerk_raw"] is None for r in excluded)
    assert validate_output(result, config=config)["status"] == "valid"
    assert run_pipeline(config, visual_encoder=FailingCocoreVisualEncoder()) == result
    path = root / "encode/eef_jerk_raw.npy"
    raw = np.load(path)
    raw[valid] += 1
    np.save(path, raw)
    with pytest.raises(ValueError, match="eef_jerk"):
        validate_output(result, config=config)


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
@pytest.mark.parametrize("delta_path,expected", [(0.001, 1.0), (100.0, None)])
def test_local_path_pipeline_cache_fusion_and_replay(tmp_path, profile, delta_path, expected):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["prototypes"]["profile"] = profile
    config["local_path_efficiency"] = {"delta_path": delta_path}
    config["reliability_metrics"] = ["local_path_efficiency"]
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    root = result.parent
    scores = np.load(root / "encode" / "local_path_efficiency.npy")
    if expected is None:
        assert np.isnan(scores).all()
    else:
        np.testing.assert_allclose(scores, expected)
    with np.load(root / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz") as nodes:
        np.testing.assert_allclose(nodes["reliability"], 1.0)
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert all(row["local_path_efficiency"] == expected for row in rows)
    assert validate_output(result, config=config)["status"] == "valid"
    encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())
    changed = copy.deepcopy(config)
    changed["local_path_efficiency"]["delta_path"] *= 2
    with pytest.raises(FileExistsError):
        encode_stage(changed, visual_encoder=FailingCocoreVisualEncoder())
    path = root / "encode" / "local_path_efficiency.npy"
    scores[0] = 0.4
    np.save(path, scores)
    manifest_path = root / "encode" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["path_checksums"]["local_path_efficiency"] = cocore_pipeline.file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="local_path_efficiency"):
        validate_output(result, config=config)


class AllInvalidJerkAdapter(CocorePipelineAdapter):
    def iter_episodes(self, **kwargs):
        for episode in super().iter_episodes(**kwargs):
            episode.timestamps[1::2] += 0.02
            yield episode


class MixedStopJerkAdapter(MixedStopCocorePipelineAdapter):
    def iter_episodes(self, **kwargs):
        for episode in super().iter_episodes(**kwargs):
            episode.timestamps[25] += 0.02
            yield episode


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_eef_jerk_only_and_dwell_compatibility(tmp_path, profile):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["prototypes"]["profile"] = profile
    config["reliability_metrics"] = ["eef_jerk"]
    config["dwell"] = dict(
        position_speed_threshold=0.1, angular_speed_threshold=0.1, gripper_speed_threshold=0.1
    )
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    with np.load(result.parent / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz") as nodes:
        np.testing.assert_allclose(nodes["reliability"], np.clip(nodes["eef_jerk"], 0.05, 1))
        assert "dwell_ratio" in nodes
    assert validate_output(result)["status"] == "valid"


def test_eef_jerk_exclusion_union_and_no_gap_edges(tmp_path):
    register_dataset_adapter("cocore_jerk_mixed", MixedStopJerkAdapter)
    config = _config(tmp_path, "sequence")
    config["dataset"]["type"] = "cocore_jerk_mixed"
    config["prototypes"]["use_stop_bucket"] = False
    config["reliability_metrics"] = ["eef_jerk"]
    config["selection"] = {"ratio": 0.5, "budget": None}
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    graph_root = result.parent / cocore_pipeline.GRAPH_DIRECTORY
    valid = np.load(result.parent / "encode/eef_jerk_valid.npy")
    prototype = np.load(graph_root / "prototype_eligible_mask.npy")
    assert ((~valid) & (~prototype)).any()
    indices = np.load(graph_root / "source_clip_indices.npy")
    np.testing.assert_array_equal(indices, np.flatnonzero(valid & prototype))
    with np.load(graph_root / "sequence_edges.npz") as edges:
        # Original near-uniform candidates are adjacent by source index; no gap bridging.
        assert np.all(indices[edges["target"]] - indices[edges["source"]] == 1)
    report = json.loads((result / "selection_report.json").read_text())
    assert report["excluded_unlabeled_clips"] == int((~prototype).sum())
    assert report["excluded_jerk_clips"] == int((~valid).sum())
    assert report["excluded_clips"] == int((~(valid & prototype)).sum())
    assert report["selected_clips"] == int(np.floor(len(indices) * 0.5 + 0.5))
    assert validate_output(result, config=config)["status"] == "valid"


def test_eef_jerk_all_invalid_and_budget_overflow(tmp_path):
    register_dataset_adapter("cocore_jerk_all_invalid", AllInvalidJerkAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_jerk_all_invalid"
    config["reliability_metrics"] = ["eef_jerk"]
    with pytest.raises(ValueError, match="no eligible candidate"):
        run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    register_dataset_adapter("cocore_jerk_irregular", IrregularJerkAdapter)
    config["dataset"]["type"] = "cocore_jerk_irregular"
    config["selection"]["budget"] = 82
    with pytest.raises(ValueError, match="budget"):
        run_pipeline(config, output_dir=tmp_path / "budget", visual_encoder=CocoreVisualEncoder())


@pytest.mark.parametrize(
    "target",
    ["contract", "missing", "recomputed_raw", "exclusions", "node", "mask", "missing_mask"],
)
def test_eef_jerk_rejects_corrupt_artifacts(tmp_path, target):
    from relcore.utils.io import file_sha256

    register_dataset_adapter("cocore_jerk_irregular", IrregularJerkAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_jerk_irregular"
    config["reliability_metrics"] = ["eef_jerk"]
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    encode = result.parent / "encode"
    graph = result.parent / cocore_pipeline.GRAPH_DIRECTORY
    if target == "contract":
        path = result / "run_manifest.json"
        data = json.loads(path.read_text())
        data["eef_jerk"]["dt"] = "fake"
        path.write_text(json.dumps(data))
    elif target == "missing":
        (encode / "eef_jerk_timestamps.npy").unlink()
    elif target == "recomputed_raw":
        path = encode / "eef_jerk_raw.npy"
        raw = np.load(path)
        raw[np.isfinite(raw)] += 1
        np.save(path, raw)
        manifest = encode / "manifest.json"
        data = json.loads(manifest.read_text())
        data["eef_jerk_checksums"]["eef_jerk_raw"] = file_sha256(path)
        manifest.write_text(json.dumps(data))
    elif target == "exclusions":
        (result / "excluded_clips.json").write_text("[]")
    elif target == "missing_mask":
        (graph / "prototype_eligible_mask.npy").unlink()
    elif target == "mask":
        path = graph / "prototype_eligible_mask.npy"
        mask = np.load(path)
        mask[:] = False
        np.save(path, mask)
    else:
        path = graph / "nodes.npz"
        with np.load(path) as nodes:
            arrays = {name: nodes[name] for name in nodes.files}
        arrays["eef_jerk_raw"] += 1
        np.savez(path, **arrays)
    with pytest.raises(ValueError):
        validate_output(result, config=config)


@pytest.mark.parametrize(
    "target", ["raw_positions", "graph", "all_rows", "selected_rows", "report"]
)
def test_local_path_diagnostics_and_tamper_detection(tmp_path, target):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["local_path_efficiency"] = {"delta_path": 100.0}
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    report_path = result / "selection_report.json"
    report = json.loads(report_path.read_text())
    assert report["local_path_efficiency_summary"]["all_mean"] is None
    assert report["local_path_efficiency_summary"]["all_invalid_count"] == len(rows)
    assert "local_path_efficiency" not in report["reliability_metrics"]
    validate_output(result, config=config)
    if target == "raw_positions":
        path = result.parent / "encode" / "path_position_sequences.npy"
        values = np.load(path)
        values.flat[0] += 0.1
        np.save(path, values)
    elif target == "graph":
        path = result.parent / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz"
        with np.load(path) as nodes:
            values = dict(nodes)
        values["local_path_efficiency"][0] = 0.5
        np.savez(path, **values)
    elif target == "all_rows":
        rows[0]["local_path_efficiency"] = 0.5
        pq.write_table(pa.Table.from_pylist(rows), result / "all_clips.parquet")
    elif target == "selected_rows":
        path = result / "selected_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["local_path_efficiency"] = 0.5
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        report["local_path_efficiency_summary"]["all_mean"] = 0.5
        report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="local_path_efficiency"):
        validate_output(result, config=config)


@pytest.mark.parametrize("metrics", [None, ["support", "eef_jerk"]])
def test_local_path_diagnostics_preserve_default_selection(tmp_path, metrics):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    if metrics is not None:
        config["reliability_metrics"] = metrics
    baseline = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    before = pq.read_table(baseline / "all_clips.parquet").to_pylist()
    config["output"]["directory"] = str(tmp_path / "diagnostic")
    config["local_path_efficiency"] = {"delta_path": 100.0}
    diagnostic = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    after = pq.read_table(diagnostic / "all_clips.parquet").to_pylist()
    assert [
        {k: v for k, v in row.items() if k != "local_path_efficiency"} for row in after
    ] == before


def test_eef_jerk_with_local_path_efficiency(tmp_path):
    register_dataset_adapter("cocore_jerk_irregular", IrregularJerkAdapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = "cocore_jerk_irregular"
    config["reliability_metrics"] = ["eef_jerk", "local_path_efficiency"]
    config["local_path_efficiency"] = {"delta_path": 100.0}
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    with np.load(result.parent / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz") as nodes:
        assert np.isnan(nodes["local_path_efficiency"]).all()
        np.testing.assert_allclose(nodes["reliability"], np.clip(nodes["eef_jerk"], 0.05, 1))
    assert validate_output(result, config=config)["status"] == "valid"


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        ["low_high_frequency_jitter"],
        ["eef_jerk", "local_path_efficiency", "low_high_frequency_jitter"],
    ],
)
def test_high_frequency_pipeline_roundtrip(tmp_path, metrics):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["high_frequency_jitter"] = dict(
        cutoff_hz=2.0, noise_floor_rms=0.01, max_frequency_resolution_hz=2.0
    )
    if metrics is not None:
        config["reliability_metrics"] = metrics
    if metrics and "local_path_efficiency" in metrics:
        config["local_path_efficiency"] = {"delta_path": 100.0}
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert rows and all(r["low_high_frequency_jitter"] is None for r in rows)
    assert all(r["high_frequency_reason"] == "low_fluctuation" for r in rows)
    if metrics == ["low_high_frequency_jitter"]:
        assert all(r["reliability"] == 1.0 for r in rows)
    report = json.loads((result / "selection_report.json").read_text())
    assert report["high_frequency_jitter_summary"]["graph"]["invalid_count"] == len(rows)
    assert validate_output(result, config=config)["status"] == "valid"


class HighFrequencyAdapter(CocorePipelineAdapter):
    def iter_episodes(self, **kwargs):
        for episode in super().iter_episodes(**kwargs):
            episode.observations["observation.state"][:, 1] = 0.01 * np.cos(
                2 * np.pi * 3 * episode.timestamps
            )
            yield episode


class HfCompatibleIrregularJerkAdapter(HighFrequencyAdapter):
    def iter_episodes(self, **kwargs):
        for episode in super().iter_episodes(**kwargs):
            # Accepted by HF's 1e-3 tolerance, rejected by jerk's 1e-4 tolerance.
            episode.timestamps[25] += 0.00005
            yield episode


def _hf_config(tmp_path):
    config = _config(tmp_path)
    config["high_frequency_jitter"] = dict(
        cutoff_hz=2.0, noise_floor_rms=1e-5, max_frequency_resolution_hz=2.0
    )
    return config


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_hf_valid_profile_and_jerk_mapping(tmp_path, profile):
    from cocore.high_frequency_jitter import HF_FIELDS

    register_dataset_adapter("hf_irregular_jerk", HfCompatibleIrregularJerkAdapter)
    config = _hf_config(tmp_path)
    config["dataset"]["type"] = "hf_irregular_jerk"
    config["prototypes"]["profile"] = profile
    config["reliability_metrics"] = ["eef_jerk", "low_high_frequency_jitter"]
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    root = result.parent
    indices = np.load(root / cocore_pipeline.GRAPH_DIRECTORY / "source_clip_indices.npy")
    with np.load(root / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz") as nodes:
        assert nodes["high_frequency_valid"].all()
        for field in HF_FIELDS:
            np.testing.assert_array_equal(
                nodes[field], np.load(root / "encode" / f"{field}.npy")[indices]
            )
        np.testing.assert_allclose(
            nodes["reliability"],
            np.clip(np.sqrt(nodes["eef_jerk"] * nodes["low_high_frequency_jitter"]), 0.05, 1),
            rtol=1e-6,
        )
    report = json.loads((result / "selection_report.json").read_text())
    assert report["excluded_jerk_clips"] > 0
    counts = report["high_frequency_jitter_summary"]
    assert counts["scanned"]["valid_count"] > counts["graph"]["valid_count"]
    assert validate_output(result, config=config)["status"] == "valid"


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_hf_diagnostics_preserve_selection(tmp_path, profile):
    from cocore.high_frequency_jitter import HF_FIELDS

    register_dataset_adapter("cocore_pipeline_synthetic", HighFrequencyAdapter)
    config = _config(tmp_path)
    config["prototypes"]["profile"] = profile
    baseline = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    before = pq.read_table(baseline / "all_clips.parquet").to_pylist()
    config["output"]["directory"] = str(tmp_path / "hf_diagnostics")
    config["high_frequency_jitter"] = _hf_config(tmp_path)["high_frequency_jitter"]
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    after = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert [{k: v for k, v in row.items() if k not in HF_FIELDS} for row in after] == before
    assert validate_output(result, config=config)["status"] == "valid"


@pytest.mark.parametrize(
    "target",
    [
        "input",
        "overlap",
        "result",
        "validity",
        "reason",
        "node",
        "row",
        "selected",
        "summary",
        "contract",
    ],
)
def test_hf_tamper_detection(tmp_path, target):
    register_dataset_adapter("cocore_pipeline_synthetic", HighFrequencyAdapter)
    config = _hf_config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    encode = result.parent / "encode"
    if target in {"input", "overlap", "result", "validity", "reason"}:
        field = {
            "input": "high_frequency_positions",
            "overlap": "high_frequency_positions",
            "result": "high_frequency_ratio",
            "validity": "high_frequency_valid",
            "reason": "high_frequency_reason",
        }[target]
        path = encode / f"{field}.npy"
        values = np.load(path)
        if target in {"input", "overlap"}:
            values[0, -1, 0] += 0.01
        elif target == "validity":
            values[0] = not values[0]
        elif target == "reason":
            values[0] = "fake"
        else:
            values[0] *= 0.5
        np.save(path, values)
        if target != "input":
            manifest_path = encode / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["high_frequency_checksums"][field] = cocore_pipeline.file_sha256(path)
            manifest_path.write_text(json.dumps(manifest))
    elif target == "node":
        path = result.parent / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz"
        with np.load(path) as nodes:
            arrays = dict(nodes)
        arrays["high_frequency_ratio"][0] *= 0.5
        np.savez(path, **arrays)
    elif target == "row":
        path = result / "all_clips.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["high_frequency_ratio"] *= 0.5
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif target == "selected":
        path = result / "selected_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["high_frequency_reason"] = "fake"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        path = result / ("selection_report.json" if target == "summary" else "run_manifest.json")
        data = json.loads(path.read_text())
        if target == "summary":
            data["high_frequency_jitter_summary"]["scanned"]["valid_count"] += 1
        else:
            data["high_frequency_jitter"]["window"] = "boxcar"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="high_frequency_jitter"):
        validate_output(result, config=config)


@pytest.mark.parametrize(
    "parameter", ["cutoff_hz", "noise_floor_rms", "max_frequency_resolution_hz", "epsilon"]
)
def test_hf_cache_parameters_invalidate(tmp_path, parameter):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _hf_config(tmp_path)
    root, _, encoded = encode_stage(config, visual_encoder=CocoreVisualEncoder())
    resumed = encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())[2]
    assert resumed.fingerprint == encoded.fingerprint
    changed = copy.deepcopy(config)
    changed["high_frequency_jitter"][parameter] = (
        config["high_frequency_jitter"].get(parameter, 1e-12) * 0.9
    )
    with pytest.raises(FileExistsError):
        encode_stage(changed, visual_encoder=FailingCocoreVisualEncoder())


def test_hf_missing_cache_requires_force(tmp_path):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _hf_config(tmp_path)
    root, _, _ = encode_stage(config, visual_encoder=CocoreVisualEncoder())
    (root / "encode" / "high_frequency_ratio.npy").unlink()
    with pytest.raises(FileExistsError):
        encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())


@pytest.mark.parametrize("problem", ["missing", "nonfinite", "shape"])
def test_hf_raw_input_error_identifies_clip(tmp_path, problem):
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
    with pytest.raises(ValueError, match="high_frequency_jitter clip ep000000_fragment_"):
        encode_stage(_hf_config(tmp_path), visual_encoder=CocoreVisualEncoder())


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_action_jump_pipeline_and_jerk_mapping(tmp_path, profile):
    register_dataset_adapter("jump_jerk", HfCompatibleIrregularJerkAdapter)
    config = _hf_config(tmp_path)
    config["dataset"]["type"] = "jump_jerk"
    config["prototypes"]["profile"] = profile
    config["reliability_metrics"] = ["action_jump", "eef_jerk", "low_high_frequency_jitter"]
    config["action_jump"] = {"threshold": 0.005}
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    root = result.parent
    indices = np.load(root / cocore_pipeline.GRAPH_DIRECTORY / "source_clip_indices.npy")
    with np.load(root / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz") as nodes:
        for field in ("action_jump_rate", "action_jump"):
            np.testing.assert_array_equal(
                nodes[field], np.load(root / "encode" / f"{field}.npy")[indices]
            )
        expected = (
            nodes["action_jump"] * nodes["eef_jerk"] * nodes["low_high_frequency_jitter"]
        ) ** (1 / 3)
        np.testing.assert_allclose(nodes["reliability"], np.clip(expected, 0.05, 1), rtol=1e-6)
    assert validate_output(result, config=config)["status"] == "valid"


def test_action_jump_default_cache_and_replay(tmp_path):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    assert rows and all(0 <= row["action_jump_rate"] <= 1 for row in rows)
    assert validate_output(result, config=config)["status"] == "valid"
    run_pipeline(config, visual_encoder=FailingCocoreVisualEncoder())


@pytest.mark.parametrize(
    "target",
    [
        "scale",
        "threshold",
        "pair_count",
        "offsets",
        "dimensions",
        "rate",
        "missing",
        "raw",
        "node",
        "row",
        "selected",
        "contract",
    ],
)
def test_action_jump_tamper_detection(tmp_path, target):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    encode = result.parent / "encode"
    if target in {
        "scale",
        "threshold",
        "pair_count",
        "offsets",
        "dimensions",
        "rate",
        "raw",
        "missing",
    }:
        field = (
            "action_jump_actions"
            if target == "raw"
            else "action_jump_rate"
            if target == "missing"
            else f"action_jump_{target}"
        )
        path = encode / f"{field}.npy"
        if target == "missing":
            path.unlink()
        else:
            values = np.load(path)
            values.flat[0] += 1
            np.save(path, values)
            if target != "raw":
                manifest_path = encode / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["action_jump_checksums"][field] = cocore_pipeline.file_sha256(path)
                manifest_path.write_text(json.dumps(manifest))
    elif target == "node":
        path = result.parent / cocore_pipeline.GRAPH_DIRECTORY / "nodes.npz"
        with np.load(path) as nodes:
            arrays = dict(nodes)
        arrays["action_jump_rate"][0] += 0.1
        np.savez(path, **arrays)
    elif target == "row":
        path = result / "all_clips.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["action_jump_rate"] += 0.1
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif target == "selected":
        path = result / "selected_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["action_jump_rate"] += 0.1
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        path = result / "selection_report.json"
        report = json.loads(path.read_text())
        report["action_jump"]["threshold_quantile"] = 0.5
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="action_jump|cache file"):
        validate_output(result, config=config)


def test_action_jump_diagnostics_and_disabled_compatibility(tmp_path):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["reliability_metrics"] = [
        "support",
        "progress",
        "action_variation",
        "visual_action_consistency",
    ]
    plain = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    original = (plain / "selected_manifest.jsonl").read_text()
    fingerprint = json.loads((plain.parent / "encode" / "manifest.json").read_text())["fingerprint"]
    assert not list((plain.parent / "encode").glob("action_jump*"))
    assert "action_jump" not in json.loads((plain.parent / "encode" / "manifest.json").read_text())
    config["action_jump"] = {}
    config["output"]["directory"] = str(tmp_path / "diagnostic")
    diagnostic = run_pipeline(config, visual_encoder=CocoreVisualEncoder())
    expected = [json.loads(line) for line in original.splitlines()]
    actual = [
        json.loads(line)
        for line in (diagnostic / "selected_manifest.jsonl").read_text().splitlines()
    ]
    assert [(r["sample_id"], r["reliability"]) for r in actual] == [
        (r["sample_id"], r["reliability"]) for r in expected
    ]
    assert (
        json.loads((diagnostic.parent / "encode" / "manifest.json").read_text())["fingerprint"]
        != fingerprint
    )
    assert validate_output(diagnostic, config=config)["status"] == "valid"


@pytest.mark.parametrize(
    "parameter,value", [("threshold", 1.0), ("threshold_quantile", 0.9), ("epsilon", 1e-6)]
)
def test_action_jump_parameters_invalidate_cache(tmp_path, parameter, value):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    encode_stage(config, visual_encoder=CocoreVisualEncoder())
    config["action_jump"] = {parameter: value}
    with pytest.raises(FileExistsError):
        encode_stage(config, visual_encoder=FailingCocoreVisualEncoder())


def test_action_jump_reference_respects_episode_limit(tmp_path):
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    config["runtime"]["max_episodes"] = 1
    _, _, encoded = encode_stage(config, visual_encoder=CocoreVisualEncoder())
    np.testing.assert_array_equal(encoded.action_jump_episode_ids, [0])
    np.testing.assert_array_equal(encoded.action_jump_offsets, [0, 605])
    assert encoded.action_jump_pair_count == 604
