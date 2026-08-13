from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from relcore.cli import main
from relcore.config import load_config
from relcore.pipeline import (
    _encode_fingerprint,
    run_pipeline,
    scan_stage,
    select_stage,
    validate_output,
)
from relcore.utils.io import publish_stage, write_json
from trajectory_data import (
    DatasetAdapter,
    EpisodeData,
    EpisodeRecord,
    register_dataset_adapter,
)


class PipelineAdapter(DatasetAdapter):
    load_images_calls: ClassVar[list[bool]] = []

    def __init__(self, _: Mapping[str, object]):
        self._records = (
            EpisodeRecord(0, 31, 0, "task zero"),
            EpisodeRecord(1, 31, 1, "task one"),
            EpisodeRecord(2, 7, 2, "short task"),
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
            observations = {
                "observation.state": np.stack(
                    [steps / 30.0, np.sin(steps), np.mod(steps, 2.0)], axis=1
                ).astype(np.float32)
            }
            if load_images:
                pixels = np.mod(steps + record.episode_id * 41, 255).astype(np.uint8)
                observations["observation.images.image"] = np.broadcast_to(
                    pixels[:, None, None, None],
                    (record.length, 2, 2, 3),
                ).copy()
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations=observations,
                actions=np.stack([steps / 30.0, np.cos(steps), np.mod(steps, 2.0)], axis=1).astype(
                    np.float32
                ),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "relcore-pipeline-adapter-v1"


class MotionPipelineAdapter(PipelineAdapter):
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
            source = episode.observations["observation.state"]
            states = np.zeros((episode.length, 8), dtype=np.float32)
            states[:, : source.shape[1]] = source
            episode.observations["observation.state"] = states
            yield episode

    def fingerprint(self) -> str:
        return "relcore-motion-pipeline-adapter-v1"


class PipelineVisualEncoder:
    output_dim = 3

    def encode(self, images: np.ndarray) -> np.ndarray:
        values = images[:, 0, 0, 0].astype(np.float32)
        return np.stack([values + 1.0, values + 2.0, values + 4.0], axis=1)


class InterruptingVisualEncoder(PipelineVisualEncoder):
    def __init__(self, fail_on_call: int | None):
        self.fail_on_call = fail_on_call
        self.episode_lengths: list[int] = []

    def encode(self, images: np.ndarray) -> np.ndarray:
        self.episode_lengths.append(len(images))
        if self.fail_on_call == len(self.episode_lengths):
            raise RuntimeError("injected CLIP interruption")
        return super().encode(images)


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "seed": 7,
        "dataset": {
            "type": "relcore_pipeline_synthetic",
            "name": "synthetic",
            "path": str(tmp_path / "dataset"),
            "use_images": True,
        },
        "clip": {"length": 15, "stride": 15},
        "visual": {"encoder": "dummy"},
        "normalization": {"epsilon": 1.0e-6},
        "relation": {
            "projection_dim": 4,
            "output_dim": 8,
            "lags": [0, 1, 2, 4],
        },
        "quality": {
            "knn": 2,
            "gripper_progress_weight": 0.5,
            "visual_progress_weight": 0.25,
            "noop_threshold": 1.0e-4,
            "gripper_action_index": -1,
            "min_reliability": 0.05,
        },
        "prototypes": {
            "count": 2,
            "batch_size": 8,
            "max_iter": 20,
            "top_r": 2,
            "temperature": 0.2,
        },
        "graph": {
            "knn": 2,
            "similarity_threshold": 0.8,
            "cooccurrence_max_gap": 4,
        },
        "objective": {
            "node_weight": 4.0,
            "transition_weight": 1.2,
            "cooccurrence_weight": 0.5,
            "sequence_weight": 2.0,
            "redundancy_weight": 2.0,
        },
        "selection": {
            "budget": 2,
            "ratio": 0.1,
            "minimum_per_task": 1,
            "engine": "sparse",
            "branches": 2,
            "seed_candidates": 8,
            "seed_similarity_threshold": 0.9,
            "transition_seed_threshold": 0.0,
            "seed_pairs_per_transition": 2,
            "global_candidates": 4,
            "residual_candidates": 4,
            "random_candidates": 2,
            "local_search": {"enabled": False},
        },
        "runtime": {"num_workers": 0, "max_episodes": None, "resume": True},
        "output": {"directory": str(tmp_path / "relcore-output")},
    }


def test_run_pipeline_publishes_aligned_outputs_and_reuses_complete_cache(
    tmp_path: Path,
):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)

    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    root = Path(config["output"]["directory"])

    assert result == root / "select-r15-g7"
    assert (root / "scan" / "manifest.json").is_file()
    assert (root / "encode" / "manifest.json").is_file()
    assert not (root / ".relcore-cache").exists()
    assert not (root / "encode" / "frame_features").exists()
    assert not (root / "encode" / "frame_features_index.json").exists()
    encode_manifest = json.loads((root / "encode" / "manifest.json").read_text())
    assert "encoded_episodes" not in encode_manifest
    assert not any(key.startswith("frame_cache_") for key in encode_manifest)
    assert (root / "graph-15" / "manifest.json").is_file()
    assert (result / "manifest.json").is_file()
    assert (result / "run_manifest.json").is_file()
    assert (result / "resolved_config.yaml").is_file()
    assert (result / "environment.json").is_file()
    manifest_rows = [
        json.loads(line) for line in (result / "selected_manifest.jsonl").read_text().splitlines()
    ]
    all_clips = pq.read_table(result / "all_clips.parquet").to_pylist()
    report = json.loads((result / "selection_report.json").read_text())
    assert len(manifest_rows) == 2
    assert len(all_clips) == 6
    assert {row["task_index"] for row in manifest_rows} == {0, 1}
    assert all("hdf5_path" not in row and "demo_key" not in row for row in manifest_rows)
    assert report["selected_clips"] == 2
    assert report["reliability_metrics"] == [
        "support",
        "progress",
        "smoothness",
        "non_noop",
    ]
    assert report["prototype_gain_metrics"] == [
        "transition",
        "cooccurrence",
        "sequence",
    ]
    assert report["task_quotas"] == {"0": 1, "1": 1}
    assert set(report["runtime_seconds"]) == {"scan", "encode", "graph", "select"}
    assert "prototype_coverage" in report
    assert report["number_of_episodes"] == 3
    assert report["skipped_short_episodes"] == [2]
    assert len(report["branches"]) == 2
    assert report["local_search"]["accepted_swaps"] == 0
    assert all("marginal_gain" in row for row in manifest_rows)

    run_manifest = json.loads((result / "run_manifest.json").read_text())
    assert run_manifest["selection_output_quota_mode"] is None
    assert run_manifest["reliability_mask"] == 15
    assert run_manifest["prototype_gain_metrics"] == [
        "transition",
        "cooccurrence",
        "sequence",
    ]
    assert run_manifest["prototype_gain_mask"] == 7
    assert run_manifest["stage_directories"] == {
        "scan": "scan",
        "encode": "encode",
        "graph": "graph-15",
        "select": "select-r15-g7",
    }

    first_mtime = (result / "selected_manifest.jsonl").stat().st_mtime_ns
    cached_encoder = InterruptingVisualEncoder(fail_on_call=1)
    cached = run_pipeline(config, visual_encoder=cached_encoder)
    assert cached == result
    assert cached_encoder.episode_lengths == []
    assert (result / "selected_manifest.jsonl").stat().st_mtime_ns == first_mtime

    validation = validate_output(result, config=config)
    assert validation["status"] == "valid"
    assert validation["selected_clips"] == 2


def test_motion_primitive_pipeline_coexists_with_kmeans_and_exports_labels(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("relcore_motion_pipeline_synthetic", MotionPipelineAdapter)
    kmeans_config = _config(tmp_path)
    kmeans_config["dataset"]["type"] = "relcore_motion_pipeline_synthetic"
    kmeans_result = run_pipeline(kmeans_config, visual_encoder=PipelineVisualEncoder())

    motion_config = copy.deepcopy(kmeans_config)
    motion_config["prototypes"]["method"] = "motion_primitives"
    motion_result = run_pipeline(motion_config, visual_encoder=PipelineVisualEncoder())
    root = Path(motion_config["output"]["directory"])

    assert kmeans_result == root / "select-r15-g7"
    assert motion_result == root / "select-r15-g7-motion-primitives"
    assert (root / "graph-15" / "prototype_centers.npy").is_file()
    graph_root = root / "graph-15-motion-primitives"
    assert not (graph_root / "prototype_centers.npy").exists()
    catalog = json.loads((graph_root / "prototype_catalog.json").read_text())
    assert catalog["method"] == "motion_primitives"
    assert catalog["constants"] == {
        "clip_anchors": [0, 7, 14],
        "dominance_ratio": 4.0,
        "fallback_weight": 0.8,
        "horizon": 8,
        "min_frequency": 0.005,
        "state_threshold": 0.03,
    }
    assert catalog["total_labels"] == 46
    graph_manifest = json.loads((graph_root / "manifest.json").read_text())
    assert graph_manifest["prototype_method"] == "motion_primitives"

    rows = [
        json.loads(line)
        for line in (motion_result / "selected_manifest.jsonl").read_text().splitlines()
    ]
    assert rows
    for row in rows:
        assert 1 <= len(row["prototype_indices"]) <= 4
        assert len(row["prototype_indices"]) == len(row["prototype_weights"])
        assert len(row["prototype_indices"]) == len(row["prototype_labels"])
        assert row["primary_prototype_label"] == row["prototype_labels"][0]
        assert all(index >= 0 for index in row["prototype_indices"])
    report = json.loads((motion_result / "selection_report.json").read_text())
    run_manifest = json.loads((motion_result / "run_manifest.json").read_text())
    assert report["prototype_method"] == "motion_primitives"
    assert run_manifest["prototype_method"] == "motion_primitives"
    assert run_manifest["stage_directories"] == {
        "scan": "scan",
        "encode": "encode",
        "graph": "graph-15-motion-primitives",
        "select": "select-r15-g7-motion-primitives",
    }
    assert validate_output(motion_result, config=motion_config) == {
        "status": "valid",
        "selected_clips": 2,
    }


def test_scan_performs_the_numeric_pass_without_loading_images(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    PipelineAdapter.load_images_calls.clear()
    config = _config(tmp_path)

    root, _, clips, _ = scan_stage(config)

    assert len(clips) == 6
    assert PipelineAdapter.load_images_calls == [False]
    assert (root / "scan" / "normalization.npz").is_file()
    manifest = json.loads((root / "scan" / "manifest.json").read_text())
    assert manifest["dataset_summary"] == {
        "source_episodes": 3,
        "indexed_episodes": 3,
        "retained_episodes": 3,
        "excluded_episodes": 0,
        "excluded_empty_task_episodes": 0,
    }
    assert manifest["scanned_episodes"] == 3
    assert manifest["skipped_short_episode_count"] == 1


def test_force_rebuilds_only_changed_selection_stage(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    root = result.parent
    encode_mtime = (root / "encode" / "manifest.json").stat().st_mtime_ns
    graph_mtime = (root / "graph-15" / "manifest.json").stat().st_mtime_ns

    changed = _config(tmp_path)
    changed["selection"]["budget"] = 4
    changed["selection"]["branches"] = 3
    run_pipeline(changed, force=True, visual_encoder=PipelineVisualEncoder())

    assert (root / "encode" / "manifest.json").stat().st_mtime_ns == encode_mtime
    assert (root / "graph-15" / "manifest.json").stat().st_mtime_ns == graph_mtime
    report = json.loads((result / "selection_report.json").read_text())
    assert report["selected_clips"] == 4
    assert len(report["branches"]) == 3


def test_reliability_metric_variants_share_encode_and_keep_graphs_and_selects(
    tmp_path: Path,
):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    root = Path(config["output"]["directory"])
    all_metrics = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    encode_mtime = (root / "encode" / "manifest.json").stat().st_mtime_ns
    baseline_reliability = np.load(root / "graph-15" / "nodes.npz")["reliability"].copy()

    selected = run_pipeline(
        config,
        reliability_metrics=["non_noop", "progress"],
        visual_encoder=PipelineVisualEncoder(),
    )

    assert all_metrics == root / "select-r15-g7"
    assert selected == root / "select-r5-g7"
    assert (root / "encode" / "manifest.json").stat().st_mtime_ns == encode_mtime
    assert (root / "graph-15" / "manifest.json").is_file()
    assert (root / "graph-5" / "manifest.json").is_file()
    assert (root / "select-r15-g7" / "manifest.json").is_file()
    assert (root / "select-r5-g7" / "manifest.json").is_file()
    nodes = np.load(root / "graph-5" / "nodes.npz")
    assert not np.array_equal(nodes["reliability"], baseline_reliability)
    np.testing.assert_allclose(
        nodes["reliability"],
        nodes["progress"] ** 0.5 * np.maximum(1.0 - nodes["noop_ratio"], 0.0) ** 0.5,
        rtol=1.0e-6,
    )
    graph_manifest = json.loads((root / "graph-5" / "manifest.json").read_text())
    encode_manifest = json.loads((root / "encode" / "manifest.json").read_text())
    assert graph_manifest["reliability_metrics"] == ["progress", "non_noop"]
    assert graph_manifest["reliability_mask"] == 5
    assert graph_manifest["upstream_fingerprint"] == encode_manifest["fingerprint"]
    report = json.loads((selected / "selection_report.json").read_text())
    assert report["reliability_metrics"] == ["progress", "non_noop"]
    assert validate_output(selected, config=config)["status"] == "valid"


def test_prototype_gain_metric_variants_reuse_graph_and_keep_selects(
    tmp_path: Path,
):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    root = Path(config["output"]["directory"])
    all_metrics = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    graph_manifest_path = root / "graph-15" / "manifest.json"
    graph_mtime = graph_manifest_path.stat().st_mtime_ns
    graph_fingerprint = json.loads(graph_manifest_path.read_text())["fingerprint"]

    selected = run_pipeline(
        config,
        prototype_gain_metrics=["sequence", "transition"],
        visual_encoder=PipelineVisualEncoder(),
    )

    assert all_metrics == root / "select-r15-g7"
    assert selected == root / "select-r15-g5"
    assert graph_manifest_path.stat().st_mtime_ns == graph_mtime
    assert json.loads(graph_manifest_path.read_text())["fingerprint"] == graph_fingerprint
    assert (all_metrics / "manifest.json").is_file()
    select_manifest = json.loads((selected / "manifest.json").read_text())
    assert select_manifest["prototype_gain_metrics"] == ["transition", "sequence"]
    assert select_manifest["prototype_gain_mask"] == 5
    report = json.loads((selected / "selection_report.json").read_text())
    assert report["prototype_gain_metrics"] == ["transition", "sequence"]
    run_manifest = json.loads((selected / "run_manifest.json").read_text())
    assert run_manifest["prototype_gain_metrics"] == ["transition", "sequence"]
    assert run_manifest["prototype_gain_mask"] == 5
    assert validate_output(selected, config=config)["status"] == "valid"


def test_ratio_scoped_selects_coexist_and_preserve_shared_outputs(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    legacy_config = _config(tmp_path)
    default_result = run_pipeline(legacy_config, visual_encoder=PipelineVisualEncoder())
    root = default_result.parent
    upstream_mtimes = {
        stage: (root / directory / "manifest.json").stat().st_mtime_ns
        for stage, directory in {
            "scan": "scan",
            "encode": "encode",
            "graph": "graph-15",
        }.items()
    }
    published = {
        name: (default_result / name).read_bytes()
        for name in (
            "selected_manifest.jsonl",
            "all_clips.parquet",
            "selection_report.json",
            "run_manifest.json",
        )
    }

    half_config = _config(tmp_path)
    half_config["selection"]["budget"] = None
    half_config["selection"]["ratio"] = 0.5
    full_config = _config(tmp_path)
    full_config["selection"]["budget"] = None
    full_config["selection"]["ratio"] = 1.0

    assert (
        select_stage(
            half_config,
            selection_output_ratio=0.5,
            visual_encoder=PipelineVisualEncoder(),
        )
        == root / "select-r15-g7-top50pct"
    )
    assert (
        select_stage(
            full_config,
            selection_output_ratio=1.0,
            visual_encoder=PipelineVisualEncoder(),
        )
        == root / "select-r15-g7-top100pct"
    )

    half_root = root / "select-r15-g7-top50pct"
    full_root = root / "select-r15-g7-top100pct"
    assert json.loads((half_root / "manifest.json").read_text())["selected_clips"] == 3
    assert json.loads((full_root / "manifest.json").read_text())["selected_clips"] == 6
    for selection_root in (half_root, full_root):
        assert {
            "manifest.json",
            "selected_manifest.jsonl",
            "all_clips.parquet",
            "selection_report.json",
        } <= {path.name for path in selection_root.iterdir()}
    assert {
        stage: (root / directory / "manifest.json").stat().st_mtime_ns
        for stage, directory in {
            "scan": "scan",
            "encode": "encode",
            "graph": "graph-15",
        }.items()
    } == upstream_mtimes
    assert {name: (default_result / name).read_bytes() for name in published} == published


def test_ratio_scoped_select_reuses_matching_cache_and_requires_force_for_changes(
    tmp_path: Path,
):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    config["selection"]["budget"] = None
    config["selection"]["ratio"] = 0.5
    selection_root = select_stage(
        config,
        selection_output_ratio=0.5,
        visual_encoder=PipelineVisualEncoder(),
    )
    first_mtime = (selection_root / "manifest.json").stat().st_mtime_ns

    select_stage(
        config,
        selection_output_ratio=0.5,
        visual_encoder=PipelineVisualEncoder(),
    )
    assert (selection_root / "manifest.json").stat().st_mtime_ns == first_mtime

    changed = _config(tmp_path)
    changed["selection"]["budget"] = None
    changed["selection"]["ratio"] = 0.5
    changed["selection"]["branches"] = 3
    with pytest.raises(FileExistsError, match="pass --force"):
        select_stage(
            changed,
            selection_output_ratio=0.5,
            visual_encoder=PipelineVisualEncoder(),
        )
    select_stage(
        changed,
        selection_output_ratio=0.5,
        force=True,
        visual_encoder=PipelineVisualEncoder(),
    )
    assert json.loads((selection_root / "manifest.json").read_text())["status"] == "complete"


def test_ratio_scoped_select_rejects_output_ratio_that_differs_from_config(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    config["selection"]["budget"] = None
    config["selection"]["ratio"] = 0.5

    with pytest.raises(ValueError, match="selection_output_ratio must match"):
        select_stage(
            config,
            selection_output_ratio=0.25,
            visual_encoder=PipelineVisualEncoder(),
        )


def test_quota_scoped_selects_coexist_with_each_other_and_legacy_output(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    proportional_config = _config(tmp_path)
    legacy_result = run_pipeline(
        proportional_config,
        visual_encoder=PipelineVisualEncoder(),
    )
    proportional_result = run_pipeline(
        proportional_config,
        selection_output_quota_mode="proportional",
        visual_encoder=PipelineVisualEncoder(),
    )

    global_config = _config(tmp_path)
    global_config["selection"]["quota_mode"] = "none"
    global_config["selection"]["minimum_per_task"] = 0
    global_result = run_pipeline(
        global_config,
        selection_output_quota_mode="none",
        visual_encoder=PipelineVisualEncoder(),
    )

    assert legacy_result == global_result.parent / "select-r15-g7"
    assert proportional_result == global_result.parent / "select-r15-g7-quota-proportional"
    assert global_result == global_result.parent / "select-r15-g7-quota-none"
    assert legacy_result.is_dir()
    assert proportional_result.is_dir()
    run_manifest = json.loads((global_result / "run_manifest.json").read_text())
    assert run_manifest["selection_output_quota_mode"] == "none"
    assert run_manifest["stage_directories"]["select"] == "select-r15-g7-quota-none"
    report = json.loads((global_result / "selection_report.json").read_text())
    assert report["quota_mode"] == "none"
    assert report["task_quotas"] is None
    assert validate_output(global_result, config=global_config) == {
        "status": "valid",
        "selected_clips": 2,
    }


def test_quota_scoped_select_rejects_mode_that_differs_from_config(tmp_path: Path) -> None:
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="selection_output_quota_mode must match"):
        select_stage(
            config,
            selection_output_quota_mode="none",
            visual_encoder=PipelineVisualEncoder(),
        )


def test_global_selection_reports_actual_tasks_without_hard_quotas(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    config["selection"]["quota_mode"] = "none"
    config["selection"]["minimum_per_task"] = 0

    root = run_pipeline(config, visual_encoder=PipelineVisualEncoder())

    report = json.loads((root / "selection_report.json").read_text())
    assert report["quota_mode"] == "none"
    assert report["task_quotas"] is None
    assert sum(report["task_counts"].values()) == 2
    assert validate_output(root, config=config) == {"status": "valid", "selected_clips": 2}


def test_global_validation_rejects_a_mismatched_recorded_budget(tmp_path: Path) -> None:
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    config["selection"]["quota_mode"] = "none"
    config["selection"]["minimum_per_task"] = 0
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["budget"] = 3
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="budget"):
        validate_output(result)


def test_seed_change_reuses_scan_but_rebuilds_randomized_stages(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    root = result.parent
    scan_mtime = (root / "scan" / "manifest.json").stat().st_mtime_ns
    encode_mtime = (root / "encode" / "manifest.json").stat().st_mtime_ns

    changed = _config(tmp_path)
    changed["seed"] = 101
    run_pipeline(changed, force=True, visual_encoder=PipelineVisualEncoder())

    assert (root / "scan" / "manifest.json").stat().st_mtime_ns == scan_mtime
    assert (root / "encode" / "manifest.json").stat().st_mtime_ns != encode_mtime


def test_validate_rejects_a_config_with_different_selection_fingerprint(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    root = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    changed = _config(tmp_path)
    changed["selection"]["budget"] = 4

    with pytest.raises(ValueError, match="fingerprint"):
        validate_output(root, config=changed)


def test_validate_rejects_noncanonical_reported_reliability_metrics(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    report_path = result / "selection_report.json"
    report = json.loads(report_path.read_text())
    report["reliability_metrics"] = ["progress", "support"]
    write_json(report_path, report)

    with pytest.raises(ValueError, match="reliability_metrics"):
        validate_output(result)


def test_validate_rejects_reliability_mask_that_differs_from_metrics(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reliability_mask"] = 5
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="reliability_mask"):
        validate_output(result, config=config)


def test_validate_rejects_noncanonical_reported_prototype_gain_metrics(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    report_path = result / "selection_report.json"
    report = json.loads(report_path.read_text())
    report["prototype_gain_metrics"] = ["sequence", "transition"]
    write_json(report_path, report)

    with pytest.raises(ValueError, match="prototype_gain_metrics"):
        validate_output(result)


def test_validate_rejects_prototype_gain_mask_that_differs_from_metrics(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prototype_gain_mask"] = 5
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="prototype_gain_mask"):
        validate_output(result, config=config)


def test_validate_rejects_legacy_manifest_without_prototype_gain_metrics(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("prototype_gain_metrics")
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="prototype_gain_metrics"):
        validate_output(result)


def test_validate_rejects_tampered_select_prototype_gain_metrics(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prototype_gain_metrics"] = ["transition"]
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="select.*prototype_gain_metrics"):
        validate_output(result)


def test_validate_rejects_tampered_stage_directory(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stage_directories"]["graph"] = "graph-5"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="stage_directories"):
        validate_output(result)


def test_validate_accepts_legacy_manifest_without_quota_scope(tmp_path: Path) -> None:
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("selection_output_quota_mode")
    write_json(manifest_path, manifest)

    assert validate_output(result, config=config) == {
        "status": "valid",
        "selected_clips": 2,
    }


def test_validate_rejects_tampered_quota_scope(tmp_path: Path) -> None:
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_output_quota_mode"] = "none"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="stage_directories"):
        validate_output(result)


def test_validate_rejects_non_string_quota_scope(tmp_path: Path) -> None:
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_output_quota_mode"] = ["none"]
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="selection_output_quota_mode"):
        validate_output(result)


def test_validate_rejects_quota_scope_that_differs_from_report(tmp_path: Path) -> None:
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    config["selection"]["quota_mode"] = "none"
    config["selection"]["minimum_per_task"] = 0
    result = run_pipeline(
        config,
        selection_output_quota_mode="none",
        visual_encoder=PipelineVisualEncoder(),
    )
    report_path = result / "selection_report.json"
    report = json.loads(report_path.read_text())
    report["quota_mode"] = "proportional"
    report["task_quotas"] = report["task_counts"]
    write_json(report_path, report)

    with pytest.raises(ValueError, match="quota mode.*run manifest"):
        validate_output(result)


def test_validate_rejects_report_quota_mode_that_differs_from_config(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    report_path = result / "selection_report.json"
    report = json.loads(report_path.read_text())
    report["quota_mode"] = "none"
    report["task_quotas"] = None
    write_json(report_path, report)

    with pytest.raises(ValueError, match="quota mode.*supplied config"):
        validate_output(result, config=config)


def test_validate_rejects_tampered_graph_metrics(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result.parent / "graph-15" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reliability_metrics"] = ["progress"]
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="graph.*reliability_metrics"):
        validate_output(result)


@pytest.mark.parametrize(
    ("stage", "directory"),
    [("graph", "graph-15"), ("select", "select-r15-g7")],
)
def test_validate_rejects_tampered_upstream_fingerprint(
    tmp_path: Path,
    stage: str,
    directory: str,
):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    manifest_path = result.parent / directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["upstream_fingerprint"] = "tampered"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=f"{stage}.*upstream"):
        validate_output(result)


def test_validate_rejects_missing_stage_artifact(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    (result.parent / "graph-15" / "sequence_edges.npz").unlink()

    with pytest.raises(ValueError, match="missing stage artifact"):
        validate_output(result, config=config)


def test_missing_aggregate_feature_invalidates_encode_cache(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    result = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    root = result.parent
    (root / "encode" / "embeddings.npy").unlink()

    with pytest.raises(FileExistsError, match="pass --force"):
        run_pipeline(config, visual_encoder=PipelineVisualEncoder())


def test_interrupted_encode_restarts_all_episode_encoding_without_frame_cache(
    tmp_path: Path,
):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    interrupted = InterruptingVisualEncoder(fail_on_call=2)

    with pytest.raises(RuntimeError, match="injected CLIP interruption"):
        run_pipeline(config, visual_encoder=interrupted)

    resumed = InterruptingVisualEncoder(fail_on_call=None)
    result = run_pipeline(config, visual_encoder=resumed)
    root = result.parent

    assert interrupted.episode_lengths == [31, 31]
    assert resumed.episode_lengths == [31, 31]
    assert not (root / ".relcore-cache").exists()
    assert not (root / "encode" / "frame_features").exists()
    assert not (root / "encode" / "frame_features_index.json").exists()
    manifest = json.loads((root / "encode" / "manifest.json").read_text())
    assert "encoded_episodes" not in manifest
    assert not any(key.startswith("frame_cache_") for key in manifest)


def test_local_model_content_invalidates_encode_fingerprint(
    tmp_path: Path,
) -> None:
    adapter = PipelineAdapter({})
    config = _config(tmp_path)
    model = tmp_path / "clip"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"weights-v1")
    config["visual"] = {
        "encoder": "clip",
        "model": str(model),
        "local_files_only": True,
        "batch_size": 1,
        "device": "cuda",
    }
    first_encode = _encode_fingerprint(adapter, config, "scan-v1")

    weights.write_bytes(b"weights-v2")

    second_encode = _encode_fingerprint(adapter, config, "scan-v1")
    assert second_encode != first_encode


def test_failed_forced_stage_build_preserves_the_previous_complete_cache(tmp_path: Path):
    destination = tmp_path / "stage"
    destination.mkdir()
    (destination / "artifact.bin").write_bytes(b"previous")
    write_json(
        destination / "manifest.json",
        {"status": "complete", "fingerprint": "previous"},
    )

    def fail_build(temporary: Path) -> None:
        (temporary / "artifact.bin").write_bytes(b"partial")
        write_json(
            temporary / "manifest.json",
            {"status": "complete", "fingerprint": "next"},
        )
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        publish_stage(
            destination,
            fingerprint="next",
            required=("artifact.bin",),
            force=True,
            resume=False,
            build=fail_build,
        )

    assert (destination / "artifact.bin").read_bytes() == b"previous"
    assert json.loads((destination / "manifest.json").read_text())["fingerprint"] == "previous"
    assert not list(tmp_path.glob(".stage.relcore-*"))


def test_cli_run_and_validate_use_exact_output_directory(tmp_path: Path, capsys):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    config_path = tmp_path / "relcore.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "chosen-output"

    main(
        [
            "run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ]
    )
    result = output / "select-r15-g7"
    main(
        [
            "validate",
            "--config",
            str(config_path),
            "--output-dir",
            str(result),
        ]
    )

    assert (result / "selected_manifest.jsonl").is_file()
    assert '"status": "valid"' in capsys.readouterr().out


@pytest.mark.real_data
def test_real_libero90_relcore_scan_maps_tasks_and_windows(tmp_path: Path):
    dataset = Path("/data/dwb/datasets/LIBERO_lerobot/libero90")
    if not dataset.is_dir():
        pytest.skip("real LIBERO-90 LeRobot dataset is not mounted")
    config = load_config(Path("relcore/config_debug.yaml"))
    config["runtime"]["max_episodes"] = 2
    config["runtime"]["num_workers"] = 0
    config["output"]["directory"] = str(tmp_path / "real-scan")

    _, adapter, clips, _ = scan_stage(config)

    assert clips
    assert all(clip.task_name and clip.task_index >= 0 for clip in clips)
    assert {clip.episode_id for clip in clips} == {
        record.episode_id for record in adapter.episodes()[:2]
    }
