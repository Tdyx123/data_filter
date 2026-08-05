from __future__ import annotations

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
from relcore.pipeline import run_pipeline, scan_stage, validate_output
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


class PipelineVisualEncoder:
    output_dim = 3

    def encode(self, images: np.ndarray) -> np.ndarray:
        values = images[:, 0, 0, 0].astype(np.float32)
        return np.stack([values + 1.0, values + 2.0, values + 4.0], axis=1)


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

    root = run_pipeline(config, visual_encoder=PipelineVisualEncoder())

    assert root == Path(config["output"]["directory"])
    assert (root / "scan" / "manifest.json").is_file()
    assert (root / "encode" / "manifest.json").is_file()
    assert (root / "graph" / "manifest.json").is_file()
    assert (root / "select" / "manifest.json").is_file()
    assert (root / "run_manifest.json").is_file()
    manifest_rows = [
        json.loads(line) for line in (root / "selected_manifest.jsonl").read_text().splitlines()
    ]
    all_clips = pq.read_table(root / "all_clips.parquet").to_pylist()
    report = json.loads((root / "selection_report.json").read_text())
    assert len(manifest_rows) == 2
    assert len(all_clips) == 6
    assert {row["task_index"] for row in manifest_rows} == {0, 1}
    assert all("hdf5_path" not in row and "demo_key" not in row for row in manifest_rows)
    assert report["selected_clips"] == 2
    assert report["task_quotas"] == {"0": 1, "1": 1}
    assert set(report["runtime_seconds"]) == {"scan", "encode", "graph", "select"}
    assert "prototype_coverage" in report
    assert report["number_of_episodes"] == 3
    assert report["skipped_short_episodes"] == [2]
    assert len(report["branches"]) == 2
    assert report["local_search"]["accepted_swaps"] == 0
    assert all("marginal_gain" in row for row in manifest_rows)

    first_mtime = (root / "selected_manifest.jsonl").stat().st_mtime_ns
    cached = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    assert cached == root
    assert (root / "selected_manifest.jsonl").stat().st_mtime_ns == first_mtime

    validation = validate_output(root, config=config)
    assert validation["status"] == "valid"
    assert validation["selected_clips"] == 2


def test_scan_performs_the_numeric_pass_without_loading_images(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    PipelineAdapter.load_images_calls.clear()
    config = _config(tmp_path)

    root, _, clips, _ = scan_stage(config)

    assert len(clips) == 6
    assert PipelineAdapter.load_images_calls == [False]
    assert (root / "scan" / "normalization.npz").is_file()


def test_force_rebuilds_only_changed_selection_stage(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    root = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    encode_mtime = (root / "encode" / "manifest.json").stat().st_mtime_ns
    graph_mtime = (root / "graph" / "manifest.json").stat().st_mtime_ns

    changed = _config(tmp_path)
    changed["selection"]["budget"] = 4
    changed["selection"]["branches"] = 3
    run_pipeline(changed, force=True, visual_encoder=PipelineVisualEncoder())

    assert (root / "encode" / "manifest.json").stat().st_mtime_ns == encode_mtime
    assert (root / "graph" / "manifest.json").stat().st_mtime_ns == graph_mtime
    report = json.loads((root / "selection_report.json").read_text())
    assert report["selected_clips"] == 4
    assert len(report["branches"]) == 3


def test_seed_change_reuses_scan_but_rebuilds_randomized_stages(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    root = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
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


def test_validate_rejects_missing_stage_artifact(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    root = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    (root / "graph" / "sequence_edges.npz").unlink()

    with pytest.raises(ValueError, match="missing stage artifact"):
        validate_output(root, config=config)


def test_missing_frame_feature_invalidates_encode_cache(tmp_path: Path):
    register_dataset_adapter("relcore_pipeline_synthetic", PipelineAdapter)
    config = _config(tmp_path)
    root = run_pipeline(config, visual_encoder=PipelineVisualEncoder())
    (root / "encode" / "frame_features" / "ep000000.npy").unlink()

    with pytest.raises(FileExistsError, match="pass --force"):
        run_pipeline(config, visual_encoder=PipelineVisualEncoder())


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
    main(
        [
            "validate",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ]
    )

    assert (output / "selected_manifest.jsonl").is_file()
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
