from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from cocore.pipeline import run_pipeline, validate_output
from trajectory_data import (
    DatasetAdapter,
    EpisodeData,
    EpisodeRecord,
    register_dataset_adapter,
)


class CocorePipelineAdapter(DatasetAdapter):
    load_images_calls: ClassVar[list[bool]] = []

    def __init__(self, _: Mapping[str, object]) -> None:
        self._records = (
            EpisodeRecord(0, 45, 0, "task zero"),
            EpisodeRecord(1, 45, 1, "task one"),
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
            states[:, 0] = steps * 0.04
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
                actions=np.stack([steps / 44.0, np.zeros_like(steps)], axis=1),
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


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "seed": 7,
        "dataset": {
            "type": "cocore_pipeline_synthetic",
            "name": "synthetic",
            "path": str(tmp_path / "dataset"),
            "use_images": True,
        },
        "visual": {"encoder": "dummy"},
        "relation": {"projection_dim": 4, "output_dim": 8, "lags": [0, 1, 2, 4]},
        "quality": {"knn": 2},
        "graph": {"knn": 2, "similarity_threshold": 0.8, "cooccurrence_max_gap": 4},
        "objective": {"cooccurrence_weight": 1.0},
        "selection": {
            "ratio": 0.5,
            "budget": None,
            "max_refreshes": 2,
        },
        "runtime": {"num_workers": 0, "max_episodes": None, "resume": True},
        "output": {"directory": str(tmp_path / "cocore-output")},
    }


def test_run_pipeline_publishes_cocore_outputs_and_validate_recomputes_them(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("cocore_pipeline_synthetic", CocorePipelineAdapter)
    config = _config(tmp_path)
    root = tmp_path / "overridden-cocore-output"

    result = run_pipeline(config, output_dir=root, visual_encoder=CocoreVisualEncoder())

    assert result == root / "select-w1-top50pct"
    assert (root / "scan" / "manifest.json").is_file()
    assert (root / "encode" / "manifest.json").is_file()
    assert (root / "graph-12-motion-primitives" / "prototype_catalog.json").is_file()
    for directory in ("scan", "encode", "graph-12-motion-primitives"):
        manifest = json.loads((root / directory / "manifest.json").read_text())
        assert manifest["producer"] == "cocore"
        assert manifest["cocore_version"] == "0.2.0"
    nodes = np.load(root / "graph-12-motion-primitives" / "nodes.npz")
    np.testing.assert_allclose(
        nodes["reliability"],
        np.maximum(nodes["support"] ** 0.5 * nodes["progress"] ** 0.5, 0.05),
        rtol=1.0e-6,
    )
    selected = [
        json.loads(line)
        for line in (result / "selected_manifest.jsonl").read_text().splitlines()
    ]
    all_rows = pq.read_table(result / "all_clips.parquet").to_pylist()
    report = json.loads((result / "selection_report.json").read_text())
    assert len(selected) == 3
    assert len(all_rows) == 6
    assert {row["selection_phase"] for row in selected} == {"coverage_seed", "heap"}
    assert all(
        {"selection_step", "selection_score_delta", "heap_refreshes"} <= row.keys()
        for row in selected
    )
    assert all({"support", "progress", "reliability", "prototype_labels"} <= row.keys() for row in all_rows)
    assert report["objective"]["total"] == report["objective"]["cooccurrence"] - report["objective"]["redundancy"]
    assert report["coverage"]["target"] == report["coverage"]["achieved"]
    assert report["algorithm"] == {
        "type": "lazy_max_heap",
        "max_refreshes": 2,
    }
    assert report["heap"] == {
        "initial_size": 5,
        "total_refreshes": sum(row["heap_refreshes"] for row in selected),
        "capped_selections": sum(row["heap_refreshes"] == 2 for row in selected),
        "max_refreshes_observed": max(row["heap_refreshes"] for row in selected),
    }
    run_manifest = json.loads((result / "run_manifest.json").read_text())
    assert run_manifest["producer"] == "cocore"
    assert run_manifest["cocore_version"] == "0.2.0"
    assert run_manifest["algorithm"] == report["algorithm"]
    resolved = yaml.safe_load((result / "resolved_config.yaml").read_text())
    assert resolved["output"]["directory"] == str(root)
    assert validate_output(result, config=config) == {"status": "valid", "selected_clips": 3}

    report["heap"]["total_refreshes"] += 1
    (result / "selection_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="heap total refreshes"):
        validate_output(result, config=config)
