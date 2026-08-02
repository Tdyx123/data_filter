from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pytest
import yaml

from sqcn.cli import main
from sqcn.pipeline import run_pipeline
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord, register_dataset_adapter


class SyntheticImageAdapter(DatasetAdapter):
    def __init__(self, _: Mapping[str, object]):
        self._records = (
            EpisodeRecord(0, 14),
            EpisodeRecord(1, 30),
            EpisodeRecord(2, 31),
        )

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ("observation.image",)

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
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            observations = {
                "observation.state": np.stack(
                    [steps / max(record.length - 1, 1), np.sin(steps)],
                    axis=1,
                ).astype(np.float32)
            }
            if load_images:
                pixels = np.mod(steps + record.episode_id * 31, 255).astype(np.uint8)
                observations["observation.image"] = np.broadcast_to(
                    pixels[:, None, None, None],
                    (record.length, 2, 2, 3),
                ).copy()
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations=observations,
                actions=np.stack([steps, np.cos(steps)], axis=1).astype(np.float32),
            )

    def fingerprint(self) -> str:
        return "sqcn-synthetic-v1"


class FakeClipVisionEncoder:
    def __init__(self, config: Mapping[str, object]):
        self.config = dict(config)

    def encode(self, frames: np.ndarray) -> np.ndarray:
        values = frames[:, 0, 0, 0].astype(np.float32)
        features = np.stack([values + 1.0, np.mod(values, 7.0) + 1.0], axis=1)
        return features / np.linalg.norm(features, axis=1, keepdims=True)


class AllShortAdapter(SyntheticImageAdapter):
    def __init__(self, config: Mapping[str, object]):
        super().__init__(config)
        self._records = (EpisodeRecord(0, 14),)


class TwoCameraAdapter(SyntheticImageAdapter):
    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ("observation.image", "observation.image2")


class NoStateAdapter(SyntheticImageAdapter):
    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ()


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "dataset": {
            "type": "sqcn_synthetic",
            "name": "synthetic",
            "path": str(tmp_path),
            "use_images": True,
        },
        "encoder": {
            "model": "/models/local-clip-vit",
            "local_files_only": True,
            "image_batch_size": 16,
            "visual_dim": 256,
            "embedding_dim": 128,
            "pca_fit_max_samples": None,
            "device": "cpu",
        },
        "quality": {
            "quantile_low": 0,
            "quantile_high": 1,
            "epsilon": 1e-8,
        },
        "coverage": {
            "sigma": 1.0,
            "batch_size": 2,
            "device": "cpu",
            "quantile_low": 0,
            "quantile_high": 1,
        },
        "novelty": {
            "k": 2,
            "backend": "numpy",
            "batch_size": 2,
            "quantile_low": 0,
            "quantile_high": 1,
        },
        "runtime": {"seed": 42, "num_workers": 0, "resume": True},
        "output": {"root": str(tmp_path / "outputs" / "sqcn")},
    }


def test_pipeline_writes_aligned_fragment_and_reference_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    register_dataset_adapter("sqcn_synthetic", SyntheticImageAdapter)
    monkeypatch.setattr("sqcn.pipeline.ClipVisionEncoder", FakeClipVisionEncoder)

    config = _config(tmp_path)
    root = run_pipeline(config)

    assert root == Path(config["output"]["root"]) / "synthetic"

    with (root / "fragment" / "scores.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    embeddings = np.load(root / "fragment" / "embeddings.npy")
    reference_embeddings = np.load(root / "reference" / "embeddings.npy")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == 5
    assert embeddings.shape == (5, 128)
    assert reference_embeddings.shape == (4, 128)
    assert set(rows[0]) == {
        "sample_id",
        "episode_id",
        "start_step",
        "end_step",
        "length",
        "quality",
        "coverage",
        "novelty",
        "sqcn",
    }
    assert all(int(row["length"]) == 15 for row in rows)
    assert [float(row["sqcn"]) for row in rows] == sorted(
        [float(row["sqcn"]) for row in rows],
        reverse=True,
    )
    assert manifest["counts"] == {
        "candidate_fragments": 5,
        "reference_fragments": 4,
        "overlap_fragments": 4,
        "pca_union_fragments": 5,
        "skipped_short_episodes": 1,
    }
    assert manifest["config"]["coverage"]["sigma"] == 1.0
    assert manifest["config"]["encoder"]["local_files_only"] is True
    assert (root / "fragment" / "features.pkl").is_file()
    assert (root / "reference" / "segments.csv").is_file()
    assert (root / "encoder_artifacts" / "visual_pca.pkl").is_file()
    assert (root / "encoder_artifacts" / "fusion_pca.pkl").is_file()
    assert (root / "encoder_artifacts" / "numeric_normalizers.pkl").is_file()

    first_mtime = (root / "fragment" / "scores.csv").stat().st_mtime_ns
    cached = run_pipeline(_config(tmp_path))
    assert cached == root
    assert (root / "fragment" / "scores.csv").stat().st_mtime_ns == first_mtime


def test_cli_output_dir_is_the_exact_cached_run_directory(
    tmp_path: Path,
    monkeypatch,
):
    register_dataset_adapter("sqcn_synthetic", SyntheticImageAdapter)
    monkeypatch.setattr("sqcn.pipeline.ClipVisionEncoder", FakeClipVisionEncoder)
    config = _config(tmp_path)
    config_path = tmp_path / "sqcn.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_dir = tmp_path / "chosen" / "libero90_sqcn"

    main(["--config", str(config_path), "--output-dir", str(output_dir)])

    scores = output_dir / "fragment" / "scores.csv"
    first_mtime = scores.stat().st_mtime_ns
    assert scores.is_file()
    assert (output_dir / "fragment" / "embeddings.npy").is_file()
    assert (output_dir / "reference" / "segments.csv").is_file()
    assert (output_dir / "encoder_artifacts" / "visual_pca.pkl").is_file()
    assert (output_dir / "run_manifest.json").is_file()
    assert not (output_dir / "synthetic").exists()
    assert not (Path(config["output"]["root"]) / "synthetic").exists()
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["outputs"]["scores"] == str(scores)

    main(["--config", str(config_path), "--output-dir", str(output_dir)])
    assert scores.stat().st_mtime_ns == first_mtime


@pytest.mark.parametrize(
    ("adapter_name", "adapter", "message"),
    [
        (
            "sqcn_two_camera",
            TwoCameraAdapter,
            "exactly one configured image observation",
        ),
        (
            "sqcn_no_state",
            NoStateAdapter,
            "at least one vector state observation",
        ),
    ],
)
def test_pipeline_rejects_invalid_modalities_before_writing_outputs(
    tmp_path: Path,
    adapter_name: str,
    adapter: type[DatasetAdapter],
    message: str,
):
    register_dataset_adapter(adapter_name, adapter)
    config = _config(tmp_path)
    config["dataset"]["type"] = adapter_name
    config["dataset"]["name"] = adapter_name

    with pytest.raises(ValueError, match=message):
        run_pipeline(config)

    assert not (Path(config["output"]["root"]) / adapter_name).exists()


def test_pipeline_records_no_success_output_when_every_episode_is_too_short(
    tmp_path: Path,
    monkeypatch,
):
    register_dataset_adapter("sqcn_all_short", AllShortAdapter)
    monkeypatch.setattr("sqcn.pipeline.ClipVisionEncoder", FakeClipVisionEncoder)
    config = _config(tmp_path)
    config["dataset"]["type"] = "sqcn_all_short"
    config["dataset"]["name"] = "all_short"

    with pytest.raises(RuntimeError, match="no complete candidate/reference fragments"):
        run_pipeline(config)

    assert not (Path(config["output"]["root"]) / "all_short").exists()


def test_pipeline_requires_local_only_clip_loading(tmp_path: Path):
    config = _config(tmp_path)
    config["encoder"]["local_files_only"] = False

    with pytest.raises(ValueError, match="local_files_only must be true"):
        run_pipeline(config)
