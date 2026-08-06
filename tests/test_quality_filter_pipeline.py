from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

import sqcn.pipeline as sqcn_pipeline
from quality_filter.pipeline import (
    encode_stage,
    filter_stage,
    quality_stage,
    run_pipeline,
    validate_output,
)
from trajectory_data import (
    DatasetAdapter,
    EpisodeData,
    EpisodeRecord,
    register_dataset_adapter,
)


class QualityFilterAdapter(DatasetAdapter):
    load_images_calls: list[bool] = []

    def __init__(self, _: Mapping[str, object]):
        self._records = (EpisodeRecord(0, 31),)

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
        self.load_images_calls.append(load_images)
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            observations = {
                "observation.state": np.stack(
                    [steps / 30.0, np.sin(steps)],
                    axis=1,
                ).astype(np.float32),
            }
            if load_images:
                pixels = np.mod(steps + record.episode_id, 255).astype(np.uint8)
                observations["observation.image"] = np.broadcast_to(
                    pixels[:, None, None, None],
                    (record.length, 2, 2, 3),
                ).copy()
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations=observations,
                actions=np.stack([steps / 30.0, np.cos(steps)], axis=1).astype(
                    np.float32
                ),
            )

    def fingerprint(self) -> str:
        return "quality-filter-synthetic-v1"


class ManyQualityFilterAdapter(QualityFilterAdapter):
    def __init__(self, _: Mapping[str, object]):
        self._records = tuple(EpisodeRecord(index, 30) for index in range(100))

    def fingerprint(self) -> str:
        return "quality-filter-many-synthetic-v1"


class TwoCameraQualityFilterAdapter(QualityFilterAdapter):
    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ("observation.image", "observation.image2")


def quality_filter_config(tmp_path: Path) -> dict[str, object]:
    return {
        "dataset": {
            "type": "quality_filter_synthetic",
            "name": "synthetic",
            "path": str(tmp_path / "dataset"),
            "use_images": True,
        },
        "clip": {"length": 15, "stride": 15},
        "encoder": {
            "model": "/models/local-clip-vit",
            "local_files_only": True,
            "image_batch_size": 16,
            "visual_dim": 128,
            "pca_fit_max_samples": None,
            "device": "cpu",
        },
        "quality": {
            "quantile_low": 0.0,
            "quantile_high": 1.0,
            "epsilon": 1.0e-8,
        },
        "filter": {"percent": 100.0, "seed": 17},
        "runtime": {
            "seed": 42,
            "num_workers": 0,
            "max_episodes": None,
            "resume": True,
        },
        "output": {"directory": str(tmp_path / "quality-filter-output")},
    }


def sqcn_config(tmp_path: Path) -> dict[str, object]:
    config = quality_filter_config(tmp_path)
    return {
        "dataset": config["dataset"],
        "encoder": config["encoder"],
        "quality": config["quality"],
        "coverage": {
            "sigma": 1.0,
            "batch_size": 32,
            "device": "cpu",
            "quantile_low": 0.0,
            "quantile_high": 1.0,
        },
        "novelty": {
            "k": 2,
            "backend": "numpy",
            "batch_size": 32,
            "quantile_low": 0.0,
            "quantile_high": 1.0,
        },
        "runtime": config["runtime"],
        "output": {"root": str(tmp_path / "sqcn-output")},
    }


class QualityFilterVisualEncoder:
    def encode(self, images: np.ndarray) -> np.ndarray:
        values = images[:, 0, 0, 0].astype(np.float32)
        features = np.stack([values + 1.0, np.mod(values, 7.0) + 1.0], axis=1)
        return features / np.linalg.norm(features, axis=1, keepdims=True)


def test_quality_stage_uses_two_numeric_passes_and_writes_aligned_artifacts(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("quality_filter_synthetic", QualityFilterAdapter)
    QualityFilterAdapter.load_images_calls.clear()

    root = quality_stage(quality_filter_config(tmp_path))

    assert root == tmp_path / "quality-filter-output"
    assert QualityFilterAdapter.load_images_calls == [False, False]
    with (root / "quality" / "scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = np.load(root / "quality" / "numeric_features.npz", allow_pickle=False)
    manifest = json.loads((root / "quality" / "manifest.json").read_text())
    assert len(rows) == 3
    assert list(rows[0]) == [
        "sample_id",
        "episode_id",
        "start_step",
        "end_step",
        "length",
        "action_smooth",
        "state_transition",
        "motion_efficiency",
        "quality",
    ]
    assert numeric["sample_ids"].tolist() == [row["sample_id"] for row in rows]
    assert numeric["state_pooled"].shape == (3, 6)
    assert numeric["action_pooled"].shape == (3, 6)
    assert numeric["progress"].shape == (3,)
    assert (root / "quality" / "numeric_normalizers.pkl").is_file()
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {
        "candidate_fragments": 3,
        "skipped_short_episodes": 0,
    }


def test_encode_stage_uses_reference_union_without_persisting_frame_features(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("quality_filter_synthetic", QualityFilterAdapter)
    QualityFilterAdapter.load_images_calls.clear()
    config = quality_filter_config(tmp_path)
    quality_stage(config)

    root = encode_stage(config, visual_encoder=QualityFilterVisualEncoder())

    embeddings = np.load(root / "encode" / "embeddings.npy", allow_pickle=False)
    manifest = json.loads((root / "encode" / "manifest.json").read_text())
    run_manifest = json.loads((root / "run_manifest.json").read_text())
    assert QualityFilterAdapter.load_images_calls == [False, False, True]
    assert embeddings.shape == (3, 141)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1.0e-6)
    assert (root / "encode" / "visual_pca.pkl").is_file()
    assert not (root / "encode" / "frame_features").exists()
    assert manifest["counts"] == {
        "candidate_fragments": 3,
        "reference_fragments": 2,
        "overlap_fragments": 2,
        "pca_union_fragments": 3,
    }
    assert manifest["embedding_dim"] == 141
    assert run_manifest["status"] == "complete"

    encode_stage(config, visual_encoder=QualityFilterVisualEncoder())
    assert QualityFilterAdapter.load_images_calls == [False, False, True]


def test_filter_stage_uses_quality_and_writes_ranked_aligned_artifacts(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("quality_filter_many", ManyQualityFilterAdapter)
    config = quality_filter_config(tmp_path)
    config["dataset"]["type"] = "quality_filter_many"
    encode_stage(config, visual_encoder=QualityFilterVisualEncoder())

    output = filter_stage(config, percent=51.0, seed=77)

    assert output == tmp_path / "quality-filter-output" / "filter" / "top51pct"
    with (output / "scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    embeddings = np.load(output / "embeddings.npy", allow_pickle=False)
    manifest = json.loads((output / "filter_manifest.json").read_text())
    assert len(rows) == 102
    assert embeddings.shape == (102, 141)
    assert [int(row["filter_rank"]) for row in rows] == list(range(1, 103))
    assert "sqcn" not in rows[0] and "coverage" not in rows[0] and "novelty" not in rows[0]
    assert manifest["status"] == "complete"
    assert manifest["algorithm"]["score_column"] == "quality"
    assert manifest["algorithm"]["seed"] == 77
    assert manifest["algorithm"]["target_size"] == 102
    assert manifest["algorithm"]["lambda"] == 0.5
    assert manifest["algorithm"]["update_count"]["promotion_minimum"] == (
        "ceil(100 + sqrt(selected_count - 100))"
    )
    penalties = [float(row["knn_penalty"]) for row in rows]
    assert any(penalty > 0.0 for penalty in penalties)
    for row in rows:
        assert float(row["adjusted_score"]) == pytest.approx(
            float(row["quality"]) - 0.5 * float(row["knn_penalty"]),
            abs=2.0e-9,
        )
    assert manifest["counts"] == {
        "input_fragments": 200,
        "selected_fragments": 102,
    }


def test_run_and_validate_check_every_stage_and_selected_filter(tmp_path: Path) -> None:
    register_dataset_adapter("quality_filter_many", ManyQualityFilterAdapter)
    config = quality_filter_config(tmp_path)
    config["dataset"]["type"] = "quality_filter_many"

    root = run_pipeline(
        config,
        percent=50.0,
        seed=77,
        visual_encoder=QualityFilterVisualEncoder(),
    )
    report = validate_output(root, config=config, percent=50.0)

    assert root == tmp_path / "quality-filter-output"
    assert report == {
        "status": "valid",
        "candidate_fragments": 200,
        "embedding_dim": 141,
        "filters": {"top50pct": 100},
    }

    filter_manifest_path = root / "filter" / "top50pct" / "filter_manifest.json"
    filter_manifest = json.loads(filter_manifest_path.read_text())
    filter_manifest["algorithm"]["lambda"] = 1.0
    filter_manifest_path.write_text(json.dumps(filter_manifest))
    with pytest.raises(ValueError, match="penalty lambda"):
        validate_output(root, config=config, percent=50.0)

    filter_manifest["algorithm"]["lambda"] = 0.5
    filter_manifest["algorithm"]["update_count"]["promotion_minimum"] = (
        "ceil(100 + log2(selected_count - 100))"
    )
    filter_manifest_path.write_text(json.dumps(filter_manifest))
    with pytest.raises(ValueError, match="promotion minimum"):
        validate_output(root, config=config, percent=50.0)

    filter_manifest["algorithm"]["update_count"]["promotion_minimum"] = (
        "ceil(100 + sqrt(selected_count - 100))"
    )
    filter_manifest_path.write_text(json.dumps(filter_manifest))

    embeddings_path = root / "encode" / "embeddings.npy"
    embeddings = np.load(embeddings_path, allow_pickle=False)
    np.save(embeddings_path, embeddings[:-1])
    with np.testing.assert_raises_regex(ValueError, "hash"):
        validate_output(root, config=config, percent=50.0)


def test_quality_filter_matches_sqcn_quality_only_quality_and_embeddings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    register_dataset_adapter("quality_filter_many", ManyQualityFilterAdapter)
    quality_config = quality_filter_config(tmp_path)
    quality_config["dataset"]["type"] = "quality_filter_many"
    sqcn_values = sqcn_config(tmp_path)
    sqcn_values["dataset"]["type"] = "quality_filter_many"
    monkeypatch.setattr(
        sqcn_pipeline,
        "ClipVisionEncoder",
        lambda _config: QualityFilterVisualEncoder(),
    )

    quality_root = run_pipeline(
        quality_config,
        percent=50.0,
        seed=77,
        visual_encoder=QualityFilterVisualEncoder(),
    )
    sqcn_root = sqcn_pipeline.run_pipeline(sqcn_values)
    with (quality_root / "quality" / "scores.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        quality_rows = list(csv.DictReader(handle))
    with (sqcn_root / "fragment" / "scores.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        sqcn_rows = list(csv.DictReader(handle))
    quality_embeddings = np.load(quality_root / "encode" / "embeddings.npy")
    sqcn_embeddings = np.load(sqcn_root / "fragment" / "embeddings.npy")
    quality_by_id = {
        row["sample_id"]: (float(row["quality"]), quality_embeddings[index])
        for index, row in enumerate(quality_rows)
    }
    sqcn_by_id = {
        row["sample_id"]: (float(row["quality"]), sqcn_embeddings[index])
        for index, row in enumerate(sqcn_rows)
    }
    assert set(quality_by_id) == set(sqcn_by_id)
    for sample_id, (quality, embedding) in quality_by_id.items():
        expected_quality, expected_embedding = sqcn_by_id[sample_id]
        assert quality == expected_quality
        np.testing.assert_allclose(embedding, expected_embedding, atol=1.0e-6)



def test_filter_algorithm_change_invalidates_legacy_cache(tmp_path: Path) -> None:
    register_dataset_adapter("quality_filter_many", ManyQualityFilterAdapter)
    config = quality_filter_config(tmp_path)
    config["dataset"]["type"] = "quality_filter_many"
    encode_stage(config, visual_encoder=QualityFilterVisualEncoder())
    output = filter_stage(config, percent=50.0, seed=77)

    manifest_path = output / "filter_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    legacy_payload = {
        "version": manifest["version"],
        "stage": "filter",
        "source": {
            "run_manifest": manifest["source"]["run_manifest_sha256"],
            "scores": manifest["source"]["scores_sha256"],
            "embeddings": manifest["source"]["embeddings_sha256"],
        },
        "percent": 50.0,
        "seed": 77,
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(FileExistsError, match="pass --force"):
        filter_stage(config, percent=50.0, seed=77)


def test_filter_rejects_a_target_smaller_than_initial_top_one_hundred(
    tmp_path: Path,
) -> None:
    register_dataset_adapter("quality_filter_many", ManyQualityFilterAdapter)
    config = quality_filter_config(tmp_path)
    config["dataset"]["type"] = "quality_filter_many"
    encode_stage(config, visual_encoder=QualityFilterVisualEncoder())

    with pytest.raises(ValueError, match="target_size must be at least 100"):
        filter_stage(config, percent=25.0, seed=1)


def test_filter_seed_change_reuses_upstream_and_requires_force(tmp_path: Path) -> None:
    register_dataset_adapter("quality_filter_many", ManyQualityFilterAdapter)
    config = quality_filter_config(tmp_path)
    config["dataset"]["type"] = "quality_filter_many"
    root = encode_stage(config, visual_encoder=QualityFilterVisualEncoder())
    filter_stage(config, percent=50.0, seed=77)
    quality_mtime = (root / "quality" / "manifest.json").stat().st_mtime_ns
    encode_mtime = (root / "encode" / "manifest.json").stat().st_mtime_ns

    with pytest.raises(FileExistsError, match="pass --force"):
        filter_stage(config, percent=50.0, seed=78)
    replaced = filter_stage(config, percent=50.0, seed=78, force=True)

    manifest = json.loads((replaced / "filter_manifest.json").read_text())
    assert manifest["algorithm"]["seed"] == 78
    assert (root / "quality" / "manifest.json").stat().st_mtime_ns == quality_mtime
    assert (root / "encode" / "manifest.json").stat().st_mtime_ns == encode_mtime


def test_quality_rejects_invalid_modalities_before_writing_output(tmp_path: Path) -> None:
    register_dataset_adapter("quality_filter_two_camera", TwoCameraQualityFilterAdapter)
    config = quality_filter_config(tmp_path)
    config["dataset"]["type"] = "quality_filter_two_camera"

    with pytest.raises(ValueError, match="exactly one configured image observation"):
        quality_stage(config)

    assert not (tmp_path / "quality-filter-output").exists()
