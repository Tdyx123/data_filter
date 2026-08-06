from __future__ import annotations

from pathlib import Path

import numpy as np

from relcore.features.frame_cache import FrameFeatureCache, directory_sha256
from trajectory_data import EpisodeRecord


def test_frame_cache_validates_only_the_contiguous_complete_prefix(tmp_path: Path) -> None:
    records = [
        EpisodeRecord(3, 2, 0, "task"),
        EpisodeRecord(7, 3, 0, "task"),
    ]
    cache = FrameFeatureCache(tmp_path / "cache", fingerprint="frame-v1")
    cache.store(records[0], np.ones((2, 4), dtype=np.float32))

    assert cache.valid_prefix(records, output_dim=4) == 1

    cache.store(records[1], np.ones((3, 4), dtype=np.float32))
    assert cache.valid_prefix(records, output_dim=4) == 2

    with cache.feature_path(records[0]).open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x00")
    assert cache.valid_prefix(records, output_dim=4) == 0


def test_frame_cache_links_valid_features_into_published_directory(tmp_path: Path) -> None:
    record = EpisodeRecord(12, 2, 1, "task")
    cache = FrameFeatureCache(tmp_path / "cache", fingerprint="frame-v1")
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    cache.store(record, expected)
    destination = tmp_path / "published"

    names = cache.publish_features([record], destination, output_dim=3)

    assert names == ["ep000012.npy"]
    np.testing.assert_array_equal(np.load(destination / names[0]), expected)


def test_local_model_directory_hash_changes_when_weight_content_changes(tmp_path: Path) -> None:
    model = tmp_path / "clip"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    weights = model / "pytorch_model.bin"
    weights.write_bytes(b"weights-v1")
    first = directory_sha256(model)

    weights.write_bytes(b"weights-v2")

    assert directory_sha256(model) != first
