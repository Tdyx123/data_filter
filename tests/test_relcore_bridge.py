from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from relcore.config import load_config, resolve_config
from relcore.data.index import build_clip_records
from trajectory_data import DatasetValidationError, LeRobotDatasetAdapter


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_episode(root: Path, episode_id: int, task_index: int, length: int = 3) -> None:
    chunk = root / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array(
                    np.arange(length * 2, dtype=np.float32).reshape(length, 2).tolist(),
                    type=pa.list_(pa.float32(), list_size=2),
                ),
                "action": pa.array(
                    np.ones((length, 2), dtype=np.float32).tolist(),
                    type=pa.list_(pa.float32(), list_size=2),
                ),
                "timestamp": pa.array(np.arange(length, dtype=np.float32)),
                "frame_index": pa.array(np.arange(length, dtype=np.int64)),
                "episode_index": pa.array([episode_id] * length, type=pa.int64()),
                "task_index": pa.array([task_index] * length, type=pa.int64()),
            }
        ),
        chunk / f"episode_{episode_id:06d}.parquet",
    )


def _bridge_fixture(root: Path) -> dict[str, object]:
    meta = root / "meta"
    meta.mkdir(parents=True)
    tasks = [
        {"task_index": 0, "task": ""},
        {"task_index": 1, "task": "named one"},
        {"task_index": 2, "task": "named two"},
    ]
    episodes = [
        {"episode_index": 0, "tasks": [""], "length": 3},
        {"episode_index": 1, "tasks": ["named one"], "length": 3},
        {"episode_index": 2, "tasks": ["named two"], "length": 3},
    ]
    _write_jsonl(meta / "tasks.jsonl", tasks)
    _write_jsonl(meta / "episodes.jsonl", episodes)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "total_episodes": 3,
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
                ),
                "video_path": None,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [2]},
                    "action": {"dtype": "float32", "shape": [2]},
                    "timestamp": {"dtype": "float32", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": [1]},
                },
            }
        ),
        encoding="utf-8",
    )
    for episode_id in range(3):
        _write_episode(root, episode_id, episode_id)
    return {
        "type": "lerobot",
        "path": str(root),
        "use_images": False,
        "feature_keys": {"vector_observations": ["observation.state"]},
    }


def test_lerobot_empty_task_policy_is_strict_by_default(tmp_path: Path) -> None:
    config = _bridge_fixture(tmp_path / "bridge")

    with pytest.raises(DatasetValidationError, match="Empty task name"):
        LeRobotDatasetAdapter(config)


def test_lerobot_excludes_empty_task_episodes_before_indexing(tmp_path: Path) -> None:
    config = _bridge_fixture(tmp_path / "bridge")
    config["empty_task_policy"] = "exclude"

    adapter = LeRobotDatasetAdapter(config)

    assert [record.episode_id for record in adapter.episodes()] == [1, 2]
    assert adapter.dataset_summary() == {
        "source_episodes": 3,
        "indexed_episodes": 2,
        "retained_episodes": 2,
        "excluded_episodes": 1,
        "excluded_empty_task_episodes": 1,
    }


def test_lerobot_exclude_policy_still_rejects_missing_task_metadata(tmp_path: Path) -> None:
    root = tmp_path / "bridge"
    config = _bridge_fixture(root)
    config["empty_task_policy"] = "exclude"
    rows = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()]
    rows[1].pop("tasks")
    _write_jsonl(root / "meta" / "episodes.jsonl", rows)

    with pytest.raises(DatasetValidationError, match="exactly one task"):
        LeRobotDatasetAdapter(config)


def test_lerobot_subset_reads_only_requested_records_in_requested_order(tmp_path: Path) -> None:
    config = _bridge_fixture(tmp_path / "bridge")
    config["empty_task_policy"] = "exclude"
    adapter = LeRobotDatasetAdapter(config)
    by_id = {record.episode_id: record for record in adapter.episodes()}

    episodes = list(
        adapter.iter_episode_subset(
            [by_id[2], by_id[1]],
            num_workers=0,
            load_images=False,
        )
    )

    assert [episode.episode_id for episode in episodes] == [2, 1]
    assert [episode.task_name for episode in episodes] == ["named two", "named one"]


def test_relcore_config_defaults_to_strict_empty_task_handling() -> None:
    assert resolve_config({})["dataset"]["empty_task_policy"] == "error"


def test_relcore_config_rejects_unknown_empty_task_policy() -> None:
    with pytest.raises(ValueError, match="empty_task_policy"):
        resolve_config({"dataset": {"empty_task_policy": "keep"}})


def test_relcore_config_accepts_global_selection_only_with_zero_minimum() -> None:
    resolved = resolve_config(
        {"selection": {"quota_mode": "none", "minimum_per_task": 0}}
    )

    assert resolved["selection"]["quota_mode"] == "none"
    with pytest.raises(ValueError, match="minimum_per_task=0"):
        resolve_config({"selection": {"quota_mode": "none", "minimum_per_task": 1}})


def test_bridge_production_config_selects_named_data_globally_on_one_gpu() -> None:
    config = load_config(Path("relcore/config_bridge.yaml"))

    assert config["dataset"]["path"] == "/data/dwb/datasets/bridge_orig_1.0.0_lerobot"
    assert config["dataset"]["empty_task_policy"] == "exclude"
    assert config["dataset"]["feature_keys"]["image_observations"] == [
        "observation.images.image_0"
    ]
    assert config["visual"]["device"] == "cuda"
    assert config["selection"]["ratio"] == 0.10
    assert config["selection"]["quota_mode"] == "none"
    assert config["selection"]["minimum_per_task"] == 0


@pytest.mark.real_data
def test_mounted_bridge_metadata_matches_relcore_acceptance_counts() -> None:
    root = Path("/data/dwb/datasets/bridge_orig_1.0.0_lerobot")
    if not root.is_dir():
        pytest.skip("Bridge LeRobot dataset is not mounted")
    config = load_config(Path("relcore/config_bridge.yaml"))
    adapter = LeRobotDatasetAdapter(config["dataset"])
    records = list(adapter.episodes())

    assert adapter.dataset_summary() == {
        "source_episodes": 53_192,
        "indexed_episodes": 38_660,
        "retained_episodes": 38_660,
        "excluded_episodes": 14_532,
        "excluded_empty_task_episodes": 14_532,
    }
    assert sum(record.length for record in records) == 1_305_714
    assert sum(record.length < 15 for record in records) == 537
    usable = [record for record in records if record.length >= 15]
    assert len(usable) == 38_123
    assert sum(record.length for record in usable) == 1_299_118
    clips = build_clip_records(records, length=15, stride=15)
    assert len(clips) == 106_625
    assert int(len(clips) * 0.10 + 0.5) == 10_663
