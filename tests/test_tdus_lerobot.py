import json
from pathlib import Path

import pytest

from tdus.dataset import LeRobotDatasetAdapter


def test_lerobot_adapter_discovers_observations_without_robot_assumptions(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    info = {
        "codebase_version": "v2.0",
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.front": {"dtype": "video", "shape": [64, 64, 3]},
            "observation.proprio": {"dtype": "float32", "shape": [5]},
            "action": {"dtype": "float32", "shape": [3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 4}) + "\n", encoding="utf-8"
    )
    adapter = LeRobotDatasetAdapter(
        {
            "path": str(tmp_path),
            "use_images": True,
            "feature_keys": {
                "vector_observations": "auto",
                "image_observations": "auto",
            },
        }
    )
    assert adapter.vector_observation_keys == ("observation.proprio",)
    assert adapter.image_observation_keys == ("observation.images.front",)


@pytest.mark.real_data
def test_real_lerobot_metadata_smoke():
    root = Path("/data/dwb/datasets/bridge_orig_1.0.0_lerobo")
    if not root.is_dir():
        pytest.skip("real LeRobot dataset is not mounted")
    adapter = LeRobotDatasetAdapter(
        {
            "path": str(root),
            "use_images": True,
            "feature_keys": {
                "vector_observations": "auto",
                "image_observations": ["observation.images.image_0"],
            },
        }
    )
    assert len(adapter.episodes()) == 53192
    assert adapter.vector_observation_keys == ("observation.state",)
    assert "observation.images.image_0" in adapter.discovered_image_keys
