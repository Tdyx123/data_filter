from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from octo_small_bridge.data import BridgeFrameDataset, BridgeFrameRef
from octo_small_bridge.preflight import inspect_bridge_dataset
from octo_small_official_pytorch.policy import load_official_action_statistics
from trajectory_data import LeRobotDatasetAdapter


BRIDGE_ROOT = Path("/data/dwb/datasets/bridge_orig_1.0.0_lerobot")
OFFICIAL_MODEL = Path("/data/dwb/models/octo-small-pytorch-official")


def _mounted_bridge_dataset() -> BridgeFrameDataset:
    adapter = LeRobotDatasetAdapter(
        {
            "path": str(BRIDGE_ROOT),
            "use_images": True,
            "empty_task_policy": "exclude",
            "feature_keys": {
                "action": "action",
                "timestamp": "timestamp",
                "frame_index": "frame_index",
                "episode_index": "episode_index",
                "vector_observations": [],
                "image_observations": ["observation.images.image_0"],
            },
        }
    )
    return BridgeFrameDataset(
        adapter,
        statistics=load_official_action_statistics(
            OFFICIAL_MODEL / "dataset_statistics.json"
        ),
        dataset_name="bridge_orig_1.0.0",
        action_horizon=4,
        history_horizon=2,
        train=False,
    )


@pytest.mark.real_data
def test_default_bridge_mount_matches_training_contract() -> None:
    if not BRIDGE_ROOT.is_dir() or not OFFICIAL_MODEL.is_dir():
        pytest.skip("default Bridge dataset or official model mount is unavailable")

    adapter = LeRobotDatasetAdapter(
        {
            "path": str(BRIDGE_ROOT),
            "use_images": True,
            "empty_task_policy": "exclude",
            "feature_keys": {
                "action": "action",
                "timestamp": "timestamp",
                "frame_index": "frame_index",
                "episode_index": "episode_index",
                "vector_observations": [],
                "image_observations": ["observation.images.image_0"],
            },
        }
    )
    report = inspect_bridge_dataset(
        BRIDGE_ROOT,
        adapter=adapter,
        statistics=load_official_action_statistics(
            OFFICIAL_MODEL / "dataset_statistics.json"
        ),
    )

    assert report["source_episodes"] == 53_192
    assert report["retained_episodes"] == 38_660
    assert report["excluded_empty_task_episodes"] == 14_532
    assert report["retained_frames"] == 1_305_714
    assert len(report["sampled_episodes"]) == 3


@pytest.mark.real_data
def test_mounted_bridge_episode_48242_frame_20_binarizes_gripper(
) -> None:
    if not BRIDGE_ROOT.is_dir() or not OFFICIAL_MODEL.is_dir():
        pytest.skip("default Bridge dataset or official model mount is unavailable")

    sample = _mounted_bridge_dataset()[
        BridgeFrameRef(epoch=0, episode_id=48_242, frame_position=20)
    ]

    assert sample["episode_index"] == 48_242
    assert sample["frame_index"] == 20
    assert sample["image_primary"].shape == (2, 3, 256, 256)
    assert sample["action"].shape == (2, 4, 7)
    assert "proprio" not in sample
    np.testing.assert_array_equal(sample["action"][1, 0, 6], np.float32(1.0))
