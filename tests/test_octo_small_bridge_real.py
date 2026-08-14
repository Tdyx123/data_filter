from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from octo_small_bridge.data import BridgeFrameDataset, BridgeFrameRef
from octo_small_bridge.preflight import inspect_bridge_dataset
from trajectory_data import LeRobotDatasetAdapter


BRIDGE_ROOT = Path("/data/dwb/datasets/bridge_orig_1.0.0_lerobo")


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
                "vector_observations": ["observation.state"],
                "image_observations": ["observation.images.image_0"],
            },
        }
    )
    return BridgeFrameDataset(
        adapter,
        dataset_name="bridge_orig_1.0.0",
        action_horizon=8,
        train=False,
    )


@pytest.mark.real_data
def test_default_bridge_mount_matches_training_contract() -> None:
    if not BRIDGE_ROOT.is_dir():
        pytest.skip(f"default Bridge mount is unavailable: {BRIDGE_ROOT}")

    report = inspect_bridge_dataset(BRIDGE_ROOT)

    assert report["source_episodes"] == 53_192
    assert report["retained_episodes"] == 38_660
    assert report["excluded_empty_task_episodes"] == 14_532
    assert report["retained_frames"] == 1_305_714
    assert len(report["sampled_episodes"]) == 3


@pytest.mark.real_data
def test_mounted_bridge_episode_48242_frame_20_keeps_continuous_gripper() -> None:
    if not BRIDGE_ROOT.is_dir():
        pytest.skip(f"default Bridge mount is unavailable: {BRIDGE_ROOT}")

    sample = _mounted_bridge_dataset()[
        BridgeFrameRef(epoch=0, episode_id=48_242, frame_position=20)
    ]

    assert sample["episode_index"] == 48_242
    assert sample["frame_index"] == 20
    np.testing.assert_allclose(sample["action"][0, 6], 0.85689795, atol=1.0e-6)
