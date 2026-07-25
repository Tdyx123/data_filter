from pathlib import Path

import numpy as np
import pytest

from qwen3_vl_groot.config import load_config
from qwen3_vl_groot.data import (
    BridgeMetadata,
    Episode,
    make_action_window,
    split_episodes,
    stable_validation_episode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_episode_split_is_stable_and_disjoint():
    episodes = [Episode(index=index, length=3, tasks=("task",)) for index in range(10_000)]
    train_a, validation_a = split_episodes(episodes, seed=42, validation_percent=1)
    train_b, validation_b = split_episodes(episodes, seed=42, validation_percent=1)
    assert [item.index for item in train_a] == [item.index for item in train_b]
    assert [item.index for item in validation_a] == [item.index for item in validation_b]
    assert set(train_a).isdisjoint(validation_a)
    assert 70 <= len(validation_a) <= 130
    assert stable_validation_episode(validation_a[0].index, 42, 1)


def test_action_window_preserves_relative_actions_and_masks_tail():
    actions = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    window, mask = make_action_window(actions, frame_index=3, horizon=8)
    np.testing.assert_array_equal(window[:2], actions[3:5])
    np.testing.assert_array_equal(window[2:], np.zeros((6, 7), dtype=np.float32))
    np.testing.assert_array_equal(mask, [1, 1, 0, 0, 0, 0, 0, 0])
    # The function copies source actions verbatim; it never subtracts state.
    assert window[0, 0] == actions[3, 0]


def test_bridge_task_fallback():
    dataset_path = Path("/data/dwb/datasets/bridge_orig_1.0.0_lerobo")
    if not dataset_path.is_dir():
        pytest.skip("Bridge dataset is not available")
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    metadata = BridgeMetadata(dataset_path, config["data"])
    episode = Episode(index=999_999, length=1, tasks=("",))
    assert (
        metadata.instruction(4, episode)
        == config["data"]["fallback_instruction"]
    )


def test_bridge_split_expected_size():
    dataset_path = Path("/data/dwb/datasets/bridge_orig_1.0.0_lerobo")
    if not dataset_path.is_dir():
        pytest.skip("Bridge dataset is not available")
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    metadata = BridgeMetadata(dataset_path, config["data"])
    train, validation = metadata.split()
    assert len(train) + len(validation) == 53_192
    assert len(validation) == 559

