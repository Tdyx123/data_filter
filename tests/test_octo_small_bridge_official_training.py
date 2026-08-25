from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trajectory_data import EpisodeData, EpisodeRecord


class _OfficialStatistics:
    def __init__(self) -> None:
        from octo_small_official_pytorch.checkpoint import (
            OFFICIAL_BRIDGE_ACTION_MASK,
            OFFICIAL_BRIDGE_ACTION_MEAN,
            OFFICIAL_BRIDGE_ACTION_STD,
        )

        self.mean = np.asarray(OFFICIAL_BRIDGE_ACTION_MEAN, dtype=np.float32)
        self.std = np.asarray(OFFICIAL_BRIDGE_ACTION_STD, dtype=np.float32)
        self.mask = np.asarray(OFFICIAL_BRIDGE_ACTION_MASK, dtype=np.bool_)

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        return np.where(
            self.mask,
            (actions - self.mean) / self.std,
            actions,
        ).astype(np.float32)


class _FakePrimaryOnlyAdapter:
    vector_observation_keys: tuple[str, ...] = ()
    image_observation_keys = ("observation.images.image_0",)
    action_key = "action"

    def __init__(self, episode: EpisodeData) -> None:
        self.episode = episode

    def episodes(self) -> tuple[EpisodeRecord, ...]:
        return (
            EpisodeRecord(
                episode_id=self.episode.episode_id,
                length=self.episode.length,
                task_index=self.episode.task_index,
                task_name=self.episode.task_name,
            ),
        )

    def load_episode(self, record: EpisodeRecord, *, load_images: bool) -> EpisodeData:
        assert record.episode_id == self.episode.episode_id
        assert load_images is True
        return self.episode


def _episode(*, identical_images: bool = False) -> EpisodeData:
    image_values = [64, 64, 64, 64, 64] if identical_images else [16, 48, 80, 112, 144]
    actions = np.asarray(
        [
            [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.0],
            [0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.4],
            [0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.6],
            [0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 1.0],
            [0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.5],
        ],
        dtype=np.float32,
    )
    return EpisodeData(
        episode_id=7,
        timestamps=np.arange(5, dtype=np.float32) / np.float32(5.0),
        frame_indices=np.arange(5, dtype=np.int64),
        observations={
            "observation.images.image_0": np.stack(
                [np.full((24, 24, 3), value, dtype=np.uint8) for value in image_values]
            )
        },
        actions=actions,
        task_index=3,
        task_name="move the block",
    )


def _dataset(*, train: bool = False, identical_images: bool = False):
    from octo_small_bridge.data import BridgeFrameDataset

    return BridgeFrameDataset(
        _FakePrimaryOnlyAdapter(_episode(identical_images=identical_images)),
        statistics=_OfficialStatistics(),
        dataset_name="bridge_orig_1.0.0",
        action_horizon=4,
        history_horizon=2,
        primary_size=(16, 16),
        train=train,
        seed=19,
    )


def test_primary_only_dataset_repeats_first_frame_and_masks_padding() -> None:
    from octo_small_bridge.data import BridgeFrameRef

    sample = _dataset()[BridgeFrameRef(epoch=0, episode_id=7, frame_position=0)]

    assert sample["image_primary"].shape == (2, 3, 16, 16)
    np.testing.assert_array_equal(sample["image_primary"][0], sample["image_primary"][1])
    np.testing.assert_array_equal(sample["timestep_pad_mask"], [False, True])
    assert sample["action"].shape == (2, 4, 7)
    np.testing.assert_array_equal(sample["action"][0], sample["action"][1])
    assert "proprio" not in sample


def test_primary_only_dataset_uses_previous_and_current_readouts_with_tail_repeat() -> None:
    from octo_small_bridge.data import BridgeFrameRef

    dataset = _dataset()
    sample = dataset[BridgeFrameRef(epoch=0, episode_id=7, frame_position=4)]
    raw = dataset._load_episode(7).actions
    statistics = dataset.statistics

    expected_previous = statistics.normalize_actions(raw[3:5])
    expected_previous[:, 6] = [1.0, -1.0]
    expected_previous = np.concatenate(
        [expected_previous, np.repeat(expected_previous[-1:], 2, axis=0)]
    )
    expected_current = statistics.normalize_actions(raw[4:5])
    expected_current[:, 6] = [-1.0]
    expected_current = np.repeat(expected_current, 4, axis=0)

    np.testing.assert_array_equal(sample["timestep_pad_mask"], [True, True])
    assert not np.array_equal(sample["image_primary"][0], sample["image_primary"][1])
    np.testing.assert_allclose(sample["action"][0], expected_previous, atol=1.0e-6)
    np.testing.assert_allclose(sample["action"][1], expected_current, atol=1.0e-6)
    assert "action_pad_mask" not in sample
    assert "proprio" not in sample


def test_gripper_intermediate_values_are_removed_backward_over_whole_trajectory() -> None:
    from octo_small_bridge.data import _standardize_gripper_trajectory

    standardized = _standardize_gripper_trajectory(
        np.asarray([0.0, 0.4, 0.6, 1.0, 0.5], dtype=np.float32),
        episode_id=7,
    )

    np.testing.assert_array_equal(standardized, [-1.0, 1.0, 1.0, 1.0, -1.0])


def test_two_frame_window_uses_one_augmentation_sample() -> None:
    from octo_small_bridge.data import BridgeFrameRef

    sample = _dataset(train=True, identical_images=True)[
        BridgeFrameRef(epoch=3, episode_id=7, frame_position=2)
    ]

    np.testing.assert_array_equal(sample["image_primary"][0], sample["image_primary"][1])


def test_official_bridge_config_defaults_and_rejects_legacy_semantics() -> None:
    from octo_small_bridge.config import load_config, validate_config

    config_path = Path(__file__).resolve().parents[1] / "configs" / "octo_small_bridge_v2_4x4090.yaml"
    config = load_config(config_path)

    assert config["paths"]["model"] == "/data/dwb/models/octo-small-pytorch-official"
    assert config["data"]["window_size"] == 2
    assert config["data"]["action_horizon"] == 4
    assert config["model"]["use_proprio"] is False
    assert config["train"]["max_steps"] == 20_000
    assert "state_obs_keys" not in config["data"]
    assert "normalization_contract" not in config["data"]

    for key, value in (
        ("window_size", 1),
        ("action_horizon", 8),
    ):
        invalid = {**config, "data": {**config["data"], key: value}}
        with pytest.raises(ValueError, match=key):
            validate_config(invalid)
    invalid = {**config, "model": {**config["model"], "use_proprio": True}}
    with pytest.raises(ValueError, match="use_proprio"):
        validate_config(invalid)


def test_preflight_uses_checkpoint_statistics_without_recomputing_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from octo_small_bridge import preflight

    statistics_path = tmp_path / "dataset_statistics.json"
    statistics_path.write_text("{}", encoding="utf-8")
    report = SimpleNamespace(
        statistics_path=statistics_path,
        as_dict=lambda: {"checkpoint_kind": "official_parity"},
    )
    adapter = SimpleNamespace(
        episodes=lambda: (
            EpisodeRecord(index, 8, index, f"task {index}") for index in range(4)
        )
    )
    statistics = SimpleNamespace(
        as_dict=lambda: {
            "path": str(statistics_path),
            "sha256": "a" * 64,
            "normalization": "mean_std",
        }
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(preflight, "validate_official_checkpoint", lambda _path: report)
    monkeypatch.setattr(preflight, "load_official_action_statistics", lambda path: statistics)
    monkeypatch.setattr(preflight, "_adapter", lambda _path: adapter)

    def inspect(_root, *, adapter, statistics, frame_positions_by_episode=None):
        observed["adapter"] = adapter
        observed["statistics"] = statistics
        observed["selection"] = frame_positions_by_episode
        return {
            "source_episodes": 4,
            "retained_episodes": 4,
            "excluded_empty_task_episodes": 1,
            "retained_frames": 32,
        }

    monkeypatch.setattr(preflight, "inspect_bridge_dataset", inspect)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.10",
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 4,
                is_bf16_supported=lambda: True,
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.2"))
    config = {
        "data": {
            "prior_selection": {"prefiltered_scores": None},
            "expected_counts": {
                "source_episodes": 4,
                "retained_episodes": 4,
                "excluded_empty_task_episodes": 1,
                "retained_frames": 32,
            },
        },
        "train": {
            "gpu_count": 4,
            "gpu_ids": [0, 1, 2, 3],
            "batch_size": 128,
        },
    }

    result = preflight.run_preflight(
        config,
        {
            "model": tmp_path / "official",
            "dataset": tmp_path / "bridge",
            "output": tmp_path / "output",
        },
    )

    assert observed == {"adapter": adapter, "statistics": statistics, "selection": None}
    assert result["route"] == "octo-small-official-bridge-pytorch"
    assert result["checkpoint"] == {"checkpoint_kind": "official_parity"}
    assert result["normalization"]["path"] == str(statistics_path)
