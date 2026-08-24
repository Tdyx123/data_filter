from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from trajectory_data import EpisodeData, EpisodeRecord


CONTRACT = "bridge_v2_q99_binary_v1"


def _statistics():
    from octo_small_bridge.normalization import BridgeV2NormalizationStatistics

    return BridgeV2NormalizationStatistics(
        state_q01=np.asarray([-1, -2, -3, -4, -5, -6, 0, 0], dtype=np.float32),
        state_q99=np.asarray([1, 2, 3, 4, 5, 6, 0, 1], dtype=np.float32),
        action_q01=np.asarray([-1, -2, -3, -4, -5, -6, 0], dtype=np.float32),
        action_q99=np.asarray([1, 2, 3, 4, 5, 6, 1], dtype=np.float32),
        metadata_sha256="a" * 64,
        retained_episodes=2,
        retained_frames=5,
        epsilon=1.0e-6,
    )


def test_bridge_v2_normalization_clips_continuous_dims_and_binarizes_grippers():
    statistics = _statistics()

    state = statistics.normalize_state(
        np.asarray([4, -8, 0, 0, 0, 0, 9, 0.5], dtype=np.float32)
    )
    action = statistics.normalize_action(
        np.asarray([4, -8, 0, 0, 0, 0, 0.50001], dtype=np.float32)
    )

    np.testing.assert_array_equal(
        state,
        np.asarray([2.2, -2.2, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        action,
        np.asarray([2.2, -2.2, 0, 0, 0, 0, 1], dtype=np.float32),
    )


def test_bridge_v2_action_denormalization_extrapolates_and_binarizes_gripper():
    statistics = _statistics()
    normalized = np.asarray(
        [
            [2.2, -2.2, 0, 0, 0, 0, 0.5],
            [2.2, -2.2, 0, 0, 0, 0, 0.50001],
        ],
        dtype=np.float32,
    )

    actual = statistics.denormalize_action(normalized)

    np.testing.assert_allclose(
        actual[:, :6],
        np.asarray(
                [
                    [2.2, -4.4, 0, 0, 0, 0],
                    [2.2, -4.4, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        rtol=0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(actual[:, 6], np.asarray([0, 1], dtype=np.float32))


def test_bridge_v2_action_denormalization_can_preserve_continuous_gripper():
    statistics = _statistics()
    normalized = np.asarray(
        [[2.2, -2.2, 0, 0, 0, 0, 0.50001]],
        dtype=np.float32,
    )

    actual = statistics.denormalize_action(normalized, binarize_gripper=False)

    np.testing.assert_allclose(
        actual,
        np.asarray([[2.2, -4.4, 0, 0, 0, 0, 0.50001]], dtype=np.float32),
        rtol=0,
        atol=1.0e-6,
    )


class _StatisticsAdapter:
    vector_observation_keys = ("observation.state",)
    action_key = "action"

    def __init__(self) -> None:
        self.load_calls = 0
        self._record = EpisodeRecord(7, 4, 1, "move the block")

    def fingerprint(self) -> str:
        return "b" * 64

    def episodes(self):
        return (self._record,)

    def load_episode(self, record, *, load_images=True):
        assert record == self._record
        assert load_images is False
        self.load_calls += 1
        state = np.asarray(
            [
                [-3, -2, -1, 0, 1, 2, 0, 0],
                [-1, 0, 1, 2, 3, 4, 0, 0.5],
                [1, 2, 3, 4, 5, 6, 0, 0.50001],
                [3, 4, 5, 6, 7, 8, 0, 1],
            ],
            dtype=np.float32,
        )
        action = state[:, :7].copy()
        action[:, 6] = state[:, 7]
        return EpisodeData(
            episode_id=7,
            timestamps=np.arange(4, dtype=np.float64),
            frame_indices=np.arange(4, dtype=np.int64),
            observations={"observation.state": state},
            actions=action,
            task_index=1,
            task_name="move the block",
        )


def test_bridge_v2_quantiles_are_cached_for_the_filtered_dataset(tmp_path: Path):
    from octo_small_bridge.normalization import compute_bridge_v2_statistics

    adapter = _StatisticsAdapter()
    cache = tmp_path / "normalization.json"

    first = compute_bridge_v2_statistics(adapter, cache, epsilon=1.0e-6)
    second = compute_bridge_v2_statistics(adapter, cache, epsilon=1.0e-6)

    assert first.contract == CONTRACT
    assert first.metadata_sha256 == "b" * 64
    assert first.retained_episodes == 1
    assert first.retained_frames == 4
    np.testing.assert_allclose(
        first.state_q01[0], np.quantile([-3, -1, 1, 3], 0.01)
    )
    np.testing.assert_allclose(
        first.state_q99[0], np.quantile([-3, -1, 1, 3], 0.99)
    )
    assert adapter.load_calls == 1
    np.testing.assert_array_equal(second.action_q99, first.action_q99)
    assert cache.is_file()


class _FrameAdapter:
    vector_observation_keys = ("observation.state",)
    image_observation_keys = ("observation.images.image_0",)

    def __init__(self) -> None:
        self._record = EpisodeRecord(3, 2, 1, "move the block")

    def episodes(self):
        return (self._record,)

    def load_episode(self, record, *, load_images=True):
        assert record == self._record
        assert load_images is True
        return EpisodeData(
            episode_id=3,
            timestamps=np.asarray([0.0, 0.2]),
            frame_indices=np.asarray([0, 1]),
            observations={
                "observation.state": np.asarray(
                    [
                        [4, -8, 0, 0, 0, 0, 99, 0.5],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                ),
                "observation.images.image_0": np.zeros(
                    (2, 8, 8, 3), dtype=np.uint8
                ),
            },
            actions=np.asarray(
                [
                    [4, -8, 0, 0, 0, 0, 0.50001],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            task_index=1,
            task_name="move the block",
        )


def test_bridge_frame_dataset_uses_bridge_v2_statistics():
    from octo_small_bridge.data import BridgeFrameDataset, BridgeFrameRef

    dataset = BridgeFrameDataset(
        _FrameAdapter(),
        statistics=_statistics(),
        dataset_name="bridge_orig_1.0.0",
        action_horizon=1,
        primary_size=(8, 8),
        train=False,
    )

    sample = dataset[BridgeFrameRef(epoch=0, episode_id=3, frame_position=0)]

    np.testing.assert_array_equal(
        sample["proprio"][0],
        np.asarray([2.2, -2.2, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        sample["action"][0],
        np.asarray([2.2, -2.2, 0, 0, 0, 0, 1], dtype=np.float32),
    )


def test_bridge_training_gripper_helper_uses_strict_binary_boundary():
    from octo_small_bridge.data import _normalize_gripper_actions

    actual = _normalize_gripper_actions(
        np.asarray([0.0, 0.5, 0.50001, 1.0], dtype=np.float32),
        episode_id=99,
    )

    np.testing.assert_array_equal(actual, np.asarray([0, 0, 1, 1], dtype=np.float32))


def test_bridge_preflight_prepares_normalization_before_dataset_inspection(
    tmp_path: Path,
    monkeypatch,
):
    from octo_small_bridge import preflight

    calls = []
    statistics = _statistics()
    adapter = object()
    monkeypatch.setattr(preflight, "inspect_octo_checkpoint", lambda path: {"ok": True})
    monkeypatch.setattr(preflight, "_adapter", lambda path: adapter)

    def compute(loaded_adapter, path, *, epsilon):
        calls.append(("normalization", loaded_adapter, Path(path), epsilon))
        return statistics

    def inspect(path, *, adapter, statistics):
        calls.append(("dataset", Path(path), adapter, statistics))
        return {
            "source_episodes": 2,
            "retained_episodes": 1,
            "excluded_empty_task_episodes": 1,
            "retained_frames": 4,
        }

    monkeypatch.setattr(preflight, "compute_bridge_v2_statistics", compute, raising=False)
    monkeypatch.setattr(preflight, "inspect_bridge_dataset", inspect)
    fake_torch = SimpleNamespace(
        __version__="2.4.1",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
            is_bf16_supported=lambda: True,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="4.44.2"))
    config = {
        "data": {
            "normalization_epsilon": 1.0e-6,
            "expected_counts": {
                "source_episodes": 2,
                "retained_episodes": 1,
                "excluded_empty_task_episodes": 1,
                "retained_frames": 4,
            },
        },
        "train": {
            "gpu_count": 4,
            "gpu_ids": [0, 1, 2, 3],
            "batch_size": 128,
        },
    }
    paths = {
        "model": tmp_path / "model",
        "dataset": tmp_path / "dataset",
        "normalization": tmp_path / "output" / "normalization.json",
    }

    report = preflight.run_preflight(config, paths)

    assert calls == [
        ("normalization", adapter, paths["normalization"], 1.0e-6),
        ("dataset", paths["dataset"], adapter, statistics),
    ]
    assert report["normalization"]["contract"] == CONTRACT
    assert report["normalization"]["path"] == str(paths["normalization"])
