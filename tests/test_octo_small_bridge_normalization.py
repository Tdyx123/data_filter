from __future__ import annotations

from pathlib import Path

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
