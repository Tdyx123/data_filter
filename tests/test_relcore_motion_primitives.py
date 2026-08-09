from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from relcore.graph.motion_primitives import (
    build_motion_primitive_prototypes,
    create_primitive_catalog,
    remove_one_atomic_action,
    soften_primitive_label,
)
from relcore.schemas import ClipRecord
from trajectory_data import EpisodeData, EpisodeRecord


def test_catalog_uses_strict_frequency_threshold_and_keeps_stop_fallback() -> None:
    catalog = create_primitive_catalog(Counter({"move forward": 1, "move right": 199}), 200)

    by_label = {category.label: category for category in catalog.categories}
    assert by_label["move forward"].proportion == pytest.approx(0.005)
    assert by_label["move forward"].retained is False
    assert by_label["move forward"].prototype_id is None
    assert by_label["move right"].retained is True
    assert by_label["move right"].prototype_id == 0
    assert by_label["stop"].retained is False
    assert by_label["stop"].prototype_id == 1
    assert catalog.labels == ("move right", "stop")


def test_remove_one_atomic_action_preserves_canonical_primitive_grammar() -> None:
    assert remove_one_atomic_action(
        "move forward right, tilt up, rotate clockwise, close gripper"
    ) == (
        "move right, tilt up, rotate clockwise, close gripper",
        "move forward, tilt up, rotate clockwise, close gripper",
        "move forward right, rotate clockwise, close gripper",
        "move forward right, tilt up, close gripper",
        "move forward right, tilt up, rotate clockwise",
    )
    assert remove_one_atomic_action("move forward") == ("stop",)
    assert remove_one_atomic_action("stop") == ()


def test_soften_primitive_uses_direct_match_single_candidate_and_stop() -> None:
    proportions = {
        "move forward right": 0.30,
        "move forward": 0.20,
        "stop": 0.10,
    }

    assert soften_primitive_label("move forward right", {"move forward right"}, proportions) == (
        ("move forward right", 1.0),
    )
    assert soften_primitive_label("move forward right", {"move forward"}, proportions) == (
        ("move forward", 0.8),
    )
    assert soften_primitive_label("tilt up, open gripper", {"move forward"}, proportions) == (
        ("stop", 0.8),
    )


def test_soften_primitive_keeps_top_two_when_frequency_ratio_is_exactly_four() -> None:
    proportions = {
        "move right, tilt up": 0.40,
        "move forward, tilt up": 0.10,
        "move forward right": 0.09,
    }

    result = soften_primitive_label(
        "move forward right, tilt up",
        set(proportions),
        proportions,
    )

    assert result == (
        ("move right, tilt up", pytest.approx(0.8)),
        ("move forward, tilt up", pytest.approx(0.2)),
    )


def test_soften_primitive_keeps_only_top_candidate_above_four_to_one() -> None:
    proportions = {
        "move right, tilt up": 0.41,
        "move forward, tilt up": 0.10,
    }

    assert soften_primitive_label(
        "move forward right, tilt up",
        set(proportions),
        proportions,
    ) == (("move right, tilt up", 0.8),)


class StateAdapter:
    def __init__(self, episodes: Sequence[EpisodeData]):
        self._episodes = tuple(episodes)
        self.load_images_calls: list[bool] = []

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    def episodes(self) -> Sequence[EpisodeRecord]:
        return tuple(
            EpisodeRecord(episode.episode_id, episode.length, 0, "task")
            for episode in self._episodes
        )

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        del num_workers
        self.load_images_calls.append(load_images)
        yield from self._episodes[:max_episodes]


def _episode(episode_id: int, states: np.ndarray) -> EpisodeData:
    length = len(states)
    return EpisodeData(
        episode_id=episode_id,
        timestamps=np.arange(length, dtype=np.float64) / 10.0,
        frame_indices=np.arange(length, dtype=np.int64),
        observations={"observation.state": states},
        actions=np.zeros((length, 7), dtype=np.float32),
        task_index=0,
        task_name="task",
    )


def _clip(episode_id: int) -> ClipRecord:
    return ClipRecord(
        sample_id=f"ep{episode_id:06d}_fragment_000000_000014",
        episode_id=episode_id,
        task_index=0,
        task_name="task",
        start_step=0,
        end_step=14,
        length=15,
        previous_sample_id=None,
        next_sample_id=None,
    )


def _two_half_motion_states() -> np.ndarray:
    states = np.zeros((15, 8), dtype=np.float32)
    states[:8, 0] = np.arange(8, dtype=np.float32) * np.float32(0.01)
    states[8:, 0] = states[7, 0]
    states[7:, 1] = -np.arange(8, dtype=np.float32) * np.float32(0.01)
    return states


def test_builder_uses_episode_horizon_statistics_and_zero_seven_fourteen_anchors() -> None:
    adapter = StateAdapter([_episode(0, _two_half_motion_states())])

    prototypes, catalog = build_motion_primitive_prototypes(adapter, [_clip(0)])

    assert catalog.total_labels == 7
    assert adapter.load_images_calls == [False]
    assert prototypes.centers is None
    assert prototypes.indices.shape == (1, 4)
    assigned = [
        (prototypes.labels[index], float(weight))
        for index, weight in zip(prototypes.indices[0], prototypes.weights[0], strict=True)
        if index >= 0
    ]
    assert assigned == [("move forward", 1.0), ("move right", 1.0)]
    np.testing.assert_array_equal(prototypes.indices[0, 2:], [-1, -1])
    np.testing.assert_array_equal(prototypes.weights[0, 2:], [0.0, 0.0])


def test_builder_merges_matching_half_labels_with_maximum_weight() -> None:
    states = np.zeros((15, 8), dtype=np.float32)
    states[:, 0] = np.arange(15, dtype=np.float32) * np.float32(0.01)

    prototypes, _ = build_motion_primitive_prototypes(
        StateAdapter([_episode(0, states)]),
        [_clip(0)],
    )

    valid = prototypes.indices[0] >= 0
    assert prototypes.indices[0, valid].tolist() == [0]
    assert prototypes.weights[0, valid].tolist() == [1.0]
    assert prototypes.labels == ("move forward", "stop")


def test_builder_does_not_cross_episodes_and_honors_max_episodes() -> None:
    adapter = StateAdapter(
        [
            _episode(0, _two_half_motion_states()),
            _episode(1, _two_half_motion_states()),
        ]
    )

    prototypes, catalog = build_motion_primitive_prototypes(
        adapter,
        [_clip(0)],
        max_episodes=1,
    )

    assert catalog.total_labels == 7
    assert prototypes.indices.shape == (1, 4)


def test_builder_rejects_missing_or_too_narrow_libero_state() -> None:
    missing = _episode(0, np.zeros((15, 8), dtype=np.float32))
    missing.observations = {}
    with pytest.raises(ValueError, match="observation.state"):
        build_motion_primitive_prototypes(StateAdapter([missing]), [_clip(0)])

    narrow = _episode(0, np.zeros((15, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        build_motion_primitive_prototypes(StateAdapter([narrow]), [_clip(0)])
