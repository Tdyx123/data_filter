from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from cocore.prototypes import (
    _euclidean_distances,
    _fit_action_clusters,
    cluster_limits,
    build_hierarchical_motion_prototypes,
    create_action_catalog,
    distance_soft_weights,
    refine_action_prototypes,
    soften_action_label,
)
from relcore.schemas import ClipRecord
from trajectory_data import EpisodeData, EpisodeRecord


def test_action_catalog_uses_strict_half_percent_threshold_and_stop_fallback() -> None:
    catalog = create_action_catalog(
        Counter({"move forward": 1, "move right": 199}),
        total_labels=200,
    )

    by_label = {category.label: category for category in catalog.action_categories}
    assert by_label["move forward"].proportion == pytest.approx(0.005)
    assert by_label["move forward"].retained is False
    assert by_label["move forward"].action_id is None
    assert by_label["move right"].retained is True
    assert by_label["move right"].action_id == 0
    assert by_label["stop"].fallback is True
    assert by_label["stop"].action_id == 1
    assert catalog.action_labels == ("move right", "stop")


@pytest.mark.parametrize(
    ("proportion", "bucket_size", "expected"),
    [
        (0.01, 100, (5, 3)),
        (0.04, 100, (9, 6)),
        (1.0, 100, (18, 13)),
        (1.0, 2, (2, 2)),
    ],
)
def test_cluster_limits_floor_formulas_and_cap_to_bucket(
    proportion: float,
    bucket_size: int,
    expected: tuple[int, int],
) -> None:
    assert cluster_limits(proportion, bucket_size) == expected


def test_distance_soft_weights_use_bucket_quantiles_and_clipped_linear_mapping() -> None:
    distances = np.arange(11, dtype=np.float32).reshape(1, 11)

    weights, lower, upper = distance_soft_weights(distances)

    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(9.0)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, 1] == pytest.approx(1.0)
    assert weights[0, 5] == pytest.approx(0.65)
    assert weights[0, 9] == pytest.approx(0.3)
    assert weights[0, 10] == pytest.approx(0.3)


def test_distance_soft_weights_keep_full_confidence_when_quantiles_are_equal() -> None:
    weights, lower, upper = distance_soft_weights(
        np.full((3, 2), 0.25, dtype=np.float32)
    )

    assert lower == pytest.approx(0.25)
    assert upper == pytest.approx(0.25)
    np.testing.assert_array_equal(weights, np.ones((3, 2), dtype=np.float32))


def test_distance_soft_weights_do_not_treat_merely_close_quantiles_as_equal() -> None:
    distances = (
        np.float32(0.25) + np.arange(11, dtype=np.float32) * np.float32(1.0e-7)
    ).reshape(1, 11)

    weights, lower, upper = distance_soft_weights(distances)

    assert upper > lower
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[0, -1] == pytest.approx(0.3)


def test_soften_action_label_preserves_existing_direct_reduction_and_dominance_rules() -> None:
    proportions = {
        "move forward right, tilt up": 0.30,
        "move right, tilt up": 0.20,
        "move forward, tilt up": 0.05,
        "stop": 0.01,
    }
    retained = frozenset(proportions)

    assert soften_action_label(
        "move forward right, tilt up", retained, proportions
    ) == (("move forward right, tilt up", 1.0),)
    assert soften_action_label("tilt down", retained, proportions) == (("stop", 0.8),)
    assert soften_action_label(
        "move forward right, tilt up", retained - {"move forward right, tilt up"}, proportions
    ) == (
        ("move right, tilt up", pytest.approx(0.8)),
        ("move forward, tilt up", pytest.approx(0.2)),
    )


def test_refinement_keeps_top_m_leaf_weights_and_multiplies_two_soft_stages() -> None:
    catalog = create_action_catalog(
        Counter({"move forward": 2, "move right": 198}),
        total_labels=200,
    )
    forward_id = catalog.action_labels.index("move forward")
    first_indices = np.full((6, 2, 1), -1, dtype=np.int32)
    first_indices[:, 0, 0] = forward_id
    first_weights = np.zeros((6, 2, 1), dtype=np.float32)
    first_weights[:, 0, 0] = np.asarray([1.0, 0.8, 0.6, 1.0, 0.8, 0.6])
    embeddings = np.asarray(
        [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98], [-1.0, 0.0], [-0.98, 0.02]],
        dtype=np.float32,
    )
    visual_halves = np.stack([embeddings, np.zeros_like(embeddings)], axis=1)

    result = refine_action_prototypes(
        catalog,
        first_indices,
        first_weights,
        visual_halves,
        batch_size=6,
        max_iter=100,
        seed=7,
    )

    by_label = {category.label: category for category in result.catalog.action_categories}
    assert by_label["move forward"].bucket_size == 6
    assert by_label["move forward"].actual_clusters == 5
    assert by_label["move forward"].top_m == 3
    assert by_label["stop"].fallback is True
    assert by_label["stop"].actual_clusters == 1
    assert result.prototypes.centers.shape == (6, 2)
    assert result.prototypes.indices.shape == (6, 3)
    for row in range(6):
        valid = result.prototypes.indices[row] >= 0
        assert valid.sum() == 3
        leaf_ids = result.prototypes.indices[row, valid]
        assert all(
            result.catalog.leaf_prototypes[int(leaf_id)].action_label == "move forward"
            for leaf_id in leaf_ids
        )
        np.testing.assert_allclose(result.action_weights[row, valid], first_weights[row, 0, 0])
        assert np.all(result.distance_weights[row, valid][:-1] >= result.distance_weights[row, valid][1:])
        np.testing.assert_allclose(
            result.prototypes.weights[row, valid],
            result.action_weights[row, valid] * result.distance_weights[row, valid],
        )


def test_refinement_normalizes_encode_embeddings_before_clustering() -> None:
    catalog = create_action_catalog(Counter({"move forward": 200}), total_labels=200)
    action_id = catalog.action_labels.index("move forward")
    indices = np.full((6, 2, 1), -1, dtype=np.int32)
    indices[:, 0, 0] = action_id
    weights = np.zeros((6, 2, 1), dtype=np.float32)
    weights[:, 0, 0] = 1.0
    unit = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.6, 0.8], [-0.6, 0.8]],
        dtype=np.float32,
    )
    scaled = unit * np.asarray([[2.0], [3.0], [4.0], [5.0], [6.0], [7.0]], dtype=np.float32)
    unit_halves = np.stack([unit, np.zeros_like(unit)], axis=1)
    scaled_halves = np.stack([scaled, np.zeros_like(scaled)], axis=1)

    expected = refine_action_prototypes(
        catalog, indices, weights, unit_halves, batch_size=6, max_iter=100, seed=5
    )
    actual = refine_action_prototypes(
        catalog, indices, weights, scaled_halves, batch_size=6, max_iter=100, seed=5
    )

    np.testing.assert_allclose(actual.prototypes.centers, expected.prototypes.centers)
    np.testing.assert_array_equal(actual.prototypes.indices, expected.prototypes.indices)
    np.testing.assert_allclose(actual.prototypes.weights, expected.prototypes.weights)


def test_refinement_uses_aligned_half_visuals_and_deduplicates_same_leaf() -> None:
    catalog = create_action_catalog(
        Counter({"move forward": 2, "move right": 198}),
        total_labels=200,
    )
    forward_id = catalog.action_labels.index("move forward")
    half_indices = np.full((3, 2, 1), forward_id, dtype=np.int32)
    half_weights = np.ones((3, 2, 1), dtype=np.float32)
    front = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    visual_halves = np.stack([front, front], axis=1)

    result = refine_action_prototypes(
        catalog,
        half_indices,
        half_weights,
        visual_halves,
        batch_size=6,
        max_iter=100,
        seed=13,
    )

    by_label = {category.label: category for category in result.catalog.action_categories}
    assert by_label["move forward"].bucket_size == 6
    assert by_label["move forward"].actual_clusters == 5
    assert by_label["move forward"].top_m == 3
    assert result.prototypes.indices.shape == (3, 3)
    assert np.all((result.prototypes.indices >= 0).sum(axis=1) == 3)

    changed_back = visual_halves.copy()
    changed_back[:, 1] = np.asarray(
        [[0.25, -1.0], [0.5, -1.0], [0.75, -1.0]], dtype=np.float32
    )
    front_only_indices = half_indices.copy()
    front_only_indices[:, 1] = -1
    front_only_weights = half_weights.copy()
    front_only_weights[:, 1] = 0.0
    expected = refine_action_prototypes(
        catalog,
        front_only_indices,
        front_only_weights,
        visual_halves,
        batch_size=6,
        max_iter=100,
        seed=13,
    )
    actual = refine_action_prototypes(
        catalog,
        front_only_indices,
        front_only_weights,
        changed_back,
        batch_size=6,
        max_iter=100,
        seed=13,
    )
    np.testing.assert_array_equal(actual.prototypes.indices, expected.prototypes.indices)
    np.testing.assert_allclose(actual.prototypes.centers, expected.prototypes.centers)


class _StateAdapter:
    def __init__(self, episode: EpisodeData) -> None:
        self.episode = episode
        self.load_images_calls: list[bool] = []

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    def episodes(self) -> Sequence[EpisodeRecord]:
        return (EpisodeRecord(self.episode.episode_id, self.episode.length, 0, "task"),)

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        del num_workers
        self.load_images_calls.append(load_images)
        if max_episodes is None or max_episodes > 0:
            yield self.episode


def _forward_episode_and_clips() -> tuple[EpisodeData, list[ClipRecord]]:
    length = 90
    steps = np.arange(length, dtype=np.float32)
    states = np.zeros((length, 8), dtype=np.float32)
    states[:, 0] = steps * np.float32(0.04)
    episode = EpisodeData(
        episode_id=0,
        timestamps=steps.astype(np.float64) / 10.0,
        frame_indices=np.arange(length, dtype=np.int64),
        observations={"observation.state": states},
        actions=np.zeros((length, 7), dtype=np.float32),
        task_index=0,
        task_name="task",
    )
    clips = [
        ClipRecord(
            sample_id=f"ep000000_fragment_{start:06d}_{start + 14:06d}",
            episode_id=0,
            task_index=0,
            task_name="task",
            start_step=start,
            end_step=start + 14,
            length=15,
            previous_sample_id=(
                None if start == 0 else f"ep000000_fragment_{start - 15:06d}_{start - 1:06d}"
            ),
            next_sample_id=(
                None if start == 75 else f"ep000000_fragment_{start + 15:06d}_{start + 29:06d}"
            ),
        )
        for start in range(0, 90, 15)
    ]
    return episode, clips


def test_full_builder_uses_half_visual_embeddings_inside_one_action_bucket() -> None:
    episode, clips = _forward_episode_and_clips()
    adapter = _StateAdapter(episode)
    angles = np.arange(12, dtype=np.float32) * np.float32(np.pi / 6.0)
    embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    visual_halves = embeddings.reshape(6, 2, 2)

    separated = build_hierarchical_motion_prototypes(
        adapter,
        clips,
        visual_halves,
        batch_size=6,
        max_iter=100,
        seed=11,
        max_episodes=None,
        num_workers=0,
    )
    collapsed = build_hierarchical_motion_prototypes(
        _StateAdapter(episode),
        clips,
        np.tile(np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32), (6, 1, 1)),
        batch_size=6,
        max_iter=100,
        seed=11,
        max_episodes=None,
        num_workers=0,
    )

    assert adapter.load_images_calls == [False]
    assert len(set(separated.prototypes.indices[:, 0].tolist())) > 1
    assert len(set(collapsed.prototypes.indices[:, 0].tolist())) == 1
    assert separated.catalog.labels[:-1] == tuple(
        f"move forward::cluster_{cluster_id}" for cluster_id in range(12)
    )
    assert separated.catalog.labels[-1] == "stop::fallback"


def test_full_builder_counts_only_the_two_halves_of_each_clip() -> None:
    episode, clips = _forward_episode_and_clips()
    adapter = _StateAdapter(episode)
    visual_halves = np.ones((len(clips), 2, 2), dtype=np.float32)

    result = build_hierarchical_motion_prototypes(
        adapter,
        clips,
        visual_halves,
        batch_size=12,
        max_iter=100,
        seed=19,
        max_episodes=None,
        num_workers=0,
    )

    by_label = {category.label: category for category in result.catalog.action_categories}
    assert result.catalog.total_labels == 2 * len(clips)
    assert by_label["move forward"].count == 2 * len(clips)
    assert result.half_action_labels.shape == (len(clips), 2)
    assert set(result.half_action_labels.ravel()) == {"move forward"}


def test_action_kmeans_uses_first_stage_soft_weights() -> None:
    center = _fit_action_clusters(
        np.asarray([[0.0], [10.0]], dtype=np.float32),
        np.asarray([9.0, 1.0], dtype=np.float32),
        clusters=1,
        batch_size=2,
        max_iter=100,
        seed=3,
    )

    assert center.shape == (1, 1)
    assert center[0, 0] < 2.0


def test_euclidean_distance_matrix_matches_hand_computed_values() -> None:
    distances = _euclidean_distances(
        np.asarray([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[0.0, 4.0], [3.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(
        distances,
        np.asarray([[4.0, 3.0], [3.0, 4.0]], dtype=np.float32),
        rtol=1.0e-6,
    )
