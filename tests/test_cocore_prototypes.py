from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from cocore import prototypes
from relcore.schemas import ClipRecord
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord


class _TrajectoryPrototypeAdapter(DatasetAdapter):
    def __init__(self, *, combined_candidate: bool = False) -> None:
        self.combined_candidate = combined_candidate
        self._records = (
            EpisodeRecord(0, 1205, 0, "forward training"),
            EpisodeRecord(1, 1205, 0, "right training"),
            EpisodeRecord(2, 15, 0, "union candidate"),
        )

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ("observation.images.image",)

    def episodes(self) -> Sequence[EpisodeRecord]:
        return self._records

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        del num_workers, load_images
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            states = np.zeros((record.length, 8), dtype=np.float32)
            if record.episode_id == 0:
                states[:, 0] = steps * np.float32(0.01)
            elif record.episode_id == 1:
                states[:, 1] = steps * np.float32(-0.01)
            elif self.combined_candidate:
                states[:, 0] = steps * np.float32(0.01)
                states[:, 1] = steps * np.float32(-0.01)
            else:
                states[:, 0] = np.minimum(steps, 7.0) * np.float32(0.02)
                states[:, 1] = np.maximum(steps - 7.0, 0.0) * np.float32(-0.02)
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations={"observation.state": states},
                actions=np.zeros((record.length, 2), dtype=np.float32),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "trajectory-prototype-test-v1"


class _PowerBoundaryAdapter(DatasetAdapter):
    def __init__(self) -> None:
        self._records = (
            EpisodeRecord(0, 47, 0, "forward"),
            EpisodeRecord(1, 52, 0, "right"),
            EpisodeRecord(2, 33, 0, "tilt up rare"),
            EpisodeRecord(3, 32, 0, "tilt down rare"),
        )

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ("observation.images.image",)

    def episodes(self) -> Sequence[EpisodeRecord]:
        return self._records

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        del num_workers, load_images
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            states = np.zeros((record.length, 8), dtype=np.float32)
            if record.episode_id == 0:
                states[:, 0] = steps * np.float32(0.01)
            elif record.episode_id == 1:
                states[:, 1] = steps * np.float32(-0.01)
            else:
                states[:, 0] = steps * np.float32(0.01)
                states[:, 1] = steps * np.float32(-0.01)
                tilt_sign = 1.0 if record.episode_id == 2 else -1.0
                states[:, 4] = steps * np.float32(0.01 * tilt_sign)
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations={"observation.state": states},
                actions=np.zeros((record.length, 2), dtype=np.float32),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "power-boundary-prototype-test-v1"


class _StreamingMeanAdapter(DatasetAdapter):
    def __init__(self, *, reverse: bool = False) -> None:
        records = (
            EpisodeRecord(0, 27, 0, "zero block"),
            EpisodeRecord(1, 27, 0, "ten block"),
        )
        self._records = tuple(reversed(records)) if reverse else records

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ("observation.images.image",)

    def episodes(self) -> Sequence[EpisodeRecord]:
        return self._records

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        del num_workers, load_images
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations={"observation.state": np.zeros((record.length, 8), dtype=np.float32)},
                actions=np.zeros((record.length, 2), dtype=np.float32),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "streaming-mean-prototype-test-v1"


class _StopFallbackAdapter(DatasetAdapter):
    def __init__(self, *, include_stop: bool) -> None:
        records = [EpisodeRecord(0, 1205, 0, "forward training")]
        if include_stop:
            records.append(EpisodeRecord(1, 15, 0, "stop training"))
        records.append(EpisodeRecord(2, 15, 0, "right fallback candidate"))
        self._records = tuple(records)

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ("observation.images.image",)

    def episodes(self) -> Sequence[EpisodeRecord]:
        return self._records

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        del num_workers, load_images
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            states = np.zeros((record.length, 8), dtype=np.float32)
            if record.episode_id == 0:
                states[:, 0] = steps * np.float32(0.01)
            elif record.episode_id == 2:
                states[:, 1] = steps * np.float32(-0.01)
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations={"observation.state": states},
                actions=np.zeros((record.length, 2), dtype=np.float32),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "stop-fallback-prototype-test-v1"


def _write_frame_caches(root: Path, adapter: DatasetAdapter) -> None:
    root.mkdir()
    for record in adapter.episodes():
        steps = np.arange(record.length, dtype=np.float32)
        features = np.stack(
            [
                np.ones(record.length, dtype=np.float32),
                np.float32(record.episode_id + 1) + steps / np.float32(100.0),
            ],
            axis=1,
        )
        np.save(root / f"ep{record.episode_id:06d}.npy", features)


def _identity_fragment_pca(frame_dim: int) -> np.ndarray:
    return np.concatenate(
        [
            np.eye(frame_dim, dtype=np.float32),
            np.zeros((frame_dim, frame_dim), dtype=np.float32),
        ],
        axis=1,
    )


def _candidate_clip() -> ClipRecord:
    return ClipRecord(
        sample_id="ep000002_chunk_000000_000014",
        episode_id=2,
        task_index=0,
        task_name="union candidate",
        start_step=0,
        end_step=14,
        length=15,
        previous_sample_id=None,
        next_sample_id=None,
    )


def test_schema_seven_action_catalog_uses_400_count_floor() -> None:
    fixed = prototypes.create_action_catalog(
        Counter({"move forward": 400, "move right": 79_600}),
        total_labels=80_000,
    )
    fractional = prototypes.create_action_catalog(
        Counter({"move forward": 402, "move backward": 401, "move right": 79_398}),
        total_labels=80_201,
    )

    fixed_by_label = {category.label: category for category in fixed.action_categories}
    fractional_by_label = {category.label: category for category in fractional.action_categories}
    assert fixed_by_label["move forward"].retained is True
    assert fractional_by_label["move forward"].retained is True
    assert fractional_by_label["move backward"].retained is False


def test_maximum_retained_parents_keeps_all_largest_atomic_subsets() -> None:
    parents = prototypes.maximum_retained_parents(
        "move forward right, tilt up, open gripper",
        {
            "move forward, tilt up": 400,
            "move right, tilt up": 800,
            "move forward right": 600,
            "move forward": 10_000,
            "rotate clockwise": 20_000,
        },
    )

    assert parents == (
        "move right, tilt up",
        "move forward right",
        "move forward, tilt up",
    )


@pytest.mark.parametrize(
    ("raw_label", "parent_label", "expected"),
    [
        ("move forward right", "move forward right", 1.0),
        ("move forward right", "move forward", 0.75),
        ("move forward right, open gripper", "move forward", 2.0 / 3.0),
        ("move forward", "stop", 0.5),
        ("stop", "stop", 1.0),
    ],
)
def test_retention_weight_linearly_maps_atomic_subset_ratio(
    raw_label: str,
    parent_label: str,
    expected: float,
) -> None:
    assert prototypes.retention_weight(raw_label, parent_label) == pytest.approx(expected)


def test_action_catalog_rejects_non_stop_data_without_a_retained_non_stop_action() -> None:
    with pytest.raises(ValueError, match="no non-stop action meets retention threshold"):
        prototypes.create_action_catalog(
            Counter({"move forward": 399, "stop": 601}),
            total_labels=1_000,
        )


def test_action_catalog_accepts_pure_stop_data_below_the_count_floor() -> None:
    catalog = prototypes.create_action_catalog(Counter({"stop": 3}), total_labels=3)

    category = catalog.action_categories[0]
    assert category.label == "stop"
    assert category.raw_count == 3
    assert category.raw_proportion == pytest.approx(1.0)
    assert category.retained is False
    assert category.training_count == 3
    assert catalog.action_labels == ("stop",)


def test_action_catalog_records_only_exact_training_memberships() -> None:
    catalog = prototypes.create_action_catalog(
        Counter(
            {
                "move right, tilt up": 800,
                "move forward right": 600,
                "move forward, tilt up": 400,
                "move forward right, tilt up, open gripper": 100,
                "stop": 100,
            }
        ),
        total_labels=2_000,
    )

    by_label = {category.label: category for category in catalog.action_categories}
    rare = by_label["move forward right, tilt up, open gripper"]
    assert rare.raw_count == 100
    assert rare.raw_proportion == pytest.approx(0.05)
    assert rare.action_id is None
    assert rare.retained is False
    assert rare.training_count == 0
    assert by_label["move right, tilt up"].training_count == 800
    assert by_label["stop"].training_count == 100


@pytest.mark.parametrize(
    "label",
    [
        "move sideways",
        "move forward, wave gripper",
    ],
)
def test_action_helpers_reject_unknown_motion_labels(label: str) -> None:
    with pytest.raises(ValueError, match="unknown motion primitive label"):
        prototypes.maximum_retained_parents(label, {"move forward": 400})
    with pytest.raises(ValueError, match="unknown motion primitive label"):
        prototypes.retention_weight(label, "stop")


@pytest.mark.parametrize(
    ("training_count", "expected"),
    [
        (1.0, 1),
        (2.0, 2),
        (3.0, 3),
        (400.0, 3),
        (1_023.0, 3),
        (1_024.0, 4),
        (65_536.0, 16),
        (1_000_000.0, 16),
    ],
)
def test_cluster_count_for_training_count_uses_capped_logarithmic_formula(
    training_count: float,
    expected: int,
) -> None:
    assert prototypes.cluster_count_for_training_count(training_count) == expected


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan, np.inf, -np.inf, True, "2"])
def test_cluster_count_for_training_count_rejects_non_positive_or_non_finite_values(
    mass: object,
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        prototypes.cluster_count_for_training_count(mass)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("trajectory_length", "expected"),
    [
        (7, ()),
        (8, (0,)),
        (9, (0, 1)),
        (10, (0, 2)),
        (11, (0, 3)),
        (12, (0, 2, 4)),
        (13, (0, 3, 5)),
        (14, (0, 3, 6)),
        (15, (0, 3, 5, 7)),
    ],
)
def test_trajectory_window_starts_cover_tail_with_rebalanced_gaps(
    trajectory_length: int,
    expected: tuple[int, ...],
) -> None:
    assert prototypes.trajectory_window_starts(trajectory_length) == expected


def test_trajectory_window_starts_preserve_full_coverage_and_gap_policy() -> None:
    for trajectory_length in range(8, 501):
        starts = prototypes.trajectory_window_starts(trajectory_length)
        gaps = tuple(right - left for left, right in zip(starts, starts[1:], strict=False))

        assert starts[0] == 0
        assert starts[-1] == trajectory_length - 8
        assert tuple(sorted(set(starts))) == starts
        assert all(1 <= gap <= 3 for gap in gaps)
        if trajectory_length == 9:
            assert gaps == (1,)
        elif trajectory_length % 3 == 0:
            assert gaps[-2:] == (2, 2)
        elif trajectory_length % 3 == 1:
            assert gaps[-1:] == (2,)
        elif gaps:
            assert all(gap == 3 for gap in gaps)


def test_nearest_distance_bounds_and_confidence_use_q10_q90_linear_mapping() -> None:
    lower, upper = prototypes.nearest_distance_bounds(np.arange(11, dtype=np.float32))

    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(9.0)
    assert prototypes.distance_confidence(0.0, lower, upper) == pytest.approx(1.0)
    assert prototypes.distance_confidence(1.0, lower, upper) == pytest.approx(1.0)
    assert prototypes.distance_confidence(5.0, lower, upper) == pytest.approx(0.65)
    assert prototypes.distance_confidence(9.0, lower, upper) == pytest.approx(0.3)
    assert prototypes.distance_confidence(10.0, lower, upper) == pytest.approx(0.3)


def test_distance_confidence_is_one_when_bounds_are_equal() -> None:
    assert prototypes.distance_confidence(0.25, 0.25, 0.25) == pytest.approx(1.0)
    assert prototypes.distance_confidence(100.0, 0.25, 0.25) == pytest.approx(1.0)


def test_merge_half_leaf_assignments_combines_only_the_same_leaf() -> None:
    assert prototypes.merge_half_leaf_assignments(((3, 0.8), (3, 0.4))) == (
        (3, pytest.approx(1.0)),
    )
    assert prototypes.merge_half_leaf_assignments(((3, 0.8), (4, 0.4))) == (
        (3, pytest.approx(0.8)),
        (4, pytest.approx(0.4)),
    )


def test_merge_half_leaf_assignments_breaks_serialized_float32_ties_by_leaf_id() -> None:
    assignments = prototypes.merge_half_leaf_assignments(((9, 0.50000001), (2, 0.5)))

    assert assignments == ((2, 0.5), (9, 0.5))


def test_clustering_projection_uses_only_sum_block_and_zero_pads() -> None:
    values = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, -1.0], [-1.0, 2.0]],
        ],
        dtype=np.float32,
    )
    components = np.asarray(
        [
            [1.0, 2.0, 100.0, 200.0],
            [3.0, 4.0, 300.0, 400.0],
        ],
        dtype=np.float32,
    )

    actual = prototypes._project_clustering_features(
        values,
        components,
        output_dim=4,
    )

    expected = np.asarray(
        [
            [[1.0, 3.0, 0.0, 0.0], [2.0, 4.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0, 0.0], [3.0, 5.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("values", "components", "output_dim", "message"),
    [
        (
            np.ones((2, 2), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            4,
            "twice the frame dimension",
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            np.ones((5, 4), dtype=np.float32),
            4,
            "cannot exceed output dimension",
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            np.asarray([[1.0, np.nan, 2.0, 3.0]], dtype=np.float32),
            4,
            "finite",
        ),
        (
            np.asarray([[1.0, np.inf]], dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
            4,
            "finite",
        ),
    ],
)
def test_clustering_projection_rejects_incompatible_inputs(
    values: np.ndarray,
    components: np.ndarray,
    output_dim: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prototypes._project_clustering_features(
            values,
            components,
            output_dim=output_dim,
        )


def test_episode_window_visuals_project_each_frame_before_pooling(tmp_path: Path) -> None:
    cache = tmp_path / "frame_embeddings"
    cache.mkdir()
    frames = np.stack(
        [np.arange(8, dtype=np.float32), np.ones(8, dtype=np.float32)],
        axis=1,
    )
    np.save(cache / "ep000000.npy", frames)
    record = EpisodeRecord(0, 8, 0, "projection")
    components = np.asarray(
        [
            [1.0, 0.0, 100.0, 200.0],
            [0.0, 2.0, 300.0, 400.0],
        ],
        dtype=np.float32,
    )

    actual = prototypes._episode_window_visuals(
        cache,
        record,
        pca_components=components,
        visual_dim=4,
    )

    expected = np.asarray([[3.5, 2.0, 0.0, 0.0]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)


def test_action_catalog_serializes_schema_eight_sampling_strategy_and_metadata() -> None:
    catalog = prototypes.create_action_catalog(
        Counter({"move forward": 400, "move right": 600}),
        total_labels=1_000,
    )
    forward = next(
        category for category in catalog.action_categories if category.label == "move forward"
    )
    updated_forward = replace(
        forward,
        requested_centers=6,
        actual_centers=5,
        nearest_distance_q10=0.1,
        nearest_distance_q90=0.9,
    )
    catalog = replace(
        catalog,
        action_categories=tuple(
            updated_forward if category.label == "move forward" else category
            for category in catalog.action_categories
        ),
        leaf_prototypes=(
            prototypes.LeafPrototype(
                prototype_id=0,
                label="move forward::center_0",
                action_id=int(updated_forward.action_id),
                action_label="move forward",
                center_id=0,
            ),
        ),
    )

    payload = catalog.to_dict()

    assert payload["schema_version"] == 8
    assert payload["strategy"] == (
        "trajectory_sampled_retained_action_then_cropped_pca_half_visual_hybrid_kmeans_nearest"
    )
    assert payload["constants"] == {
        "state_threshold": 0.03,
        "min_action_count": 400,
        "min_action_frequency": 0.005,
        "max_visual_centers": 16,
        "full_kmeans_max_training_count": 65536,
        "full_kmeans_openmp_threads": 1,
        "minibatch_kmeans_openmp_threads": 4,
        "kmeans_n_init": 1,
        "large_bucket_parallelism": "serial",
        "trajectory_window_length": 8,
        "trajectory_window_policy": "full_coverage_max_gap_3_tail_rebalanced",
        "visual_half_windows": [[0, 8], [7, 15]],
        "visual_projection": "frame @ visual_pca.components[:, :frame_embedding_dim].T",
        "visual_projection_centering": "none",
        "visual_projection_padding": "right_zero_to_128",
        "visual_half_encoding": "l2_normalized_mean_of_eight_projected_frames",
        "cluster_count": (
            "min(training_count, min(16, max(3, "
            "floor(2 * log2(training_count) - 16))))"
        ),
        "retention_weight": "0.5 + 0.5 * retained_atomic_ratio",
        "distance_quantiles": [0.1, 0.9],
        "distance_weight_range": [1.0, 0.3],
        "duplicate_merge": "max + 0.5 * min",
    }
    assert payload["total_raw_actions"] == 1_000
    forward_payload = next(
        category for category in payload["action_categories"] if category["label"] == "move forward"
    )
    assert forward_payload["raw_count"] == 400
    assert forward_payload["raw_proportion"] == pytest.approx(0.4)
    assert forward_payload["training_count"] == 400
    assert forward_payload["requested_centers"] == 6
    assert forward_payload["actual_centers"] == 5
    assert forward_payload["nearest_distance_q10"] == pytest.approx(0.1)
    assert forward_payload["nearest_distance_q90"] == pytest.approx(0.9)
    assert payload["leaf_prototypes"] == [
        {
            "prototype_id": 0,
            "label": "move forward::center_0",
            "action_id": updated_forward.action_id,
            "action_label": "move forward",
            "center_id": 0,
        }
    ]


def test_full_trajectory_builder_trains_exact_buckets_and_labels_each_half_once(
    tmp_path: Path,
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    clip = _candidate_clip()
    candidate_frames = np.load(cache / "ep000002.npy", allow_pickle=False)
    candidate_halves = np.stack(
        [candidate_frames[:8].mean(axis=0), candidate_frames[7:].mean(axis=0)]
    )
    candidate_halves /= np.linalg.norm(candidate_halves, axis=1, keepdims=True)
    components = np.asarray(
        [[1.0, 0.0, 900.0, 900.0], [0.0, 1.0, 900.0, 900.0]],
        dtype=np.float32,
    )

    result = prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [clip],
        candidate_halves[None, :, :],
        pca_components=components,
        visual_dim=128,
        frame_cache_dir=cache,
        batch_size=32,
        max_iter=2,
        seed=23,
        max_episodes=None,
        num_workers=0,
    )

    assert result.half_action_labels.dtype.kind == "U"
    assert result.half_action_labels.shape == (1, 2)
    assert result.half_action_labels.tolist() == [["move forward", "move right"]]
    by_label = {category.label: category for category in result.catalog.action_categories}
    assert by_label["move forward"].training_count == by_label["move forward"].raw_count
    assert by_label["move right"].training_count == by_label["move right"].raw_count
    assert by_label["move forward right"].training_count == 0
    assert by_label["move forward"].requested_centers == 3
    assert by_label["move right"].requested_centers == 3
    assert by_label["move forward"].actual_centers == 3
    assert by_label["move right"].actual_centers == 3
    assert by_label["move forward"].nearest_distance_q10 is not None
    assert by_label["move forward"].nearest_distance_q90 is not None
    assert result.prototypes.centers is not None
    assert result.prototypes.centers.shape == (6, 128)
    np.testing.assert_array_equal(result.prototypes.centers[:, 2:], 0.0)
    assert result.prototypes.indices.shape == (1, 2)
    assert np.all(result.prototypes.indices[0] >= 0)
    assert np.all((result.prototypes.weights[0] >= 0.3) & (result.prototypes.weights[0] <= 1.0))
    assigned_actions = {
        result.catalog.leaf_prototypes[int(leaf_id)].action_label
        for leaf_id in result.prototypes.indices[0]
    }
    assert assigned_actions == {"move forward", "move right"}


def test_full_trajectory_builder_materializes_each_episode_visual_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    candidate_frames = np.load(cache / "ep000002.npy", allow_pickle=False)
    candidate_halves = np.stack(
        [candidate_frames[:8].mean(axis=0), candidate_frames[7:].mean(axis=0)]
    )
    candidate_halves /= np.linalg.norm(candidate_halves, axis=1, keepdims=True)
    calls: Counter[int] = Counter()
    materialized: dict[int, prototypes._ActionTrainingData] = {}
    real_episode_window_visuals = prototypes._episode_window_visuals
    real_materialize = prototypes._materialize_action_training_data

    def counting_episode_window_visuals(
        cache_root: Path,
        record: EpisodeRecord,
        *,
        pca_components: np.ndarray,
        visual_dim: int,
    ) -> np.ndarray:
        calls[record.episode_id] += 1
        return real_episode_window_visuals(
            cache_root,
            record,
            pca_components=pca_components,
            visual_dim=visual_dim,
        )

    def capture_materialized(*args: object, **kwargs: object):
        result = real_materialize(*args, **kwargs)
        materialized.update(result)
        return result

    monkeypatch.setattr(prototypes, "_episode_window_visuals", counting_episode_window_visuals)
    monkeypatch.setattr(prototypes, "_materialize_action_training_data", capture_materialized)

    prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        candidate_halves[None, :, :],
        pca_components=_identity_fragment_pca(2),
        visual_dim=2,
        frame_cache_dir=cache,
        batch_size=32,
        max_iter=100,
        seed=23,
        max_episodes=None,
        num_workers=0,
    )

    assert calls == Counter({0: 1, 1: 1, 2: 1})
    assert set(materialized) == {0, 1}
    for action_id, training_data in materialized.items():
        assert training_data.action_id == action_id
        assert training_data.values.shape == (401, 2)
        assert training_data.values.dtype == np.float32
        assert training_data.values.flags.c_contiguous
        assert training_data.episode_ranges == ((0, 400), (400, 401))


def test_full_trajectory_builder_runs_action_fit_and_statistics_in_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    candidate_frames = np.load(cache / "ep000002.npy", allow_pickle=False)
    candidate_halves = np.stack(
        [candidate_frames[:8].mean(axis=0), candidate_frames[7:].mean(axis=0)]
    )
    candidate_halves /= np.linalg.norm(candidate_halves, axis=1, keepdims=True)
    fit_barrier = threading.Barrier(2, timeout=5.0)
    statistics_barrier = threading.Barrier(2, timeout=5.0)
    fit_threads: set[int] = set()
    statistics_threads: set[int] = set()
    real_fit_action_model = prototypes._fit_action_model
    real_compute_statistics = prototypes._compute_action_center_statistics

    def synchronized_fit(*args: object, **kwargs: object) -> object:
        fit_threads.add(threading.get_ident())
        fit_barrier.wait()
        return real_fit_action_model(*args, **kwargs)

    def synchronized_statistics(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        statistics_threads.add(threading.get_ident())
        statistics_barrier.wait()
        return real_compute_statistics(*args, **kwargs)

    monkeypatch.setattr(prototypes, "_fit_action_model", synchronized_fit)
    monkeypatch.setattr(
        prototypes,
        "_compute_action_center_statistics",
        synchronized_statistics,
    )

    prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        candidate_halves[None, :, :],
        pca_components=_identity_fragment_pca(2),
        visual_dim=2,
        frame_cache_dir=cache,
        batch_size=32,
        max_iter=2,
        seed=23,
        max_episodes=None,
        num_workers=0,
        num_threads=2,
    )

    assert len(fit_threads) == 2
    assert len(statistics_threads) == 2


def test_full_trajectory_builder_matches_single_and_multi_thread_outputs(
    tmp_path: Path,
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    candidate_frames = np.load(cache / "ep000002.npy", allow_pickle=False)
    candidate_halves = np.stack(
        [candidate_frames[:8].mean(axis=0), candidate_frames[7:].mean(axis=0)]
    )
    candidate_halves /= np.linalg.norm(candidate_halves, axis=1, keepdims=True)

    results = [
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [_candidate_clip()],
            candidate_halves[None, :, :],
            pca_components=_identity_fragment_pca(2),
            visual_dim=2,
            frame_cache_dir=cache,
            batch_size=32,
            max_iter=3,
            seed=23,
            max_episodes=None,
            num_workers=0,
            num_threads=num_threads,
        )
        for num_threads in (1, 4)
    ]

    assert results[0].catalog.to_dict() == results[1].catalog.to_dict()
    np.testing.assert_array_equal(
        results[0].prototypes.centers,
        results[1].prototypes.centers,
    )
    np.testing.assert_array_equal(
        results[0].prototypes.indices,
        results[1].prototypes.indices,
    )
    np.testing.assert_array_equal(
        results[0].prototypes.weights,
        results[1].prototypes.weights,
    )
    np.testing.assert_array_equal(results[0].half_action_labels, results[1].half_action_labels)


@pytest.mark.parametrize("num_threads", [0, -1, 1.5, True, "4"])
def test_full_trajectory_builder_rejects_invalid_thread_counts(
    tmp_path: Path,
    num_threads: object,
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)

    with pytest.raises(ValueError, match="thread count must be a positive integer"):
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [_candidate_clip()],
            np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
            pca_components=_identity_fragment_pca(2),
            visual_dim=2,
            frame_cache_dir=cache,
            batch_size=32,
            max_iter=2,
            seed=23,
            max_episodes=None,
            num_workers=0,
            num_threads=num_threads,  # type: ignore[arg-type]
        )


def test_full_trajectory_builder_propagates_lowest_action_failure_and_closes_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    barrier = threading.Barrier(2, timeout=5.0)

    def failing_fit(training_data: object, **_: object) -> object:
        barrier.wait()
        action_id = int(getattr(training_data, "action_id"))
        raise RuntimeError(f"fit failed for action {action_id}")

    monkeypatch.setattr(prototypes, "_fit_action_model", failing_fit)

    with pytest.raises(RuntimeError, match="fit failed for action 0"):
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [_candidate_clip()],
            np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
            pca_components=_identity_fragment_pca(2),
            visual_dim=2,
            frame_cache_dir=cache,
            batch_size=32,
            max_iter=2,
            seed=23,
            max_episodes=None,
            num_workers=0,
            num_threads=2,
        )

    assert not any(
        thread.name.startswith("cocore-prototypes") for thread in threading.enumerate()
    )


def test_full_trajectory_builder_reports_aggregate_step_timings(tmp_path: Path) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    candidate_frames = np.load(cache / "ep000002.npy", allow_pickle=False)
    candidate_halves = np.stack(
        [candidate_frames[:8].mean(axis=0), candidate_frames[7:].mean(axis=0)]
    )
    candidate_halves /= np.linalg.norm(candidate_halves, axis=1, keepdims=True)
    events: list[tuple[str, float]] = []

    prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        candidate_halves[None, :, :],
        pca_components=_identity_fragment_pca(2),
        visual_dim=2,
        frame_cache_dir=cache,
        batch_size=32,
        max_iter=2,
        seed=23,
        max_episodes=None,
        num_workers=0,
        timing_callback=lambda step, seconds: events.append((step, seconds)),
    )

    assert [step for step, _ in events] == [
        "graph.prototypes.action_scan",
        "graph.prototypes.training_data",
        "graph.prototypes.kmeans",
        "graph.prototypes.center_statistics",
        "graph.prototypes.candidate_assignment",
    ]
    assert all(seconds >= 0.0 for _, seconds in events)


@pytest.mark.parametrize(
    ("candidate_visual", "expected_action"),
    [
        ([0.0, 1.0], "move right"),
        ([2.0**-0.5, 2.0**-0.5], "move forward"),
    ],
)
def test_tied_maximum_parent_actions_choose_global_nearest_leaf_then_leaf_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_visual: list[float],
    expected_action: str,
) -> None:
    adapter = _TrajectoryPrototypeAdapter(combined_candidate=True)

    def block_visuals(
        cache_root: Path,
        record: EpisodeRecord,
        *,
        pca_components: np.ndarray,
        visual_dim: int,
    ) -> np.ndarray:
        del cache_root
        np.testing.assert_array_equal(pca_components, _identity_fragment_pca(2))
        assert visual_dim == 2
        values = {
            0: np.asarray([1.0, 0.0], dtype=np.float32),
            1: np.asarray([0.0, 1.0], dtype=np.float32),
            2: np.asarray(candidate_visual, dtype=np.float32),
        }[record.episode_id]
        window_count = {0: 400, 1: 400, 2: 4}[record.episode_id]
        return np.broadcast_to(values, (window_count, 2)).copy()

    monkeypatch.setattr(prototypes, "_episode_window_visuals", block_visuals)
    half = np.asarray(candidate_visual, dtype=np.float32)
    result = prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        np.stack([half, half])[None, :, :],
        pca_components=_identity_fragment_pca(2),
        visual_dim=2,
        frame_cache_dir=tmp_path,
        batch_size=32,
        max_iter=2,
        seed=23,
        max_episodes=None,
        num_workers=0,
    )

    assert result.half_action_labels.tolist() == [["move forward right", "move forward right"]]
    assert result.prototypes.indices[0, 1] == -1
    assert result.prototypes.weights[0].tolist() == pytest.approx([1.125, 0.0])
    leaf = result.catalog.leaf_prototypes[int(result.prototypes.indices[0, 0])]
    assert leaf.action_label == expected_action


def test_non_stop_half_falls_back_to_stop_with_absolute_merged_confidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StopFallbackAdapter(include_stop=True)

    def block_visuals(
        cache_root: Path,
        record: EpisodeRecord,
        *,
        pca_components: np.ndarray,
        visual_dim: int,
    ) -> np.ndarray:
        del cache_root
        np.testing.assert_array_equal(pca_components, _identity_fragment_pca(2))
        assert visual_dim == 2
        values = {
            0: np.asarray([1.0, 0.0], dtype=np.float32),
            1: np.asarray([0.0, 1.0], dtype=np.float32),
            2: np.asarray([0.0, -1.0], dtype=np.float32),
        }[record.episode_id]
        window_count = {0: 400, 1: 4, 2: 4}[record.episode_id]
        return np.broadcast_to(values, (window_count, 2)).copy()

    monkeypatch.setattr(prototypes, "_episode_window_visuals", block_visuals)
    result = prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        np.asarray([[[0.0, -1.0], [0.0, -1.0]]], dtype=np.float32),
        pca_components=_identity_fragment_pca(2),
        visual_dim=2,
        frame_cache_dir=tmp_path,
        batch_size=32,
        max_iter=2,
        seed=23,
        max_episodes=None,
        num_workers=0,
    )

    by_label = {category.label: category for category in result.catalog.action_categories}
    assert by_label["stop"].training_count == 4
    assert by_label["stop"].actual_centers == 3
    assert result.half_action_labels.tolist() == [["move right", "move right"]]
    assert result.prototypes.indices[0, 0] >= 0
    assert result.prototypes.indices[0, 1] == -1
    assert result.prototypes.weights[0].tolist() == pytest.approx([0.75, 0.0])
    leaf = result.catalog.leaf_prototypes[int(result.prototypes.indices[0, 0])]
    assert leaf.action_label == "stop"


def test_non_stop_fallback_fails_when_stop_has_no_visual_center(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StopFallbackAdapter(include_stop=False)

    def block_visuals(
        cache_root: Path,
        record: EpisodeRecord,
        *,
        pca_components: np.ndarray,
        visual_dim: int,
    ) -> np.ndarray:
        del cache_root
        np.testing.assert_array_equal(pca_components, _identity_fragment_pca(2))
        assert visual_dim == 2
        values = np.asarray(
            [1.0, 0.0] if record.episode_id == 0 else [0.0, -1.0],
            dtype=np.float32,
        )
        window_count = {0: 400, 2: 4}[record.episode_id]
        return np.broadcast_to(values, (window_count, 2)).copy()

    monkeypatch.setattr(prototypes, "_episode_window_visuals", block_visuals)
    with pytest.raises(ValueError, match="missing visual centers for parent action 'stop'"):
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [_candidate_clip()],
            np.asarray([[[0.0, -1.0], [0.0, -1.0]]], dtype=np.float32),
            pca_components=_identity_fragment_pca(2),
            visual_dim=2,
            frame_cache_dir=tmp_path,
            batch_size=32,
            max_iter=2,
            seed=23,
            max_episodes=None,
            num_workers=0,
        )


@pytest.mark.parametrize(
    ("training_count", "expected_model_name", "expected_openmp_threads"),
    [
        (65_536, "KMeans", 1),
        (65_537, "MiniBatchKMeans", 4),
    ],
)
def test_action_model_switches_at_large_bucket_boundary_and_converges_early(
    training_count: int,
    expected_model_name: str,
    expected_openmp_threads: int,
) -> None:
    values = np.zeros((training_count, 1), dtype=np.float32)
    training_data = prototypes._ActionTrainingData(
        action_id=0,
        values=values,
        episode_ranges=((0, training_count),),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        model = prototypes._fit_action_models(
            executor,
            {0: training_data},
            {0: 1},
            batch_size=4096,
            max_iter=100,
            tol=1.0e-4,
            seed=23,
        )[0]

    assert type(model).__name__ == expected_model_name
    assert model.n_init == 1
    assert model.n_iter_ < 100
    assert model._n_threads == expected_openmp_threads


def test_large_action_buckets_run_serially_after_parallel_small_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_thread = threading.get_ident()
    training_data = {
        0: prototypes._ActionTrainingData(
            action_id=0,
            values=np.zeros((2, 1), dtype=np.float32),
            episode_ranges=((0, 2),),
        ),
        1: prototypes._ActionTrainingData(
            action_id=1,
            values=np.zeros((65_537, 1), dtype=np.float32),
            episode_ranges=((0, 65_537),),
        ),
        2: prototypes._ActionTrainingData(
            action_id=2,
            values=np.zeros((65_538, 1), dtype=np.float32),
            episode_ranges=((0, 65_538),),
        ),
    }
    active_openmp_threads: list[int | None] = [None]
    calls: list[tuple[int, int, int | None]] = []

    @contextmanager
    def recording_threadpool_limits(*, limits: int, user_api: str):
        assert user_api == "openmp"
        active_openmp_threads[0] = limits
        try:
            yield
        finally:
            active_openmp_threads[0] = None

    def recording_fit(training: object, **_: object) -> object:
        action_id = int(getattr(training, "action_id"))
        calls.append((action_id, threading.get_ident(), active_openmp_threads[0]))
        return SimpleNamespace(cluster_centers_=np.zeros((1, 1), dtype=np.float32))

    monkeypatch.setattr(prototypes, "threadpool_limits", recording_threadpool_limits)
    monkeypatch.setattr(prototypes, "_fit_action_model", recording_fit)

    with ThreadPoolExecutor(max_workers=2) as executor:
        models = prototypes._fit_action_models(
            executor,
            training_data,
            {0: 1, 1: 1, 2: 1},
            batch_size=4096,
            max_iter=100,
            tol=1.0e-4,
            seed=23,
        )

    assert set(models) == {0, 1, 2}
    assert calls[0][0] == 0
    assert calls[0][1] != main_thread
    assert calls[0][2] == 1
    assert calls[1:] == [(1, main_thread, 4), (2, main_thread, 4)]


def test_parallel_full_kmeans_accepts_degenerate_buckets_without_warning() -> None:
    from sklearn.exceptions import ConvergenceWarning

    training_data = {
        action_id: prototypes._ActionTrainingData(
            action_id=action_id,
            values=np.zeros((400, 2), dtype=np.float32),
            episode_ranges=((0, 400),),
        )
        for action_id in range(4)
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=4) as executor:
            models = prototypes._fit_action_models(
                executor,
                training_data,
                {action_id: 3 for action_id in training_data},
                batch_size=64,
                max_iter=100,
                tol=1.0e-4,
                seed=23,
            )

    assert all(model.cluster_centers_.shape == (3, 2) for model in models.values())
    assert not any(isinstance(item.message, ConvergenceWarning) for item in caught)


def test_full_trajectory_builder_rejects_missing_frame_cache(tmp_path: Path) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    (cache / "ep000001.npy").unlink()
    candidate = np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)

    with pytest.raises(ValueError, match="missing frame embedding cache"):
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [_candidate_clip()],
            candidate,
            pca_components=_identity_fragment_pca(2),
            visual_dim=2,
            frame_cache_dir=cache,
            batch_size=32,
            max_iter=50,
            seed=23,
            max_episodes=None,
            num_workers=0,
        )


def test_full_trajectory_builder_rejects_data_without_valid_windows(
    tmp_path: Path,
) -> None:
    class _ShortAdapter(_TrajectoryPrototypeAdapter):
        def __init__(self) -> None:
            self._records = (EpisodeRecord(0, 7, 0, "short"),)

    adapter = _ShortAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)

    with pytest.raises(ValueError, match="no valid trajectory windows"):
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [],
            np.empty((0, 2, 2), dtype=np.float32),
            pca_components=_identity_fragment_pca(2),
            visual_dim=2,
            frame_cache_dir=cache,
            batch_size=32,
            max_iter=50,
            seed=23,
            max_episodes=None,
            num_workers=0,
        )
