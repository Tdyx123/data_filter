from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cocore import prototypes
from relcore.schemas import ClipRecord
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord


class _TrajectoryPrototypeAdapter(DatasetAdapter):
    def __init__(self) -> None:
        self._records = (
            EpisodeRecord(0, 47, 0, "forward training"),
            EpisodeRecord(1, 47, 0, "right training"),
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


def test_action_catalog_retains_counts_at_fixed_and_fractional_thresholds() -> None:
    fixed = prototypes.create_action_catalog(
        Counter({"move forward": 40, "move right": 7_960}),
        total_labels=8_000,
    )
    fractional = prototypes.create_action_catalog(
        Counter({"move forward": 42, "move backward": 41, "move right": 8_118}),
        total_labels=8_201,
    )

    fixed_by_label = {category.label: category for category in fixed.action_categories}
    fractional_by_label = {category.label: category for category in fractional.action_categories}
    assert fixed_by_label["move forward"].retained is True
    assert fractional_by_label["move forward"].retained is True
    assert fractional_by_label["move backward"].retained is False
    assert fixed.action_labels == ("move right", "move forward", "stop")
    assert fractional.action_labels == ("move right", "move forward", "stop")


def test_action_catalog_rejects_non_stop_data_without_a_retained_non_stop_action() -> None:
    with pytest.raises(ValueError, match="no non-stop action meets retention threshold"):
        prototypes.create_action_catalog(
            Counter({"move forward": 39, "stop": 61}),
            total_labels=100,
        )


def test_action_catalog_accepts_pure_stop_data_below_the_count_floor() -> None:
    catalog = prototypes.create_action_catalog(Counter({"stop": 3}), total_labels=3)

    category = catalog.action_categories[0]
    assert category.label == "stop"
    assert category.raw_count == 3
    assert category.raw_proportion == pytest.approx(1.0)
    assert category.retained is False
    assert tuple((parent.label, parent.probability) for parent in category.parents) == (
        ("stop", 1.0),
    )
    assert catalog.action_labels == ("stop",)


def test_action_parent_distribution_uses_all_maximum_cardinality_subsets() -> None:
    distribution = prototypes.action_parent_distribution(
        "move forward right, tilt up, open gripper",
        {
            "move forward, tilt up": 40,
            "move right, tilt up": 80,
            "move forward right": 60,
            "move forward": 1_000,
            "rotate clockwise": 2_000,
        },
    )

    assert distribution == (
        ("move right, tilt up", pytest.approx(4.0 / 9.0)),
        ("move forward right", pytest.approx(3.0 / 9.0)),
        ("move forward, tilt up", pytest.approx(2.0 / 9.0)),
    )


def test_action_parent_distribution_keeps_a_retained_action_exactly() -> None:
    assert prototypes.action_parent_distribution(
        "move forward right, tilt up",
        {
            "move forward right, tilt up": 40,
            "move forward": 400,
            "move right, tilt up": 80,
        },
    ) == (("move forward right, tilt up", 1.0),)


def test_action_parent_distribution_uses_stop_only_without_a_nonempty_subset() -> None:
    assert prototypes.action_parent_distribution(
        "tilt down, close gripper",
        {"move forward": 400, "stop": 500},
    ) == (("stop", 1.0),)


def test_action_catalog_records_raw_values_and_parent_assignments() -> None:
    catalog = prototypes.create_action_catalog(
        Counter(
            {
                "move right, tilt up": 80,
                "move forward right": 60,
                "move forward, tilt up": 40,
                "move forward right, tilt up, open gripper": 10,
                "stop": 10,
            }
        ),
        total_labels=200,
    )

    by_label = {category.label: category for category in catalog.action_categories}
    rare = by_label["move forward right, tilt up, open gripper"]
    assert rare.raw_count == 10
    assert rare.raw_proportion == pytest.approx(0.05)
    assert rare.action_id is None
    assert rare.retained is False
    assert tuple((parent.label, parent.probability) for parent in rare.parents) == (
        ("move right, tilt up", pytest.approx(4.0 / 9.0)),
        ("move forward right", pytest.approx(3.0 / 9.0)),
        ("move forward, tilt up", pytest.approx(2.0 / 9.0)),
    )


@pytest.mark.parametrize(
    "label",
    [
        "move sideways",
        "move forward, wave gripper",
    ],
)
def test_action_helpers_reject_unknown_motion_labels(label: str) -> None:
    with pytest.raises(ValueError, match="unknown motion primitive label"):
        prototypes.action_parent_distribution(label, {"move forward": 40})
    with pytest.raises(ValueError, match="unknown motion primitive label"):
        prototypes.canonical_clip_action(label, "stop")


def test_canonical_clip_action_unions_deduplicates_and_uses_semantic_order() -> None:
    assert (
        prototypes.canonical_clip_action(
            "open gripper, move left up, tilt down",
            "move forward left, rotate clockwise, open gripper",
        )
        == "move forward left up, tilt down, rotate clockwise, open gripper"
    )
    assert prototypes.canonical_clip_action("stop", "stop") == "stop"


@pytest.mark.parametrize(
    ("mass", "expected"),
    [
        (1.0, 1),
        (2.0, 2),
        (8.0, 4),
        (16_384.0, 15),
        (32_768.0, 16),
        (1_000_000.0, 16),
    ],
)
def test_cluster_count_for_mass_uses_capped_logarithmic_formula(
    mass: float,
    expected: int,
) -> None:
    assert prototypes.cluster_count_for_mass(mass) == expected


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan, np.inf, -np.inf, True, "2"])
def test_cluster_count_for_mass_rejects_non_positive_or_non_finite_values(
    mass: object,
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        prototypes.cluster_count_for_mass(mass)  # type: ignore[arg-type]


def test_visual_center_probabilities_use_stable_all_center_squared_distance_softmax() -> None:
    probabilities = prototypes.visual_center_probabilities(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(
        probabilities,
        np.asarray(
            [
                [0.9999546021312976, 0.0000453978687024, 4.248161389e-18],
                [0.0000453958078295, 0.9999092083843409, 0.0000453958078295],
            ],
            dtype=np.float64,
        ),
        rtol=1.0e-12,
        atol=1.0e-18,
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities > 0.0)


def test_visual_center_probabilities_return_exact_one_for_one_center() -> None:
    probabilities = prototypes.visual_center_probabilities(
        np.asarray([[0.0, 0.0], [2.0, -3.0]], dtype=np.float32),
        np.asarray([[100.0, 100.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(probabilities, np.ones((2, 1), dtype=np.float64))


@pytest.mark.parametrize(
    ("values", "centers"),
    [
        (np.asarray([1.0, 2.0]), np.asarray([[1.0, 2.0]])),
        (np.empty((0, 2)), np.asarray([[1.0, 2.0]])),
        (np.asarray([[1.0, 2.0]]), np.empty((0, 2))),
        (np.asarray([[1.0, 2.0]]), np.asarray([[1.0, 2.0, 3.0]])),
        (np.asarray([[np.nan, 2.0]]), np.asarray([[1.0, 2.0]])),
        (np.asarray([[1.0, 2.0]]), np.asarray([[np.inf, 2.0]])),
    ],
)
def test_visual_center_probabilities_reject_malformed_or_non_finite_inputs(
    values: np.ndarray,
    centers: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="finite non-empty matrices"):
        prototypes.visual_center_probabilities(values, centers)


def test_action_catalog_serializes_schema_four_strategy_and_metadata() -> None:
    catalog = prototypes.create_action_catalog(
        Counter({"move forward": 40, "move right": 60}),
        total_labels=100,
    )
    forward = next(
        category for category in catalog.action_categories if category.label == "move forward"
    )
    updated_forward = replace(
        forward,
        effective_mass=42.5,
        requested_centers=6,
        actual_centers=5,
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

    assert payload["schema_version"] == 4
    assert payload["strategy"] == "trajectory_action_subset_then_visual_softmax"
    assert payload["constants"] == {
        "state_threshold": 0.03,
        "min_action_count": 40,
        "min_action_frequency": 0.005,
        "max_visual_centers": 16,
        "visual_softmax_temperature": 0.1,
        "cluster_count": "min(16, 1 + floor(log2(effective_mass)))",
    }
    assert payload["total_raw_actions"] == 100
    forward_payload = next(
        category for category in payload["action_categories"] if category["label"] == "move forward"
    )
    assert forward_payload["raw_count"] == 40
    assert forward_payload["raw_proportion"] == pytest.approx(0.4)
    assert forward_payload["parents"] == [
        {
            "action_id": updated_forward.action_id,
            "label": "move forward",
            "probability": 1.0,
        }
    ]
    assert forward_payload["effective_mass"] == pytest.approx(42.5)
    assert forward_payload["requested_centers"] == 6
    assert forward_payload["actual_centers"] == 5
    assert payload["leaf_prototypes"] == [
        {
            "prototype_id": 0,
            "label": "move forward::center_0",
            "action_id": updated_forward.action_id,
            "action_label": "move forward",
            "center_id": 0,
        }
    ]


def test_full_trajectory_builder_uses_weighted_parent_buckets_and_all_centers(
    tmp_path: Path,
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    clip = _candidate_clip()
    candidate_frames = np.load(cache / "ep000002.npy", allow_pickle=False)
    candidate_mean = candidate_frames.mean(axis=0)
    candidate_mean /= np.linalg.norm(candidate_mean)

    result = prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [clip],
        candidate_mean[None, :],
        frame_cache_dir=cache,
        batch_size=32,
        max_iter=50,
        seed=23,
        max_episodes=None,
        num_workers=0,
    )

    assert result.clip_action_labels.dtype.kind == "U"
    assert result.clip_action_labels.shape == (1,)
    assert result.clip_action_labels.tolist() == ["move forward right"]
    by_label = {category.label: category for category in result.catalog.action_categories}
    assert by_label["move forward"].raw_count == 42
    assert by_label["move right"].raw_count == 42
    assert by_label["move forward right"].raw_count == 4
    assert by_label["move forward"].effective_mass == pytest.approx(44.0)
    assert by_label["move right"].effective_mass == pytest.approx(44.0)
    assert by_label["move forward"].requested_centers == 6
    assert by_label["move right"].requested_centers == 6
    assert by_label["move forward"].actual_centers == 6
    assert by_label["move right"].actual_centers == 6
    assert result.prototypes.centers is not None
    assert result.prototypes.centers.shape == (12, 2)
    assert result.prototypes.indices.shape == (1, 12)
    assert set(result.prototypes.indices[0]) == set(range(12))
    assert np.all(result.prototypes.weights[0] > 0.0)
    np.testing.assert_allclose(result.prototypes.weights.sum(axis=1), 1.0)
    forward_leaf_ids = {
        leaf.prototype_id
        for leaf in result.catalog.leaf_prototypes
        if leaf.action_label == "move forward"
    }
    right_leaf_ids = {
        leaf.prototype_id
        for leaf in result.catalog.leaf_prototypes
        if leaf.action_label == "move right"
    }
    assert len(forward_leaf_ids) == 6
    assert len(right_leaf_ids) == 6
    assert sum(
        weight
        for leaf_id, weight in zip(
            result.prototypes.indices[0],
            result.prototypes.weights[0],
            strict=True,
        )
        if leaf_id in forward_leaf_ids
    ) == pytest.approx(0.5)
    assert sum(
        weight
        for leaf_id, weight in zip(
            result.prototypes.indices[0],
            result.prototypes.weights[0],
            strict=True,
        )
        if leaf_id in right_leaf_ids
    ) == pytest.approx(0.5)


def test_full_trajectory_builder_bounds_every_kmeans_update_by_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.cluster import MiniBatchKMeans

    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    candidate_frames = np.load(cache / "ep000002.npy", allow_pickle=False)
    candidate_mean = candidate_frames.mean(axis=0)
    candidate_mean /= np.linalg.norm(candidate_mean)
    observed_batch_rows: list[int] = []
    real_fit = MiniBatchKMeans.fit
    real_partial_fit = MiniBatchKMeans.partial_fit

    def recording_fit(
        self: MiniBatchKMeans,
        features: np.ndarray,
        labels: object = None,
        sample_weight: np.ndarray | None = None,
    ) -> MiniBatchKMeans:
        observed_batch_rows.append(len(features))
        return real_fit(self, features, labels, sample_weight=sample_weight)

    def recording_partial_fit(
        self: MiniBatchKMeans,
        features: np.ndarray,
        labels: object = None,
        sample_weight: np.ndarray | None = None,
    ) -> MiniBatchKMeans:
        observed_batch_rows.append(len(features))
        return real_partial_fit(self, features, labels, sample_weight=sample_weight)

    monkeypatch.setattr(MiniBatchKMeans, "fit", recording_fit)
    monkeypatch.setattr(MiniBatchKMeans, "partial_fit", recording_partial_fit)

    prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        candidate_mean[None, :],
        frame_cache_dir=cache,
        batch_size=8,
        max_iter=50,
        seed=23,
        max_episodes=None,
        num_workers=0,
    )

    assert observed_batch_rows
    assert max(observed_batch_rows) <= 8


def test_full_trajectory_builder_sorts_the_definitive_stored_probabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    candidate_frames = np.load(cache / "ep000002.npy", allow_pickle=False)
    candidate_mean = candidate_frames.mean(axis=0)
    candidate_mean /= np.linalg.norm(candidate_mean)

    def uniform_probabilities(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.full(
            (len(values), len(centers)),
            1.0 / len(centers),
            dtype=np.float64,
        )

    monkeypatch.setattr(
        prototypes,
        "visual_center_probabilities",
        uniform_probabilities,
    )

    result = prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        candidate_mean[None, :],
        frame_cache_dir=cache,
        batch_size=32,
        max_iter=50,
        seed=23,
        max_episodes=None,
        num_workers=0,
    )

    stored = list(
        zip(
            result.prototypes.indices[0].tolist(),
            result.prototypes.weights[0].tolist(),
            strict=True,
        )
    )
    assert stored == sorted(stored, key=lambda item: (-item[1], item[0]))
    assert sum(weight for _, weight in stored) == pytest.approx(1.0, abs=1.0e-7)


def test_full_trajectory_builder_uses_stable_mass_at_a_power_of_two(
    tmp_path: Path,
) -> None:
    adapter = _PowerBoundaryAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    candidate_frames = np.load(cache / "ep000000.npy", allow_pickle=False)
    candidate_mean = candidate_frames[:15].mean(axis=0)
    candidate_mean /= np.linalg.norm(candidate_mean)
    clip = ClipRecord(
        sample_id="ep000000_chunk_000000_000014",
        episode_id=0,
        task_index=0,
        task_name="forward",
        start_step=0,
        end_step=14,
        length=15,
        previous_sample_id=None,
        next_sample_id=None,
    )

    result = prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [clip],
        candidate_mean[None, :],
        frame_cache_dir=cache,
        batch_size=32,
        max_iter=50,
        seed=23,
        max_episodes=None,
        num_workers=0,
    )

    forward = next(
        category
        for category in result.catalog.action_categories
        if category.label == "move forward"
    )
    assert forward.effective_mass == 64.0
    assert forward.requested_centers == 7
    assert forward.actual_centers == 7


def test_full_trajectory_builder_rejects_missing_frame_cache(tmp_path: Path) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    (cache / "ep000001.npy").unlink()
    candidate = np.asarray([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="missing frame embedding cache"):
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [_candidate_clip()],
            candidate,
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
            np.empty((0, 2), dtype=np.float32),
            frame_cache_dir=cache,
            batch_size=32,
            max_iter=50,
            seed=23,
            max_episodes=None,
            num_workers=0,
        )
