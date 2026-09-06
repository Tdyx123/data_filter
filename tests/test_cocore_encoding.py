from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from cocore.encoding import (
    CocoreNumericNormalizers,
    CocorePCAProjector,
    encode_cocore_dataset,
    fuse_fragment_features,
    temporal_pool,
    visual_half_means,
    visual_fragment_feature,
)
from cocore.index import uniform_clip_windows
from segment_filter_core.encoding import (
    NumericNormalizers,
    PCAProjector,
    fuse_fragment_features as reference_fuse_fragment_features,
)
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord


class _EncodingAdapter(DatasetAdapter):
    def __init__(self) -> None:
        self._records = (
            EpisodeRecord(0, 46, 0, "task zero"),
            EpisodeRecord(1, 15, 1, "task one"),
            EpisodeRecord(2, 7, 2, "short task"),
        )
        self.load_images_calls: list[bool] = []

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
        del num_workers
        self.load_images_calls.append(load_images)
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            state = np.zeros((record.length, 8), dtype=np.float32)
            state[:, 0] = steps * np.float32(0.04)
            observations = {"observation.state": state}
            if load_images:
                pixels = np.mod(steps + record.episode_id * 31, 255).astype(np.uint8)
                observations["observation.images.image"] = np.broadcast_to(
                    pixels[:, None, None, None], (record.length, 2, 2, 3)
                ).copy()
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations=observations,
                actions=np.stack([steps / 30.0, np.cos(steps)], axis=1),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "cocore-encoding-test-v1"


class _CountingVisualEncoder:
    output_dim = 3

    def __init__(self) -> None:
        self.episode_lengths: list[int] = []

    def encode(self, images: np.ndarray) -> np.ndarray:
        self.episode_lengths.append(len(images))
        values = images[:, 0, 0, 0].astype(np.float32)
        return np.stack([values + 1.0, values + 2.0, values + 4.0], axis=1)


class _FailingVisualEncoder(_CountingVisualEncoder):
    def encode(self, images: np.ndarray) -> np.ndarray:
        raise RuntimeError(f"injected visual failure for {len(images)} frames")


class _ActionVariationAdapter(DatasetAdapter):
    def __init__(self) -> None:
        self._records = (
            EpisodeRecord(0, 15, 0, "task zero"),
            EpisodeRecord(1, 15, 1, "task one"),
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
        del num_workers
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            action = np.zeros((record.length, 2), dtype=np.float32)
            action[5:, 0] = 1.0
            observations = {"observation.state": np.stack([steps, np.zeros_like(steps)], axis=1)}
            if load_images:
                observations["observation.images.image"] = np.broadcast_to(
                    ((record.episode_id + 1) * steps)[:, None, None, None],
                    (record.length, 2, 2, 3),
                ).astype(np.uint8)
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64),
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations=observations,
                actions=action,
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "cocore-action-variation-test-v1"


class _OverlappingVacAdapter(_ActionVariationAdapter):
    def __init__(self) -> None:
        self._records = (EpisodeRecord(0, 16, 0, "overlapping VAC"),)

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        del num_workers
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            observations = {"observation.state": np.zeros((record.length, 2), dtype=np.float32)}
            if load_images:
                pixels = np.concatenate(
                    [np.asarray([0.0], dtype=np.float32), np.arange(100.0, 115.0)]
                )
                observations["observation.images.image"] = np.broadcast_to(
                    pixels[:, None, None, None],
                    (record.length, 2, 2, 3),
                ).astype(np.uint8)
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64),
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations=observations,
                actions=np.zeros((record.length, 2), dtype=np.float32),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "cocore-overlapping-vac-test-v1"


def test_quality_style_primitives_match_shared_reference() -> None:
    actions = np.asarray(
        [[-5.0, 2.0], [0.0, 4.0], [5.0, 8.0], [10.0, 16.0]],
        dtype=np.float32,
    )
    observations = {
        "state.a": np.asarray([[-10.0], [0.0], [10.0], [20.0]], dtype=np.float32),
        "state.b": np.asarray([1.0, 3.0, 5.0, 9.0], dtype=np.float32),
    }
    episodes = [(actions, observations)]

    actual_normalizers = CocoreNumericNormalizers.fit(
        episodes,
        ("state.a", "state.b"),
        quantile_low=0.25,
        quantile_high=0.75,
        epsilon=1.0e-8,
    )
    reference_normalizers = NumericNormalizers.fit(
        episodes,
        ("state.a", "state.b"),
        quantile_low=0.25,
        quantile_high=0.75,
        epsilon=1.0e-8,
    )
    np.testing.assert_array_equal(
        actual_normalizers.action(actions), reference_normalizers.action(actions)
    )
    np.testing.assert_array_equal(
        actual_normalizers.state(observations), reference_normalizers.state(observations)
    )

    raw = np.arange(42, dtype=np.float32).reshape(7, 6)
    actual_projector = CocorePCAProjector(output_dim=8, seed=17)
    reference_projector = PCAProjector(output_dim=8, seed=17)
    np.testing.assert_allclose(
        actual_projector.fit_transform(raw, max_samples=5),
        reference_projector.fit_transform(raw, max_samples=5),
        atol=1.0e-6,
    )


def test_quality_style_pca_requires_positive_output_dimension() -> None:
    with pytest.raises(ValueError, match="output_dim must be positive"):
        CocorePCAProjector(output_dim=0)


def test_visual_half_means_use_overlapping_eight_frame_windows() -> None:
    steps = np.arange(15, dtype=np.float32)
    frames = np.stack([steps, steps**2 + 1.0], axis=1)

    actual = visual_half_means(frames)

    expected = np.stack([frames[:8].mean(axis=0), frames[7:].mean(axis=0)])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected, atol=1.0e-7)


def test_visual_half_means_accept_overlapping_four_frame_windows() -> None:
    steps = np.arange(7, dtype=np.float32)
    frames = np.stack([steps + 1.0, steps**2 + 1.0], axis=1)

    actual = visual_half_means(frames, half_windows=((0, 4), (3, 7)))

    expected = np.stack([frames[:4].mean(axis=0), frames[3:].mean(axis=0)])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected, atol=1.0e-7)


def test_cocore_encoder_uses_bridge_profile_clip_geometry(tmp_path: Path) -> None:
    encoded = encode_cocore_dataset(
        _EncodingAdapter(),
        _CountingVisualEncoder(),
        profile="bridge_v2",
        frame_cache_dir=tmp_path / "frame_embeddings",
    )

    assert len(encoded.clips) == 11
    assert {clip.length for clip in encoded.clips} == {7}
    assert encoded.state_sequences.shape == (11, 7, 8)
    assert encoded.action_sequences.shape == (11, 7, 2)
    assert encoded.visual_half_embeddings.shape == (11, 2, 3)


def test_cocore_encoder_computes_top_three_action_variation_from_full_episode(
    tmp_path: Path,
) -> None:
    encoded = encode_cocore_dataset(
        _ActionVariationAdapter(),
        _CountingVisualEncoder(),
        quantile_low=0.0,
        quantile_high=1.0,
        frame_cache_dir=tmp_path / "frame_embeddings",
    )

    np.testing.assert_allclose(
        encoded.action_variation_raw,
        [0.74666667, 0.74666667],
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(
        encoded.action_variation,
        np.zeros(2, dtype=np.float32),
    )


def test_cocore_encoder_computes_top_three_vac_from_full_episode(tmp_path: Path) -> None:
    visual = _CountingVisualEncoder()
    encoded = encode_cocore_dataset(
        _ActionVariationAdapter(),
        visual,
        quantile_low=0.0,
        quantile_high=1.0,
        epsilon=0.5,
        frame_cache_dir=tmp_path / "frame_embeddings",
    )

    np.testing.assert_allclose(
        encoded.visual_action_consistency_raw,
        [2.0 * np.sqrt(3.0), 4.0 * np.sqrt(3.0)],
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        encoded.visual_action_consistency,
        [0.0, 1.0],
        rtol=0.0,
        atol=1.0e-7,
    )
    assert visual.episode_lengths == [15, 15]


def test_overlapping_candidates_reuse_the_same_full_episode_vac(tmp_path: Path) -> None:
    visual = _CountingVisualEncoder()
    encoded = encode_cocore_dataset(
        _OverlappingVacAdapter(),
        visual,
        quantile_low=0.0,
        quantile_high=1.0,
        epsilon=0.5,
        frame_cache_dir=tmp_path / "frame_embeddings",
    )

    np.testing.assert_allclose(
        encoded.visual_action_consistency_raw,
        [134.0 * np.sqrt(3.0), 68.0 * np.sqrt(3.0)],
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(encoded.visual_action_consistency, [1.0, 0.0], atol=1.0e-7)
    assert visual.episode_lengths == [16]


def test_cocore_encoder_uses_quality_fusion_and_caches_episode_frames(
    tmp_path: Path,
) -> None:
    adapter = _EncodingAdapter()
    visual = _CountingVisualEncoder()
    cache = tmp_path / "frame_embeddings"

    cocore = encode_cocore_dataset(
        adapter,
        visual,
        visual_dim=128,
        pca_fit_max_samples=None,
        quantile_low=0.01,
        quantile_high=0.99,
        epsilon=1.0e-8,
        seed=17,
        frame_cache_dir=cache,
    )

    assert adapter.load_images_calls == [False, True]
    assert visual.episode_lengths == [46, 15, 7]
    assert [clip.start_step for clip in cocore.clips] == [0, 10, 20, 31, 0]
    assert cocore.pca_fit_fragment_count == len(cocore.clips) == 5
    assert cocore.embeddings.shape == (5, 159)
    np.testing.assert_allclose(np.linalg.norm(cocore.embeddings, axis=1), 1.0, atol=1.0e-6)
    assert [
        (entry.episode_id, entry.frames, entry.embedding_dim) for entry in cocore.frame_embeddings
    ] == [
        (0, 46, 3),
        (1, 15, 3),
        (2, 7, 3),
    ]
    frame_features = {
        entry.episode_id: np.load(cache / entry.filename, allow_pickle=False)
        for entry in cocore.frame_embeddings
    }
    assert frame_features[0].shape == (46, 3)
    assert frame_features[0].dtype == np.float32
    np.testing.assert_array_equal(
        frame_features[0][0], np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    )

    union_raw: list[np.ndarray] = []
    union_positions: dict[tuple[int, int, int], int] = {}
    for record in adapter.episodes():
        for start, end in uniform_clip_windows(record.length):
            union_positions[(record.episode_id, start, end)] = len(union_raw)
            union_raw.append(
                visual_fragment_feature(frame_features[record.episode_id][start : end + 1])
            )
    visual_projected = PCAProjector(output_dim=128, seed=17).fit_transform(np.stack(union_raw))
    candidate_visual = visual_projected[
        np.asarray(
            [
                union_positions[(clip.episode_id, clip.start_step, clip.end_step)]
                for clip in cocore.clips
            ]
        )
    ]
    state_pooled = np.stack([temporal_pool(values) for values in cocore.state_sequences])
    action_pooled = np.stack([temporal_pool(values) for values in cocore.action_sequences])
    lengths = {record.episode_id: record.length for record in adapter.episodes()}
    positions = np.asarray(
        [clip.start_step / lengths[clip.episode_id] for clip in cocore.clips],
        dtype=np.float32,
    )
    _, expected = reference_fuse_fragment_features(
        candidate_visual,
        state_pooled,
        action_pooled,
        positions,
    )
    np.testing.assert_allclose(cocore.embeddings, expected, atol=1.0e-6)
    _, local_expected = fuse_fragment_features(
        candidate_visual,
        state_pooled,
        action_pooled,
        positions,
    )
    np.testing.assert_array_equal(cocore.embeddings, local_expected)

    assert cocore.visual_half_embeddings.shape == (5, 2, 3)
    first_half = np.asarray([4.5, 5.5, 7.5], dtype=np.float32)
    second_half = np.asarray([11.5, 12.5, 14.5], dtype=np.float32)
    np.testing.assert_allclose(
        cocore.visual_half_embeddings[0, 0],
        first_half / np.linalg.norm(first_half),
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        cocore.visual_half_embeddings[0, 1],
        second_half / np.linalg.norm(second_half),
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.linalg.norm(cocore.visual_half_embeddings, axis=2),
        1.0,
        atol=1.0e-6,
    )


def test_cocore_encoder_reports_completed_step_timings(tmp_path: Path) -> None:
    events: list[tuple[str, float]] = []

    encode_cocore_dataset(
        _EncodingAdapter(),
        _CountingVisualEncoder(),
        frame_cache_dir=tmp_path / "frame_embeddings",
        timing_callback=lambda step, seconds: events.append((step, seconds)),
    )

    assert [step for step, _ in events] == [
        "encode.numeric_normalization",
        "encode.visual_cache",
        "encode.pca_fusion",
    ]
    assert all(seconds >= 0.0 for _, seconds in events)


def test_cocore_encoder_does_not_report_failed_visual_step(tmp_path: Path) -> None:
    events: list[tuple[str, float]] = []

    with pytest.raises(RuntimeError, match="injected visual failure"):
        encode_cocore_dataset(
            _EncodingAdapter(),
            _FailingVisualEncoder(),
            frame_cache_dir=tmp_path / "frame_embeddings",
            timing_callback=lambda step, seconds: events.append((step, seconds)),
        )

    assert [step for step, _ in events] == ["encode.numeric_normalization"]
