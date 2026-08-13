from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from cocore.encoding import encode_cocore_dataset, visual_half_means
from relcore.features.encoding import encode_dataset as encode_relcore_dataset
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord


class _EncodingAdapter(DatasetAdapter):
    def __init__(self) -> None:
        self._records = (
            EpisodeRecord(0, 31, 0, "task zero"),
            EpisodeRecord(1, 15, 1, "task one"),
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


def test_visual_half_means_share_the_middle_frame() -> None:
    frame_features = np.arange(30, dtype=np.float32).reshape(15, 2)

    result = visual_half_means(frame_features)

    np.testing.assert_array_equal(
        result,
        np.asarray([[7.0, 8.0], [21.0, 22.0]], dtype=np.float32),
    )


def test_cocore_encoder_matches_relation_encoding_and_adds_visual_halves() -> None:
    relcore = encode_relcore_dataset(
        _EncodingAdapter(),
        _CountingVisualEncoder(),
        projection_dim=4,
        output_dim=8,
        lags=(0, 1, 2, 4),
        seed=17,
    )
    adapter = _EncodingAdapter()
    visual = _CountingVisualEncoder()

    cocore = encode_cocore_dataset(
        adapter,
        visual,
        projection_dim=4,
        output_dim=8,
        lags=(0, 1, 2, 4),
        seed=17,
    )

    assert adapter.load_images_calls == [False, True]
    assert visual.episode_lengths == [31, 15]
    assert [clip.sample_id for clip in cocore.clips] == [
        clip.sample_id for clip in relcore.clips
    ]
    np.testing.assert_array_equal(cocore.raw_relations, relcore.raw_relations)
    np.testing.assert_array_equal(cocore.embeddings, relcore.embeddings)
    np.testing.assert_array_equal(cocore.state_sequences, relcore.state_sequences)
    np.testing.assert_array_equal(cocore.action_sequences, relcore.action_sequences)
    np.testing.assert_array_equal(cocore.visual_progress, relcore.visual_progress)
    assert cocore.visual_half_embeddings.shape == (4, 2, 3)
    np.testing.assert_array_equal(
        cocore.visual_half_embeddings[0],
        np.asarray([[4.5, 5.5, 7.5], [11.5, 12.5, 14.5]], dtype=np.float32),
    )
