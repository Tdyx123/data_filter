from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from relcore.features.encoding import encode_dataset, fit_numeric_normalizers
from relcore.features.visual_encoder import FrozenClipEncoder
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord


class CountingAdapter(DatasetAdapter):
    def __init__(self):
        self._records = (
            EpisodeRecord(0, 31, 2, "task two"),
            EpisodeRecord(1, 15, 5, "task five"),
            EpisodeRecord(2, 7, 6, "short task"),
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
            observations = {
                "observation.state": np.stack(
                    [steps, steps / max(record.length - 1, 1)], axis=1
                ).astype(np.float32)
            }
            if load_images:
                pixels = np.mod(steps + record.episode_id * 17, 255).astype(np.uint8)
                observations["observation.images.image"] = np.broadcast_to(
                    pixels[:, None, None, None],
                    (record.length, 2, 2, 3),
                ).copy()
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 10.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations=observations,
                actions=np.stack([steps, np.cos(steps)], axis=1).astype(np.float32),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "counting-adapter-v1"


class CountingVisualEncoder:
    output_dim = 3

    def __init__(self):
        self.episode_lengths: list[int] = []

    def encode(self, images: np.ndarray) -> np.ndarray:
        self.episode_lengths.append(len(images))
        values = images[:, 0, 0, 0].astype(np.float32)
        return np.stack([values + 1.0, values + 2.0, values + 3.0], axis=1)


def test_encode_dataset_uses_numeric_then_image_pass_and_one_visual_call_per_episode(
    capsys: pytest.CaptureFixture[str],
):
    adapter = CountingAdapter()
    visual = CountingVisualEncoder()

    encoded = encode_dataset(
        adapter,
        visual,
        clip_length=15,
        clip_stride=15,
        projection_dim=4,
        output_dim=8,
        lags=(0, 1, 2, 4),
        seed=11,
        progress_interval=1,
    )

    assert adapter.load_images_calls == [False, True]
    assert visual.episode_lengths == [31, 15]
    assert [clip.sample_id for clip in encoded.clips] == [
        "ep000000_fragment_000000_000014",
        "ep000000_fragment_000015_000029",
        "ep000000_fragment_000016_000030",
        "ep000001_fragment_000000_000014",
    ]
    assert encoded.embeddings.shape == (4, 8)
    np.testing.assert_allclose(
        np.linalg.norm(encoded.embeddings, axis=1),
        np.ones(4),
        atol=1.0e-6,
    )
    assert encoded.state_sequences.shape == (4, 15, 2)
    assert encoded.action_sequences.shape == (4, 15, 2)
    assert capsys.readouterr().err.splitlines() == [
        "relcore_encode completed=1 remaining=1",
        "relcore_encode completed=2 remaining=0",
    ]


def test_encode_dataset_is_deterministic_for_same_seed():
    first = encode_dataset(
        CountingAdapter(),
        CountingVisualEncoder(),
        projection_dim=4,
        output_dim=8,
        seed=23,
    )
    second = encode_dataset(
        CountingAdapter(),
        CountingVisualEncoder(),
        projection_dim=4,
        output_dim=8,
        seed=23,
    )

    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    np.testing.assert_array_equal(first.raw_relations, second.raw_relations)


def test_supplied_normalizers_match_direct_encoding() -> None:
    direct = encode_dataset(
        CountingAdapter(),
        CountingVisualEncoder(),
        projection_dim=4,
        output_dim=8,
        seed=23,
    )
    adapter = CountingAdapter()
    records = list(adapter.episodes())
    action_normalizer, state_normalizer = fit_numeric_normalizers(adapter, records)
    supplied = encode_dataset(
        adapter,
        CountingVisualEncoder(),
        projection_dim=4,
        output_dim=8,
        seed=23,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
    )

    assert [clip.sample_id for clip in supplied.clips] == [
        clip.sample_id for clip in direct.clips
    ]
    np.testing.assert_array_equal(supplied.raw_relations, direct.raw_relations)
    np.testing.assert_array_equal(supplied.embeddings, direct.embeddings)
    np.testing.assert_array_equal(supplied.state_sequences, direct.state_sequences)
    np.testing.assert_array_equal(supplied.action_sequences, direct.action_sequences)
    np.testing.assert_array_equal(supplied.visual_progress, direct.visual_progress)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skip(reason="Long-running GPU encoder smoke test")
def test_local_clip_encoder_smoke_on_gpu():
    model = Path("/data/dwb/models/clip-vit-base-patch32")
    if not model.is_dir():
        pytest.skip("local CLIP model is not mounted")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    encoder = FrozenClipEncoder(
        {
            "model": str(model),
            "local_files_only": True,
            "device": "cuda",
            "batch_size": 1,
        }
    )

    features = encoder.encode(np.zeros((1, 224, 224, 3), dtype=np.uint8))

    assert features.shape == (1, encoder.output_dim)
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), [1.0], atol=1.0e-5)
