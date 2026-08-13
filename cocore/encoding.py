"""Cocore-owned one-pass relation and half-clip visual encoding."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from relcore.data.index import build_clip_records
from relcore.features.encoding import fit_numeric_normalizers
from relcore.features.normalization import RobustNormalizer
from relcore.features.projection import RelationProjector
from relcore.features.relation_encoder import RelationEncoder
from relcore.features.visual_encoder import VisualEncoder
from relcore.schemas import ClipRecord
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord


CLIP_LENGTH = 15
CLIP_STRIDE = 15
HALF_WINDOWS = ((0, 8), (7, 15))


@dataclass
class CocoreEncodedClips:
    clips: list[ClipRecord]
    embeddings: np.ndarray
    raw_relations: np.ndarray
    visual_half_embeddings: np.ndarray
    state_sequences: np.ndarray
    action_sequences: np.ndarray
    visual_progress: np.ndarray
    state_normalizer: RobustNormalizer
    action_normalizer: RobustNormalizer
    relation_encoder: RelationEncoder
    relation_projector: RelationProjector


@dataclass(frozen=True)
class CocoreEncodedArtifact:
    clips: list[ClipRecord]
    embeddings: np.ndarray
    visual_half_embeddings: np.ndarray
    state_sequences: np.ndarray
    action_sequences: np.ndarray
    visual_progress: np.ndarray
    fingerprint: str


def visual_half_means(frame_features: np.ndarray) -> np.ndarray:
    """Average frames 0..7 and 7..14 of one fixed 15-frame clip."""

    values = np.asarray(frame_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != CLIP_LENGTH or values.shape[1] == 0:
        raise ValueError(
            "visual clip features must have shape [15, positive dimensions]"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("visual clip features must be finite")
    return np.stack(
        [values[start:end].mean(axis=0) for start, end in HALF_WINDOWS]
    ).astype(np.float32)


def _records_by_id(records: list[EpisodeRecord]) -> dict[int, EpisodeRecord]:
    result = {record.episode_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("episode metadata contains duplicate episode ids")
    return result


def _validate_episode_metadata(
    episode: EpisodeData,
    expected: dict[int, EpisodeRecord],
    seen: set[int],
) -> None:
    if episode.episode_id in seen or episode.episode_id not in expected:
        raise ValueError(f"unexpected or duplicate image episode {episode.episode_id}")
    seen.add(episode.episode_id)
    record = expected[episode.episode_id]
    if episode.length != record.length:
        raise ValueError(f"episode {episode.episode_id}: metadata length mismatch")
    if (episode.task_index, episode.task_name) != (record.task_index, record.task_name):
        raise ValueError(f"episode {episode.episode_id}: task metadata mismatch")
    if not np.array_equal(episode.frame_indices, np.arange(episode.length, dtype=np.int64)):
        raise ValueError(f"episode {episode.episode_id} frame indices must be contiguous from zero")


def _state_values(adapter: DatasetAdapter, observations: dict[str, np.ndarray]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in adapter.vector_observation_keys:
        if key not in observations:
            raise ValueError(f"episode is missing vector observation {key!r}")
        values = np.asarray(observations[key], dtype=np.float32)
        if values.ndim not in {1, 2} or not np.all(np.isfinite(values)):
            raise ValueError(f"vector observation {key!r} must be finite [time, dim]")
        parts.append(values[:, None] if values.ndim == 1 else values)
    if not parts:
        raise ValueError("cocore requires at least one vector observation")
    return np.concatenate(parts, axis=1)


def encode_cocore_dataset(
    adapter: DatasetAdapter,
    visual_encoder: VisualEncoder,
    *,
    projection_dim: int = 32,
    output_dim: int = 256,
    lags: tuple[int, ...] = (0, 1, 2, 4),
    seed: int = 42,
    epsilon: float = 1.0e-6,
    num_workers: int = 0,
    max_episodes: int | None = None,
    progress_interval: int = 0,
    action_normalizer: RobustNormalizer | None = None,
    state_normalizer: RobustNormalizer | None = None,
) -> CocoreEncodedClips:
    """Encode relation and half-clip visual features in one image pass."""

    records = list(adapter.episodes())
    if max_episodes is not None:
        records = records[:max_episodes]
    clips = build_clip_records(records, length=CLIP_LENGTH, stride=CLIP_STRIDE)
    if not clips:
        raise ValueError("dataset contains no complete clips")

    if (action_normalizer is None) != (state_normalizer is None):
        raise ValueError("action and state normalizers must be provided together")
    if action_normalizer is None or state_normalizer is None:
        action_normalizer, state_normalizer = fit_numeric_normalizers(
            adapter,
            records,
            epsilon=epsilon,
            num_workers=num_workers,
            max_episodes=max_episodes,
        )

    clips_by_episode: dict[int, list[tuple[int, ClipRecord]]] = {}
    for index, clip in enumerate(clips):
        clips_by_episode.setdefault(clip.episode_id, []).append((index, clip))
    usable_records = [record for record in records if record.episode_id in clips_by_episode]
    records_by_id = _records_by_id(usable_records)
    raw_by_index: dict[int, np.ndarray] = {}
    half_visual_by_index: dict[int, np.ndarray] = {}
    state_by_index: dict[int, np.ndarray] = {}
    action_by_index: dict[int, np.ndarray] = {}
    progress_by_index: dict[int, float] = {}
    relation_encoder = RelationEncoder(projection_dim, lags, seed)
    image_key = adapter.image_observation_keys[0]
    seen: set[int] = set()
    for completed, episode in enumerate(
        adapter.iter_episode_subset(
            usable_records,
            num_workers=num_workers,
            load_images=True,
        ),
        start=1,
    ):
        _validate_episode_metadata(episode, records_by_id, seen)
        if image_key not in episode.observations:
            raise ValueError(f"episode {episode.episode_id} is missing image {image_key!r}")
        frame_features = np.asarray(
            visual_encoder.encode(episode.observations[image_key]), dtype=np.float32
        )
        if (
            frame_features.ndim != 2
            or frame_features.shape[0] != episode.length
            or frame_features.shape[1] == 0
            or not np.all(np.isfinite(frame_features))
        ):
            raise ValueError(f"episode {episode.episode_id}: visual feature length mismatch")
        normalized_state = state_normalizer.transform(_state_values(adapter, episode.observations))
        normalized_action = action_normalizer.transform(episode.actions)
        for clip_index, clip in clips_by_episode[episode.episode_id]:
            window = slice(clip.start_step, clip.end_step + 1)
            visual = frame_features[window]
            state = normalized_state[window]
            action = normalized_action[window]
            raw_by_index[clip_index] = relation_encoder.encode_raw(visual, state, action)
            half_visual_by_index[clip_index] = visual_half_means(visual)
            state_by_index[clip_index] = state
            action_by_index[clip_index] = action
            normalized_visual = visual / np.maximum(
                np.linalg.norm(visual, axis=1, keepdims=True), 1.0e-8
            )
            progress_by_index[clip_index] = float(
                np.linalg.norm(normalized_visual[-1] - normalized_visual[0])
            )
        if progress_interval > 0 and (
            completed % progress_interval == 0 or completed == len(usable_records)
        ):
            print(
                "cocore_encode "
                f"completed={completed} remaining={len(usable_records) - completed}",
                file=sys.stderr,
                flush=True,
            )
    if seen != set(records_by_id):
        raise ValueError("image pass did not yield every indexed episode exactly once")
    if len(raw_by_index) != len(clips) or len(half_visual_by_index) != len(clips):
        raise ValueError("encoded clip count does not match clip index")

    raw_relations = np.stack([raw_by_index[index] for index in range(len(clips))])
    projector = RelationProjector(output_dim=output_dim, seed=seed)
    embeddings = projector.fit_transform(raw_relations)
    return CocoreEncodedClips(
        clips=clips,
        embeddings=embeddings,
        raw_relations=raw_relations,
        visual_half_embeddings=np.stack(
            [half_visual_by_index[index] for index in range(len(clips))]
        ).astype(np.float32),
        state_sequences=np.stack([state_by_index[index] for index in range(len(clips))]),
        action_sequences=np.stack([action_by_index[index] for index in range(len(clips))]),
        visual_progress=np.asarray(
            [progress_by_index[index] for index in range(len(clips))], dtype=np.float32
        ),
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        relation_encoder=relation_encoder,
        relation_projector=projector,
    )
