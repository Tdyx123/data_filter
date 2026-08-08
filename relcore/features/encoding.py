"""SQCN-style two-pass episode encoding for relcore."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord

from relcore.data.index import build_clip_records
from relcore.schemas import ClipRecord

from .normalization import RobustNormalizer
from .projection import RelationProjector
from .relation_encoder import RelationEncoder
from .visual_encoder import VisualEncoder


@dataclass
class EncodedClips:
    clips: list[ClipRecord]
    embeddings: np.ndarray
    raw_relations: np.ndarray
    state_sequences: np.ndarray
    action_sequences: np.ndarray
    visual_progress: np.ndarray
    state_normalizer: RobustNormalizer
    action_normalizer: RobustNormalizer
    relation_encoder: RelationEncoder
    relation_projector: RelationProjector


def _records_by_id(records: list[EpisodeRecord]) -> dict[int, EpisodeRecord]:
    result = {record.episode_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("episode metadata contains duplicate episode ids")
    return result


def _validate_episode_metadata(
    episode: EpisodeData,
    expected: dict[int, EpisodeRecord],
    seen: set[int],
    pass_name: str,
) -> EpisodeRecord:
    if episode.episode_id in seen or episode.episode_id not in expected:
        raise ValueError(f"unexpected or duplicate {pass_name} episode {episode.episode_id}")
    seen.add(episode.episode_id)
    record = expected[episode.episode_id]
    if episode.length != record.length:
        raise ValueError(f"episode {episode.episode_id}: metadata length mismatch")
    if (episode.task_index, episode.task_name) != (record.task_index, record.task_name):
        raise ValueError(f"episode {episode.episode_id}: task metadata mismatch")
    if not np.array_equal(episode.frame_indices, np.arange(episode.length, dtype=np.int64)):
        raise ValueError(f"episode {episode.episode_id} frame indices must be contiguous from zero")
    return record


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
        raise ValueError("relcore requires at least one vector observation")
    return np.concatenate(parts, axis=1)


def fit_numeric_normalizers(
    adapter: DatasetAdapter,
    records: list[EpisodeRecord],
    *,
    epsilon: float = 1.0e-6,
    num_workers: int = 0,
    max_episodes: int | None = None,
) -> tuple[RobustNormalizer, RobustNormalizer]:
    """Run the image-free first pass and return action/state normalizers."""

    expected = _records_by_id(records)
    action_arrays: list[np.ndarray] = []
    state_arrays: list[np.ndarray] = []
    seen: set[int] = set()
    action_dim: int | None = None
    state_dim: int | None = None
    for episode in adapter.iter_episodes(
        num_workers=num_workers,
        max_episodes=max_episodes,
        load_images=False,
    ):
        _validate_episode_metadata(episode, expected, seen, "numeric")
        actions = np.asarray(episode.actions, dtype=np.float32)
        states = _state_values(adapter, episode.observations)
        if actions.ndim != 2 or not np.all(np.isfinite(actions)):
            raise ValueError(f"episode {episode.episode_id}: actions must be finite [time, dim]")
        action_dim = actions.shape[1] if action_dim is None else action_dim
        state_dim = states.shape[1] if state_dim is None else state_dim
        if actions.shape[1] != action_dim or states.shape[1] != state_dim:
            raise ValueError(f"episode {episode.episode_id}: numeric dimensions changed")
        action_arrays.append(actions)
        state_arrays.append(states)
    if seen != set(expected):
        raise ValueError("numeric pass did not yield every indexed episode exactly once")
    return (
        RobustNormalizer.fit(action_arrays, epsilon=epsilon),
        RobustNormalizer.fit(state_arrays, epsilon=epsilon),
    )


def encode_dataset(
    adapter: DatasetAdapter,
    visual_encoder: VisualEncoder,
    *,
    clip_length: int = 15,
    clip_stride: int = 15,
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
) -> EncodedClips:
    """Use fitted statistics or run pass one, then encode each usable episode once."""

    records = list(adapter.episodes())
    if max_episodes is not None:
        records = records[:max_episodes]
    clips = build_clip_records(records, length=clip_length, stride=clip_stride)
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
    state_by_index: dict[int, np.ndarray] = {}
    action_by_index: dict[int, np.ndarray] = {}
    progress_by_index: dict[int, float] = {}
    relation_encoder = RelationEncoder(projection_dim, lags, seed)
    image_key = adapter.image_observation_keys[0]
    image_episode_ids: set[int] = set()
    for completed, episode in enumerate(
        adapter.iter_episode_subset(
            usable_records,
            num_workers=num_workers,
            load_images=True,
        ),
        start=1,
    ):
        _validate_episode_metadata(episode, records_by_id, image_episode_ids, "image")
        if image_key not in episode.observations:
            raise ValueError(f"episode {episode.episode_id} is missing image {image_key!r}")
        frame_features = np.asarray(
            visual_encoder.encode(episode.observations[image_key]), dtype=np.float32
        )
        if (
            frame_features.ndim != 2
            or frame_features.shape[0] != episode.length
            or not np.all(np.isfinite(frame_features))
        ):
            raise ValueError(f"episode {episode.episode_id}: visual feature length mismatch")
        normalized_state = state_normalizer.transform(_state_values(adapter, episode.observations))
        normalized_action = action_normalizer.transform(episode.actions)
        for clip_index, clip in clips_by_episode.get(episode.episode_id, []):
            window = slice(clip.start_step, clip.end_step + 1)
            visual = frame_features[window]
            state = normalized_state[window]
            action = normalized_action[window]
            raw_by_index[clip_index] = relation_encoder.encode_raw(visual, state, action)
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
                "relcore_encode "
                f"completed={completed} remaining={len(usable_records) - completed}",
                file=sys.stderr,
                flush=True,
            )
    if image_episode_ids != set(records_by_id):
        raise ValueError("image pass did not yield every indexed episode exactly once")
    if len(raw_by_index) != len(clips):
        raise ValueError("encoded clip count does not match clip index")
    raw_relations = np.stack([raw_by_index[index] for index in range(len(clips))])
    projector = RelationProjector(output_dim=output_dim, seed=seed)
    embeddings = projector.fit_transform(raw_relations)
    return EncodedClips(
        clips=clips,
        embeddings=embeddings,
        raw_relations=raw_relations,
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
