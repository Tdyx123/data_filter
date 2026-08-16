"""Cocore-owned Quality-style fragment and overlapping-half visual encoding."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from relcore.features.visual_encoder import VisualEncoder
from relcore.schemas import ClipRecord
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord

from cocore.index import CLIP_LENGTH, build_clip_records
from cocore.timing import TimingCallback, timed_step


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _robust_bounds(
    arrays: Iterable[np.ndarray],
    *,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    materialized = [np.asarray(value, dtype=np.float32) for value in arrays]
    if not materialized:
        raise ValueError("cannot fit robust bounds without arrays")
    merged = np.concatenate(materialized, axis=0)
    if merged.ndim == 1:
        merged = merged[:, None]
    if merged.ndim != 2 or not np.all(np.isfinite(merged)):
        raise ValueError("numeric features must be finite [time, dim] arrays")
    return (
        np.quantile(merged, low, axis=0).astype(np.float32),
        np.quantile(merged, high, axis=0).astype(np.float32),
    )


def _robust_scale(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    denominator = np.maximum(np.asarray(upper) - np.asarray(lower), epsilon)
    return np.clip((array - lower) / denominator, 0.0, 1.0).astype(np.float32)


@dataclass
class CocoreNumericNormalizers:
    """Quality-compatible robust per-dimension action and state bounds."""

    action_lower: np.ndarray
    action_upper: np.ndarray
    observation_bounds: dict[str, tuple[np.ndarray, np.ndarray]]
    vector_keys: tuple[str, ...]
    quantile_low: float = 0.01
    quantile_high: float = 0.99
    epsilon: float = 1.0e-8

    @classmethod
    def fit(
        cls,
        episodes: Iterable[tuple[np.ndarray, Mapping[str, np.ndarray]]],
        vector_keys: Sequence[str],
        *,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        epsilon: float = 1.0e-8,
    ) -> "CocoreNumericNormalizers":
        actions: list[np.ndarray] = []
        observations: dict[str, list[np.ndarray]] = {str(key): [] for key in vector_keys}
        for action_values, observation_values in episodes:
            actions.append(np.asarray(action_values, dtype=np.float32))
            for key in observations:
                if key not in observation_values:
                    raise ValueError(f"episode is missing vector observation {key!r}")
                observations[key].append(np.asarray(observation_values[key], dtype=np.float32))
        action_lower, action_upper = _robust_bounds(
            actions,
            low=quantile_low,
            high=quantile_high,
        )
        bounds = {
            key: _robust_bounds(values, low=quantile_low, high=quantile_high)
            for key, values in observations.items()
        }
        return cls(
            action_lower,
            action_upper,
            bounds,
            tuple(str(key) for key in vector_keys),
            float(quantile_low),
            float(quantile_high),
            float(epsilon),
        )

    def action(self, values: np.ndarray) -> np.ndarray:
        return _robust_scale(
            values,
            self.action_lower,
            self.action_upper,
            epsilon=self.epsilon,
        )

    def state(self, observations: Mapping[str, np.ndarray]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key in self.vector_keys:
            if key not in observations:
                raise ValueError(f"segment is missing vector observation {key!r}")
            lower, upper = self.observation_bounds[key]
            scaled = _robust_scale(
                observations[key],
                lower,
                upper,
                epsilon=self.epsilon,
            )
            parts.append(scaled[:, None] if scaled.ndim == 1 else scaled)
        if not parts:
            raise ValueError("cocore requires at least one vector state observation")
        return np.concatenate(parts, axis=1)


@dataclass
class CocorePCAProjector:
    """Quality-compatible standardized PCA with fixed-width zero padding."""

    output_dim: int
    seed: int = 42
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = int(self.output_dim)
        self.seed = int(self.seed)

    def fit(
        self,
        features: np.ndarray,
        *,
        max_samples: int | None = None,
    ) -> "CocorePCAProjector":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("features must have shape [samples, dimensions]")
        if not np.all(np.isfinite(values)):
            raise ValueError("PCA features contain NaN or infinity")
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1.0e-8] = 1.0
        standardized = (values - self.mean_) / self.scale_
        fit_data = standardized
        if max_samples and len(fit_data) > int(max_samples):
            rng = np.random.default_rng(self.seed)
            indices = np.sort(rng.choice(len(fit_data), size=int(max_samples), replace=False))
            fit_data = fit_data[indices]
        components = min(
            self.output_dim,
            fit_data.shape[1],
            max(1, len(fit_data) - 1),
        )
        try:
            from sklearn.decomposition import PCA

            model = PCA(
                n_components=components,
                svd_solver=("randomized" if components < min(fit_data.shape) else "full"),
                random_state=self.seed,
            )
            model.fit(fit_data)
            self.components_ = model.components_.astype(np.float32)
            self.explained_variance_ratio_ = model.explained_variance_ratio_.astype(np.float32)
        except ImportError:
            _, singular, right = np.linalg.svd(fit_data, full_matrices=False)
            self.components_ = right[:components].astype(np.float32)
            variance = singular**2
            total = float(variance.sum()) or 1.0
            self.explained_variance_ratio_ = (variance[:components] / total).astype(np.float32)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise RuntimeError("CocorePCAProjector must be fitted before transform")
        values = np.asarray(features, dtype=np.float32)
        standardized = (values - self.mean_) / self.scale_
        projected = standardized @ self.components_.T
        if projected.shape[1] < self.output_dim:
            projected = np.pad(
                projected,
                ((0, 0), (0, self.output_dim - projected.shape[1])),
            )
        return projected.astype(np.float32)

    def fit_transform(
        self,
        features: np.ndarray,
        *,
        max_samples: int | None = None,
    ) -> np.ndarray:
        return self.fit(features, max_samples=max_samples).transform(features)


def temporal_pool(sequence: np.ndarray) -> np.ndarray:
    """Concatenate mean, standard deviation, and maximum over time."""

    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"expected non-empty [time, dim] sequence, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("temporal sequence contains NaN or infinity")
    return np.concatenate([values.mean(axis=0), values.std(axis=0), values.max(axis=0)]).astype(
        np.float32
    )


def visual_fragment_feature(frame_features: np.ndarray) -> np.ndarray:
    """Return ``[sum(v_0..v_14), v_14-v_0]`` for one fragment."""

    values = np.asarray(frame_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != CLIP_LENGTH:
        raise ValueError(f"visual fragment features must have shape [15, dim], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("visual fragment features contain NaN or infinity")
    return np.concatenate([values.sum(axis=0), values[-1] - values[0]]).astype(np.float32)


def _l2_normalize_rows(features: np.ndarray, *, epsilon: float = 1.0e-8) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("features must have shape [samples, positive dimensions]")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must contain only finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, float(epsilon))).astype(np.float32)


def fuse_fragment_features(
    visual_embeddings: np.ndarray,
    state_pooled: np.ndarray,
    action_pooled: np.ndarray,
    progress: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate Quality fragment features and L2-normalize each row."""

    visual = np.asarray(visual_embeddings, dtype=np.float32)
    states = np.asarray(state_pooled, dtype=np.float32)
    actions = np.asarray(action_pooled, dtype=np.float32)
    progress_values = np.asarray(progress, dtype=np.float32)
    row_count = len(visual)
    if (
        visual.ndim != 2
        or states.ndim != 2
        or actions.ndim != 2
        or progress_values.ndim != 1
        or len(states) != row_count
        or len(actions) != row_count
        or len(progress_values) != row_count
    ):
        raise ValueError("fused fragment features must have aligned sample rows")
    fused_raw = np.concatenate([visual, states, actions, progress_values[:, None]], axis=1).astype(
        np.float32
    )
    return fused_raw, _l2_normalize_rows(fused_raw)


def visual_half_means(frame_features: np.ndarray) -> np.ndarray:
    """Return normalized means for overlapping frames ``[0..7]`` and ``[7..14]``."""

    values = np.asarray(frame_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != CLIP_LENGTH or values.shape[1] == 0:
        raise ValueError("visual clip features must have shape [15, positive dimensions]")
    if not np.all(np.isfinite(values)):
        raise ValueError("visual clip features must be finite")
    means = np.stack([values[:8].mean(axis=0), values[7:].mean(axis=0)])
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1.0e-8):
        raise ValueError("visual half means must have finite positive norms")
    return (means / norms).astype(np.float32)


@dataclass(frozen=True)
class FrameEmbeddingEntry:
    episode_id: int
    filename: str
    frames: int
    embedding_dim: int
    sha256: str


@dataclass
class CocoreEncodedClips:
    clips: list[ClipRecord]
    embeddings: np.ndarray
    visual_half_embeddings: np.ndarray
    state_sequences: np.ndarray
    action_sequences: np.ndarray
    visual_progress: np.ndarray
    numeric_normalizers: CocoreNumericNormalizers
    visual_projector: CocorePCAProjector
    frame_embeddings: list[FrameEmbeddingEntry]
    pca_fit_fragment_count: int


@dataclass(frozen=True)
class CocoreEncodedArtifact:
    clips: list[ClipRecord]
    embeddings: np.ndarray
    visual_half_embeddings: np.ndarray
    state_sequences: np.ndarray
    action_sequences: np.ndarray
    visual_progress: np.ndarray
    fingerprint: str


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
) -> None:
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


def encode_cocore_dataset(
    adapter: DatasetAdapter,
    visual_encoder: VisualEncoder,
    *,
    visual_dim: int = 128,
    pca_fit_max_samples: int | None = None,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    epsilon: float = 1.0e-8,
    seed: int = 42,
    frame_cache_dir: str | Path,
    num_workers: int = 0,
    max_episodes: int | None = None,
    progress_interval: int = 0,
    timing_callback: TimingCallback | None = None,
) -> CocoreEncodedClips:
    """Encode Quality-style fragments and cache each usable episode's frames."""

    records = list(adapter.episodes())
    if max_episodes is not None:
        records = records[:max_episodes]
    records_by_id = _records_by_id(records)
    clips = build_clip_records(records)
    if not clips:
        raise ValueError("dataset contains no complete clips")

    with timed_step("encode.numeric_normalization", timing_callback):
        numeric_episodes: list[tuple[np.ndarray, Mapping[str, np.ndarray]]] = []
        numeric_seen: set[int] = set()
        for episode in adapter.iter_episodes(
            num_workers=num_workers,
            max_episodes=max_episodes,
            load_images=False,
        ):
            _validate_episode_metadata(episode, records_by_id, numeric_seen, "numeric")
            numeric_episodes.append((episode.actions, episode.observations))
        if numeric_seen != set(records_by_id):
            raise ValueError("numeric pass did not yield every indexed episode exactly once")
        normalizers = CocoreNumericNormalizers.fit(
            numeric_episodes,
            adapter.vector_observation_keys,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            epsilon=epsilon,
        )

    with timed_step("encode.visual_cache", timing_callback):
        clips_by_episode: dict[int, list[tuple[int, ClipRecord]]] = {}
        for index, clip in enumerate(clips):
            clips_by_episode.setdefault(clip.episode_id, []).append((index, clip))
        candidate_windows = {
            record.episode_id: [
                (clip.start_step, clip.end_step)
                for _, clip in clips_by_episode.get(record.episode_id, ())
            ]
            for record in records
        }
        candidate_order = [
            (record.episode_id, start, end)
            for record in records
            for start, end in candidate_windows[record.episode_id]
        ]

        cache_root = Path(frame_cache_dir)
        if cache_root.exists() and any(cache_root.iterdir()):
            raise FileExistsError(f"frame cache directory must be empty: {cache_root}")
        cache_root.mkdir(parents=True, exist_ok=True)
        raw_visual: dict[tuple[int, int, int], np.ndarray] = {}
        half_visual_by_index: dict[int, np.ndarray] = {}
        state_by_index: dict[int, np.ndarray] = {}
        action_by_index: dict[int, np.ndarray] = {}
        position_by_index: dict[int, float] = {}
        progress_by_index: dict[int, float] = {}
        frame_entries: dict[int, FrameEmbeddingEntry] = {}
        image_key = adapter.image_observation_keys[0]
        image_seen: set[int] = set()
        feature_dim: int | None = None
        for completed, episode in enumerate(
            adapter.iter_episode_subset(
                records,
                num_workers=num_workers,
                load_images=True,
            ),
            start=1,
        ):
            _validate_episode_metadata(episode, records_by_id, image_seen, "image")
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
            feature_dim = frame_features.shape[1] if feature_dim is None else feature_dim
            if frame_features.shape[1] != feature_dim:
                raise ValueError(f"episode {episode.episode_id}: visual feature dimension changed")
            filename = f"ep{episode.episode_id:06d}.npy"
            frame_path = cache_root / filename
            np.save(frame_path, frame_features)
            frame_entries[episode.episode_id] = FrameEmbeddingEntry(
                episode_id=episode.episode_id,
                filename=filename,
                frames=episode.length,
                embedding_dim=feature_dim,
                sha256=_sha256(frame_path),
            )

            normalized_state = normalizers.state(episode.observations)
            normalized_action = normalizers.action(episode.actions)
            for start, end in candidate_windows[episode.episode_id]:
                raw_visual[(episode.episode_id, start, end)] = visual_fragment_feature(
                    frame_features[start : end + 1]
                )
            for clip_index, clip in clips_by_episode.get(episode.episode_id, ()):
                window = slice(clip.start_step, clip.end_step + 1)
                visual = frame_features[window]
                state = normalized_state[window]
                action = normalized_action[window]
                half_visual_by_index[clip_index] = visual_half_means(visual)
                state_by_index[clip_index] = state
                action_by_index[clip_index] = action
                position_by_index[clip_index] = float(clip.start_step) / float(episode.length)
                normalized_visual = visual / np.maximum(
                    np.linalg.norm(visual, axis=1, keepdims=True), 1.0e-8
                )
                progress_by_index[clip_index] = float(
                    np.linalg.norm(normalized_visual[-1] - normalized_visual[0])
                )
            if progress_interval > 0 and (
                completed % progress_interval == 0 or completed == len(records)
            ):
                print(
                    f"cocore_encode completed={completed} remaining={len(records) - completed}",
                    file=sys.stderr,
                    flush=True,
                )
        if image_seen != set(records_by_id):
            raise ValueError("image pass did not yield every indexed episode exactly once")
        if len(half_visual_by_index) != len(clips):
            raise ValueError("encoded clip count does not match clip index")
        if set(raw_visual) != set(candidate_order):
            raise ValueError("encoded visual features do not match candidate fragment windows")

    with timed_step("encode.pca_fusion", timing_callback):
        visual_raw = np.stack([raw_visual[key] for key in candidate_order])
        projector = CocorePCAProjector(output_dim=visual_dim, seed=seed)
        candidate_projected = projector.fit_transform(
            visual_raw,
            max_samples=pca_fit_max_samples,
        )
        candidate_index = {key: index for index, key in enumerate(candidate_order)}
        candidate_visual = candidate_projected[
            np.asarray(
                [
                    candidate_index[(clip.episode_id, clip.start_step, clip.end_step)]
                    for clip in clips
                ],
                dtype=np.int64,
            )
        ]
        state_sequences = np.stack([state_by_index[index] for index in range(len(clips))]).astype(
            dtype=np.float32,
        )
        action_sequences = np.stack(
            [action_by_index[index] for index in range(len(clips))]
        ).astype(dtype=np.float32)
        _, embeddings = fuse_fragment_features(
            candidate_visual,
            np.stack([temporal_pool(values) for values in state_sequences]),
            np.stack([temporal_pool(values) for values in action_sequences]),
            np.asarray(
                [position_by_index[index] for index in range(len(clips))],
                dtype=np.float32,
            ),
        )
        return CocoreEncodedClips(
            clips=clips,
            embeddings=embeddings,
            visual_half_embeddings=np.stack(
                [half_visual_by_index[index] for index in range(len(clips))]
            ).astype(np.float32),
            state_sequences=state_sequences,
            action_sequences=action_sequences,
            visual_progress=np.asarray(
                [progress_by_index[index] for index in range(len(clips))],
                dtype=np.float32,
            ),
            numeric_normalizers=normalizers,
            visual_projector=projector,
            frame_embeddings=[frame_entries[record.episode_id] for record in records],
            pca_fit_fragment_count=len(candidate_order),
        )
