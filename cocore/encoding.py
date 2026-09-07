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
from cocore.local_backtracking import backtracking_arrays, resolve_backtracking_config
from cocore.local_path_efficiency import compute_local_path_efficiency, resolve_path_config
from cocore.eef_jerk import jerk_arrays
from cocore.action_jump import build_jump_arrays, resolve_jump_config
from cocore.action_execution_deviation import (
    execution_arrays, execution_inputs, resolve_execution_config,
)
from cocore.high_frequency_jitter import hf_arrays, resolve_hf_config
from cocore.dwell import compute_dwell_ratio, resolve_dwell_config
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord

from cocore.action_variation import (
    compute_step_action_variation,
    normalize_action_variation,
    top_k_mean,
)
from cocore.index import CLIP_LENGTH, build_clip_records
from cocore.temporal import resolve_temporal_geometry
from cocore.timing import TimingCallback, timed_step
from cocore.visual_action_consistency import (
    compute_step_visual_action_consistency,
    normalize_visual_action_consistency,
)


DEFAULT_VISUAL_HALF_WINDOWS = resolve_temporal_geometry("libero").visual_half_windows


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


def visual_fragment_feature(
    frame_features: np.ndarray,
    *,
    clip_length: int = CLIP_LENGTH,
) -> np.ndarray:
    """Return the summed frames and endpoint delta for one fragment."""

    values = np.asarray(frame_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != clip_length:
        raise ValueError(
            f"visual fragment features must have shape [{clip_length}, dim], got {values.shape}"
        )
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
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate visual, state, and action features, then L2-normalize."""

    visual = np.asarray(visual_embeddings, dtype=np.float32)
    states = np.asarray(state_pooled, dtype=np.float32)
    actions = np.asarray(action_pooled, dtype=np.float32)
    row_count = len(visual)
    if (
        visual.ndim != 2
        or states.ndim != 2
        or actions.ndim != 2
        or len(states) != row_count
        or len(actions) != row_count
    ):
        raise ValueError("fused fragment features must have aligned sample rows")
    fused_raw = np.concatenate([visual, states, actions], axis=1).astype(np.float32)
    return fused_raw, _l2_normalize_rows(fused_raw)


def visual_half_means(
    frame_features: np.ndarray,
    *,
    half_windows: tuple[tuple[int, int], tuple[int, int]] = DEFAULT_VISUAL_HALF_WINDOWS,
) -> np.ndarray:
    """Return normalized means for two explicit overlapping half windows."""

    values = np.asarray(frame_features, dtype=np.float32)
    windows = tuple((int(start), int(end)) for start, end in half_windows)
    if (
        len(windows) != 2
        or any(start < 0 or end <= start for start, end in windows)
        or values.ndim != 2
        or values.shape[0] != max(end for _, end in windows)
        or values.shape[1] == 0
    ):
        raise ValueError("visual clip features must match two valid half windows")
    if not np.all(np.isfinite(values)):
        raise ValueError("visual clip features must be finite")
    means = np.stack([values[start:end].mean(axis=0) for start, end in windows])
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
    action_variation_raw: np.ndarray
    action_variation: np.ndarray
    visual_action_consistency_raw: np.ndarray
    visual_action_consistency: np.ndarray
    visual_progress: np.ndarray
    numeric_normalizers: CocoreNumericNormalizers
    visual_projector: CocorePCAProjector
    frame_embeddings: list[FrameEmbeddingEntry]
    pca_fit_fragment_count: int
    dwell_state_sequences: np.ndarray | None = None
    dwell_timestamps: np.ndarray | None = None
    dwell_ratio: np.ndarray | None = None
    non_dwell: np.ndarray | None = None
    eef_jerk_positions: np.ndarray | None = None
    eef_jerk_timestamps: np.ndarray | None = None
    eef_jerk_raw: np.ndarray | None = None
    eef_jerk: np.ndarray | None = None
    eef_jerk_valid: np.ndarray | None = None
    eef_jerk_reason: np.ndarray | None = None
    high_frequency_positions: np.ndarray | None = None
    high_frequency_timestamps: np.ndarray | None = None
    high_frequency_ratio: np.ndarray | None = None
    high_frequency_rms: np.ndarray | None = None
    total_fluctuation_rms: np.ndarray | None = None
    low_high_frequency_jitter: np.ndarray | None = None
    high_frequency_resolution_hz: np.ndarray | None = None
    high_frequency_valid: np.ndarray | None = None
    high_frequency_reason: np.ndarray | None = None
    local_backtracking_positions: np.ndarray | None = None
    local_backtracking_timestamps: np.ndarray | None = None
    local_backtracking_rate: np.ndarray | None = None
    low_local_backtracking: np.ndarray | None = None
    local_backtracking_valid_count: np.ndarray | None = None
    local_backtracking_count: np.ndarray | None = None
    local_backtracking_valid: np.ndarray | None = None
    local_backtracking_reason: np.ndarray | None = None
    path_position_sequences: np.ndarray | None = None
    local_path_efficiency: np.ndarray | None = None
    action_jump_actions: np.ndarray | None = None
    action_jump_episode_ids: np.ndarray | None = None
    action_jump_offsets: np.ndarray | None = None
    action_jump_dimensions: np.ndarray | None = None
    action_jump_scale: np.ndarray | None = None
    action_jump_threshold: np.ndarray | None = None
    action_jump_pair_count: np.ndarray | None = None
    action_jump_rate: np.ndarray | None = None
    action_jump: np.ndarray | None = None
    action_execution_deviation_positions: np.ndarray | None = None
    action_execution_deviation_actions: np.ndarray | None = None
    action_execution_deviation_timestamps: np.ndarray | None = None
    action_execution_deviation_raw: np.ndarray | None = None
    low_action_execution_deviation: np.ndarray | None = None
    action_execution_deviation_valid: np.ndarray | None = None
    action_execution_deviation_reason: np.ndarray | None = None


@dataclass(frozen=True)
class CocoreEncodedArtifact:
    clips: list[ClipRecord]
    embeddings: np.ndarray
    visual_half_embeddings: np.ndarray
    state_sequences: np.ndarray
    action_sequences: np.ndarray
    action_variation_raw: np.ndarray
    action_variation: np.ndarray
    visual_action_consistency_raw: np.ndarray
    visual_action_consistency: np.ndarray
    visual_progress: np.ndarray
    fingerprint: str
    dwell_state_sequences: np.ndarray | None = None
    dwell_timestamps: np.ndarray | None = None
    dwell_ratio: np.ndarray | None = None
    non_dwell: np.ndarray | None = None
    eef_jerk_positions: np.ndarray | None = None
    eef_jerk_timestamps: np.ndarray | None = None
    eef_jerk_raw: np.ndarray | None = None
    eef_jerk: np.ndarray | None = None
    eef_jerk_valid: np.ndarray | None = None
    eef_jerk_reason: np.ndarray | None = None
    high_frequency_positions: np.ndarray | None = None
    high_frequency_timestamps: np.ndarray | None = None
    high_frequency_ratio: np.ndarray | None = None
    high_frequency_rms: np.ndarray | None = None
    total_fluctuation_rms: np.ndarray | None = None
    low_high_frequency_jitter: np.ndarray | None = None
    high_frequency_resolution_hz: np.ndarray | None = None
    high_frequency_valid: np.ndarray | None = None
    high_frequency_reason: np.ndarray | None = None
    local_backtracking_positions: np.ndarray | None = None
    local_backtracking_timestamps: np.ndarray | None = None
    local_backtracking_rate: np.ndarray | None = None
    low_local_backtracking: np.ndarray | None = None
    local_backtracking_valid_count: np.ndarray | None = None
    local_backtracking_count: np.ndarray | None = None
    local_backtracking_valid: np.ndarray | None = None
    local_backtracking_reason: np.ndarray | None = None
    path_position_sequences: np.ndarray | None = None
    local_path_efficiency: np.ndarray | None = None
    action_jump_actions: np.ndarray | None = None
    action_jump_episode_ids: np.ndarray | None = None
    action_jump_offsets: np.ndarray | None = None
    action_jump_dimensions: np.ndarray | None = None
    action_jump_scale: np.ndarray | None = None
    action_jump_threshold: np.ndarray | None = None
    action_jump_pair_count: np.ndarray | None = None
    action_jump_rate: np.ndarray | None = None
    action_jump: np.ndarray | None = None
    action_execution_deviation_positions: np.ndarray | None = None
    action_execution_deviation_actions: np.ndarray | None = None
    action_execution_deviation_timestamps: np.ndarray | None = None
    action_execution_deviation_raw: np.ndarray | None = None
    low_action_execution_deviation: np.ndarray | None = None
    action_execution_deviation_valid: np.ndarray | None = None
    action_execution_deviation_reason: np.ndarray | None = None


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


def _position_clip_inputs(
    episode: EpisodeData, clip: ClipRecord, *, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate raw inputs before normalized features can obscure clip context."""
    window = slice(clip.start_step, clip.end_step + 1)
    try:
        positions = np.asarray(
            episode.observations["observation.state"][window, :3], dtype=np.float64
        )
        timestamps = np.asarray(episode.timestamps[window], dtype=np.float64)
        if (
            positions.shape != (clip.length, 3)
            or timestamps.shape != (clip.length,)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(timestamps))
        ):
            raise ValueError("expected finite positions and matching timestamps")
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            f"{metric} clip {clip.sample_id}: missing or invalid positions/timestamps"
        ) from error
    return positions, timestamps


def encode_cocore_dataset(
    adapter: DatasetAdapter,
    visual_encoder: VisualEncoder,
    *,
    profile: str = "libero",
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
    dwell: Mapping[str, object] | None = None,
    eef_jerk: bool = False,
    local_path_efficiency: Mapping[str, object] | None = None,
    high_frequency_jitter: Mapping[str, object] | None = None,
    local_backtracking: Mapping[str, object] | None = None,
    action_jump: Mapping[str, object] | None = None,
    action_execution_deviation: Mapping[str, object] | None = None,
    gripper_action_index: int = -1,
) -> CocoreEncodedClips:
    """Encode Quality-style fragments and cache each usable episode's frames."""

    dwell = resolve_dwell_config(dwell)
    path_settings = resolve_path_config(local_path_efficiency)
    hf_settings = resolve_hf_config(high_frequency_jitter)
    backtracking_settings = resolve_backtracking_config(local_backtracking)
    jump_settings = resolve_jump_config(action_jump)
    execution_settings = resolve_execution_config(action_execution_deviation)
    geometry = resolve_temporal_geometry(profile)
    records = list(adapter.episodes())
    if max_episodes is not None:
        records = records[:max_episodes]
    records_by_id = _records_by_id(records)
    clips = build_clip_records(records, clip_length=geometry.clip_length)
    if not clips:
        raise ValueError("dataset contains no complete clips")

    clips_by_episode: dict[int, list[tuple[int, ClipRecord]]] = {}
    for index, clip in enumerate(clips):
        clips_by_episode.setdefault(clip.episode_id, []).append((index, clip))

    with timed_step("encode.numeric_normalization", timing_callback):
        numeric_episodes: list[tuple[np.ndarray, Mapping[str, np.ndarray]]] = []
        numeric_seen: set[int] = set()
        jump_episodes = {}
        execution_by_index = {}
        for episode in adapter.iter_episodes(
            num_workers=num_workers,
            max_episodes=max_episodes,
            load_images=False,
        ):
            _validate_episode_metadata(episode, records_by_id, numeric_seen, "numeric")
            if execution_settings is not None:
                for clip_index, clip in clips_by_episode.get(episode.episode_id, ()):
                    try:
                        p, t = _position_clip_inputs(
                            episode, clip, metric="action_execution_deviation"
                        )
                        a = episode.actions[clip.start_step : clip.end_step + 1, :3]
                        execution_by_index[clip_index] = execution_inputs(p, a, t)
                    except (AttributeError, IndexError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"action_execution_deviation clip {clip.sample_id}: {error}"
                        ) from error
            if hf_settings is not None:
                for _, clip in clips_by_episode.get(episode.episode_id, ()):
                    _position_clip_inputs(episode, clip, metric="high_frequency_jitter")
            if backtracking_settings is not None:
                for _, clip in clips_by_episode.get(episode.episode_id, ()):
                    _position_clip_inputs(episode, clip, metric="local_backtracking")
            numeric_episodes.append((episode.actions, episode.observations))
            if jump_settings is not None:
                jump_episodes[episode.episode_id] = episode.actions
        if numeric_seen != set(records_by_id):
            raise ValueError("numeric pass did not yield every indexed episode exactly once")
        normalizers = CocoreNumericNormalizers.fit(
            numeric_episodes,
            adapter.vector_observation_keys,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            epsilon=epsilon,
        )

    execution_values = (
        execution_arrays(
            *[
                np.stack([execution_by_index[i][axis] for i in range(len(clips))])
                for axis in range(3)
            ],
            config=execution_settings,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            epsilon=epsilon,
        )
        if execution_settings is not None
        else {}
    )
    jump_values = (
        build_jump_arrays(
            [jump_episodes[record.episode_id] for record in records],
            [record.episode_id for record in records],
            clips,
            config=jump_settings,
            gripper_action_index=gripper_action_index,
        )
        if jump_settings is not None
        else {}
    )

    with timed_step("encode.visual_cache", timing_callback):
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
        backtracking_positions: dict[int, np.ndarray] = {}
        backtracking_times: dict[int, np.ndarray] = {}
        hf_positions: dict[int, np.ndarray] = {}
        hf_times: dict[int, np.ndarray] = {}
        path_positions: dict[int, np.ndarray] = {}
        path_scores: dict[int, float] = {}
        jerk_positions: dict[int, np.ndarray] = {}
        jerk_times: dict[int, np.ndarray] = {}
        dwell_states: dict[int, np.ndarray] = {}
        dwell_times: dict[int, np.ndarray] = {}
        dwell_scores: dict[int, float] = {}
        state_by_index: dict[int, np.ndarray] = {}
        action_by_index: dict[int, np.ndarray] = {}
        action_variation_by_index: dict[int, float] = {}
        visual_action_consistency_by_index: dict[int, float] = {}
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
            episode_action_variation = compute_step_action_variation(normalized_action)
            episode_visual_action_consistency = None
            if candidate_windows[episode.episode_id]:
                episode_visual_action_consistency = compute_step_visual_action_consistency(
                    frame_features,
                    normalized_action,
                    epsilon=epsilon,
                )
            for start, end in candidate_windows[episode.episode_id]:
                raw_visual[(episode.episode_id, start, end)] = visual_fragment_feature(
                    frame_features[start : end + 1],
                    clip_length=geometry.clip_length,
                )
            for clip_index, clip in clips_by_episode.get(episode.episode_id, ()):
                if episode_visual_action_consistency is None:
                    raise RuntimeError("VAC scores are missing for a candidate episode")
                window = slice(clip.start_step, clip.end_step + 1)
                visual = frame_features[window]
                if hf_settings is not None:
                    hf_positions[clip_index], hf_times[clip_index] = _position_clip_inputs(
                        episode, clip, metric="high_frequency_jitter"
                    )
                if backtracking_settings is not None:
                    backtracking_positions[clip_index], backtracking_times[clip_index] = (
                        _position_clip_inputs(episode, clip, metric="local_backtracking")
                    )
                if path_settings is not None:
                    positions = np.asarray(
                        episode.observations["observation.state"][window, :3], dtype=np.float64
                    )
                    path_scores[clip_index] = compute_local_path_efficiency(
                        positions, **path_settings
                    )
                    path_positions[clip_index] = positions
                if eef_jerk:
                    jerk_positions[clip_index] = np.asarray(
                        episode.observations["observation.state"][window, :3], dtype=np.float64
                    )
                    jerk_times[clip_index] = np.asarray(
                        episode.timestamps[window], dtype=np.float64
                    )
                if dwell is not None:
                    raw_state = np.asarray(
                        episode.observations["observation.state"][window], dtype=np.float64
                    )
                    timestamps = np.asarray(episode.timestamps[window], dtype=np.float64)
                    dwell_scores[clip_index] = compute_dwell_ratio(
                        raw_state, timestamps, profile=profile, config=dwell
                    )
                    dwell_states[clip_index] = raw_state
                    dwell_times[clip_index] = timestamps
                state = normalized_state[window]
                action = normalized_action[window]
                half_visual_by_index[clip_index] = visual_half_means(
                    visual,
                    half_windows=geometry.visual_half_windows,
                )
                state_by_index[clip_index] = state
                action_by_index[clip_index] = action
                action_variation_by_index[clip_index] = top_k_mean(episode_action_variation[window])
                visual_action_consistency_by_index[clip_index] = top_k_mean(
                    episode_visual_action_consistency[window]
                )
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
        action_sequences = np.stack([action_by_index[index] for index in range(len(clips))]).astype(
            dtype=np.float32
        )
        action_variation_raw = np.asarray(
            [action_variation_by_index[index] for index in range(len(clips))],
            dtype=np.float32,
        )
        action_variation = normalize_action_variation(
            action_variation_raw,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            epsilon=epsilon,
        )
        visual_action_consistency_raw = np.asarray(
            [visual_action_consistency_by_index[index] for index in range(len(clips))],
            dtype=np.float32,
        )
        visual_action_consistency = normalize_visual_action_consistency(
            visual_action_consistency_raw,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            epsilon=epsilon,
        )
        _, embeddings = fuse_fragment_features(
            candidate_visual,
            np.stack([temporal_pool(values) for values in state_sequences]),
            np.stack([temporal_pool(values) for values in action_sequences]),
        )
        dwell_values = (
            np.asarray([dwell_scores[i] for i in range(len(clips))], dtype=np.float64)
            if dwell is not None
            else None
        )
        return CocoreEncodedClips(
            **jump_values,
            **execution_values,
            clips=clips,
            embeddings=embeddings,
            visual_half_embeddings=np.stack(
                [half_visual_by_index[index] for index in range(len(clips))]
            ).astype(np.float32),
            state_sequences=state_sequences,
            action_sequences=action_sequences,
            action_variation_raw=action_variation_raw,
            action_variation=action_variation,
            visual_action_consistency_raw=visual_action_consistency_raw,
            visual_action_consistency=visual_action_consistency,
            visual_progress=np.asarray(
                [progress_by_index[index] for index in range(len(clips))],
                dtype=np.float32,
            ),
            numeric_normalizers=normalizers,
            visual_projector=projector,
            frame_embeddings=[frame_entries[record.episode_id] for record in records],
            pca_fit_fragment_count=len(candidate_order),
            dwell_state_sequences=np.stack([dwell_states[i] for i in range(len(clips))])
            if dwell is not None
            else None,
            dwell_timestamps=np.stack([dwell_times[i] for i in range(len(clips))])
            if dwell is not None
            else None,
            **(
                hf_arrays(
                    np.stack([hf_positions[i] for i in range(len(clips))]),
                    np.stack([hf_times[i] for i in range(len(clips))]),
                    config=hf_settings,
                    clips=clips,
                )
                if hf_settings is not None
                else {}
            ),
            **(
                backtracking_arrays(
                    np.stack([backtracking_positions[i] for i in range(len(clips))]),
                    np.stack([backtracking_times[i] for i in range(len(clips))]),
                    config=backtracking_settings,
                    clips=clips,
                )
                if backtracking_settings is not None
                else {}
            ),
            path_position_sequences=np.stack([path_positions[i] for i in range(len(clips))])
            if path_settings is not None
            else None,
            local_path_efficiency=np.asarray(
                [path_scores[i] for i in range(len(clips))], dtype=np.float64
            )
            if path_settings is not None
            else None,
            dwell_ratio=dwell_values,
            non_dwell=1.0 - dwell_values if dwell_values is not None else None,
            **(
                jerk_arrays(
                    np.stack([jerk_positions[i] for i in range(len(clips))]),
                    np.stack([jerk_times[i] for i in range(len(clips))]),
                    quantile_low=quantile_low,
                    quantile_high=quantile_high,
                    epsilon=epsilon,
                )
                if eef_jerk
                else {}
            ),
        )
