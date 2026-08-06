"""SQCN temporal pooling, PCA projection, and row normalization."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


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
class NumericNormalizers:
    """Robust per-dimension action and state normalization statistics."""

    action_lower: np.ndarray
    action_upper: np.ndarray
    observation_bounds: dict[str, tuple[np.ndarray, np.ndarray]]
    vector_keys: tuple[str, ...]
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
    ) -> "NumericNormalizers":
        actions: list[np.ndarray] = []
        observations: dict[str, list[np.ndarray]] = {
            str(key): [] for key in vector_keys
        }
        for action_values, observation_values in episodes:
            actions.append(np.asarray(action_values, dtype=np.float32))
            for key in observations:
                if key not in observation_values:
                    raise ValueError(f"episode is missing vector observation {key!r}")
                observations[key].append(
                    np.asarray(observation_values[key], dtype=np.float32)
                )
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
            epsilon,
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
            raise ValueError("SQCN requires at least one vector state observation")
        return np.concatenate(parts, axis=1)


def _clip_feature_tensor(output: Any) -> Any:
    import torch

    if torch.is_tensor(output):
        return output
    pooled = getattr(output, "pooler_output", None)
    if torch.is_tensor(pooled):
        return pooled
    raise TypeError("CLIP get_image_features returned no tensor feature")


class ClipVisionEncoder:
    """Strict local CLIP ViT feature extractor with no fallback backend."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            configured_device = str(config.get("device", "auto"))
            self.device = (
                "cuda"
                if configured_device == "auto" and torch.cuda.is_available()
                else configured_device
            )
            if self.device == "auto":
                self.device = "cpu"
            model_path = str(config["model"])
            local_only = bool(config.get("local_files_only", True))
            self.processor = CLIPProcessor.from_pretrained(
                model_path,
                local_files_only=local_only,
            )
            self.model = (
                CLIPModel.from_pretrained(model_path, local_files_only=local_only)
                .eval()
                .to(self.device)
            )
        except Exception as error:
            raise RuntimeError(f"CLIP ViT could not be loaded: {error}") from error

    def encode(self, frames: np.ndarray) -> np.ndarray:
        values = np.asarray(frames)
        if values.ndim != 4 or values.shape[-1] != 3 or len(values) == 0:
            raise ValueError(f"frames must have shape [time, height, width, 3], got {values.shape}")
        import torch

        batch_size = int(self.config.get("image_batch_size", 64))
        outputs: list[np.ndarray] = []
        for start in range(0, len(values), batch_size):
            batch = [frame for frame in values[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = _clip_feature_tensor(self.model.get_image_features(**inputs))
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            outputs.append(features.cpu().numpy().astype(np.float32))
        result = np.concatenate(outputs, axis=0)
        if not np.all(np.isfinite(result)):
            raise ValueError("CLIP ViT produced NaN or infinity")
        return result


def temporal_pool(sequence: np.ndarray) -> np.ndarray:
    """Concatenate mean, standard deviation, and maximum over time."""

    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 1:
        sequence = sequence[:, None]
    if sequence.ndim != 2 or len(sequence) == 0:
        raise ValueError(f"expected non-empty [time, dim] sequence, got {sequence.shape}")
    if not np.all(np.isfinite(sequence)):
        raise ValueError("temporal sequence contains NaN or infinity")
    return np.concatenate(
        [sequence.mean(axis=0), sequence.std(axis=0), sequence.max(axis=0)]
    ).astype(np.float32)


def visual_fragment_feature(frame_features: np.ndarray) -> np.ndarray:
    """Return ``[sum(v_0..v_14), v_14-v_0]`` for one 15-frame fragment."""

    values = np.asarray(frame_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 15:
        raise ValueError(
            f"visual fragment features must have shape [15, dim], got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("visual fragment features contain NaN or infinity")
    return np.concatenate([values.sum(axis=0), values[-1] - values[0]]).astype(
        np.float32
    )


def l2_normalize_rows(
    features: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """L2-normalize finite feature rows while preserving zero rows."""

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
    """Concatenate fragment features in contract order and L2-normalize rows."""

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
    fused_raw = np.concatenate(
        [visual, states, actions, progress_values[:, None]],
        axis=1,
    ).astype(np.float32)
    return fused_raw, l2_normalize_rows(fused_raw)


class PCAProjector:
    """Standardize, project, and zero-pad feature rows."""

    def __init__(self, output_dim: int, seed: int = 42):
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(
        self,
        features: np.ndarray,
        *,
        max_samples: int | None = None,
    ) -> "PCAProjector":
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
            indices = np.sort(
                rng.choice(len(fit_data), size=int(max_samples), replace=False)
            )
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
                svd_solver=(
                    "randomized" if components < min(fit_data.shape) else "full"
                ),
                random_state=self.seed,
            )
            model.fit(fit_data)
            self.components_ = model.components_.astype(np.float32)
            self.explained_variance_ratio_ = model.explained_variance_ratio_.astype(
                np.float32
            )
        except ImportError:
            _, singular, right = np.linalg.svd(fit_data, full_matrices=False)
            self.components_ = right[:components].astype(np.float32)
            variance = singular**2
            total = float(variance.sum()) or 1.0
            self.explained_variance_ratio_ = (
                variance[:components] / total
            ).astype(np.float32)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise RuntimeError("PCAProjector must be fitted before transform")
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

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
