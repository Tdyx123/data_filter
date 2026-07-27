"""Trajectory encoders used by TDUS.

The encoder is model-free with respect to the robot policy: CLIP is used only as
a frozen image feature extractor.  When CLIP cannot be loaded, resized pixels
flow through the same final PCA projection, providing an offline fallback.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .dataset import TrajectorySegment


def robust_bounds(
    arrays: Iterable[np.ndarray],
    *,
    low: float = 0.01,
    high: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension robust bounds from time-major arrays."""

    materialized = [np.asarray(value, dtype=np.float32) for value in arrays]
    if not materialized:
        raise ValueError("cannot fit robust bounds without arrays")
    merged = np.concatenate(materialized, axis=0)
    if merged.ndim == 1:
        merged = merged[:, None]
    return (
        np.quantile(merged, low, axis=0).astype(np.float32),
        np.quantile(merged, high, axis=0).astype(np.float32),
    )


def robust_scale(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Clip and scale values to [0, 1] using per-dimension quantiles."""

    values = np.asarray(values, dtype=np.float32)
    denominator = np.maximum(np.asarray(upper) - np.asarray(lower), epsilon)
    return np.clip((values - lower) / denominator, 0.0, 1.0).astype(np.float32)


def temporal_pool(sequence: np.ndarray) -> np.ndarray:
    """Concatenate mean, standard deviation, and maximum over time."""

    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 1:
        sequence = sequence[:, None]
    if sequence.ndim != 2 or len(sequence) == 0:
        raise ValueError(f"expected non-empty [time, dim] sequence, got {sequence.shape}")
    return np.concatenate(
        [sequence.mean(axis=0), sequence.std(axis=0), sequence.max(axis=0)]
    ).astype(np.float32)


def uniformly_sample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= 0:
        return np.empty((0,), dtype=np.int64)
    count = min(length, max(1, maximum))
    return np.unique(np.linspace(0, length - 1, num=count).round().astype(np.int64))


@dataclass
class NumericNormalizers:
    """Robust normalization statistics for actions and vector observations."""

    action_lower: np.ndarray
    action_upper: np.ndarray
    observation_bounds: dict[str, tuple[np.ndarray, np.ndarray]]
    epsilon: float = 1.0e-8

    @classmethod
    def fit(
        cls,
        segments: Iterable[TrajectorySegment],
        vector_keys: Sequence[str],
        *,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        epsilon: float = 1.0e-8,
    ) -> "NumericNormalizers":
        actions: list[np.ndarray] = []
        observations: dict[str, list[np.ndarray]] = {key: [] for key in vector_keys}
        for segment in segments:
            actions.append(segment.actions)
            for key in vector_keys:
                if key in segment.observations:
                    observations[key].append(segment.observations[key])
        action_lower, action_upper = robust_bounds(
            actions, low=quantile_low, high=quantile_high
        )
        bounds = {
            key: robust_bounds(values, low=quantile_low, high=quantile_high)
            for key, values in observations.items()
            if values
        }
        return cls(action_lower, action_upper, bounds, epsilon)

    def action(self, values: np.ndarray) -> np.ndarray:
        return robust_scale(
            values,
            self.action_lower,
            self.action_upper,
            epsilon=self.epsilon,
        )

    def observation(self, key: str, values: np.ndarray) -> np.ndarray:
        lower, upper = self.observation_bounds[key]
        scaled = robust_scale(values, lower, upper, epsilon=self.epsilon)
        return scaled[:, None] if scaled.ndim == 1 else scaled

    def state_sequence(self, segment: TrajectorySegment) -> np.ndarray | None:
        """Concatenate every available normalized vector observation."""

        parts = [
            self.observation(key, segment.observations[key])
            for key in self.observation_bounds
            if key in segment.observations
        ]
        return np.concatenate(parts, axis=1) if parts else None


class VisionFeatureExtractor:
    """Frozen CLIP image encoder with an offline resized-pixel fallback."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.backend = "pixels"
        self.device = "cpu"
        self.model: Any = None
        self.processor: Any = None
        self.pixel_projector: PCAProjector | None = None
        requested = str(config.get("vision_backend", "auto")).lower()
        if requested not in {"auto", "clip", "pixels"}:
            raise ValueError("vision_backend must be auto, clip, or pixels")
        if requested in {"auto", "clip"}:
            try:
                self._load_clip()
            except Exception as error:
                if requested == "clip":
                    raise
                warnings.warn(
                    f"CLIP is unavailable ({error}); using resized-pixel PCA fallback",
                    RuntimeWarning,
                )

    def _load_clip(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        configured_device = str(self.config.get("device", "auto"))
        self.device = (
            "cuda"
            if configured_device == "auto" and torch.cuda.is_available()
            else configured_device
        )
        if self.device == "auto":
            self.device = "cpu"
        model_name = str(
            self.config.get("clip_model", "openai/clip-vit-base-patch32")
        )
        local_only = bool(self.config.get("clip_local_files_only", False))
        self.processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=local_only
        )
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=local_only
        ).eval().to(self.device)
        self.backend = "clip"

    @staticmethod
    def _resize_pixels(frames: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image

        flattened: list[np.ndarray] = []
        for frame in frames:
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
                (size, size), Image.Resampling.BILINEAR
            )
            flattened.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
        return np.stack(flattened)

    def encode(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        if len(frames) == 0:
            raise ValueError("cannot encode an empty image sequence")
        if self.backend == "pixels":
            pixels = self._resize_pixels(
                frames, int(self.config.get("fallback_image_size", 32))
            )
            if self.pixel_projector is None:
                raise RuntimeError(
                    "pixel fallback must be fitted with fit_pixel_fallback() first"
                )
            return self.pixel_projector.transform(pixels)

        import torch

        batch_size = int(self.config.get("image_batch_size", 64))
        outputs: list[np.ndarray] = []
        for start in range(0, len(frames), batch_size):
            batch = [frame for frame in frames[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            outputs.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def fit_pixel_fallback(self, flattened_frames: np.ndarray, *, seed: int) -> None:
        """Fit the offline frame-level PCA used when CLIP is unavailable."""

        if self.backend != "pixels":
            return
        output_dim = int(self.config.get("fallback_frame_dim", 128))
        self.pixel_projector = PCAProjector(output_dim=output_dim, seed=seed)
        self.pixel_projector.fit(np.asarray(flattened_frames, dtype=np.float32))


class TrajectoryEncoder:
    """Create pooled action/state features for one trajectory segment."""

    def __init__(
        self,
        config: Mapping[str, Any],
        normalizers: NumericNormalizers,
        vector_keys: Sequence[str],
        image_keys: Sequence[str],
    ):
        self.config = dict(config)
        self.normalizers = normalizers
        self.vector_keys = tuple(vector_keys)
        self.image_keys = tuple(image_keys)
        self.vision = VisionFeatureExtractor(config) if self.image_keys else None

    @property
    def backend(self) -> str:
        return self.vision.backend if self.vision else "vector-only"

    def encode_raw(
        self, segment: TrajectorySegment
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return pooled feature and optional frame-level visual state.

        The frame-level visual sequence is returned for image-only quality
        scoring.  Vector observations remain the preferred quality state.
        """

        parts = [temporal_pool(self.normalizers.action(segment.actions))]
        numeric_state = self.normalizers.state_sequence(segment)
        if numeric_state is not None:
            parts.append(temporal_pool(numeric_state))

        visual_for_quality: list[np.ndarray] = []
        max_frames = int(self.config.get("max_frames_per_segment", 8))
        for key in self.image_keys:
            if key not in segment.observations:
                continue
            indices = uniformly_sample_indices(segment.length, max_frames)
            visual = self.vision.encode(  # type: ignore[union-attr]
                segment.observations[key][indices]
            )
            parts.append(temporal_pool(visual))
            visual_for_quality.append(visual)
        quality_state = (
            np.concatenate(visual_for_quality, axis=1) if visual_for_quality else None
        )
        return np.concatenate(parts).astype(np.float32), quality_state


class PCAProjector:
    """Standardize and project pooled features to a fixed output dimension."""

    def __init__(self, output_dim: int = 128, seed: int = 42):
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, *, max_samples: int | None = None
    ) -> "PCAProjector":
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("features must have shape [samples, dimensions]")
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1.0e-8] = 1.0
        standardized = (features - self.mean_) / self.scale_
        fit_data = standardized
        if max_samples and len(fit_data) > max_samples:
            rng = np.random.default_rng(self.seed)
            indices = np.sort(rng.choice(len(fit_data), size=max_samples, replace=False))
            fit_data = fit_data[indices]
        components = min(self.output_dim, fit_data.shape[1], max(1, len(fit_data) - 1))
        try:
            from sklearn.decomposition import PCA

            model = PCA(
                n_components=components,
                svd_solver="randomized" if components < min(fit_data.shape) else "full",
                random_state=self.seed,
            )
            model.fit(fit_data)
            self.components_ = model.components_.astype(np.float32)
            self.explained_variance_ratio_ = model.explained_variance_ratio_.astype(
                np.float32
            )
        except ImportError:
            # NumPy SVD keeps the core usable in minimal environments and tests.
            _, singular, vt = np.linalg.svd(fit_data, full_matrices=False)
            self.components_ = vt[:components].astype(np.float32)
            variance = singular**2
            total = float(variance.sum()) or 1.0
            self.explained_variance_ratio_ = (variance[:components] / total).astype(
                np.float32
            )
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise RuntimeError("PCAProjector must be fitted before transform")
        standardized = (np.asarray(features, dtype=np.float32) - self.mean_) / self.scale_
        projected = standardized @ self.components_.T
        if projected.shape[1] < self.output_dim:
            projected = np.pad(
                projected,
                ((0, 0), (0, self.output_dim - projected.shape[1])),
            )
        # A common L2 scale makes cosine, Euclidean KNN, and RBF distances stable.
        norm = np.linalg.norm(projected, axis=1, keepdims=True)
        return (projected / np.maximum(norm, 1.0e-8)).astype(np.float32)

    def fit_transform(
        self, features: np.ndarray, *, max_samples: int | None = None
    ) -> np.ndarray:
        return self.fit(features, max_samples=max_samples).transform(features)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)
