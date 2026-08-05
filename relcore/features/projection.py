"""Standardized deterministic PCA with fixed-width zero padding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RelationProjector:
    output_dim: int
    seed: int = 42
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    components_: np.ndarray | None = None

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("PCA inputs must be finite [samples, dim]")
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1.0e-8] = 1.0
        standardized = (values - self.mean_) / self.scale_
        components = min(
            self.output_dim,
            standardized.shape[1],
            max(1, len(standardized) - 1),
        )
        from sklearn.decomposition import PCA

        model = PCA(
            n_components=components,
            svd_solver=("randomized" if components < min(standardized.shape) else "full"),
            random_state=self.seed,
        )
        projected = model.fit_transform(standardized).astype(np.float32)
        self.components_ = model.components_.astype(np.float32)
        if projected.shape[1] < self.output_dim:
            projected = np.pad(
                projected,
                ((0, 0), (0, self.output_dim - projected.shape[1])),
            )
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return (projected / np.maximum(norms, 1.0e-8)).astype(np.float32)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise RuntimeError("RelationProjector must be fitted before transform")
        standardized = (np.asarray(features, dtype=np.float32) - self.mean_) / self.scale_
        projected = standardized @ self.components_.T
        if projected.shape[1] < self.output_dim:
            projected = np.pad(
                projected,
                ((0, 0), (0, self.output_dim - projected.shape[1])),
            )
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return (projected / np.maximum(norms, 1.0e-8)).astype(np.float32)
