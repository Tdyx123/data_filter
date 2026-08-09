"""Deterministic soft behavior prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrototypeData:
    centers: np.ndarray | None
    indices: np.ndarray
    weights: np.ndarray
    labels: tuple[str, ...] | None = None

    @property
    def count(self) -> int:
        if self.labels is not None:
            return len(self.labels)
        if self.centers is None:
            raise ValueError("prototype data has neither centers nor labels")
        return len(self.centers)


def valid_prototype_assignments(
    indices: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prototype_indices = np.asarray(indices)
    prototype_weights = np.asarray(weights)
    if prototype_indices.ndim != 1 or prototype_indices.shape != prototype_weights.shape:
        raise ValueError("prototype indices and weights must be matching one-dimensional arrays")
    if len(prototype_indices) == 0 or (
        prototype_indices[-1] >= 0 and prototype_weights[-1] > 0.0
    ):
        return prototype_indices, prototype_weights
    mask = (prototype_indices >= 0) & (prototype_weights > 0.0)
    return prototype_indices[mask], prototype_weights[mask]


def discover_prototypes(
    embeddings: np.ndarray,
    *,
    count: int = 64,
    batch_size: int = 4096,
    max_iter: int = 100,
    top_r: int = 3,
    temperature: float = 0.1,
    seed: int = 42,
) -> PrototypeData:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or len(values) < count or not np.all(np.isfinite(values)):
        raise ValueError("prototype count cannot exceed the number of embeddings")
    if not 0 < top_r <= count or temperature <= 0:
        raise ValueError("prototype top_r/temperature configuration is invalid")
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=count,
        batch_size=min(batch_size, len(values)),
        max_iter=max_iter,
        random_state=seed,
        n_init=10,
    )
    model.fit(values)
    centers = model.cluster_centers_.astype(np.float32)
    distances2 = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    logits = -distances2 / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    indices = np.argsort(-probabilities, axis=1, kind="stable")[:, :top_r]
    weights = np.take_along_axis(probabilities, indices, axis=1)
    return PrototypeData(centers, indices.astype(np.int32), weights.astype(np.float32))
