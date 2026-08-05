"""Deterministic soft behavior prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrototypeData:
    centers: np.ndarray
    indices: np.ndarray
    weights: np.ndarray


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
