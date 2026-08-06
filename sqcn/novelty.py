"""Candidate-fragment k-nearest-neighbor novelty."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from segment_filter_core.scaling import robust_unit_interval


def _faiss_knn(embeddings: np.ndarray, k: int) -> np.ndarray:
    import faiss

    values = np.ascontiguousarray(embeddings.astype(np.float32))
    index = faiss.IndexFlatL2(values.shape[1])
    if hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
        try:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        except Exception:
            pass
    index.add(values)
    distances, _ = index.search(values, k + 1)
    return np.sqrt(np.maximum(distances[:, 1:], 0.0)).mean(axis=1)


def _sklearn_knn(embeddings: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    model.fit(embeddings)
    distances, _ = model.kneighbors(embeddings)
    return distances[:, 1:].mean(axis=1)


def _numpy_knn(embeddings: np.ndarray, k: int, batch_size: int) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    result = np.empty((len(values),), dtype=np.float32)
    right_norm = np.sum(values * values, axis=1)
    for start in range(0, len(values), batch_size):
        block = values[start : start + batch_size]
        distances2 = (
            np.sum(block * block, axis=1, keepdims=True)
            + right_norm[None, :]
            - 2.0 * block @ values.T
        )
        distances2 = np.maximum(distances2, 0.0)
        rows = np.arange(len(block))
        distances2[rows, start + rows] = np.inf
        nearest = np.partition(distances2, kth=k - 1, axis=1)[:, :k]
        result[start : start + len(block)] = np.sqrt(nearest).mean(axis=1)
    return result


def novelty_scores(
    embeddings: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("embeddings must have shape [samples, dimensions]")
    if len(values) == 1:
        zeros = np.zeros((1,), dtype=np.float32)
        return zeros, zeros, (0.0, 0.0)
    k = min(int(config.get("k", 20)), len(values) - 1)
    if k <= 0:
        raise ValueError("novelty k must be positive")
    backend = str(config.get("backend", "auto")).lower()
    raw: np.ndarray
    if backend in {"auto", "faiss"}:
        try:
            raw = _faiss_knn(values, k)
        except ImportError:
            if backend == "faiss":
                raise
        else:
            normalized, bounds = robust_unit_interval(
                raw,
                low=float(config.get("quantile_low", 0.01)),
                high=float(config.get("quantile_high", 0.99)),
            )
            return normalized, raw.astype(np.float32), bounds
    if backend in {"auto", "sklearn"}:
        try:
            raw = _sklearn_knn(values, k)
        except ImportError:
            if backend == "sklearn":
                raise
        else:
            normalized, bounds = robust_unit_interval(
                raw,
                low=float(config.get("quantile_low", 0.01)),
                high=float(config.get("quantile_high", 0.99)),
            )
            return normalized, raw.astype(np.float32), bounds
    raw = _numpy_knn(values, k, int(config.get("batch_size", 2048)))
    normalized, bounds = robust_unit_interval(
        raw,
        low=float(config.get("quantile_low", 0.01)),
        high=float(config.get("quantile_high", 0.99)),
    )
    return normalized, raw.astype(np.float32), bounds
