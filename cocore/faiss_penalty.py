"""Exact FAISS Flat cosine indexes for Cocore branch redundancy."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _load_faiss():
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "random_multibranch requires faiss-cpu>=1.9; install cocore/requirements.txt"
        ) from error

    return faiss


class FaissFlatPenaltySpace:
    """Shared normalized embeddings and penalty constants for branch-local indexes."""

    def __init__(
        self,
        embeddings: np.ndarray,
        reliability: np.ndarray,
        *,
        similarity_threshold: float,
        redundancy_normalizer: float,
        epsilon: float = 1.0e-8,
    ) -> None:
        values = np.asarray(embeddings, dtype=np.float32)
        quality = np.asarray(reliability, dtype=np.float32)
        threshold = float(similarity_threshold)
        normalizer = float(redundancy_normalizer)
        tolerance = float(epsilon)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] == 0
            or quality.shape != (values.shape[0],)
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(quality))
        ):
            raise ValueError("FAISS penalty inputs must be finite and aligned")
        if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ValueError("similarity_threshold must be in [0, 1)")
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise ValueError("redundancy_normalizer must be finite and positive")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("epsilon must be finite and positive")

        norms = np.linalg.norm(values, axis=1, keepdims=True)
        self.vectors = np.ascontiguousarray(values / np.maximum(norms, tolerance))
        self.reliability = quality
        self.similarity_threshold = threshold
        self.redundancy_normalizer = normalizer
        self.epsilon = tolerance
        self._faiss = _load_faiss()

    def create_index(
        self,
        indices: Sequence[int] = (),
    ) -> FaissFlatPenaltyIndex:
        return FaissFlatPenaltyIndex(self, indices)

    def penalty(
        self,
        candidate: int,
        indexes: Sequence[FaissFlatPenaltyIndex],
    ) -> float:
        index = int(candidate)
        matches = sorted(
            match for penalty_index in indexes for match in penalty_index.matches(index)
        )
        denominator = max(1.0 - self.similarity_threshold, self.epsilon)
        raw_penalty = math.fsum(
            float(self.reliability[index])
            * float(self.reliability[other])
            * max(0.0, similarity - self.similarity_threshold)
            / denominator
            for other, similarity in matches
        )
        return float(raw_penalty / self.redundancy_normalizer)


class FaissFlatPenaltyIndex:
    """Mutable exact inner-product index over one selected reference set."""

    def __init__(
        self,
        space: FaissFlatPenaltySpace,
        indices: Sequence[int] = (),
    ) -> None:
        self.space = space
        self.index = space._faiss.IndexFlatIP(space.vectors.shape[1])
        self.global_indices = np.empty(0, dtype=np.int64)
        self._global_index_set: set[int] = set()
        self.add(indices)

    def add(self, indices: Sequence[int]) -> None:
        added = tuple(int(index) for index in indices)
        if len(added) != len(set(added)):
            raise ValueError("FAISS penalty indices cannot contain duplicates")
        candidate_count = len(self.space.vectors)
        if any(index < 0 or index >= candidate_count for index in added):
            raise ValueError("FAISS penalty indices must be in range")
        if any(index in self._global_index_set for index in added):
            raise ValueError("FAISS penalty index already contains a requested index")
        if not added:
            return
        added_array = np.asarray(added, dtype=np.int64)
        self.index.add(np.ascontiguousarray(self.space.vectors[added_array]))
        self.global_indices = np.concatenate((self.global_indices, added_array))
        self._global_index_set.update(added)

    def matches(self, candidate: int) -> tuple[tuple[int, float], ...]:
        index = int(candidate)
        if index < 0 or index >= len(self.space.vectors):
            raise ValueError("FAISS penalty candidate must be in range")
        if index in self._global_index_set:
            raise ValueError("FAISS penalty candidate is already indexed")
        if self.index.ntotal == 0:
            return ()

        limits, similarities, local_indices = self.index.range_search(
            self.space.vectors[index : index + 1],
            float(
                np.nextafter(
                    np.float32(self.space.similarity_threshold),
                    np.float32(-np.inf),
                )
            ),
        )
        start, stop = int(limits[0]), int(limits[1])
        return tuple(
            sorted(
                (
                    int(self.global_indices[int(local)]),
                    min(1.0, max(0.0, float(similarity))),
                )
                for local, similarity in zip(
                    local_indices[start:stop],
                    similarities[start:stop],
                    strict=True,
                )
            )
        )

    def penalty(self, candidate: int) -> float:
        return self.space.penalty(candidate, (self,))
