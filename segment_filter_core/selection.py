"""Diversity-aware reranking over SQCN scores and fragment embeddings."""

from __future__ import annotations

import heapq
import math
import secrets
from dataclasses import dataclass
from typing import Sequence

import numpy as np


SIGMA_EPSILON = 1.0e-8
PENALTY_LAMBDA = 1.0
UPDATE_BATCH_SIZE = 512


@dataclass(frozen=True)
class _AlgorithmParameters:
    init_select_size: int = 100
    candidate_capacity: int = 100
    neighbor_count: int = 5


@dataclass(frozen=True)
class SelectionResult:
    """Ordered source rows and their diagnostics at selection time."""

    selected_indices: np.ndarray
    adjusted_scores: np.ndarray
    knn_penalties: np.ndarray
    sigma_raw: float
    sigma_effective: float
    seed: int


def _validate_inputs(
    scores: np.ndarray,
    embeddings: np.ndarray,
    sample_ids: Sequence[str],
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    encoded = np.asarray(embeddings)
    identifiers = np.asarray(tuple(str(value) for value in sample_ids), dtype=str)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if encoded.ndim != 2 or encoded.shape[0] != len(values):
        raise ValueError("embeddings must have one row for every score")
    if encoded.shape[1] == 0:
        raise ValueError("embeddings must have a positive feature dimension")
    if len(identifiers) != len(values):
        raise ValueError("sample_ids must have one value for every score")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("scores must contain finite values in [0, 1]")
    if encoded.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("embeddings must contain numeric real values")
    if not np.all(np.isfinite(encoded)):
        raise ValueError("embeddings must contain only finite values")
    if len(set(identifiers.tolist())) != len(identifiers) or np.any(identifiers == ""):
        raise ValueError("sample_ids must be non-empty and unique")
    if len(values) < 100:
        raise ValueError("scores must contain at least 100 fragments")
    if isinstance(target_size, bool) or not isinstance(target_size, (int, np.integer)):
        raise ValueError("target_size must be an integer")
    if int(target_size) < 100:
        raise ValueError("target_size must be at least 100")
    if int(target_size) > len(values):
        raise ValueError("target_size must be in [100, sample_count]")
    return values, encoded.astype(np.float64, copy=False), identifiers


def _raw_order(scores: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((sample_ids, -scores)).astype(np.int64, copy=False)


def _sigma_from_top_scores(
    embeddings: np.ndarray,
    raw_order: np.ndarray,
) -> tuple[float, float]:
    top = embeddings[raw_order[: min(100, len(raw_order))]]
    if len(top) < 2:
        raw = 0.0
    else:
        deltas = top[:, None, :] - top[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        raw = float(distances[np.triu_indices(len(top), k=1)].mean())
    effective = max(raw, SIGMA_EPSILON)
    return raw, effective


class _DiverseSelector:
    def __init__(
        self,
        scores: np.ndarray,
        embeddings: np.ndarray,
        sample_ids: np.ndarray,
        *,
        seed: int,
        sigma_effective: float,
        raw_order: np.ndarray | None = None,
        parameters: _AlgorithmParameters = _AlgorithmParameters(),
    ):
        self.scores = scores
        self.embeddings = embeddings
        self.sample_ids = sample_ids
        self.rng = np.random.default_rng(seed)
        self.sigma = sigma_effective
        self.parameters = parameters
        self._raw_order = (
            _raw_order(scores, sample_ids)
            if raw_order is None
            else np.asarray(raw_order, dtype=np.int64)
        )
        sample_id_order = np.argsort(sample_ids)
        self._sample_id_ranks = np.empty(len(sample_ids), dtype=np.int64)
        self._sample_id_ranks[sample_id_order] = np.arange(len(sample_ids), dtype=np.int64)
        self.penalties = np.zeros(len(scores), dtype=np.float64)
        self.adjusted = scores.copy()
        self.update_counts = np.zeros(len(scores), dtype=np.int64)
        self._neighbor_indices = np.full(
            (len(scores), self.parameters.neighbor_count),
            -1,
            dtype=np.int64,
        )
        self._neighbor_similarities = np.zeros(
            (len(scores), self.parameters.neighbor_count),
            dtype=np.float64,
        )
        self.selected: list[int] = []
        self.candidates: set[int] = set()
        self.silent: set[int] = set()
        self._silent_heap: list[tuple[float, float, str, int, int]] = []
        self._silent_versions = np.zeros(len(scores), dtype=np.int64)
        self._selected_adjusted: list[float] = []
        self._selected_knn_penalties: list[float] = []

    def _rank_key(self, index: int) -> tuple[float, float, str]:
        return (
            -float(self.adjusted[index]),
            -float(self.scores[index]),
            str(self.sample_ids[index]),
        )

    def _best(self, indices: set[int] | list[int]) -> int:
        return min(indices, key=self._rank_key)

    def _update(self, indices: set[int] | list[int], references: list[int]) -> None:
        if not indices or not references:
            return
        targets = np.asarray(sorted(indices), dtype=np.int64)
        reference = np.unique(np.asarray(references, dtype=np.int64))
        for start in range(0, len(targets), UPDATE_BATCH_SIZE):
            self._update_batch(targets[start : start + UPDATE_BATCH_SIZE], reference)

    def _update_batch(self, target: np.ndarray, reference: np.ndarray) -> None:
        deltas = self.embeddings[target, None, :] - self.embeddings[reference, :][None, :, :]
        squared_distances = np.einsum("ijk,ijk->ij", deltas, deltas)
        similarities = np.exp(-squared_distances / (2.0 * self.sigma * self.sigma))
        retained_indices = self._neighbor_indices[target].copy()
        retained_similarities = self._neighbor_similarities[target].copy()
        duplicate_retained = np.any(
            retained_indices[:, :, None] == reference[None, None, :],
            axis=2,
        )
        retained_indices[duplicate_retained] = -1
        retained_similarities[duplicate_retained] = 0.0

        new_indices = np.broadcast_to(reference, similarities.shape)
        combined_indices = np.concatenate((retained_indices, new_indices), axis=1)
        combined_similarities = np.concatenate(
            (retained_similarities, similarities),
            axis=1,
        )
        valid = combined_indices >= 0
        safe_indices = np.where(valid, combined_indices, 0)
        id_ranks = np.where(
            valid,
            self._sample_id_ranks[safe_indices],
            np.iinfo(np.int64).max,
        )
        nearest_positions = np.lexsort(
            (id_ranks, -combined_similarities),
            axis=1,
        )[:, : self.parameters.neighbor_count]
        nearest_indices = np.take_along_axis(combined_indices, nearest_positions, axis=1)
        nearest_similarities = np.take_along_axis(
            combined_similarities,
            nearest_positions,
            axis=1,
        )
        nearest_valid = nearest_indices >= 0
        self._neighbor_indices[target] = np.where(nearest_valid, nearest_indices, -1)
        self._neighbor_similarities[target] = np.where(
            nearest_valid,
            nearest_similarities,
            0.0,
        )
        nearest_safe_indices = np.where(nearest_valid, nearest_indices, 0)
        weighted_scores = np.where(
            nearest_valid,
            nearest_similarities * self.scores[nearest_safe_indices],
            0.0,
        )
        neighbor_counts = nearest_valid.sum(axis=1)
        self.penalties[target] = np.divide(
            weighted_scores.sum(axis=1),
            neighbor_counts,
            out=np.zeros(len(target), dtype=np.float64),
            where=neighbor_counts > 0,
        )
        self.adjusted[target] = self.scores[target] - PENALTY_LAMBDA * self.penalties[target]

    def _push_silent(self, index: int) -> None:
        self.silent.add(index)
        self._silent_versions[index] += 1
        version = int(self._silent_versions[index])
        heapq.heappush(
            self._silent_heap,
            (
                -float(self.adjusted[index]),
                -float(self.scores[index]),
                str(self.sample_ids[index]),
                index,
                version,
            ),
        )

    def _peek_silent(self) -> int:
        while self._silent_heap:
            _, _, _, index, version = self._silent_heap[0]
            if index in self.silent and version == self._silent_versions[index]:
                return index
            heapq.heappop(self._silent_heap)
        raise RuntimeError("silent heap is empty while silent items remain")

    def _remove_silent(self, index: int) -> None:
        self.silent.remove(index)

    def _pop_silent(self) -> int:
        index = self._peek_silent()
        self._remove_silent(index)
        return index

    def _fill_initial_candidates(self) -> None:
        count = min(self.parameters.candidate_capacity, len(self.silent))
        for _ in range(count):
            self.candidates.add(self._pop_silent())

    def _required_update_count(self, selected_count: int) -> int:
        excess = selected_count - self.parameters.init_select_size
        if excess < 1:
            raise ValueError("selected_count must exceed init_select_size")
        return math.ceil(self.parameters.init_select_size + math.log2(excess))

    def _sample_catch_up_references(self, count: int) -> list[int]:
        if count == 0:
            return []
        positions = self.rng.choice(
            len(self.selected),
            size=count,
            replace=False,
        )
        return [self.selected[int(position)] for position in positions]

    def _promote_one_silent(self) -> None:
        if not self.silent:
            return
        index = self._pop_silent()
        required = self._required_update_count(len(self.selected))
        deficit = max(0, required - int(self.update_counts[index]))
        references = self._sample_catch_up_references(deficit)
        self._update([index], references)
        self.update_counts[index] += deficit
        self.candidates.add(index)

    def _record_selected(self, index: int) -> None:
        self.selected.append(index)
        self._selected_adjusted.append(float(self.adjusted[index]))
        self._selected_knn_penalties.append(float(self.penalties[index]))

    def select(self, target_size: int) -> tuple[np.ndarray, ...]:
        initial_count = min(
            self.parameters.init_select_size,
            target_size,
            len(self.scores),
        )
        for value in self._raw_order[:initial_count]:
            index = int(value)
            self._record_selected(index)
        if len(self.selected) >= target_size:
            return self._selection_arrays()

        remaining = [int(index) for index in self._raw_order[initial_count:]]
        self._update(remaining, self.selected)
        self.update_counts[remaining] = initial_count
        for index in remaining:
            self._push_silent(index)
        self._fill_initial_candidates()

        while len(self.selected) < target_size:
            if not self.candidates:
                break

            chosen = self._best(self.candidates)
            self.candidates.remove(chosen)
            self._record_selected(chosen)
            if len(self.selected) >= target_size:
                break
            candidate_indices = sorted(self.candidates)
            self._update(candidate_indices, [chosen])
            self.update_counts[candidate_indices] += 1
            self._promote_one_silent()

        if len(self.selected) != target_size:
            raise RuntimeError(
                "SQCN filtering stopped before target_size: "
                f"selected={len(self.selected)}, candidates={len(self.candidates)}, "
                f"silent={len(self.silent)}, target={target_size}"
            )
        return self._result_arrays()

    def _result_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        groups = (
            set(self.selected),
            self.candidates,
            self.silent,
        )
        if sum(len(group) for group in groups) != len(self.scores) or set().union(*groups) != set(
            range(len(self.scores))
        ):
            raise RuntimeError(
                "SQCN filtering state partition is inconsistent: "
                f"selected={len(groups[0])}, candidates={len(groups[1])}, "
                f"silent={len(groups[2])}"
            )
        return self._selection_arrays()

    def _selection_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.selected, dtype=np.int64),
            np.asarray(self._selected_adjusted, dtype=np.float64),
            np.asarray(self._selected_knn_penalties, dtype=np.float64),
        )


def select_diverse_fragments(
    scores: np.ndarray,
    embeddings: np.ndarray,
    sample_ids: Sequence[str],
    target_size: int,
    *,
    seed: int | None = None,
) -> SelectionResult:
    """Select and order ``target_size`` fragments by score and diversity."""

    values, encoded, identifiers = _validate_inputs(
        scores,
        embeddings,
        sample_ids,
        target_size,
    )
    if seed is not None and (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= int(seed) < 2**64
    ):
        raise ValueError("seed must be an integer in [0, 2**64)")
    actual_seed = secrets.randbits(64) if seed is None else int(seed)
    order = _raw_order(values, identifiers)
    sigma_raw, sigma_effective = _sigma_from_top_scores(encoded, order)
    selector = _DiverseSelector(
        values,
        encoded,
        identifiers,
        seed=actual_seed,
        sigma_effective=sigma_effective,
        raw_order=order,
    )
    selected, adjusted_scores, knn_penalties = selector.select(int(target_size))
    return SelectionResult(
        selected_indices=selected,
        adjusted_scores=adjusted_scores,
        knn_penalties=knn_penalties,
        sigma_raw=sigma_raw,
        sigma_effective=sigma_effective,
        seed=actual_seed,
    )
