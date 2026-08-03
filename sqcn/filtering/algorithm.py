"""Diversity-aware reranking over SQCN scores and fragment embeddings."""

from __future__ import annotations

import heapq
import secrets
from dataclasses import dataclass
from typing import Sequence

import numpy as np


SIGMA_EPSILON = 1.0e-8
WEIGHT_EPSILON = 1.0e-12
PENALTY_LAMBDA = 1.0


@dataclass(frozen=True)
class _AlgorithmParameters:
    init_select_size: int = 100
    new_batch_size: int = 100
    high_ref_size: int = 50
    random_ref_size: int = 50
    candidate_threshold: int = 50
    unseen_rank: int = 300
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
    if isinstance(target_size, bool) or not isinstance(target_size, (int, np.integer)):
        raise ValueError("target_size must be an integer")
    if not 0 < int(target_size) <= len(values):
        raise ValueError("target_size must be in [1, sample_count]")
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
        parameters: _AlgorithmParameters = _AlgorithmParameters(),
    ):
        self.scores = scores
        self.embeddings = embeddings
        self.sample_ids = sample_ids
        self.rng = np.random.default_rng(seed)
        self.sigma = sigma_effective
        self.parameters = parameters
        self.penalties = np.zeros(len(scores), dtype=np.float64)
        self.adjusted = scores.copy()
        self._neighbor_indices = np.full(
            (len(scores), self.parameters.neighbor_count),
            -1,
            dtype=np.int64,
        )
        self._neighbor_similarities = np.zeros(
            (len(scores), self.parameters.neighbor_count),
            dtype=np.float64,
        )
        self.refresh_round = np.full(len(scores), -1, dtype=np.int64)
        self.selected: list[int] = []
        self.candidates: set[int] = set()
        self.silent: set[int] = set()
        self.unseen: set[int] = set(range(len(scores)))
        self._silent_heap: list[tuple[float, float, str, int, int]] = []
        self._silent_versions = np.zeros(len(scores), dtype=np.int64)
        self._silent_heap_round = -1
        self._unseen_threshold = float("-inf")
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

    def _worst(self, indices: set[int]) -> int:
        return max(indices, key=self._rank_key)

    def _raw_ranked(self, indices: set[int] | list[int]) -> list[int]:
        return sorted(
            indices,
            key=lambda index: (
                -float(self.scores[index]),
                str(self.sample_ids[index]),
            ),
        )

    def _reference_set(self) -> list[int]:
        ranked = self._raw_ranked(self.selected)
        high_count = min(self.parameters.high_ref_size, len(ranked))
        high = ranked[:high_count]
        remaining = ranked[high_count:]
        random_count = min(self.parameters.random_ref_size, len(remaining))
        if random_count == 0:
            return high
        if random_count == len(remaining):
            sampled = remaining
        else:
            weights = np.maximum(self.scores[remaining], WEIGHT_EPSILON)
            probabilities = weights / weights.sum()
            positions = self.rng.choice(
                len(remaining),
                size=random_count,
                replace=False,
                p=probabilities,
            )
            sampled = [remaining[int(position)] for position in positions]
        return high + sampled

    def _update(self, indices: set[int] | list[int], references: list[int]) -> None:
        if not indices or not references:
            return
        target = np.asarray(sorted(indices), dtype=np.int64)
        reference = np.asarray(references, dtype=np.int64)
        deltas = self.embeddings[target, None, :] - self.embeddings[reference, :][None, :, :]
        squared_distances = np.einsum("ijk,ijk->ij", deltas, deltas)
        similarities = np.exp(-squared_distances / (2.0 * self.sigma * self.sigma))
        for row, index in enumerate(target):
            retained = {
                int(neighbor): float(similarity)
                for neighbor, similarity in zip(
                    self._neighbor_indices[index],
                    self._neighbor_similarities[index],
                    strict=True,
                )
                if neighbor >= 0
            }
            retained.update(
                {
                    int(neighbor): float(similarity)
                    for neighbor, similarity in zip(
                        reference,
                        similarities[row],
                        strict=True,
                    )
                }
            )
            nearest = sorted(
                retained.items(),
                key=lambda item: (
                    -item[1],
                    str(self.sample_ids[item[0]]),
                ),
            )[: self.parameters.neighbor_count]
            self._neighbor_indices[index].fill(-1)
            self._neighbor_similarities[index].fill(0.0)
            if nearest:
                neighbor_indices = np.fromiter(
                    (neighbor for neighbor, _ in nearest),
                    dtype=np.int64,
                    count=len(nearest),
                )
                neighbor_similarities = np.fromiter(
                    (similarity for _, similarity in nearest),
                    dtype=np.float64,
                    count=len(nearest),
                )
                self._neighbor_indices[index, : len(nearest)] = neighbor_indices
                self._neighbor_similarities[index, : len(nearest)] = neighbor_similarities
                self.penalties[index] = float(
                    np.mean(neighbor_similarities * self.scores[neighbor_indices])
                )
            else:
                self.penalties[index] = 0.0
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

    def _prepare_silent_round(self, round_id: int) -> None:
        if self._silent_heap_round == round_id:
            return
        self._silent_heap_round = round_id
        self._silent_heap.clear()
        for index in self.silent:
            self._silent_versions[index] += 1
            version = int(self._silent_versions[index])
            heapq.heappush(
                self._silent_heap,
                (
                    -float(self.scores[index]),
                    -float(self.scores[index]),
                    str(self.sample_ids[index]),
                    index,
                    version,
                ),
            )

    def _remove_silent(self, index: int) -> None:
        self.silent.remove(index)

    def _refresh_silent(
        self,
        index: int,
        references: list[int],
        round_id: int,
    ) -> None:
        self._update([index], references)
        self.refresh_round[index] = round_id
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

    def _sample_unseen(self) -> list[int]:
        count = min(self.parameters.new_batch_size, len(self.unseen))
        if count == 0:
            return []
        population = sorted(
            self.unseen,
            key=lambda index: str(self.sample_ids[index]),
        )
        if count == len(population):
            sampled = population
        else:
            positions = self.rng.choice(len(population), size=count, replace=False)
            sampled = [population[int(position)] for position in positions]
        self.unseen.difference_update(sampled)
        return sampled

    def _refresh_unseen_threshold(self) -> None:
        rank = self.parameters.unseen_rank
        if len(self.unseen) < rank:
            self._unseen_threshold = float("-inf")
            return
        unseen_scores = np.fromiter(
            (self.scores[index] for index in self.unseen),
            dtype=np.float64,
            count=len(self.unseen),
        )
        position = len(unseen_scores) - rank
        self._unseen_threshold = float(np.partition(unseen_scores, position)[position])

    def _reactivate_silent(self, round_id: int) -> None:
        if not self.silent:
            return
        self._prepare_silent_round(round_id)
        references = self._reference_set()
        count = min(self.parameters.new_batch_size, len(self.silent))
        reactivated: set[int] = set()
        while len(reactivated) < count and self.silent:
            best = self._peek_silent()
            if self.refresh_round[best] != round_id:
                self._refresh_silent(best, references, round_id)
                continue
            self._remove_silent(best)
            reactivated.add(best)
        self.candidates.update(reactivated)

    def _add_candidates(self, round_id: int) -> None:
        references = self._reference_set()
        self._prepare_silent_round(round_id)
        newly_sampled = self._sample_unseen()
        self._refresh_unseen_threshold()
        if not newly_sampled:
            self._reactivate_silent(round_id)
            return
        self.penalties[newly_sampled] = 0.0
        self.adjusted[newly_sampled] = self.scores[newly_sampled]
        self._neighbor_indices[newly_sampled] = -1
        self._neighbor_similarities[newly_sampled] = 0.0
        self._update(newly_sampled, references)
        self.refresh_round[newly_sampled] = round_id
        admission = set(newly_sampled)
        while self.silent and admission:
            worst = self._worst(admission)
            best_silent = self._peek_silent()
            if self.refresh_round[best_silent] != round_id:
                self._refresh_silent(best_silent, references, round_id)
                continue
            if self.adjusted[best_silent] <= self.adjusted[worst]:
                break
            if self.adjusted[best_silent] > self.adjusted[worst]:
                admission.remove(worst)
                self._push_silent(worst)
                self._remove_silent(best_silent)
                admission.add(best_silent)
        self.candidates.update(admission)
        for index in newly_sampled:
            if index not in admission and index not in self.silent:
                self._push_silent(index)

    def _record_selected(self, index: int) -> None:
        self.selected.append(index)
        self._selected_adjusted.append(float(self.adjusted[index]))
        self._selected_knn_penalties.append(float(self.penalties[index]))

    def select(self, target_size: int, raw_order: np.ndarray) -> tuple[np.ndarray, ...]:
        initial_count = min(
            self.parameters.init_select_size,
            target_size,
            len(self.scores),
        )
        for value in raw_order[:initial_count]:
            index = int(value)
            self.unseen.remove(index)
            self._record_selected(index)
        if len(self.selected) >= target_size:
            return self._result_arrays()

        round_id = 0
        while len(self.selected) < target_size:
            if len(self.candidates) <= self.parameters.candidate_threshold:
                round_id += 1
                self._add_candidates(round_id)
            if not self.candidates:
                if not self.unseen and not self.silent:
                    break
                if not self.unseen and self.silent:
                    round_id += 1
                    self._reactivate_silent(round_id)
            if not self.candidates:
                break

            chosen = self._best(self.candidates)
            self.candidates.remove(chosen)
            self._record_selected(chosen)
            if len(self.selected) >= target_size:
                break
            self._update(self.candidates, [chosen])
            move_to_silent = [
                index for index in self.candidates if self.adjusted[index] < self._unseen_threshold
            ]
            for index in move_to_silent:
                self.candidates.remove(index)
                self._push_silent(index)

        if len(self.selected) != target_size:
            raise RuntimeError(
                "SQCN filtering stopped before target_size: "
                f"selected={len(self.selected)}, candidates={len(self.candidates)}, "
                f"silent={len(self.silent)}, unseen={len(self.unseen)}, "
                f"target={target_size}"
            )
        return self._result_arrays()

    def _result_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        groups = (
            set(self.selected),
            self.candidates,
            self.silent,
            self.unseen,
        )
        if sum(len(group) for group in groups) != len(self.scores) or set().union(*groups) != set(
            range(len(self.scores))
        ):
            raise RuntimeError(
                "SQCN filtering state partition is inconsistent: "
                f"selected={len(groups[0])}, candidates={len(groups[1])}, "
                f"silent={len(groups[2])}, unseen={len(groups[3])}"
            )
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
    )
    selected, adjusted_scores, knn_penalties = selector.select(int(target_size), order)
    return SelectionResult(
        selected_indices=selected,
        adjusted_scores=adjusted_scores,
        knn_penalties=knn_penalties,
        sigma_raw=sigma_raw,
        sigma_effective=sigma_effective,
        seed=actual_seed,
    )
