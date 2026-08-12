"""Coverage seeding and deterministic lazy maximum-heap selection."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .objective import CocoreObjectiveContext


@dataclass(frozen=True)
class CoverageSeed:
    selected_indices: tuple[int, ...]
    target_coverage: np.ndarray
    achieved_coverage: np.ndarray


@dataclass(frozen=True)
class HeapSelectionResult:
    selected_indices: tuple[int, ...]
    score_deltas: tuple[float, ...]
    selection_phases: tuple[str, ...]
    selection_steps: tuple[int, ...]
    heap_refreshes: tuple[int, ...]
    objective_value: float
    cooccurrence: float
    redundancy: float
    initial_heap_size: int
    total_refreshes: int
    capped_selections: int
    max_refreshes_observed: int


def build_max_coverage_seed(
    context: CocoreObjectiveContext,
    *,
    budget: int,
) -> CoverageSeed:
    if budget <= 0 or budget > len(context.graph.sample_ids):
        raise ValueError("selection budget must be within candidate count")
    target = context.prototype_mass.max(axis=0)
    selected: list[int] = []
    selected_set: set[int] = set()
    for prototype, maximum in enumerate(target):
        if maximum <= 0.0:
            continue
        matches = np.flatnonzero(context.prototype_mass[:, prototype] == maximum)
        winner = min(matches.tolist(), key=lambda index: context.graph.sample_ids[index])
        if winner not in selected_set:
            selected.append(winner)
            selected_set.add(winner)
    if len(selected) > budget:
        raise ValueError(
            f"coverage seed exceeds selection budget; minimum required budget is {len(selected)}"
        )
    achieved = (
        context.prototype_mass[np.asarray(selected, dtype=np.int64)].max(axis=0)
        if selected
        else np.zeros_like(target)
    )
    return CoverageSeed(tuple(selected), target.copy(), achieved)


_HeapEntry = tuple[float, str, int, int]


class LazyHeapSelector:
    """Approximate greedy selection with bounded lazy marginal-gain refreshes."""

    def __init__(
        self,
        context: CocoreObjectiveContext,
        *,
        max_refreshes: int = 100,
    ) -> None:
        if isinstance(max_refreshes, bool) or not isinstance(
            max_refreshes, (int, np.integer)
        ):
            raise ValueError("max_refreshes must be a positive integer")
        if int(max_refreshes) <= 0:
            raise ValueError("max_refreshes must be a positive integer")
        self.context = context
        self.max_refreshes = int(max_refreshes)

    @staticmethod
    def _finite_gain(gain: float, candidate: int) -> float:
        value = float(gain)
        if not math.isfinite(value):
            raise ValueError(f"candidate {candidate} has a non-finite marginal gain")
        return value

    def _entry(self, gain: float, candidate: int, version: int) -> _HeapEntry:
        return (
            -self._finite_gain(gain, candidate),
            self.context.graph.sample_ids[candidate],
            candidate,
            version,
        )

    @staticmethod
    def _is_full_refresh_step(step: int) -> bool:
        return step >= 8 and (step & (step - 1)) == 0

    def _build_heap(self, state, version: int) -> list[_HeapEntry]:
        heap = [
            self._entry(self.context.marginal_gain(state, candidate), candidate, version)
            for candidate in range(len(self.context.graph.sample_ids))
            if not state.selected_mask[candidate]
        ]
        heapq.heapify(heap)
        return heap

    def select(
        self,
        budget: int,
        *,
        initial_indices: list[int] | tuple[int, ...],
    ) -> HeapSelectionResult:
        candidate_count = len(self.context.graph.sample_ids)
        initial = tuple(int(index) for index in initial_indices)
        if len(initial) != len(set(initial)):
            raise ValueError("initial_indices cannot contain duplicates")
        if any(index < 0 or index >= candidate_count for index in initial):
            raise ValueError("initial_indices must contain in-range values")
        if not 0 < len(initial) <= budget <= candidate_count:
            raise ValueError("budget must contain a non-empty initial selection")

        state = self.context.empty_state()
        selected = list(initial)
        score_deltas: list[float] = []
        for index in initial:
            gain = self._finite_gain(self.context.marginal_gain(state, index), index)
            self.context.add_candidate(state, index)
            score_deltas.append(gain)

        selection_phases = ["coverage_seed"] * len(initial)
        selection_steps = [0] * len(initial)
        refresh_counts = [0] * len(initial)
        if len(initial) == budget:
            return self._result(
                state,
                selected,
                score_deltas,
                selection_phases,
                selection_steps,
                refresh_counts,
                initial_heap_size=0,
                capped_selections=0,
            )

        current_version = 0
        heap = self._build_heap(state, current_version)
        initial_heap_size = len(heap)
        capped_selections = 0
        heap_step = 1

        while len(selected) < budget:
            if self._is_full_refresh_step(heap_step):
                heap = self._build_heap(state, current_version)

            refreshed: list[_HeapEntry] = []
            chosen_index: int | None = None
            chosen_gain = 0.0
            hit_refresh_cap = False

            while heap:
                negative_gain, sample_id, candidate, version = heapq.heappop(heap)
                if state.selected_mask[candidate]:
                    continue
                if version == current_version:
                    chosen_index = candidate
                    chosen_gain = -negative_gain
                    break

                gain = self.context.marginal_gain(state, candidate)
                entry = self._entry(gain, candidate, current_version)
                heapq.heappush(heap, entry)
                refreshed.append(entry)
                if len(refreshed) >= self.max_refreshes:
                    best = min(refreshed)
                    chosen_gain = -best[0]
                    chosen_index = best[2]
                    hit_refresh_cap = True
                    break

            if chosen_index is None:
                raise ValueError("no candidate remains before the selection budget is reached")

            self.context.add_candidate(state, chosen_index)
            selected.append(chosen_index)
            score_deltas.append(float(chosen_gain))
            selection_phases.append("heap")
            selection_steps.append(heap_step)
            refresh_counts.append(len(refreshed))
            capped_selections += int(hit_refresh_cap)
            current_version += 1
            heap_step += 1

        if len(selected) != len(set(selected)):
            raise RuntimeError("lazy heap selection produced duplicate indices")
        return self._result(
            state,
            selected,
            score_deltas,
            selection_phases,
            selection_steps,
            refresh_counts,
            initial_heap_size=initial_heap_size,
            capped_selections=capped_selections,
        )

    @staticmethod
    def _result(
        state,
        selected: list[int],
        score_deltas: list[float],
        selection_phases: list[str],
        selection_steps: list[int],
        refresh_counts: list[int],
        *,
        initial_heap_size: int,
        capped_selections: int,
    ) -> HeapSelectionResult:
        return HeapSelectionResult(
            selected_indices=tuple(selected),
            score_deltas=tuple(score_deltas),
            selection_phases=tuple(selection_phases),
            selection_steps=tuple(selection_steps),
            heap_refreshes=tuple(refresh_counts),
            objective_value=float(state.score),
            cooccurrence=float(state.cooccurrence),
            redundancy=float(state.redundancy),
            initial_heap_size=initial_heap_size,
            total_refreshes=sum(refresh_counts),
            capped_selections=capped_selections,
            max_refreshes_observed=max(refresh_counts, default=0),
        )
