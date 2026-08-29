"""Selection strategies shared by Cocore ablation runs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from cocore.objective import CocoreObjectiveContext
from cocore.random_multibranch import RandomMultiBranchSelectionResult
from cocore.selection import build_max_coverage_seed


def initial_selection(
    context: CocoreObjectiveContext,
    *,
    budget: int,
    use_coverage_seed: bool,
) -> tuple[int, ...]:
    if not isinstance(use_coverage_seed, bool):
        raise ValueError("use_coverage_seed must be a boolean")
    if not use_coverage_seed:
        return ()
    return build_max_coverage_seed(context, budget=budget).selected_indices


class SeededRandomSelector:
    """Uniform sampling without replacement over the eligible graph nodes."""

    def __init__(self, context: CocoreObjectiveContext, *, seed: int = 42) -> None:
        self.context = context
        self.seed = int(seed)

    def select(
        self,
        budget: int,
        *,
        initial_indices: Sequence[int],
    ) -> RandomMultiBranchSelectionResult:
        candidate_count = len(self.context.graph.sample_ids)
        initial = tuple(int(index) for index in initial_indices)
        if len(initial) != len(set(initial)):
            raise ValueError("initial_indices cannot contain duplicates")
        if any(index < 0 or index >= candidate_count for index in initial):
            raise ValueError("initial_indices must contain in-range values")
        if not 0 <= len(initial) <= budget <= candidate_count or budget <= 0:
            raise ValueError("budget must include the initial selection and fit the pool")

        initial_set = set(initial)
        available = np.asarray(
            [index for index in range(candidate_count) if index not in initial_set],
            dtype=np.int64,
        )
        remaining = budget - len(initial)
        chosen = (
            tuple(
                int(index)
                for index in np.random.default_rng(self.seed)
                .choice(available, size=remaining, replace=False)
                .tolist()
            )
            if remaining
            else ()
        )
        selected = initial + chosen
        state = self.context.empty_state()
        gains: list[float] = []
        for index in selected:
            gain = self.context.marginal_gain(state, index)
            if not math.isfinite(gain):
                raise ValueError(f"candidate {index} has a non-finite marginal gain")
            gains.append(float(gain))
            self.context.add_candidate(state, index)
        return RandomMultiBranchSelectionResult(
            selected_indices=selected,
            score_deltas=tuple(gains),
            selection_phases=("coverage_seed",) * len(initial)
            + ("random",) * len(chosen),
            selection_steps=(0,) * len(initial) + (1,) * len(chosen),
            objective_value=float(state.score),
            relation=float(state.relation),
            redundancy=float(state.redundancy),
            rounds=0,
            evaluated_branches=0,
            recombinations=0,
            committed_clips=0,
            final_active_clips=len(chosen),
            round_runtime_seconds=(),
            recombination_runtime_seconds=(),
        )
