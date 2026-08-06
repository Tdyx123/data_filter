"""Independent seeded sparse branches with bounded 1-swap refinement."""

from __future__ import annotations

from collections.abc import Mapping

from .greedy import LocalSwap, SelectionResult
from .objective import ObjectiveContext
from .seeds import generate_seed_pairs
from .sparse_greedy import SparseGreedySelector


def _replay_result(
    context: ObjectiveContext,
    selected: list[int],
    branch_id: int,
    local_swaps: tuple[LocalSwap, ...] = (),
) -> SelectionResult:
    state = context.empty_state()
    gains: list[float] = []
    for index in selected:
        gains.append(context.marginal_gain(state, index))
        context.add_candidate(state, index)
    return SelectionResult(selected, gains, float(state.objective_value), branch_id, local_swaps)


def _local_search(
    result: SelectionResult,
    context: ObjectiveContext,
    *,
    task_quotas: Mapping[int, int] | None,
    max_selected_candidates: int,
    max_unselected_candidates: int,
    max_rounds: int,
) -> SelectionResult:
    current = result
    accepted = list(result.local_swaps)
    for _ in range(max_rounds):
        state = context.state_from_indices(current.selected_indices)
        selected_positions = sorted(
            range(len(current.selected_indices)),
            key=lambda position: (
                current.marginal_gains[position],
                context.graph.sample_ids[current.selected_indices[position]],
            ),
        )[:max_selected_candidates]
        unselected = [
            index
            for index in range(len(context.graph.sample_ids))
            if not state.selected_mask[index]
        ]
        unselected.sort(
            key=lambda index: (
                -context.marginal_gain(state, index),
                context.graph.sample_ids[index],
            )
        )
        unselected = unselected[:max_unselected_candidates]
        best_order: list[int] | None = None
        best_value = current.objective_value
        best_swap: tuple[int, int] | None = None
        without_states = {}
        for position in selected_positions:
            without_state = context.clone_state(state)
            context.remove_candidate(without_state, current.selected_indices[position])
            without_states[position] = without_state
        for position in selected_positions:
            removed = current.selected_indices[position]
            removed_task = int(context.graph.task_indices[removed])
            without_state = without_states[position]
            for candidate in unselected:
                if (
                    task_quotas is not None
                    and int(context.graph.task_indices[candidate]) != removed_task
                ):
                    continue
                value = without_state.objective_value + context.marginal_gain(
                    without_state, candidate
                )
                if value > best_value + 1.0e-12:
                    best_value = value
                    trial = current.selected_indices.copy()
                    trial[position] = candidate
                    best_order = trial
                    best_swap = (removed, candidate)
        if best_order is None:
            break
        assert best_swap is not None
        accepted.append(
            LocalSwap(
                removed_index=best_swap[0],
                added_index=best_swap[1],
                improvement=float(best_value - current.objective_value),
            )
        )
        current = _replay_result(context, best_order, current.branch_id, tuple(accepted))
    return current


class MultiBranchSelector:
    def __init__(
        self,
        context: ObjectiveContext,
        task_quotas: Mapping[int, int] | None,
        *,
        branches: int = 8,
        seed: int = 42,
        seed_candidates: int = 128,
        seed_similarity_threshold: float = 0.9,
        transition_seed_threshold: float = 0.0,
        pairs_per_transition: int = 4,
        global_candidates: int = 256,
        residual_candidates: int = 128,
        random_candidates: int = 128,
        local_search_enabled: bool = True,
        local_search_selected: int = 128,
        local_search_unselected: int = 256,
        local_search_rounds: int = 2,
    ):
        self.context = context
        self.task_quotas = task_quotas
        self.branches = branches
        self.seed = seed
        self.seed_candidates = seed_candidates
        self.seed_similarity_threshold = seed_similarity_threshold
        self.transition_seed_threshold = transition_seed_threshold
        self.pairs_per_transition = pairs_per_transition
        self.global_candidates = global_candidates
        self.residual_candidates = residual_candidates
        self.random_candidates = random_candidates
        self.local_search_enabled = local_search_enabled
        self.local_search_selected = local_search_selected
        self.local_search_unselected = local_search_unselected
        self.local_search_rounds = local_search_rounds
        self.branch_results: list[SelectionResult] = []

    def select(self, budget: int) -> SelectionResult:
        seed_slots = max(self.branches - 1, 0)
        seeds = (
            generate_seed_pairs(
                self.context,
                self.task_quotas,
                budget=budget,
                branches=seed_slots,
                seed_candidates=self.seed_candidates,
                seed_similarity_threshold=self.seed_similarity_threshold,
                transition_seed_threshold=self.transition_seed_threshold,
                pairs_per_transition=self.pairs_per_transition,
            )
            if seed_slots
            else []
        )
        branch_seeds: list[list[int]] = [[]]
        branch_seeds.extend([list(pair) for pair in seeds])
        branch_seeds.extend([[] for _ in range(max(0, self.branches - len(branch_seeds)))])
        branch_seeds = branch_seeds[: self.branches]
        results: list[SelectionResult] = []
        for branch_id, initial in enumerate(branch_seeds):
            selector = SparseGreedySelector(
                self.context,
                self.task_quotas,
                seed=self.seed + branch_id,
                global_candidates=self.global_candidates,
                residual_candidates=self.residual_candidates,
                random_candidates=self.random_candidates,
            )
            results.append(
                selector.select(
                    budget,
                    initial_indices=initial,
                    branch_id=branch_id,
                )
            )
        self.branch_results = list(results)
        best = min(
            results,
            key=lambda result: (
                -result.objective_value,
                tuple(
                    sorted(
                        self.context.graph.sample_ids[index] for index in result.selected_indices
                    )
                ),
            ),
        )
        if self.local_search_enabled:
            best = _local_search(
                best,
                self.context,
                task_quotas=self.task_quotas,
                max_selected_candidates=self.local_search_selected,
                max_unselected_candidates=self.local_search_unselected,
                max_rounds=self.local_search_rounds,
            )
        return best
