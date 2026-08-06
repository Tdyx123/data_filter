"""Exact quota-constrained greedy reference selector."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .objective import ObjectiveContext


@dataclass(frozen=True)
class LocalSwap:
    removed_index: int
    added_index: int
    improvement: float


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: list[int]
    marginal_gains: list[float]
    objective_value: float
    branch_id: int = 0
    local_swaps: tuple[LocalSwap, ...] = ()


class ExactGreedySelector:
    def __init__(
        self,
        context: ObjectiveContext,
        task_quotas: Mapping[int, int] | None,
    ):
        self.context = context
        self.task_quotas = (
            None
            if task_quotas is None
            else {int(task): int(value) for task, value in task_quotas.items()}
        )

    def select(
        self,
        budget: int,
        *,
        initial_indices: list[int] | None = None,
        branch_id: int = 0,
    ) -> SelectionResult:
        if self.task_quotas is not None and sum(self.task_quotas.values()) != budget:
            raise ValueError("task quotas must sum to the selection budget")
        if budget <= 0 or budget > len(self.context.graph.sample_ids):
            raise ValueError("selection budget must be within candidate count")
        selected = list(initial_indices or [])
        state = self.context.empty_state()
        gains: list[float] = []
        for index in selected:
            if self.task_quotas is not None:
                task = int(self.context.graph.task_indices[index])
                position = self.context.task_position[task]
                if state.task_counts[position] >= self.task_quotas[task]:
                    raise ValueError("initial selection exceeds a task quota")
            gain = self.context.marginal_gain(state, index)
            self.context.add_candidate(state, index)
            gains.append(gain)
        while len(selected) < budget:
            eligible = []
            for index, sample_id in enumerate(self.context.graph.sample_ids):
                if state.selected_mask[index]:
                    continue
                if self.task_quotas is None:
                    eligible.append((sample_id, index))
                    continue
                task = int(self.context.graph.task_indices[index])
                position = self.context.task_position[task]
                if state.task_counts[position] < self.task_quotas[task]:
                    eligible.append((sample_id, index))
            if not eligible:
                raise ValueError("no quota-feasible candidate remains")
            eligible.sort()
            best_index = eligible[0][1]
            best_gain = self.context.marginal_gain(state, best_index)
            for _, index in eligible[1:]:
                gain = self.context.marginal_gain(state, index)
                if gain > best_gain:
                    best_index, best_gain = index, gain
            self.context.add_candidate(state, best_index)
            selected.append(best_index)
            gains.append(float(best_gain))
        return SelectionResult(selected, gains, float(state.objective_value), branch_id)
