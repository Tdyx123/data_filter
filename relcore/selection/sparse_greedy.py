"""Quota-constrained greedy selection over a dynamic sparse candidate pool."""

from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np

from .greedy import SelectionResult
from .objective import ObjectiveContext


class SparseGreedySelector:
    def __init__(
        self,
        context: ObjectiveContext,
        task_quotas: dict[int, int],
        *,
        seed: int = 42,
        global_candidates: int = 256,
        residual_candidates: int = 128,
        random_candidates: int = 128,
    ):
        self.context = context
        self.task_quotas = {int(task): int(value) for task, value in task_quotas.items()}
        self.seed = int(seed)
        self.global_candidates = int(global_candidates)
        self.residual_candidates = int(residual_candidates)
        self.random_candidates = int(random_candidates)
        self.node_task_positions = np.asarray(
            [context.task_position[int(task)] for task in context.graph.task_indices],
            dtype=np.int64,
        )
        self.quota_limits = np.asarray(
            [self.task_quotas[task] for task in context.task_labels],
            dtype=np.int64,
        )
        self.neighbors: list[set[int]] = [set() for _ in context.graph.sample_ids]
        for table in (context.graph.sequence_edges, context.graph.similarity_edges):
            for source, target in zip(table.source, table.target, strict=True):
                left, right = int(source), int(target)
                self.neighbors[left].add(right)
                self.neighbors[right].add(left)
        self.prototype_nodes: list[list[int]] = [[] for _ in range(context.prototype_count)]
        self.task_nodes: dict[int, list[int]] = defaultdict(list)
        for index in range(len(context.graph.sample_ids)):
            self.task_nodes[int(context.graph.task_indices[index])].append(index)
            for prototype in context.graph.prototype_indices[index]:
                self.prototype_nodes[int(prototype)].append(index)

        def ordering(index: int) -> tuple[float, str]:
            return (
                -float(context.graph.reliability[index]),
                context.graph.sample_ids[index],
            )

        for nodes in self.prototype_nodes:
            nodes.sort(key=ordering)
        for nodes in self.task_nodes.values():
            nodes.sort(key=ordering)

    def _eligible(self, state, index: int) -> bool:
        if state.selected_mask[index]:
            return False
        task = int(self.context.graph.task_indices[index])
        position = self.context.task_position[task]
        return state.task_counts[position] < self.task_quotas[task]

    def select(
        self,
        budget: int,
        *,
        initial_indices: list[int] | None = None,
        branch_id: int = 0,
    ) -> SelectionResult:
        if sum(self.task_quotas.values()) != budget:
            raise ValueError("task quotas must sum to the selection budget")
        state = self.context.empty_state()
        selected: list[int] = []
        gains: list[float] = []
        for index in initial_indices or []:
            if not self._eligible(state, index):
                raise ValueError("initial seed violates task quotas")
            gain = self.context.marginal_gain(state, index)
            self.context.add_candidate(state, index)
            selected.append(index)
            gains.append(gain)
        stale = np.asarray(
            [
                self.context.marginal_gain(self.context.empty_state(), index)
                for index in range(len(self.context.graph.sample_ids))
            ],
            dtype=np.float64,
        )
        versions = np.zeros(len(stale), dtype=np.int64)
        stale_heap = [
            (-float(score), self.context.graph.sample_ids[index], index, 0)
            for index, score in enumerate(stale)
        ]
        heapq.heapify(stale_heap)
        rng = np.random.default_rng(self.seed)
        while len(selected) < budget:
            open_tasks = state.task_counts < self.quota_limits
            eligible = np.flatnonzero(~state.selected_mask & open_tasks[self.node_task_positions])
            if not len(eligible):
                raise ValueError("no quota-feasible candidate remains")
            candidate_pool: set[int] = set()
            popped: list[tuple[float, str, int, int]] = []
            while stale_heap and len(candidate_pool) < self.global_candidates:
                entry = heapq.heappop(stale_heap)
                _, _, index, version = entry
                if version != int(versions[index]) or not self._eligible(state, index):
                    continue
                candidate_pool.add(index)
                popped.append(entry)
            for entry in popped:
                heapq.heappush(stale_heap, entry)
            if selected:
                candidate_pool.update(
                    index for index in self.neighbors[selected[-1]] if self._eligible(state, index)
                )
            residual_order = np.argsort(state.prototype_coverage, kind="stable")
            residual_nodes: list[int] = []
            for prototype in residual_order:
                match = next(
                    (
                        index
                        for index in self.prototype_nodes[int(prototype)]
                        if self._eligible(state, index)
                    ),
                    None,
                )
                if match is not None:
                    residual_nodes.append(match)
                if len(residual_nodes) >= self.residual_candidates:
                    break
            candidate_pool.update(residual_nodes)
            sample_size = min(self.random_candidates, len(eligible))
            if sample_size:
                weights = self.context.graph.reliability[eligible].astype(np.float64)
                weights /= weights.sum()
                candidate_pool.update(
                    int(index)
                    for index in rng.choice(eligible, size=sample_size, replace=False, p=weights)
                )
            for task, quota in self.task_quotas.items():
                position = self.context.task_position[task]
                if state.task_counts[position] >= quota:
                    continue
                task_node = next(
                    (index for index in self.task_nodes[task] if self._eligible(state, index)),
                    None,
                )
                if task_node is not None:
                    candidate_pool.add(task_node)
            evaluated = sorted(
                (index for index in candidate_pool if self._eligible(state, index)),
                key=lambda index: self.context.graph.sample_ids[index],
            )
            if not evaluated:
                evaluated = sorted(
                    eligible.tolist(),
                    key=lambda index: self.context.graph.sample_ids[index],
                )
            best_index = evaluated[0]
            best_gain = self.context.marginal_gain(state, best_index)
            stale[best_index] = best_gain
            versions[best_index] += 1
            heapq.heappush(
                stale_heap,
                (
                    -float(best_gain),
                    self.context.graph.sample_ids[best_index],
                    best_index,
                    int(versions[best_index]),
                ),
            )
            for index in evaluated[1:]:
                gain = self.context.marginal_gain(state, index)
                stale[index] = gain
                versions[index] += 1
                heapq.heappush(
                    stale_heap,
                    (
                        -float(gain),
                        self.context.graph.sample_ids[index],
                        index,
                        int(versions[index]),
                    ),
                )
                if gain > best_gain:
                    best_index, best_gain = index, gain
            self.context.add_candidate(state, best_index)
            selected.append(best_index)
            gains.append(float(best_gain))
            if len(stale_heap) > 4 * len(stale):
                stale_heap = [
                    (
                        -float(stale[index]),
                        self.context.graph.sample_ids[index],
                        index,
                        int(versions[index]),
                    )
                    for index in range(len(stale))
                    if not state.selected_mask[index]
                ]
                heapq.heapify(stale_heap)
        return SelectionResult(selected, gains, float(state.objective_value), branch_id)
