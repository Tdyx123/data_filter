"""Incremental co-occurrence-minus-redundancy objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from relcore.graph.prototypes import valid_prototype_assignments
from relcore.schemas import GraphData


@dataclass
class CocoreObjectiveState:
    selected_mask: np.ndarray
    prototype_mass: np.ndarray
    task_counts: np.ndarray
    cooccurrence: float = 0.0
    redundancy: float = 0.0
    score: float = 0.0


class CocoreObjectiveContext:
    def __init__(
        self,
        graph: GraphData,
        cooccurrence_weight: float = 1.0,
        *,
        similarity_threshold: float,
        epsilon: float = 1.0e-8,
    ) -> None:
        weight = float(cooccurrence_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("cooccurrence_weight must be finite and non-negative")
        threshold = float(similarity_threshold)
        if not 0.0 <= threshold < 1.0:
            raise ValueError("similarity_threshold must be in [0, 1)")
        self.graph = graph
        self.cooccurrence_weight = weight
        self.similarity_threshold = threshold
        self.epsilon = float(epsilon)
        self.cooccurrence_matrix = graph.cooccurrence_matrix.toarray().astype(np.float64)
        prototype_count = int(self.cooccurrence_matrix.shape[0])
        if self.cooccurrence_matrix.shape != (prototype_count, prototype_count):
            raise ValueError("cooccurrence matrix must be square")
        self.prototype_mass = np.zeros(
            (len(graph.sample_ids), prototype_count), dtype=np.float64
        )
        for node, (indices, assignments) in enumerate(
            zip(graph.prototype_indices, graph.prototype_weights, strict=True)
        ):
            valid_indices, valid_assignments = valid_prototype_assignments(indices, assignments)
            self.prototype_mass[node, valid_indices] = (
                float(graph.reliability[node]) * valid_assignments.astype(np.float64)
            )
        self.task_labels = sorted({int(task) for task in graph.task_indices})
        self.task_position = {task: position for position, task in enumerate(self.task_labels)}
        self.redundancy_adjacency: list[list[tuple[int, float]]] = [
            [] for _ in graph.sample_ids
        ]
        raw_total = 0.0
        denominator = max(1.0 - threshold, self.epsilon)
        for source, target, similarity in zip(
            graph.similarity_edges.source,
            graph.similarity_edges.target,
            graph.similarity_edges.weight,
            strict=True,
        ):
            left, right = int(source), int(target)
            penalty = (
                float(graph.reliability[left])
                * float(graph.reliability[right])
                * max(0.0, float(similarity) - threshold)
                / denominator
            )
            if penalty <= 0.0:
                continue
            self.redundancy_adjacency[left].append((right, penalty))
            self.redundancy_adjacency[right].append((left, penalty))
            raw_total += penalty
        self.redundancy_normalizer = max(raw_total, self.epsilon)

    def empty_state(self) -> CocoreObjectiveState:
        return CocoreObjectiveState(
            selected_mask=np.zeros(len(self.graph.sample_ids), dtype=bool),
            prototype_mass=np.zeros(self.prototype_mass.shape[1], dtype=np.float64),
            task_counts=np.zeros(len(self.task_labels), dtype=np.int64),
        )

    @staticmethod
    def clone_state(state: CocoreObjectiveState) -> CocoreObjectiveState:
        return CocoreObjectiveState(
            selected_mask=state.selected_mask.copy(),
            prototype_mass=state.prototype_mass.copy(),
            task_counts=state.task_counts.copy(),
            cooccurrence=float(state.cooccurrence),
            redundancy=float(state.redundancy),
            score=float(state.score),
        )

    def _candidate_values(
        self, state: CocoreObjectiveState, candidate: int
    ) -> tuple[float, float, float]:
        if state.selected_mask[candidate]:
            raise ValueError("candidate is already selected")
        old_mass = state.prototype_mass
        added_mass = self.prototype_mass[candidate]
        cooccurrence_delta = float(
            added_mass @ self.cooccurrence_matrix @ old_mass
            + old_mass @ self.cooccurrence_matrix @ added_mass
            + added_mass @ self.cooccurrence_matrix @ added_mass
        )
        redundancy_delta = (
            sum(
                penalty
                for other, penalty in self.redundancy_adjacency[candidate]
                if state.selected_mask[other]
            )
            / self.redundancy_normalizer
        )
        cooccurrence = state.cooccurrence + cooccurrence_delta
        redundancy = state.redundancy + redundancy_delta
        score = self.cooccurrence_weight * cooccurrence - redundancy
        return cooccurrence, redundancy, float(score)

    def marginal_gain(self, state: CocoreObjectiveState, candidate: int) -> float:
        if state.selected_mask[candidate]:
            return float("-inf")
        return self._candidate_values(state, candidate)[2] - state.score

    def add_candidate(self, state: CocoreObjectiveState, candidate: int) -> None:
        cooccurrence, redundancy, score = self._candidate_values(state, candidate)
        state.prototype_mass += self.prototype_mass[candidate]
        task = int(self.graph.task_indices[candidate])
        state.task_counts[self.task_position[task]] += 1
        state.selected_mask[candidate] = True
        state.cooccurrence = cooccurrence
        state.redundancy = redundancy
        state.score = score

    def state_from_indices(self, selected_indices: list[int] | tuple[int, ...]) -> CocoreObjectiveState:
        state = self.empty_state()
        for index in selected_indices:
            self.add_candidate(state, int(index))
        return state


def recompute_objective(
    selected_indices: list[int] | tuple[int, ...],
    graph: GraphData,
    *,
    cooccurrence_weight: float = 1.0,
    similarity_threshold: float,
) -> CocoreObjectiveState:
    context = CocoreObjectiveContext(
        graph,
        cooccurrence_weight,
        similarity_threshold=similarity_threshold,
    )
    return context.state_from_indices(selected_indices)
