"""Incremental relation-minus-redundancy objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from relcore.schemas import GraphData
from relcore.selection.relation_metrics import RelationMetricDelta, RelationMetricKernel


@dataclass
class CocoreObjectiveState:
    selected_mask: np.ndarray
    prototype_coverage: np.ndarray
    sequence_relation_counts: np.ndarray
    task_counts: np.ndarray
    relation: float = 0.0
    redundancy: float = 0.0
    score: float = 0.0


class CocoreObjectiveContext:
    def __init__(
        self,
        graph: GraphData,
        relation_type: str,
        relation_weight: float = 1.0,
        *,
        similarity_threshold: float,
        epsilon: float = 1.0e-8,
    ) -> None:
        relation = str(relation_type)
        if relation not in {"sequence", "cooccurrence"}:
            raise ValueError("relation_type must be sequence or cooccurrence")
        weight = float(relation_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("relation_weight must be finite and non-negative")
        threshold = float(similarity_threshold)
        if not 0.0 <= threshold < 1.0:
            raise ValueError("similarity_threshold must be in [0, 1)")
        self.graph = graph
        self.relation_type = relation
        self.relation_weight = weight
        self.similarity_threshold = threshold
        self.epsilon = float(epsilon)
        self.relation_metrics = RelationMetricKernel(
            graph,
            epsilon=self.epsilon,
        )
        self.prototype_mass = self.relation_metrics.prototype_mass
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
            prototype_coverage=self.relation_metrics.empty_prototype_coverage(),
            sequence_relation_counts=self.relation_metrics.empty_sequence_counts(),
            task_counts=np.zeros(len(self.task_labels), dtype=np.int64),
        )

    @staticmethod
    def clone_state(state: CocoreObjectiveState) -> CocoreObjectiveState:
        return CocoreObjectiveState(
            selected_mask=state.selected_mask.copy(),
            prototype_coverage=state.prototype_coverage.copy(),
            sequence_relation_counts=state.sequence_relation_counts.copy(),
            task_counts=state.task_counts.copy(),
            relation=float(state.relation),
            redundancy=float(state.redundancy),
            score=float(state.score),
        )

    def _candidate_values(
        self, state: CocoreObjectiveState, candidate: int
    ) -> tuple[float, float, float, RelationMetricDelta]:
        if state.selected_mask[candidate]:
            raise ValueError("candidate is already selected")
        metric_delta = self.relation_metrics.candidate_delta(
            state.selected_mask,
            state.prototype_coverage,
            state.sequence_relation_counts,
            candidate,
        )
        relation_delta = float(getattr(metric_delta, self.relation_type))
        redundancy_delta = (
            sum(
                penalty
                for other, penalty in self.redundancy_adjacency[candidate]
                if state.selected_mask[other]
            )
            / self.redundancy_normalizer
        )
        relation = state.relation + relation_delta
        redundancy = state.redundancy + redundancy_delta
        score = self.relation_weight * relation - redundancy
        return relation, redundancy, float(score), metric_delta

    def marginal_gain(self, state: CocoreObjectiveState, candidate: int) -> float:
        if state.selected_mask[candidate]:
            return float("-inf")
        return self._candidate_values(state, candidate)[2] - state.score

    def add_candidate(self, state: CocoreObjectiveState, candidate: int) -> None:
        relation, redundancy, score, metric_delta = self._candidate_values(state, candidate)
        state.prototype_coverage[:] = metric_delta.prototype_coverage
        self.relation_metrics.update_sequence_counts(
            state.sequence_relation_counts,
            state.selected_mask,
            candidate,
            1.0,
        )
        task = int(self.graph.task_indices[candidate])
        state.task_counts[self.task_position[task]] += 1
        state.selected_mask[candidate] = True
        state.relation = relation
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
    relation_type: str,
    relation_weight: float = 1.0,
    similarity_threshold: float,
) -> CocoreObjectiveState:
    context = CocoreObjectiveContext(
        graph,
        relation_type,
        relation_weight,
        similarity_threshold=similarity_threshold,
    )
    return context.state_from_indices(selected_indices)
