"""Full and incrementally maintained relational set objective."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from relcore.graph.prototypes import valid_prototype_assignments
from relcore.schemas import GraphData, ObjectiveState
from relcore.selection.prototype_gain import (
    PROTOTYPE_GAIN_METRICS,
    normalize_prototype_gain_metrics,
)
from relcore.selection.relation_metrics import RelationMetricKernel


@dataclass(frozen=True)
class ObjectiveWeights:
    node: float = 4.0
    transition: float = 1.2
    cooccurrence: float = 0.5
    sequence: float = 2.0
    redundancy: float = 2.0


@dataclass(frozen=True)
class ObjectiveBreakdown:
    node: float
    transition: float
    cooccurrence: float
    sequence: float
    redundancy: float
    total: float


class ObjectiveContext:
    def __init__(
        self,
        graph: GraphData,
        weights: ObjectiveWeights,
        *,
        similarity_threshold: float,
        prototype_gain_metrics: Sequence[str] = PROTOTYPE_GAIN_METRICS,
        epsilon: float = 1.0e-8,
    ):
        self.graph = graph
        self.weights = weights
        self.prototype_gain_metrics = normalize_prototype_gain_metrics(
            prototype_gain_metrics
        )
        self._enabled_prototype_gains = frozenset(self.prototype_gain_metrics)
        self.similarity_threshold = float(similarity_threshold)
        self.epsilon = float(epsilon)
        self.relation_metrics = RelationMetricKernel(graph, epsilon=self.epsilon)
        self.prototype_count = self.relation_metrics.prototype_count
        self.transition = self.relation_metrics.transition
        self.cooccurrence = self.relation_metrics.cooccurrence
        frequency = np.zeros(self.prototype_count, dtype=np.float64)
        for indices, values in zip(graph.prototype_indices, graph.prototype_weights, strict=True):
            valid_indices, valid_values = valid_prototype_assignments(indices, values)
            np.add.at(frequency, valid_indices, valid_values)
        inverse = np.zeros(self.prototype_count, dtype=np.float64)
        reachable = frequency > 0.0
        inverse[reachable] = 1.0 / np.sqrt(frequency[reachable] + self.epsilon)
        if not np.any(reachable):
            raise ValueError("graph contains no reachable prototypes")
        self.prototype_weights = inverse / inverse.sum()
        self.prototype_members = self.relation_metrics.prototype_members
        self.task_labels = sorted({int(task) for task in graph.task_indices})
        self.task_position = {task: position for position, task in enumerate(self.task_labels)}
        self.sequence_adjacency = self.relation_metrics.sequence_adjacency
        self.sequence_denominator = self.relation_metrics.sequence_denominator

        self.redundancy_adjacency: list[list[tuple[int, float]]] = [[] for _ in graph.sample_ids]
        total_redundancy = 0.0
        denominator = max(1.0 - self.similarity_threshold, self.epsilon)
        for source, target, similarity in zip(
            graph.similarity_edges.source,
            graph.similarity_edges.target,
            graph.similarity_edges.weight,
            strict=True,
        ):
            source_index, target_index = int(source), int(target)
            penalty = (
                float(graph.reliability[source_index])
                * float(graph.reliability[target_index])
                * max(0.0, float(similarity) - self.similarity_threshold)
                / denominator
            )
            self.redundancy_adjacency[source_index].append((target_index, penalty))
            self.redundancy_adjacency[target_index].append((source_index, penalty))
            total_redundancy += penalty
        self.redundancy_normalizer = max(total_redundancy, self.epsilon)

    def empty_state(self) -> ObjectiveState:
        return ObjectiveState(
            selected_mask=np.zeros(len(self.graph.sample_ids), dtype=bool),
            task_counts=np.zeros(len(self.task_labels), dtype=np.int64),
            prototype_coverage=self.relation_metrics.empty_prototype_coverage(),
            sequence_relation_counts=self.relation_metrics.empty_sequence_counts(),
            redundancy_sum=0.0,
            objective_value=0.0,
        )

    @staticmethod
    def clone_state(state: ObjectiveState) -> ObjectiveState:
        return ObjectiveState(
            state.selected_mask.copy(),
            state.task_counts.copy(),
            state.prototype_coverage.copy(),
            state.sequence_relation_counts.copy(),
            float(state.redundancy_sum),
            float(state.objective_value),
        )

    def _candidate_gain(self, state: ObjectiveState, candidate: int) -> float:
        relation_delta = self.relation_metrics.candidate_delta(
            state.selected_mask,
            state.prototype_coverage,
            state.sequence_relation_counts,
            candidate,
        )
        new_coverage = relation_delta.prototype_coverage
        affected = relation_delta.affected_prototypes
        node_delta = float(
            self.prototype_weights[affected]
            @ (new_coverage[affected] - state.prototype_coverage[affected])
        )

        redundancy_delta = (
            sum(
                penalty
                for other, penalty in self.redundancy_adjacency[candidate]
                if state.selected_mask[other]
            )
            / self.redundancy_normalizer
        )
        return float(
            self.weights.node * node_delta
            + (
                self.weights.transition * relation_delta.transition
                if "transition" in self._enabled_prototype_gains
                else 0.0
            )
            + (
                self.weights.cooccurrence * relation_delta.cooccurrence
                if "cooccurrence" in self._enabled_prototype_gains
                else 0.0
            )
            + (
                self.weights.sequence * relation_delta.sequence
                if "sequence" in self._enabled_prototype_gains
                else 0.0
            )
            - self.weights.redundancy * redundancy_delta
        )

    def breakdown(self, state: ObjectiveState) -> ObjectiveBreakdown:
        coverage = state.prototype_coverage
        relation_values = self.relation_metrics.values(
            coverage,
            state.sequence_relation_counts,
        )
        node = float(self.prototype_weights @ coverage)
        transition = relation_values.transition
        cooccurrence = relation_values.cooccurrence
        sequence = relation_values.sequence
        redundancy = float(state.redundancy_sum / self.redundancy_normalizer)
        total = (
            self.weights.node * node
            + (
                self.weights.transition * transition
                if "transition" in self._enabled_prototype_gains
                else 0.0
            )
            + (
                self.weights.cooccurrence * cooccurrence
                if "cooccurrence" in self._enabled_prototype_gains
                else 0.0
            )
            + (
                self.weights.sequence * sequence
                if "sequence" in self._enabled_prototype_gains
                else 0.0
            )
            - self.weights.redundancy * redundancy
        )
        return ObjectiveBreakdown(
            node, transition, cooccurrence, sequence, redundancy, float(total)
        )

    def add_candidate(self, state: ObjectiveState, candidate: int) -> None:
        if state.selected_mask[candidate]:
            raise ValueError("candidate is already selected")
        gain = self._candidate_gain(state, candidate)
        state.prototype_coverage[:] = np.maximum(
            state.prototype_coverage,
            self.relation_metrics.prototype_mass[candidate],
        )
        self.relation_metrics.update_sequence_counts(
            state.sequence_relation_counts,
            state.selected_mask,
            candidate,
            1.0,
        )
        for other, penalty in self.redundancy_adjacency[candidate]:
            if state.selected_mask[other]:
                state.redundancy_sum += penalty
        task = int(self.graph.task_indices[candidate])
        state.task_counts[self.task_position[task]] += 1
        state.selected_mask[candidate] = True
        state.objective_value += gain

    def marginal_gain(self, state: ObjectiveState, candidate: int) -> float:
        if state.selected_mask[candidate]:
            return float("-inf")
        return self._candidate_gain(state, candidate)

    def remove_candidate(self, state: ObjectiveState, candidate: int) -> None:
        if not state.selected_mask[candidate]:
            raise ValueError("candidate is not selected")
        self.relation_metrics.update_sequence_counts(
            state.sequence_relation_counts,
            state.selected_mask,
            candidate,
            -1.0,
        )
        for other, penalty in self.redundancy_adjacency[candidate]:
            if state.selected_mask[other]:
                state.redundancy_sum -= penalty
        state.redundancy_sum = max(state.redundancy_sum, 0.0)
        task = int(self.graph.task_indices[candidate])
        state.task_counts[self.task_position[task]] -= 1
        state.selected_mask[candidate] = False
        candidate_indices, _ = valid_prototype_assignments(
            self.graph.prototype_indices[candidate], self.graph.prototype_weights[candidate]
        )
        for prototype in candidate_indices:
            state.prototype_coverage[prototype] = max(
                (
                    float(self.graph.reliability[node]) * assignment
                    for node, assignment in self.prototype_members[int(prototype)]
                    if state.selected_mask[node]
                ),
                default=0.0,
            )
        state.objective_value = self.breakdown(state).total

    def state_from_indices(self, selected_indices: list[int]) -> ObjectiveState:
        state = self.empty_state()
        for index in selected_indices:
            self.add_candidate(state, int(index))
        return state


def recompute_objective(
    selected_indices: list[int],
    graph: GraphData,
    weights: ObjectiveWeights,
    *,
    similarity_threshold: float,
    prototype_gain_metrics: Sequence[str] = PROTOTYPE_GAIN_METRICS,
) -> ObjectiveBreakdown:
    context = ObjectiveContext(
        graph,
        weights,
        similarity_threshold=similarity_threshold,
        prototype_gain_metrics=prototype_gain_metrics,
    )
    selected = np.zeros(len(graph.sample_ids), dtype=bool)
    selected[np.asarray(selected_indices, dtype=np.int64)] = True
    coverage = context.relation_metrics.coverage_from_mask(selected)
    sequence_counts = context.relation_metrics.sequence_counts_from_mask(selected)
    redundancy_sum = 0.0
    denominator = max(1.0 - similarity_threshold, context.epsilon)
    for source, target, similarity in zip(
        graph.similarity_edges.source,
        graph.similarity_edges.target,
        graph.similarity_edges.weight,
        strict=True,
    ):
        source_index, target_index = int(source), int(target)
        if selected[source_index] and selected[target_index]:
            redundancy_sum += (
                float(graph.reliability[source_index])
                * float(graph.reliability[target_index])
                * max(0.0, float(similarity) - similarity_threshold)
                / denominator
            )
    task_counts = np.zeros(len(context.task_labels), dtype=np.int64)
    for index in np.flatnonzero(selected):
        task_counts[context.task_position[int(graph.task_indices[index])]] += 1
    state = ObjectiveState(
        selected_mask=selected,
        task_counts=task_counts,
        prototype_coverage=coverage,
        sequence_relation_counts=sequence_counts,
        redundancy_sum=redundancy_sum,
        objective_value=0.0,
    )
    return context.breakdown(state)
