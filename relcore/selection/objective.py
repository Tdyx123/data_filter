"""Full and incrementally maintained relational set objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relcore.graph.prototypes import valid_prototype_assignments
from relcore.schemas import GraphData, ObjectiveState


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
        epsilon: float = 1.0e-8,
    ):
        self.graph = graph
        self.weights = weights
        self.similarity_threshold = float(similarity_threshold)
        self.epsilon = float(epsilon)
        self.prototype_count = int(graph.transition_matrix.shape[0])
        self.transition = graph.transition_matrix.toarray().astype(np.float64)
        self.cooccurrence = graph.cooccurrence_matrix.toarray().astype(np.float64)
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
        self.prototype_members: list[list[tuple[int, float]]] = [
            [] for _ in range(self.prototype_count)
        ]
        for node, (indices, assignments) in enumerate(
            zip(graph.prototype_indices, graph.prototype_weights, strict=True)
        ):
            valid_indices, valid_assignments = valid_prototype_assignments(indices, assignments)
            for prototype, assignment in zip(valid_indices, valid_assignments, strict=True):
                self.prototype_members[int(prototype)].append((node, float(assignment)))
        self.task_labels = sorted({int(task) for task in graph.task_indices})
        self.task_position = {task: position for position, task in enumerate(self.task_labels)}
        self.sequence_adjacency: list[list[tuple[int, int]]] = [[] for _ in graph.sample_ids]
        for source, target in zip(
            graph.sequence_edges.source, graph.sequence_edges.target, strict=True
        ):
            source_index, target_index = int(source), int(target)
            self.sequence_adjacency[source_index].append((source_index, target_index))
            self.sequence_adjacency[target_index].append((source_index, target_index))
        full_sequence = np.zeros((self.prototype_count, self.prototype_count), dtype=np.float64)
        for source, target in zip(
            graph.sequence_edges.source, graph.sequence_edges.target, strict=True
        ):
            self._add_sequence_relation(full_sequence, int(source), int(target), 1.0)
        self.sequence_denominator = np.log1p(full_sequence) + self.epsilon

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

    def _add_sequence_relation(
        self,
        counts: np.ndarray,
        source: int,
        target: int,
        sign: float,
    ) -> None:
        node_weight = (
            float(self.graph.reliability[source]) * float(self.graph.reliability[target]) * sign
        )
        source_indices, source_weights = valid_prototype_assignments(
            self.graph.prototype_indices[source], self.graph.prototype_weights[source]
        )
        target_indices, target_weights = valid_prototype_assignments(
            self.graph.prototype_indices[target], self.graph.prototype_weights[target]
        )
        for source_proto, source_weight in zip(source_indices, source_weights, strict=True):
            for target_proto, target_weight in zip(target_indices, target_weights, strict=True):
                counts[source_proto, target_proto] += (
                    node_weight * float(source_weight) * float(target_weight)
                )

    def empty_state(self) -> ObjectiveState:
        return ObjectiveState(
            selected_mask=np.zeros(len(self.graph.sample_ids), dtype=bool),
            task_counts=np.zeros(len(self.task_labels), dtype=np.int64),
            prototype_coverage=np.zeros(self.prototype_count, dtype=np.float64),
            sequence_relation_counts=np.zeros(
                (self.prototype_count, self.prototype_count), dtype=np.float64
            ),
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

    @staticmethod
    def _pair_coverage_delta(
        matrix: np.ndarray,
        old_coverage: np.ndarray,
        new_coverage: np.ndarray,
        affected: np.ndarray,
    ) -> float:
        if not len(affected):
            return 0.0
        delta = 0.0
        unaffected_mask = np.ones(len(old_coverage), dtype=bool)
        unaffected_mask[affected] = False
        for prototype in affected:
            old_row = np.sqrt(np.maximum(old_coverage[prototype] * old_coverage, 0.0))
            new_row = np.sqrt(np.maximum(new_coverage[prototype] * new_coverage, 0.0))
            delta += float(matrix[prototype] @ (new_row - old_row))
            if np.any(unaffected_mask):
                old_column = np.sqrt(
                    np.maximum(
                        old_coverage[unaffected_mask] * old_coverage[prototype],
                        0.0,
                    )
                )
                new_column = np.sqrt(
                    np.maximum(
                        new_coverage[unaffected_mask] * new_coverage[prototype],
                        0.0,
                    )
                )
                delta += float(matrix[unaffected_mask, prototype] @ (new_column - old_column))
        return delta

    def _candidate_gain(self, state: ObjectiveState, candidate: int) -> float:
        new_coverage = state.prototype_coverage.copy()
        candidate_indices, candidate_weights = valid_prototype_assignments(
            self.graph.prototype_indices[candidate], self.graph.prototype_weights[candidate]
        )
        for prototype, assignment in zip(candidate_indices, candidate_weights, strict=True):
            new_coverage[prototype] = max(
                new_coverage[prototype],
                float(self.graph.reliability[candidate]) * float(assignment),
            )
        affected = np.flatnonzero(new_coverage != state.prototype_coverage)
        node_delta = float(
            self.prototype_weights[affected]
            @ (new_coverage[affected] - state.prototype_coverage[affected])
        )
        transition_delta = self._pair_coverage_delta(
            self.transition,
            state.prototype_coverage,
            new_coverage,
            affected,
        )
        cooccurrence_delta = self._pair_coverage_delta(
            self.cooccurrence,
            state.prototype_coverage,
            new_coverage,
            affected,
        )

        relation_delta: dict[tuple[int, int], float] = {}
        for source, target in self.sequence_adjacency[candidate]:
            other = target if source == candidate else source
            if not state.selected_mask[other]:
                continue
            edge_quality = float(self.graph.reliability[source]) * float(
                self.graph.reliability[target]
            )
            source_indices, source_weights = valid_prototype_assignments(
                self.graph.prototype_indices[source], self.graph.prototype_weights[source]
            )
            target_indices, target_weights = valid_prototype_assignments(
                self.graph.prototype_indices[target], self.graph.prototype_weights[target]
            )
            for source_proto, source_weight in zip(source_indices, source_weights, strict=True):
                for target_proto, target_weight in zip(target_indices, target_weights, strict=True):
                    key = (int(source_proto), int(target_proto))
                    relation_delta[key] = relation_delta.get(key, 0.0) + (
                        edge_quality * float(source_weight) * float(target_weight)
                    )
        sequence_delta = 0.0
        for (source_proto, target_proto), increment in relation_delta.items():
            old_count = state.sequence_relation_counts[source_proto, target_proto]
            sequence_delta += float(
                self.transition[source_proto, target_proto]
                * (np.log1p(old_count + increment) - np.log1p(old_count))
                / self.sequence_denominator[source_proto, target_proto]
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
            + self.weights.transition * transition_delta
            + self.weights.cooccurrence * cooccurrence_delta
            + self.weights.sequence * sequence_delta
            - self.weights.redundancy * redundancy_delta
        )

    def breakdown(self, state: ObjectiveState) -> ObjectiveBreakdown:
        coverage = state.prototype_coverage
        node = float(self.prototype_weights @ coverage)
        pair_coverage = np.sqrt(np.maximum(coverage[:, None] * coverage[None, :], 0.0))
        transition = float(np.sum(self.transition * pair_coverage))
        cooccurrence = float(np.sum(self.cooccurrence * pair_coverage))
        sequence = float(
            np.sum(
                self.transition
                * np.log1p(np.maximum(state.sequence_relation_counts, 0.0))
                / self.sequence_denominator
            )
        )
        redundancy = float(state.redundancy_sum / self.redundancy_normalizer)
        total = (
            self.weights.node * node
            + self.weights.transition * transition
            + self.weights.cooccurrence * cooccurrence
            + self.weights.sequence * sequence
            - self.weights.redundancy * redundancy
        )
        return ObjectiveBreakdown(
            node, transition, cooccurrence, sequence, redundancy, float(total)
        )

    def add_candidate(self, state: ObjectiveState, candidate: int) -> None:
        if state.selected_mask[candidate]:
            raise ValueError("candidate is already selected")
        gain = self._candidate_gain(state, candidate)
        candidate_indices, candidate_weights = valid_prototype_assignments(
            self.graph.prototype_indices[candidate], self.graph.prototype_weights[candidate]
        )
        for prototype, assignment in zip(candidate_indices, candidate_weights, strict=True):
            state.prototype_coverage[prototype] = max(
                state.prototype_coverage[prototype],
                float(self.graph.reliability[candidate]) * float(assignment),
            )
        for source, target in self.sequence_adjacency[candidate]:
            other = target if source == candidate else source
            if state.selected_mask[other]:
                self._add_sequence_relation(state.sequence_relation_counts, source, target, 1.0)
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
        for source, target in self.sequence_adjacency[candidate]:
            other = target if source == candidate else source
            if state.selected_mask[other]:
                self._add_sequence_relation(state.sequence_relation_counts, source, target, -1.0)
        state.sequence_relation_counts[:] = np.maximum(state.sequence_relation_counts, 0.0)
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
) -> ObjectiveBreakdown:
    context = ObjectiveContext(graph, weights, similarity_threshold=similarity_threshold)
    selected = np.zeros(len(graph.sample_ids), dtype=bool)
    selected[np.asarray(selected_indices, dtype=np.int64)] = True
    coverage = np.zeros(context.prototype_count, dtype=np.float64)
    for index in np.flatnonzero(selected):
        candidate_indices, candidate_weights = valid_prototype_assignments(
            graph.prototype_indices[index], graph.prototype_weights[index]
        )
        for prototype, assignment in zip(candidate_indices, candidate_weights, strict=True):
            coverage[prototype] = max(
                coverage[prototype],
                float(graph.reliability[index]) * float(assignment),
            )
    sequence_counts = np.zeros((context.prototype_count, context.prototype_count), dtype=np.float64)
    for source, target in zip(
        graph.sequence_edges.source, graph.sequence_edges.target, strict=True
    ):
        if selected[int(source)] and selected[int(target)]:
            context._add_sequence_relation(sequence_counts, int(source), int(target), 1.0)
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
