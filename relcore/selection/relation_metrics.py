"""Shared incremental prototype-relation metric calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relcore.graph.prototypes import valid_prototype_assignments
from relcore.schemas import GraphData


@dataclass(frozen=True)
class RelationMetricDelta:
    prototype_coverage: np.ndarray
    affected_prototypes: np.ndarray
    transition: float
    cooccurrence: float
    sequence: float


@dataclass(frozen=True)
class RelationMetricValues:
    transition: float
    cooccurrence: float
    sequence: float


class RelationMetricKernel:
    """Single source of truth for RelCore prototype relation metrics."""

    def __init__(self, graph: GraphData, *, epsilon: float = 1.0e-8) -> None:
        self.graph = graph
        self.epsilon = float(epsilon)
        self.prototype_count = int(graph.transition_matrix.shape[0])
        self.transition = graph.transition_matrix.toarray().astype(np.float64)
        self.cooccurrence = graph.cooccurrence_matrix.toarray().astype(np.float64)
        expected_shape = (self.prototype_count, self.prototype_count)
        if self.transition.shape != expected_shape or self.cooccurrence.shape != expected_shape:
            raise ValueError("prototype relation matrices must be square and have equal shapes")

        self.prototype_mass = np.zeros(
            (len(graph.sample_ids), self.prototype_count), dtype=np.float64
        )
        self.prototype_members: list[list[tuple[int, float]]] = [
            [] for _ in range(self.prototype_count)
        ]
        for node, (indices, assignments) in enumerate(
            zip(graph.prototype_indices, graph.prototype_weights, strict=True)
        ):
            valid_indices, valid_assignments = valid_prototype_assignments(indices, assignments)
            reliable_assignments = (
                float(graph.reliability[node]) * valid_assignments.astype(np.float64)
            )
            self.prototype_mass[node, valid_indices] = reliable_assignments
            for prototype, assignment in zip(
                valid_indices, valid_assignments, strict=True
            ):
                self.prototype_members[int(prototype)].append((node, float(assignment)))

        self.sequence_adjacency: list[list[tuple[int, int]]] = [
            [] for _ in graph.sample_ids
        ]
        full_sequence = self.empty_sequence_counts()
        for source, target in zip(
            graph.sequence_edges.source, graph.sequence_edges.target, strict=True
        ):
            source_index, target_index = int(source), int(target)
            self.sequence_adjacency[source_index].append((source_index, target_index))
            self.sequence_adjacency[target_index].append((source_index, target_index))
            full_sequence += self._edge_relation(source_index, target_index)
        self.sequence_denominator = np.log1p(full_sequence) + self.epsilon

    def empty_prototype_coverage(self) -> np.ndarray:
        return np.zeros(self.prototype_count, dtype=np.float64)

    def empty_sequence_counts(self) -> np.ndarray:
        return np.zeros((self.prototype_count, self.prototype_count), dtype=np.float64)

    def _edge_relation(self, source: int, target: int) -> np.ndarray:
        return np.outer(self.prototype_mass[source], self.prototype_mass[target])

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
                delta += float(
                    matrix[unaffected_mask, prototype] @ (new_column - old_column)
                )
        return delta

    def _sequence_delta(
        self,
        selected_mask: np.ndarray,
        sequence_counts: np.ndarray,
        candidate: int,
    ) -> float:
        increment = self.empty_sequence_counts()
        for source, target in self.sequence_adjacency[candidate]:
            other = target if source == candidate else source
            if selected_mask[other]:
                increment += self._edge_relation(source, target)
        return float(
            np.sum(
                self.transition
                * (np.log1p(sequence_counts + increment) - np.log1p(sequence_counts))
                / self.sequence_denominator
            )
        )

    def candidate_delta(
        self,
        selected_mask: np.ndarray,
        prototype_coverage: np.ndarray,
        sequence_counts: np.ndarray,
        candidate: int,
    ) -> RelationMetricDelta:
        new_coverage = np.maximum(
            prototype_coverage,
            self.prototype_mass[int(candidate)],
        )
        affected = np.flatnonzero(new_coverage != prototype_coverage)
        return RelationMetricDelta(
            prototype_coverage=new_coverage,
            affected_prototypes=affected,
            transition=self._pair_coverage_delta(
                self.transition,
                prototype_coverage,
                new_coverage,
                affected,
            ),
            cooccurrence=self._pair_coverage_delta(
                self.cooccurrence,
                prototype_coverage,
                new_coverage,
                affected,
            ),
            sequence=self._sequence_delta(
                selected_mask,
                sequence_counts,
                int(candidate),
            ),
        )

    def update_sequence_counts(
        self,
        sequence_counts: np.ndarray,
        selected_mask: np.ndarray,
        candidate: int,
        sign: float,
    ) -> None:
        for source, target in self.sequence_adjacency[int(candidate)]:
            other = target if source == candidate else source
            if selected_mask[other]:
                sequence_counts += float(sign) * self._edge_relation(source, target)
        if sign < 0.0:
            sequence_counts[:] = np.maximum(sequence_counts, 0.0)

    def coverage_from_mask(self, selected_mask: np.ndarray) -> np.ndarray:
        selected = np.flatnonzero(selected_mask)
        if not len(selected):
            return self.empty_prototype_coverage()
        return self.prototype_mass[selected].max(axis=0)

    def sequence_counts_from_mask(self, selected_mask: np.ndarray) -> np.ndarray:
        counts = self.empty_sequence_counts()
        for source, target in zip(
            self.graph.sequence_edges.source,
            self.graph.sequence_edges.target,
            strict=True,
        ):
            source_index, target_index = int(source), int(target)
            if selected_mask[source_index] and selected_mask[target_index]:
                counts += self._edge_relation(source_index, target_index)
        return counts

    def values(
        self,
        prototype_coverage: np.ndarray,
        sequence_counts: np.ndarray,
    ) -> RelationMetricValues:
        pair_coverage = np.sqrt(
            np.maximum(
                prototype_coverage[:, None] * prototype_coverage[None, :],
                0.0,
            )
        )
        return RelationMetricValues(
            transition=float(np.sum(self.transition * pair_coverage)),
            cooccurrence=float(np.sum(self.cooccurrence * pair_coverage)),
            sequence=float(
                np.sum(
                    self.transition
                    * np.log1p(np.maximum(sequence_counts, 0.0))
                    / self.sequence_denominator
                )
            ),
        )
