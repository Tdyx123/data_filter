"""Incremental relation-minus-redundancy objective."""

from __future__ import annotations

import math
from collections.abc import Sequence
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


@dataclass(frozen=True)
class CocoreObjectiveUpdateState:
    """Sparse branch-local changes relative to one shared main state."""

    main_state: CocoreObjectiveState
    selected_indices: tuple[int, ...]
    redundancy_deltas: tuple[float, ...]
    prototype_override_indices: np.ndarray
    prototype_override_values: np.ndarray
    sequence_override_flat_indices: np.ndarray
    sequence_override_values: np.ndarray
    task_count_deltas: np.ndarray
    relation: float
    redundancy: float
    score: float


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

    def extend_state(
        self,
        state: CocoreObjectiveState,
        indices: Sequence[int],
    ) -> CocoreObjectiveState:
        extended = self.clone_state(state)
        for index in indices:
            self.add_candidate(extended, int(index))
        return extended

    def empty_update_state(
        self,
        main_state: CocoreObjectiveState,
    ) -> CocoreObjectiveUpdateState:
        return CocoreObjectiveUpdateState(
            main_state=main_state,
            selected_indices=(),
            redundancy_deltas=(),
            prototype_override_indices=np.empty(0, dtype=np.int64),
            prototype_override_values=np.empty(0, dtype=np.float64),
            sequence_override_flat_indices=np.empty(0, dtype=np.int64),
            sequence_override_values=np.empty(0, dtype=np.float64),
            task_count_deltas=np.zeros_like(main_state.task_counts),
            relation=float(main_state.relation),
            redundancy=0.0,
            score=self.relation_weight * float(main_state.relation),
        )

    def materialize_update_state(
        self,
        update_state: CocoreObjectiveUpdateState,
    ) -> CocoreObjectiveState:
        main = update_state.main_state
        selected_mask = main.selected_mask.copy()
        if update_state.selected_indices:
            selected_mask[np.asarray(update_state.selected_indices, dtype=np.int64)] = True
        prototype_coverage = main.prototype_coverage.copy()
        prototype_coverage[update_state.prototype_override_indices] = (
            update_state.prototype_override_values
        )
        sequence_relation_counts = main.sequence_relation_counts.copy()
        sequence_relation_counts.ravel()[update_state.sequence_override_flat_indices] = (
            update_state.sequence_override_values
        )
        return CocoreObjectiveState(
            selected_mask=selected_mask,
            prototype_coverage=prototype_coverage,
            sequence_relation_counts=sequence_relation_counts,
            task_counts=main.task_counts + update_state.task_count_deltas,
            relation=float(update_state.relation),
            redundancy=float(update_state.redundancy),
            score=float(update_state.score),
        )

    def redundancy_from_indices(self, indices: Sequence[int]) -> float:
        normalized = tuple(int(index) for index in indices)
        if len(normalized) != len(set(normalized)):
            raise ValueError("similarity penalty indices cannot contain duplicates")
        candidate_count = len(self.graph.sample_ids)
        if any(index < 0 or index >= candidate_count for index in normalized):
            raise ValueError("similarity penalty indices must contain in-range values")
        selected: set[int] = set()
        redundancy = 0.0
        for index in normalized:
            redundancy_delta = (
                sum(
                    penalty
                    for other, penalty in self.redundancy_adjacency[index]
                    if other in selected
                )
                / self.redundancy_normalizer
            )
            redundancy += redundancy_delta
            selected.add(index)
        return float(redundancy)

    def extend_update_state(
        self,
        update_state: CocoreObjectiveUpdateState,
        indices: Sequence[int],
        *,
        similarity_main_indices: Sequence[int],
    ) -> CocoreObjectiveUpdateState:
        main = update_state.main_state
        sample = tuple(int(index) for index in similarity_main_indices)
        if len(sample) != len(set(sample)):
            raise ValueError("similarity main indices cannot contain duplicates")
        if any(
            index < 0
            or index >= len(self.graph.sample_ids)
            or not main.selected_mask[index]
            for index in sample
        ):
            raise ValueError("similarity main indices must belong to the main state")

        added = tuple(int(index) for index in indices)
        state = self.materialize_update_state(update_state)
        penalty_selected = set(sample)
        penalty_selected.update(update_state.selected_indices)
        redundancy_deltas = list(update_state.redundancy_deltas)
        for index in added:
            previous_redundancy = float(state.redundancy)
            self.add_candidate(state, index)
            redundancy_delta = (
                sum(
                    penalty
                    for other, penalty in self.redundancy_adjacency[index]
                    if other in penalty_selected
                )
                / self.redundancy_normalizer
            )
            state.redundancy = previous_redundancy + redundancy_delta
            state.score = self.relation_weight * state.relation - state.redundancy
            penalty_selected.add(index)
            redundancy_deltas.append(float(redundancy_delta))
        selected_indices = update_state.selected_indices + added

        prototype_indices = np.flatnonzero(
            state.prototype_coverage != main.prototype_coverage
        ).astype(np.int64, copy=False)
        sequence_flat = state.sequence_relation_counts.ravel()
        main_sequence_flat = main.sequence_relation_counts.ravel()
        sequence_indices = np.flatnonzero(sequence_flat != main_sequence_flat).astype(
            np.int64,
            copy=False,
        )
        return CocoreObjectiveUpdateState(
            main_state=main,
            selected_indices=selected_indices,
            redundancy_deltas=tuple(redundancy_deltas),
            prototype_override_indices=prototype_indices,
            prototype_override_values=state.prototype_coverage[prototype_indices].copy(),
            sequence_override_flat_indices=sequence_indices,
            sequence_override_values=sequence_flat[sequence_indices].copy(),
            task_count_deltas=(state.task_counts - main.task_counts).copy(),
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
