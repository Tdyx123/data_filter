from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from cocore.objective import CocoreObjectiveContext, recompute_objective
from cocore.selection import (
    LazyHeapSelector,
    build_max_coverage_seed,
)
from relcore.schemas import EdgeTable, GraphData


def _edges(rows: list[tuple[int, int, float]], edge_type: str) -> EdgeTable:
    return EdgeTable(
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        np.asarray([row[2] for row in rows], dtype=np.float32),
        edge_type,
    )


def _graph() -> GraphData:
    return GraphData(
        sample_ids=["b", "c", "a"],
        task_indices=np.asarray([0, 1, 1], dtype=np.int64),
        embeddings=np.eye(3, dtype=np.float32),
        reliability=np.asarray([1.0, 0.5, 0.8], dtype=np.float32),
        prototype_indices=np.asarray([[0, -1], [1, -1], [0, 1]], dtype=np.int32),
        prototype_weights=np.asarray([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        sequence_edges=_edges([], "sequence"),
        similarity_edges=_edges([(0, 2, 0.9)], "similarity"),
        transition_matrix=sparse.csr_matrix((2, 2), dtype=np.float32),
        cooccurrence_matrix=sparse.csr_matrix(
            np.asarray([[0.0, 0.25], [0.75, 0.0]], dtype=np.float32)
        ),
        prototype_labels=("move forward", "open gripper"),
    )


def test_incremental_asymmetric_cooccurrence_and_redundancy_match_literal_values() -> None:
    context = CocoreObjectiveContext(_graph(), cooccurrence_weight=2.0, similarity_threshold=0.8)
    state = context.state_from_indices([0, 1])

    assert state.prototype_mass.tolist() == pytest.approx([1.0, 0.5])
    assert state.cooccurrence == pytest.approx(0.5)
    assert state.redundancy == pytest.approx(0.0)
    assert state.score == pytest.approx(1.0)
    assert context.marginal_gain(state, 2) == pytest.approx(0.52)

    context.add_candidate(state, 2)

    assert state.prototype_mass.tolist() == pytest.approx([1.4, 0.9])
    assert state.cooccurrence == pytest.approx(1.26)
    assert state.redundancy == pytest.approx(1.0)
    assert state.score == pytest.approx(1.52)
    assert recompute_objective(
        [0, 1, 2],
        _graph(),
        cooccurrence_weight=2.0,
        similarity_threshold=0.8,
    ).score == pytest.approx(state.score)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_objective_rejects_invalid_cooccurrence_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="cooccurrence_weight"):
        CocoreObjectiveContext(_graph(), weight, similarity_threshold=0.8)


def test_max_coverage_seed_uses_reliable_soft_assignments_and_stable_ties() -> None:
    graph = _graph()
    graph.sample_ids[0] = "z"
    graph.sample_ids.append("a0")
    graph.task_indices = np.append(graph.task_indices, 0)
    graph.embeddings = np.vstack([graph.embeddings, np.zeros((1, 3), dtype=np.float32)])
    graph.reliability = np.append(graph.reliability, 1.0).astype(np.float32)
    graph.prototype_indices = np.vstack([graph.prototype_indices, [0, -1]])
    graph.prototype_weights = np.vstack([graph.prototype_weights, [1.0, 0.0]])

    context = CocoreObjectiveContext(graph, 1.0, similarity_threshold=0.8)
    seed = build_max_coverage_seed(context, budget=3)

    assert seed.selected_indices == (3, 1)
    assert seed.target_coverage.tolist() == pytest.approx([1.0, 0.5])
    assert seed.achieved_coverage.tolist() == pytest.approx([1.0, 0.5])


def test_max_coverage_seed_rejects_budget_smaller_than_required_union() -> None:
    context = CocoreObjectiveContext(_graph(), 1.0, similarity_threshold=0.8)

    with pytest.raises(ValueError, match="minimum required budget is 2"):
        build_max_coverage_seed(context, budget=1)


def _heap_graph(count: int = 14) -> GraphData:
    reliabilities = np.linspace(0.1, 1.0, count, dtype=np.float32)
    return GraphData(
        sample_ids=[f"node-{index:02d}" for index in range(count)],
        task_indices=np.zeros(count, dtype=np.int64),
        embeddings=np.eye(count, dtype=np.float32),
        reliability=reliabilities,
        prototype_indices=np.zeros((count, 1), dtype=np.int32),
        prototype_weights=np.ones((count, 1), dtype=np.float32),
        sequence_edges=_edges([], "sequence"),
        similarity_edges=_edges([], "similarity"),
        transition_matrix=sparse.csr_matrix((1, 1), dtype=np.float32),
        cooccurrence_matrix=sparse.csr_matrix(np.ones((1, 1), dtype=np.float32)),
        prototype_labels=("move forward",),
    )


def test_lazy_heap_selects_current_top_then_refreshes_the_next_stale_top() -> None:
    graph = _heap_graph()
    context = CocoreObjectiveContext(graph, 1.0, similarity_threshold=0.8)
    seed = build_max_coverage_seed(context, budget=3)
    selector = LazyHeapSelector(context, max_refreshes=100)

    result = selector.select(3, initial_indices=seed.selected_indices)

    assert result.selected_indices == (13, 12, 11)
    assert result.selection_phases == ("coverage_seed", "heap", "heap")
    assert result.selection_steps == (0, 1, 2)
    assert result.heap_refreshes == (0, 0, 1)
    assert result.initial_heap_size == 13
    assert result.total_refreshes == 1
    assert result.capped_selections == 0
    assert result.max_refreshes_observed == 1


def test_lazy_heap_returns_seed_without_building_a_heap_when_budget_is_full() -> None:
    graph = _heap_graph()
    context = CocoreObjectiveContext(graph, 1.0, similarity_threshold=0.8)
    seed = build_max_coverage_seed(context, budget=1)
    selector = LazyHeapSelector(context)

    result = selector.select(1, initial_indices=seed.selected_indices)

    assert result.selected_indices == seed.selected_indices
    assert result.initial_heap_size == 0
    assert result.total_refreshes == 0


@dataclass
class _ScriptedState:
    selected_mask: np.ndarray
    score: float = 0.0
    cooccurrence: float = 0.0
    redundancy: float = 0.0


class _ScriptedContext:
    def __init__(self, sample_ids: list[str], gain) -> None:
        self.graph = SimpleNamespace(sample_ids=sample_ids)
        self._gain = gain

    def empty_state(self) -> _ScriptedState:
        return _ScriptedState(np.zeros(len(self.graph.sample_ids), dtype=bool))

    def marginal_gain(self, state: _ScriptedState, index: int) -> float:
        return float(self._gain(int(state.selected_mask.sum()), index))

    def add_candidate(self, state: _ScriptedState, index: int) -> None:
        gain = float(self._gain(int(state.selected_mask.sum()), index))
        state.selected_mask[index] = True
        state.score += gain
        state.cooccurrence = state.score


def test_lazy_heap_keeps_refreshing_when_a_recomputed_gain_falls_below_stale_entries() -> None:
    def gain(selected_count: int, index: int) -> float:
        if selected_count == 0:
            return 0.0
        if selected_count == 1:
            return {1: 10.0, 2: 9.0, 3: 8.0}[index]
        return {2: 1.0, 3: 7.0}[index]

    selector = LazyHeapSelector(
        _ScriptedContext(["seed", "first", "falls", "winner"], gain),
        max_refreshes=100,
    )

    result = selector.select(3, initial_indices=[0])

    assert result.selected_indices == (0, 1, 3)
    assert result.heap_refreshes == (0, 0, 2)


def test_lazy_heap_selects_a_refreshed_entry_as_soon_as_it_returns_to_the_top() -> None:
    def gain(selected_count: int, index: int) -> float:
        if selected_count == 0:
            return 0.0
        if selected_count == 1:
            return {1: 10.0, 2: 9.0, 3: 8.0}[index]
        return {2: 20.0, 3: 7.0}[index]

    selector = LazyHeapSelector(
        _ScriptedContext(["seed", "first", "winner", "other"], gain),
        max_refreshes=100,
    )

    result = selector.select(3, initial_indices=[0])

    assert result.selected_indices == (0, 1, 2)
    assert result.heap_refreshes == (0, 0, 1)


def test_lazy_heap_caps_refreshes_and_lazily_skips_the_selected_heap_entry() -> None:
    sample_ids = [f"node-{index:03d}" for index in range(102)]

    def gain(selected_count: int, index: int) -> float:
        if selected_count == 0:
            return 0.0
        if selected_count == 1:
            return 10_000.0 - index
        return float(index)

    selector = LazyHeapSelector(_ScriptedContext(sample_ids, gain), max_refreshes=100)

    result = selector.select(4, initial_indices=[0])

    assert result.selected_indices == (0, 1, 101, 100)
    assert result.heap_refreshes == (0, 0, 100, 1)
    assert len(set(result.selected_indices)) == 4
    assert result.total_refreshes == 101
    assert result.capped_selections == 1
    assert result.max_refreshes_observed == 100


def test_lazy_heap_breaks_equal_gain_ties_by_sample_id() -> None:
    context = _ScriptedContext(
        ["seed", "z-candidate", "a-candidate"],
        lambda selected_count, index: 0.0 if selected_count == 0 else 1.0,
    )

    result = LazyHeapSelector(context).select(2, initial_indices=[0])

    assert result.selected_indices == (0, 2)


def test_lazy_heap_has_no_task_quota_and_can_select_one_task_repeatedly() -> None:
    graph = _heap_graph(6)
    graph.task_indices = np.asarray([1, 1, 1, 0, 0, 1], dtype=np.int64)
    context = CocoreObjectiveContext(graph, 1.0, similarity_threshold=0.8)
    seed = build_max_coverage_seed(context, budget=3)

    result = LazyHeapSelector(context).select(3, initial_indices=seed.selected_indices)

    assert result.selected_indices == (5, 4, 3)
    assert graph.task_indices[list(result.selected_indices)].tolist() == [1, 0, 0]


def test_lazy_heap_rejects_non_finite_gains() -> None:
    context = _ScriptedContext(
        ["seed", "bad"],
        lambda selected_count, index: 0.0 if selected_count == 0 else float("nan"),
    )

    with pytest.raises(ValueError, match="finite marginal gain"):
        LazyHeapSelector(context).select(2, initial_indices=[0])


class _CountingObjectiveContext(CocoreObjectiveContext):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.marginal_gain_calls = 0

    def marginal_gain(self, state, candidate: int) -> float:
        self.marginal_gain_calls += 1
        return super().marginal_gain(state, candidate)


def test_lazy_heap_bounds_marginal_gain_recomputations() -> None:
    graph = _heap_graph(20)
    context = _CountingObjectiveContext(graph, 1.0, similarity_threshold=0.8)
    budget = 10
    seed = build_max_coverage_seed(context, budget=budget)
    selector = LazyHeapSelector(context, max_refreshes=3)

    result = selector.select(budget, initial_indices=seed.selected_indices)

    heap_selections = budget - len(seed.selected_indices)
    remaining_candidates = len(graph.sample_ids) - len(seed.selected_indices)
    calls_excluding_seed = context.marginal_gain_calls - len(seed.selected_indices)
    assert calls_excluding_seed <= remaining_candidates + 3 * heap_selections
    recomputed = CocoreObjectiveContext(
        graph, 1.0, similarity_threshold=0.8
    ).state_from_indices(result.selected_indices)
    assert result.objective_value == pytest.approx(recomputed.score)
    assert result.cooccurrence == pytest.approx(recomputed.cooccurrence)
    assert result.redundancy == pytest.approx(recomputed.redundancy)
