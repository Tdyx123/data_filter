from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from relcore.schemas import EdgeTable, GraphData
from relcore.selection.greedy import ExactGreedySelector
from relcore.selection.objective import ObjectiveContext, ObjectiveWeights, recompute_objective
from relcore.selection.quota import allocate_task_quotas


def _edge_table(edges: list[tuple[int, int, float]], edge_type: str) -> EdgeTable:
    return EdgeTable(
        np.asarray([edge[0] for edge in edges], dtype=np.int64),
        np.asarray([edge[1] for edge in edges], dtype=np.int64),
        np.asarray([edge[2] for edge in edges], dtype=np.float32),
        edge_type,
    )


def _small_graph() -> GraphData:
    return GraphData(
        sample_ids=["a1", "a2", "b1", "b2"],
        task_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        embeddings=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.01, 0.99]],
            dtype=np.float32,
        ),
        reliability=np.asarray([0.9, 0.9, 0.8, 0.8], dtype=np.float32),
        prototype_indices=np.asarray([[0], [1], [0], [1]], dtype=np.int32),
        prototype_weights=np.ones((4, 1), dtype=np.float32),
        sequence_edges=_edge_table([(0, 1, 0.9)], "sequence"),
        similarity_edges=_edge_table([(0, 2, 0.99), (1, 3, 0.99)], "similarity"),
        transition_matrix=sparse.csr_matrix(np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)),
        cooccurrence_matrix=sparse.csr_matrix((2, 2), dtype=np.float32),
    )


def test_incremental_objective_matches_full_recomputation():
    graph = _small_graph()
    context = ObjectiveContext(graph, ObjectiveWeights(), similarity_threshold=0.8)
    state = context.empty_state()

    for candidate in (0, 1, 3):
        expected_gain = (
            recompute_objective(
                np.flatnonzero(state.selected_mask).tolist() + [candidate],
                graph,
                ObjectiveWeights(),
                similarity_threshold=0.8,
            ).total
            - state.objective_value
        )
        actual_gain = context.marginal_gain(state, candidate)
        assert actual_gain == pytest.approx(expected_gain, abs=1.0e-6)
        context.add_candidate(state, candidate)

    expected = recompute_objective([0, 1, 3], graph, ObjectiveWeights(), similarity_threshold=0.8)
    assert state.objective_value == pytest.approx(expected.total, abs=1.0e-6)


def test_incremental_remove_matches_full_recomputation_and_allows_reinsertion():
    graph = _small_graph()
    weights = ObjectiveWeights()
    context = ObjectiveContext(graph, weights, similarity_threshold=0.8)
    state = context.state_from_indices([0, 1, 3])

    context.remove_candidate(state, 1)

    expected = recompute_objective([0, 3], graph, weights, similarity_threshold=0.8)
    assert state.objective_value == pytest.approx(expected.total, abs=1.0e-6)
    expected_gain = (
        recompute_objective([0, 1, 3], graph, weights, similarity_threshold=0.8).total
        - expected.total
    )
    assert context.marginal_gain(state, 1) == pytest.approx(expected_gain, abs=1.0e-6)


def test_padded_soft_assignments_match_unpadded_objective() -> None:
    graph = _small_graph()
    padded = replace(
        graph,
        prototype_indices=np.pad(
            graph.prototype_indices,
            ((0, 0), (0, 3)),
            constant_values=-1,
        ),
        prototype_weights=np.pad(
            graph.prototype_weights,
            ((0, 0), (0, 3)),
            constant_values=0.0,
        ),
    )

    expected = recompute_objective([0, 1, 3], graph, ObjectiveWeights(), similarity_threshold=0.8)
    actual = recompute_objective([0, 1, 3], padded, ObjectiveWeights(), similarity_threshold=0.8)

    assert actual == expected


def test_objective_members_exclude_padded_soft_assignments() -> None:
    graph = _small_graph()
    padded = replace(
        graph,
        prototype_indices=np.pad(
            graph.prototype_indices,
            ((0, 0), (0, 3)),
            constant_values=-1,
        ),
        prototype_weights=np.pad(
            graph.prototype_weights,
            ((0, 0), (0, 3)),
            constant_values=0.0,
        ),
    )

    context = ObjectiveContext(padded, ObjectiveWeights(), similarity_threshold=0.8)

    assert context.prototype_members == [
        [(0, 1.0), (2, 1.0)],
        [(1, 1.0), (3, 1.0)],
    ]


def test_objective_gives_unreachable_catalog_prototypes_zero_rarity_weight() -> None:
    graph = _small_graph()
    graph.prototype_indices[:] = 0
    graph.transition_matrix = sparse.csr_matrix((2, 2), dtype=np.float32)
    graph.cooccurrence_matrix = sparse.csr_matrix((2, 2), dtype=np.float32)

    context = ObjectiveContext(graph, ObjectiveWeights(), similarity_threshold=0.8)

    np.testing.assert_array_equal(context.prototype_weights, [1.0, 0.0])


def test_incremental_objective_matches_full_recomputation_on_random_soft_graph():
    rng = np.random.default_rng(19)
    node_count, prototype_count = 10, 4
    prototype_indices = np.stack(
        [rng.choice(prototype_count, size=2, replace=False) for _ in range(node_count)]
    ).astype(np.int32)
    prototype_weights = rng.uniform(0.1, 0.8, size=(node_count, 2)).astype(np.float32)
    prototype_weights *= np.minimum(
        1.0,
        0.95 / prototype_weights.sum(axis=1, keepdims=True),
    )
    transition = rng.uniform(size=(prototype_count, prototype_count)).astype(np.float32)
    transition /= transition.sum()
    cooccurrence = rng.uniform(size=(prototype_count, prototype_count)).astype(np.float32)
    cooccurrence /= cooccurrence.sum()
    graph = GraphData(
        sample_ids=[f"node-{index:02d}" for index in range(node_count)],
        task_indices=np.asarray([0] * 5 + [1] * 5, dtype=np.int64),
        embeddings=rng.normal(size=(node_count, 6)).astype(np.float32),
        reliability=rng.uniform(0.1, 1.0, size=node_count).astype(np.float32),
        prototype_indices=prototype_indices,
        prototype_weights=prototype_weights,
        sequence_edges=_edge_table(
            [(index, index + 1, 0.5) for index in range(node_count - 1)],
            "sequence",
        ),
        similarity_edges=_edge_table(
            [(index, index + 2, 0.9) for index in range(node_count - 2)],
            "similarity",
        ),
        transition_matrix=sparse.csr_matrix(transition),
        cooccurrence_matrix=sparse.csr_matrix(cooccurrence),
    )
    weights = ObjectiveWeights()
    context = ObjectiveContext(graph, weights, similarity_threshold=0.8)
    order = rng.permutation(node_count).tolist()
    state = context.empty_state()
    selected: list[int] = []

    for candidate in order:
        expected_gain = (
            recompute_objective(
                selected + [candidate], graph, weights, similarity_threshold=0.8
            ).total
            - recompute_objective(selected, graph, weights, similarity_threshold=0.8).total
        )
        assert context.marginal_gain(state, candidate) == pytest.approx(expected_gain, abs=1.0e-6)
        context.add_candidate(state, candidate)
        selected.append(candidate)
        assert state.objective_value == pytest.approx(
            recompute_objective(selected, graph, weights, similarity_threshold=0.8).total,
            abs=1.0e-6,
        )


def test_true_sequence_successor_has_more_gain_than_nonadjacent_equivalent():
    context = ObjectiveContext(_small_graph(), ObjectiveWeights(), similarity_threshold=0.8)
    state = context.empty_state()
    context.add_candidate(state, 0)

    assert context.marginal_gain(state, 1) > context.marginal_gain(state, 3)


def test_similarity_neighbor_loses_gain_after_duplicate_is_selected():
    context = ObjectiveContext(_small_graph(), ObjectiveWeights(), similarity_threshold=0.8)
    empty = context.empty_state()
    gain_before = context.marginal_gain(empty, 2)
    context.add_candidate(empty, 0)

    assert context.marginal_gain(empty, 2) < gain_before


def test_capacity_aware_hamilton_quotas_sum_to_budget_and_respect_minimum():
    quotas = allocate_task_quotas(
        np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 2]),
        budget=5,
        minimum_per_task=1,
    )

    assert quotas == {0: 2, 1: 2, 2: 1}


def test_exact_greedy_fills_budget_without_exceeding_task_quotas():
    graph = _small_graph()
    quotas = {0: 1, 1: 1}
    result = ExactGreedySelector(
        ObjectiveContext(graph, ObjectiveWeights(), similarity_threshold=0.8),
        quotas,
    ).select(2)

    assert len(result.selected_indices) == 2
    assert {int(graph.task_indices[index]) for index in result.selected_indices} == {0, 1}
    assert (
        result.objective_value
        == recompute_objective(
            result.selected_indices,
            graph,
            ObjectiveWeights(),
            similarity_threshold=0.8,
        ).total
    )


def test_exact_greedy_global_mode_can_fill_budget_from_one_task():
    graph = _small_graph()
    result = ExactGreedySelector(
        ObjectiveContext(graph, ObjectiveWeights(), similarity_threshold=0.8),
        None,
    ).select(2)

    assert result.selected_indices == [0, 1]


def test_exact_greedy_fills_budget_even_when_the_best_remaining_gain_is_negative():
    graph = GraphData(
        sample_ids=["a", "b"],
        task_indices=np.zeros(2, dtype=np.int64),
        embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        reliability=np.ones(2, dtype=np.float32),
        prototype_indices=np.zeros((2, 1), dtype=np.int32),
        prototype_weights=np.ones((2, 1), dtype=np.float32),
        sequence_edges=_edge_table([], "sequence"),
        similarity_edges=_edge_table([(0, 1, 0.99)], "similarity"),
        transition_matrix=sparse.csr_matrix((1, 1), dtype=np.float32),
        cooccurrence_matrix=sparse.csr_matrix((1, 1), dtype=np.float32),
    )
    context = ObjectiveContext(
        graph,
        ObjectiveWeights(node=0.0, transition=0.0, cooccurrence=0.0, sequence=0.0),
        similarity_threshold=0.8,
    )

    result = ExactGreedySelector(context, {0: 2}).select(2)

    assert result.selected_indices == [0, 1]
    assert result.marginal_gains[-1] < 0.0


def test_low_quality_node_in_an_anomalous_chain_is_not_selected_for_continuity_alone():
    reliability = np.asarray([0.9, 0.9, 0.05, 0.9, 0.8, 0.8, 0.8, 0.8], dtype=np.float32)
    prototype_indices = np.asarray([[0], [1], [2], [3], [0], [1], [2], [3]], dtype=np.int32)
    transition = np.zeros((4, 4), dtype=np.float32)
    transition[0, 1] = reliability[0] * reliability[1] + reliability[4] * reliability[5]
    transition[1, 2] = reliability[1] * reliability[2] + reliability[5] * reliability[6]
    transition[2, 3] = reliability[2] * reliability[3] + reliability[6] * reliability[7]
    transition /= transition.sum()
    graph = GraphData(
        sample_ids=[f"a{index + 1}" for index in range(4)]
        + [f"b{index + 1}" for index in range(4)],
        task_indices=np.zeros(8, dtype=np.int64),
        embeddings=np.eye(8, dtype=np.float32),
        reliability=reliability,
        prototype_indices=prototype_indices,
        prototype_weights=np.ones((8, 1), dtype=np.float32),
        sequence_edges=_edge_table(
            [(0, 1, 0.9), (1, 2, 0.05), (2, 3, 0.05), (4, 5, 0.8), (5, 6, 0.8), (6, 7, 0.8)],
            "sequence",
        ),
        similarity_edges=_edge_table([], "similarity"),
        transition_matrix=sparse.csr_matrix(transition),
        cooccurrence_matrix=sparse.csr_matrix((4, 4), dtype=np.float32),
    )
    weights = ObjectiveWeights()
    context = ObjectiveContext(graph, weights, similarity_threshold=0.8)

    result = ExactGreedySelector(context, {0: 4}).select(4)

    assert 2 not in result.selected_indices
    assert (
        recompute_objective([4, 5, 6, 7], graph, weights, similarity_threshold=0.8).total
        > recompute_objective([0, 1, 2, 3], graph, weights, similarity_threshold=0.8).total
    )
