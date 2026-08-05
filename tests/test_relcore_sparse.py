from __future__ import annotations

import numpy as np
from scipy import sparse

from relcore.schemas import EdgeTable, GraphData
from relcore.selection.multibranch import MultiBranchSelector
from relcore.selection.objective import ObjectiveContext, ObjectiveWeights
from relcore.selection.seeds import generate_seed_pairs
from relcore.selection.sparse_greedy import SparseGreedySelector
from tests.test_relcore_objective import _small_graph


def _empty_edges(edge_type: str) -> EdgeTable:
    return EdgeTable(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        edge_type,
    )


def test_seed_generation_includes_high_value_real_sequence_pair():
    context = ObjectiveContext(_small_graph(), ObjectiveWeights(), similarity_threshold=0.8)

    seeds = generate_seed_pairs(
        context,
        {0: 2, 1: 2},
        budget=4,
        branches=4,
        seed_candidates=16,
        seed_similarity_threshold=0.9,
        transition_seed_threshold=0.0,
        pairs_per_transition=2,
    )

    assert seeds
    assert seeds[0] == (0, 1)


def test_seed_generation_reserves_a_low_frequency_reliable_transition():
    sample_ids = [f"ep{index:06d}_fragment_000000_000014" for index in range(8)]
    primary = np.asarray([[0], [0], [0], [1], [1], [1], [2], [3]], dtype=np.int32)
    graph = GraphData(
        sample_ids=sample_ids,
        task_indices=np.zeros(8, dtype=np.int64),
        embeddings=np.eye(8, dtype=np.float32),
        reliability=np.asarray([1.0] * 6 + [0.2, 0.2], dtype=np.float32),
        prototype_indices=primary,
        prototype_weights=np.ones((8, 1), dtype=np.float32),
        sequence_edges=_empty_edges("sequence"),
        similarity_edges=_empty_edges("similarity"),
        transition_matrix=sparse.csr_matrix(
            np.asarray(
                [
                    [0.0, 0.99, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.01],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        ),
        cooccurrence_matrix=sparse.csr_matrix((4, 4), dtype=np.float32),
    )
    context = ObjectiveContext(graph, ObjectiveWeights(), similarity_threshold=0.8)

    seeds = generate_seed_pairs(
        context,
        {0: 2},
        budget=2,
        branches=4,
        seed_candidates=4,
        seed_similarity_threshold=0.9999,
        transition_seed_threshold=0.0,
        pairs_per_transition=4,
    )

    assert any((int(primary[left, 0]), int(primary[right, 0])) == (2, 3) for left, right in seeds)


def test_sparse_greedy_fills_exact_budget_and_hard_quotas():
    graph = _small_graph()
    context = ObjectiveContext(graph, ObjectiveWeights(), similarity_threshold=0.8)

    result = SparseGreedySelector(
        context,
        {0: 1, 1: 1},
        seed=17,
        global_candidates=2,
        residual_candidates=2,
        random_candidates=1,
    ).select(2)

    assert len(result.selected_indices) == 2
    assert {int(graph.task_indices[index]) for index in result.selected_indices} == {0, 1}


def test_multibranch_returns_at_least_the_empty_seed_sparse_objective():
    graph = _small_graph()
    context = ObjectiveContext(graph, ObjectiveWeights(), similarity_threshold=0.8)
    quotas = {0: 1, 1: 1}
    baseline = SparseGreedySelector(context, quotas, seed=31).select(2)

    selector = MultiBranchSelector(
        context,
        quotas,
        branches=4,
        seed=31,
        local_search_enabled=True,
        local_search_rounds=1,
    )
    result = selector.select(2)

    assert len(result.selected_indices) == 2
    assert result.objective_value >= baseline.objective_value - 1.0e-9
    assert len(selector.branch_results) == 4
