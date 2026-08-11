from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from cocore.objective import CocoreObjectiveContext, recompute_objective
from cocore.selection import (
    BeamRolloutSelector,
    CandidatePoolConfig,
    allocate_residual_task_quotas,
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


def test_residual_hamilton_quotas_ignore_seed_distribution() -> None:
    quotas = allocate_residual_task_quotas(
        np.asarray([0, 0, 1, 1, 1], dtype=np.int64),
        selected_indices=[0],
        budget=4,
    )

    assert quotas == {0: 1, 1: 2}


def _beam_graph(count: int = 14) -> GraphData:
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


def test_beam_rollout_uses_root_eight_node_four_exact_dedup_and_top_eight() -> None:
    graph = _beam_graph()
    context = CocoreObjectiveContext(graph, 1.0, similarity_threshold=0.8)
    seed = build_max_coverage_seed(context, budget=3)
    quotas = allocate_residual_task_quotas(
        graph.task_indices,
        selected_indices=seed.selected_indices,
        budget=3,
    )
    selector = BeamRolloutSelector(
        context,
        quotas,
        seed=17,
        pool_config=CandidatePoolConfig(
            global_candidates=32,
            prototype_candidates=0,
            similarity_candidates=0,
            random_candidates=0,
        ),
    )

    result = selector.select(3, initial_indices=seed.selected_indices)

    assert result.selected_indices == (13, 12, 11)
    assert result.selection_phases == ("coverage_seed", "rollout", "rollout")
    assert result.rollout_depths == (0, 1, 2)
    assert result.layer_stats[0].generated == 8
    assert result.layer_stats[0].unique == 8
    assert result.layer_stats[0].retained == 8
    assert result.layer_stats[1].generated == 32
    assert result.layer_stats[1].unique < result.layer_stats[1].generated
    assert result.layer_stats[1].retained == 8
    assert len(result.final_beam_scores) == 8


def test_sparse_candidate_pool_random_component_depends_on_set_not_path_order() -> None:
    graph = _beam_graph(20)
    context = CocoreObjectiveContext(graph, 1.0, similarity_threshold=0.8)
    selector = BeamRolloutSelector(
        context,
        {0: 10},
        seed=123,
        pool_config=CandidatePoolConfig(
            global_candidates=0,
            prototype_candidates=0,
            similarity_candidates=0,
            random_candidates=5,
        ),
    )
    left = context.state_from_indices([0, 1])
    right = context.state_from_indices([1, 0])

    left_pool = selector.candidate_pool(left, {0: 0})
    right_pool = selector.candidate_pool(right, {0: 0})

    assert left_pool == right_pool
    assert len(left_pool) >= 5


def test_beam_rollout_returns_seed_without_generating_a_layer_when_budget_is_full() -> None:
    graph = _beam_graph()
    context = CocoreObjectiveContext(graph, 1.0, similarity_threshold=0.8)
    seed = build_max_coverage_seed(context, budget=1)
    selector = BeamRolloutSelector(context, {0: 0}, seed=1)

    result = selector.select(1, initial_indices=seed.selected_indices)

    assert result.selected_indices == seed.selected_indices
    assert result.layer_stats == ()
