from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from cocore.objective import CocoreObjectiveContext, recompute_objective
from cocore.random_multibranch import RandomMultiBranchSelector
from cocore.selection import (
    LazyHeapSelector,
    build_max_coverage_seed,
)
from relcore.graph import build_graph
from relcore.graph.prototypes import PrototypeData
from relcore.schemas import ClipRecord, EdgeTable, GraphData
from relcore.selection.objective import (
    ObjectiveWeights,
    recompute_objective as recompute_relcore_objective,
)


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


def test_cocore_graph_consumes_absolute_prototype_confidence_without_normalizing() -> None:
    clips = [
        ClipRecord("a", 0, 0, "task", 0, 14, 15, None, "b"),
        ClipRecord("b", 0, 0, "task", 15, 29, 15, "a", None),
    ]
    prototypes = PrototypeData(
        centers=np.asarray([[1.0]], dtype=np.float32),
        indices=np.zeros((2, 1), dtype=np.int32),
        weights=np.asarray([[1.5], [0.5]], dtype=np.float32),
        labels=("leaf",),
    )

    graph = build_graph(
        clips,
        np.eye(2, dtype=np.float32),
        np.asarray([0.8, 0.5], dtype=np.float32),
        prototypes,
        knn=1,
        normalize_prototype_relations=False,
    )

    assert graph.transition_matrix.toarray()[0, 0] == pytest.approx(0.3)


def test_incremental_asymmetric_relation_and_redundancy_match_literal_values() -> None:
    context = CocoreObjectiveContext(
        _graph(), "cooccurrence", relation_weight=2.0, similarity_threshold=0.8
    )
    state = context.state_from_indices([0, 1])

    assert state.prototype_coverage.tolist() == pytest.approx([1.0, 0.5])
    assert state.relation == pytest.approx(2.0**-0.5)
    assert state.redundancy == pytest.approx(0.0)
    assert state.score == pytest.approx(2.0**0.5)
    assert context.marginal_gain(state, 2) == pytest.approx(-1.0)

    context.add_candidate(state, 2)

    assert state.prototype_coverage.tolist() == pytest.approx([1.0, 0.5])
    assert state.relation == pytest.approx(2.0**-0.5)
    assert state.redundancy == pytest.approx(1.0)
    assert state.score == pytest.approx(2.0**0.5 - 1.0)
    assert recompute_objective(
        [0, 1, 2],
        _graph(),
        relation_type="cooccurrence",
        relation_weight=2.0,
        similarity_threshold=0.8,
    ).score == pytest.approx(state.score)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_objective_rejects_invalid_relation_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="relation_weight"):
        CocoreObjectiveContext(
            _graph(), "cooccurrence", weight, similarity_threshold=0.8
        )


def test_objective_rejects_unknown_relation_type() -> None:
    with pytest.raises(ValueError, match="relation_type"):
        CocoreObjectiveContext(_graph(), "transition", 1.0, similarity_threshold=0.8)


def test_sequence_relation_counts_only_selected_directed_sequence_edges() -> None:
    graph = _graph()
    graph.sequence_edges = _edges([(0, 1, 1.0)], "sequence")
    graph.transition_matrix = sparse.csr_matrix(
        np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    )

    adjacent = CocoreObjectiveContext(
        graph, "sequence", relation_weight=2.0, similarity_threshold=0.8
    ).state_from_indices([0, 1])
    nonadjacent = CocoreObjectiveContext(
        graph, "sequence", relation_weight=2.0, similarity_threshold=0.8
    ).state_from_indices([0, 2])

    assert adjacent.relation == pytest.approx(1.0, abs=1.0e-7)
    assert adjacent.score == pytest.approx(2.0, abs=1.0e-7)
    assert nonadjacent.relation == pytest.approx(0.0)


@pytest.mark.parametrize("relation", ["sequence", "cooccurrence"])
@pytest.mark.parametrize("selected", [[], [0], [0, 1], [0, 1, 2]])
def test_cocore_raw_relation_matches_relcore_breakdown(
    relation: str,
    selected: list[int],
) -> None:
    graph = _graph()
    graph.sequence_edges = _edges([(0, 1, 1.0), (1, 2, 1.0)], "sequence")
    graph.transition_matrix = sparse.csr_matrix(
        np.asarray([[0.0, 0.6], [0.4, 0.0]], dtype=np.float32)
    )

    actual = CocoreObjectiveContext(
        graph, relation, relation_weight=1.0, similarity_threshold=0.8
    ).state_from_indices(selected)
    expected = recompute_relcore_objective(
        selected,
        graph,
        ObjectiveWeights(),
        similarity_threshold=0.8,
        prototype_gain_metrics=[relation],
    )

    assert actual.relation == pytest.approx(getattr(expected, relation), abs=1.0e-7)


@pytest.mark.parametrize("relation", ["sequence", "cooccurrence"])
def test_cocore_marginal_relation_gain_matches_relcore(relation: str) -> None:
    graph = _graph()
    graph.sequence_edges = _edges([(0, 1, 1.0), (1, 2, 1.0)], "sequence")
    graph.transition_matrix = sparse.csr_matrix(
        np.asarray([[0.0, 0.6], [0.4, 0.0]], dtype=np.float32)
    )
    context = CocoreObjectiveContext(
        graph, relation, relation_weight=1.0, similarity_threshold=0.8
    )
    state = context.state_from_indices([0])
    before = recompute_relcore_objective(
        [0],
        graph,
        ObjectiveWeights(),
        similarity_threshold=0.8,
        prototype_gain_metrics=[relation],
    )
    after = recompute_relcore_objective(
        [0, 1],
        graph,
        ObjectiveWeights(),
        similarity_threshold=0.8,
        prototype_gain_metrics=[relation],
    )

    assert context.marginal_gain(state, 1) == pytest.approx(
        getattr(after, relation) - getattr(before, relation),
        abs=1.0e-7,
    )


@pytest.mark.parametrize("relation", ["sequence", "cooccurrence"])
def test_objective_update_state_materializes_without_mutating_shared_main(
    relation: str,
) -> None:
    graph = _graph()
    graph.sequence_edges = _edges([(0, 1, 1.0), (1, 2, 1.0)], "sequence")
    graph.transition_matrix = sparse.csr_matrix(
        np.asarray([[0.0, 0.6], [0.4, 0.0]], dtype=np.float32)
    )
    context = CocoreObjectiveContext(
        graph, relation, relation_weight=1.0, similarity_threshold=0.8
    )
    main = context.state_from_indices([0])
    main_snapshot = context.clone_state(main)

    root = context.empty_update_state(main)
    first = context.extend_update_state(
        root,
        (1,),
        similarity_main_indices=(0,),
    )
    second = context.extend_update_state(
        root,
        (2,),
        similarity_main_indices=(0,),
    )
    materialized = context.materialize_update_state(first)
    expected = context.state_from_indices([0, 1])

    assert first.main_state is main
    assert second.main_state is main
    assert first.selected_indices == (1,)
    assert second.selected_indices == (2,)
    np.testing.assert_array_equal(materialized.selected_mask, expected.selected_mask)
    np.testing.assert_array_equal(materialized.prototype_coverage, expected.prototype_coverage)
    np.testing.assert_array_equal(
        materialized.sequence_relation_counts,
        expected.sequence_relation_counts,
    )
    np.testing.assert_array_equal(materialized.task_counts, expected.task_counts)
    assert materialized.relation == expected.relation
    assert materialized.redundancy == expected.redundancy
    assert materialized.score == expected.score
    np.testing.assert_array_equal(main.selected_mask, main_snapshot.selected_mask)
    np.testing.assert_array_equal(main.prototype_coverage, main_snapshot.prototype_coverage)
    np.testing.assert_array_equal(
        main.sequence_relation_counts,
        main_snapshot.sequence_relation_counts,
    )
    np.testing.assert_array_equal(main.task_counts, main_snapshot.task_counts)
    assert main.relation == main_snapshot.relation
    assert main.redundancy == main_snapshot.redundancy
    assert main.score == main_snapshot.score


def test_objective_update_state_penalizes_only_sampled_main_plus_active_pairs() -> None:
    graph = _heap_graph(103)
    graph.reliability = np.ones(103, dtype=np.float32)
    graph.similarity_edges = _edges(
        [
            (0, 1, 0.9),
            (0, 101, 0.9),
            (101, 102, 0.9),
            (100, 101, 0.9),
        ],
        "similarity",
    )
    context = CocoreObjectiveContext(
        graph, "cooccurrence", relation_weight=1.0, similarity_threshold=0.8
    )
    main = context.state_from_indices(list(range(101)))

    update = context.extend_update_state(
        context.empty_update_state(main),
        (101, 102),
        similarity_main_indices=tuple(range(100)),
    )

    assert update.similarity_main_indices == tuple(range(100))
    assert update.relation == pytest.approx(1.0)
    assert update.redundancy == pytest.approx(0.75)
    assert update.score == pytest.approx(0.25)


def test_max_coverage_seed_uses_reliable_soft_assignments_and_stable_ties() -> None:
    graph = _graph()
    graph.sample_ids[0] = "z"
    graph.sample_ids.append("a0")
    graph.task_indices = np.append(graph.task_indices, 0)
    graph.embeddings = np.vstack([graph.embeddings, np.zeros((1, 3), dtype=np.float32)])
    graph.reliability = np.append(graph.reliability, 1.0).astype(np.float32)
    graph.prototype_indices = np.vstack([graph.prototype_indices, [0, -1]])
    graph.prototype_weights = np.vstack([graph.prototype_weights, [1.0, 0.0]])

    context = CocoreObjectiveContext(graph, "cooccurrence", 1.0, similarity_threshold=0.8)
    seed = build_max_coverage_seed(context, budget=3)

    assert seed.selected_indices == (3, 1)
    assert seed.target_coverage.tolist() == pytest.approx([1.0, 0.5])
    assert seed.achieved_coverage.tolist() == pytest.approx([1.0, 0.5])


def test_max_coverage_seed_rejects_budget_smaller_than_required_union() -> None:
    context = CocoreObjectiveContext(
        _graph(), "cooccurrence", 1.0, similarity_threshold=0.8
    )

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


def test_random_multibranch_returns_coverage_without_starting_search() -> None:
    context = CocoreObjectiveContext(
        _heap_graph(), "cooccurrence", 1.0, similarity_threshold=0.8
    )
    seed = build_max_coverage_seed(context, budget=1)

    result = RandomMultiBranchSelector(context, seed=7).select(
        1, initial_indices=seed.selected_indices
    )

    assert result.selected_indices == seed.selected_indices
    assert result.selection_phases == ("coverage_seed",)
    assert result.selection_steps == (0,)
    assert result.heap_refreshes == (None,)
    assert result.rounds == 0
    assert result.evaluated_branches == 0
    assert result.recombinations == 0
    assert result.committed_clips == 0
    assert result.final_active_clips == 0
    assert result.round_runtime_seconds == ()
    assert result.recombination_runtime_seconds == ()


def test_random_multibranch_fills_a_partial_final_batch_reproducibly() -> None:
    graph = _heap_graph(30)
    first_context = CocoreObjectiveContext(
        graph, "cooccurrence", 1.0, similarity_threshold=0.8
    )
    first_seed = build_max_coverage_seed(first_context, budget=17)
    first = RandomMultiBranchSelector(first_context, seed=11).select(
        17, initial_indices=first_seed.selected_indices
    )
    second_context = CocoreObjectiveContext(
        graph, "cooccurrence", 1.0, similarity_threshold=0.8
    )
    second_seed = build_max_coverage_seed(second_context, budget=17)
    second = RandomMultiBranchSelector(second_context, seed=11).select(
        17, initial_indices=second_seed.selected_indices
    )

    assert first == second
    assert first.selected_indices == (
        29,
        21,
        6,
        12,
        27,
        4,
        0,
        24,
        2,
        23,
        18,
        16,
        5,
        1,
        25,
        9,
        15,
    )
    assert len(first.selected_indices) == 17
    assert len(set(first.selected_indices)) == 17
    assert first.rounds == 2
    assert first.evaluated_branches == 40
    assert first.recombinations == 0
    assert first.committed_clips == 0
    assert first.final_active_clips == 16
    assert first.selection_phases.count("coverage_seed") == 1
    assert first.selection_phases.count("branch_final") == 16
    assert first.selection_steps[-16:] == (2,) * 16


def test_random_multibranch_times_rounds_separately_from_recombination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = CocoreObjectiveContext(
        _heap_graph(10), "cooccurrence", 1.0, similarity_threshold=0.8
    )
    coverage = build_max_coverage_seed(context, budget=4)
    ticks = count(step=0.25)
    monkeypatch.setattr("cocore.random_multibranch.BATCH_SIZE", 1)
    monkeypatch.setattr("cocore.random_multibranch.FIRST_RECOMBINATION_ROUND", 2)
    monkeypatch.setattr("cocore.random_multibranch.RECOMBINATION_INTERVAL", 10)
    monkeypatch.setattr("cocore.random_multibranch.COMMIT_SIZE", 1)
    monkeypatch.setattr("cocore.random_multibranch.RETAINED_SIZE", 1)
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    result = RandomMultiBranchSelector(context, seed=11).select(
        4,
        initial_indices=coverage.selected_indices,
    )

    assert result.rounds == 3
    assert result.recombinations == 1
    assert result.round_runtime_seconds == pytest.approx((0.25, 0.25, 0.25))
    assert result.recombination_runtime_seconds == ((2, 0.25),)


def test_random_multibranch_does_not_clone_complete_branch_states(monkeypatch) -> None:
    context = CocoreObjectiveContext(
        _heap_graph(30), "cooccurrence", 1.0, similarity_threshold=0.8
    )
    coverage = build_max_coverage_seed(context, budget=17)

    def reject_clone(state):
        del state
        raise AssertionError("random multibranch copied a complete branch state")

    monkeypatch.setattr(context, "clone_state", reject_clone)

    result = RandomMultiBranchSelector(context, seed=11).select(
        17,
        initial_indices=coverage.selected_indices,
    )

    assert result.selected_indices == (
        29,
        21,
        6,
        12,
        27,
        4,
        0,
        24,
        2,
        23,
        18,
        16,
        5,
        1,
        25,
        9,
        15,
    )
    assert result.similarity_penalty_indices == result.selected_indices


def test_random_multibranch_similarity_sampling_uses_strict_100_clip_threshold() -> None:
    context = CocoreObjectiveContext(
        _heap_graph(101), "cooccurrence", 1.0, similarity_threshold=0.8
    )
    selector = RandomMultiBranchSelector(context, seed=17)

    class RejectingGenerator:
        def choice(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("100 fixed clips must not consume the similarity RNG")

    fixed_100 = tuple(range(100))
    assert selector._sample_similarity_main(RejectingGenerator(), fixed_100) == fixed_100

    sampled = selector._sample_similarity_main(
        np.random.default_rng(17),
        tuple(range(101)),
    )
    assert len(sampled) == 100
    assert len(set(sampled)) == 100
    assert set(sampled) < set(range(101))


def test_random_multibranch_resamples_similarity_main_for_every_new_branch() -> None:
    context = CocoreObjectiveContext(
        _heap_graph(112), "cooccurrence", 1.0, similarity_threshold=0.8
    )

    class RecordingSelector(RandomMultiBranchSelector):
        def __init__(self) -> None:
            super().__init__(context, seed=29)
            self.similarity_samples: list[tuple[int, ...]] = []

        def _sample_similarity_main(self, rng, fixed):
            sample = super()._sample_similarity_main(rng, fixed)
            self.similarity_samples.append(sample)
            return sample

    selector = RecordingSelector()
    result = selector.select(112, initial_indices=tuple(range(101)))

    assert len(selector.similarity_samples) == 8 + 32
    assert all(len(sample) == 100 for sample in selector.similarity_samples)
    assert all(set(sample) < set(range(101)) for sample in selector.similarity_samples)
    assert len(result.similarity_penalty_indices) == 100 + result.final_active_clips


def test_random_multibranch_final_result_uses_winner_similarity_sample() -> None:
    graph = _heap_graph(103)
    graph.reliability = np.ones(103, dtype=np.float32)
    graph.similarity_edges = _edges(
        [
            (0, 1, 0.9),
            (0, 101, 0.9),
            (101, 102, 0.9),
            (100, 101, 0.9),
        ],
        "similarity",
    )
    context = CocoreObjectiveContext(
        graph, "cooccurrence", 1.0, similarity_threshold=0.8
    )

    result = RandomMultiBranchSelector(context, seed=31).select(
        103,
        initial_indices=tuple(range(101)),
    )

    assert set(result.selected_indices[-2:]) == {101, 102}
    assert result.similarity_penalty_indices[-2:] == result.selected_indices[-2:]
    assert len(result.similarity_penalty_indices) == 102
    assert len(set(result.similarity_penalty_indices[:-2])) == 100
    assert set(result.similarity_penalty_indices[:-2]) < set(range(101))
    assert result.redundancy == pytest.approx(
        context.redundancy_from_indices(result.similarity_penalty_indices)
    )
    assert result.objective_value == pytest.approx(
        result.relation - result.redundancy
    )
    assert sum(result.score_deltas) == pytest.approx(result.objective_value)


def test_random_multibranch_uses_seed_to_break_equal_branch_scores() -> None:
    graph = _heap_graph(30)

    selected = []
    for random_seed in (3, 19):
        context = CocoreObjectiveContext(
            graph, "cooccurrence", 1.0, similarity_threshold=0.8
        )
        coverage = build_max_coverage_seed(context, budget=17)
        selected.append(
            RandomMultiBranchSelector(context, seed=random_seed).select(
                17, initial_indices=coverage.selected_indices
            ).selected_indices
        )

    assert selected[0] != selected[1]


def test_random_multibranch_selects_the_highest_scoring_initial_branch() -> None:
    graph = _heap_graph(9)
    graph.similarity_edges = _edges(
        [(8, index, 0.9) for index in range(1, 8)], "similarity"
    )
    context = CocoreObjectiveContext(
        graph, "cooccurrence", 1.0, similarity_threshold=0.8
    )
    coverage = build_max_coverage_seed(context, budget=2)

    class ScheduledSelector(RandomMultiBranchSelector):
        def __init__(self) -> None:
            super().__init__(context, seed=5)
            self.samples = iter(range(8))

        def _sample(self, rng, *, fixed, active, size):
            del rng, fixed, active
            assert size == 1
            return (next(self.samples),)

    result = ScheduledSelector().select(2, initial_indices=coverage.selected_indices)

    assert result.selected_indices == (8, 0)


def test_random_multibranch_ranks_recombination_clips_by_branch_frequency() -> None:
    context = CocoreObjectiveContext(
        _heap_graph(5), "cooccurrence", 1.0, similarity_threshold=0.8
    )
    selector = RandomMultiBranchSelector(context, seed=13)

    ranked = selector._rank_indices(
        [0, 1, 2, 3],
        Counter({0: 8, 1: 4, 2: 4, 3: 1}),
        np.random.default_rng(13),
        limit=3,
    )

    assert ranked[0] == 0
    assert set(ranked[1:]) == {1, 2}


def test_random_multibranch_recombines_first_at_20_then_every_10_rounds() -> None:
    graph = _heap_graph(340)
    context = CocoreObjectiveContext(
        graph, "cooccurrence", 1.0, similarity_threshold=0.8
    )
    coverage = build_max_coverage_seed(context, budget=326)

    result = RandomMultiBranchSelector(context, seed=23).select(
        326, initial_indices=coverage.selected_indices
    )

    assert len(result.selected_indices) == 326
    assert len(set(result.selected_indices)) == 326
    assert result.rounds == 33
    assert result.evaluated_branches == 8 + 32 * 32
    assert result.recombinations == 2
    assert result.committed_clips == 200
    assert result.final_active_clips == 125
    commit_steps = [
        step
        for phase, step in zip(result.selection_phases, result.selection_steps, strict=True)
        if phase == "branch_commit"
    ]
    assert commit_steps == [20] * 100 + [30] * 100
    assert result.selection_phases.count("branch_final") == 125
    assert result.selection_steps[-125:] == (33,) * 125
    replay = CocoreObjectiveContext(
        graph, "cooccurrence", 1.0, similarity_threshold=0.8
    ).state_from_indices(result.selected_indices)
    assert result.objective_value == pytest.approx(replay.score)
    assert result.relation == pytest.approx(replay.relation)
    assert result.redundancy == pytest.approx(replay.redundancy)


def test_lazy_heap_selects_current_top_then_refreshes_the_next_stale_top() -> None:
    graph = _heap_graph()
    context = CocoreObjectiveContext(graph, "cooccurrence", 1.0, similarity_threshold=0.8)
    seed = build_max_coverage_seed(context, budget=3)
    selector = LazyHeapSelector(context, max_refreshes=100)

    result = selector.select(3, initial_indices=seed.selected_indices)

    assert result.selected_indices == (13, 0, 1)
    assert result.selection_phases == ("coverage_seed", "heap", "heap")
    assert result.selection_steps == (0, 1, 2)
    assert result.heap_refreshes == (0, 0, 1)
    assert result.initial_heap_size == 13
    assert result.total_refreshes == 1
    assert result.capped_selections == 0
    assert result.max_refreshes_observed == 1


def test_lazy_heap_returns_seed_without_building_a_heap_when_budget_is_full() -> None:
    graph = _heap_graph()
    context = CocoreObjectiveContext(graph, "cooccurrence", 1.0, similarity_threshold=0.8)
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
    relation: float = 0.0
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
        state.relation = state.score


class _RecordingScriptedContext(_ScriptedContext):
    def __init__(self, sample_ids: list[str], gain) -> None:
        super().__init__(sample_ids, gain)
        self.marginal_gain_candidates: dict[int, list[int]] = {}

    def marginal_gain(self, state: _ScriptedState, index: int) -> float:
        selected_count = int(state.selected_mask.sum())
        self.marginal_gain_candidates.setdefault(selected_count, []).append(index)
        return super().marginal_gain(state, index)


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


def test_lazy_heap_fully_rebuilds_before_heap_steps_8_16_and_32() -> None:
    candidate_count = 42
    context = _RecordingScriptedContext(
        [f"node-{index:02d}" for index in range(candidate_count)],
        lambda selected_count, index: float(candidate_count - index),
    )
    initial = [0, 1, 2]

    result = LazyHeapSelector(context, max_refreshes=3).select(
        36,
        initial_indices=initial,
    )

    assert result.selected_indices == tuple(range(36))
    assert len(context.marginal_gain_candidates[8]) == 1
    assert len(context.marginal_gain_candidates[10]) == 32
    assert len(context.marginal_gain_candidates[18]) == 24
    assert len(context.marginal_gain_candidates[34]) == 8
    assert result.heap_refreshes[9] == 1
    assert result.heap_refreshes[10] == 0
    assert result.heap_refreshes[11] == 1
    assert result.heap_refreshes[18] == 0
    assert result.heap_refreshes[34] == 0


def test_lazy_heap_step_8_rebuild_selects_the_new_global_maximum() -> None:
    def gain(selected_count: int, index: int) -> float:
        if selected_count >= 8 and index == 9:
            return 200.0
        if index <= 7:
            return float(100 - index)
        return {8: 20.0, 9: 0.0, 10: 10.0, 11: 5.0}[index]

    selector = LazyHeapSelector(
        _ScriptedContext([f"node-{index:02d}" for index in range(12)], gain),
        max_refreshes=3,
    )

    result = selector.select(9, initial_indices=[0])

    assert result.selected_indices == (0, 1, 2, 3, 4, 5, 6, 7, 9)
    assert result.score_deltas[-1] == pytest.approx(200.0)
    assert result.heap_refreshes[-1] == 0


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
    context = CocoreObjectiveContext(graph, "cooccurrence", 1.0, similarity_threshold=0.8)
    seed = build_max_coverage_seed(context, budget=3)

    result = LazyHeapSelector(context).select(3, initial_indices=seed.selected_indices)

    assert result.selected_indices == (5, 0, 1)
    assert graph.task_indices[list(result.selected_indices)].tolist() == [1, 1, 1]


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
    context = _CountingObjectiveContext(
        graph, "cooccurrence", 1.0, similarity_threshold=0.8
    )
    budget = 10
    seed = build_max_coverage_seed(context, budget=budget)
    selector = LazyHeapSelector(context, max_refreshes=3)

    result = selector.select(budget, initial_indices=seed.selected_indices)

    heap_selections = budget - len(seed.selected_indices)
    remaining_candidates = len(graph.sample_ids) - len(seed.selected_indices)
    full_refresh_candidates = sum(
        remaining_candidates - (step - 1)
        for step in range(8, heap_selections + 1)
        if (step & (step - 1)) == 0
    )
    calls_excluding_seed = context.marginal_gain_calls - len(seed.selected_indices)
    assert calls_excluding_seed <= (
        remaining_candidates + 3 * heap_selections + full_refresh_candidates
    )
    recomputed = CocoreObjectiveContext(
        graph, "cooccurrence", 1.0, similarity_threshold=0.8
    ).state_from_indices(result.selected_indices)
    assert result.objective_value == pytest.approx(recomputed.score)
    assert result.relation == pytest.approx(recomputed.relation)
    assert result.redundancy == pytest.approx(recomputed.redundancy)
