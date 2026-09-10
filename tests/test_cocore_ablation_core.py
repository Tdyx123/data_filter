from __future__ import annotations

import numpy as np
import pytest

from cocore.objective import CocoreObjectiveContext, recompute_objective
from cocore.random_multibranch import RandomMultiBranchSelector
from cocore_ablation.reliability import fuse_reliability
from cocore_ablation.selection import SeededRandomSelector, initial_selection
from tests.test_cocore_core import _graph


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (("support_old", "action_jump"), [0.3, 0.8]),
        (("support_old",), [0.25, 0.8]),
        (("action_jump",), [0.36, 0.8]),
        ((), [1.0, 1.0]),
    ],
)
def test_reliability_fusion_uses_cocore_exponents(
    metrics: tuple[str, ...], expected: list[float]
) -> None:
    actual = fuse_reliability(
        np.asarray([0.25, 0.8], dtype=np.float32),
        np.asarray([0.36, 0.8], dtype=np.float32),
        metrics,
        min_reliability=0.05,
    )

    assert actual.tolist() == pytest.approx(expected)


def test_redundancy_weight_scales_objective_without_changing_raw_penalty() -> None:
    context = CocoreObjectiveContext(
        _graph(),
        "cooccurrence",
        relation_weight=2.0,
        redundancy_weight=0.25,
        similarity_threshold=0.8,
    )
    state = context.state_from_indices([0, 1, 2])

    assert state.relation == pytest.approx(2.0**-0.5)
    assert state.redundancy == pytest.approx(1.0)
    assert state.score == pytest.approx(2.0**0.5 - 0.25)
    replayed = recompute_objective(
        [0, 1, 2],
        _graph(),
        relation_type="cooccurrence",
        relation_weight=2.0,
        redundancy_weight=0.25,
        similarity_threshold=0.8,
    )
    assert replayed.score == pytest.approx(state.score)


def test_zero_objective_weights_remove_the_corresponding_marginal_term() -> None:
    graph = _graph()
    no_relation = CocoreObjectiveContext(
        graph,
        "cooccurrence",
        relation_weight=0.0,
        redundancy_weight=0.25,
        similarity_threshold=0.8,
    )
    no_redundancy = CocoreObjectiveContext(
        graph,
        "cooccurrence",
        relation_weight=2.0,
        redundancy_weight=0.0,
        similarity_threshold=0.8,
    )
    no_relation_state = no_relation.state_from_indices([0, 1])
    no_redundancy_state = no_redundancy.state_from_indices([0, 1])
    relation_delta = (
        no_redundancy.state_from_indices([0, 1, 2]).relation
        - no_redundancy_state.relation
    )
    redundancy_delta = (
        no_relation.state_from_indices([0, 1, 2]).redundancy
        - no_relation_state.redundancy
    )

    assert no_relation.marginal_gain(no_relation_state, 2) == pytest.approx(
        -0.25 * redundancy_delta
    )
    assert no_redundancy.marginal_gain(no_redundancy_state, 2) == pytest.approx(
        2.0 * relation_delta
    )


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_objective_rejects_invalid_redundancy_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="redundancy_weight"):
        CocoreObjectiveContext(
            _graph(),
            "cooccurrence",
            redundancy_weight=weight,
            similarity_threshold=0.8,
        )


def test_random_selector_samples_without_replacement_and_replays_objective() -> None:
    context = CocoreObjectiveContext(
        _graph(),
        "cooccurrence",
        relation_weight=2.0,
        redundancy_weight=0.25,
        similarity_threshold=0.8,
    )

    result = SeededRandomSelector(context, seed=7).select(2, initial_indices=())

    assert result.selected_indices == (1, 2)
    assert len(set(result.selected_indices)) == 2
    assert result.selection_phases == ("random", "random")
    replayed = context.state_from_indices(result.selected_indices)
    assert result.relation == pytest.approx(replayed.relation)
    assert result.redundancy == pytest.approx(replayed.redundancy)
    assert result.objective_value == pytest.approx(replayed.score)
    assert sum(result.score_deltas) == pytest.approx(result.objective_value)


def test_random_selector_is_reproducible_and_seed_sensitive() -> None:
    context = CocoreObjectiveContext(
        _graph(), "cooccurrence", similarity_threshold=0.8
    )

    first = SeededRandomSelector(context, seed=1).select(2, initial_indices=())
    repeated = SeededRandomSelector(context, seed=1).select(2, initial_indices=())
    changed = SeededRandomSelector(context, seed=7).select(2, initial_indices=())

    assert first.selected_indices == repeated.selected_indices
    assert changed.selected_indices != first.selected_indices


def test_random_selector_preserves_coverage_seed_before_random_fill() -> None:
    context = CocoreObjectiveContext(
        _graph(), "cooccurrence", similarity_threshold=0.8
    )

    result = SeededRandomSelector(context, seed=7).select(3, initial_indices=(0,))

    assert result.selected_indices[0] == 0
    assert result.selection_phases == ("coverage_seed", "random", "random")


def test_initial_selection_can_disable_maximum_coverage_seed() -> None:
    context = CocoreObjectiveContext(
        _graph(), "cooccurrence", similarity_threshold=0.8
    )

    assert initial_selection(context, budget=3, use_coverage_seed=False) == ()
    assert initial_selection(context, budget=3, use_coverage_seed=True)


def test_random_multibranch_accepts_empty_initial_selection() -> None:
    context = CocoreObjectiveContext(
        _graph(), "cooccurrence", similarity_threshold=0.8
    )

    result = RandomMultiBranchSelector(context, seed=7).select(
        2, initial_indices=()
    )

    assert len(result.selected_indices) == 2
    assert len(set(result.selected_indices)) == 2
