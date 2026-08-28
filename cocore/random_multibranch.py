"""Seeded random multi-branch selection with periodic recombination."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .faiss_penalty import FaissFlatPenaltyIndex, FaissFlatPenaltySpace
from .objective import CocoreObjectiveContext, CocoreObjectiveUpdateState


BRANCH_COUNT = 8
CHILDREN_PER_BRANCH = 4
BATCH_SIZE = 10
FIRST_RECOMBINATION_ROUND = 20
RECOMBINATION_INTERVAL = 10
COMMIT_SIZE = 100
RETAINED_SIZE = 100


@dataclass(frozen=True)
class RandomMultiBranchSelectionResult:
    selected_indices: tuple[int, ...]
    score_deltas: tuple[float, ...]
    selection_phases: tuple[str, ...]
    selection_steps: tuple[int, ...]
    objective_value: float
    relation: float
    redundancy: float
    rounds: int
    evaluated_branches: int
    recombinations: int
    committed_clips: int
    final_active_clips: int
    round_runtime_seconds: tuple[float, ...] = field(compare=False)
    recombination_runtime_seconds: tuple[tuple[int, float], ...] = field(compare=False)


@dataclass(frozen=True)
class _Branch:
    update_state: CocoreObjectiveUpdateState
    serial: int

    @property
    def active_indices(self) -> tuple[int, ...]:
        return self.update_state.selected_indices


class RandomMultiBranchSelector:
    """Search sparse branch updates over one shared main objective state."""

    def __init__(self, context: CocoreObjectiveContext, *, seed: int = 42) -> None:
        self.context = context
        self.seed = int(seed)
        self.penalty_space = FaissFlatPenaltySpace(
            context.graph.embeddings,
            context.graph.reliability,
            similarity_threshold=context.similarity_threshold,
            redundancy_normalizer=context.redundancy_normalizer,
            epsilon=context.epsilon,
        )

    @staticmethod
    def _finite_score(state: CocoreObjectiveUpdateState) -> float:
        score = float(state.score)
        if not math.isfinite(score):
            raise ValueError("random multibranch produced a non-finite branch score")
        return score

    def _sample(
        self,
        rng: np.random.Generator,
        *,
        fixed: set[int],
        active: tuple[int, ...],
        size: int,
    ) -> tuple[int, ...]:
        candidate_count = len(self.context.graph.sample_ids)
        active_set = set(active)
        available_count = candidate_count - len(fixed) - len(active_set)
        if size > available_count:
            raise ValueError("not enough unselected candidates to fill a branch batch")
        if size == 0:
            return ()
        if available_count <= 1_024 or available_count * 4 < candidate_count:
            available = [
                index
                for index in range(candidate_count)
                if index not in fixed and index not in active_set
            ]
            chosen = rng.choice(
                np.asarray(available, dtype=np.int64), size=size, replace=False
            )
            return tuple(int(index) for index in chosen.tolist())

        chosen: list[int] = []
        chosen_set: set[int] = set()
        while len(chosen) < size:
            draw_count = max(16, 2 * (size - len(chosen)))
            draws = rng.integers(0, candidate_count, size=draw_count)
            for raw_index in draws:
                index = int(raw_index)
                if (
                    index in fixed
                    or index in active_set
                    or index in chosen_set
                ):
                    continue
                chosen.append(index)
                chosen_set.add(index)
                if len(chosen) == size:
                    break
        return tuple(chosen)

    def _extend_update_state(
        self,
        update_state: CocoreObjectiveUpdateState,
        indices: tuple[int, ...],
        *,
        main_penalty_index: FaissFlatPenaltyIndex,
    ) -> CocoreObjectiveUpdateState:
        active_penalty_index = self.penalty_space.create_index(
            update_state.selected_indices
        )
        redundancy_deltas: list[float] = []
        for index in indices:
            redundancy_deltas.append(
                self.penalty_space.penalty(
                    index,
                    (main_penalty_index, active_penalty_index),
                )
            )
            active_penalty_index.add((index,))
        extended = self.context.extend_update_state(
            update_state,
            indices,
            redundancy_deltas=redundancy_deltas,
        )
        self._finite_score(extended)
        return extended

    def _rank_branches(
        self,
        branches: list[_Branch],
        rng: np.random.Generator,
        *,
        limit: int,
    ) -> list[_Branch]:
        tie_breakers = {branch.serial: float(rng.random()) for branch in branches}
        return sorted(
            branches,
            key=lambda branch: (
                -self._finite_score(branch.update_state),
                tie_breakers[branch.serial],
                branch.serial,
            ),
        )[:limit]

    def _rank_committed_indices(
        self,
        indices: tuple[int, ...] | list[int],
        counts: Counter[int],
        rng: np.random.Generator,
        *,
        limit: int,
    ) -> tuple[int, ...]:
        tie_breakers = {index: float(rng.random()) for index in indices}
        ranked = sorted(
            indices,
            key=lambda index: (
                -counts[index],
                tie_breakers[index],
                self.context.graph.sample_ids[index],
            ),
        )
        return tuple(ranked[:limit])

    def _rank_retained_indices(
        self,
        indices: tuple[int, ...] | list[int],
        counts: Counter[int],
        *,
        limit: int,
    ) -> tuple[int, ...]:
        ranked = sorted(
            indices,
            key=lambda index: (
                -counts[index],
                -float(self.context.graph.reliability[index]),
                self.context.graph.sample_ids[index],
            ),
        )
        return tuple(ranked[:limit])

    @staticmethod
    def _is_recombination_round(round_number: int) -> bool:
        return round_number >= FIRST_RECOMBINATION_ROUND and (
            round_number - FIRST_RECOMBINATION_ROUND
        ) % RECOMBINATION_INTERVAL == 0

    def _result(
        self,
        selected: list[int],
        phases: list[str],
        steps: list[int],
        *,
        active_relation_deltas: tuple[float, ...],
        active_redundancy_deltas: tuple[float, ...],
        redundancy: float,
        rounds: int,
        evaluated_branches: int,
        recombinations: int,
        committed_clips: int,
        final_active_clips: int,
        round_runtime_seconds: list[float],
        recombination_runtime_seconds: list[tuple[int, float]],
    ) -> RandomMultiBranchSelectionResult:
        if len(round_runtime_seconds) != rounds:
            raise ValueError("round timing count must match completed rounds")
        if len(recombination_runtime_seconds) != recombinations:
            raise ValueError("recombination timing count must match completed recombinations")
        if len(active_relation_deltas) != final_active_clips:
            raise ValueError("active relation deltas must match final active clips")
        if len(active_redundancy_deltas) != final_active_clips:
            raise ValueError("active redundancy deltas must match final active clips")
        fixed_count = len(selected) - final_active_clips
        if self.context.relation_type == "sequence":
            fixed_relation_deltas = [0.0] * fixed_count
        else:
            fixed_relation_deltas = []
            fixed_state = self.context.empty_state()
            for index in selected[:fixed_count]:
                previous_relation = float(fixed_state.relation)
                self.context.add_candidate(fixed_state, index)
                fixed_relation_deltas.append(
                    float(fixed_state.relation) - previous_relation
                )
        relation_deltas = tuple(fixed_relation_deltas) + active_relation_deltas
        redundancy_deltas = (0.0,) * fixed_count + active_redundancy_deltas
        if not math.isclose(
            math.fsum(redundancy_deltas),
            float(redundancy),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("active redundancy deltas do not match branch redundancy")
        relation = math.fsum(relation_deltas)
        gains = [
            self.context.relation_weight * relation_delta - redundancy_delta
            for relation_delta, redundancy_delta in zip(
                relation_deltas,
                redundancy_deltas,
                strict=True,
            )
        ]
        for index, gain in zip(selected, gains, strict=True):
            if not math.isfinite(gain):
                raise ValueError(f"candidate {index} has a non-finite marginal gain")
        score = self.context.relation_weight * relation - float(redundancy)
        if not math.isclose(
            math.fsum(gains),
            score,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("incremental score deltas do not match branch objective")
        return RandomMultiBranchSelectionResult(
            selected_indices=tuple(selected),
            score_deltas=tuple(gains),
            selection_phases=tuple(phases),
            selection_steps=tuple(steps),
            objective_value=float(score),
            relation=float(relation),
            redundancy=float(redundancy),
            rounds=rounds,
            evaluated_branches=evaluated_branches,
            recombinations=recombinations,
            committed_clips=committed_clips,
            final_active_clips=final_active_clips,
            round_runtime_seconds=tuple(round_runtime_seconds),
            recombination_runtime_seconds=tuple(recombination_runtime_seconds),
        )

    def select(
        self,
        budget: int,
        *,
        initial_indices: list[int] | tuple[int, ...],
    ) -> RandomMultiBranchSelectionResult:
        candidate_count = len(self.context.graph.sample_ids)
        initial = tuple(int(index) for index in initial_indices)
        if len(initial) != len(set(initial)):
            raise ValueError("initial_indices cannot contain duplicates")
        if any(index < 0 or index >= candidate_count for index in initial):
            raise ValueError("initial_indices must contain in-range values")
        if not 0 < len(initial) <= budget <= candidate_count:
            raise ValueError("budget must contain a non-empty initial selection")

        fixed = list(initial)
        fixed_set = set(initial)
        phases = ["coverage_seed"] * len(initial)
        steps = [0] * len(initial)
        rng = np.random.default_rng(self.seed)
        round_runtime_seconds: list[float] = []
        recombination_runtime_seconds: list[tuple[int, float]] = []
        if len(fixed) == budget:
            return self._result(
                fixed,
                phases,
                steps,
                active_relation_deltas=(),
                active_redundancy_deltas=(),
                redundancy=0.0,
                rounds=0,
                evaluated_branches=0,
                recombinations=0,
                committed_clips=0,
                final_active_clips=0,
                round_runtime_seconds=round_runtime_seconds,
                recombination_runtime_seconds=recombination_runtime_seconds,
            )

        main_state = (
            self.context.extend_sequence_main_state(self.context.empty_state(), fixed)
            if self.context.relation_type == "sequence"
            else self.context.state_from_indices(fixed)
        )
        root_update = self.context.empty_update_state(main_state)
        main_penalty_index = self.penalty_space.create_index(fixed)
        initial_batch_size = min(BATCH_SIZE, budget - len(fixed))
        branches: list[_Branch] = []
        next_serial = 0
        round_started = time.perf_counter()
        for _ in range(BRANCH_COUNT):
            active = self._sample(
                rng,
                fixed=fixed_set,
                active=(),
                size=initial_batch_size,
            )
            update_state = self._extend_update_state(
                root_update,
                active,
                main_penalty_index=main_penalty_index,
            )
            branches.append(_Branch(update_state, next_serial))
            next_serial += 1
        round_runtime_seconds.append(time.perf_counter() - round_started)

        rounds = 1
        evaluated_branches = BRANCH_COUNT
        recombinations = 0
        committed_clips = 0

        while True:
            active_size = len(branches[0].active_indices)
            if len(fixed) + active_size == budget:
                winner = self._rank_branches(branches, rng, limit=1)[0]
                selected = fixed + list(winner.active_indices)
                return self._result(
                    selected,
                    phases + ["branch_final"] * active_size,
                    steps + [rounds] * active_size,
                    active_relation_deltas=winner.update_state.relation_deltas,
                    active_redundancy_deltas=winner.update_state.redundancy_deltas,
                    redundancy=winner.update_state.redundancy,
                    rounds=rounds,
                    evaluated_branches=evaluated_branches,
                    recombinations=recombinations,
                    committed_clips=committed_clips,
                    final_active_clips=active_size,
                    round_runtime_seconds=round_runtime_seconds,
                    recombination_runtime_seconds=recombination_runtime_seconds,
                )

            batch_size = min(BATCH_SIZE, budget - len(fixed) - active_size)
            children: list[_Branch] = []
            round_started = time.perf_counter()
            for branch in branches:
                for _ in range(CHILDREN_PER_BRANCH):
                    added = self._sample(
                        rng,
                        fixed=fixed_set,
                        active=branch.active_indices,
                        size=batch_size,
                    )
                    update_state = self._extend_update_state(
                        branch.update_state,
                        added,
                        main_penalty_index=main_penalty_index,
                    )
                    children.append(_Branch(update_state, next_serial))
                    next_serial += 1
            rounds += 1
            evaluated_branches += len(children)
            branches = self._rank_branches(children, rng, limit=BRANCH_COUNT)
            round_runtime_seconds.append(time.perf_counter() - round_started)

            active_size = len(branches[0].active_indices)
            if len(fixed) + active_size == budget:
                continue
            if not self._is_recombination_round(rounds):
                continue

            recombination_started = time.perf_counter()
            counts = Counter(
                index for branch in branches for index in branch.active_indices
            )
            winner = branches[0]
            committed = self._rank_committed_indices(
                winner.active_indices,
                counts,
                rng,
                limit=COMMIT_SIZE,
            )
            fixed.extend(committed)
            fixed_set.update(committed)
            main_penalty_index.add(committed)
            phases.extend(["branch_commit"] * len(committed))
            steps.extend([rounds] * len(committed))
            committed_clips += len(committed)
            recombinations += 1

            main_state = (
                self.context.extend_sequence_main_state(main_state, committed)
                if self.context.relation_type == "sequence"
                else self.context.extend_state(main_state, committed)
            )
            root_update = self.context.empty_update_state(main_state)
            recombined: list[_Branch] = []
            for branch in branches:
                remaining = [
                    index for index in branch.active_indices if index not in fixed_set
                ]
                retained = self._rank_retained_indices(
                    remaining,
                    counts,
                    limit=RETAINED_SIZE,
                )
                if len(retained) != RETAINED_SIZE:
                    raise ValueError("recombination could not retain 100 active clips")
                update_state = self._extend_update_state(
                    root_update,
                    retained,
                    main_penalty_index=main_penalty_index,
                )
                recombined.append(_Branch(update_state, next_serial))
                next_serial += 1
            branches = recombined
            recombination_runtime_seconds.append(
                (rounds, time.perf_counter() - recombination_started)
            )
