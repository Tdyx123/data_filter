"""Coverage seeding, residual quotas, and deterministic sparse beam rollout."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from relcore.selection.quota import allocate_task_quotas
from relcore.utils.io import stable_hash

from .objective import CocoreObjectiveContext


@dataclass(frozen=True)
class CoverageSeed:
    selected_indices: tuple[int, ...]
    target_coverage: np.ndarray
    achieved_coverage: np.ndarray


@dataclass(frozen=True)
class CandidatePoolConfig:
    global_candidates: int = 256
    prototype_candidates: int = 128
    similarity_candidates: int = 128
    random_candidates: int = 128

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.global_candidates,
                self.prototype_candidates,
                self.similarity_candidates,
                self.random_candidates,
            )
        ):
            raise ValueError("candidate pool sizes cannot be negative")


@dataclass(frozen=True)
class BeamLayerStats:
    depth: int
    generated: int
    unique: int
    retained: int


@dataclass(frozen=True)
class BeamSelectionResult:
    selected_indices: tuple[int, ...]
    score_deltas: tuple[float, ...]
    selection_phases: tuple[str, ...]
    rollout_depths: tuple[int, ...]
    objective_value: float
    cooccurrence: float
    redundancy: float
    layer_stats: tuple[BeamLayerStats, ...]
    final_beam_scores: tuple[float, ...]


@dataclass
class _BeamPath:
    selected_indices: tuple[int, ...]
    score_deltas: tuple[float, ...]
    selection_phases: tuple[str, ...]
    rollout_depths: tuple[int, ...]
    rollout_task_counts: dict[int, int]
    state: object


def build_max_coverage_seed(
    context: CocoreObjectiveContext,
    *,
    budget: int,
) -> CoverageSeed:
    if budget <= 0 or budget > len(context.graph.sample_ids):
        raise ValueError("selection budget must be within candidate count")
    target = context.prototype_mass.max(axis=0)
    selected: list[int] = []
    selected_set: set[int] = set()
    for prototype, maximum in enumerate(target):
        if maximum <= 0.0:
            continue
        matches = np.flatnonzero(context.prototype_mass[:, prototype] == maximum)
        winner = min(matches.tolist(), key=lambda index: context.graph.sample_ids[index])
        if winner not in selected_set:
            selected.append(winner)
            selected_set.add(winner)
    if len(selected) > budget:
        raise ValueError(
            f"coverage seed exceeds selection budget; minimum required budget is {len(selected)}"
        )
    achieved = (
        context.prototype_mass[np.asarray(selected, dtype=np.int64)].max(axis=0)
        if selected
        else np.zeros_like(target)
    )
    return CoverageSeed(tuple(selected), target.copy(), achieved)


def allocate_residual_task_quotas(
    task_indices: np.ndarray,
    *,
    selected_indices: list[int] | tuple[int, ...],
    budget: int,
) -> dict[int, int]:
    tasks = np.asarray(task_indices, dtype=np.int64)
    if tasks.ndim != 1 or len(tasks) == 0:
        raise ValueError("task_indices must be a non-empty vector")
    selected = np.asarray(selected_indices, dtype=np.int64)
    if len(np.unique(selected)) != len(selected) or np.any((selected < 0) | (selected >= len(tasks))):
        raise ValueError("selected_indices must contain unique in-range values")
    if budget < len(selected) or budget > len(tasks):
        raise ValueError("selection budget must contain the seed and fit candidate count")
    remaining_budget = budget - len(selected)
    remaining_mask = np.ones(len(tasks), dtype=bool)
    remaining_mask[selected] = False
    remaining_tasks = tasks[remaining_mask]
    if remaining_budget == 0:
        return {int(task): 0 for task in sorted(set(remaining_tasks.tolist()))}
    return allocate_task_quotas(
        remaining_tasks,
        budget=remaining_budget,
        minimum_per_task=0,
    )


class BeamRolloutSelector:
    BEAM_WIDTH = 8
    ROOT_ROLLOUTS = 8
    NODE_ROLLOUTS = 4

    def __init__(
        self,
        context: CocoreObjectiveContext,
        residual_task_quotas: Mapping[int, int],
        *,
        seed: int = 42,
        pool_config: CandidatePoolConfig = CandidatePoolConfig(),
    ) -> None:
        self.context = context
        self.residual_task_quotas = {
            int(task): int(quota) for task, quota in residual_task_quotas.items()
        }
        if any(quota < 0 for quota in self.residual_task_quotas.values()):
            raise ValueError("residual task quotas cannot be negative")
        self.seed = int(seed)
        self.pool_config = pool_config
        graph = context.graph
        self._static_potential = np.einsum(
            "ni,ij,nj->n",
            context.prototype_mass,
            context.cooccurrence_matrix,
            context.prototype_mass,
        )
        self._global_order = sorted(
            range(len(graph.sample_ids)),
            key=lambda index: (
                -float(self._static_potential[index]),
                -float(graph.reliability[index]),
                graph.sample_ids[index],
            ),
        )
        self._prototype_members: list[list[int]] = [
            [] for _ in range(context.prototype_mass.shape[1])
        ]
        for index in range(len(graph.sample_ids)):
            for prototype in np.flatnonzero(context.prototype_mass[index] > 0.0):
                self._prototype_members[int(prototype)].append(index)
        for prototype, members in enumerate(self._prototype_members):
            members.sort(
                key=lambda index: (
                    -float(context.prototype_mass[index, prototype]),
                    graph.sample_ids[index],
                )
            )
        self._similarity_adjacency: list[list[tuple[int, float]]] = [
            [] for _ in graph.sample_ids
        ]
        for source, target, similarity in zip(
            graph.similarity_edges.source,
            graph.similarity_edges.target,
            graph.similarity_edges.weight,
            strict=True,
        ):
            left, right, value = int(source), int(target), float(similarity)
            self._similarity_adjacency[left].append((right, value))
            self._similarity_adjacency[right].append((left, value))
        self._task_members: dict[int, list[int]] = {}
        for task in sorted({int(value) for value in graph.task_indices}):
            members = np.flatnonzero(graph.task_indices == task).tolist()
            members.sort(
                key=lambda index: (-float(graph.reliability[index]), graph.sample_ids[index])
            )
            self._task_members[task] = members

    def _eligible(
        self,
        state,
        rollout_task_counts: Mapping[int, int],
        candidate: int,
    ) -> bool:
        if state.selected_mask[candidate]:
            return False
        task = int(self.context.graph.task_indices[candidate])
        return int(rollout_task_counts.get(task, 0)) < self.residual_task_quotas.get(task, 0)

    def candidate_pool(
        self,
        state,
        rollout_task_counts: Mapping[int, int],
    ) -> tuple[int, ...]:
        graph = self.context.graph
        selected: set[int] = set()

        global_count = 0
        for index in self._global_order:
            if not self._eligible(state, rollout_task_counts, index):
                continue
            selected.add(index)
            global_count += 1
            if global_count >= self.pool_config.global_candidates:
                break

        prototype_limit = self.pool_config.prototype_candidates
        if prototype_limit:
            direction = (self.context.cooccurrence_matrix + self.context.cooccurrence_matrix.T) @ (
                state.prototype_mass
            )
            prototypes = sorted(
                range(len(direction)), key=lambda prototype: (-float(direction[prototype]), prototype)
            )
            positions = {prototype: 0 for prototype in prototypes}
            added = 0
            while added < prototype_limit:
                progressed = False
                for prototype in prototypes:
                    members = self._prototype_members[prototype]
                    position = positions[prototype]
                    while position < len(members) and not self._eligible(
                        state, rollout_task_counts, members[position]
                    ):
                        position += 1
                    positions[prototype] = position + 1
                    if position >= len(members):
                        continue
                    before = len(selected)
                    selected.add(members[position])
                    added += len(selected) - before
                    progressed = True
                    if added >= prototype_limit:
                        break
                if not progressed:
                    break

        similarity: dict[int, float] = {}
        for source in np.flatnonzero(state.selected_mask):
            for candidate, value in self._similarity_adjacency[int(source)]:
                if self._eligible(state, rollout_task_counts, candidate):
                    similarity[candidate] = max(similarity.get(candidate, 0.0), value)
        selected.update(
            sorted(
                similarity,
                key=lambda index: (-similarity[index], graph.sample_ids[index]),
            )[: self.pool_config.similarity_candidates]
        )

        eligible = np.asarray(
            [
                index
                for index in range(len(graph.sample_ids))
                if self._eligible(state, rollout_task_counts, index)
            ],
            dtype=np.int64,
        )
        sample_size = min(self.pool_config.random_candidates, len(eligible))
        if sample_size:
            selected_ids = sorted(
                graph.sample_ids[index] for index in np.flatnonzero(state.selected_mask)
            )
            digest = stable_hash({"seed": self.seed, "selected": selected_ids})
            rng = np.random.default_rng(int(digest[:16], 16))
            weights = graph.reliability[eligible].astype(np.float64)
            weights /= weights.sum()
            selected.update(
                int(index)
                for index in rng.choice(
                    eligible,
                    size=sample_size,
                    replace=False,
                    p=weights,
                )
            )

        for task, quota in self.residual_task_quotas.items():
            if int(rollout_task_counts.get(task, 0)) >= quota:
                continue
            representative = next(
                (
                    index
                    for index in self._task_members.get(task, ())
                    if self._eligible(state, rollout_task_counts, index)
                ),
                None,
            )
            if representative is not None:
                selected.add(representative)

        if not selected and len(eligible):
            selected.add(
                min(
                    eligible.tolist(),
                    key=lambda index: (-float(graph.reliability[index]), graph.sample_ids[index]),
                )
            )
        return tuple(sorted(selected, key=lambda index: graph.sample_ids[index]))

    def _path_rank(
        self, path: _BeamPath
    ) -> tuple[float, tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
        graph = self.context.graph
        canonical = tuple(sorted(graph.sample_ids[index] for index in path.selected_indices))
        prefix_scores = tuple(-float(value) for value in np.cumsum(path.score_deltas))
        order = tuple(graph.sample_ids[index] for index in path.selected_indices)
        return (-float(path.state.score), canonical, prefix_scores, order)

    def _expand(self, path: _BeamPath, *, count: int, depth: int) -> list[_BeamPath]:
        candidates = self.candidate_pool(path.state, path.rollout_task_counts)
        ranked = sorted(
            candidates,
            key=lambda index: (
                -float(path.state.score + self.context.marginal_gain(path.state, index)),
                self.context.graph.sample_ids[index],
            ),
        )[:count]
        children: list[_BeamPath] = []
        for candidate in ranked:
            state = self.context.clone_state(path.state)
            gain = self.context.marginal_gain(state, candidate)
            self.context.add_candidate(state, candidate)
            counts = dict(path.rollout_task_counts)
            task = int(self.context.graph.task_indices[candidate])
            counts[task] = counts.get(task, 0) + 1
            children.append(
                _BeamPath(
                    selected_indices=path.selected_indices + (candidate,),
                    score_deltas=path.score_deltas + (float(gain),),
                    selection_phases=path.selection_phases + ("rollout",),
                    rollout_depths=path.rollout_depths + (depth,),
                    rollout_task_counts=counts,
                    state=state,
                )
            )
        return children

    def select(
        self,
        budget: int,
        *,
        initial_indices: list[int] | tuple[int, ...],
    ) -> BeamSelectionResult:
        initial = tuple(int(index) for index in initial_indices)
        if len(initial) != len(set(initial)):
            raise ValueError("initial_indices cannot contain duplicates")
        if not 0 < len(initial) <= budget <= len(self.context.graph.sample_ids):
            raise ValueError("budget must contain a non-empty initial selection")
        if sum(self.residual_task_quotas.values()) != budget - len(initial):
            raise ValueError("residual task quotas must sum to remaining budget")
        initial_state = self.context.empty_state()
        initial_gains: list[float] = []
        for index in initial:
            gain = self.context.marginal_gain(initial_state, index)
            self.context.add_candidate(initial_state, index)
            initial_gains.append(float(gain))
        beam = [
            _BeamPath(
                selected_indices=initial,
                score_deltas=tuple(initial_gains),
                selection_phases=("coverage_seed",) * len(initial),
                rollout_depths=(0,) * len(initial),
                rollout_task_counts={task: 0 for task in self.residual_task_quotas},
                state=initial_state,
            )
        ]
        layer_stats: list[BeamLayerStats] = []
        depth = 1
        while len(beam[0].selected_indices) < budget:
            rollout_count = self.ROOT_ROLLOUTS if depth == 1 else self.NODE_ROLLOUTS
            generated = [
                child
                for path in beam
                for child in self._expand(path, count=rollout_count, depth=depth)
            ]
            if not generated:
                raise ValueError("no quota-feasible rollout candidate remains")
            unique: dict[tuple[int, ...], _BeamPath] = {}
            for path in generated:
                key = tuple(sorted(path.selected_indices))
                previous = unique.get(key)
                if previous is None or self._path_rank(path) < self._path_rank(previous):
                    unique[key] = path
            beam = sorted(unique.values(), key=self._path_rank)[: self.BEAM_WIDTH]
            layer_stats.append(
                BeamLayerStats(depth, len(generated), len(unique), len(beam))
            )
            depth += 1
        best = min(beam, key=self._path_rank)
        return BeamSelectionResult(
            selected_indices=best.selected_indices,
            score_deltas=best.score_deltas,
            selection_phases=best.selection_phases,
            rollout_depths=best.rollout_depths,
            objective_value=float(best.state.score),
            cooccurrence=float(best.state.cooccurrence),
            redundancy=float(best.state.redundancy),
            layer_stats=tuple(layer_stats),
            final_beam_scores=tuple(float(path.state.score) for path in beam),
        )
