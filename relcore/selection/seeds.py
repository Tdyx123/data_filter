"""Diverse pair seed generation for multi-branch selection."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .objective import ObjectiveContext


def _pair_feasible(
    context: ObjectiveContext,
    pair: tuple[int, int],
    task_quotas: dict[int, int],
) -> bool:
    counts = Counter(int(context.graph.task_indices[index]) for index in pair)
    return all(count <= task_quotas.get(task, 0) for task, count in counts.items())


def _pair_score(context: ObjectiveContext, pair: tuple[int, int]) -> float:
    state = context.empty_state()
    for index in pair:
        context.add_candidate(state, index)
    return float(state.objective_value)


def _episode_key(sample_id: str) -> str:
    return sample_id.split("_fragment_", 1)[0]


def generate_seed_pairs(
    context: ObjectiveContext,
    task_quotas: dict[int, int],
    *,
    budget: int,
    branches: int = 8,
    seed_candidates: int = 128,
    seed_similarity_threshold: float = 0.9,
    transition_seed_threshold: float = 0.0,
    pairs_per_transition: int = 4,
) -> list[tuple[int, int]]:
    if budget < 2 or branches <= 0:
        return []
    graph = context.graph
    pairs: set[tuple[int, int]] = set()
    for source, target in zip(
        graph.sequence_edges.source, graph.sequence_edges.target, strict=True
    ):
        pair = (int(source), int(target))
        if _pair_feasible(context, pair, task_quotas):
            pairs.add(pair)

    primary = graph.prototype_indices[:, 0]
    by_prototype: dict[int, list[int]] = defaultdict(list)
    for index, prototype in enumerate(primary):
        by_prototype[int(prototype)].append(index)
    for nodes in by_prototype.values():
        nodes.sort(key=lambda index: (-float(graph.reliability[index]), graph.sample_ids[index]))
    transition = graph.transition_matrix.toarray()
    transitions = [
        (float(transition[source, target]), source, target)
        for source, target in zip(*np.nonzero(transition), strict=True)
        if float(transition[source, target]) > transition_seed_threshold
    ]
    transitions.sort(key=lambda item: (-item[0], item[1], item[2]))
    transition_representatives: list[tuple[float, tuple[int, int]]] = []
    for transition_weight, source_prototype, target_prototype in transitions:
        added = 0
        representative: tuple[int, int] | None = None
        for source in by_prototype[source_prototype]:
            for target in by_prototype[target_prototype]:
                if source == target or _episode_key(graph.sample_ids[source]) == _episode_key(
                    graph.sample_ids[target]
                ):
                    continue
                pair = (source, target)
                if not _pair_feasible(context, pair, task_quotas):
                    continue
                pairs.add(pair)
                if representative is None:
                    representative = pair
                added += 1
                if added >= pairs_per_transition:
                    break
            if added >= pairs_per_transition:
                break
        if representative is not None:
            transition_representatives.append((transition_weight, representative))

    ranked = sorted(
        ((_pair_score(context, pair), pair) for pair in pairs),
        key=lambda item: (-item[0], graph.sample_ids[item[1][0]], graph.sample_ids[item[1][1]]),
    )[:seed_candidates]
    rare_pair = (
        min(
            transition_representatives,
            key=lambda item: (
                item[0],
                -float(graph.reliability[item[1][0]] * graph.reliability[item[1][1]]),
                graph.sample_ids[item[1][0]],
                graph.sample_ids[item[1][1]],
            ),
        )[1]
        if transition_representatives
        else None
    )
    if rare_pair is not None and all(pair != rare_pair for _, pair in ranked):
        rare_item = (_pair_score(context, rare_pair), rare_pair)
        insertion = max(0, min(len(ranked), branches) - 1)
        if len(ranked) < seed_candidates:
            ranked.insert(insertion, rare_item)
        elif ranked:
            ranked[insertion] = rare_item
    selected: list[tuple[int, int]] = []
    representations: list[np.ndarray] = []
    for _, pair in ranked:
        representation = np.concatenate(
            [graph.embeddings[pair[0]], graph.embeddings[pair[1]]]
        ).astype(np.float32)
        representation /= max(float(np.linalg.norm(representation)), 1.0e-8)
        duplicate = False
        for old_pair, old_representation in zip(selected, representations, strict=True):
            overlap = len(set(pair) & set(old_pair))
            cosine = float(representation @ old_representation)
            if overlap == 2 or cosine >= seed_similarity_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(pair)
        representations.append(representation)
        if len(selected) >= branches:
            break
    return selected
