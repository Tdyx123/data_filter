"""Cocore-owned sparse graph construction for ordered candidate clips."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from relcore.graph.prototypes import PrototypeData, valid_prototype_assignments
from relcore.schemas import ClipRecord, EdgeTable, GraphData


SEQUENCE_ADJACENCY = "ordered_candidates"


def _edge_table(edges: list[tuple[int, int, float]], edge_type: str) -> EdgeTable:
    if not edges:
        return EdgeTable(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            edge_type,
        )
    return EdgeTable(
        np.asarray([edge[0] for edge in edges], dtype=np.int64),
        np.asarray([edge[1] for edge in edges], dtype=np.int64),
        np.asarray([edge[2] for edge in edges], dtype=np.float32),
        edge_type,
    )


def _prototype_matrix(
    edges: list[tuple[int, int, float]],
    prototypes: PrototypeData,
    *,
    normalize: bool,
) -> sparse.csr_matrix:
    matrix = np.zeros((prototypes.count, prototypes.count), dtype=np.float64)
    for source, target, edge_weight in edges:
        source_indices, source_weights = valid_prototype_assignments(
            prototypes.indices[source], prototypes.weights[source]
        )
        target_indices, target_weights = valid_prototype_assignments(
            prototypes.indices[target], prototypes.weights[target]
        )
        for source_index, source_weight in zip(source_indices, source_weights, strict=True):
            for target_index, target_weight in zip(target_indices, target_weights, strict=True):
                matrix[source_index, target_index] += (
                    edge_weight * float(source_weight) * float(target_weight)
                )
    if normalize:
        total = float(matrix.sum())
        if total > 0:
            matrix /= total
    return sparse.csr_matrix(matrix.astype(np.float32))


def _cosine_neighbors(values: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    contiguous = np.ascontiguousarray(values.astype(np.float32))
    try:
        import faiss

        index = faiss.IndexFlatIP(contiguous.shape[1])
        index.add(contiguous)
        similarities, indices = index.search(contiguous, neighbors)
        return (1.0 - similarities).astype(np.float32), indices
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=neighbors, metric="cosine")
        return model.fit(contiguous).kneighbors(contiguous)


def _episode_nodes(clips: Sequence[ClipRecord]) -> dict[int, list[int]]:
    nodes: dict[int, list[int]] = defaultdict(list)
    for index, clip in enumerate(clips):
        nodes[clip.episode_id].append(index)
    for episode_indices in nodes.values():
        episode_indices.sort(key=lambda index: (clips[index].start_step, clips[index].sample_id))
    return nodes


def _ordered_sequence_edges(
    clips: Sequence[ClipRecord],
    reliability: np.ndarray,
    episode_nodes: dict[int, list[int]],
) -> list[tuple[int, int, float]]:
    edges: list[tuple[int, int, float]] = []
    for nodes in episode_nodes.values():
        for position, source in enumerate(nodes):
            clip = clips[source]
            expected_previous = clips[nodes[position - 1]].sample_id if position > 0 else None
            expected_next = (
                clips[nodes[position + 1]].sample_id if position + 1 < len(nodes) else None
            )
            if clip.previous_sample_id != expected_previous or clip.next_sample_id != expected_next:
                raise ValueError("cocore sequence links do not match ordered episode candidates")
            if expected_next is None:
                continue
            target = nodes[position + 1]
            if clips[target].start_step <= clip.start_step:
                raise ValueError("cocore sequence candidate starts must be strictly increasing")
            edges.append((source, target, float(min(reliability[source], reliability[target]))))
    return edges


def build_graph(
    clips: Sequence[ClipRecord],
    embeddings: np.ndarray,
    reliability: np.ndarray,
    prototypes: PrototypeData,
    *,
    included_indices: np.ndarray | None = None,
    knn: int = 32,
    similarity_threshold: float = 0.8,
    cooccurrence_max_gap: int = 4,
    normalize_prototype_relations: bool = True,
) -> GraphData:
    """Build Cocore relations with sequence edges between ordered candidates."""

    values = np.asarray(embeddings, dtype=np.float32)
    quality = np.asarray(reliability, dtype=np.float32)
    if (
        values.ndim != 2
        or len(clips) != len(values)
        or quality.shape != (len(clips),)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(quality))
        or len({clip.sample_id for clip in clips}) != len(clips)
    ):
        raise ValueError("graph inputs do not share the same node count")

    if included_indices is None:
        included = np.arange(len(clips), dtype=np.int64)
    else:
        included = np.asarray(included_indices)
        if (
            included.ndim != 1
            or len(included) == 0
            or included.dtype.kind not in {"i", "u"}
            or np.any(included < 0)
            or np.any(included >= len(clips))
            or (len(included) > 1 and np.any(included[1:] <= included[:-1]))
        ):
            raise ValueError("included_indices must be non-empty, sorted, unique, and in range")
        included = included.astype(np.int64, copy=False)
    if (
        prototypes.indices.ndim != 2
        or prototypes.weights.shape != prototypes.indices.shape
        or prototypes.indices.shape[0] != len(clips)
    ):
        raise ValueError("prototype assignments do not share the full candidate count")

    selected_clips = [clips[int(index)] for index in included]
    selected_values = values[included]
    selected_quality = quality[included]
    selected_prototypes = PrototypeData(
        centers=prototypes.centers,
        indices=prototypes.indices[included],
        weights=prototypes.weights[included],
        labels=prototypes.labels,
    )
    remap = {int(source): target for target, source in enumerate(included)}

    episode_nodes = _episode_nodes(clips)
    full_sequence = _ordered_sequence_edges(clips, quality, episode_nodes)
    sequence = [
        (remap[source], remap[target], weight)
        for source, target, weight in full_sequence
        if source in remap and target in remap
    ]
    sequence_relations = [
        (
            remap[source],
            remap[target],
            float(quality[source] * quality[target]),
        )
        for source, target, _ in full_sequence
        if source in remap and target in remap
    ]

    normalized = selected_values / np.maximum(
        np.linalg.norm(selected_values, axis=1, keepdims=True), 1.0e-8
    )
    similarity_by_pair: dict[tuple[int, int], float] = {}
    if len(selected_values) > 1:
        neighbors = min(int(knn) + 1, len(selected_values))
        distances, indices = _cosine_neighbors(normalized, neighbors)
        for source in range(len(selected_values)):
            for target, distance in zip(indices[source], distances[source], strict=True):
                target = int(target)
                if target == source:
                    continue
                score = min(1.0, max(0.0, 1.0 - float(distance)))
                if score < similarity_threshold:
                    continue
                pair = (min(source, target), max(source, target))
                similarity_by_pair[pair] = max(similarity_by_pair.get(pair, 0.0), score)
    similarity = [
        (source, target, score) for (source, target), score in sorted(similarity_by_pair.items())
    ]

    cooccurrence: list[tuple[int, int, float]] = []
    for nodes in episode_nodes.values():
        for left_position, source in enumerate(nodes):
            stop = min(len(nodes), left_position + cooccurrence_max_gap + 1)
            for target_position in range(left_position + 2, stop):
                target = nodes[target_position]
                if source in remap and target in remap:
                    cooccurrence.append(
                        (
                            remap[source],
                            remap[target],
                            float(quality[source] * quality[target]),
                        )
                    )

    return GraphData(
        sample_ids=[clip.sample_id for clip in selected_clips],
        task_indices=np.asarray([clip.task_index for clip in selected_clips], dtype=np.int64),
        embeddings=selected_values,
        reliability=selected_quality,
        prototype_indices=selected_prototypes.indices,
        prototype_weights=selected_prototypes.weights,
        sequence_edges=_edge_table(sequence, "sequence"),
        similarity_edges=_edge_table(similarity, "similarity"),
        transition_matrix=_prototype_matrix(
            sequence_relations,
            selected_prototypes,
            normalize=normalize_prototype_relations,
        ),
        cooccurrence_matrix=_prototype_matrix(
            cooccurrence,
            selected_prototypes,
            normalize=normalize_prototype_relations,
        ),
        prototype_labels=selected_prototypes.labels,
    )
