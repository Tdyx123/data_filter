"""Build sparse sequence, similarity, transition, and cooccurrence relations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from relcore.schemas import ClipRecord, EdgeTable, GraphData

from .prototypes import PrototypeData


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
) -> sparse.csr_matrix:
    count = len(prototypes.centers)
    matrix = np.zeros((count, count), dtype=np.float64)
    for source, target, edge_weight in edges:
        for source_index, source_weight in zip(
            prototypes.indices[source], prototypes.weights[source], strict=True
        ):
            for target_index, target_weight in zip(
                prototypes.indices[target], prototypes.weights[target], strict=True
            ):
                matrix[source_index, target_index] += (
                    edge_weight * float(source_weight) * float(target_weight)
                )
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


def build_graph(
    clips: Sequence[ClipRecord],
    embeddings: np.ndarray,
    reliability: np.ndarray,
    prototypes: PrototypeData,
    *,
    knn: int = 32,
    similarity_threshold: float = 0.8,
    cooccurrence_max_gap: int = 4,
) -> GraphData:
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
    id_to_index = {clip.sample_id: index for index, clip in enumerate(clips)}
    sequence: list[tuple[int, int, float]] = []
    for source, clip in enumerate(clips):
        if clip.next_sample_id is None:
            continue
        target = id_to_index[clip.next_sample_id]
        if clip.end_step + 1 != clips[target].start_step:
            raise ValueError("sequence edge is not exactly contiguous")
        sequence.append((source, target, float(min(quality[source], quality[target]))))

    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-8)
    similarity_by_pair: dict[tuple[int, int], float] = {}
    if len(values) > 1:
        neighbors = min(int(knn) + 1, len(values))
        distances, indices = _cosine_neighbors(normalized, neighbors)
        for source in range(len(values)):
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

    episode_nodes: dict[int, list[int]] = defaultdict(list)
    for index, clip in enumerate(clips):
        episode_nodes[clip.episode_id].append(index)
    cooccurrence: list[tuple[int, int, float]] = []
    for nodes in episode_nodes.values():
        nodes.sort(key=lambda index: (clips[index].start_step, clips[index].sample_id))
        for left_position, source in enumerate(nodes):
            stop = min(len(nodes), left_position + cooccurrence_max_gap + 1)
            for target_position in range(left_position + 2, stop):
                target = nodes[target_position]
                cooccurrence.append((source, target, float(quality[source] * quality[target])))
    return GraphData(
        sample_ids=[clip.sample_id for clip in clips],
        task_indices=np.asarray([clip.task_index for clip in clips], dtype=np.int64),
        embeddings=values,
        reliability=quality,
        prototype_indices=prototypes.indices,
        prototype_weights=prototypes.weights,
        sequence_edges=_edge_table(sequence, "sequence"),
        similarity_edges=_edge_table(similarity, "similarity"),
        transition_matrix=_prototype_matrix(
            [
                (
                    source,
                    target,
                    float(quality[source] * quality[target]),
                )
                for source, target, _ in sequence
            ],
            prototypes,
        ),
        cooccurrence_matrix=_prototype_matrix(cooccurrence, prototypes),
    )
