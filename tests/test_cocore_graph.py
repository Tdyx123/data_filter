from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cocore.graph import build_graph
from cocore.index import build_clip_records
from relcore.data.index import build_clip_records as build_relcore_clip_records
from relcore.graph import build_graph as build_relcore_graph
from relcore.graph.prototypes import PrototypeData
from trajectory_data import EpisodeRecord


def _prototypes(count: int) -> PrototypeData:
    return PrototypeData(
        centers=np.eye(count, dtype=np.float32),
        indices=np.arange(count, dtype=np.int32)[:, None],
        weights=np.ones((count, 1), dtype=np.float32),
        labels=tuple(f"prototype {index}" for index in range(count)),
    )


def test_cocore_graph_links_overlapping_ordered_candidates() -> None:
    clips = build_clip_records([EpisodeRecord(0, 31, 0, "task")])
    reliability = np.asarray([1.0, 0.8, 0.6], dtype=np.float32)

    graph = build_graph(
        clips,
        np.eye(3, dtype=np.float32),
        reliability,
        _prototypes(3),
        knn=1,
        similarity_threshold=1.0,
        cooccurrence_max_gap=4,
        normalize_prototype_relations=False,
    )

    np.testing.assert_array_equal(graph.sequence_edges.source, [0, 1])
    np.testing.assert_array_equal(graph.sequence_edges.target, [1, 2])
    np.testing.assert_allclose(graph.sequence_edges.weight, [0.8, 0.6])
    np.testing.assert_allclose(
        graph.transition_matrix.toarray()[[0, 1], [1, 2]],
        [0.8, 0.48],
    )


def test_cocore_graph_matches_relcore_for_non_overlapping_candidates() -> None:
    episodes = [EpisodeRecord(0, 45, 0, "task")]
    cocore_clips = build_clip_records(episodes)
    relcore_clips = build_relcore_clip_records(episodes, length=15, stride=15)
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )
    reliability = np.asarray([0.9, 0.7, 0.5], dtype=np.float32)
    prototypes = _prototypes(3)

    actual = build_graph(
        cocore_clips,
        embeddings,
        reliability,
        prototypes,
        knn=2,
        similarity_threshold=0.5,
        cooccurrence_max_gap=4,
        normalize_prototype_relations=False,
    )
    expected = build_relcore_graph(
        relcore_clips,
        embeddings,
        reliability,
        prototypes,
        knn=2,
        similarity_threshold=0.5,
        cooccurrence_max_gap=4,
        normalize_prototype_relations=False,
    )

    np.testing.assert_array_equal(actual.sequence_edges.source, expected.sequence_edges.source)
    np.testing.assert_array_equal(actual.sequence_edges.target, expected.sequence_edges.target)
    np.testing.assert_allclose(actual.sequence_edges.weight, expected.sequence_edges.weight)
    np.testing.assert_array_equal(actual.similarity_edges.source, expected.similarity_edges.source)
    np.testing.assert_array_equal(actual.similarity_edges.target, expected.similarity_edges.target)
    np.testing.assert_allclose(actual.similarity_edges.weight, expected.similarity_edges.weight)
    np.testing.assert_allclose(
        actual.transition_matrix.toarray(),
        expected.transition_matrix.toarray(),
    )
    np.testing.assert_allclose(
        actual.cooccurrence_matrix.toarray(),
        expected.cooccurrence_matrix.toarray(),
    )


def test_cocore_graph_rejects_links_that_skip_an_ordered_candidate() -> None:
    clips = build_clip_records([EpisodeRecord(0, 31, 0, "task")])
    clips[0] = replace(clips[0], next_sample_id=clips[2].sample_id)

    with pytest.raises(ValueError, match="ordered episode candidates"):
        build_graph(
            clips,
            np.eye(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            _prototypes(3),
            knn=1,
        )


def test_cocore_graph_builds_induced_graph_without_compressing_temporal_gaps() -> None:
    clips = build_clip_records([EpisodeRecord(0, 60, 0, "task")])
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.8, 0.6],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )

    graph = build_graph(
        clips,
        embeddings,
        np.ones(4, dtype=np.float32),
        _prototypes(4),
        included_indices=np.asarray([0, 2], dtype=np.int64),
        knn=1,
        similarity_threshold=0.75,
        cooccurrence_max_gap=2,
        normalize_prototype_relations=False,
    )

    assert graph.sample_ids == [clips[0].sample_id, clips[2].sample_id]
    np.testing.assert_array_equal(graph.prototype_indices, [[0], [2]])
    np.testing.assert_array_equal(graph.sequence_edges.source, [])
    np.testing.assert_array_equal(graph.sequence_edges.target, [])
    np.testing.assert_array_equal(graph.similarity_edges.source, [0])
    np.testing.assert_array_equal(graph.similarity_edges.target, [1])
    np.testing.assert_allclose(graph.similarity_edges.weight, [0.8])
    np.testing.assert_allclose(graph.cooccurrence_matrix.toarray()[0, 2], 1.0)


@pytest.mark.parametrize(
    "included_indices",
    [
        np.asarray([], dtype=np.int64),
        np.asarray([1, 0], dtype=np.int64),
        np.asarray([1, 0], dtype=np.uint64),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([0, 4], dtype=np.int64),
    ],
)
def test_cocore_graph_rejects_invalid_included_indices(included_indices: np.ndarray) -> None:
    clips = build_clip_records([EpisodeRecord(0, 60, 0, "task")])

    with pytest.raises(ValueError, match="included_indices"):
        build_graph(
            clips,
            np.eye(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            _prototypes(4),
            included_indices=included_indices,
            knn=1,
        )
