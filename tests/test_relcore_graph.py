from __future__ import annotations

import numpy as np

from relcore.graph.build import build_graph
from relcore.graph.prototypes import discover_prototypes
from relcore.scoring.reliability import compute_reliability
from relcore.schemas import ClipRecord


def test_noop_clip_has_lower_reliability_than_supported_smooth_motion():
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    states = np.zeros((3, 15, 3), dtype=np.float32)
    actions = np.zeros((3, 15, 3), dtype=np.float32)
    steps = np.linspace(0.0, 1.0, 15, dtype=np.float32)
    states[1, :, 0] = steps
    actions[1, :, 0] = steps
    states[2, :, 1] = steps
    actions[2, :, 1] = np.sin(steps * np.pi)

    result = compute_reliability(
        embeddings,
        states,
        actions,
        np.asarray([0.0, 0.5, 0.5], dtype=np.float32),
        knn=1,
        noop_threshold=1.0e-4,
        gripper_action_index=-1,
        min_reliability=0.05,
    )

    assert result.noop_ratio[0] == 1.0
    assert result.noop_ratio[1] < 0.1
    assert result.reliability[0] == 0.05
    assert result.reliability[1] > result.reliability[0]


def test_progress_uses_candidate_pool_standard_deviation_scales():
    states = np.zeros((2, 15, 2), dtype=np.float32)
    states[0, -1] = [1.0, 0.0]
    states[1, -1] = [3.0, 2.0]
    actions = np.zeros((2, 15, 2), dtype=np.float32)

    result = compute_reliability(
        np.eye(2, dtype=np.float32),
        states,
        actions,
        np.zeros(2, dtype=np.float32),
        knn=1,
        gripper_progress_weight=0.5,
    )

    expected_raw = np.asarray([1.0, 4.0], dtype=np.float32)
    expected = 0.5 + 0.5 * (1.0 - np.exp(-expected_raw))
    np.testing.assert_allclose(result.progress, expected, rtol=1.0e-6)


def test_soft_prototypes_are_deterministic_and_keep_original_top_weights():
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    first = discover_prototypes(
        embeddings, count=2, batch_size=4, max_iter=20, top_r=1, temperature=0.2, seed=7
    )
    second = discover_prototypes(
        embeddings, count=2, batch_size=4, max_iter=20, top_r=1, temperature=0.2, seed=7
    )

    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_allclose(first.centers, second.centers)
    assert first.indices.shape == (4, 1)
    assert np.all(first.weights > 0.5)
    assert np.all(first.weights <= 1.0)


def test_graph_keeps_directed_sequence_and_one_undirected_similarity_edge():
    clips = [
        ClipRecord(
            "ep000000_fragment_000000_000014",
            0,
            0,
            "task zero",
            0,
            14,
            15,
            None,
            "ep000000_fragment_000015_000029",
        ),
        ClipRecord(
            "ep000000_fragment_000015_000029",
            0,
            0,
            "task zero",
            15,
            29,
            15,
            "ep000000_fragment_000000_000014",
            None,
        ),
        ClipRecord(
            "ep000001_fragment_000000_000014",
            1,
            1,
            "task one",
            0,
            14,
            15,
            None,
            None,
        ),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.999, 0.001]], dtype=np.float32)
    reliability = np.asarray([0.8, 0.6, 0.7], dtype=np.float32)
    prototypes = discover_prototypes(
        embeddings, count=2, batch_size=3, max_iter=20, top_r=2, temperature=0.2, seed=3
    )

    graph = build_graph(
        clips,
        embeddings,
        reliability,
        prototypes,
        knn=2,
        similarity_threshold=0.95,
        cooccurrence_max_gap=4,
    )

    assert graph.sequence_edges.source.tolist() == [0]
    assert graph.sequence_edges.target.tolist() == [1]
    assert list(zip(graph.similarity_edges.source, graph.similarity_edges.target)) == [(0, 2)]
    assert graph.transition_matrix.shape == (2, 2)
    assert graph.cooccurrence_matrix.shape == (2, 2)
    assert graph.transition_matrix.nnz > 0
    assert graph.cooccurrence_matrix.nnz == 0
