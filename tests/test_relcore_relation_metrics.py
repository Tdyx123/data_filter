from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

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
        sample_ids=["source", "target", "weaker"],
        task_indices=np.zeros(3, dtype=np.int64),
        embeddings=np.eye(3, dtype=np.float32),
        reliability=np.asarray([1.0, 0.5, 0.8], dtype=np.float32),
        prototype_indices=np.asarray([[0, -1], [1, -1], [0, 1]], dtype=np.int32),
        prototype_weights=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32
        ),
        sequence_edges=_edges([(0, 1, 1.0)], "sequence"),
        similarity_edges=_edges([], "similarity"),
        transition_matrix=sparse.csr_matrix(
            np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        ),
        cooccurrence_matrix=sparse.csr_matrix(
            np.asarray([[0.0, 0.25], [0.75, 0.0]], dtype=np.float32)
        ),
    )


def test_relation_kernel_uses_max_reliable_coverage_for_pair_metrics() -> None:
    from relcore.selection.relation_metrics import RelationMetricKernel

    kernel = RelationMetricKernel(_graph())
    selected = np.asarray([True, True, False])
    coverage = kernel.coverage_from_mask(selected)
    sequence_counts = kernel.sequence_counts_from_mask(selected)
    values = kernel.values(coverage, sequence_counts)

    assert coverage.tolist() == pytest.approx([1.0, 0.5])
    assert values.cooccurrence == pytest.approx(2.0**-0.5)
    assert values.sequence == pytest.approx(1.0, abs=1.0e-7)

    weaker = kernel.candidate_delta(selected, coverage, sequence_counts, 2)
    assert weaker.prototype_coverage.tolist() == pytest.approx([1.0, 0.5])
    assert weaker.cooccurrence == pytest.approx(0.0)
    assert weaker.sequence == pytest.approx(0.0)


def test_relation_kernel_adds_and_removes_only_selected_sequence_edges() -> None:
    from relcore.selection.relation_metrics import RelationMetricKernel

    kernel = RelationMetricKernel(_graph())
    selected = np.asarray([True, False, False])
    coverage = kernel.coverage_from_mask(selected)
    sequence_counts = kernel.empty_sequence_counts()

    adjacent = kernel.candidate_delta(selected, coverage, sequence_counts, 1)
    nonadjacent = kernel.candidate_delta(selected, coverage, sequence_counts, 2)

    assert adjacent.sequence == pytest.approx(1.0, abs=1.0e-7)
    assert nonadjacent.sequence == pytest.approx(0.0)

    kernel.update_sequence_counts(sequence_counts, selected, 1, 1.0)
    selected[1] = True
    np.testing.assert_allclose(
        sequence_counts,
        np.asarray([[0.0, 0.5], [0.0, 0.0]]),
        atol=1.0e-12,
    )
    assert kernel.values(adjacent.prototype_coverage, sequence_counts).sequence == pytest.approx(
        1.0, abs=1.0e-7
    )

    kernel.update_sequence_counts(sequence_counts, selected, 1, -1.0)
    selected[1] = False
    np.testing.assert_allclose(sequence_counts, 0.0, atol=1.0e-12)


def test_sequence_metric_uses_log1p_saturation_against_full_pool_counts() -> None:
    from relcore.selection.relation_metrics import RelationMetricKernel

    graph = _graph()
    graph.sample_ids.extend(["source-2", "target-2"])
    graph.task_indices = np.zeros(5, dtype=np.int64)
    graph.embeddings = np.eye(5, dtype=np.float32)
    graph.reliability = np.append(graph.reliability, [1.0, 0.5]).astype(np.float32)
    graph.prototype_indices = np.vstack(
        [graph.prototype_indices, [[0, -1], [1, -1]]]
    )
    graph.prototype_weights = np.vstack(
        [graph.prototype_weights, [[1.0, 0.0], [1.0, 0.0]]]
    )
    graph.sequence_edges = _edges([(0, 1, 1.0), (3, 4, 1.0)], "sequence")
    kernel = RelationMetricKernel(graph)
    selected = np.asarray([True, True, False, False, False])

    values = kernel.values(
        kernel.coverage_from_mask(selected),
        kernel.sequence_counts_from_mask(selected),
    )

    assert values.sequence == pytest.approx(0.584962492, abs=1.0e-8)
