import numpy as np
import pytest

from relcore.scoring.reliability import compute_reliability


def _inputs(points):
    count = len(points)
    return (
        np.asarray(points, dtype=np.float32).reshape(count, 1),
        np.zeros((count, 3, 2), dtype=np.float32),
        np.zeros((count, 3, 2), dtype=np.float32),
        np.zeros(count, dtype=np.float32),
    )


@pytest.mark.parametrize(
    "points,k,expected",
    [
        ([0, 1, 2, 4, 20], 2, [1, 1, 1, 2 / 3, 1 / 3]),
        ([0, 1, 3, 10], 1, [1, 1, 0.5, 0.5]),
        ([0, 0, 0, 0, 10], 2, [1, 1, 1, 1, 1 / 3]),
        ([0, 0, 0, 0], 2, [1, 1, 1, 1]),
        ([0, 1, 10], 10, [1, 1, 1]),
        ([0, 10], 10, [1, 1]),
        ([0], 10, [1]),
    ],
)
def test_median_radius_support_includes_self_and_caps_count(points, k, expected):
    result = compute_reliability(
        *_inputs(points), knn=k, support_mode="median_radius_count_with_self"
    )
    np.testing.assert_allclose(result.support, expected, rtol=1e-6)
    assert result.support.dtype == np.float32
    assert np.all((result.support >= 0) & (result.support <= 1))


def test_default_support_keeps_exponential_formula():
    result = compute_reliability(*_inputs([0, 1, 2, 4, 20]), knn=2)
    np.testing.assert_allclose(
        result.support, np.exp(-np.asarray([2, 1, 2, 3, 18]) / (2 + 1e-8)),
        rtol=1e-6,
    )


def test_unknown_support_mode_is_rejected_even_for_single_sample():
    with pytest.raises(ValueError, match="support_mode"):
        compute_reliability(*_inputs([0]), support_mode="unknown")
