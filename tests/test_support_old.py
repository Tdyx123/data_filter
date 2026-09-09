"""Regression coverage for the selectable legacy distance support."""

import numpy as np
import pytest

from cocore.action_variation import fuse_reliability, normalize_reliability_metrics
from cocore.config import resolve_config
from relcore.scoring.reliability import compute_reliability


@pytest.mark.parametrize(
    "points,k,kth",
    [
        ([0, 1, 2, 4, 20], 2, [2, 1, 2, 3, 18]),
        ([0, 1, 3, 10], 1, [1, 1, 2, 7]),
        ([0, 0, 0, 0, 10], 2, [0, 0, 0, 0, 10]),
        ([0, 0, 0, 0], 2, [0, 0, 0, 0]),
        ([0, 1, 10], 10, [10, 9, 10]),
        ([0], 10, [0]),
    ],
)
def test_dual_support_preserves_legacy_formula(points, k, kth):
    n = len(points)
    inputs = (
        np.asarray(points, dtype=np.float32).reshape(n, 1),
        np.zeros((n, 3, 2)),
        np.zeros((n, 3, 2)),
        np.zeros(n),
    )
    result = compute_reliability(
        *inputs, knn=k, support_mode="median_radius_count_with_self", compute_support_old=True
    )
    expected = np.exp(-np.asarray(kth) / (np.median(kth) + 1e-8))
    np.testing.assert_allclose(result.support_old, expected, rtol=1e-6)
    assert result.support_old.dtype == np.float32
    current = compute_reliability(*inputs, knn=k, support_mode="median_radius_count_with_self")
    np.testing.assert_array_equal(result.support, current.support)
    legacy = compute_reliability(*inputs, knn=k)
    np.testing.assert_allclose(result.support_old, legacy.support, rtol=1e-6)


def test_old_support_selection_and_exclusion():
    assert normalize_reliability_metrics(["progress", "support_old"]) == ("support_old", "progress")
    assert normalize_reliability_metrics(["progress"]) == ("progress",)
    with pytest.raises(ValueError, match="mutually exclusive"):
        normalize_reliability_metrics(["support", "support_old"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_config(
            {
                "objective": {"relation": "sequence", "relation_weight": 1},
                "reliability_metrics": ["support", "support_old"],
            }
        )


def test_old_support_fusion_uses_only_selected_component():
    current = np.array([1.0, 1.0])
    progress = np.array([1.0, 0.25])
    old = np.array([0.25, 0.0])
    args = (current, progress, current, current, ["support_old", "progress"])
    result = fuse_reliability(*args, support_old=old, min_reliability=0.05)
    np.testing.assert_allclose(result, [0.5, 0.05])
    with pytest.raises(ValueError, match="support_old.*requires"):
        fuse_reliability(*args, min_reliability=0.05)
    for invalid in (np.array([np.nan, 0]), np.array([1.1, 0]), np.array([0.5])):
        with pytest.raises(ValueError):
            fuse_reliability(*args, support_old=invalid, min_reliability=0.05)
