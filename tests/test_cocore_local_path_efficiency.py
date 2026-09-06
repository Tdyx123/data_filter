import numpy as np
import pytest

from cocore.local_path_efficiency import compute_local_path_efficiency


@pytest.mark.parametrize(
    "points,expected",
    [
        ([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], 1),
        ([[0, 0, 0], [0.125, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], 0.8),
        ([[0, 0, 0], [1, 0, 0], [0, 0, 0]], 0),
        ([[0, 0, 0], [1, 2, 3]], 1),
    ],
)
def test_geometry(points, expected):
    assert compute_local_path_efficiency(points, delta_path=1e-6) == pytest.approx(expected)


def test_stationary_and_threshold_boundary():
    assert np.isnan(compute_local_path_efficiency(np.zeros((3, 3)), delta_path=0.1))
    assert np.isnan(compute_local_path_efficiency([[0, 0, 0], [0.1, 0, 0]], delta_path=0.1))


@pytest.mark.parametrize("points", [[], [[0, 0, 0]], [[0, 0], [1, 1]], [[0, 0, 0], [np.nan, 0, 0]]])
def test_invalid_positions(points):
    with pytest.raises(ValueError):
        compute_local_path_efficiency(points, delta_path=0.1)


@pytest.mark.parametrize("threshold", [0, -1, np.nan, np.inf, True, None])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        compute_local_path_efficiency(np.zeros((3, 3)), delta_path=threshold)


def test_optional_config_and_nan_fusion():
    from cocore.config import resolve_config
    from cocore.action_variation import fuse_reliability

    base = {"objective": {"relation": "sequence", "relation_weight": 1}}
    assert "local_path_efficiency" not in resolve_config(base)["reliability_metrics"]
    with pytest.raises(ValueError, match="local_path_efficiency"):
        resolve_config({**base, "reliability_metrics": ["local_path_efficiency"]})
    resolve_config(
        {
            **base,
            "local_path_efficiency": {"delta_path": 0.01},
            "reliability_metrics": ["local_path_efficiency"],
        }
    )
    values = np.array([0.25, 0.25, 0.25])
    result = fuse_reliability(
        values,
        values,
        values,
        values,
        ["support", "local_path_efficiency"],
        min_reliability=0.05,
        local_path_efficiency=np.array([1, np.nan, 0]),
    )
    np.testing.assert_allclose(result, [0.5, 0.25, 0.05])
    result = fuse_reliability(
        values,
        values,
        values,
        values,
        ["local_path_efficiency"],
        min_reliability=0.05,
        local_path_efficiency=np.array([1, np.nan, 0]),
    )
    np.testing.assert_allclose(result, [1, 1, 0.05])
