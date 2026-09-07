"""Physical tracking error must not cancel directions or consume the final action."""

import numpy as np
import pytest

from cocore.config import resolve_config


SETTINGS = {
    "action_source": "original_command",
    "action_semantics": "delta_from_observed_position",
    "action_scale": [1, 1, 1],
    "alignment_confirmed": True,
}


def compute(p, a, t, config=None):
    from cocore.action_execution_deviation import compute_action_execution_deviation

    return compute_action_execution_deviation(p, a, t, config=config or SETTINGS)


def test_physical_error_and_final_action():
    raw, reason = compute([[0, 0, 0], [0.006, 0.003, 0]], [[0.010, 0, 0], [999, 0, 0]], [0, 1])
    assert raw == pytest.approx(0.005)
    assert reason == ""


def test_norm_before_mean_and_nonuniform_time():
    raw, reason = compute([[0, 0, 0]] * 3, [[0.01, 0, 0], [-0.01, 0, 0], [0, 0, 0]], [0, 1, 3])
    assert raw == pytest.approx(0.01)
    assert reason == ""


def test_explicit_per_axis_scaling_and_true_zero():
    raw, reason = compute(
        [[0, 0, 0], [0.01, 0.02, 0.03]],
        [[1, 1, 1], [0, 0, 0]],
        [0, 1],
        {**SETTINGS, "action_scale": [0.01, 0.02, 0.03]},
    )
    assert raw == 0
    assert reason == ""


@pytest.mark.parametrize(
    "p,a,t,reason",
    [
        ([[0, 0, 0]], [[0, 0, 0]], [0], "too_short"),
        ([[0, 0, 0]] * 2, [[0, 0, 0]] * 2, [1, 1], "non_increasing_time"),
        ([[0, 0, 0]] * 2, [[0, 0, 0]] * 2, [1, 0], "non_increasing_time"),
        ([[-1e308, 0, 0], [1e308, 0, 0]], [[0, 0, 0]] * 2, [0, 1], "overflow"),
    ],
)
def test_uncomputable_is_not_zero(p, a, t, reason):
    raw, actual_reason = compute(p, a, t)
    assert np.isnan(raw)
    assert actual_reason == reason


@pytest.mark.parametrize(
    "p,a,t",
    [
        ([[0, 0]] * 2, [[0, 0, 0]] * 2, [0, 1]),
        ([[0, 0, 0]] * 2, [[0, 0]] * 2, [0, 1]),
        ([[0, 0, 0]] * 2, [[0, 0, 0]], [0, 1]),
        ([[0, 0, np.nan]] * 2, [[0, 0, 0]] * 2, [0, 1]),
        ([[0, 0, 0]] * 2, [[0, 0, np.inf]] * 2, [0, 1]),
        ([[0, 0, 0]] * 2, [[0, 0, 0]] * 2, [0, np.inf]),
    ],
)
def test_malformed_inputs_raise(p, a, t):
    with pytest.raises(ValueError, match="action_execution_deviation"):
        compute(p, a, t)


@pytest.mark.parametrize(
    "override",
    [
        {"action_source": "state_difference"},
        {"action_source": "unknown"},
        {"action_semantics": "absolute_position"},
        {"alignment_confirmed": False},
        {"alignment_confirmed": 1},
        {"action_scale": [1, 1]},
        {"action_scale": [0, 1, 1]},
        {"action_scale": [True, 1, 1]},
        {"action_scale": [1, np.inf, 1]},
        {"unknown": 1},
    ],
)
def test_rejects_unconfirmed_or_wrong_contract(override):
    from cocore.action_execution_deviation import resolve_execution_config

    with pytest.raises(ValueError, match="action_execution_deviation"):
        resolve_execution_config({**SETTINGS, **override})


@pytest.mark.parametrize("missing", list(SETTINGS))
def test_requires_every_contract_field(missing):
    from cocore.action_execution_deviation import resolve_execution_config

    with pytest.raises(ValueError, match="action_execution_deviation"):
        resolve_execution_config({key: value for key, value in SETTINGS.items() if key != missing})


def test_optional_metric_requires_config_and_does_not_change_defaults():
    base = {"objective": {"relation": "sequence", "relation_weight": 1}}
    with pytest.raises(ValueError, match="requires explicit action_execution_deviation"):
        resolve_config({**base, "reliability_metrics": ["low_action_execution_deviation"]})
    resolved = resolve_config({**base, "action_execution_deviation": SETTINGS})
    assert "low_action_execution_deviation" not in resolved["reliability_metrics"]


def test_reverse_quantiles_ignore_invalid_and_constant_pool():
    from cocore.action_execution_deviation import normalize_execution_deviation

    kwargs = dict(quantile_low=0, quantile_high=1, epsilon=1e-8)
    np.testing.assert_allclose(
        normalize_execution_deviation(np.array([0.0, 1.0, 2.0, np.nan]), **kwargs),
        [1, 0.5, 0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_array_equal(normalize_execution_deviation([10.0, 10.0], **kwargs), [1, 1])
    assert np.isnan(normalize_execution_deviation([np.nan], **kwargs)[0])


def test_fusion_uses_execution_score_and_rejects_unavailable_selected_values():
    from cocore.action_variation import fuse_reliability

    values = np.array([0.25, 0.25])
    args = (values, values, values, values, ["support", "low_action_execution_deviation"])
    np.testing.assert_allclose(
        fuse_reliability(*args, min_reliability=0.05, low_action_execution_deviation=[1, 0]),
        [0.5, 0.05],
    )
    with pytest.raises(ValueError, match="execution_deviation"):
        fuse_reliability(*args, min_reliability=0.05)
    with pytest.raises(ValueError, match="execution_deviation"):
        fuse_reliability(*args, min_reliability=0.05, low_action_execution_deviation=[1, np.nan])
