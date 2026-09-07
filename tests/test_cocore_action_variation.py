from __future__ import annotations

import itertools

import numpy as np
import pytest

from cocore.action_variation import (
    RELIABILITY_METRICS,
    compute_step_action_variation,
    fuse_reliability,
    normalize_action_variation,
    normalize_reliability_metrics,
    top_k_mean,
)


def test_step_action_variation_uses_l2_delta_and_truncated_future_population_variance() -> None:
    actions = np.asarray(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [3.0, 6.0],
        ],
        dtype=np.float32,
    )

    actual = compute_step_action_variation(actions)

    np.testing.assert_allclose(actual, [0.5, 10.0, 4.0], rtol=0.0, atol=1.0e-7)


def test_step_action_variation_future_window_is_exactly_five_actions() -> None:
    actions = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [100.0]])

    actual = compute_step_action_variation(actions)

    assert float(actual[0]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "actions",
    [
        np.asarray([], dtype=np.float32),
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([[1.0, np.inf]], dtype=np.float32),
    ],
)
def test_step_action_variation_rejects_invalid_actions(actions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="action variation"):
        compute_step_action_variation(actions)


def test_top_k_mean_uses_only_the_three_largest_values() -> None:
    values = np.asarray([1.0, 9.0, 2.0, 8.0, 3.0], dtype=np.float32)

    assert top_k_mean(values) == pytest.approx(20.0 / 3.0)


def test_action_variation_quantile_normalization_clips_both_tails() -> None:
    values = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    actual = normalize_action_variation(
        values,
        quantile_low=0.25,
        quantile_high=0.75,
        epsilon=1.0e-8,
    )

    np.testing.assert_allclose(actual, [0.0, 0.0, 0.5, 1.0, 1.0])


@pytest.mark.parametrize("constant", [0.0, 2.0])
def test_constant_action_variation_normalizes_to_zero(constant: float) -> None:
    actual = normalize_action_variation(
        np.full(3, constant, dtype=np.float32),
        quantile_low=0.01,
        quantile_high=0.99,
        epsilon=1.0e-8,
    )

    np.testing.assert_array_equal(actual, np.zeros(3, dtype=np.float32))


def test_reliability_is_geometric_mean_of_every_nonempty_metric_subset() -> None:
    components = {
        "action_jump": 0.9,
        "support": 0.125,
        "progress": 0.216,
        "action_variation": 0.343,
        "non_dwell": 0.729,
        "eef_jerk": 0.81,
        "local_path_efficiency": 0.64,
        "low_high_frequency_jitter": 0.49,
        "low_local_backtracking": 0.81,
        "low_action_execution_deviation": 0.36,
        "visual_action_consistency": 0.512,
    }
    for size in range(1, len(RELIABILITY_METRICS) + 1):
        for metrics in itertools.combinations(RELIABILITY_METRICS, size):
            actual = fuse_reliability(
                np.asarray([components["support"]], dtype=np.float32),
                np.asarray([components["progress"]], dtype=np.float32),
                np.asarray([components["action_variation"]], dtype=np.float32),
                np.asarray([components["visual_action_consistency"]], dtype=np.float32),
                metrics,
                min_reliability=0.01,
                action_jump=np.asarray([components["action_jump"]]),
                non_dwell=np.asarray([components["non_dwell"]], dtype=np.float32),
                eef_jerk=np.asarray([components["eef_jerk"]], dtype=np.float32),
                local_path_efficiency=np.asarray([components["local_path_efficiency"]]),
                low_high_frequency_jitter=np.asarray([components["low_high_frequency_jitter"]]),
                low_local_backtracking=np.asarray([components["low_local_backtracking"]]),
                low_action_execution_deviation=np.asarray(
                    [components["low_action_execution_deviation"]]
                ),
            )
            expected = np.prod([components[metric] for metric in metrics]) ** (1.0 / size)

            assert float(actual[0]) == pytest.approx(expected, rel=1.0e-6)


def test_metric_normalization_accepts_every_nonempty_subset_in_canonical_order() -> None:
    for size in range(1, len(RELIABILITY_METRICS) + 1):
        for metrics in itertools.combinations(reversed(RELIABILITY_METRICS), size):
            expected = tuple(metric for metric in RELIABILITY_METRICS if metric in metrics)
            assert normalize_reliability_metrics(metrics) == expected


@pytest.mark.parametrize(
    "metrics",
    [
        (),
        ("support", "support"),
        ("smoothness",),
        "support",
    ],
)
def test_metric_normalization_rejects_empty_duplicate_unknown_and_string_inputs(
    metrics: object,
) -> None:
    with pytest.raises(ValueError, match="reliability metrics"):
        normalize_reliability_metrics(metrics)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([1.0, np.nan], dtype=np.float32),
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([], dtype=np.float32),
    ],
)
def test_action_variation_normalization_rejects_invalid_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="action variation"):
        normalize_action_variation(
            values,
            quantile_low=0.01,
            quantile_high=0.99,
            epsilon=1.0e-8,
        )


def test_reliability_fusion_rejects_mismatched_or_nonfinite_components() -> None:
    with pytest.raises(ValueError, match="reliability components"):
        fuse_reliability(
            np.asarray([0.5], dtype=np.float32),
            np.asarray([0.5, 0.6], dtype=np.float32),
            np.asarray([np.nan], dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
            ("support",),
            min_reliability=0.05,
        )
