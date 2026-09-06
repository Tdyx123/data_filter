from __future__ import annotations

import numpy as np
import pytest

import cocore.visual_action_consistency as vac


def test_step_vac_uses_l2_ratio_and_copies_first_valid_value() -> None:
    visual = np.asarray([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]], dtype=np.float32)
    actions = np.asarray([[0.0, 0.0], [3.0, 4.0], [3.0, 6.0]], dtype=np.float32)

    actual = vac.compute_step_visual_action_consistency(visual, actions, epsilon=1.0)

    np.testing.assert_allclose(actual, [5.0 / 6.0, 5.0 / 6.0, 2.0], atol=1.0e-7)


def test_step_vac_uses_epsilon_when_action_does_not_change() -> None:
    visual = np.asarray([[0.0], [2.0]], dtype=np.float32)
    actions = np.asarray([[1.0], [1.0]], dtype=np.float32)

    actual = vac.compute_step_visual_action_consistency(visual, actions, epsilon=0.5)

    np.testing.assert_allclose(actual, [4.0, 4.0], atol=1.0e-7)


def test_vac_quantile_normalization_clips_both_tails() -> None:
    actual = vac.normalize_visual_action_consistency(
        np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        quantile_low=0.25,
        quantile_high=0.75,
        epsilon=1.0e-8,
    )

    np.testing.assert_allclose(actual, [0.0, 0.0, 0.5, 1.0, 1.0])


@pytest.mark.parametrize("constant", [0.0, 2.0])
def test_constant_vac_normalizes_to_zero(constant: float) -> None:
    actual = vac.normalize_visual_action_consistency(
        np.full(3, constant, dtype=np.float32),
        quantile_low=0.01,
        quantile_high=0.99,
        epsilon=1.0e-8,
    )

    np.testing.assert_array_equal(actual, np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize(
    ("visual", "actions", "epsilon"),
    [
        (np.asarray([], dtype=np.float32), np.zeros((2, 1), dtype=np.float32), 1.0e-8),
        (np.zeros((2, 1), dtype=np.float32), np.zeros((3, 1), dtype=np.float32), 1.0e-8),
        (np.zeros((2, 1), dtype=np.float32), np.asarray([0.0, 1.0]), 1.0e-8),
        (np.asarray([[0.0], [np.inf]]), np.zeros((2, 1), dtype=np.float32), 1.0e-8),
        (np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32), 0.0),
    ],
)
def test_step_vac_rejects_invalid_inputs(
    visual: np.ndarray,
    actions: np.ndarray,
    epsilon: float,
) -> None:
    with pytest.raises(ValueError, match="visual-action consistency"):
        vac.compute_step_visual_action_consistency(visual, actions, epsilon=epsilon)


def test_vac_contract_records_the_fixed_candidate_pipeline() -> None:
    assert vac.visual_action_consistency_contract(
        quantile_low=0.01,
        quantile_high=0.99,
        epsilon=1.0e-8,
    ) == {
        "formula": "l2(v_t-v_t_minus_1)/(l2(a_t-a_t_minus_1)+epsilon)",
        "visual_input": "configured_encoder_full_episode_frame_features",
        "action_input": "robust_scaled_full_episode_actions",
        "visual_difference": "l2_norm_current_minus_previous",
        "action_difference": "l2_norm_current_minus_previous",
        "ratio": "visual_difference_over_action_difference_plus_epsilon",
        "first_step": "copy_first_valid_ratio",
        "clip_aggregation": "top_k_mean",
        "top_k": 3,
        "normalization": "clip_quantile_scale_to_zero_one",
        "quantile_low": 0.01,
        "quantile_high": 0.99,
        "epsilon": 1.0e-8,
    }
