from __future__ import annotations

import numpy as np
import pytest

from libero_motion_primitives import (
    PrimitiveConfig,
    classify_motion_primitive,
    compute_primitive_statistics,
    filter_frequent_primitives,
    generate_motion_primitives,
    make_libero_config,
)


def _libero_states(delta_by_axis: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    current = np.zeros(8, dtype=np.float64)
    future = current.copy()
    for axis, delta in delta_by_axis.items():
        future[axis] = delta
    return current, future


@pytest.mark.parametrize(
    ("delta_by_axis", "expected"),
    [
        ({}, "stop"),
        ({1: 0.04}, "move left"),
        ({0: 0.04, 1: -0.04, 2: -0.04}, "move forward right down"),
        ({2: 0.04, 7: 0.04}, "move up, open gripper"),
        ({1: 0.04, 5: 0.04}, "move left, rotate counterclockwise"),
        ({7: -0.04}, "close gripper"),
        ({4: 0.04}, "tilt up"),
    ],
)
def test_libero_config_classifies_expected_primitives(
    delta_by_axis: dict[int, float],
    expected: str,
) -> None:
    current, future = _libero_states(delta_by_axis)

    result = classify_motion_primitive(current, future, make_libero_config())

    assert result == expected


def test_values_exactly_at_threshold_are_not_significant() -> None:
    current, future = _libero_states({0: 0.03, 1: -0.03, 2: 0.03, 4: -0.03, 5: 0.03, 7: -0.03})

    result = classify_motion_primitive(current, future, make_libero_config())

    assert result == "stop"


def test_custom_sign_mapping_flips_every_semantic_direction() -> None:
    config = PrimitiveConfig(
        forward_axis=0,
        left_right_axis=1,
        vertical_axis=2,
        tilt_axis=3,
        rotation_axis=4,
        gripper_axis=5,
        forward_positive=False,
        right_positive=False,
        up_positive=False,
        tilt_up_positive=False,
        counterclockwise_positive=False,
        gripper_open_positive=False,
    )
    current = np.zeros(6, dtype=np.float64)
    future = np.full(6, 0.04, dtype=np.float64)

    result = classify_motion_primitive(current, future, config)

    assert result == ("move backward left down, tilt down, rotate clockwise, close gripper")


@pytest.mark.parametrize("horizon", [3, 4, 8])
def test_libero_factory_accepts_supported_horizons(horizon: int) -> None:
    current, future = _libero_states({0: 0.04})

    result = classify_motion_primitive(
        current,
        future,
        make_libero_config(horizon=horizon),
    )

    assert result == "move forward"


@pytest.mark.parametrize("horizon", [2, 9])
def test_config_rejects_horizon_outside_supported_range(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon.*3.*8"):
        make_libero_config(horizon=horizon)


@pytest.mark.parametrize("threshold", [-0.01, np.nan, np.inf])
def test_config_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        make_libero_config(threshold=threshold)


def test_config_rejects_duplicate_semantic_axes() -> None:
    with pytest.raises(ValueError, match="unique"):
        PrimitiveConfig(
            forward_axis=0,
            left_right_axis=0,
            vertical_axis=2,
            tilt_axis=3,
            rotation_axis=4,
            gripper_axis=5,
            forward_positive=True,
            right_positive=True,
            up_positive=True,
            tilt_up_positive=True,
            counterclockwise_positive=True,
            gripper_open_positive=True,
        )


def _forward_ramp_states() -> np.ndarray:
    states = np.zeros((7, 8), dtype=np.float64)
    states[:, 0] = np.arange(7, dtype=np.float64) * 0.02
    return states


def test_truncate_omits_timesteps_without_full_horizon() -> None:
    result = generate_motion_primitives(
        _forward_ramp_states(),
        make_libero_config(horizon=3, tail_strategy="truncate"),
    )

    assert result == ["move forward"] * 4


def test_clip_uses_last_state_for_timesteps_near_tail() -> None:
    result = generate_motion_primitives(
        _forward_ramp_states(),
        make_libero_config(horizon=3, tail_strategy="clip"),
    )

    assert result == ["move forward"] * 5 + ["stop", "stop"]


def test_pad_last_repeats_last_valid_label_to_trajectory_length() -> None:
    states = np.zeros((5, 8), dtype=np.float64)
    states[:, 0] = [0.0, 0.04, 0.04, 0.04, 0.04]

    result = generate_motion_primitives(
        states,
        make_libero_config(horizon=3, tail_strategy="pad_last"),
    )

    assert result == ["move forward", "stop", "stop", "stop", "stop"]


def test_pad_last_fills_too_short_trajectory_with_stop() -> None:
    states = np.zeros((3, 8), dtype=np.float64)
    states[:, 0] = [0.0, 0.04, 0.08]

    result = generate_motion_primitives(
        states,
        make_libero_config(horizon=3, tail_strategy="pad_last"),
    )

    assert result == ["stop", "stop", "stop"]


@pytest.mark.parametrize("tail_strategy", ["truncate", "clip", "pad_last"])
def test_empty_trajectory_returns_no_labels(tail_strategy: str) -> None:
    result = generate_motion_primitives(
        np.empty((0, 8), dtype=np.float64),
        make_libero_config(tail_strategy=tail_strategy),  # type: ignore[arg-type]
    )

    assert result == []


def test_truncate_returns_no_labels_for_too_short_trajectory() -> None:
    result = generate_motion_primitives(
        np.zeros((3, 8), dtype=np.float64),
        make_libero_config(horizon=3, tail_strategy="truncate"),
    )

    assert result == []


def test_clip_still_labels_a_too_short_trajectory() -> None:
    states = np.zeros((3, 8), dtype=np.float64)
    states[:, 0] = [0.0, 0.02, 0.04]

    result = generate_motion_primitives(
        states,
        make_libero_config(horizon=3, tail_strategy="clip"),
    )

    assert result == ["move forward", "stop", "stop"]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_trajectory_rejects_non_finite_values(bad_value: float) -> None:
    states = np.zeros((5, 8), dtype=np.float64)
    states[2, 4] = bad_value

    with pytest.raises(ValueError, match="NaN or infinity"):
        generate_motion_primitives(states, make_libero_config())


def test_trajectory_rejects_non_numeric_values() -> None:
    states = np.full((5, 8), "not-a-number", dtype=object)

    with pytest.raises(ValueError, match="numeric"):
        generate_motion_primitives(states, make_libero_config())


def test_trajectory_must_be_two_dimensional() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        generate_motion_primitives(np.zeros(8), make_libero_config())


def test_trajectory_dimension_must_cover_configured_indices() -> None:
    with pytest.raises(ValueError, match="dimension 7.*axis index 7"):
        generate_motion_primitives(np.zeros((5, 7)), make_libero_config())


def test_classification_rejects_mismatched_state_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        classify_motion_primitive(
            np.zeros(8),
            np.zeros(9),
            make_libero_config(),
        )


def test_classification_rejects_non_vector_state() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        classify_motion_primitive(
            np.zeros((1, 8)),
            np.zeros((1, 8)),
            make_libero_config(),
        )


def test_config_rejects_unknown_tail_strategy() -> None:
    with pytest.raises(ValueError, match="tail_strategy"):
        make_libero_config(tail_strategy="unknown")  # type: ignore[arg-type]


def test_statistics_count_proportions_and_sort_deterministically() -> None:
    result = compute_primitive_statistics(
        ["stop", "move left", "stop", "move left", "open gripper"]
    )

    assert result == [
        ("move left", 2, 0.4),
        ("stop", 2, 0.4),
        ("open gripper", 1, 0.2),
    ]


def test_statistics_of_empty_input_are_empty() -> None:
    assert compute_primitive_statistics([]) == []


def test_statistics_reject_non_string_labels() -> None:
    with pytest.raises(TypeError, match="strings"):
        compute_primitive_statistics(["stop", 7])  # type: ignore[list-item]


def test_frequency_filter_includes_exact_threshold() -> None:
    statistics = [
        ("at threshold", 1, 0.001),
        ("below threshold", 1, 0.0009),
        ("common", 998, 0.9981),
    ]

    result = filter_frequent_primitives(statistics)

    assert result == [
        ("at threshold", 1, 0.001),
        ("common", 998, 0.9981),
    ]


@pytest.mark.parametrize("min_frequency", [-0.01, 1.01, np.nan, np.inf])
def test_frequency_filter_rejects_invalid_minimum(min_frequency: float) -> None:
    with pytest.raises(ValueError, match="min_frequency"):
        filter_frequent_primitives([], min_frequency=min_frequency)


def test_frequency_filter_rejects_invalid_statistic_frequency() -> None:
    with pytest.raises(ValueError, match="frequency"):
        filter_frequent_primitives([("stop", 1, np.nan)])
