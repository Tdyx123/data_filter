"""Generate ECoT motion-primitive labels from robot state changes."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, Sequence, TypeAlias

import numpy as np

TailStrategy: TypeAlias = Literal["truncate", "clip", "pad_last"]


@dataclass(frozen=True, kw_only=True)
class PrimitiveConfig:
    """Configuration mapping state-vector axes and signs to motion semantics.

    Every ``*_positive`` field states whether a positive change along the
    corresponding raw state axis maps to the positive semantic named by that
    field. For example, ``right_positive=False`` maps a negative raw change to
    ``"right"`` and a positive raw change to ``"left"``.
    """

    horizon: int = 4
    threshold: float = 0.03

    forward_axis: int
    left_right_axis: int
    vertical_axis: int
    tilt_axis: int
    rotation_axis: int
    gripper_axis: int

    forward_positive: bool
    right_positive: bool
    up_positive: bool
    tilt_up_positive: bool
    counterclockwise_positive: bool
    gripper_open_positive: bool

    tail_strategy: TailStrategy = "truncate"

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, Integral):
            raise ValueError("horizon must be an integer between 3 and 8")
        if not 3 <= int(self.horizon) <= 8:
            raise ValueError("horizon must be between 3 and 8 inclusive")
        object.__setattr__(self, "horizon", int(self.horizon))

        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, Real)
            or not math.isfinite(float(self.threshold))
            or float(self.threshold) < 0.0
        ):
            raise ValueError("threshold must be a finite, non-negative number")
        object.__setattr__(self, "threshold", float(self.threshold))

        if self.tail_strategy not in {"truncate", "clip", "pad_last"}:
            raise ValueError("tail_strategy must be one of 'truncate', 'clip', or 'pad_last'")

        axis_fields = (
            "forward_axis",
            "left_right_axis",
            "vertical_axis",
            "tilt_axis",
            "rotation_axis",
            "gripper_axis",
        )
        axes: list[int] = []
        for field_name in axis_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
            normalized = int(value)
            object.__setattr__(self, field_name, normalized)
            axes.append(normalized)
        if len(set(axes)) != len(axes):
            raise ValueError("semantic axis indices must be unique")

        sign_fields = (
            "forward_positive",
            "right_positive",
            "up_positive",
            "tilt_up_positive",
            "counterclockwise_positive",
            "gripper_open_positive",
        )
        for field_name in sign_fields:
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a bool")


def make_libero_config(
    *,
    horizon: int = 4,
    threshold: float = 0.03,
    tail_strategy: TailStrategy = "truncate",
) -> PrimitiveConfig:
    """Return the explicit primitive mapping for this repository's LIBERO state.

    The expected state layout is ``[x, y, z, rx, ry, rz, placeholder,
    gripper_qpos]``. Positive x is forward, negative y is right, positive z is
    up, positive ry tilts up, positive rz rotates counterclockwise, and a
    positive gripper position change opens the gripper.
    """
    return PrimitiveConfig(
        horizon=horizon,
        threshold=threshold,
        forward_axis=0,
        left_right_axis=1,
        vertical_axis=2,
        tilt_axis=4,
        rotation_axis=5,
        gripper_axis=7,
        forward_positive=True,
        right_positive=False,
        up_positive=True,
        tilt_up_positive=True,
        counterclockwise_positive=True,
        gripper_open_positive=True,
        tail_strategy=tail_strategy,
    )


def _as_state_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, found shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _required_state_dimension(config: PrimitiveConfig) -> int:
    return 1 + max(
        config.forward_axis,
        config.left_right_axis,
        config.vertical_axis,
        config.tilt_axis,
        config.rotation_axis,
        config.gripper_axis,
    )


def _mapped_token(
    delta: float,
    *,
    threshold: float,
    positive_token: str,
    negative_token: str,
    positive_delta_is_positive_token: bool,
) -> str | None:
    if delta > threshold:
        return positive_token if positive_delta_is_positive_token else negative_token
    if delta < -threshold:
        return negative_token if positive_delta_is_positive_token else positive_token
    return None


def classify_motion_primitive(
    current_state: np.ndarray,
    future_state: np.ndarray,
    config: PrimitiveConfig,
) -> str:
    """Classify the motion between two state vectors as one ECoT label.

    Args:
        current_state: Current one-dimensional robot state.
        future_state: Future state with the same shape as ``current_state``.
        config: Axis, sign, threshold, and horizon configuration.

    Returns:
        A dynamically composed English motion-primitive label, or ``"stop"``.

    Raises:
        ValueError: If either state is malformed, non-finite, mismatched, or too
            short for the configured semantic axes.
    """
    current = _as_state_vector(current_state, name="current_state")
    future = _as_state_vector(future_state, name="future_state")
    if current.shape != future.shape:
        raise ValueError(
            "current_state and future_state must have the same shape, "
            f"found {current.shape} and {future.shape}"
        )
    required_dimension = _required_state_dimension(config)
    if current.shape[0] < required_dimension:
        raise ValueError(
            f"state dimension {current.shape[0]} is too small for configured "
            f"axis index {required_dimension - 1}"
        )

    delta = future - current
    move_tokens = [
        _mapped_token(
            float(delta[config.forward_axis]),
            threshold=config.threshold,
            positive_token="forward",
            negative_token="backward",
            positive_delta_is_positive_token=config.forward_positive,
        ),
        _mapped_token(
            float(delta[config.left_right_axis]),
            threshold=config.threshold,
            positive_token="right",
            negative_token="left",
            positive_delta_is_positive_token=config.right_positive,
        ),
        _mapped_token(
            float(delta[config.vertical_axis]),
            threshold=config.threshold,
            positive_token="up",
            negative_token="down",
            positive_delta_is_positive_token=config.up_positive,
        ),
    ]
    blocks: list[str] = []
    active_move_tokens = [token for token in move_tokens if token is not None]
    if active_move_tokens:
        blocks.append("move " + " ".join(active_move_tokens))

    tilt = _mapped_token(
        float(delta[config.tilt_axis]),
        threshold=config.threshold,
        positive_token="tilt up",
        negative_token="tilt down",
        positive_delta_is_positive_token=config.tilt_up_positive,
    )
    if tilt is not None:
        blocks.append(tilt)

    rotation = _mapped_token(
        float(delta[config.rotation_axis]),
        threshold=config.threshold,
        positive_token="rotate counterclockwise",
        negative_token="rotate clockwise",
        positive_delta_is_positive_token=config.counterclockwise_positive,
    )
    if rotation is not None:
        blocks.append(rotation)

    gripper = _mapped_token(
        float(delta[config.gripper_axis]),
        threshold=config.threshold,
        positive_token="open gripper",
        negative_token="close gripper",
        positive_delta_is_positive_token=config.gripper_open_positive,
    )
    if gripper is not None:
        blocks.append(gripper)

    return ", ".join(blocks) if blocks else "stop"


def _as_state_matrix(states: np.ndarray, config: PrimitiveConfig) -> np.ndarray:
    try:
        array = np.asarray(states, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("states must contain numeric values") from error
    if array.ndim != 2:
        raise ValueError(f"states must be two-dimensional [T, D], found shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("states contains NaN or infinity")
    required_dimension = _required_state_dimension(config)
    if array.shape[1] < required_dimension:
        raise ValueError(
            f"state dimension {array.shape[1]} is too small for configured "
            f"axis index {required_dimension - 1}"
        )
    return array


def generate_motion_primitives(
    states: np.ndarray,
    config: PrimitiveConfig,
) -> list[str]:
    """Generate ECoT motion-primitive labels for a ``[T, D]`` trajectory.

    ``truncate`` omits timesteps without a complete future horizon. ``clip``
    compares those timesteps with the final state. ``pad_last`` repeats the
    final valid label; when no valid comparison exists it emits ``"stop"`` for
    every timestep.

    Args:
        states: Finite numeric state sequence with shape ``[T, D]``.
        config: Primitive configuration, including horizon and tail strategy.

    Returns:
        Labels ordered by their corresponding current timestep.

    Raises:
        ValueError: If ``states`` is malformed, non-finite, or too narrow for
            the configured semantic axes.
    """
    state_values = _as_state_matrix(states, config)
    trajectory_length = int(state_values.shape[0])
    if trajectory_length == 0:
        return []

    if config.tail_strategy == "truncate":
        valid_count = max(trajectory_length - config.horizon, 0)
        return [
            classify_motion_primitive(
                state_values[timestep],
                state_values[timestep + config.horizon],
                config,
            )
            for timestep in range(valid_count)
        ]

    if config.tail_strategy == "clip":
        final_index = trajectory_length - 1
        return [
            classify_motion_primitive(
                state_values[timestep],
                state_values[min(timestep + config.horizon, final_index)],
                config,
            )
            for timestep in range(trajectory_length)
        ]

    valid_count = trajectory_length - config.horizon
    if valid_count <= 0:
        return ["stop"] * trajectory_length
    labels = [
        classify_motion_primitive(
            state_values[timestep],
            state_values[timestep + config.horizon],
            config,
        )
        for timestep in range(valid_count)
    ]
    labels.extend([labels[-1]] * (trajectory_length - len(labels)))
    return labels


def compute_primitive_statistics(
    primitives: Sequence[str],
) -> list[tuple[str, int, float]]:
    """Count primitive labels and return deterministic frequency statistics.

    Results are ordered by descending count, then alphabetically by label for
    equal counts. An empty input produces an empty result.

    Args:
        primitives: Generated primitive labels.

    Returns:
        ``(label, count, proportion)`` tuples.

    Raises:
        TypeError: If any primitive label is not a string.
    """
    labels = list(primitives)
    if not all(isinstance(label, str) for label in labels):
        raise TypeError("primitive labels must be strings")
    if not labels:
        return []

    counts = Counter(labels)
    total = len(labels)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [(label, count, count / total) for label, count in ordered]


def filter_frequent_primitives(
    statistics: Sequence[tuple[str, int, float]],
    min_frequency: float = 0.001,
) -> list[tuple[str, int, float]]:
    """Return statistics whose proportion is at least ``min_frequency``.

    This helper is only for distribution analysis. It is intentionally not
    called by either label-generation function and never removes samples.

    Args:
        statistics: ``(label, count, proportion)`` tuples, typically from
            :func:`compute_primitive_statistics`.
        min_frequency: Inclusive minimum proportion in the closed interval
            ``[0, 1]``.

    Returns:
        Validated statistics in their original order whose proportions meet
        the threshold.

    Raises:
        TypeError: If a statistics entry has invalid field types or shape.
        ValueError: If a count, frequency, or minimum frequency is outside its
            valid range or is not finite.
    """
    if (
        isinstance(min_frequency, bool)
        or not isinstance(min_frequency, Real)
        or not math.isfinite(float(min_frequency))
        or not 0.0 <= float(min_frequency) <= 1.0
    ):
        raise ValueError("min_frequency must be a finite number between 0 and 1")
    minimum = float(min_frequency)

    filtered: list[tuple[str, int, float]] = []
    for index, statistic in enumerate(statistics):
        if not isinstance(statistic, (tuple, list)) or len(statistic) != 3:
            raise TypeError(f"statistics entry {index} must contain label, count, frequency")
        label, count, frequency = statistic
        if not isinstance(label, str):
            raise TypeError(f"statistics entry {index} label must be a string")
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(f"statistics entry {index} count must be an integer")
        if int(count) < 0:
            raise ValueError(f"statistics entry {index} count must be non-negative")
        if isinstance(frequency, bool) or not isinstance(frequency, Real):
            raise TypeError(f"statistics entry {index} frequency must be numeric")
        normalized_frequency = float(frequency)
        if not math.isfinite(normalized_frequency) or not 0.0 <= normalized_frequency <= 1.0:
            raise ValueError(
                f"statistics entry {index} frequency must be finite and between 0 and 1"
            )
        normalized = (label, int(count), normalized_frequency)
        if normalized_frequency >= minimum:
            filtered.append(normalized)
    return filtered


__all__ = [
    "PrimitiveConfig",
    "TailStrategy",
    "classify_motion_primitive",
    "compute_primitive_statistics",
    "filter_frequent_primitives",
    "generate_motion_primitives",
    "make_libero_config",
]
