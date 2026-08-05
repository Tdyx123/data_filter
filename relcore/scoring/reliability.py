"""Reliability components used as graph node and edge weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityResult:
    reliability: np.ndarray
    support: np.ndarray
    progress: np.ndarray
    smoothness: np.ndarray
    noop_ratio: np.ndarray


def _motion_without_gripper(values: np.ndarray, gripper_index: int) -> np.ndarray:
    index = gripper_index if gripper_index >= 0 else values.shape[-1] + gripper_index
    if index < 0 or index >= values.shape[-1]:
        raise ValueError("gripper_action_index is outside the action dimension")
    return np.delete(values, index, axis=-1)


def compute_reliability(
    embeddings: np.ndarray,
    state_sequences: np.ndarray,
    action_sequences: np.ndarray,
    visual_progress: np.ndarray,
    *,
    knn: int = 10,
    gripper_progress_weight: float = 0.5,
    visual_progress_weight: float = 0.25,
    noop_threshold: float = 1.0e-4,
    gripper_action_index: int = -1,
    min_reliability: float = 0.05,
    epsilon: float = 1.0e-8,
) -> ReliabilityResult:
    values = np.asarray(embeddings, dtype=np.float32)
    states = np.asarray(state_sequences, dtype=np.float32)
    actions = np.asarray(action_sequences, dtype=np.float32)
    visual = np.asarray(visual_progress, dtype=np.float32)
    count = len(values)
    if (
        values.ndim != 2
        or states.ndim != 3
        or actions.ndim != 3
        or visual.shape != (count,)
        or len(states) != count
        or len(actions) != count
    ):
        raise ValueError("reliability inputs have inconsistent sample dimensions")
    if not all(np.all(np.isfinite(array)) for array in (values, states, actions, visual)):
        raise ValueError("reliability inputs contain NaN or infinity")

    if count == 1:
        support = np.ones(1, dtype=np.float32)
    else:
        from sklearn.neighbors import NearestNeighbors

        effective_k = min(int(knn), count - 1)
        neighbors = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
        distances, _ = neighbors.fit(values).kneighbors(values)
        kth = distances[:, -1]
        support = np.exp(-kth / (np.median(kth) + epsilon)).astype(np.float32)

    state_motion = states[..., :-1] if states.shape[-1] > 1 else states
    state_gripper = states[..., -1]
    ee_delta = np.linalg.norm(state_motion[:, -1] - state_motion[:, 0], axis=1)
    gripper_delta = np.abs(state_gripper[:, -1] - state_gripper[:, 0])
    ee_scale = float(np.std(ee_delta))
    gripper_scale = float(np.std(gripper_delta))
    progress_raw = (
        ee_delta / (ee_scale + epsilon)
        + gripper_progress_weight * gripper_delta / (gripper_scale + epsilon)
        + visual_progress_weight * visual
    )
    progress = (0.5 + 0.5 * (1.0 - np.exp(-progress_raw))).astype(np.float32)

    action_motion = _motion_without_gripper(actions, gripper_action_index)
    acceleration = np.diff(action_motion, axis=1)
    jerk_values = np.diff(acceleration, axis=1)
    if jerk_values.shape[1] == 0:
        jerk = np.zeros(count, dtype=np.float32)
    else:
        jerk = np.median(np.linalg.norm(jerk_values, axis=2), axis=1)
    smoothness = np.exp(-jerk / (np.median(jerk) + epsilon)).astype(np.float32)

    motion_small = np.linalg.norm(action_motion, axis=2) < noop_threshold
    action_gripper = actions[..., gripper_action_index]
    gripper_change = np.abs(np.diff(action_gripper, axis=1, prepend=action_gripper[:, :1]))
    noop_ratio = np.mean(motion_small & (gripper_change < noop_threshold), axis=1).astype(
        np.float32
    )
    reliability = (
        support**0.5 * progress**0.5 * smoothness**0.25 * np.maximum(1.0 - noop_ratio, 0.0) ** 0.5
    )
    reliability = np.clip(reliability, min_reliability, 1.0).astype(np.float32)
    return ReliabilityResult(reliability, support, progress, smoothness, noop_ratio)
