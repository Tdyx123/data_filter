"""Directional within-clip visual/state/action relation features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_sequence(values: np.ndarray, name: str) -> np.ndarray:
    sequence = np.asarray(values, dtype=np.float32)
    if sequence.ndim != 2 or len(sequence) == 0 or not np.all(np.isfinite(sequence)):
        raise ValueError(f"{name} must be a finite non-empty [time, dim] array")
    return sequence


def _first_order(sequence: np.ndarray) -> np.ndarray:
    return np.concatenate([sequence.mean(axis=0), sequence.std(axis=0), sequence[-1] - sequence[0]])


@dataclass
class RelationEncoder:
    projection_dim: int = 32
    lags: tuple[int, ...] = (0, 1, 2, 4)
    seed: int = 42
    projection_matrices: dict[str, np.ndarray] = field(default_factory=dict)

    def _ensure_projections(self, visual_dim: int, state_dim: int, action_dim: int) -> None:
        dimensions = {"visual": visual_dim, "state": state_dim, "action": action_dim}
        if self.projection_matrices:
            for name, input_dim in dimensions.items():
                if self.projection_matrices[name].shape != (input_dim, self.projection_dim):
                    raise ValueError(f"{name} dimension changed after fitting projections")
            return
        rng = np.random.default_rng(self.seed)
        scale = np.sqrt(float(self.projection_dim))
        self.projection_matrices = {
            name: (rng.standard_normal((input_dim, self.projection_dim)) / scale).astype(np.float32)
            for name, input_dim in dimensions.items()
        }

    @staticmethod
    def _relation(left: np.ndarray, right: np.ndarray, lag: int) -> np.ndarray:
        count = min(len(left), len(right))
        if lag < 0 or lag >= count:
            raise ValueError(f"lag {lag} is invalid for relation length {count}")
        return (left[: count - lag] * right[lag:count]).mean(axis=0)

    def encode_raw(
        self,
        visual: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        visual_values = _as_sequence(visual, "visual")
        state_values = _as_sequence(state, "state")
        action_values = _as_sequence(action, "action")
        if len({len(visual_values), len(state_values), len(action_values)}) != 1:
            raise ValueError("visual/state/action sequences must have the same length")
        visual_values = visual_values / np.maximum(
            np.linalg.norm(visual_values, axis=1, keepdims=True), 1.0e-8
        )
        self._ensure_projections(
            visual_values.shape[1], state_values.shape[1], action_values.shape[1]
        )
        projected = {
            "visual": visual_values @ self.projection_matrices["visual"],
            "state": state_values @ self.projection_matrices["state"],
            "action": action_values @ self.projection_matrices["action"],
        }
        projected["delta_visual"] = np.diff(projected["visual"], axis=0)
        projected["delta_state"] = np.diff(projected["state"], axis=0)
        projected["delta_action"] = np.diff(projected["action"], axis=0)
        features = [
            _first_order(projected[name])
            for name in (
                "visual",
                "state",
                "action",
                "delta_visual",
                "delta_state",
                "delta_action",
            )
        ]
        pairs = (
            ("visual", "visual"),
            ("state", "state"),
            ("action", "action"),
            ("visual", "action"),
            ("state", "action"),
            ("delta_visual", "action"),
            ("delta_state", "action"),
        )
        for left_name, right_name in pairs:
            for lag in self.lags:
                features.append(
                    self._relation(projected[left_name], projected[right_name], int(lag))
                )
        return np.concatenate(features).astype(np.float32)
