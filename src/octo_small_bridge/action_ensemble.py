"""SimplerEnv-compatible temporal ensemble for Octo action chunks."""

from __future__ import annotations

from collections import deque

import numpy as np

from simpler_bridge.evaluation import SimplerEvaluationError


class OctoTemporalActionEnsembler:
    """Align overlapping Octo chunks and average their current predictions."""

    def __init__(
        self,
        *,
        prediction_horizon: int = 8,
        temperature: float = 0.0,
    ) -> None:
        if (
            isinstance(prediction_horizon, bool)
            or not isinstance(prediction_horizon, int)
            or prediction_horizon <= 0
        ):
            raise SimplerEvaluationError(
                "Octo ensemble prediction_horizon must be a positive integer"
            )
        if not np.isfinite(temperature):
            raise SimplerEvaluationError("Octo ensemble temperature must be finite")
        self.prediction_horizon = prediction_horizon
        self.temperature = float(temperature)
        self._history: deque[np.ndarray] = deque(maxlen=prediction_horizon)

    def reset(self) -> None:
        self._history.clear()

    def select_action(self, actions: np.ndarray) -> np.ndarray:
        chunk = np.asarray(actions, dtype=np.float32)
        expected = (1, self.prediction_horizon, 7)
        if chunk.shape != expected or not np.all(np.isfinite(chunk)):
            raise SimplerEvaluationError(
                f"Octo temporal ensemble expects finite {expected} actions; "
                f"found {chunk.shape}"
            )
        self._history.append(chunk[0].copy())
        count = len(self._history)
        aligned = np.stack(
            [
                predicted[index]
                for index, predicted in zip(
                    range(count - 1, -1, -1),
                    self._history,
                    strict=True,
                )
            ]
        )
        weights = np.exp(-self.temperature * np.arange(count))
        weights /= weights.sum()
        selected = np.sum(weights[:, None] * aligned, axis=0)
        return selected.astype(np.float32, copy=False)
