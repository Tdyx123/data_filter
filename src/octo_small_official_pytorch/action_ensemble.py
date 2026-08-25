from __future__ import annotations

from collections import deque

import numpy as np

from simpler_bridge.evaluation import SimplerEvaluationError


class OfficialTemporalActionEnsembler:
    """Official SimplerEnv chunk alignment specialized to Octo's horizon four."""

    def __init__(self, *, temperature: float = 0.0) -> None:
        if not np.isfinite(temperature):
            raise SimplerEvaluationError("Octo ensemble temperature must be finite")
        self.prediction_horizon = 4
        self.temperature = float(temperature)
        self._history: deque[np.ndarray] = deque(maxlen=self.prediction_horizon)

    def reset(self) -> None:
        self._history.clear()

    def select_action(self, actions: np.ndarray) -> np.ndarray:
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.shape != (1, 4, 7) or not np.all(np.isfinite(chunk)):
            raise SimplerEvaluationError(
                "Official Octo temporal ensemble expects finite (1, 4, 7) actions; "
                f"found {chunk.shape}"
            )
        self._history.append(chunk[0].copy())
        count = len(self._history)
        aligned = np.stack(
            [
                predicted[index]
                for index, predicted in zip(range(count - 1, -1, -1), self._history, strict=True)
            ]
        )
        weights = np.exp(-self.temperature * np.arange(count, dtype=np.float64))
        weights /= weights.sum()
        return np.sum(weights[:, None] * aligned, axis=0).astype(np.float32, copy=False)
