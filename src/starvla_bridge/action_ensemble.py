"""StarVLA-compatible adaptive action-chunk integration."""

from __future__ import annotations

from collections import deque

import numpy as np

from simpler_bridge.evaluation import SimplerEvaluationError


class AdaptiveActionEnsembler:
    """Align and integrate overlapping action chunks at the current timestep."""

    def __init__(self, *, horizon: int = 7, alpha: float = 0.1) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise SimplerEvaluationError("adaptive ensemble horizon must be a positive integer")
        if not np.isfinite(alpha):
            raise SimplerEvaluationError("adaptive ensemble alpha must be finite")
        self.horizon = horizon
        self.alpha = float(alpha)
        self._history: deque[np.ndarray] = deque(maxlen=horizon)

    def reset(self) -> None:
        self._history.clear()

    def select_action(self, actions: np.ndarray) -> np.ndarray:
        chunk = np.asarray(actions, dtype=np.float32)
        if (
            chunk.ndim != 3
            or chunk.shape[0] != 1
            or chunk.shape[1] < self.horizon
            or chunk.shape[2] != 7
            or not np.all(np.isfinite(chunk))
        ):
            raise SimplerEvaluationError(
                "adaptive ensemble expects finite actions with shape "
                f"(1, T, 7) and T >= {self.horizon}; found {chunk.shape}"
            )

        binary_chunk = chunk[0].copy()
        binary_chunk[:, 6] = (binary_chunk[:, 6] > 0.5).astype(np.float32)
        self._history.append(binary_chunk)

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
        reference = aligned[-1]
        cosine = np.sum(aligned * reference, axis=1) / (
            np.linalg.norm(aligned, axis=1) * np.linalg.norm(reference) + 1.0e-7
        )
        weights = np.exp(self.alpha * cosine)
        weights /= weights.sum()
        return np.sum(weights[:, None] * aligned, axis=0).astype(np.float32, copy=False)
