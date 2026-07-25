from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuantileStats:
    state_q01: np.ndarray
    state_q99: np.ndarray
    action_q01: np.ndarray
    action_q99: np.ndarray
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        arrays = (
            ("state_q01", self.state_q01),
            ("state_q99", self.state_q99),
            ("action_q01", self.action_q01),
            ("action_q99", self.action_q99),
        )
        for name, value in arrays:
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite 1-D array")
            object.__setattr__(self, name, array)
        if self.state_q01.shape != self.state_q99.shape:
            raise ValueError("State quantiles have different shapes")
        if self.action_q01.shape != self.action_q99.shape:
            raise ValueError("Action quantiles have different shapes")

    @staticmethod
    def _safe_span(low: np.ndarray, high: np.ndarray, epsilon: float) -> np.ndarray:
        span = high - low
        return np.where(np.abs(span) < epsilon, 1.0, span)

    def normalize_state(self, value: np.ndarray, clip: bool = True) -> np.ndarray:
        constant = np.abs(self.state_q99 - self.state_q01) < self.epsilon
        result = (
            2.0
            * (np.asarray(value, dtype=np.float32) - self.state_q01)
            / self._safe_span(self.state_q01, self.state_q99, self.epsilon)
            - 1.0
        )
        result = np.where(constant, 0.0, result)
        return np.clip(result, -1.0, 1.0) if clip else result

    def normalize_action(self, value: np.ndarray, clip: bool = True) -> np.ndarray:
        constant = np.abs(self.action_q99 - self.action_q01) < self.epsilon
        result = (
            2.0
            * (np.asarray(value, dtype=np.float32) - self.action_q01)
            / self._safe_span(self.action_q01, self.action_q99, self.epsilon)
            - 1.0
        )
        result = np.where(constant, 0.0, result)
        return np.clip(result, -1.0, 1.0) if clip else result

    def denormalize_state(self, value: np.ndarray) -> np.ndarray:
        result = (
            (np.asarray(value, dtype=np.float32) + 1.0)
            * 0.5
            * self._safe_span(self.state_q01, self.state_q99, self.epsilon)
            + self.state_q01
        )
        constant = np.abs(self.state_q99 - self.state_q01) < self.epsilon
        return np.where(constant, self.state_q01, result)

    def denormalize_action(self, value: np.ndarray) -> np.ndarray:
        result = (
            (np.asarray(value, dtype=np.float32) + 1.0)
            * 0.5
            * self._safe_span(self.action_q01, self.action_q99, self.epsilon)
            + self.action_q01
        )
        constant = np.abs(self.action_q99 - self.action_q01) < self.epsilon
        return np.where(constant, self.action_q01, result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "q01_q99_to_minus1_plus1",
            "state_q01": self.state_q01.tolist(),
            "state_q99": self.state_q99.tolist(),
            "action_q01": self.action_q01.tolist(),
            "action_q99": self.action_q99.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QuantileStats":
        return cls(
            state_q01=np.asarray(value["state_q01"], dtype=np.float32),
            state_q99=np.asarray(value["state_q99"], dtype=np.float32),
            action_q01=np.asarray(value["action_q01"], dtype=np.float32),
            action_q99=np.asarray(value["action_q99"], dtype=np.float32),
            epsilon=float(value.get("epsilon", 1.0e-6)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "QuantileStats":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if extra:
            payload.update(extra)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(target)
