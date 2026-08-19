"""Bridge V2 q01/q99 and binary-gripper normalization contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_data import DatasetValidationError

from .config import BRIDGE_V2_NORMALIZATION_CONTRACT


STATE_DIM = 8
ACTION_DIM = 7
CONTINUOUS_CLIP = np.float32(2.2)


@dataclass(frozen=True)
class BridgeV2NormalizationStatistics:
    state_q01: np.ndarray
    state_q99: np.ndarray
    action_q01: np.ndarray
    action_q99: np.ndarray
    metadata_sha256: str
    retained_episodes: int
    retained_frames: int
    epsilon: float = 1.0e-6

    @property
    def contract(self) -> str:
        return BRIDGE_V2_NORMALIZATION_CONTRACT

    def __post_init__(self) -> None:
        dimensions = {
            "state_q01": (self.state_q01, STATE_DIM),
            "state_q99": (self.state_q99, STATE_DIM),
            "action_q01": (self.action_q01, ACTION_DIM),
            "action_q99": (self.action_q99, ACTION_DIM),
        }
        for name, (value, dimension) in dimensions.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (dimension,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape ({dimension},)")
            object.__setattr__(self, name, array)
        if np.any(self.state_q01 > self.state_q99):
            raise ValueError("state q01 values must not exceed q99")
        if np.any(self.action_q01 > self.action_q99):
            raise ValueError("action q01 values must not exceed q99")
        if not self.metadata_sha256:
            raise ValueError("metadata_sha256 must not be empty")
        if self.retained_episodes <= 0 or self.retained_frames <= 0:
            raise ValueError("retained episode and frame counts must be positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

    def _normalize_continuous(
        self,
        value: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
    ) -> np.ndarray:
        span = high - low
        constant = np.abs(span) < self.epsilon
        safe_span = np.where(constant, np.float32(1.0), span)
        normalized = np.float32(2.0) * (value - low) / safe_span - np.float32(1.0)
        normalized = np.where(constant, np.float32(0.0), normalized)
        return np.clip(normalized, -CONTINUOUS_CLIP, CONTINUOUS_CLIP)

    @staticmethod
    def _array(value: Any, *, dimension: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim < 1 or array.shape[-1] != dimension:
            raise ValueError(f"{name} must have last dimension {dimension}, found {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values")
        return array

    def normalize_state(self, value: Any) -> np.ndarray:
        array = self._array(value, dimension=STATE_DIM, name="Bridge state")
        continuous = self._normalize_continuous(
            array[..., :-1], self.state_q01[:-1], self.state_q99[:-1]
        )
        gripper = (array[..., -1:] > np.float32(0.5)).astype(np.float32)
        return np.concatenate([continuous, gripper], axis=-1).astype(
            np.float32, copy=False
        )

    def normalize_action(self, value: Any) -> np.ndarray:
        array = self._array(value, dimension=ACTION_DIM, name="Bridge action")
        continuous = self._normalize_continuous(
            array[..., :-1], self.action_q01[:-1], self.action_q99[:-1]
        )
        gripper = (array[..., -1:] > np.float32(0.5)).astype(np.float32)
        return np.concatenate([continuous, gripper], axis=-1).astype(
            np.float32, copy=False
        )

    def denormalize_action(self, value: Any) -> np.ndarray:
        array = self._array(value, dimension=ACTION_DIM, name="Normalized action")
        span = self.action_q99[:-1] - self.action_q01[:-1]
        constant = np.abs(span) < self.epsilon
        continuous = (
            (array[..., :-1] + np.float32(1.0))
            * np.float32(0.5)
            * np.where(constant, np.float32(1.0), span)
            + self.action_q01[:-1]
        )
        continuous = np.where(constant, self.action_q01[:-1], continuous)
        gripper = (array[..., -1:] > np.float32(0.5)).astype(np.float32)
        return np.concatenate([continuous, gripper], axis=-1).astype(
            np.float32, copy=False
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "method": "q01_q99_continuous_clip_2.2_binary_gripper",
            "state_q01": self.state_q01.tolist(),
            "state_q99": self.state_q99.tolist(),
            "action_q01": self.action_q01.tolist(),
            "action_q99": self.action_q99.tolist(),
            "metadata_sha256": self.metadata_sha256,
            "retained_episodes": self.retained_episodes,
            "retained_frames": self.retained_frames,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BridgeV2NormalizationStatistics":
        if value.get("contract") != BRIDGE_V2_NORMALIZATION_CONTRACT:
            raise ValueError(
                "normalization contract must equal "
                f"{BRIDGE_V2_NORMALIZATION_CONTRACT!r}"
            )
        return cls(
            state_q01=np.asarray(value["state_q01"], dtype=np.float32),
            state_q99=np.asarray(value["state_q99"], dtype=np.float32),
            action_q01=np.asarray(value["action_q01"], dtype=np.float32),
            action_q99=np.asarray(value["action_q99"], dtype=np.float32),
            metadata_sha256=str(value["metadata_sha256"]),
            retained_episodes=int(value["retained_episodes"]),
            retained_frames=int(value["retained_frames"]),
            epsilon=float(value.get("epsilon", 1.0e-6)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BridgeV2NormalizationStatistics":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(target)


def compute_bridge_v2_statistics(
    adapter: Any,
    cache_path: str | Path,
    *,
    epsilon: float = 1.0e-6,
) -> BridgeV2NormalizationStatistics:
    """Compute exact quantiles from the adapter's filtered episode index."""

    cache = Path(cache_path)
    records = tuple(adapter.episodes())
    metadata_sha256 = str(adapter.fingerprint())
    retained_frames = sum(int(record.length) for record in records)
    if cache.is_file():
        try:
            cached = BridgeV2NormalizationStatistics.load(cache)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            cached = None
        if (
            cached is not None
            and cached.metadata_sha256 == metadata_sha256
            and cached.retained_episodes == len(records)
            and cached.retained_frames == retained_frames
            and cached.epsilon == float(epsilon)
        ):
            return cached

    if not records or retained_frames <= 0:
        raise DatasetValidationError("Bridge normalization requires retained frames")
    state_key = adapter.vector_observation_keys[0]
    states = np.empty((retained_frames, STATE_DIM), dtype=np.float32)
    actions = np.empty((retained_frames, ACTION_DIM), dtype=np.float32)
    offset = 0
    for record in records:
        episode = adapter.load_episode(record, load_images=False)
        state = np.asarray(episode.observations[state_key], dtype=np.float32)
        action = np.asarray(episode.actions, dtype=np.float32)
        expected_state = (int(record.length), STATE_DIM)
        expected_action = (int(record.length), ACTION_DIM)
        if state.shape != expected_state or not np.all(np.isfinite(state)):
            raise DatasetValidationError(
                f"Episode {record.episode_id} state must be finite with shape {expected_state}"
            )
        if action.shape != expected_action or not np.all(np.isfinite(action)):
            raise DatasetValidationError(
                f"Episode {record.episode_id} action must be finite with shape {expected_action}"
            )
        end = offset + int(record.length)
        states[offset:end] = state
        actions[offset:end] = action
        offset = end
    if offset != retained_frames:
        raise DatasetValidationError(
            f"Read {offset} Bridge frames, expected {retained_frames}"
        )

    statistics = BridgeV2NormalizationStatistics(
        state_q01=np.quantile(states, 0.01, axis=0).astype(np.float32),
        state_q99=np.quantile(states, 0.99, axis=0).astype(np.float32),
        action_q01=np.quantile(actions, 0.01, axis=0).astype(np.float32),
        action_q99=np.quantile(actions, 0.99, axis=0).astype(np.float32),
        metadata_sha256=metadata_sha256,
        retained_episodes=len(records),
        retained_frames=retained_frames,
        epsilon=float(epsilon),
    )
    statistics.save(cache)
    return statistics
