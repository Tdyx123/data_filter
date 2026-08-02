"""Dataset-neutral episode and segment interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


class DatasetValidationError(RuntimeError):
    """Raised when a dataset does not satisfy its adapter contract."""


@dataclass(frozen=True)
class EpisodeRecord:
    """Small, serializable episode index entry."""

    episode_id: int
    length: int


@dataclass
class EpisodeData:
    """One decoded episode before it is split into segments."""

    episode_id: int
    timestamps: np.ndarray
    frame_indices: np.ndarray
    observations: dict[str, np.ndarray]
    actions: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.actions))


@dataclass
class TrajectorySegment:
    """Dataset-neutral segment with inclusive source frame indices."""

    sample_id: str
    episode_id: int
    start_step: int
    end_step: int
    timestamps: np.ndarray
    observations: dict[str, np.ndarray]
    actions: np.ndarray
    kind: str

    @property
    def length(self) -> int:
        return int(len(self.actions))

    def metadata(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_id": self.episode_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "length": self.length,
            "kind": self.kind,
        }


def aligned_chunk_windows(
    length: int,
    chunk_length: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Return inclusive windows, preserving short episodes and the final frame."""

    if length <= 0:
        return []
    if chunk_length <= 0 or stride <= 0:
        raise ValueError("chunk_length and stride must be positive")
    if length <= chunk_length:
        return [(0, length - 1)]
    starts = list(range(0, length - chunk_length + 1, stride))
    final_start = length - chunk_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + chunk_length - 1) for start in starts]


def segment_episode(
    episode: EpisodeData,
    start: int,
    end: int,
    *,
    kind: str,
) -> TrajectorySegment:
    """Slice one decoded episode using inclusive positional offsets."""

    if start < 0 or end < start or end >= episode.length:
        raise ValueError(
            f"invalid segment offsets [{start}, {end}] for episode length {episode.length}"
        )
    index = slice(start, end + 1)
    start_step = int(episode.frame_indices[start])
    end_step = int(episode.frame_indices[end])
    if kind == "trajectory":
        sample_id = f"ep{episode.episode_id:06d}_trajectory"
    elif kind == "chunk":
        sample_id = (
            f"ep{episode.episode_id:06d}_chunk_{start_step:06d}_{end_step:06d}"
        )
    else:
        sample_id = (
            f"ep{episode.episode_id:06d}_{kind}_{start_step:06d}_{end_step:06d}"
        )
    return TrajectorySegment(
        sample_id=sample_id,
        episode_id=episode.episode_id,
        start_step=start_step,
        end_step=end_step,
        timestamps=episode.timestamps[index],
        observations={key: value[index] for key, value in episode.observations.items()},
        actions=episode.actions[index],
        kind=kind,
    )


class DatasetAdapter(ABC):
    """Abstract adapter that exposes decoded episodes and standard segments."""

    @property
    @abstractmethod
    def vector_observation_keys(self) -> tuple[str, ...]:
        """Return vector-valued observation fields selected for encoding."""

    @property
    @abstractmethod
    def image_observation_keys(self) -> tuple[str, ...]:
        """Return image/video observation fields selected for encoding."""

    @abstractmethod
    def episodes(self) -> Sequence[EpisodeRecord]:
        """Return stable episode records."""

    @abstractmethod
    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        """Stream decoded episodes in deterministic order."""

    def iter_segments(
        self,
        modes: Sequence[str],
        *,
        chunk_length: int,
        stride: int,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[TrajectorySegment]:
        """Stream standard trajectory/chunk segments from the episode API."""

        normalized_modes = tuple(dict.fromkeys(str(mode).lower() for mode in modes))
        invalid = sorted(set(normalized_modes) - {"trajectory", "chunk"})
        if invalid:
            raise ValueError(f"Unsupported segmentation modes: {invalid}")
        for episode in self.iter_episodes(
            num_workers=num_workers,
            max_episodes=max_episodes,
            load_images=load_images,
        ):
            if "trajectory" in normalized_modes:
                yield segment_episode(
                    episode,
                    0,
                    episode.length - 1,
                    kind="trajectory",
                )
            if "chunk" in normalized_modes:
                for start, end in aligned_chunk_windows(
                    episode.length,
                    chunk_length,
                    stride,
                ):
                    yield segment_episode(episode, start, end, kind="chunk")

    @abstractmethod
    def fingerprint(self) -> str:
        """Return a stable source fingerprint used to validate caches."""


_ADAPTERS: dict[str, Callable[[Mapping[str, Any]], DatasetAdapter]] = {}


def register_dataset_adapter(
    name: str,
    factory: Callable[[Mapping[str, Any]], DatasetAdapter],
) -> None:
    """Register a dataset adapter factory."""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("adapter name cannot be empty")
    _ADAPTERS[normalized] = factory


def create_dataset(config: Mapping[str, Any]) -> DatasetAdapter:
    """Create the adapter selected by ``dataset.type``."""

    adapter_name = str(config.get("type", "")).strip().lower()
    if adapter_name not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS)) or "(none)"
        raise ValueError(
            f"Unknown dataset adapter {adapter_name!r}; registered adapters: {available}"
        )
    return _ADAPTERS[adapter_name](config)
