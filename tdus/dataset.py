"""Dataset adapters and trajectory/chunk construction.

Only this module understands the LeRobot on-disk layout.  All downstream
modules receive the same ``TrajectorySegment`` representation and are therefore
independent of Bridge, WidowX, or any other concrete robot dataset.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


class DatasetValidationError(RuntimeError):
    """Raised when a dataset does not satisfy its adapter's contract."""


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
    """Dataset-neutral input consumed by encoders and quality metrics.

    ``start_step`` and ``end_step`` are inclusive frame indices.  Observation
    arrays always have time as their first dimension.
    """

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


class DatasetAdapter(ABC):
    """Abstract adapter required by the TDUS pipeline."""

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
        """Stream trajectory/chunk segments in deterministic order."""

    @abstractmethod
    def fingerprint(self) -> str:
        """Return a stable source fingerprint used to validate caches."""


_ADAPTERS: dict[str, Callable[[Mapping[str, Any]], DatasetAdapter]] = {}


def register_dataset_adapter(
    name: str,
    factory: Callable[[Mapping[str, Any]], DatasetAdapter],
) -> None:
    """Register a dataset adapter factory.

    Registration is intentionally tiny: a third-party dataset only needs to
    construct a ``DatasetAdapter`` from its section of ``config.yaml``.
    """

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


def aligned_chunk_windows(length: int, chunk_length: int, stride: int) -> list[tuple[int, int]]:
    """Return inclusive windows, preserving short episodes and the final frame.

    Regular stride-aligned full windows are emitted first.  If the final regular
    window does not reach the episode end, one extra full window is aligned to
    the end.  This avoids padding and avoids tiny tail-only chunks.
    """

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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise DatasetValidationError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
    return rows


def _decode_video(path: Path) -> np.ndarray:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required to decode LeRobot video observations") from error
    frames: list[np.ndarray] = []
    try:
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception as error:
        raise DatasetValidationError(f"Could not decode video {path}: {error}") from error
    if not frames:
        raise DatasetValidationError(f"Video contains no frames: {path}")
    return np.stack(frames)


def _load_lerobot_episode(payload: Mapping[str, Any]) -> EpisodeData:
    """Process-safe LeRobot episode loader used by the multiprocessing path."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to read LeRobot Parquet files") from error

    root = Path(str(payload["root"]))
    episode_id = int(payload["episode_id"])
    expected_length = int(payload["length"])
    chunks_size = int(payload["chunks_size"])
    chunk_id = episode_id // chunks_size
    format_values = {"episode_chunk": chunk_id, "episode_index": episode_id}
    parquet_path = root / str(payload["data_path"]).format(**format_values)
    if not parquet_path.is_file():
        raise DatasetValidationError(f"Missing episode Parquet file: {parquet_path}")

    vector_keys = tuple(payload["vector_keys"])
    action_key = str(payload["action_key"])
    timestamp_key = str(payload["timestamp_key"])
    frame_key = str(payload["frame_key"])
    episode_key = str(payload["episode_key"])
    columns = list(dict.fromkeys([*vector_keys, action_key, timestamp_key, frame_key, episode_key]))
    table = pq.read_table(parquet_path, columns=columns)
    if len(table) != expected_length:
        raise DatasetValidationError(
            f"Episode {episode_id}: Parquet rows={len(table)}, metadata length={expected_length}"
        )

    def column_array(key: str, dtype: Any) -> np.ndarray:
        return np.asarray(table[key].to_pylist(), dtype=dtype)

    actions = column_array(action_key, np.float32)
    timestamps = column_array(timestamp_key, np.float64).reshape(-1)
    frame_indices = column_array(frame_key, np.int64).reshape(-1)
    episode_indices = column_array(episode_key, np.int64).reshape(-1)
    if actions.ndim != 2:
        raise DatasetValidationError(
            f"Episode {episode_id}: action must have shape [time, dim], got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(timestamps)):
        raise DatasetValidationError(
            f"Episode {episode_id}: action/timestamp contains NaN or infinity"
        )
    if np.any(episode_indices != episode_id):
        raise DatasetValidationError(f"Episode {episode_id}: inconsistent episode index column")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
        raise DatasetValidationError(f"Episode {episode_id}: timestamps are not monotonic")
    if len(frame_indices) > 1 and np.any(np.diff(frame_indices) <= 0):
        raise DatasetValidationError(f"Episode {episode_id}: frame indices are not increasing")

    observations: dict[str, np.ndarray] = {}
    for key in vector_keys:
        values = column_array(key, np.float32)
        if len(values) != expected_length or not np.all(np.isfinite(values)):
            raise DatasetValidationError(
                f"Episode {episode_id}, {key}: invalid length or non-finite values"
            )
        observations[key] = values
    video_path_pattern = str(payload["video_path"])
    for image_key in payload["image_keys"]:
        video_values = {**format_values, "video_key": image_key}
        video_path = root / video_path_pattern.format(**video_values)
        frames = _decode_video(video_path)
        if len(frames) != expected_length:
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}: video frames={len(frames)}, "
                f"expected={expected_length}"
            )
        observations[str(image_key)] = frames

    return EpisodeData(
        episode_id=episode_id,
        timestamps=timestamps,
        frame_indices=frame_indices,
        observations=observations,
        actions=actions,
    )


class LeRobotDatasetAdapter(DatasetAdapter):
    """LeRobot v2 adapter driven entirely by ``meta/info.json``."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.root = Path(str(config["path"])).expanduser().resolve()
        info_path = self.root / "meta" / "info.json"
        episodes_path = self.root / "meta" / "episodes.jsonl"
        missing = [str(path) for path in (info_path, episodes_path) if not path.is_file()]
        if missing:
            raise DatasetValidationError(f"Missing LeRobot metadata files: {missing}")
        with info_path.open("r", encoding="utf-8") as handle:
            self.info = json.load(handle)
        if not str(self.info.get("codebase_version", "")).startswith("v2"):
            raise DatasetValidationError(
                f"LeRobot adapter currently supports v2 datasets, found "
                f"{self.info.get('codebase_version')!r}"
            )
        self.features: dict[str, dict[str, Any]] = dict(self.info.get("features", {}))
        self._discover_keys()
        rows = _read_jsonl(episodes_path)
        self._episodes = tuple(
            EpisodeRecord(int(row["episode_index"]), int(row["length"])) for row in rows
        )
        if any(episode.length <= 0 for episode in self._episodes):
            raise DatasetValidationError("LeRobot episodes must contain at least one frame")
        expected = int(self.info.get("total_episodes", len(self._episodes)))
        if len(self._episodes) != expected:
            raise DatasetValidationError(
                f"episodes.jsonl has {len(self._episodes)} entries, expected {expected}"
            )

    def _discover_keys(self) -> None:
        overrides = dict(self.config.get("feature_keys", {}))

        def choose_key(
            override_name: str,
            preferred: str,
            predicate: Callable[[str], bool],
        ) -> str:
            override = overrides.get(override_name, "auto")
            if override not in {None, "auto"}:
                return str(override)
            if preferred in self.features:
                return preferred
            candidates = sorted(key for key in self.features if predicate(key))
            if len(candidates) != 1:
                raise DatasetValidationError(
                    f"Could not uniquely discover {override_name}; candidates={candidates}. "
                    "Set dataset.feature_keys explicitly."
                )
            return candidates[0]

        self.action_key = choose_key(
            "action", "action", lambda key: key == "action" or key.startswith("action.")
        )
        self.timestamp_key = choose_key(
            "timestamp", "timestamp", lambda key: key.endswith("timestamp")
        )
        self.frame_key = choose_key(
            "frame_index", "frame_index", lambda key: key.endswith("frame_index")
        )
        self.episode_key = choose_key(
            "episode_index", "episode_index", lambda key: key.endswith("episode_index")
        )
        required = [self.action_key, self.timestamp_key, self.frame_key, self.episode_key]
        missing = [key for key in required if key not in self.features]
        if missing:
            raise DatasetValidationError(f"Missing required LeRobot features: {missing}")

        discovered_vectors = [
            key
            for key, feature in self.features.items()
            if key.startswith("observation.") and feature.get("dtype") != "video"
        ]
        configured_vectors = overrides.get("vector_observations", "auto")
        if configured_vectors == "auto":
            self._vector_keys = tuple(sorted(discovered_vectors))
        else:
            self._vector_keys = tuple(str(key) for key in configured_vectors)

        discovered_images = sorted(
            key
            for key, feature in self.features.items()
            if key.startswith("observation.") and feature.get("dtype") == "video"
        )
        use_images = bool(self.config.get("use_images", True))
        configured_images = overrides.get(
            "image_observations", "auto"
        )
        if not use_images:
            selected_images: list[str] = []
        elif configured_images == "all":
            selected_images = discovered_images
        elif configured_images == "auto":
            selected_images = discovered_images[:1]
        else:
            selected_images = [str(key) for key in configured_images]
        unknown = [
            key for key in (*self._vector_keys, *selected_images) if key not in self.features
        ]
        if unknown:
            raise DatasetValidationError(f"Configured observation features not found: {unknown}")
        self._image_keys = tuple(selected_images)
        if not self._vector_keys and not self._image_keys:
            raise DatasetValidationError("No observation features were selected")

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return self._vector_keys

    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return self._image_keys

    @property
    def discovered_image_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                key
                for key, value in self.features.items()
                if key.startswith("observation.") and value.get("dtype") == "video"
            )
        )

    def episodes(self) -> Sequence[EpisodeRecord]:
        return self._episodes

    def _worker_payload(
        self, episode: EpisodeRecord, *, load_images: bool = True
    ) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "episode_id": episode.episode_id,
            "length": episode.length,
            "chunks_size": int(self.info["chunks_size"]),
            "data_path": self.info["data_path"],
            "video_path": self.info.get("video_path", ""),
            "vector_keys": self._vector_keys,
            "image_keys": self._image_keys if load_images else (),
            "action_key": self.action_key,
            "timestamp_key": self.timestamp_key,
            "frame_key": self.frame_key,
            "episode_key": self.episode_key,
        }

    def _iter_episodes(
        self, *, num_workers: int, max_episodes: int | None, load_images: bool
    ) -> Iterator[EpisodeData]:
        episodes = self._episodes[:max_episodes] if max_episodes else self._episodes
        if num_workers <= 1:
            for episode in episodes:
                yield _load_lerobot_episode(
                    self._worker_payload(episode, load_images=load_images)
                )
            return
        # Spawn prevents a parent CUDA context from being inherited by workers.
        # Keep only 2x workers in flight so decoded videos cannot accumulate for
        # all 53k episodes behind a slower GPU consumer.
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as pool:
            episode_iterator = iter(episodes)
            pending: deque[Any] = deque()
            for _ in range(num_workers * 2):
                try:
                    episode = next(episode_iterator)
                except StopIteration:
                    break
                pending.append(
                    pool.submit(
                        _load_lerobot_episode,
                        self._worker_payload(episode, load_images=load_images),
                    )
                )
            while pending:
                yield pending.popleft().result()
                try:
                    episode = next(episode_iterator)
                except StopIteration:
                    continue
                pending.append(
                    pool.submit(
                        _load_lerobot_episode,
                        self._worker_payload(episode, load_images=load_images),
                    )
                )

    @staticmethod
    def _segment(episode: EpisodeData, start: int, end: int, kind: str) -> TrajectorySegment:
        index = slice(start, end + 1)
        if kind == "trajectory":
            sample_id = f"ep{episode.episode_id:06d}_trajectory"
        else:
            sample_id = f"ep{episode.episode_id:06d}_chunk_{start:06d}_{end:06d}"
        return TrajectorySegment(
            sample_id=sample_id,
            episode_id=episode.episode_id,
            start_step=int(episode.frame_indices[start]),
            end_step=int(episode.frame_indices[end]),
            timestamps=episode.timestamps[index],
            observations={key: value[index] for key, value in episode.observations.items()},
            actions=episode.actions[index],
            kind=kind,
        )

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
        normalized_modes = tuple(dict.fromkeys(str(mode).lower() for mode in modes))
        invalid = sorted(set(normalized_modes) - {"trajectory", "chunk"})
        if invalid:
            raise ValueError(f"Unsupported segmentation modes: {invalid}")
        for episode in self._iter_episodes(
            num_workers=num_workers,
            max_episodes=max_episodes,
            load_images=load_images,
        ):
            if "trajectory" in normalized_modes:
                yield self._segment(episode, 0, episode.length - 1, "trajectory")
            if "chunk" in normalized_modes:
                for start, end in aligned_chunk_windows(
                    episode.length, chunk_length, stride
                ):
                    yield self._segment(episode, start, end, "chunk")

    def fingerprint(self) -> str:
        sha = hashlib.sha256()
        for path in (
            self.root / "meta" / "info.json",
            self.root / "meta" / "episodes.jsonl",
        ):
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    sha.update(block)
        sha.update(json.dumps(self.config, sort_keys=True).encode("utf-8"))
        return sha.hexdigest()


register_dataset_adapter("lerobot", LeRobotDatasetAdapter)
