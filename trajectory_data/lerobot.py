"""LeRobot v2 dataset adapter."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .core import DatasetAdapter, DatasetValidationError, EpisodeData, EpisodeRecord


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


def _decode_embedded_images(
    values: Sequence[Any],
    *,
    episode_id: int,
    image_key: str,
    expected_shape: Sequence[int],
) -> np.ndarray:
    """Decode LeRobot ``image`` structs containing embedded image bytes."""

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required to decode LeRobot image observations") from error

    try:
        shape = tuple(int(dimension) for dimension in expected_shape)
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(
            f"Episode {episode_id}, {image_key}: invalid declared image shape {expected_shape!r}"
        ) from error
    if len(shape) != 3 or shape[-1] != 3:
        raise DatasetValidationError(
            f"Episode {episode_id}, {image_key}: expected an RGB image shape, got {shape}"
        )

    frames: list[np.ndarray] = []
    for frame_index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}, frame {frame_index}: "
                "embedded image must be a {bytes, path} mapping"
            )
        data = value.get("bytes")
        path = value.get("path")
        if path not in {None, ""}:
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}, frame {frame_index}: "
                f"embedded image path must be empty, got {path!r}"
            )
        if data is None:
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}, frame {frame_index}: "
                "embedded image bytes are missing"
            )
        try:
            encoded = bytes(data)
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}, frame {frame_index}: "
                "embedded image bytes are invalid"
            ) from error
        if not encoded:
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}, frame {frame_index}: "
                "embedded image bytes are empty"
            )
        try:
            with Image.open(BytesIO(encoded)) as image:
                frame = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        except Exception as error:
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}, frame {frame_index}: "
                f"could not decode embedded image: {error}"
            ) from error
        if frame.shape != shape:
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}, frame {frame_index}: "
                f"decoded shape={frame.shape}, expected={shape}"
            )
        frames.append(frame)

    if not frames:
        raise DatasetValidationError(
            f"Episode {episode_id}, {image_key}: embedded image sequence is empty"
        )
    return np.stack(frames)


def _load_lerobot_episode(payload: Mapping[str, Any]) -> EpisodeData:
    """Process-safe LeRobot episode loader used by multiprocessing."""

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
    image_features = {
        str(key): dict(value) for key, value in dict(payload.get("image_features", {})).items()
    }
    embedded_image_keys = tuple(
        key for key, feature in image_features.items() if feature.get("dtype") == "image"
    )
    columns = list(
        dict.fromkeys(
            [
                *vector_keys,
                *embedded_image_keys,
                action_key,
                timestamp_key,
                frame_key,
                episode_key,
            ]
        )
    )
    expected_task_index = payload.get("task_index")
    if expected_task_index is not None:
        columns.append("task_index")
    try:
        table = pq.read_table(parquet_path, columns=columns)
    except Exception as error:
        raise DatasetValidationError(
            f"Episode {episode_id}: could not read required columns {columns} "
            f"from {parquet_path}: {error}"
        ) from error
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
    if expected_task_index is not None:
        task_indices = column_array("task_index", np.int64).reshape(-1)
        if np.any(task_indices != int(expected_task_index)):
            raise DatasetValidationError(
                f"Episode {episode_id}: inconsistent task index column; "
                f"expected {int(expected_task_index)}"
            )

    observations: dict[str, np.ndarray] = {}
    for key in vector_keys:
        values = column_array(key, np.float32)
        if len(values) != expected_length or not np.all(np.isfinite(values)):
            raise DatasetValidationError(
                f"Episode {episode_id}, {key}: invalid length or non-finite values"
            )
        observations[key] = values

    video_path_pattern = payload.get("video_path")
    for image_key, feature in image_features.items():
        storage = str(feature.get("dtype", ""))
        if storage == "image":
            frames = _decode_embedded_images(
                table[image_key].to_pylist(),
                episode_id=episode_id,
                image_key=image_key,
                expected_shape=feature.get("shape", ()),
            )
        elif storage == "video":
            if not video_path_pattern:
                raise DatasetValidationError(
                    f"Episode {episode_id}, {image_key}: video_path is not configured"
                )
            video_values = {**format_values, "video_key": image_key}
            video_path = root / str(video_path_pattern).format(**video_values)
            frames = _decode_video(video_path)
        else:
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}: unsupported image dtype {storage!r}"
            )
        if len(frames) != expected_length:
            raise DatasetValidationError(
                f"Episode {episode_id}, {image_key}: image frames={len(frames)}, "
                f"expected={expected_length}"
            )
        observations[image_key] = frames

    return EpisodeData(
        episode_id=episode_id,
        timestamps=timestamps,
        frame_indices=frame_indices,
        observations=observations,
        actions=actions,
        task_index=(int(expected_task_index) if expected_task_index is not None else None),
        task_name=(str(payload["task_name"]) if payload.get("task_name") is not None else None),
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
                "LeRobot adapter currently supports v2 datasets, found "
                f"{self.info.get('codebase_version')!r}"
            )
        self.features: dict[str, dict[str, Any]] = dict(self.info.get("features", {}))
        self._discover_keys()
        rows = _read_jsonl(episodes_path)
        task_by_name = self._load_task_index()
        records: list[EpisodeRecord] = []
        for row in rows:
            task_names = row.get("tasks")
            task_name: str | None = None
            task_index: int | None = None
            if task_names is not None:
                if not isinstance(task_names, list) or len(task_names) != 1:
                    raise DatasetValidationError(
                        "LeRobot episodes must reference exactly one task when task "
                        f"metadata is present: episode {row.get('episode_index')}"
                    )
                task_name = str(task_names[0])
                if not task_name:
                    raise DatasetValidationError(
                        f"Empty LeRobot task name for episode {row.get('episode_index')}"
                    )
                if task_name in task_by_name:
                    task_index = task_by_name[task_name]
                elif task_by_name:
                    raise DatasetValidationError(
                        f"Unknown LeRobot task {task_name!r} for episode {row.get('episode_index')}"
                    )
            records.append(
                EpisodeRecord(
                    int(row["episode_index"]),
                    int(row["length"]),
                    task_index,
                    task_name,
                )
            )
        self._episodes = tuple(records)
        if any(episode.length <= 0 for episode in self._episodes):
            raise DatasetValidationError("LeRobot episodes must contain at least one frame")
        expected = int(self.info.get("total_episodes", len(self._episodes)))
        if len(self._episodes) != expected:
            raise DatasetValidationError(
                f"episodes.jsonl has {len(self._episodes)} entries, expected {expected}"
            )

    def _load_task_index(self) -> dict[str, int]:
        path = self.root / "meta" / "tasks.jsonl"
        if not path.is_file():
            return {}
        by_name: dict[str, int] = {}
        by_index: dict[int, str] = {}
        for row in _read_jsonl(path):
            task_name = str(row.get("task", ""))
            try:
                task_index = int(row["task_index"])
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetValidationError(f"Invalid task_index in {path}: {row!r}") from error
            if not task_name:
                raise DatasetValidationError(f"Empty task name in {path}")
            if task_name in by_name or task_index in by_index:
                raise DatasetValidationError(
                    f"Duplicate task name or index in {path}: {task_name!r}, {task_index}"
                )
            by_name[task_name] = task_index
            by_index[task_index] = task_name
        return by_name

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
            "action",
            "action",
            lambda key: key == "action" or key.startswith("action."),
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

        image_dtypes = {"image", "video"}
        discovered_vectors = [
            key
            for key, feature in self.features.items()
            if key.startswith("observation.") and feature.get("dtype") not in image_dtypes
        ]
        configured_vectors = overrides.get("vector_observations", "auto")
        if configured_vectors == "auto":
            self._vector_keys = tuple(sorted(discovered_vectors))
        else:
            self._vector_keys = tuple(str(key) for key in configured_vectors)

        discovered_images = sorted(
            key
            for key, feature in self.features.items()
            if key.startswith("observation.") and feature.get("dtype") in image_dtypes
        )
        use_images = bool(self.config.get("use_images", True))
        configured_images = overrides.get("image_observations", "auto")
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
        invalid_vectors = [
            key for key in self._vector_keys if self.features[key].get("dtype") in image_dtypes
        ]
        if invalid_vectors:
            raise DatasetValidationError(
                f"Configured vector observations are image features: {invalid_vectors}"
            )
        invalid_images = [
            key for key in selected_images if self.features[key].get("dtype") not in image_dtypes
        ]
        if invalid_images:
            raise DatasetValidationError(
                f"Configured image observations are not image/video features: {invalid_images}"
            )
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
                if key.startswith("observation.") and value.get("dtype") in {"image", "video"}
            )
        )

    def episodes(self) -> Sequence[EpisodeRecord]:
        return self._episodes

    def _worker_payload(
        self,
        episode: EpisodeRecord,
        *,
        load_images: bool = True,
    ) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "episode_id": episode.episode_id,
            "length": episode.length,
            "task_index": episode.task_index,
            "task_name": episode.task_name,
            "chunks_size": int(self.info["chunks_size"]),
            "data_path": self.info["data_path"],
            "video_path": self.info.get("video_path", ""),
            "vector_keys": self._vector_keys,
            "image_features": (
                {key: self.features[key] for key in self._image_keys} if load_images else {}
            ),
            "action_key": self.action_key,
            "timestamp_key": self.timestamp_key,
            "frame_key": self.frame_key,
            "episode_key": self.episode_key,
        }

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        episodes = self._episodes[:max_episodes] if max_episodes else self._episodes
        if num_workers <= 1:
            for episode in episodes:
                yield _load_lerobot_episode(self._worker_payload(episode, load_images=load_images))
            return

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
