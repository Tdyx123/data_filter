from __future__ import annotations

import hashlib
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .errors import LiberoDataError


CODEBASE_VERSION = "v2.0"
CHUNKS_SIZE = 1000
DEFAULT_FPS = 10
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
PRIMARY_IMAGE_KEY = "observation.images.image"
WRIST_IMAGE_KEY = "observation.images.image2"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STANDARD_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
METADATA_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "stats.json")
METADATA_HASH_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl")


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    length: int
    tasks: tuple[str, ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise LiberoDataError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise LiberoDataError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise LiberoDataError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise LiberoDataError(f"Expected a JSON object in {path}")
    return value


class LeRobotV2Metadata:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        meta = self.root / "meta"
        missing = [str(meta / name) for name in METADATA_FILES if not (meta / name).is_file()]
        if missing:
            raise LiberoDataError(f"Missing LeRobot v2 metadata files: {missing}")

        self.info = _load_json(meta / "info.json")
        self.stats = _load_json(meta / "stats.json")
        episode_rows = read_jsonl(meta / "episodes.jsonl")
        task_rows = read_jsonl(meta / "tasks.jsonl")
        try:
            self.tasks = {int(row["task_index"]): str(row["task"]).strip() for row in task_rows}
            self.episodes = tuple(
                EpisodeRecord(
                    episode_index=int(row["episode_index"]),
                    length=int(row["length"]),
                    tasks=tuple(str(task).strip() for task in row.get("tasks", [])),
                )
                for row in episode_rows
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LiberoDataError(f"Invalid LeRobot v2 metadata under {meta}: {error}") from error
        if len(self.tasks) != len(task_rows):
            raise LiberoDataError(f"{meta}/tasks.jsonl contains duplicate task indices")
        try:
            self._validate()
        except (KeyError, TypeError, ValueError) as error:
            raise LiberoDataError(f"Invalid LeRobot v2 metadata under {meta}: {error}") from error
        offset = 0
        self.global_offsets: dict[int, int] = {}
        for episode in self.episodes:
            self.global_offsets[episode.episode_index] = offset
            offset += episode.length

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.info["features"]

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def chunks_size(self) -> int:
        return int(self.info["chunks_size"])

    def episode_path(self, episode_index: int) -> Path:
        relative = str(self.info["data_path"]).format(
            episode_chunk=episode_index // self.chunks_size,
            episode_index=episode_index,
        )
        return self.root / relative

    def _validate_feature(
        self,
        key: str,
        *,
        dtype: str,
        shape: Sequence[int],
    ) -> None:
        feature = self.features.get(key)
        if not isinstance(feature, dict):
            raise LiberoDataError(f"{self.root}: missing LeRobot feature {key!r}")
        actual_shape = list(feature.get("shape", []))
        if feature.get("dtype") != dtype or actual_shape != list(shape):
            raise LiberoDataError(
                f"{self.root}: feature {key!r} must be dtype={dtype}, "
                f"shape={list(shape)}; found {feature}"
            )

    def _validate(self) -> None:
        if self.info.get("codebase_version") != CODEBASE_VERSION:
            raise LiberoDataError(
                f"{self.root}: expected codebase_version={CODEBASE_VERSION}, "
                f"found {self.info.get('codebase_version')!r}"
            )
        if self.info.get("robot_type") != "libero":
            raise LiberoDataError(
                f"{self.root}: expected robot_type='libero', found {self.info.get('robot_type')!r}"
            )
        if self.info.get("video_path") is not None or int(self.info.get("total_videos", -1)) != 0:
            raise LiberoDataError(f"{self.root}: expected image-based LeRobot data without videos")
        if self.fps != DEFAULT_FPS:
            raise LiberoDataError(f"{self.root}: expected fps={DEFAULT_FPS}, found {self.fps}")
        if self.chunks_size != CHUNKS_SIZE:
            raise LiberoDataError(
                f"{self.root}: expected chunks_size={CHUNKS_SIZE}, found {self.chunks_size}"
            )
        if self.info.get("data_path") != DATA_PATH:
            raise LiberoDataError(
                f"{self.root}: unsupported data_path {self.info.get('data_path')!r}"
            )

        self._validate_feature(PRIMARY_IMAGE_KEY, dtype="image", shape=(128, 128, 3))
        self._validate_feature(WRIST_IMAGE_KEY, dtype="image", shape=(128, 128, 3))
        self._validate_feature(STATE_KEY, dtype="float32", shape=(8,))
        self._validate_feature(ACTION_KEY, dtype="float32", shape=(7,))
        for key, feature in STANDARD_FEATURES.items():
            self._validate_feature(
                key,
                dtype=str(feature["dtype"]),
                shape=tuple(feature["shape"]),
            )

        total_episodes = int(self.info.get("total_episodes", -1))
        total_frames = int(self.info.get("total_frames", -1))
        total_tasks = int(self.info.get("total_tasks", -1))
        if total_episodes != len(self.episodes):
            raise LiberoDataError(
                f"{self.root}: episodes.jsonl has {len(self.episodes)} rows, "
                f"info.json reports {total_episodes}"
            )
        if total_frames != sum(episode.length for episode in self.episodes):
            raise LiberoDataError(f"{self.root}: total_frames does not match episode lengths")
        if total_tasks != len(self.tasks):
            raise LiberoDataError(f"{self.root}: total_tasks does not match tasks.jsonl")
        if [episode.episode_index for episode in self.episodes] != list(range(total_episodes)):
            raise LiberoDataError(f"{self.root}: episode indices must be contiguous from zero")
        if any(episode.length <= 0 for episode in self.episodes):
            raise LiberoDataError(f"{self.root}: every episode must contain at least one frame")
        if sorted(self.tasks) != list(range(total_tasks)):
            raise LiberoDataError(f"{self.root}: task indices must be contiguous from zero")
        if any(not task for task in self.tasks.values()):
            raise LiberoDataError(f"{self.root}: tasks.jsonl contains an empty task")
        known_tasks = set(self.tasks.values())
        if any(
            not episode.tasks or any(task not in known_tasks for task in episode.tasks)
            for episode in self.episodes
        ):
            raise LiberoDataError(f"{self.root}: episodes.jsonl contains an invalid task mapping")
        expected_chunks = (total_episodes + self.chunks_size - 1) // self.chunks_size
        if int(self.info.get("total_chunks", -1)) != expected_chunks:
            raise LiberoDataError(f"{self.root}: total_chunks is inconsistent")
        if self.info.get("splits") != {"train": f"0:{total_episodes}"}:
            raise LiberoDataError(f"{self.root}: expected a single full train split")

        statistic_shapes = {
            PRIMARY_IMAGE_KEY: (3, 1, 1),
            WRIST_IMAGE_KEY: (3, 1, 1),
            STATE_KEY: (8,),
            ACTION_KEY: (7,),
            **{key: (1,) for key in STANDARD_FEATURES},
        }
        for key, shape in statistic_shapes.items():
            entry = self.stats.get(key)
            if not isinstance(entry, dict):
                raise LiberoDataError(f"{self.root}: missing stats for {key}")
            for statistic in ("mean", "std", "min", "max"):
                values = entry.get(statistic)
                array = np.asarray(values)
                if not isinstance(values, list) or array.shape != shape:
                    raise LiberoDataError(f"{self.root}: {key}.{statistic} must have shape {shape}")
                if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
                    raise LiberoDataError(
                        f"{self.root}: {key}.{statistic} must contain finite numbers"
                    )
            if np.any(np.asarray(entry["std"]) < 0):
                raise LiberoDataError(f"{self.root}: {key}.std must be non-negative")
            if np.any(np.asarray(entry["min"]) > np.asarray(entry["max"])):
                raise LiberoDataError(f"{self.root}: {key}.min must not exceed max")

    def missing_episode_files(self) -> list[Path]:
        return [
            self.episode_path(episode.episode_index)
            for episode in self.episodes
            if not self.episode_path(episode.episode_index).is_file()
        ]

    def metadata_sha256(self) -> str:
        digest = hashlib.sha256()
        for name in METADATA_HASH_FILES:
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            with (self.root / "meta" / name).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        data = value.get("bytes")
        path = value.get("path")
    else:
        try:
            data = value["bytes"].as_py()
            path = value["path"].as_py()
        except (KeyError, TypeError, AttributeError) as error:
            raise LiberoDataError(f"Invalid embedded LeRobot image value: {value!r}") from error
    if data is not None:
        if path is not None:
            raise LiberoDataError("Embedded LeRobot image must have path=null")
        return bytes(data)
    raise LiberoDataError(
        f"LeRobot image must contain embedded PNG bytes with path=null, found path={path!r}"
    )


def load_episode(
    metadata: LeRobotV2Metadata,
    episode: EpisodeRecord,
    *,
    include_wrist: bool = True,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to read LeRobotDataset v2") from error

    path = metadata.episode_path(episode.episode_index)
    if not path.is_file():
        raise LiberoDataError(f"Missing LeRobot episode Parquet: {path}")
    columns = [
        PRIMARY_IMAGE_KEY,
        STATE_KEY,
        ACTION_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    if include_wrist:
        columns.insert(1, WRIST_IMAGE_KEY)
    try:
        table = pq.read_table(path, columns=columns)
    except Exception as error:
        raise LiberoDataError(f"Could not read LeRobot episode {path}: {error}") from error
    if len(table) != episode.length:
        raise LiberoDataError(f"{path}: found {len(table)} rows, expected {episode.length}")

    def array(key: str, dtype: Any) -> np.ndarray:
        return np.asarray(table[key].to_pylist(), dtype=dtype)

    state = array(STATE_KEY, np.float32)
    action = array(ACTION_KEY, np.float32)
    timestamp = array("timestamp", np.float32).reshape(-1)
    frame_index = array("frame_index", np.int64).reshape(-1)
    episode_index = array("episode_index", np.int64).reshape(-1)
    global_index = array("index", np.int64).reshape(-1)
    task_index = array("task_index", np.int64).reshape(-1)
    if state.shape != (episode.length, 8) or action.shape != (episode.length, 7):
        raise LiberoDataError(f"{path}: invalid state/action shapes {state.shape}/{action.shape}")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise LiberoDataError(f"{path}: state/action contains NaN or infinity")
    if np.any(episode_index != episode.episode_index):
        raise LiberoDataError(f"{path}: inconsistent episode_index column")
    if not np.array_equal(frame_index, np.arange(episode.length, dtype=np.int64)):
        raise LiberoDataError(f"{path}: frame_index must be contiguous from zero")
    expected_timestamp = frame_index.astype(np.float32) / np.float32(metadata.fps)
    if not np.allclose(timestamp, expected_timestamp, atol=1.0e-5, rtol=0.0):
        raise LiberoDataError(f"{path}: timestamp is inconsistent with fps={metadata.fps}")
    if len(global_index) > 1 and np.any(np.diff(global_index) != 1):
        raise LiberoDataError(f"{path}: index must increase by one")
    expected_global_index = np.arange(
        metadata.global_offsets[episode.episode_index],
        metadata.global_offsets[episode.episode_index] + episode.length,
        dtype=np.int64,
    )
    if not np.array_equal(global_index, expected_global_index):
        raise LiberoDataError(f"{path}: index is inconsistent with prior episode lengths")
    if any(int(value) not in metadata.tasks for value in task_index):
        raise LiberoDataError(f"{path}: unknown task_index value")

    primary = [_image_bytes(value) for value in table[PRIMARY_IMAGE_KEY].to_pylist()]
    language = [metadata.tasks[int(value)] for value in task_index]
    if not set(language).issubset(episode.tasks):
        raise LiberoDataError(f"{path}: task_index values do not match episodes.jsonl")
    result = {
        "image_primary": np.asarray(primary, dtype=object),
        "state": state,
        "action": action,
        "language_instruction": np.asarray(language, dtype=object),
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_index,
        "index": global_index,
        "task_index": task_index,
    }
    if include_wrist:
        wrist = [_image_bytes(value) for value in table[WRIST_IMAGE_KEY].to_pylist()]
        result["image_wrist"] = np.asarray(wrist, dtype=object)
    return result


def iter_episodes(
    metadata: LeRobotV2Metadata,
    *,
    num_workers: int,
) -> Iterator[dict[str, Any]]:
    episodes = iter(metadata.episodes)
    if num_workers <= 1:
        for episode in episodes:
            yield load_episode(metadata, episode)
        return

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending: deque[Any] = deque()
        for _ in range(num_workers * 2):
            try:
                episode = next(episodes)
            except StopIteration:
                break
            pending.append(pool.submit(load_episode, metadata, episode))
        while pending:
            yield pending.popleft().result()
            try:
                episode = next(episodes)
            except StopIteration:
                continue
            pending.append(pool.submit(load_episode, metadata, episode))
