from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .normalization import QuantileStats

try:
    import torch as _torch

    _IterableDatasetBase = _torch.utils.data.IterableDataset
except ImportError:
    class _IterableDatasetBase:  # type: ignore[no-redef]
        pass


class DatasetValidationError(RuntimeError):
    """Raised when a Bridge dataset does not match the expected contract."""


@dataclass(frozen=True)
class Episode:
    index: int
    length: int
    tasks: tuple[str, ...]


def stable_validation_episode(
    episode_index: int,
    seed: int = 42,
    validation_percent: int = 1,
) -> bool:
    if not 0 <= validation_percent <= 100:
        raise ValueError("validation_percent must be between 0 and 100")
    digest = hashlib.blake2b(
        f"{seed}:{episode_index}".encode("utf-8"), digest_size=8
    ).digest()
    bucket = int.from_bytes(digest, byteorder="big") % 100
    return bucket < validation_percent


def split_episodes(
    episodes: Sequence[Episode],
    seed: int = 42,
    validation_percent: int = 1,
) -> tuple[list[Episode], list[Episode]]:
    train: list[Episode] = []
    validation: list[Episode] = []
    for episode in episodes:
        target = (
            validation
            if stable_validation_episode(episode.index, seed, validation_percent)
            else train
        )
        target.append(episode)
    if episodes and (not train or not validation):
        raise DatasetValidationError(
            "Hash split produced an empty partition; increase dataset size or validation percent"
        )
    return train, validation


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


class BridgeMetadata:
    def __init__(self, root: str | Path, config: dict[str, Any]):
        self.root = Path(root).expanduser().resolve()
        self.config = config
        meta = self.root / "meta"
        required = [meta / "info.json", meta / "episodes.jsonl", meta / "tasks.jsonl"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise DatasetValidationError(f"Missing Bridge metadata files: {missing}")

        with (meta / "info.json").open("r", encoding="utf-8") as handle:
            self.info = json.load(handle)
        self._validate_info()

        task_rows = _read_jsonl(meta / "tasks.jsonl")
        self.tasks = {
            int(row["task_index"]): str(row.get("task", "")).strip() for row in task_rows
        }
        episode_rows = _read_jsonl(meta / "episodes.jsonl")
        self.episodes = [
            Episode(
                index=int(row["episode_index"]),
                length=int(row["length"]),
                tasks=tuple(str(task).strip() for task in row.get("tasks", [])),
            )
            for row in episode_rows
        ]
        if len(self.episodes) != int(self.info["total_episodes"]):
            raise DatasetValidationError(
                f"episodes.jsonl contains {len(self.episodes)} rows, "
                f"expected {self.info['total_episodes']}"
            )

    def _validate_info(self) -> None:
        info = self.info
        data = self.config
        if info.get("codebase_version") != "v2.0":
            raise DatasetValidationError("Only LeRobot codebase_version=v2.0 is supported")
        if info.get("robot_type") != "widowx":
            raise DatasetValidationError(f"Expected widowx, found {info.get('robot_type')}")
        features = info.get("features", {})
        checks = {
            data["image_key"]: ("video", [256, 256, 3]),
            data["state_key"]: ("float32", [data["state_dim"]]),
            data["action_key"]: ("float32", [data["action_dim"]]),
        }
        for key, (dtype, shape) in checks.items():
            feature = features.get(key)
            if feature is None:
                raise DatasetValidationError(f"Missing feature: {key}")
            if feature.get("dtype") != dtype or feature.get("shape") != shape:
                raise DatasetValidationError(
                    f"Feature {key} must be dtype={dtype}, shape={shape}; found {feature}"
                )
        video_info = features[data["image_key"]].get("info", {})
        if video_info.get("video.codec") != "av1":
            raise DatasetValidationError(
                f"Expected AV1 videos, found {video_info.get('video.codec')}"
            )

    def parquet_path(self, episode_index: int) -> Path:
        chunk = episode_index // int(self.info["chunks_size"])
        relative = self.info["data_path"].format(
            episode_chunk=chunk, episode_index=episode_index
        )
        return self.root / relative

    def video_path(self, episode_index: int) -> Path:
        chunk = episode_index // int(self.info["chunks_size"])
        relative = self.info["video_path"].format(
            episode_chunk=chunk,
            video_key=self.config["image_key"],
            episode_index=episode_index,
        )
        return self.root / relative

    def instruction(self, task_index: int, episode: Episode) -> str:
        instruction = self.tasks.get(task_index, "").strip()
        if not instruction:
            instruction = next((task for task in episode.tasks if task), "")
        return instruction or self.config["fallback_instruction"]

    def split(self) -> tuple[list[Episode], list[Episode]]:
        return split_episodes(
            self.episodes,
            seed=int(self.config["split_seed"]),
            validation_percent=int(self.config["validation_percent"]),
        )

    def fingerprint(self) -> dict[str, Any]:
        sha = hashlib.sha256()
        paths = [
            self.root / "meta" / "info.json",
            self.root / "meta" / "episodes.jsonl",
            self.root / "meta" / "tasks.jsonl",
        ]
        for path in paths:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    sha.update(chunk)
        return {
            "algorithm": "sha256",
            "metadata_sha256": sha.hexdigest(),
            "root": str(self.root),
            "total_episodes": len(self.episodes),
            "total_frames": int(self.info["total_frames"]),
            "image_key": self.config["image_key"],
            "split_seed": int(self.config["split_seed"]),
            "validation_percent": int(self.config["validation_percent"]),
        }


def make_action_window(
    actions: np.ndarray,
    frame_index: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError("actions must have shape [time, action_dim]")
    if not 0 <= frame_index < len(actions):
        raise IndexError(frame_index)
    result = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
    mask = np.zeros((horizon,), dtype=np.float32)
    count = min(horizon, len(actions) - frame_index)
    result[:count] = actions[frame_index : frame_index + count]
    mask[:count] = 1.0
    return result, mask


def decode_video(path: str | Path) -> list[np.ndarray]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required to decode Bridge AV1 videos") from error

    frames: list[np.ndarray] = []
    try:
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception as error:
        raise DatasetValidationError(
            f"Could not decode AV1 video {path}. Install PyAV with libdav1d support, "
            "or transcode to a separate dataset copy. The source dataset was not changed. "
            f"Original error: {error}"
        ) from error
    return frames


class BridgeImageTransform:
    def __init__(self, config: dict[str, Any], train: bool):
        self.train = train
        self.crop_size = int(config["train_crop_size"])
        self.output_size = int(config["output_image_size"])
        jitter = config["color_jitter"]
        try:
            from torchvision.transforms import ColorJitter, InterpolationMode
            from torchvision.transforms import functional as functional
        except ImportError as error:
            raise RuntimeError("torchvision is required for Bridge image transforms") from error
        self.functional = functional
        self.interpolation = InterpolationMode.BICUBIC
        self.jitter = ColorJitter(
            brightness=float(jitter["brightness"]),
            contrast=float(jitter["contrast"]),
            saturation=float(jitter["saturation"]),
            hue=float(jitter["hue"]),
        )

    def __call__(self, image: np.ndarray, rng: random.Random):
        from PIL import Image

        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        width, height = pil.size
        if min(width, height) < self.crop_size:
            raise DatasetValidationError(
                f"Image {width}x{height} is smaller than crop {self.crop_size}"
            )
        if self.train:
            left = rng.randint(0, width - self.crop_size)
            top = rng.randint(0, height - self.crop_size)
        else:
            left = (width - self.crop_size) // 2
            top = (height - self.crop_size) // 2
        pil = self.functional.crop(pil, top, left, self.crop_size, self.crop_size)
        pil = self.functional.resize(
            pil,
            [self.output_size, self.output_size],
            interpolation=self.interpolation,
            antialias=True,
        )
        return self.jitter(pil) if self.train else pil


class BridgeEpisodeDataset(_IterableDatasetBase):
    """Iterable episode-sharded dataset.

    Each rank/worker receives disjoint episodes. A worker decodes an entire episode
    video once and retains a small LRU cache. Training repeats indefinitely; validation
    performs one deterministic pass.
    """

    def __init__(
        self,
        metadata: BridgeMetadata,
        episodes: Sequence[Episode],
        *,
        train: bool,
        rank: int,
        world_size: int,
        seed: int,
        action_horizon: int,
        video_cache_size: int = 2,
    ):
        super().__init__()
        self.metadata = metadata
        self.episodes = list(episodes)
        self.train = train
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.action_horizon = action_horizon
        self.video_cache_size = video_cache_size
        self.transform = BridgeImageTransform(metadata.config, train=train)
        self._video_cache: OrderedDict[int, list[np.ndarray]] = OrderedDict()

    def _load_episode(self, episode: Episode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("pyarrow is required to read Bridge Parquet files") from error
        path = self.metadata.parquet_path(episode.index)
        if not path.is_file():
            raise DatasetValidationError(f"Missing Parquet file: {path}")
        table = pq.read_table(
            path,
            columns=[
                self.metadata.config["state_key"],
                self.metadata.config["action_key"],
                "task_index",
            ],
        )
        state = np.asarray(
            table[self.metadata.config["state_key"]].to_pylist(), dtype=np.float32
        )
        action = np.asarray(
            table[self.metadata.config["action_key"]].to_pylist(), dtype=np.float32
        )
        task_index = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        if len(table) != episode.length:
            raise DatasetValidationError(
                f"Episode {episode.index}: Parquet has {len(table)} rows, "
                f"metadata says {episode.length}"
            )
        if state.shape != (episode.length, self.metadata.config["state_dim"]):
            raise DatasetValidationError(
                f"Episode {episode.index}: invalid state shape {state.shape}"
            )
        if action.shape != (episode.length, self.metadata.config["action_dim"]):
            raise DatasetValidationError(
                f"Episode {episode.index}: invalid action shape {action.shape}"
            )
        return state, action, task_index

    def _video(self, episode: Episode) -> list[np.ndarray]:
        if episode.index in self._video_cache:
            frames = self._video_cache.pop(episode.index)
            self._video_cache[episode.index] = frames
            return frames
        path = self.metadata.video_path(episode.index)
        if not path.is_file():
            raise DatasetValidationError(f"Missing video: {path}")
        frames = decode_video(path)
        if len(frames) != episode.length:
            raise DatasetValidationError(
                f"Episode {episode.index}: video has {len(frames)} frames, "
                f"Parquet/metadata has {episode.length}"
            )
        self._video_cache[episode.index] = frames
        while len(self._video_cache) > self.video_cache_size:
            self._video_cache.popitem(last=False)
        return frames

    def __iter__(self) -> Iterator[dict[str, Any]]:
        import torch

        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        workers_per_rank = worker.num_workers if worker else 1
        global_worker = self.rank * workers_per_rank + worker_id
        total_workers = self.world_size * workers_per_rank
        epoch = 0

        while True:
            episode_order = list(self.episodes)
            epoch_rng = random.Random(self.seed + epoch)
            if self.train:
                epoch_rng.shuffle(episode_order)
            local_episodes = episode_order[global_worker::total_workers]

            for episode in local_episodes:
                state, actions, task_indices = self._load_episode(episode)
                frames = self._video(episode)
                frame_order = list(range(episode.length))
                if self.train:
                    epoch_rng.shuffle(frame_order)
                for frame_index in frame_order:
                    action_window, action_mask = make_action_window(
                        actions, frame_index, self.action_horizon
                    )
                    sample_rng = random.Random(
                        self.seed
                        + epoch * 10_000_019
                        + episode.index * 1_009
                        + frame_index
                    )
                    yield {
                        "image": self.transform(frames[frame_index], sample_rng),
                        "state": state[frame_index],
                        "actions": action_window,
                        "action_mask": action_mask,
                        "instruction": self.metadata.instruction(
                            int(task_indices[frame_index]), episode
                        ),
                        "episode_index": episode.index,
                        "frame_index": frame_index,
                    }
            if not self.train:
                return
            epoch += 1


def as_torch_iterable(dataset: BridgeEpisodeDataset):
    """Return an IterableDataset while keeping torch optional for metadata-only tools."""
    if "_torch" not in globals():
        raise RuntimeError("PyTorch is required for Bridge data loading")
    return dataset


def bridge_collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "images": [sample["image"] for sample in samples],
        "state": torch.as_tensor(
            np.stack([sample["state"] for sample in samples]), dtype=torch.float32
        ),
        "actions": torch.as_tensor(
            np.stack([sample["actions"] for sample in samples]), dtype=torch.float32
        ),
        "action_mask": torch.as_tensor(
            np.stack([sample["action_mask"] for sample in samples]), dtype=torch.float32
        ),
        "instructions": [sample["instruction"] for sample in samples],
        "episode_index": torch.as_tensor(
            [sample["episode_index"] for sample in samples], dtype=torch.int64
        ),
        "frame_index": torch.as_tensor(
            [sample["frame_index"] for sample in samples], dtype=torch.int64
        ),
    }


def compute_quantile_stats(
    metadata: BridgeMetadata,
    train_episodes: Sequence[Episode],
    cache_path: str | Path,
    *,
    epsilon: float = 1.0e-6,
) -> QuantileStats:
    cache_path = Path(cache_path)
    fingerprint = metadata.fingerprint()["metadata_sha256"]
    if cache_path.is_file():
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("metadata_sha256") == fingerprint:
            return QuantileStats.from_dict(cached)

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to compute quantile statistics") from error

    total_frames = sum(episode.length for episode in train_episodes)
    state_dim = int(metadata.config["state_dim"])
    action_dim = int(metadata.config["action_dim"])
    states = np.empty((total_frames, state_dim), dtype=np.float32)
    actions = np.empty((total_frames, action_dim), dtype=np.float32)
    offset = 0
    for episode in train_episodes:
        table = pq.read_table(
            metadata.parquet_path(episode.index),
            columns=[metadata.config["state_key"], metadata.config["action_key"]],
        )
        count = len(table)
        states[offset : offset + count] = np.asarray(
            table[metadata.config["state_key"]].to_pylist(), dtype=np.float32
        )
        actions[offset : offset + count] = np.asarray(
            table[metadata.config["action_key"]].to_pylist(), dtype=np.float32
        )
        offset += count
    if offset != total_frames:
        raise DatasetValidationError(
            f"Read {offset} training frames, expected {total_frames}"
        )
    stats = QuantileStats(
        state_q01=np.quantile(states, 0.01, axis=0).astype(np.float32),
        state_q99=np.quantile(states, 0.99, axis=0).astype(np.float32),
        action_q01=np.quantile(actions, 0.01, axis=0).astype(np.float32),
        action_q99=np.quantile(actions, 0.99, axis=0).astype(np.float32),
        epsilon=epsilon,
    )
    stats.save(
        cache_path,
        extra={
            "metadata_sha256": fingerprint,
            "training_episodes": len(train_episodes),
            "training_frames": total_frames,
        },
    )
    return stats


def validate_episode(
    metadata: BridgeMetadata,
    episode: Episode,
    *,
    decode: bool = True,
) -> dict[str, Any]:
    dataset = BridgeEpisodeDataset(
        metadata,
        [episode],
        train=False,
        rank=0,
        world_size=1,
        seed=int(metadata.config["split_seed"]),
        action_horizon=int(metadata.config["action_horizon"]),
        video_cache_size=1,
    )
    state, action, task_index = dataset._load_episode(episode)
    result = {
        "episode_index": episode.index,
        "frames": episode.length,
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "task_index_min": int(task_index.min()),
        "task_index_max": int(task_index.max()),
        "video": str(metadata.video_path(episode.index)),
    }
    if decode:
        frames = dataset._video(episode)
        result["decoded_frames"] = len(frames)
        result["frame_shape"] = list(frames[0].shape)
    return result
