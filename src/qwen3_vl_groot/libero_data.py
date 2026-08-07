from __future__ import annotations

import hashlib
import random
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from libero_lerobot.metadata import ACTION_KEY, STATE_KEY, LeRobotV2Metadata, load_episode
from libero_lerobot.sampling import (
    FrameIndex,
    GloballyBalancedDistributedBatchSampler,
    normalized_sample_weights,
)
from libero_lerobot.selection import resolve_prior_selection
from libero_lerobot.targets import (
    TargetTaskSelection,
    resolve_target_task_selection,
    training_selection_sha256,
)

from .data import BridgeImageTransform, bridge_collate, make_action_window
from .normalization import QuantileStats


@dataclass(frozen=True)
class LiberoSources:
    target_selection: TargetTaskSelection
    prior_selection: Any | None
    sample_weights: tuple[float, ...]
    normalization_root: Path
    training_mode: str


def resolve_libero_sources(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> LiberoSources:
    target_selection = resolve_target_task_selection(config, paths)
    if bool(config["data"].get("target_only", False)):
        return LiberoSources(
            target_selection=target_selection,
            prior_selection=None,
            sample_weights=(1.0,),
            normalization_root=Path(paths["target_dataset"]).resolve(),
            training_mode="target_only",
        )
    prior_selection = resolve_prior_selection(config, paths)
    return LiberoSources(
        target_selection=target_selection,
        prior_selection=prior_selection,
        sample_weights=normalized_sample_weights(config["data"]["sample_weights"]),
        normalization_root=Path(paths["prior_dataset"]).resolve(),
        training_mode="mixed",
    )


def compute_libero_quantile_stats(
    dataset_root: str | Path,
    cache_path: str | Path,
    *,
    epsilon: float,
) -> QuantileStats:
    metadata = LeRobotV2Metadata(dataset_root)
    cache = Path(cache_path)
    fingerprint = metadata.metadata_sha256()
    if cache.is_file():
        import json

        with cache.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("metadata_sha256") == fingerprint:
            return QuantileStats.from_dict(cached)
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to compute LIBERO quantiles") from error

    total_frames = int(metadata.info["total_frames"])
    states = np.empty((total_frames, 8), dtype=np.float32)
    actions = np.empty((total_frames, 7), dtype=np.float32)
    offset = 0
    for episode in metadata.episodes:
        table = pq.read_table(
            metadata.episode_path(episode.episode_index),
            columns=[STATE_KEY, ACTION_KEY],
        )
        count = len(table)
        states[offset : offset + count] = np.asarray(
            table[STATE_KEY].to_pylist(), dtype=np.float32
        )
        actions[offset : offset + count] = np.asarray(
            table[ACTION_KEY].to_pylist(), dtype=np.float32
        )
        offset += count
    if offset != total_frames:
        raise RuntimeError(f"read {offset} LIBERO frames, expected {total_frames}")
    stats = QuantileStats(
        state_q01=np.quantile(states, 0.01, axis=0).astype(np.float32),
        state_q99=np.quantile(states, 0.99, axis=0).astype(np.float32),
        action_q01=np.quantile(actions, 0.01, axis=0).astype(np.float32),
        action_q99=np.quantile(actions, 0.99, axis=0).astype(np.float32),
        epsilon=float(epsilon),
    )
    stats.save(
        cache,
        extra={
            "dataset_path": str(metadata.root),
            "metadata_sha256": fingerprint,
            "training_frames": total_frames,
        },
    )
    return stats


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_libero_dataset_manifest(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    sources: LiberoSources,
    *,
    normalization_path: str | Path,
) -> dict[str, Any]:
    normalization = Path(normalization_path).resolve()
    normalization_sha256 = _sha256_file(normalization)
    target_weight = float(sources.sample_weights[0])
    prior_manifest = None
    if sources.training_mode == "mixed":
        if sources.prior_selection is None:
            prior_metadata = LeRobotV2Metadata(paths["prior_dataset"])
            selection_manifest = {
                "enabled": True,
                "mode": "full_dataset",
                "episodes": int(prior_metadata.info["total_episodes"]),
                "frames": int(prior_metadata.info["total_frames"]),
                "metadata_sha256": prior_metadata.metadata_sha256(),
            }
        else:
            selection_manifest = sources.prior_selection.as_manifest()
        prior_manifest = {
            "dataset_name": str(config["data"]["prior_dataset"]),
            "dataset_path": str(Path(paths["prior_dataset"]).resolve()),
            "sample_weight": float(sources.sample_weights[1]),
            "selection": selection_manifest,
        }
    return {
        "format": "qwen3-vl-groot-libero-selection-v1",
        "training_mode": sources.training_mode,
        "sample_weights": list(sources.sample_weights),
        "target": {
            "dataset_name": str(config["data"]["target_dataset"]),
            "dataset_path": str(Path(paths["target_dataset"]).resolve()),
            "sample_weight": target_weight,
            "selection": sources.target_selection.as_manifest(),
        },
        "prior": prior_manifest,
        "normalization": {
            "contract": "q01_q99_to_minus_one_plus_one",
            "dataset_path": str(sources.normalization_root),
            "stats_path": str(normalization),
            "sha256": normalization_sha256,
        },
        "learning_rates": {
            "lora": float(config["train"]["lora_learning_rate"]),
            "action_head": float(config["train"]["head_learning_rate"]),
        },
        "training_selection_sha256": training_selection_sha256(
            sources.target_selection,
            sources.prior_selection,
            sources.sample_weights,
            training_mode=sources.training_mode,
            normalization_sha256=normalization_sha256,
        ),
    }


class QwenLiberoFrameDataset:
    """Random-access Qwen samples over embedded-PNG LIBERO LeRobot v2 data."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_name: str,
        action_horizon: int,
        train: bool,
        seed: int,
        episode_cache_size: int,
        config: dict[str, Any] | None = None,
        frame_indices: Sequence[int] | None = None,
        transform: Callable[[np.ndarray, random.Random], Any] | None = None,
    ) -> None:
        if action_horizon <= 0 or episode_cache_size <= 0:
            raise ValueError("action_horizon and episode_cache_size must be positive")
        self.metadata = LeRobotV2Metadata(root)
        missing = self.metadata.missing_episode_files()
        if missing:
            raise RuntimeError(f"missing LIBERO episode file: {missing[0]}")
        self.dataset_name = str(dataset_name)
        self.action_horizon = int(action_horizon)
        self.train = bool(train)
        self.seed = int(seed)
        self.episode_cache_size = int(episode_cache_size)
        self.transform = (
            transform
            if transform is not None
            else BridgeImageTransform(config or {}, train=self.train)
        )
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._episode_ends: list[int] = []
        total = 0
        for episode in self.metadata.episodes:
            total += episode.length
            self._episode_ends.append(total)
        self._total_frames = total
        if frame_indices is None:
            self._frame_indices: tuple[int, ...] | None = None
            self._length = total
        else:
            selected = tuple(int(frame) for frame in frame_indices)
            if not selected or any(frame < 0 or frame >= total for frame in selected):
                raise ValueError("selected frame indices must be non-empty and in bounds")
            if any(left >= right for left, right in zip(selected, selected[1:])):
                raise ValueError("selected frame indices must be unique and sorted")
            self._frame_indices = selected
            self._length = len(selected)

    def __len__(self) -> int:
        return self._length

    def _global_frame(self, index: int) -> int:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        return index if self._frame_indices is None else self._frame_indices[index]

    def _episode(self, position: int) -> dict[str, Any]:
        record = self.metadata.episodes[position]
        if record.episode_index in self._episode_cache:
            value = self._episode_cache.pop(record.episode_index)
            self._episode_cache[record.episode_index] = value
            return value
        value = load_episode(self.metadata, record, include_wrist=False)
        self._episode_cache[record.episode_index] = value
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return value

    @staticmethod
    def _decode_primary(value: bytes) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Pillow is required to decode LIBERO PNG images") from error
        with Image.open(BytesIO(value)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    def __getitem__(self, index: int) -> dict[str, Any]:
        global_frame = self._global_frame(int(index))
        episode_position = bisect_right(self._episode_ends, global_frame)
        start = 0 if episode_position == 0 else self._episode_ends[episode_position - 1]
        frame_position = global_frame - start
        episode = self._episode(episode_position)
        actions, action_mask = make_action_window(
            episode["action"], frame_position, self.action_horizon
        )
        rng = random.Random(self.seed + global_frame * 1_009)
        return {
            "image": self.transform(
                self._decode_primary(episode["image_primary"][frame_position]),
                rng,
            ),
            "state": episode["state"][frame_position],
            "actions": actions,
            "action_mask": action_mask,
            "instruction": str(episode["language_instruction"][frame_position]),
            "dataset_name": self.dataset_name,
            "episode_index": int(episode["episode_index"][frame_position]),
            "frame_index": int(episode["frame_index"][frame_position]),
        }


class QwenLiberoCombinedDataset:
    """Route sampler indices to one of the independently selected data sources."""

    def __init__(self, sources: Sequence[QwenLiberoFrameDataset]) -> None:
        if not sources:
            raise ValueError("at least one LIBERO source is required")
        self.sources = tuple(sources)
        self.source_sizes = tuple(len(source) for source in self.sources)
        if any(size <= 0 for size in self.source_sizes):
            raise ValueError("LIBERO data sources must be non-empty")

    def __len__(self) -> int:
        return sum(self.source_sizes)

    def __getitem__(self, index: FrameIndex | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, FrameIndex):
            source, frame = index.source, index.frame
        else:
            source, frame = index
        return self.sources[int(source)][int(frame)]


@dataclass(frozen=True)
class QwenLiberoTrainingData:
    dataloader: Any
    dataset: QwenLiberoCombinedDataset
    batch_sampler: GloballyBalancedDistributedBatchSampler
    sources: LiberoSources


def make_libero_training_data(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    rank: int,
    world_size: int,
    transform: Callable[[np.ndarray, random.Random], Any] | None = None,
) -> QwenLiberoTrainingData:
    """Build the selected LIBERO datasets and an exact globally balanced loader."""
    from torch.utils.data import DataLoader

    data_config = config["data"]
    train_config = config["train"]
    sources = resolve_libero_sources(config, paths)
    common = {
        "action_horizon": int(data_config["action_horizon"]),
        "train": True,
        "seed": int(train_config["seed"]),
        "episode_cache_size": int(data_config["episode_cache_size"]),
        "config": dict(data_config),
        "transform": transform,
    }
    datasets = [
        QwenLiberoFrameDataset(
            paths["target_dataset"],
            dataset_name=str(data_config["target_dataset"]),
            frame_indices=sources.target_selection.frame_indices,
            **common,
        )
    ]
    if sources.training_mode == "mixed":
        datasets.append(
            QwenLiberoFrameDataset(
                paths["prior_dataset"],
                dataset_name=str(data_config["prior_dataset"]),
                frame_indices=(
                    None
                    if sources.prior_selection is None
                    else sources.prior_selection.frame_indices
                ),
                **common,
            )
        )
    dataset = QwenLiberoCombinedDataset(datasets)
    batch_sampler = GloballyBalancedDistributedBatchSampler(
        dataset.source_sizes,
        local_batch_size=int(train_config["micro_batch_size"]),
        sample_weights=sources.sample_weights,
        rank=int(rank),
        world_size=int(world_size),
        seed=int(train_config["seed"]),
        num_batches=int(train_config["max_steps"])
        * int(train_config["gradient_accumulation_steps"]),
    )
    workers = int(data_config["num_workers"])
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": batch_sampler,
        "collate_fn": bridge_collate,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(data_config["prefetch_factor"])
    dataloader = DataLoader(dataset, **loader_kwargs)
    return QwenLiberoTrainingData(
        dataloader=dataloader,
        dataset=dataset,
        batch_sampler=batch_sampler,
        sources=sources,
    )
