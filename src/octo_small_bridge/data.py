"""Frame-level BridgeData V2 input pipeline for Octo-small."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image, ImageEnhance

from trajectory_data import DatasetValidationError, EpisodeData, EpisodeRecord
from trajectory_data.lerobot import LeRobotDatasetAdapter

from .normalization import BridgeV2NormalizationStatistics


@dataclass(frozen=True)
class BridgeFrameRef:
    """A deterministic reference to one training frame."""

    epoch: int
    episode_id: int
    frame_position: int


def _normalize_gripper_actions(
    gripper: np.ndarray,
    *,
    episode_id: int,
) -> np.ndarray:
    tolerance = np.float32(1.0e-5)
    lower = -tolerance
    upper = np.float32(1.0) + tolerance
    if (
        np.any(~np.isfinite(gripper))
        or np.any(gripper < lower)
        or np.any(gripper > upper)
    ):
        raise DatasetValidationError(
            f"Episode {episode_id} gripper actions must be finite and within "
            "[-1e-5, 1+1e-5]"
        )
    clipped = np.clip(gripper, np.float32(0.0), np.float32(1.0))
    return (clipped > np.float32(0.5)).astype(np.float32)


class BridgeDistributedBatchSampler:
    """Yield deterministic, resumable, episode-local batches for one DDP rank."""

    def __init__(
        self,
        records: Sequence[EpisodeRecord],
        *,
        local_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        num_batches: int,
    ) -> None:
        if local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if num_batches < 0:
            raise ValueError("num_batches must be non-negative")
        if any(record.length <= 0 for record in records):
            raise ValueError("all episode records must have positive length")
        self._records = tuple(records)
        if len(self._records) < world_size:
            raise ValueError("world_size cannot exceed the number of episodes")
        self.local_batch_size = int(local_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.num_batches = int(num_batches)
        self._batches_emitted = 0
        self._batches_committed: int | None = None

    def _epoch_records(self, epoch: int) -> list[EpisodeRecord]:
        records = list(self._records)
        random.Random(self.seed + epoch).shuffle(records)
        return records[self.rank :: self.world_size]

    def _epoch_size(self, epoch: int) -> int:
        records = list(self._records)
        random.Random(self.seed + epoch).shuffle(records)
        frames_per_rank = [
            sum(record.length for record in records[rank :: self.world_size])
            for rank in range(self.world_size)
        ]
        common_batches = min(frames_per_rank) // self.local_batch_size
        if common_batches <= 0:
            raise ValueError(
                "each rank must have at least one complete local batch per epoch"
            )
        return common_batches * self.local_batch_size

    def _epoch_refs(self, epoch: int) -> Iterator[BridgeFrameRef]:
        emitted = 0
        limit = self._epoch_size(epoch)
        for record in self._epoch_records(epoch):
            positions = list(range(record.length))
            frame_seed = self.seed + epoch * 1_000_003 + record.episode_id * 97
            random.Random(frame_seed).shuffle(positions)
            for frame_position in positions:
                if emitted >= limit:
                    return
                yield BridgeFrameRef(
                    epoch=epoch,
                    episode_id=record.episode_id,
                    frame_position=frame_position,
                )
                emitted += 1

    def _refs_from_offset(self, offset: int) -> Iterator[BridgeFrameRef]:
        epoch = 0
        while True:
            epoch_size = self._epoch_size(epoch)
            if offset < epoch_size:
                break
            offset -= epoch_size
            epoch += 1
        within_epoch = offset
        while True:
            refs = self._epoch_refs(epoch)
            for _ in range(within_epoch):
                next(refs)
            yield from refs
            epoch += 1
            within_epoch = 0

    def __iter__(self) -> Iterator[list[BridgeFrameRef]]:
        refs = self._refs_from_offset(self._batches_emitted * self.local_batch_size)
        while self._batches_emitted < self.num_batches:
            batch = [next(refs) for _ in range(self.local_batch_size)]
            self._batches_emitted += 1
            yield batch

    def __len__(self) -> int:
        return max(0, self.num_batches - self._batches_emitted)

    def state_dict(self) -> dict[str, int]:
        batches = (
            self._batches_emitted
            if self._batches_committed is None
            else self._batches_committed
        )
        return {"batches_emitted": batches}

    def mark_batch_consumed(self) -> None:
        """Commit one consumed batch, excluding DataLoader prefetch from checkpoints."""

        if self._batches_committed is None:
            self._batches_committed = 0
        if self._batches_committed >= self._batches_emitted:
            raise RuntimeError("cannot commit a batch before it has been emitted")
        self._batches_committed += 1

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        try:
            batches_emitted = int(state["batches_emitted"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("sampler state must contain an integer batches_emitted") from error
        if batches_emitted < 0 or batches_emitted > self.num_batches:
            raise ValueError(
                f"batches_emitted must be in [0, {self.num_batches}], got {batches_emitted}"
            )
        self._batches_emitted = batches_emitted
        self._batches_committed = batches_emitted


class BridgeFrameDataset:
    """Decode Bridge V2 episodes lazily and emit Octo-small training samples."""

    def __init__(
        self,
        adapter: LeRobotDatasetAdapter,
        *,
        statistics: BridgeV2NormalizationStatistics,
        dataset_name: str,
        action_horizon: int,
        primary_size: tuple[int, int] = (256, 256),
        episode_cache_size: int = 2,
        seed: int = 42,
        train: bool = True,
        tokenizer: Any | None = None,
        language_max_length: int = 16,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if len(primary_size) != 2 or any(int(value) <= 0 for value in primary_size):
            raise ValueError("primary_size must contain two positive integers")
        if episode_cache_size <= 0:
            raise ValueError("episode_cache_size must be positive")
        if len(adapter.vector_observation_keys) != 1:
            raise DatasetValidationError(
                "Bridge Octo-small requires exactly one vector observation feature"
            )
        if len(adapter.image_observation_keys) != 1:
            raise DatasetValidationError(
                "Bridge Octo-small requires exactly one primary image observation feature"
            )
        self.adapter = adapter
        self.statistics = statistics
        self.dataset_name = str(dataset_name)
        self.action_horizon = int(action_horizon)
        self.primary_size = tuple(int(value) for value in primary_size)
        self.episode_cache_size = int(episode_cache_size)
        self.seed = int(seed)
        self.train = bool(train)
        self.tokenizer = tokenizer
        self.language_max_length = int(language_max_length)
        self._records = {record.episode_id: record for record in adapter.episodes()}
        self._cache: OrderedDict[int, EpisodeData] = OrderedDict()
        self._token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._state_key = adapter.vector_observation_keys[0]
        self._image_key = adapter.image_observation_keys[0]

    def __len__(self) -> int:
        return sum(record.length for record in self._records.values())

    def _load_episode(self, episode_id: int) -> EpisodeData:
        if episode_id in self._cache:
            episode = self._cache.pop(episode_id)
            self._cache[episode_id] = episode
            return episode
        try:
            record = self._records[episode_id]
        except KeyError as error:
            raise IndexError(f"unknown Bridge episode {episode_id}") from error
        episode = self.adapter.load_episode(record, load_images=True)
        self._cache[episode_id] = episode
        while len(self._cache) > self.episode_cache_size:
            self._cache.popitem(last=False)
        return episode

    def _primary_image(
        self,
        frame: np.ndarray,
        reference: BridgeFrameRef,
    ) -> np.ndarray:
        array = np.asarray(frame, dtype=np.uint8)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise DatasetValidationError(f"Primary image must be RGB, got {array.shape}")
        image = Image.fromarray(array, mode="RGB")
        if self.train:
            image = self._augment_image(image, reference)
        image = image.resize(self.primary_size[::-1], resample=Image.Resampling.LANCZOS)
        channel_first = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        return channel_first / np.float32(127.5) - np.float32(1.0)

    def _augment_image(
        self,
        image: Image.Image,
        reference: BridgeFrameRef,
    ) -> Image.Image:
        randomizer = random.Random(
            self.seed
            + reference.epoch * 1_000_003
            + reference.episode_id * 97
            + reference.frame_position * 193
        )
        width, height = image.size
        area = width * height
        crop_width, crop_height = width, height
        for _ in range(10):
            target_area = area * randomizer.uniform(0.8, 1.0)
            aspect = randomizer.uniform(0.9, 1.1)
            candidate_width = int(round(math.sqrt(target_area * aspect)))
            candidate_height = int(round(math.sqrt(target_area / aspect)))
            if 0 < candidate_width <= width and 0 < candidate_height <= height:
                crop_width, crop_height = candidate_width, candidate_height
                break
        left = randomizer.randint(0, width - crop_width)
        top = randomizer.randint(0, height - crop_height)
        image = image.crop((left, top, left + crop_width, top + crop_height))
        image = ImageEnhance.Brightness(image).enhance(randomizer.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(randomizer.uniform(0.9, 1.1))
        image = ImageEnhance.Color(image).enhance(randomizer.uniform(0.9, 1.1))
        hue_shift = int(round(randomizer.uniform(-0.05, 0.05) * 255))
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 256
        return Image.fromarray(hsv, mode="HSV").convert("RGB")

    def _tokens(self, instruction: str) -> tuple[np.ndarray, np.ndarray] | None:
        if self.tokenizer is None:
            return None
        cached = self._token_cache.get(instruction)
        if cached is not None:
            return cached
        encoded = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=self.language_max_length,
            return_tensors="np",
        )
        result = (
            np.asarray(encoded["input_ids"][0], dtype=np.int64),
            np.asarray(encoded["attention_mask"][0], dtype=np.int64),
        )
        self._token_cache[instruction] = result
        return result

    def __getitem__(self, reference: BridgeFrameRef) -> dict[str, Any]:
        if not isinstance(reference, BridgeFrameRef):
            raise TypeError("BridgeFrameDataset indices must be BridgeFrameRef values")
        episode = self._load_episode(reference.episode_id)
        position = int(reference.frame_position)
        if position < 0 or position >= episode.length:
            raise IndexError(
                f"frame position {position} is outside episode {episode.episode_id}"
            )
        actions = np.asarray(episode.actions, dtype=np.float32)
        if actions.shape != (episode.length, 7):
            raise DatasetValidationError(
                f"Episode {episode.episode_id} actions must have shape "
                f"({episode.length}, 7), got {actions.shape}"
            )
        valid_steps = min(self.action_horizon, episode.length - position)
        action_window = actions[position : position + valid_steps]
        if valid_steps < self.action_horizon:
            action_window = np.concatenate(
                [
                    action_window,
                    np.repeat(
                        action_window[-1:], self.action_horizon - valid_steps, axis=0
                    ),
                ],
                axis=0,
            )
        normalized_action = self.statistics.normalize_action(action_window)
        normalized_action[:, 6] = _normalize_gripper_actions(
            action_window[:, 6],
            episode_id=episode.episode_id,
        )
        action_pad_mask = np.zeros((self.action_horizon, 7), dtype=np.bool_)
        action_pad_mask[:valid_steps] = True

        state = np.asarray(episode.observations[self._state_key][position], dtype=np.float32)
        if state.shape != (8,):
            raise DatasetValidationError(
                f"Episode {episode.episode_id} state must have shape (8,), got {state.shape}"
            )
        proprio = self.statistics.normalize_state(state)
        image = self._primary_image(
            episode.observations[self._image_key][position], reference
        )
        instruction = str(episode.task_name or "").strip()
        if not instruction:
            raise DatasetValidationError(
                f"Episode {episode.episode_id} has an empty language instruction"
            )
        sample: dict[str, Any] = {
            "image_primary": image[None, ...].astype(np.float32, copy=False),
            "proprio": proprio[None, ...].astype(np.float32, copy=False),
            "timestep_pad_mask": np.ones((1,), dtype=np.bool_),
            "action": normalized_action.astype(np.float32, copy=False),
            "action_pad_mask": action_pad_mask,
            "language_instruction": instruction,
            "dataset_name": self.dataset_name,
            "episode_index": episode.episode_id,
            "frame_index": int(episode.frame_indices[position]),
        }
        tokens = self._tokens(instruction)
        if tokens is not None:
            sample["language_input_ids"], sample["language_attention_mask"] = tokens
        return sample


@dataclass(frozen=True)
class BridgeTrainingData:
    dataset: BridgeFrameDataset
    batch_sampler: BridgeDistributedBatchSampler
    dataloader: Any
    selection_sha256: str
    normalization_path: Path


def _selection_sha256(
    adapter: LeRobotDatasetAdapter,
    *,
    dataset_name: str,
    action_horizon: int,
    sampling_contract: Mapping[str, Any],
    normalization_path: Path,
) -> str:
    digest = hashlib.sha256()
    digest.update(adapter.fingerprint().encode("ascii"))
    digest.update(dataset_name.encode("utf-8"))
    digest.update(str(action_horizon).encode("ascii"))
    digest.update(
        json.dumps(sampling_contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    with normalization_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    for record in adapter.episodes():
        digest.update(
            f"{record.episode_id}:{record.length}:{record.task_index}:{record.task_name}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def make_training_dataset(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    tokenizer: Any,
    rank: int,
    world_size: int,
) -> BridgeTrainingData:
    """Build the lazy Bridge dataset and rank-local PyTorch DataLoader."""

    try:
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError("PyTorch is required to build the Bridge training loader") from error

    data = config["data"]
    train = config["train"]
    adapter = LeRobotDatasetAdapter(
        {
            "path": str(paths["dataset"]),
            "use_images": True,
            "empty_task_policy": "exclude",
            "feature_keys": {
                "action": data["action_key"],
                "timestamp": "timestamp",
                "frame_index": "frame_index",
                "episode_index": "episode_index",
                "vector_observations": data["state_obs_keys"],
                "image_observations": [data["image_obs_keys"]["primary"]],
            },
        }
    )
    normalization_path = Path(paths["normalization"])
    statistics = BridgeV2NormalizationStatistics.load(normalization_path)
    retained_frames = sum(record.length for record in adapter.episodes())
    if (
        statistics.metadata_sha256 != adapter.fingerprint()
        or statistics.retained_episodes != len(adapter.episodes())
        or statistics.retained_frames != retained_frames
    ):
        raise DatasetValidationError(
            "Bridge normalization statistics do not match the filtered dataset"
        )
    dataset = BridgeFrameDataset(
        adapter,
        statistics=statistics,
        dataset_name=data["dataset_name"],
        action_horizon=int(data["action_horizon"]),
        primary_size=tuple(data["resize"]["primary"]),
        episode_cache_size=int(train["episode_cache_size"]),
        seed=int(train["seed"]),
        train=True,
        tokenizer=tokenizer,
    )
    batch_sampler = BridgeDistributedBatchSampler(
        adapter.episodes(),
        local_batch_size=int(train["micro_batch_size_per_gpu"]),
        rank=rank,
        world_size=world_size,
        seed=int(train["seed"]),
        num_batches=(
            int(train["max_steps"]) * int(train["gradient_accumulation_steps"])
        ),
    )
    worker_count = int(train["num_workers_per_rank"])
    loader_options: dict[str, Any] = {
        "batch_sampler": batch_sampler,
        "num_workers": worker_count,
        "pin_memory": True,
        "persistent_workers": worker_count > 0,
    }
    if worker_count > 0:
        loader_options["prefetch_factor"] = int(train["prefetch_factor"])
    dataloader = DataLoader(dataset, **loader_options)
    return BridgeTrainingData(
        dataset=dataset,
        batch_sampler=batch_sampler,
        dataloader=dataloader,
        selection_sha256=_selection_sha256(
            adapter,
            dataset_name=data["dataset_name"],
            action_horizon=int(data["action_horizon"]),
            normalization_path=normalization_path,
            sampling_contract={
                "seed": int(train["seed"]),
                "world_size": int(world_size),
                "local_batch_size": int(train["micro_batch_size_per_gpu"]),
                "gradient_accumulation_steps": int(
                    train["gradient_accumulation_steps"]
                ),
                "primary_size": list(data["resize"]["primary"]),
                "image_key": data["image_obs_keys"]["primary"],
                "state_keys": list(data["state_obs_keys"]),
                "action_key": data["action_key"],
            },
        ),
        normalization_path=normalization_path,
    )
