from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

try:
    from torch.utils.data import Dataset as _TorchDataset
    from torch.utils.data import Sampler as _TorchSampler
except ImportError:  # Keep raw HDF5 -> LeRobot conversion usable without PyTorch.
    class _TorchDataset:  # type: ignore[no-redef]
        pass

    class _TorchSampler:  # type: ignore[no-redef]
        pass


DEFAULT_TARGET_TASK = (
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
)
TARGET_DEMONSTRATIONS = 5


class LiberoDataError(RuntimeError):
    """Raised when LIBERO HDF5 or LeRobot v2 data violates the expected contract."""


@dataclass(frozen=True)
class DemoRef:
    path: Path
    demo_id: str
    task_name: str
    language_instruction: str
    index: int


@dataclass
class RunningMoments:
    dimension: int

    def __post_init__(self) -> None:
        self.count = 0
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)
        self.minimum = np.full(self.dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(self.dimension, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.dimension:
            raise LiberoDataError(f"Expected [time, {self.dimension}] values, found {batch.shape}")
        if not len(batch):
            return
        batch_count = len(batch)
        batch_mean = np.mean(batch, axis=0)
        batch_m2 = np.sum((batch - batch_mean) ** 2, axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta**2 * self.count * batch_count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, np.min(batch, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(batch, axis=0))

    def as_dict(self) -> dict[str, list[float]]:
        if self.count <= 0:
            raise LiberoDataError("Cannot finalize empty statistics")
        return {
            "mean": self.mean.tolist(),
            "std": np.sqrt(self.m2 / self.count).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def task_to_dataset_name(task_name: str) -> str:
    value = task_name.strip()
    if value.endswith(".hdf5"):
        value = value[: -len(".hdf5")]
    if value.endswith("_demo"):
        value = value[: -len("_demo")]
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not value or any(character not in allowed for character in value):
        raise LiberoDataError(f"Invalid LIBERO task name: {task_name!r}")
    return value.lower()


def _numeric_demo_key(demo_id: str) -> int:
    try:
        return int(demo_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise LiberoDataError(f"Invalid LIBERO demo id: {demo_id!r}") from error


def select_target_demo_ids(
    demo_ids: Sequence[str],
    task_name: str,
    *,
    count: int = TARGET_DEMONSTRATIONS,
) -> list[str]:
    ordered = sorted(demo_ids)
    if len(ordered) < count:
        raise LiberoDataError(
            f"Target task {task_name!r} has {len(ordered)} demos; {count} are required"
        )
    digest = hashlib.sha256(task_name.encode("utf-8")).digest()
    seed = int.from_bytes(digest, byteorder="big") % (2**31 - 1)
    rng = np.random.RandomState(seed)
    return [str(value) for value in rng.permutation(ordered)[:count]]


def _problem_info(data_group: Any, path: Path) -> dict[str, Any]:
    raw = data_group.attrs.get("problem_info")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if raw is None:
        return {}
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as error:
        raise LiberoDataError(f"Invalid problem_info JSON in {path}") from error
    return value if isinstance(value, dict) else {}


def _fallback_instruction(task_name: str) -> str:
    value = task_name
    if "SCENE" in value:
        suffix = value.split("SCENE", 1)[1]
        value = suffix.split("_", 1)[1] if "_" in suffix else suffix
    return value.replace("_", " ").strip()


def inspect_hdf5_file(path: str | Path) -> tuple[list[str], str, str]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to read raw LIBERO demonstrations") from error

    target = Path(path).expanduser().resolve()
    try:
        with h5py.File(target, "r") as handle:
            if "data" not in handle:
                raise LiberoDataError(f"{target} has no data group")
            data = handle["data"]
            demo_ids = sorted(data.keys(), key=_numeric_demo_key)
            task_name = target.stem.removesuffix("_demo")
            info = _problem_info(data, target)
            instruction = str(info.get("language_instruction", "")).strip()
            return demo_ids, task_name, instruction or _fallback_instruction(task_name)
    except OSError as error:
        raise LiberoDataError(f"Could not read {target}: {error}") from error


def standardize_demo_arrays(
    observation: dict[str, np.ndarray],
    actions: np.ndarray,
    *,
    include_images: bool = True,
) -> dict[str, np.ndarray]:
    required = {"ee_pos", "ee_ori", "gripper_states"}
    if include_images:
        required.update({"agentview_rgb", "eye_in_hand_rgb"})
    missing = required.difference(observation)
    if missing:
        raise LiberoDataError(f"LIBERO observation is missing keys: {sorted(missing)}")

    action_values = np.asarray(actions, dtype=np.float32).copy()
    if action_values.ndim != 2 or action_values.shape[1] != 7:
        raise LiberoDataError(
            f"LIBERO actions must have shape [time, 7], found {action_values.shape}"
        )
    action_values[:, -1] = np.clip((1.0 - action_values[:, -1]) / 2.0, 0.0, 1.0)
    length = len(action_values)

    ee_pos = np.asarray(observation["ee_pos"], dtype=np.float32)
    ee_ori = np.asarray(observation["ee_ori"], dtype=np.float32)
    gripper = np.asarray(observation["gripper_states"], dtype=np.float32)
    if ee_pos.shape != (length, 3) or ee_ori.shape != (length, 3):
        raise LiberoDataError(
            f"Expected ee_pos/ee_ori shapes {(length, 3)}, found {ee_pos.shape}/{ee_ori.shape}"
        )
    if gripper.ndim != 2 or gripper.shape[0] != length or gripper.shape[1] < 1:
        raise LiberoDataError(f"Invalid gripper_states shape: {gripper.shape}")
    state = np.concatenate(
        [
            ee_pos,
            ee_ori,
            np.zeros((length, 1), dtype=np.float32),
            gripper[:, :1],
        ],
        axis=-1,
    )

    result = {"action": action_values, "state": state}
    if include_images:
        primary = np.asarray(observation["agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(observation["eye_in_hand_rgb"], dtype=np.uint8)
        expected = (length, 128, 128, 3)
        if primary.shape != expected or wrist.shape != expected:
            raise LiberoDataError(
                f"LIBERO RGB observations must have shape {expected}, "
                f"found {primary.shape}/{wrist.shape}"
            )
        result["image"] = np.flip(primary, axis=1).copy()
        result["wrist_image"] = np.flip(wrist, axis=1).copy()
    return result


def load_hdf5_demo(
    reference: DemoRef,
    *,
    include_images: bool = True,
) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to read raw LIBERO demonstrations") from error

    with h5py.File(reference.path, "r") as handle:
        try:
            demo = handle["data"][reference.demo_id]
            observation = {
                "ee_pos": demo["obs"]["ee_pos"][:],
                "ee_ori": demo["obs"]["ee_ori"][:],
                "gripper_states": demo["obs"]["gripper_states"][:],
            }
            if include_images:
                observation.update(
                    {
                        "agentview_rgb": demo["obs"]["agentview_rgb"][:],
                        "eye_in_hand_rgb": demo["obs"]["eye_in_hand_rgb"][:],
                    }
                )
            actions = demo["actions"][:]
        except KeyError as error:
            raise LiberoDataError(
                f"{reference.path}:{reference.demo_id} is missing {error}"
            ) from error
    return standardize_demo_arrays(observation, actions, include_images=include_images)


def iter_prior_demo_refs(source_root: str | Path) -> Iterator[DemoRef]:
    root = Path(source_root).expanduser().resolve()
    source = root / "libero_90"
    files = sorted(source.glob("*.hdf5"))
    if not files:
        raise LiberoDataError(f"No LIBERO-90 HDF5 files found in {source}")
    index = 0
    for path in files:
        demo_ids, task_name, instruction = inspect_hdf5_file(path)
        for demo_id in demo_ids:
            yield DemoRef(path, demo_id, task_name, instruction, index)
            index += 1


def target_demo_refs(
    source_root: str | Path,
    task_name: str = DEFAULT_TARGET_TASK,
) -> list[DemoRef]:
    root = Path(source_root).expanduser().resolve()
    path = root / "libero_10" / f"{task_name}_demo.hdf5"
    if not path.is_file():
        raise LiberoDataError(f"Missing LIBERO-10 target file: {path}")
    demo_ids, canonical_name, instruction = inspect_hdf5_file(path)
    selected = select_target_demo_ids(demo_ids, canonical_name)
    return [
        DemoRef(path, demo_id, canonical_name, instruction, index)
        for index, demo_id in enumerate(selected)
    ]


@dataclass(frozen=True)
class FrameIndex:
    """Index emitted by the balanced sampler and consumed by the combined dataset."""

    source: int
    frame: int


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for Octo-small LeRobot training; "
            "install requirements-octo-pytorch.txt"
        ) from error
    return torch


class LeRobotFrameDataset:
    """Random-access view over every frame in one LeRobotDataset v2 source."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_name: str,
        statistics: dict[str, Any],
        action_horizon: int = 8,
        primary_size: tuple[int, int] = (256, 256),
        wrist_size: tuple[int, int] = (128, 128),
        episode_cache_size: int = 2,
        tokenizer: Any | None = None,
        max_language_length: int = 16,
    ):
        from .lerobot_v2 import LeRobotV2Metadata

        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if episode_cache_size <= 0:
            raise ValueError("episode_cache_size must be positive")
        self.metadata = LeRobotV2Metadata(root)
        missing = self.metadata.missing_episode_files()
        if missing:
            raise LiberoDataError(
                f"{self.metadata.root}: missing {len(missing)} episode files; first is {missing[0]}"
            )
        self.dataset_name = dataset_name
        self.action_horizon = int(action_horizon)
        self.primary_size = tuple(int(value) for value in primary_size)
        self.wrist_size = tuple(int(value) for value in wrist_size)
        self.episode_cache_size = int(episode_cache_size)
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        self._episode_ends: list[int] = []
        total = 0
        for episode in self.metadata.episodes:
            total += episode.length
            self._episode_ends.append(total)
        self._length = total

        action = statistics["action"]
        proprio = statistics["proprio"]
        self.action_mean = np.asarray(action["mean"], dtype=np.float32)
        self.action_std = np.asarray(action["std"], dtype=np.float32) + np.float32(1.0e-8)
        self.proprio_mean = np.asarray(proprio["mean"], dtype=np.float32)
        self.proprio_std = np.asarray(proprio["std"], dtype=np.float32) + np.float32(1.0e-8)
        if self.action_mean.shape != (7,) or self.action_std.shape != (7,):
            raise LiberoDataError("Action normalization statistics must have shape [7]")
        if self.proprio_mean.shape != (8,) or self.proprio_std.shape != (8,):
            raise LiberoDataError("Proprio normalization statistics must have shape [8]")

        self._encoded_tasks: dict[str, tuple[Any, Any]] = {}
        if tokenizer is not None:
            for instruction in sorted(set(self.metadata.tasks.values())):
                encoded = tokenizer(
                    instruction,
                    padding="max_length",
                    truncation=True,
                    max_length=max_language_length,
                    return_tensors="pt",
                )
                self._encoded_tasks[instruction] = (
                    encoded["input_ids"][0].to(dtype=_require_torch().long),
                    encoded["attention_mask"][0].to(dtype=_require_torch().bool),
                )

    def __len__(self) -> int:
        return self._length

    def _locate(self, frame: int) -> tuple[int, int]:
        if frame < 0:
            frame += self._length
        if frame < 0 or frame >= self._length:
            raise IndexError(frame)
        episode_position = bisect_right(self._episode_ends, frame)
        start = 0 if episode_position == 0 else self._episode_ends[episode_position - 1]
        return episode_position, frame - start

    def _episode(self, episode_position: int) -> dict[str, Any]:
        from .lerobot_v2 import load_episode

        record = self.metadata.episodes[episode_position]
        key = record.episode_index
        if key in self._episode_cache:
            value = self._episode_cache.pop(key)
            self._episode_cache[key] = value
            return value
        value = load_episode(self.metadata, record)
        self._episode_cache[key] = value
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return value

    @staticmethod
    def _decode_image(value: bytes, size: tuple[int, int]) -> Any:
        torch = _require_torch()
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Pillow is required to decode LeRobot PNG images") from error
        with Image.open(BytesIO(value)) as image:
            image = image.convert("RGB")
            if image.size != (size[1], size[0]):
                image = image.resize((size[1], size[0]), resample=Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)

    def __getitem__(self, frame: int) -> dict[str, Any]:
        torch = _require_torch()
        episode_position, frame_position = self._locate(int(frame))
        episode = self._episode(episode_position)
        length = len(episode["action"])
        stop = min(frame_position + self.action_horizon, length)
        valid = stop - frame_position

        actions = np.zeros((self.action_horizon, 7), dtype=np.float32)
        raw_actions = np.asarray(episode["action"][frame_position:stop], dtype=np.float32)
        actions[:valid, :6] = (
            raw_actions[:, :6] - self.action_mean[None, :6]
        ) / self.action_std[None, :6]
        actions[:valid, 6] = raw_actions[:, 6]
        if valid < self.action_horizon:
            actions[valid:, 6] = raw_actions[-1, 6]
        action_pad_mask = np.zeros((self.action_horizon, 7), dtype=bool)
        action_pad_mask[:valid] = True

        proprio = np.asarray(episode["state"][frame_position], dtype=np.float32)
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        instruction = str(episode["language_instruction"][frame_position])
        result: dict[str, Any] = {
            "image_primary": self._decode_image(
                episode["image_primary"][frame_position], self.primary_size
            ).unsqueeze(0),
            "image_wrist": self._decode_image(
                episode["image_wrist"][frame_position], self.wrist_size
            ).unsqueeze(0),
            "proprio": torch.from_numpy(proprio.copy()).unsqueeze(0),
            "action": torch.from_numpy(actions),
            "action_pad_mask": torch.from_numpy(action_pad_mask),
            "timestep_pad_mask": torch.ones(1, dtype=torch.bool),
            "language_instruction": instruction,
            "dataset_name": self.dataset_name,
            "episode_index": int(episode["episode_index"][frame_position]),
            "frame_index": int(episode["frame_index"][frame_position]),
        }
        if instruction in self._encoded_tasks:
            input_ids, attention_mask = self._encoded_tasks[instruction]
            result["language_input_ids"] = input_ids.clone()
            result["language_attention_mask"] = attention_mask.clone()
        return result


class CombinedLeRobotDataset(_TorchDataset):
    """Routes sampler-generated source/frame pairs to target and prior datasets."""

    def __init__(self, sources: Sequence[LeRobotFrameDataset]):
        if len(sources) != 2:
            raise ValueError("DataMIL training requires exactly target and prior sources")
        self.sources = tuple(sources)

    @property
    def source_sizes(self) -> tuple[int, int]:
        return tuple(len(source) for source in self.sources)  # type: ignore[return-value]

    def __len__(self) -> int:
        return sum(self.source_sizes)

    def __getitem__(self, index: FrameIndex | tuple[int, int]) -> dict[str, Any]:
        if not isinstance(index, FrameIndex):
            index = FrameIndex(int(index[0]), int(index[1]))
        return self.sources[index.source][index.frame]


class BalancedDistributedBatchSampler(_TorchSampler):
    """Deterministic 1:1 target/prior batches shared consistently across DDP ranks."""

    def __init__(
        self,
        source_sizes: Sequence[int],
        *,
        local_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        num_batches: int | None = None,
    ):
        torch = _require_torch()
        if len(source_sizes) != 2 or any(int(size) <= 0 for size in source_sizes):
            raise ValueError("source_sizes must contain two positive frame counts")
        if local_batch_size <= 0 or local_batch_size % 2:
            raise ValueError("local_batch_size must be a positive even number")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.source_sizes = tuple(int(size) for size in source_sizes)
        self.local_batch_size = int(local_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.num_batches = num_batches
        self.step = 0
        self._generators = []
        self._permutations = []
        self._positions = [0, 0]
        for source, size in enumerate(self.source_sizes):
            generator = torch.Generator()
            generator.manual_seed(self.seed + source * 1_000_003)
            self._generators.append(generator)
            self._permutations.append(torch.randperm(size, generator=generator).tolist())

    def _take(self, source: int, count: int) -> list[int]:
        torch = _require_torch()
        values: list[int] = []
        while len(values) < count:
            position = self._positions[source]
            permutation = self._permutations[source]
            available = min(count - len(values), len(permutation) - position)
            values.extend(permutation[position : position + available])
            position += available
            if position == len(permutation):
                permutation = torch.randperm(
                    self.source_sizes[source], generator=self._generators[source]
                ).tolist()
                position = 0
                self._permutations[source] = permutation
            self._positions[source] = position
        return values

    def __iter__(self) -> Iterator[list[FrameIndex]]:
        produced = 0
        while self.num_batches is None or produced < self.num_batches:
            per_source_global = self.local_batch_size // 2 * self.world_size
            global_indices = [
                self._take(source, per_source_global) for source in range(2)
            ]
            start = self.rank * (self.local_batch_size // 2)
            batch: list[FrameIndex] = []
            for offset in range(self.local_batch_size // 2):
                batch.append(FrameIndex(0, global_indices[0][start + offset]))
                batch.append(FrameIndex(1, global_indices[1][start + offset]))
            self.step += 1
            produced += 1
            yield batch

    def __len__(self) -> int:
        if self.num_batches is None:
            raise TypeError("An unbounded sampler has no length")
        return self.num_batches

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "positions": list(self._positions),
            "permutations": [list(values) for values in self._permutations],
            "generator_states": [generator.get_state() for generator in self._generators],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if len(state.get("positions", [])) != 2 or len(state.get("permutations", [])) != 2:
            raise ValueError("Invalid balanced sampler state")
        self.step = int(state["step"])
        self._positions = [int(value) for value in state["positions"]]
        self._permutations = [
            [int(value) for value in values] for values in state["permutations"]
        ]
        generator_states = state.get("generator_states", [])
        if len(generator_states) != 2:
            raise ValueError("Invalid balanced sampler generator state")
        for generator, generator_state in zip(
            self._generators, generator_states, strict=True
        ):
            generator.set_state(generator_state)


@dataclass
class TorchTrainingData:
    dataloader: Any
    dataset: CombinedLeRobotDataset
    batch_sampler: BalancedDistributedBatchSampler
    sample_weights: np.ndarray


def make_training_dataset(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    tokenizer: Any,
    rank: int = 0,
    world_size: int = 1,
) -> TorchTrainingData:
    """Build the native PyTorch LeRobot v2 DataLoader used by Octo-small."""
    torch = _require_torch()
    from torch.utils.data import DataLoader

    from .checkpoint import load_lerobot_statistics

    data = config["data"]
    train = config["train"]
    statistics = load_lerobot_statistics(paths["prior_dataset"])
    names = [data["target_dataset"], data["prior_dataset"]]
    roots = [paths["target_dataset"], paths["prior_dataset"]]
    cache_size = int(train["episode_cache_size"])
    sources = [
        LeRobotFrameDataset(
            root,
            dataset_name=name,
            statistics=statistics,
            action_horizon=int(data["action_horizon"]),
            primary_size=tuple(data["resize"]["primary"]),
            wrist_size=tuple(data["resize"]["wrist"]),
            episode_cache_size=cache_size,
            tokenizer=tokenizer,
        )
        for name, root in zip(names, roots, strict=True)
    ]
    dataset = CombinedLeRobotDataset(sources)
    micro_batch_size = int(train["micro_batch_size_per_gpu"])
    num_batches = int(train["max_steps"]) * int(train["gradient_accumulation_steps"])
    sampler = BalancedDistributedBatchSampler(
        dataset.source_sizes,
        local_batch_size=micro_batch_size,
        rank=rank,
        world_size=world_size,
        seed=int(train["seed"]),
        num_batches=num_batches,
    )
    workers = int(train["num_workers_per_rank"])
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(train["prefetch_factor"]),
            }
        )
    dataloader = DataLoader(dataset, **loader_kwargs)
    return TorchTrainingData(
        dataloader=dataloader,
        dataset=dataset,
        batch_sampler=sampler,
        sample_weights=np.asarray([0.5, 0.5], dtype=np.float64),
    )
