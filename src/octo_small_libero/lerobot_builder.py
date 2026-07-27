from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import (
    DEFAULT_TARGET_TASK,
    DemoRef,
    LiberoDataError,
    RunningMoments,
    iter_prior_demo_refs,
    load_hdf5_demo,
    target_demo_refs,
    task_to_dataset_name,
)
from .lerobot_v2 import (
    ACTION_KEY,
    CHUNKS_SIZE,
    CODEBASE_VERSION,
    DATA_PATH,
    DEFAULT_FPS,
    PRIMARY_IMAGE_KEY,
    STANDARD_FEATURES,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)


@dataclass
class ImageMoments:
    def __post_init__(self) -> None:
        self.count = 0
        self.total = np.zeros(3, dtype=np.float64)
        self.total_squared = np.zeros(3, dtype=np.float64)
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)

    def update(self, images: np.ndarray) -> None:
        values = np.asarray(images, dtype=np.float64) / 255.0
        if values.ndim != 4 or values.shape[-1] != 3:
            raise LiberoDataError(
                f"Expected RGB images [time, height, width, 3], found {values.shape}"
            )
        pixels = values.reshape(-1, 3)
        self.count += len(pixels)
        self.total += np.sum(pixels, axis=0)
        self.total_squared += np.sum(np.square(pixels), axis=0)
        self.minimum = np.minimum(self.minimum, np.min(pixels, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(pixels, axis=0))

    def as_dict(self) -> dict[str, list[list[list[float]]]]:
        if self.count <= 0:
            raise LiberoDataError("Cannot finalize empty image statistics")
        mean = self.total / self.count
        variance = np.maximum(self.total_squared / self.count - np.square(mean), 0.0)

        def channel_shape(values: np.ndarray) -> list[list[list[float]]]:
            return [[[float(value)]] for value in values]

        return {
            "mean": channel_shape(mean),
            "std": channel_shape(np.sqrt(variance)),
            "min": channel_shape(self.minimum),
            "max": channel_shape(self.maximum),
        }


def _features() -> Any:
    try:
        from datasets import Features, Image, Sequence, Value
    except ImportError as error:
        raise RuntimeError(
            "datasets, pyarrow, and Pillow are required to convert LIBERO to LeRobotDataset v2"
        ) from error

    return Features(
        {
            PRIMARY_IMAGE_KEY: Image(),
            WRIST_IMAGE_KEY: Image(),
            STATE_KEY: Sequence(Value("float32"), length=8),
            ACTION_KEY: Sequence(Value("float32"), length=7),
            "timestamp": Value("float32"),
            "frame_index": Value("int64"),
            "episode_index": Value("int64"),
            "index": Value("int64"),
            "task_index": Value("int64"),
        }
    )


def _feature_info() -> dict[str, dict[str, Any]]:
    return {
        PRIMARY_IMAGE_KEY: {
            "dtype": "image",
            "shape": [128, 128, 3],
            "names": ["height", "width", "channel"],
        },
        WRIST_IMAGE_KEY: {
            "dtype": "image",
            "shape": [128, 128, 3],
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": [8],
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]},
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": [7],
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
        },
        **STANDARD_FEATURES,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _remove_dataset_directory(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != output_root.resolve():
        raise LiberoDataError(f"Refusing to remove unexpected path: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def _build_dataset(
    references: Iterable[DemoRef],
    output: Path,
    *,
    fps: int,
) -> dict[str, Any]:
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError(
            "datasets, pyarrow, and Pillow are required to convert LIBERO to LeRobotDataset v2"
        ) from error

    references = list(references)
    if not references:
        raise LiberoDataError(f"Cannot build an empty LeRobot dataset at {output}")

    task_names: list[str] = []
    task_to_index: dict[str, int] = {}
    episodes: list[dict[str, Any]] = []
    action_stats = RunningMoments(7)
    state_stats = RunningMoments(8)
    primary_stats = ImageMoments()
    wrist_stats = ImageMoments()
    scalar_stats = {
        key: RunningMoments(1)
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
    }
    features = _features()
    global_index = 0

    for episode_index, reference in enumerate(references):
        values = load_hdf5_demo(reference, include_images=True)
        length = len(values[ACTION_KEY])
        if reference.language_instruction not in task_to_index:
            task_to_index[reference.language_instruction] = len(task_names)
            task_names.append(reference.language_instruction)
        task_index = task_to_index[reference.language_instruction]
        frame_index = np.arange(length, dtype=np.int64)
        timestamps = frame_index.astype(np.float32) / np.float32(fps)
        indices = np.arange(global_index, global_index + length, dtype=np.int64)
        episode_indices = np.full(length, episode_index, dtype=np.int64)
        task_indices = np.full(length, task_index, dtype=np.int64)

        payload = {
            PRIMARY_IMAGE_KEY: list(values["image"]),
            WRIST_IMAGE_KEY: list(values["wrist_image"]),
            STATE_KEY: values["state"],
            ACTION_KEY: values["action"],
            "timestamp": timestamps,
            "frame_index": frame_index,
            "episode_index": episode_indices,
            "index": indices,
            "task_index": task_indices,
        }
        dataset = Dataset.from_dict(payload, features=features, split="train")
        parquet_path = output / DATA_PATH.format(
            episode_chunk=episode_index // CHUNKS_SIZE,
            episode_index=episode_index,
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(parquet_path))

        action_stats.update(values["action"])
        state_stats.update(values["state"])
        primary_stats.update(values["image"])
        wrist_stats.update(values["wrist_image"])
        for key, array in (
            ("timestamp", timestamps),
            ("frame_index", frame_index),
            ("episode_index", episode_indices),
            ("index", indices),
            ("task_index", task_indices),
        ):
            scalar_stats[key].update(np.asarray(array, dtype=np.float64)[:, None])
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [reference.language_instruction],
                "length": length,
            }
        )
        global_index += length

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "libero",
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": len(task_names),
        "total_videos": 0,
        "total_chunks": (len(episodes) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH,
        "video_path": None,
        "features": _feature_info(),
    }
    stats = {
        PRIMARY_IMAGE_KEY: primary_stats.as_dict(),
        WRIST_IMAGE_KEY: wrist_stats.as_dict(),
        STATE_KEY: state_stats.as_dict(),
        ACTION_KEY: action_stats.as_dict(),
        **{key: value.as_dict() for key, value in scalar_stats.items()},
    }
    _write_json(output / "meta" / "info.json", info)
    _write_json(output / "meta" / "stats.json", stats)
    _write_jsonl(output / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(
        output / "meta" / "tasks.jsonl",
        ({"task_index": index, "task": task} for index, task in enumerate(task_names)),
    )
    return {
        "path": str(output),
        "episodes": len(episodes),
        "frames": global_index,
        "tasks": len(task_names),
        "source_files": sorted({str(reference.path) for reference in references}),
        "stats": str(output / "meta" / "stats.json"),
    }


def prepare_lerobot_v2(
    source_root: str | Path,
    output_root: str | Path,
    *,
    target_task: str = DEFAULT_TARGET_TASK,
    fps: int = DEFAULT_FPS,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not (source / "libero_90").is_dir() or not (source / "libero_10").is_dir():
        raise LiberoDataError(f"{source} must contain libero_90 and libero_10 directories")
    if fps != DEFAULT_FPS:
        raise LiberoDataError(f"Strict LeRobotDataset v2 output requires fps={DEFAULT_FPS}")

    target_name = task_to_dataset_name(target_task)
    prior_dir = output / "libero90"
    target_dir = output / target_name
    if prior_dir == target_dir:
        raise LiberoDataError("Target dataset name must differ from libero90")
    manifest_path = output / "conversion_manifest.json"
    existing = [path for path in (prior_dir, target_dir, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise LiberoDataError(
            "LeRobotDataset v2 output already exists; pass --overwrite to rebuild: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        invalid = [
            path
            for path in (prior_dir, target_dir)
            if path.is_symlink() or (path.exists() and not path.is_dir())
        ]
        if manifest_path.exists() and not manifest_path.is_file():
            invalid.append(manifest_path)
        if invalid:
            raise LiberoDataError(
                "Refusing to overwrite unexpected non-dataset paths: "
                + ", ".join(str(path) for path in invalid)
            )
        for path in (prior_dir, target_dir):
            _remove_dataset_directory(path, output)
        if manifest_path.is_file():
            manifest_path.unlink()

    output.mkdir(parents=True, exist_ok=True)
    target_references = target_demo_refs(source, target_task)
    prior_report = _build_dataset(iter_prior_demo_refs(source), prior_dir, fps=fps)
    target_report = _build_dataset(target_references, target_dir, fps=fps)
    report = {
        "format": "LeRobotDataset v2.0",
        "source": str(source),
        "output": str(output),
        "fps": fps,
        "prior_dataset": "libero90",
        "target_dataset": target_name,
        "prior": prior_report,
        "target": {
            **target_report,
            "demo_ids": [reference.demo_id for reference in target_references],
        },
        "source_files": sorted(
            set(prior_report["source_files"]) | set(target_report["source_files"])
        ),
        # Keep flat counts in the manifest for simple downstream auditing.
        "prior_trajectories": prior_report["episodes"],
        "prior_transitions": prior_report["frames"],
        "target_trajectories": target_report["episodes"],
        "target_transitions": target_report["frames"],
        "target_demo_ids": [reference.demo_id for reference in target_references],
    }
    report["manifest"] = str(manifest_path)
    _write_json(manifest_path, report)
    return report
