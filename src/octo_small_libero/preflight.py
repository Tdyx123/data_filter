from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from .checkpoint import inspect_octo_checkpoint, load_lerobot_statistics
from .lerobot_v2 import (
    ACTION_KEY,
    PRIMARY_IMAGE_KEY,
    STATE_KEY,
    WRIST_IMAGE_KEY,
    LeRobotV2Metadata,
    load_episode,
)


class PreflightError(RuntimeError):
    """Raised when the Octo-small LIBERO route cannot safely start."""


def _verify_png(value: bytes, *, source: str) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required to validate LeRobot images") from error

    try:
        with Image.open(BytesIO(value)) as image:
            image_format = image.format
            image_mode = image.mode
            image_size = image.size
            image.verify()
    except Exception as error:
        raise RuntimeError(f"Could not decode embedded image from {source}: {error}") from error
    if image_format != "PNG" or image_mode != "RGB" or image_size != (128, 128):
        raise RuntimeError(
            f"{source}: expected embedded 128x128 RGB PNG, "
            f"found format={image_format}, mode={image_mode}, size={image_size}"
        )


def _inspect_lerobot_dataset(root: Path) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to inspect LeRobotDataset v2") from error

    metadata = LeRobotV2Metadata(root)
    if not metadata.episodes:
        raise RuntimeError(f"LeRobot dataset is empty: {metadata.root}")
    missing = metadata.missing_episode_files()
    if missing:
        raise RuntimeError(
            f"{metadata.root}: missing {len(missing)} episode files; first is {missing[0]}"
        )

    expected_columns = {
        PRIMARY_IMAGE_KEY,
        WRIST_IMAGE_KEY,
        STATE_KEY,
        ACTION_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    expected_schema = pa.schema(
        [
            pa.field(PRIMARY_IMAGE_KEY, image_type),
            pa.field(WRIST_IMAGE_KEY, image_type),
            pa.field(STATE_KEY, pa.list_(pa.float32(), list_size=8)),
            pa.field(ACTION_KEY, pa.list_(pa.float32(), list_size=7)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )
    for record in metadata.episodes:
        path = metadata.episode_path(record.episode_index)
        try:
            parquet = pq.ParquetFile(path)
        except Exception as error:
            raise RuntimeError(f"Could not read LeRobot Parquet footer {path}: {error}") from error
        columns = set(parquet.schema_arrow.names)
        if columns != expected_columns:
            raise RuntimeError(
                f"{path}: Parquet columns differ from LeRobot v2 schema; "
                f"missing={sorted(expected_columns - columns)}, "
                f"extra={sorted(columns - expected_columns)}"
            )
        if not parquet.schema_arrow.remove_metadata().equals(expected_schema):
            raise RuntimeError(f"{path}: Parquet physical types differ from LeRobot v2 schema")
        if b"huggingface" not in (parquet.schema_arrow.metadata or {}):
            raise RuntimeError(f"{path}: missing Hugging Face feature schema metadata")
        if parquet.metadata.num_rows != record.length:
            raise RuntimeError(
                f"{path}: Parquet has {parquet.metadata.num_rows} rows, "
                f"episodes.jsonl reports {record.length}"
            )

    sampled_records = [metadata.episodes[0]]
    if len(metadata.episodes) > 1:
        sampled_records.append(metadata.episodes[-1])
    for record in sampled_records:
        episode = load_episode(metadata, record)
        if not set(episode["language_instruction"]).issubset(record.tasks):
            raise RuntimeError(
                f"{metadata.episode_path(record.episode_index)}: task mapping mismatch"
            )
        for key in ("image_primary", "image_wrist"):
            _verify_png(
                episode[key][0],
                source=f"{metadata.episode_path(record.episode_index)}:{key}[0]",
            )
            _verify_png(
                episode[key][-1],
                source=f"{metadata.episode_path(record.episode_index)}:{key}[-1]",
            )

    return {
        "path": str(metadata.root),
        "codebase_version": metadata.info["codebase_version"],
        "fps": metadata.fps,
        "chunks_size": metadata.chunks_size,
        "episodes": int(metadata.info["total_episodes"]),
        "frames": int(metadata.info["total_frames"]),
        "tasks": int(metadata.info["total_tasks"]),
        "metadata_sha256": metadata.metadata_sha256(),
        "action_shape": list(metadata.features[ACTION_KEY]["shape"]),
        "state_shape": list(metadata.features[STATE_KEY]["shape"]),
        "image_shape": list(metadata.features[PRIMARY_IMAGE_KEY]["shape"]),
        "wrist_image_shape": list(metadata.features[WRIST_IMAGE_KEY]["shape"]),
        "embedded_png_samples_checked": len(sampled_records) * 4,
    }


def run_preflight(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    try:
        checkpoint = inspect_octo_checkpoint(paths["model"])
        statistics = load_lerobot_statistics(paths["prior_dataset"])
        prior = _inspect_lerobot_dataset(paths["prior_dataset"])
        target = _inspect_lerobot_dataset(paths["target_dataset"])
    except (OSError, RuntimeError, ValueError) as error:
        raise PreflightError(str(error)) from error

    if prior["episodes"] != int(statistics["num_trajectories"]):
        raise PreflightError(
            "LIBERO-90 LeRobot episode count differs from its normalization statistics"
        )
    if prior["episodes"] <= 0:
        raise PreflightError("LIBERO-90 prior must contain at least one episode")
    if target["episodes"] != 5:
        raise PreflightError(f"Expected the DataMIL five-demo target, found {target['episodes']}")
    if target["tasks"] != 1:
        raise PreflightError(f"Expected one target task mapping, found {target['tasks']} tasks")

    try:
        import torch
        import transformers
    except ImportError as error:
        raise PreflightError(
            "PyTorch and Transformers must be installed from "
            "requirements-octo-pytorch.txt"
        ) from error
    if not torch.cuda.is_available():
        raise PreflightError("CUDA is required for Octo-small training")
    visible_devices = torch.cuda.device_count()
    requested_devices = int(config["train"]["gpu_count"])
    if visible_devices != requested_devices:
        raise PreflightError(
            f"PyTorch sees {visible_devices} CUDA devices, expected {requested_devices}"
        )
    if not torch.cuda.is_bf16_supported():
        raise PreflightError("The selected CUDA device does not support BF16")
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, requested_devices):
        raise PreflightError(
            f"WORLD_SIZE must be 1 for preflight or {requested_devices} for training"
        )

    report = {
        "route": "octo-small-libero-pytorch",
        "independent_from": "qwen3-vl",
        "runtime": {
            "framework": "pytorch",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "checkpoint": checkpoint,
        "lerobot": {
            "root": str(paths["lerobot"]),
            "prior": prior,
            "target": target,
            "sample_weights": list(config["data"]["sample_weights"]),
            "raw_hdf5_checked": False,
        },
        "normalization": {
            "path": str(paths["statistics"]),
            "source_dataset": config["data"]["prior_dataset"],
            "trajectories": int(statistics["num_trajectories"]),
            "transitions": int(statistics["num_transitions"]),
            "action_mask": list(config["data"]["action_normalization_mask"]),
        },
        "restructure": {
            "retained_tokenizers": list(config["model"]["required_observation_tokenizers"]),
            "added_tokenizers": ["proprio"],
            "window_size": int(config["data"]["window_size"]),
            "action_horizon": int(config["data"]["action_horizon"]),
        },
        "devices": {
            "gpu_ids": list(config["train"]["gpu_ids"]),
            "visible_cuda_devices": visible_devices,
            "world_size": world_size,
            "precision": config["train"]["precision"],
            "global_batch_size": int(config["train"]["batch_size"]),
            "micro_batch_size_per_gpu": int(
                config["train"]["micro_batch_size_per_gpu"]
            ),
            "gradient_accumulation_steps": int(
                config["train"]["gradient_accumulation_steps"]
            ),
        },
    }
    if int(__import__("os").environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report
