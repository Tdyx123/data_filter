from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from .checkpoint import inspect_octo_checkpoint, load_lerobot_statistics
from .data import resolve_target_task_selection
from .lerobot_v2 import (
    ACTION_KEY,
    PRIMARY_IMAGE_KEY,
    STATE_KEY,
    WRIST_IMAGE_KEY,
    LeRobotV2Metadata,
    load_episode,
)
from .libero10_tasks import LIBERO_10_DEMOS_PER_TASK, LIBERO_10_TASK_COUNT


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
    target_only = bool(config["data"].get("target_only", False))
    try:
        checkpoint = inspect_octo_checkpoint(paths["model"])
        statistics_root = (
            paths["target_dataset"] if target_only else paths["prior_dataset"]
        )
        statistics = load_lerobot_statistics(statistics_root)
        target = _inspect_lerobot_dataset(paths["target_dataset"])
        target_selection = resolve_target_task_selection(config, paths)
        if target_only:
            prior = None
            prior_selection = None
        else:
            from .selection import resolve_prior_selection

            prior = _inspect_lerobot_dataset(paths["prior_dataset"])
            prior_selection = resolve_prior_selection(config, paths)
    except (OSError, RuntimeError, ValueError) as error:
        raise PreflightError(str(error)) from error

    if target_only:
        if target["episodes"] != int(statistics["num_trajectories"]):
            raise PreflightError(
                "LIBERO-10 target episode count differs from its normalization statistics"
            )
    else:
        assert prior is not None
        if prior["episodes"] != int(statistics["num_trajectories"]):
            raise PreflightError(
                "LIBERO-90 LeRobot episode count differs from its normalization statistics"
            )
        if prior["episodes"] <= 0:
            raise PreflightError("LIBERO-90 prior must contain at least one episode")
    expected_target_episodes = LIBERO_10_TASK_COUNT * LIBERO_10_DEMOS_PER_TASK
    if target["episodes"] != expected_target_episodes:
        raise PreflightError(
            f"Expected {expected_target_episodes} LIBERO-10 target episodes, "
            f"found {target['episodes']}"
        )
    if target["tasks"] != LIBERO_10_TASK_COUNT:
        raise PreflightError(
            f"Expected {LIBERO_10_TASK_COUNT} target task mappings, "
            f"found {target['tasks']} tasks"
        )
    expected_selected_episodes = (
        len(target_selection.task_indices) * LIBERO_10_DEMOS_PER_TASK
    )
    if target_selection.episodes != expected_selected_episodes:
        raise PreflightError(
            f"Expected {expected_selected_episodes} selected target episodes, "
            f"found {target_selection.episodes}"
        )
    try:
        target_metadata = LeRobotV2Metadata(paths["target_dataset"])
        selected_task_indices = set(target_selection.task_indices)
        for episode_index in target_selection.episode_indices:
            record = target_metadata.episodes[episode_index]
            episode = load_episode(target_metadata, record)
            expected_task_index = episode_index // LIBERO_10_DEMOS_PER_TASK
            if expected_task_index not in selected_task_indices:
                raise RuntimeError(
                    f"{target_metadata.episode_path(episode_index)}: episode is outside "
                    "the selected LIBERO-10 task set"
                )
            if set(int(value) for value in episode["task_index"]) != {
                expected_task_index
            }:
                raise RuntimeError(
                    f"{target_metadata.episode_path(episode_index)}: task_index column "
                    f"does not match evaluation index {expected_task_index}"
                )
    except (OSError, RuntimeError, ValueError) as error:
        raise PreflightError(str(error)) from error

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

    lerobot_report = {
        "root": str(paths["lerobot"]),
        "target": target,
        "target_selection": target_selection.as_manifest(),
        "sample_weights": (
            [1.0] if target_only else list(config["data"]["sample_weights"])
        ),
        "raw_hdf5_checked": False,
    }
    if not target_only:
        assert prior is not None
        lerobot_report.update(
            {
                "prior": prior,
                "prior_selection": (
                    prior_selection.as_manifest()
                    if prior_selection is not None
                    else {"enabled": False}
                ),
            }
        )
    report = {
        "route": "octo-small-libero-pytorch",
        "training_mode": "target_only" if target_only else "mixed",
        "independent_from": "qwen3-vl",
        "runtime": {
            "framework": "pytorch",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "checkpoint": checkpoint,
        "lerobot": lerobot_report,
        "normalization": {
            "path": str(paths["statistics"]),
            "source_dataset": config["data"][
                "target_dataset" if target_only else "prior_dataset"
            ],
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
