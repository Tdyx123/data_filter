"""Fail-fast validation for the Bridge Octo-small route."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from octo_small_libero.checkpoint import inspect_octo_checkpoint
from trajectory_data import DatasetValidationError, LeRobotDatasetAdapter

from .data import BridgeFrameDataset, BridgeFrameRef


class PreflightError(RuntimeError):
    """Raised when Bridge training cannot safely start."""


def _adapter(root: Path) -> LeRobotDatasetAdapter:
    return LeRobotDatasetAdapter(
        {
            "path": str(root),
            "use_images": True,
            "empty_task_policy": "exclude",
            "feature_keys": {
                "action": "action",
                "timestamp": "timestamp",
                "frame_index": "frame_index",
                "episode_index": "episode_index",
                "vector_observations": ["observation.state"],
                "image_observations": ["observation.images.image_0"],
            },
        }
    )


def inspect_bridge_dataset(root: str | Path) -> dict[str, Any]:
    """Validate metadata/stats and decode the first/middle/last retained episodes."""

    adapter = _adapter(Path(root).expanduser().resolve())
    if str(adapter.info.get("robot_type", "")).lower() != "widowx":
        raise DatasetValidationError("Bridge dataset robot_type must be widowx")
    if float(adapter.info.get("fps", 0)) != 5.0:
        raise DatasetValidationError("Bridge dataset must use 5 Hz observations")
    image_feature = adapter.features["observation.images.image_0"]
    if image_feature.get("dtype") != "video":
        raise DatasetValidationError("Bridge image_0 must use video storage")
    video_info = image_feature.get("info", {})
    if str(video_info.get("video.codec", "")).lower() != "av1":
        raise DatasetValidationError("Bridge image_0 video codec must be AV1")
    if adapter.features["observation.state"].get("shape") != [8]:
        raise DatasetValidationError("Bridge observation.state must have shape [8]")
    if adapter.features["action"].get("shape") != [7]:
        raise DatasetValidationError("Bridge action must have shape [7]")

    dataset = BridgeFrameDataset(
        adapter,
        dataset_name="bridge_orig_1.0.0",
        action_horizon=8,
        primary_size=(256, 256),
        train=False,
    )
    records = tuple(adapter.episodes())
    if not records:
        raise DatasetValidationError("Bridge dataset has no non-empty language episodes")
    sampled_indices = sorted({0, len(records) // 2, len(records) - 1})
    sampled_episodes: list[int] = []
    for index in sampled_indices:
        record = records[index]
        sample = dataset[
            BridgeFrameRef(epoch=0, episode_id=record.episode_id, frame_position=0)
        ]
        if sample["image_primary"].shape != (1, 3, 256, 256):
            raise DatasetValidationError(
                f"Episode {record.episode_id} produced an invalid primary image"
            )
        if not np.all(np.isfinite(sample["proprio"])):
            raise DatasetValidationError(
                f"Episode {record.episode_id} produced non-finite proprio"
            )
        sampled_episodes.append(record.episode_id)
    summary = adapter.dataset_summary()
    return {
        "path": str(adapter.root),
        "codebase_version": adapter.info["codebase_version"],
        "robot_type": adapter.info["robot_type"],
        "fps": adapter.info["fps"],
        **summary,
        "retained_frames": sum(record.length for record in records),
        "image_key": adapter.image_observation_keys[0],
        "state_key": adapter.vector_observation_keys[0],
        "action_key": adapter.action_key,
        "sampled_episodes": sampled_episodes,
        "metadata_sha256": adapter.fingerprint(),
    }


def run_preflight(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    try:
        checkpoint = inspect_octo_checkpoint(paths["model"])
        dataset = inspect_bridge_dataset(paths["dataset"])
    except (OSError, RuntimeError, ValueError) as error:
        raise PreflightError(str(error)) from error
    expected = config["data"]["expected_counts"]
    for name, expected_value in expected.items():
        if int(dataset[name]) != int(expected_value):
            raise PreflightError(
                f"Bridge {name}={dataset[name]}, expected {int(expected_value)}"
            )
    try:
        import torch
        import transformers
    except ImportError as error:
        raise PreflightError(
            "PyTorch and Transformers must be installed from "
            "requirements-octo-pytorch.txt"
        ) from error
    if not torch.cuda.is_available():
        raise PreflightError("CUDA is required for Octo-small Bridge training")
    requested = int(config["train"]["gpu_count"])
    visible = int(torch.cuda.device_count())
    if visible != requested:
        raise PreflightError(f"PyTorch sees {visible} CUDA devices, expected {requested}")
    if not torch.cuda.is_bf16_supported():
        raise PreflightError("The selected CUDA device does not support BF16")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, requested):
        raise PreflightError(f"WORLD_SIZE must be 1 or {requested}, got {world_size}")
    report = {
        "route": "octo-small-bridge-v2-pytorch",
        "checkpoint": checkpoint,
        "dataset": dataset,
        "runtime": {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "devices": {
            "gpu_ids": config["train"]["gpu_ids"],
            "visible_cuda_devices": visible,
            "world_size": world_size,
            "precision": "bf16",
            "global_batch_size": config["train"]["batch_size"],
        },
    }
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

