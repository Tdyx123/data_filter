"""Fail-fast validation for the Bridge Octo-small route."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from octo_small_official_pytorch.checkpoint import validate_official_checkpoint
from octo_small_official_pytorch.policy import (
    OfficialActionStatistics,
    load_official_action_statistics,
)
from trajectory_data import DatasetValidationError, LeRobotDatasetAdapter

from .data import (
    BridgeDistributedBatchSampler,
    BridgeFrameDataset,
    BridgeFrameRef,
)
from .selection import (
    BridgePrefilteredSelection,
    resolve_bridge_prefiltered_selection,
)


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
                "vector_observations": [],
                "image_observations": ["observation.images.image_0"],
            },
        }
    )


def inspect_bridge_dataset(
    root: str | Path,
    *,
    adapter: LeRobotDatasetAdapter,
    statistics: OfficialActionStatistics,
    frame_positions_by_episode: Mapping[int, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Validate metadata/stats and decode representative retained training starts."""

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
    if adapter.features["action"].get("shape") != [7]:
        raise DatasetValidationError("Bridge action must have shape [7]")

    dataset = BridgeFrameDataset(
        adapter,
        statistics=statistics,
        dataset_name="bridge_orig_1.0.0",
        action_horizon=4,
        history_horizon=2,
        primary_size=(256, 256),
        train=False,
    )
    records = tuple(adapter.episodes())
    if not records:
        raise DatasetValidationError("Bridge dataset has no non-empty language episodes")
    if frame_positions_by_episode is None:
        available_refs = [
            BridgeFrameRef(epoch=0, episode_id=records[0].episode_id, frame_position=0),
            BridgeFrameRef(
                epoch=0,
                episode_id=records[len(records) // 2].episode_id,
                frame_position=records[len(records) // 2].length // 2,
            ),
            BridgeFrameRef(
                epoch=0,
                episode_id=records[-1].episode_id,
                frame_position=records[-1].length - 1,
            ),
        ]
    else:
        available_refs = [
            BridgeFrameRef(
                epoch=0,
                episode_id=record.episode_id,
                frame_position=frame_position,
            )
            for record in records
            for frame_position in frame_positions_by_episode.get(record.episode_id, ())
        ]
        if not available_refs:
            raise DatasetValidationError("Bridge prefiltered selection has no training starts")
    sampled_indices = sorted({0, len(available_refs) // 2, len(available_refs) - 1})
    records_by_id = {record.episode_id: record for record in records}
    sampled_episodes: list[int] = []
    sampled_frames: list[dict[str, int]] = []
    for index in sampled_indices:
        reference = available_refs[index]
        sample = dataset[reference]
        if sample["image_primary"].shape != (2, 3, 256, 256):
            raise DatasetValidationError(
                f"Episode {reference.episode_id} produced an invalid primary image"
            )
        if sample["action"].shape != (2, 4, 7):
            raise DatasetValidationError(
                f"Episode {reference.episode_id} produced an invalid action chunk"
            )
        if "proprio" in sample:
            raise DatasetValidationError("Official Bridge samples must not contain proprio")
        if not np.all(np.isfinite(sample["action"])):
            raise DatasetValidationError(
                f"Episode {reference.episode_id} produced non-finite actions"
            )
        expected_timestep_mask = np.asarray(
            [reference.frame_position > 0, True], dtype=np.bool_
        )
        if not np.array_equal(sample["timestep_pad_mask"], expected_timestep_mask):
            raise DatasetValidationError(
                f"Episode {reference.episode_id} produced an invalid timestep mask"
            )
        if reference.frame_position == records_by_id[reference.episode_id].length - 1:
            if not np.allclose(sample["action"][1], sample["action"][1, :1]):
                raise DatasetValidationError(
                    f"Episode {reference.episode_id} did not repeat its final action"
                )
        sampled_episodes.append(reference.episode_id)
        sampled_frames.append(
            {
                "episode_id": reference.episode_id,
                "frame_position": reference.frame_position,
            }
        )
    summary = adapter.dataset_summary()
    return {
        "path": str(adapter.root),
        "codebase_version": adapter.info["codebase_version"],
        "robot_type": adapter.info["robot_type"],
        "fps": adapter.info["fps"],
        **summary,
        "retained_frames": sum(record.length for record in records),
        "image_key": adapter.image_observation_keys[0],
        "action_key": adapter.action_key,
        "sampled_episodes": sampled_episodes,
        "sampled_frames": sampled_frames,
        "metadata_sha256": adapter.fingerprint(),
    }


def _validate_prefiltered_sampling(
    config: Mapping[str, Any],
    records: tuple[Any, ...],
    selection: BridgePrefilteredSelection,
) -> None:
    selected_ids = set(selection.frame_positions_by_episode)
    sampled_records = tuple(
        record for record in records if record.episode_id in selected_ids
    )
    train = config["train"]
    world_size = int(train["gpu_count"])
    for rank in range(world_size):
        sampler = BridgeDistributedBatchSampler(
            sampled_records,
            local_batch_size=int(train["micro_batch_size_per_gpu"]),
            rank=rank,
            world_size=world_size,
            seed=int(train["seed"]),
            num_batches=1,
            frame_positions_by_episode=selection.frame_positions_by_episode,
        )
        next(iter(sampler))


def run_preflight(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    try:
        checkpoint = validate_official_checkpoint(paths["model"])
        statistics = load_official_action_statistics(checkpoint.statistics_path)
        adapter = _adapter(paths["dataset"])
        prior_selection = None
        if (
            config["data"]
            .get("prior_selection", {})
            .get("prefiltered_scores")
            is not None
        ):
            records = tuple(adapter.episodes())
            prior_selection = resolve_bridge_prefiltered_selection(
                config,
                paths,
                records=records,
            )
            assert prior_selection is not None
            _validate_prefiltered_sampling(config, records, prior_selection)
        if prior_selection is not None:
            dataset = inspect_bridge_dataset(
                paths["dataset"],
                adapter=adapter,
                statistics=statistics,
                frame_positions_by_episode=prior_selection.frame_positions_by_episode,
            )
        else:
            dataset = inspect_bridge_dataset(
                paths["dataset"],
                adapter=adapter,
                statistics=statistics,
            )
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
        "route": "octo-small-official-bridge-pytorch",
        "checkpoint": checkpoint.as_dict(),
        "dataset": dataset,
        "normalization": statistics.as_dict(),
        "selection": (
            prior_selection.as_manifest()
            if prior_selection is not None
            else {
                "enabled": False,
                "mode": "all_non_empty_episodes",
                "training_starts": dataset["retained_frames"],
            }
        ),
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
