from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import LiberoDataError
from .metadata import LeRobotV2Metadata
from .sampling import ACTION_WINDOW_POLICY, normalized_sample_weights
from .tasks import LIBERO_10_DEMOS_PER_TASK, LIBERO_10_TASK_COUNT, LIBERO_10_TASKS


@dataclass(frozen=True)
class TargetTaskSelection:
    dataset_name: str
    selection_mode: str
    task_index: int | None
    task_name: str | None
    language_instruction: str | None
    task_indices: tuple[int, ...]
    task_names: tuple[str, ...]
    language_instructions: tuple[str, ...]
    episode_indices: tuple[int, ...]
    frame_indices: tuple[int, ...]
    metadata_sha256: str
    selection_sha256: str

    @property
    def episodes(self) -> int:
        return len(self.episode_indices)

    @property
    def frames(self) -> int:
        return len(self.frame_indices)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self.selection_mode,
            "task_index": self.task_index,
            "task_name": self.task_name,
            "language_instruction": self.language_instruction,
            "task_indices": list(self.task_indices),
            "task_names": list(self.task_names),
            "language_instructions": list(self.language_instructions),
            "tasks": len(self.task_indices),
            "episode_indices": list(self.episode_indices),
            "episodes": self.episodes,
            "frames": self.frames,
            "action_window_policy": ACTION_WINDOW_POLICY,
            "metadata_sha256": self.metadata_sha256,
            "selection_sha256": self.selection_sha256,
        }


def resolve_target_selection(
    dataset_root: str | Path,
    *,
    dataset_name: str,
    task_indices: Sequence[int],
) -> TargetTaskSelection:
    indices = tuple(int(index) for index in task_indices)
    if not indices or len(indices) != len(set(indices)) or tuple(sorted(indices)) != indices:
        raise LiberoDataError("target task indices must be non-empty, unique, and sorted")
    if any(index < 0 or index >= LIBERO_10_TASK_COUNT for index in indices):
        raise LiberoDataError(
            f"target task indices must be in [0, {LIBERO_10_TASK_COUNT - 1}]"
        )

    metadata = LeRobotV2Metadata(dataset_root)
    expected_tasks = {
        index: task.language_instruction for index, task in enumerate(LIBERO_10_TASKS)
    }
    expected_episodes = LIBERO_10_TASK_COUNT * LIBERO_10_DEMOS_PER_TASK
    if metadata.tasks != expected_tasks:
        raise LiberoDataError(
            f"{metadata.root}: task_index mapping does not match the pinned LIBERO-10 order"
        )
    if (
        int(metadata.info["total_tasks"]) != LIBERO_10_TASK_COUNT
        or int(metadata.info["total_episodes"]) != expected_episodes
    ):
        raise LiberoDataError(
            f"{metadata.root}: expected {LIBERO_10_TASK_COUNT} tasks and "
            f"{expected_episodes} episodes"
        )
    for position, episode in enumerate(metadata.episodes):
        expected_instruction = LIBERO_10_TASKS[
            position // LIBERO_10_DEMOS_PER_TASK
        ].language_instruction
        if episode.tasks != (expected_instruction,):
            raise LiberoDataError(
                f"{metadata.root}: episode {episode.episode_index} is not ordered by "
                "the pinned LIBERO-10 task index"
            )

    selected_tasks = tuple(LIBERO_10_TASKS[index] for index in indices)
    selected_records = tuple(
        metadata.episodes[position]
        for index in indices
        for position in range(
            index * LIBERO_10_DEMOS_PER_TASK,
            (index + 1) * LIBERO_10_DEMOS_PER_TASK,
        )
    )
    selection_mode = (
        "all"
        if indices == tuple(range(LIBERO_10_TASK_COUNT))
        else ("single" if len(indices) == 1 else "subset")
    )
    episode_indices = tuple(record.episode_index for record in selected_records)
    frame_indices = tuple(
        frame
        for record in selected_records
        for frame in range(
            metadata.global_offsets[record.episode_index],
            metadata.global_offsets[record.episode_index] + record.length,
        )
    )
    if len(episode_indices) != len(indices) * LIBERO_10_DEMOS_PER_TASK or not frame_indices:
        raise LiberoDataError(f"{metadata.root}: target selection is empty or incomplete")

    metadata_sha256 = metadata.metadata_sha256()
    signature_payload = {
        "dataset_name": dataset_name,
        "mode": selection_mode,
        "task_indices": indices,
        "task_names": tuple(task.name for task in selected_tasks),
        "language_instructions": tuple(
            task.language_instruction for task in selected_tasks
        ),
        "episode_indices": episode_indices,
        "frames": len(frame_indices),
        "metadata_sha256": metadata_sha256,
    }
    selection_sha256 = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    single = selected_tasks[0] if len(indices) == 1 else None
    return TargetTaskSelection(
        dataset_name=dataset_name,
        selection_mode=selection_mode,
        task_index=indices[0] if single is not None else None,
        task_name=single.name if single is not None else None,
        language_instruction=single.language_instruction if single is not None else None,
        task_indices=indices,
        task_names=tuple(task.name for task in selected_tasks),
        language_instructions=tuple(
            task.language_instruction for task in selected_tasks
        ),
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        metadata_sha256=metadata_sha256,
        selection_sha256=selection_sha256,
    )


def resolve_target_task_selection(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> TargetTaskSelection:
    task_index = config["data"].get("target_task_index")
    all_tasks = bool(config["data"].get("target_all_tasks", False))
    if task_index is None and not all_tasks:
        raise LiberoDataError("a target task index or all-tasks selection is required")
    if task_index is not None and all_tasks:
        raise LiberoDataError("target task index and all-tasks cannot be combined")
    indices = (
        tuple(range(LIBERO_10_TASK_COUNT))
        if all_tasks
        else (int(task_index),)
    )
    return resolve_target_selection(
        paths["target_dataset"],
        dataset_name=str(config["data"]["target_dataset"]),
        task_indices=indices,
    )


def training_selection_sha256(
    target_selection: TargetTaskSelection,
    prior_selection: Any | None,
    sample_weights: Sequence[float] | None = None,
    *,
    training_mode: str | None = None,
    normalization_sha256: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "target": target_selection.selection_sha256,
        "prior": (
            prior_selection.selection_sha256 if prior_selection is not None else None
        ),
        "action_window_policy": ACTION_WINDOW_POLICY,
    }
    if sample_weights is not None:
        normalized = normalized_sample_weights(sample_weights)
        if normalized != tuple(0.5 for _ in normalized):
            payload["sample_weights"] = normalized
    if training_mode is not None:
        payload["training_mode"] = training_mode
    if normalization_sha256 is not None:
        payload["normalization_sha256"] = normalization_sha256
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
