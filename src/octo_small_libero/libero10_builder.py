from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from .data import (
    DemoRef,
    LiberoDataError,
    inspect_hdf5_file,
    load_hdf5_demo,
    select_target_demo_ids,
    task_to_dataset_name,
)
from .lerobot_builder import _build_dataset
from .lerobot_v2 import DEFAULT_FPS, LeRobotV2Metadata
from .libero10_tasks import (
    LIBERO_10_DEMOS_PER_TASK,
    LIBERO_10_TASK_COUNT,
    LIBERO_10_TASK_NAMES,
    LIBERO_10_TASKS,
)


LIBERO10_DATASET_NAME = "libero10_5"
LIBERO10_MANIFEST_NAME = "libero10_5_conversion_manifest.json"
LIBERO10_EXPECTED_TASKS = LIBERO_10_TASK_COUNT
LIBERO10_DEMOS_PER_TASK = LIBERO_10_DEMOS_PER_TASK
SELECTION_METHOD = "task_name_sha256"


def collect_libero10_five_demo_refs(
    source_root: str | Path,
) -> tuple[list[DemoRef], list[dict[str, Any]]]:
    """Select five deterministic demonstrations from every LIBERO-10 task."""
    source = Path(source_root).expanduser().resolve()
    libero10 = source / "libero_10"
    if not libero10.is_dir():
        raise LiberoDataError(f"Missing LIBERO-10 source directory: {libero10}")

    files = sorted(libero10.glob("*_demo.hdf5"))
    if len(files) != LIBERO10_EXPECTED_TASKS:
        raise LiberoDataError(
            f"Expected exactly {LIBERO10_EXPECTED_TASKS} LIBERO-10 task files in "
            f"{libero10}, found {len(files)}"
        )

    discovered: dict[str, tuple[Path, list[str], str]] = {}
    instructions: set[str] = set()
    for path in files:
        demo_ids, task_name, instruction = inspect_hdf5_file(path)
        task_to_dataset_name(task_name)
        instruction = instruction.strip()
        if not instruction:
            raise LiberoDataError(f"LIBERO-10 task has no language instruction: {path}")
        if task_name in discovered:
            raise LiberoDataError(f"Duplicate LIBERO-10 task name: {task_name}")
        if instruction in instructions:
            raise LiberoDataError(
                f"Duplicate LIBERO-10 language instruction: {instruction!r}"
            )
        instructions.add(instruction)
        discovered[task_name] = (path, demo_ids, instruction)

    discovered_names = set(discovered)
    expected_names = set(LIBERO_10_TASK_NAMES)
    if discovered_names != expected_names:
        raise LiberoDataError(
            "LIBERO-10 source tasks differ from the pinned evaluation order; "
            f"missing={sorted(expected_names - discovered_names)}, "
            f"extra={sorted(discovered_names - expected_names)}"
        )

    references: list[DemoRef] = []
    tasks: list[dict[str, Any]] = []
    for task_index, canonical in enumerate(LIBERO_10_TASKS):
        path, demo_ids, instruction = discovered[canonical.name]
        if instruction != canonical.language_instruction:
            raise LiberoDataError(
                f"LIBERO-10 task {canonical.name} has language instruction "
                f"{instruction!r}; expected {canonical.language_instruction!r}"
            )
        selected = select_target_demo_ids(
            demo_ids,
            canonical.name,
            count=LIBERO10_DEMOS_PER_TASK,
        )
        task_references = [
            DemoRef(
                path=path,
                demo_id=demo_id,
                task_name=canonical.name,
                language_instruction=instruction,
                index=len(references) + offset,
            )
            for offset, demo_id in enumerate(selected)
        ]
        # Validate state/action readability before creating any output directory.
        for reference in task_references:
            load_hdf5_demo(reference, include_images=False)
        references.extend(task_references)
        tasks.append(
            {
                "task_index": task_index,
                "task_name": canonical.name,
                "language_instruction": instruction,
                "source_file": str(path.resolve()),
                "demo_ids": selected,
                "episodes": LIBERO10_DEMOS_PER_TASK,
            }
        )

    expected_episodes = LIBERO10_EXPECTED_TASKS * LIBERO10_DEMOS_PER_TASK
    if len(references) != expected_episodes:
        raise LiberoDataError(
            f"Expected {expected_episodes} selected LIBERO-10 episodes, "
            f"found {len(references)}"
        )
    return references, tasks


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _validate_destination(dataset: Path, manifest: Path, *, overwrite: bool) -> None:
    existing = [path for path in (dataset, manifest) if path.exists()]
    if existing and not overwrite:
        raise LiberoDataError(
            "LIBERO-10 LeRobotDataset v2 output already exists; "
            "pass --overwrite to rebuild: "
            + ", ".join(str(path) for path in existing)
        )
    invalid: list[Path] = []
    if dataset.is_symlink() or (dataset.exists() and not dataset.is_dir()):
        invalid.append(dataset)
    if manifest.is_symlink() or (manifest.exists() and not manifest.is_file()):
        invalid.append(manifest)
    if invalid:
        raise LiberoDataError(
            "Refusing to overwrite unexpected LIBERO-10 output paths: "
            + ", ".join(str(path) for path in invalid)
        )


def prepare_libero10_lerobot_v2(
    source_root: str | Path,
    output_root: str | Path,
    *,
    fps: int = DEFAULT_FPS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one 10-task, 50-episode LeRobotDataset v2 from LIBERO-10."""
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if fps != DEFAULT_FPS:
        raise LiberoDataError(f"Strict LeRobotDataset v2 output requires fps={DEFAULT_FPS}")

    dataset = output / LIBERO10_DATASET_NAME
    manifest = output / LIBERO10_MANIFEST_NAME
    _validate_destination(dataset, manifest, overwrite=overwrite)
    references, tasks = collect_libero10_five_demo_refs(source)

    output.mkdir(parents=True, exist_ok=True)
    build_id = uuid4().hex
    staging = output / f".{LIBERO10_DATASET_NAME}.build-{build_id}"
    manifest_staging = output / f".{LIBERO10_MANIFEST_NAME}.{build_id}.tmp"
    backup = output / f".{LIBERO10_DATASET_NAME}.backup-{build_id}"
    try:
        dataset_report = _build_dataset(references, staging, fps=fps)
        metadata = LeRobotV2Metadata(staging)
        missing = metadata.missing_episode_files()
        if missing:
            raise LiberoDataError(
                f"Staged LIBERO-10 dataset is missing episode files: {missing[:3]}"
            )
        expected_episodes = LIBERO10_EXPECTED_TASKS * LIBERO10_DEMOS_PER_TASK
        if int(metadata.info["total_episodes"]) != expected_episodes:
            raise LiberoDataError(
                f"Staged LIBERO-10 dataset has {metadata.info['total_episodes']} episodes, "
                f"expected {expected_episodes}"
            )
        if int(metadata.info["total_tasks"]) != LIBERO10_EXPECTED_TASKS:
            raise LiberoDataError(
                f"Staged LIBERO-10 dataset has {metadata.info['total_tasks']} tasks, "
                f"expected {LIBERO10_EXPECTED_TASKS}"
            )
        expected_instructions = [task["language_instruction"] for task in tasks]
        if list(metadata.tasks.values()) != expected_instructions:
            raise LiberoDataError("Staged LIBERO-10 task mapping is inconsistent")

        episode_lengths = [episode.length for episode in metadata.episodes]
        for task_index, task in enumerate(tasks):
            start = task_index * LIBERO10_DEMOS_PER_TASK
            stop = start + LIBERO10_DEMOS_PER_TASK
            task["frames"] = sum(episode_lengths[start:stop])

        final_report = {
            **dataset_report,
            "path": str(dataset),
            "stats": str(dataset / "meta" / "stats.json"),
        }
        report = {
            "format": "LeRobotDataset v2.0",
            "source": str(source),
            "output": str(output),
            "dataset_name": LIBERO10_DATASET_NAME,
            "dataset_path": str(dataset),
            "fps": fps,
            "selection_method": SELECTION_METHOD,
            "demos_per_task": LIBERO10_DEMOS_PER_TASK,
            "total_tasks": LIBERO10_EXPECTED_TASKS,
            "total_episodes": expected_episodes,
            "total_frames": int(metadata.info["total_frames"]),
            "metadata_sha256": metadata.metadata_sha256(),
            "tasks": tasks,
            "dataset": final_report,
            "source_files": [task["source_file"] for task in tasks],
            "manifest": str(manifest),
        }
        _write_json(manifest_staging, report)

        if dataset.exists():
            dataset.rename(backup)
        try:
            staging.rename(dataset)
            manifest_staging.replace(manifest)
        except Exception:
            if dataset.is_dir():
                shutil.rmtree(dataset)
            if backup.is_dir():
                backup.rename(dataset)
            raise
        if backup.is_dir():
            shutil.rmtree(backup)
        return report
    finally:
        if staging.is_dir():
            shutil.rmtree(staging)
        if manifest_staging.is_file():
            manifest_staging.unlink()
        if backup.is_dir() and not dataset.exists():
            backup.rename(dataset)
