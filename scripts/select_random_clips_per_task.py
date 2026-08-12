#!/usr/bin/env python3
"""Select a seeded random Top K percent of RelCore-compatible clips per task."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relcore.data import build_clip_records  # noqa: E402
from trajectory_data import DatasetValidationError, LeRobotDatasetAdapter  # noqa: E402


CLIP_LENGTH = 15
CLIP_STRIDE = 15
SELECTED_MANIFEST_NAME = "selected_manifest.jsonl"
SELECTION_REPORT_NAME = "selection_report.json"
TOP_K_PERCENT_ERROR = "top-k percent must be finite and in (0, 100]"


def _validate_top_k_percent(value: float) -> float:
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise ValueError(TOP_K_PERCENT_ERROR)
    return value


def _top_k_percent(value: str) -> float:
    try:
        return _validate_top_k_percent(float(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(TOP_K_PERCENT_ERROR) from error


def select_random_clips(
    dataset_root: Path,
    *,
    top_k_percent: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build RelCore candidates and randomly select a percentage within each task."""

    top_k_percent = _validate_top_k_percent(top_k_percent)
    resolved_dataset = dataset_root.expanduser().resolve()
    adapter = LeRobotDatasetAdapter(
        {
            "path": str(resolved_dataset),
            "use_images": False,
            "feature_keys": {"vector_observations": ["observation.state"]},
        }
    )
    episodes = list(adapter.episodes())
    clips = build_clip_records(
        episodes,
        length=CLIP_LENGTH,
        stride=CLIP_STRIDE,
    )
    if not clips:
        raise ValueError("dataset contains no complete 15-frame clips")

    clips_by_task: dict[int, list[Any]] = defaultdict(list)
    task_names: dict[int, str] = {}
    for clip in clips:
        clips_by_task[clip.task_index].append(clip)
        task_names[clip.task_index] = clip.task_name

    generator = random.Random(seed)
    chosen = []
    task_counts: dict[str, int] = {}
    for task_index in sorted(clips_by_task):
        candidates = clips_by_task[task_index]
        budget = math.ceil(len(candidates) * top_k_percent / 100.0)
        selected_for_task = generator.sample(candidates, k=budget)
        chosen.extend(selected_for_task)
        task_counts[str(task_index)] = budget

    dataset_name = resolved_dataset.name
    selected_rows: list[dict[str, Any]] = []
    for selection_order, clip in enumerate(chosen, start=1):
        row = asdict(clip)
        row.update(
            {
                "dataset_name": dataset_name,
                "dataset_path": str(resolved_dataset),
                "selected": True,
                "selection_order": selection_order,
            }
        )
        selected_rows.append(row)

    report = {
        "selection_method": "per_task_random",
        "dataset_name": dataset_name,
        "dataset_path": str(resolved_dataset),
        "top_k_percent": float(top_k_percent),
        "seed": seed,
        "clip": {"length": CLIP_LENGTH, "stride": CLIP_STRIDE},
        "number_of_episodes": len(episodes),
        "number_of_clips": len(clips),
        "selected_clips": len(selected_rows),
        "selection_ratio": len(selected_rows) / len(clips),
        "skipped_short_episodes": [
            episode.episode_id for episode in episodes if episode.length < CLIP_LENGTH
        ],
        "task_candidate_counts": {
            str(task_index): len(clips_by_task[task_index])
            for task_index in sorted(clips_by_task)
        },
        "task_counts": task_counts,
        "task_names": {
            str(task_index): task_names[task_index] for task_index in sorted(task_names)
        },
    }
    return selected_rows, report


def write_outputs(
    output_dir: Path,
    *,
    selected_rows: list[dict[str, Any]],
    report: dict[str, Any],
    force: bool = False,
) -> tuple[Path, Path]:
    """Atomically write the selection manifest and report."""

    resolved_output = output_dir.expanduser().resolve()
    manifest_path = resolved_output / SELECTED_MANIFEST_NAME
    report_path = resolved_output / SELECTION_REPORT_NAME
    existing = [path for path in (manifest_path, report_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"selection output already exists ({names}); pass --force")

    manifest_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected_rows
    )
    report_payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    resolved_output.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        for destination, payload in (
            (manifest_path, manifest_payload),
            (report_path, report_payload),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=resolved_output,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                temporary_paths.append(temporary)
                handle.write(payload)
        os.replace(temporary_paths[0], manifest_path)
        os.replace(temporary_paths[1], report_path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return manifest_path, report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a seeded random Top K percent of 15-frame clips per task"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-percent", type=_top_k_percent, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        selected_rows, report = select_random_clips(
            arguments.dataset_root,
            top_k_percent=arguments.top_k_percent,
            seed=arguments.seed,
        )
        manifest_path, report_path = write_outputs(
            arguments.output_dir,
            selected_rows=selected_rows,
            report=report,
            force=arguments.force,
        )
    except (DatasetValidationError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Candidates: {report['number_of_clips']}")
    print(f"Selected: {report['selected_clips']}")
    print(f"Selection ratio: {report['selection_ratio']:.6f}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
