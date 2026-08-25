#!/usr/bin/env python3
"""Randomly select complete nonempty-task Bridge episodes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


DEFAULT_DATASET_ROOT = Path("/data/dwb/datasets/bridge_orig_1.0.0_lerobot/")
PERCENT_ERROR = "percent must be finite and in (0, 100]"


def _load_candidates(dataset_root: Path) -> tuple[int, list[tuple[int, int]]]:
    meta = dataset_root.expanduser().resolve() / "meta"
    with (meta / "info.json").open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if not isinstance(info, dict):
        raise ValueError("info.json must contain a JSON object")
    expected_count = info.get("total_episodes")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ValueError("info.json total_episodes must be a nonnegative integer")

    rows: list[tuple[int, dict[str, Any]]] = []
    episodes_path = meta / "episodes.jsonl"
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {episodes_path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"episodes.jsonl line {line_number}: row must be a JSON object"
                )
            rows.append((line_number, row))

    if len(rows) != expected_count:
        raise ValueError(
            f"episodes.jsonl has {len(rows)} entries, expected {expected_count}"
        )

    candidates: list[tuple[int, int]] = []
    seen_episode_ids: set[int] = set()
    for line_number, row in rows:
        episode_id = row.get("episode_index")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int):
            raise ValueError(
                f"episodes.jsonl line {line_number}: episode_index must be an integer"
            )
        if episode_id < 0:
            raise ValueError(
                f"episodes.jsonl line {line_number}: episode_index must be nonnegative"
            )
        if episode_id in seen_episode_ids:
            raise ValueError(f"duplicate episode_index={episode_id}")
        seen_episode_ids.add(episode_id)

        length = row.get("length")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"episodes.jsonl line {line_number}: length must be a positive integer"
            )

        tasks = row.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError(
                f"episodes.jsonl line {line_number}: tasks must be a one-item list"
            )
        task = tasks[0]
        if not isinstance(task, str):
            raise ValueError(
                f"episodes.jsonl line {line_number}: task must be a string"
            )
        if task.strip():
            candidates.append((episode_id, length))

    if not candidates:
        raise ValueError("dataset contains no nonempty-task episodes")
    return len(rows), candidates


def _percent(value: str) -> float:
    try:
        percent = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(PERCENT_ERROR) from error
    if not math.isfinite(percent) or not 0.0 < percent <= 100.0:
        raise argparse.ArgumentTypeError(PERCENT_ERROR)
    return percent


def _write_output(
    output_path: Path,
    selected: Sequence[tuple[int, int]],
    *,
    force: bool,
) -> Path:
    resolved_output = output_path.expanduser().resolve()
    if resolved_output.exists() and not force:
        raise FileExistsError("output already exists; pass --force")

    payload = "".join(
        json.dumps(
            {
                "episode_id": episode_id,
                "start_step": 0,
                "end_step": length - 1,
            }
        )
        + "\n"
        for episode_id, length in selected
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{resolved_output.name}.",
            suffix=".tmp",
            dir=resolved_output.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
        os.replace(temporary_path, resolved_output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return resolved_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Randomly select complete nonempty-task Bridge episodes"
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--percent", type=_percent, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        source_count, candidates = _load_candidates(arguments.dataset_root)
        selected_count = math.ceil(len(candidates) * arguments.percent / 100.0)
        selected = random.Random(arguments.seed).sample(candidates, k=selected_count)
        selected.sort(key=lambda episode: episode[0])
        output_path = _write_output(
            arguments.output_path,
            selected,
            force=arguments.force,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Source episodes: {source_count}")
    print(f"Excluded empty-task episodes: {source_count - len(candidates)}")
    print(f"Selected episodes: {len(selected)}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
