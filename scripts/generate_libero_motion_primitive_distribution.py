#!/usr/bin/env python3
"""Generate a global LIBERO ECoT motion-primitive distribution CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero_motion_primitives import (  # noqa: E402
    TailStrategy,
    compute_primitive_statistics,
    generate_motion_primitives,
    make_libero_config,
)
from trajectory_data import DatasetValidationError, LeRobotDatasetAdapter  # noqa: E402


STATE_KEY = "observation.state"
OUTPUT_NAME = "motion_primitive_distribution.csv"


def generate_distribution(
    dataset_root: Path,
    *,
    horizon: int,
    threshold: float,
    tail_strategy: TailStrategy,
) -> tuple[list[tuple[str, int, float]], int, int]:
    """Generate global primitive statistics without crossing episode boundaries."""
    adapter = LeRobotDatasetAdapter(
        {
            "path": str(dataset_root),
            "use_images": False,
            "feature_keys": {"vector_observations": [STATE_KEY]},
        }
    )
    config = make_libero_config(
        horizon=horizon,
        threshold=threshold,
        tail_strategy=tail_strategy,
    )

    labels: list[str] = []
    for episode in adapter.iter_episodes(num_workers=0, load_images=False):
        labels.extend(
            generate_motion_primitives(
                episode.observations[STATE_KEY],
                config,
            )
        )

    return compute_primitive_statistics(labels), len(adapter.episodes()), len(labels)


def write_distribution_csv(
    statistics: Sequence[tuple[str, int, float]],
    output_dir: Path,
) -> Path:
    """Atomically write primitive statistics to the configured output directory."""
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / OUTPUT_NAME
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{OUTPUT_NAME}.",
            suffix=".tmp",
            dir=resolved_output_dir,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(("primitive", "count", "proportion"))
            writer.writerows(statistics)
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate global LIBERO ECoT motion-primitive counts and proportions "
            "from a LeRobot v2 dataset."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument(
        "--tail-strategy",
        choices=("truncate", "clip", "pad_last"),
        default="truncate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the distribution command."""
    arguments = build_parser().parse_args(argv)
    try:
        statistics, episode_count, label_count = generate_distribution(
            arguments.dataset_root,
            horizon=arguments.horizon,
            threshold=arguments.threshold,
            tail_strategy=arguments.tail_strategy,
        )
        output_path = write_distribution_csv(statistics, arguments.output_dir)
    except (DatasetValidationError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Processed episodes: {episode_count}")
    print(f"Generated labels: {label_count}")
    print(f"CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
