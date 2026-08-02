"""SQCN command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .pipeline import load_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Segment Quality-Coverage-Novelty scores"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="SQCN YAML configuration",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="temporary episode limit for a smoke test",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="exact SQCN run directory (overrides output.root/<dataset>)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an incompatible existing SQCN output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise SystemExit("--max-episodes must be positive")
        config["runtime"]["max_episodes"] = args.max_episodes
    root = run_pipeline(config, force=args.force, output_dir=args.output_dir)
    print(f"sqcn_output={root}")
