"""Run SQCN diversity filtering with ``python -m sqcn.filtering``."""

from __future__ import annotations

import argparse
from typing import Sequence

from .artifacts import filter_sqcn_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter a completed SQCN run with diversity-aware reranking"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="completed SQCN run directory",
    )
    parser.add_argument(
        "--percent",
        required=True,
        type=float,
        help="global percentage of fragments to keep, in (0, 100]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional reproducible seed; omitted means a fresh random seed",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="exact filter output directory",
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="use quality instead of sqcn as the diversity reranking score",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing filter output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    root = filter_sqcn_run(
        arguments.input_dir,
        arguments.percent,
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        force=arguments.force,
        quality_only=arguments.quality_only,
    )
    print(f"sqcn_filter_output={root}")


if __name__ == "__main__":
    main()
