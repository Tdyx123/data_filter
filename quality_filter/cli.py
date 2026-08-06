"""Command-line interface for staged Quality Filter runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .pipeline import (
    encode_stage,
    filter_stage,
    quality_stage,
    run_pipeline,
    validate_output,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _percent(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("percent must be a number") from error
    if not 0.0 < parsed <= 100.0:
        raise argparse.ArgumentTypeError("percent must be in (0, 100]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SQCN-compatible quality-only fragment filtering"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_config = str(Path(__file__).with_name("config_libero90.yaml"))
    for command in ("quality", "encode", "filter", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default=default_config)
        child.add_argument("--output-dir", default=None)
        child.add_argument("--max-episodes", type=_positive_int, default=None)
        child.add_argument("--force", action="store_true")
        if command in {"filter", "run"}:
            child.add_argument("--percent", type=_percent, default=None)
            child.add_argument("--seed", type=int, default=None)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--config", default=None)
    validate.add_argument("--percent", type=_percent, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "validate":
        config = load_config(arguments.config) if arguments.config else None
        result = validate_output(
            arguments.output_dir,
            config=config,
            percent=arguments.percent,
        )
        print(json.dumps(result, sort_keys=True))
        return
    config = load_config(arguments.config)
    if arguments.max_episodes is not None:
        config["runtime"]["max_episodes"] = arguments.max_episodes
    common = {
        "output_dir": arguments.output_dir,
        "force": arguments.force,
    }
    if arguments.command == "quality":
        root = quality_stage(config, **common)
        print(f"quality_filter_output={root}")
    elif arguments.command == "encode":
        root = encode_stage(config, **common)
        print(f"quality_filter_output={root}")
    elif arguments.command == "filter":
        root = filter_stage(
            config,
            percent=arguments.percent,
            seed=arguments.seed,
            **common,
        )
        print(f"quality_filter_filter_output={root}")
    else:
        root = run_pipeline(
            config,
            percent=arguments.percent,
            seed=arguments.seed,
            **common,
        )
        print(f"quality_filter_output={root}")


if __name__ == "__main__":
    main()
