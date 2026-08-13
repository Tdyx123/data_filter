"""Command-line entrypoint for staged relational coreset selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .pipeline import (
    encode_stage,
    graph_directory_name,
    graph_stage,
    run_pipeline,
    scan_stage,
    select_stage,
    validate_output,
)
from .scoring import RELIABILITY_METRICS, normalize_reliability_metrics
from .selection import PROTOTYPE_GAIN_METRICS, normalize_prototype_gain_metrics


def _selection_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("selection ratio must be a number") from error
    if not 0.0 < ratio <= 1.0:
        raise argparse.ArgumentTypeError("selection ratio must be in (0, 1]")
    return ratio


def _reliability_metrics(value: str) -> tuple[str, ...]:
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "reliability metrics must be a comma-separated list of metric names"
        )
    try:
        return normalize_reliability_metrics(tuple(part.strip() for part in parts))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"reliability metrics {error}") from error


def _prototype_gain_metrics(value: str) -> tuple[str, ...]:
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "prototype gain metrics must be a comma-separated list of metric names"
        )
    try:
        return normalize_prototype_gain_metrics(tuple(part.strip() for part in parts))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"prototype gain metrics {error}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeRobot relational coreset selection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_config = str(Path(__file__).with_name("config_libero90.yaml"))
    for command in ("scan", "encode", "build-graph", "select", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default=default_config)
        child.add_argument("--output-dir", default=None)
        child.add_argument("--max-episodes", type=int, default=None)
        child.add_argument("--force", action="store_true")
        if command in {"build-graph", "select", "run"}:
            child.add_argument(
                "--reliability-metrics",
                type=_reliability_metrics,
                default=RELIABILITY_METRICS,
                metavar="NAMES",
                help="comma-separated subset of support,progress,smoothness,non_noop",
            )
        if command in {"build-graph", "run"}:
            child.add_argument(
                "--prototype-method",
                choices=("kmeans", "motion_primitives"),
                default=None,
                help="override prototypes.method while building the graph",
            )
        if command in {"select", "run"}:
            child.add_argument(
                "--prototype-gain-metrics",
                type=_prototype_gain_metrics,
                default=PROTOTYPE_GAIN_METRICS,
                metavar="NAMES",
                help="comma-separated subset of transition,cooccurrence,sequence",
            )
            child.add_argument(
                "--selection-ratio",
                type=_selection_ratio,
                default=None,
                metavar="FLOAT",
                help="override selection ratio in (0, 1] and ignore configured budget",
            )
            child.add_argument(
                "--quota-mode",
                choices=("proportional", "none"),
                default=None,
                help="override task quota mode and isolate the selection output",
            )
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--config", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        config = load_config(args.config) if args.config is not None else None
        print(json.dumps(validate_output(args.output_dir, config=config), sort_keys=True))
        return
    config = load_config(args.config)
    if getattr(args, "prototype_method", None) is not None:
        config["prototypes"]["method"] = args.prototype_method
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise SystemExit("--max-episodes must be positive")
        config["runtime"]["max_episodes"] = args.max_episodes
    if args.command in {"select", "run"} and args.selection_ratio is not None:
        config["selection"]["ratio"] = args.selection_ratio
        config["selection"]["budget"] = None
    if args.command in {"select", "run"} and args.quota_mode is not None:
        config["selection"]["quota_mode"] = args.quota_mode
        if args.quota_mode == "none":
            config["selection"]["minimum_per_task"] = 0
    if args.command == "scan":
        root, _, clips, _ = scan_stage(config, output_dir=args.output_dir, force=args.force)
        print(f"relcore_output={root} clips={len(clips)}")
    elif args.command == "encode":
        root, _, artifact = encode_stage(config, output_dir=args.output_dir, force=args.force)
        print(f"relcore_output={root} clips={len(artifact.clips)}")
    elif args.command == "build-graph":
        root, _, _, graph, _ = graph_stage(
            config,
            output_dir=args.output_dir,
            force=args.force,
            reliability_metrics=args.reliability_metrics,
        )
        print(
            f"relcore_output={root / graph_directory_name(args.reliability_metrics, config['prototypes']['method'])} "
            f"nodes={len(graph.sample_ids)}"
        )
    elif args.command == "select":
        result = select_stage(
            config,
            output_dir=args.output_dir,
            force=args.force,
            selection_output_ratio=args.selection_ratio,
            selection_output_quota_mode=args.quota_mode,
            reliability_metrics=args.reliability_metrics,
            prototype_gain_metrics=args.prototype_gain_metrics,
        )
        print(f"relcore_output={result}")
    else:
        result = run_pipeline(
            config,
            output_dir=args.output_dir,
            force=args.force,
            selection_output_ratio=args.selection_ratio,
            selection_output_quota_mode=args.quota_mode,
            reliability_metrics=args.reliability_metrics,
            prototype_gain_metrics=args.prototype_gain_metrics,
        )
        print(f"relcore_output={result}")


if __name__ == "__main__":
    main()
