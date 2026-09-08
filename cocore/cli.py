"""Command-line entrypoint for Cocore."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from .action_variation import RELIABILITY_METRICS, normalize_reliability_metrics
from .config import load_config
from .pipeline import (
    GRAPH_DIRECTORY,
    encode_stage,
    graph_stage,
    run_pipeline,
    scan_stage,
    select_stage,
    validate_output,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _selection_ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("selection ratio must be in (0, 1]")
    return parsed


def _relation_weight(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("relation weight must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LIBERO motion-primitive relation filter")
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
                "--support-k", type=_positive_int, default=None,
                help="override quality.knn for support (positive integer; default: configuration)",
            )
            child.add_argument("--no-use-stop-bucket", action="store_true")
            child.add_argument(
                "--reliability-metrics",
                nargs="+",
                choices=RELIABILITY_METRICS,
                default=None,
            )
        if command in {"select", "run"}:
            child.add_argument("--selection-ratio", type=_selection_ratio, default=None)
            child.add_argument("--relation", choices=("sequence", "cooccurrence"), default=None)
            child.add_argument("--relation-weight", type=_relation_weight, default=None)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--config", default=None)
    validate.add_argument(
        "--support-k", type=_positive_int, default=None,
        help="validate the support k against the saved output configuration",
    )
    validate.add_argument("--no-use-stop-bucket", action="store_true")
    validate.add_argument(
        "--reliability-metrics",
        nargs="+",
        choices=RELIABILITY_METRICS,
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "reliability_metrics", None) is not None:
        try:
            args.reliability_metrics = list(normalize_reliability_metrics(args.reliability_metrics))
        except ValueError as error:
            raise SystemExit(str(error)) from error
    if args.command == "validate":
        config_path = args.config
        if config_path is None and (
            args.no_use_stop_bucket or args.reliability_metrics is not None
            or args.support_k is not None
        ):
            config_path = Path(args.output_dir).expanduser() / "resolved_config.yaml"
        config = load_config(config_path) if config_path is not None else None
        if args.support_k is not None:
            config.setdefault("quality", {})["knn"] = args.support_k
        if args.no_use_stop_bucket:
            config.setdefault("prototypes", {})["use_stop_bucket"] = False
        if args.reliability_metrics is not None:
            config["reliability_metrics"] = args.reliability_metrics
        result = validate_output(args.output_dir, config=config)
        import json

        print(json.dumps(result, sort_keys=True))
        return
    config = load_config(args.config)
    if getattr(args, "support_k", None) is not None:
        config.setdefault("quality", {})["knn"] = args.support_k
    if getattr(args, "no_use_stop_bucket", False):
        config.setdefault("prototypes", {})["use_stop_bucket"] = False
    if getattr(args, "reliability_metrics", None) is not None:
        config["reliability_metrics"] = args.reliability_metrics
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise SystemExit("--max-episodes must be positive")
        config.setdefault("runtime", {})["max_episodes"] = args.max_episodes
    if args.command in {"select", "run"}:
        if args.selection_ratio is not None:
            config.setdefault("selection", {})["ratio"] = args.selection_ratio
            config["selection"]["budget"] = None
        if args.relation is not None:
            config.setdefault("objective", {})["relation"] = args.relation
        if args.relation_weight is not None:
            config.setdefault("objective", {})["relation_weight"] = args.relation_weight
    kwargs = {"output_dir": args.output_dir, "force": args.force}
    if args.command == "scan":
        root, _, clips, _ = scan_stage(config, **kwargs)
        print(f"cocore_output={root} clips={len(clips)}")
    elif args.command == "encode":
        root, _, artifact = encode_stage(config, **kwargs)
        print(f"cocore_output={root} clips={len(artifact.clips)}")
    elif args.command == "build-graph":
        root, _, _, graph, _ = graph_stage(config, **kwargs)
        print(f"cocore_output={root / GRAPH_DIRECTORY} nodes={len(graph.sample_ids)}")
    elif args.command == "select":
        print(f"cocore_output={select_stage(config, **kwargs)}")
    else:
        print(f"cocore_output={run_pipeline(config, **kwargs)}")
