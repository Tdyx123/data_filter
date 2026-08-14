"""Command-line entrypoint for the Cocore BridgeData V2 adapter."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

from cocore.pipeline import (
    GRAPH_DIRECTORY,
    encode_stage,
    graph_stage,
    run_pipeline,
    scan_stage,
    select_stage,
    validate_output,
)

from .config import DEFAULT_DATASET_PATH, DEFAULT_SELECTION_RATIO, build_config
from .preflight import validate_bridge_dataset


_EXECUTION_COMMANDS = ("scan", "encode", "build-graph", "select", "run")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _relation_weight(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("relation weight must be finite and non-negative")
    return parsed


def _selection_ratio(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("selection ratio must be finite and in (0, 1]")
    return parsed


def _add_objective_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--relation",
        choices=("sequence", "cooccurrence"),
        required=True,
    )
    parser.add_argument("--relation-weight", type=_relation_weight, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cocore adapter for BridgeData V2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in _EXECUTION_COMMANDS:
        child = subparsers.add_parser(command)
        _add_objective_arguments(child)
        child.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
        child.add_argument("--output-dir", default=None)
        child.add_argument("--max-episodes", type=_positive_int, default=None)
        child.add_argument("--force", action="store_true")
        if command in {"select", "run"}:
            child.add_argument(
                "--selection-ratio",
                type=_selection_ratio,
                default=DEFAULT_SELECTION_RATIO,
            )

    validate = subparsers.add_parser("validate")
    _add_objective_arguments(validate)
    validate.add_argument("--output-dir", required=True)
    validate.add_argument(
        "--selection-ratio",
        type=_selection_ratio,
        default=DEFAULT_SELECTION_RATIO,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    selection_ratio = getattr(args, "selection_ratio", DEFAULT_SELECTION_RATIO)
    config = build_config(
        relation=args.relation,
        relation_weight=args.relation_weight,
        selection_ratio=selection_ratio,
        dataset_path=getattr(args, "dataset_path", DEFAULT_DATASET_PATH),
        max_episodes=getattr(args, "max_episodes", None),
    )
    if args.command == "validate":
        result = validate_output(args.output_dir, config=config)
        print(json.dumps(result, sort_keys=True))
        return

    validate_bridge_dataset(args.dataset_path)
    kwargs = {"output_dir": args.output_dir, "force": args.force}
    if args.command == "scan":
        root, _, clips, _ = scan_stage(config, **kwargs)
        print(f"cocore_output={root} clips={len(clips)}")
    elif args.command == "encode":
        root, _, artifact = encode_stage(config, **kwargs)
        print(f"cocore_output={root} clips={len(artifact.clips)}")
    elif args.command == "build-graph":
        root, _, _, graph, _ = graph_stage(config, **kwargs)
        print(f"cocore_output={Path(root) / GRAPH_DIRECTORY} nodes={len(graph.sample_ids)}")
    elif args.command == "select":
        print(f"cocore_output={select_stage(config, **kwargs)}")
    else:
        print(f"cocore_output={run_pipeline(config, **kwargs)}")
