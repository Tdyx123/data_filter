"""Command-line entrypoint for the Cocore BridgeData V2 adapter."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

from cocore.action_variation import RELIABILITY_METRICS, normalize_reliability_metrics
from cocore.pipeline import (
    GRAPH_DIRECTORY,
    encode_stage,
    graph_stage,
    run_pipeline,
    scan_stage,
    select_stage,
    validate_output,
)

from .config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RELIABILITY_METRICS,
    DEFAULT_SELECTION_RATIO,
    build_config,
)
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
        "--execution-action-source", choices=("original_command",), default=None,
        help="declare original issued commands; state-difference labels are not supported",
    )
    parser.add_argument(
        "--execution-action-semantics", choices=("delta_from_observed_position",),
        default=None, help="declare translation relative to the current observed position",
    )
    parser.add_argument(
        "--execution-action-scale", nargs=3, type=float, metavar=("SX", "SY", "SZ"),
        default=None, help="three finite positive command-to-metre factors (no defaults)",
    )
    parser.add_argument(
        "--execution-alignment-confirmed", action="store_true", default=None,
        help="declare coordinate/time alignment and upstream clipping; not automatic verification. "
        "All four --execution-* options must be supplied together",
    )
    for name in ("position-speed-threshold", "gripper-speed-threshold", "angular-speed-threshold"):
        parser.add_argument(f"--dwell-{name}", type=float, default=None)
    parser.add_argument("--dwell-gripper-mode", choices=("continuous", "binary"), default=None)
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
        if command in {"build-graph", "select", "run"}:
            child.add_argument(
                "--support-k", type=_positive_int, default=None,
                help="override quality.knn shared by support/support_old (positive integer; default: configuration)",
            )
            child.add_argument("--no-use-stop-bucket", action="store_true")
            child.add_argument(
                "--reliability-metrics",
                nargs="+",
                choices=RELIABILITY_METRICS,
                default=list(DEFAULT_RELIABILITY_METRICS),
            )
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
        "--support-k", type=_positive_int, default=None,
        help="validate the shared support/support_old k against the saved output configuration",
    )
    validate.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    validate.add_argument(
        "--selection-ratio",
        type=_selection_ratio,
        default=DEFAULT_SELECTION_RATIO,
    )
    validate.add_argument("--max-episodes", type=_positive_int, default=None)
    validate.add_argument("--no-use-stop-bucket", action="store_true")
    validate.add_argument(
        "--reliability-metrics",
        nargs="+",
        choices=RELIABILITY_METRICS,
        default=list(DEFAULT_RELIABILITY_METRICS),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.reliability_metrics = list(
            normalize_reliability_metrics(
                getattr(args, "reliability_metrics", DEFAULT_RELIABILITY_METRICS)
            )
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    selection_ratio = getattr(args, "selection_ratio", DEFAULT_SELECTION_RATIO)
    dwell = {
        name: getattr(args, f"dwell_{name}")
        for name in (
            "position_speed_threshold",
            "gripper_speed_threshold",
            "angular_speed_threshold",
            "gripper_mode",
        )
        if getattr(args, f"dwell_{name}", None) is not None
    }
    config = build_config(
        action_execution_deviation={
            name: getattr(args, f"execution_{name}")
            for name in (
                "action_source", "action_semantics", "action_scale", "alignment_confirmed"
            )
            if getattr(args, f"execution_{name}") is not None
        } or None,
        support_k=getattr(args, "support_k", None),
        dwell=dwell or None,
        relation=args.relation,
        relation_weight=args.relation_weight,
        selection_ratio=selection_ratio,
        dataset_path=getattr(args, "dataset_path", DEFAULT_DATASET_PATH),
        max_episodes=getattr(args, "max_episodes", None),
        use_stop_bucket=not getattr(args, "no_use_stop_bucket", False),
        reliability_metrics=getattr(args, "reliability_metrics", DEFAULT_RELIABILITY_METRICS),
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
