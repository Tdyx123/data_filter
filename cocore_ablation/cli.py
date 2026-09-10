"""Command-line interface for LIBERO Cocore component ablations."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import load_config
from .pipeline import _validate_subfolder_name, run_pipeline, validate_output


def _nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("weight must be finite and non-negative")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("selection ratio must be in (0, 1]")
    return parsed


class _MetricsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        names = [name for value in values for name in value.split(",")]
        choices = [[], ["support_old"], ["action_jump"], ["support_old", "action_jump"]]
        metrics = [] if names == ["none"] else names
        if metrics not in choices:
            parser.error("reliability metrics must be none, support_old, action_jump, or support_old action_jump")
        setattr(namespace, self.dest, metrics)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("support k must be a positive integer")
    return parsed


def _subfolder_name(value: str) -> str:
    try:
        return _validate_subfolder_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _add_common_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reliability-metrics", nargs="+", action=_MetricsAction, default=None)
    parser.add_argument(
        "--prototype-representation",
        choices=("action_visual", "action_only"),
        default=None,
    )
    parser.add_argument("--support-k", type=_positive_int, default=None)
    parser.add_argument("--no-assignment-confidence", action="store_true")
    parser.add_argument("--no-use-stop-bucket", action="store_true")
    parser.add_argument("--relation", choices=("sequence", "cooccurrence"), default=None)
    parser.add_argument("--relation-weight", type=_nonnegative, default=None)
    parser.add_argument("--redundancy-weight", type=_nonnegative, default=None)
    parser.add_argument("--no-coverage-seed", action="store_true")
    parser.add_argument(
        "--selection-strategy",
        choices=("random_multibranch", "random"),
        default=None,
    )
    parser.add_argument("--selection-ratio", type=_ratio, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated LIBERO Cocore component ablations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_config = str(Path(__file__).with_name("config_libero90.yaml"))
    run = subparsers.add_parser("run")
    run.add_argument("--config", default=default_config)
    run.add_argument("--output-dir", default=None)
    run.add_argument("--subfolder-name", type=_subfolder_name, required=True)
    run.add_argument("--force", action="store_true")
    _add_common_overrides(run)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--config", default=None)
    _add_common_overrides(validate)
    return parser


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.reliability_metrics is not None:
        config["reliability_metrics"] = args.reliability_metrics
    if args.support_k is not None:
        config.setdefault("quality", {})["knn"] = args.support_k
    prototypes = config.setdefault("prototypes", {})
    if args.prototype_representation is not None:
        prototypes["representation"] = args.prototype_representation
    if args.no_assignment_confidence:
        prototypes["use_assignment_confidence"] = False
    if args.no_use_stop_bucket:
        prototypes["use_stop_bucket"] = False
    objective = config.setdefault("objective", {})
    if args.relation is not None:
        objective["relation"] = args.relation
    if args.relation_weight is not None:
        objective["relation_weight"] = args.relation_weight
    if args.redundancy_weight is not None:
        objective["redundancy_weight"] = args.redundancy_weight
    selection = config.setdefault("selection", {})
    if args.no_coverage_seed:
        selection["use_coverage_seed"] = False
    if args.selection_strategy is not None:
        selection["strategy"] = args.selection_strategy
    if args.selection_ratio is not None:
        selection["ratio"] = args.selection_ratio
        selection["budget"] = None


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        config_path = args.config
        if config_path is None and any(
            (
                args.reliability_metrics is not None,
                args.support_k is not None,
                args.prototype_representation is not None,
                args.no_assignment_confidence,
                args.no_use_stop_bucket,
                args.relation is not None,
                args.relation_weight is not None,
                args.redundancy_weight is not None,
                args.no_coverage_seed,
                args.selection_strategy is not None,
                args.selection_ratio is not None,
            )
        ):
            config_path = Path(args.output_dir) / "resolved_config.yaml"
        config = load_config(config_path) if config_path is not None else None
        if config is not None:
            _apply_overrides(config, args)
        print(json.dumps(validate_output(args.output_dir, config=config), sort_keys=True))
        return

    config = load_config(args.config)
    _apply_overrides(config, args)
    result = run_pipeline(
        config,
        output_dir=args.output_dir,
        subfolder_name=args.subfolder_name,
        force=args.force,
    )
    print(f"cocore_ablation_output={result}")
