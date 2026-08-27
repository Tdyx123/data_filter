#!/usr/bin/env python3
"""Print read-only Cocore BridgeData V2 action-classification diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cocore_bridge_v2.action_diagnostics import (  # noqa: E402
    analyze_bridge_action_windows,
    validate_reference_acceptance,
)
from cocore_bridge_v2.config import DEFAULT_DATASET_PATH, build_config  # noqa: E402
from cocore_bridge_v2.preflight import validate_bridge_dataset  # noqa: E402
from trajectory_data import create_dataset  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only BridgeData V2 motion-primitive threshold diagnostics"
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-episodes", type=_positive_int, default=None)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--verify-reference", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        validate_bridge_dataset(arguments.dataset_path)
        config = build_config(
            relation="sequence",
            relation_weight=1.0,
            dataset_path=arguments.dataset_path,
            max_episodes=arguments.max_episodes,
        )
        adapter = create_dataset(config["dataset"])
        report = analyze_bridge_action_windows(
            adapter,
            max_episodes=arguments.max_episodes,
            num_workers=arguments.num_workers,
        )
        failures = validate_reference_acceptance(report) if arguments.verify_reference else ()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    report["reference_acceptance"] = {
        "checked": arguments.verify_reference,
        "passed": arguments.verify_reference and not failures,
        "failures": list(failures),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
