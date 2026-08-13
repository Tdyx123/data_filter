#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_ROOT = Path("/data/dwb/octo_small_libero/")
TASK_COUNT = 10


class InvalidExperiment(ValueError):
    """Raised when one experiment does not contain ten valid completed results."""


def parse_task_success_rate(path: Path) -> float:
    value = json.loads(path.read_text(encoding="utf-8"))
    return float(value["summary"]["success_rate"])


def summarize_experiment(path: Path) -> float:
    rates = [
        parse_task_success_rate(path / f"task-{index}" / "results.json")
        for index in range(TASK_COUNT)
    ]
    return math.fsum(rates) / TASK_COUNT


def collect_success_rates(root: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for experiment in sorted(root.iterdir(), key=lambda item: item.name):
        if not experiment.is_dir():
            continue
        try:
            mean = summarize_experiment(experiment)
        except InvalidExperiment as error:
            print(f"skip {experiment.name}: {error}", file=sys.stderr)
            continue
        rows.append((experiment.name, mean))
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print mean LIBERO-10 task success rates for complete experiments."
    )
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = build_parser().parse_args(argv).root
    if not root.is_dir():
        print(f"error: result root is not a directory: {root}", file=sys.stderr)
        return 1
    try:
        rows = collect_success_rates(root)
    except OSError as error:
        print(f"error: cannot read result root {root}: {error}", file=sys.stderr)
        return 1
    for name, rate in rows:
        print(f"{name}  {rate:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
