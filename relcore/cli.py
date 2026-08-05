"""Command-line entrypoint for staged relational coreset selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .pipeline import (
    encode_stage,
    graph_stage,
    run_pipeline,
    scan_stage,
    select_stage,
    validate_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LIBERO relational coreset selection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_config = str(Path(__file__).with_name("config_libero90.yaml"))
    for command in ("scan", "encode", "build-graph", "select", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default=default_config)
        child.add_argument("--output-dir", default=None)
        child.add_argument("--max-episodes", type=int, default=None)
        child.add_argument("--force", action="store_true")
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
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise SystemExit("--max-episodes must be positive")
        config["runtime"]["max_episodes"] = args.max_episodes
    if args.command == "scan":
        root, _, clips, _ = scan_stage(config, output_dir=args.output_dir, force=args.force)
        print(f"relcore_output={root} clips={len(clips)}")
    elif args.command == "encode":
        root, _, artifact = encode_stage(config, output_dir=args.output_dir, force=args.force)
        print(f"relcore_output={root} clips={len(artifact.clips)}")
    elif args.command == "build-graph":
        root, _, _, graph, _ = graph_stage(config, output_dir=args.output_dir, force=args.force)
        print(f"relcore_output={root} nodes={len(graph.sample_ids)}")
    elif args.command == "select":
        root = select_stage(config, output_dir=args.output_dir, force=args.force)
        print(f"relcore_output={root}")
    else:
        root = run_pipeline(config, output_dir=args.output_dir, force=args.force)
        print(f"relcore_output={root}")


if __name__ == "__main__":
    main()
