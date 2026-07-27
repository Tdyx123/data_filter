from __future__ import annotations

import argparse
import json

from .data import DEFAULT_TARGET_TASK


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Convert LIBERO-90 and one five-demo LIBERO-10 target to LeRobotDataset v2.0")
    )
    parser.add_argument("--source", default="/data/dwb/datasets/LIBERO")
    parser.add_argument("--output", default="/data/dwb/datasets/LIBERO/lerobot")
    parser.add_argument("--target-task", default=DEFAULT_TARGET_TASK)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    from .lerobot_builder import prepare_lerobot_v2

    report = prepare_lerobot_v2(
        arguments.source,
        arguments.output,
        target_task=arguments.target_task,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
