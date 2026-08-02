from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert five deterministic demonstrations from every LIBERO-10 task "
            "into one LeRobotDataset v2.0 dataset"
        )
    )
    parser.add_argument("--source", default="/data/dwb/datasets/LIBERO")
    parser.add_argument("--output", default="/data/dwb/datasets/LIBERO_lerobot")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    from .libero10_builder import prepare_libero10_lerobot_v2

    report = prepare_libero10_lerobot_v2(
        arguments.source,
        arguments.output,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

