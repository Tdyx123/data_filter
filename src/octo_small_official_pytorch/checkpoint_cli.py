from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .checkpoint import (
    OfficialCheckpointError,
    validate_official_or_finetuned_checkpoint,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an official base or fine-tuned Octo-small PyTorch artifact."
    )
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = validate_official_or_finetuned_checkpoint(arguments.checkpoint)
    except OfficialCheckpointError as error:
        parser.exit(2, f"invalid official checkpoint: {error}\n")
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
