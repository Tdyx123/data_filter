"""SimplerEnv client CLI for the managed Qwen-VL OFT model service."""

from __future__ import annotations

from typing import Sequence

from qwen3_vl_groot.evaluate_simpler import build_parser as _shared_build_parser
from qwen3_vl_groot.evaluate_simpler import main as _shared_main


EVALUATION_ROUTE = "qwen-vl-oft-simpler-widowx-eval"
PREFLIGHT_ROUTE = "qwen-vl-oft-simpler-widowx-preflight"


def build_parser():
    return _shared_build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    return _shared_main(
        argv,
        evaluation_route=EVALUATION_ROUTE,
        preflight_route=PREFLIGHT_ROUTE,
        display_name="Qwen-VL OFT",
    )


if __name__ == "__main__":
    raise SystemExit(main())
