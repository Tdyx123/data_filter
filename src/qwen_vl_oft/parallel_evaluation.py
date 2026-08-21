"""Multi-replica orchestration for Qwen-VL OFT SimplerEnv evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from qwen3_vl_groot import parallel_evaluation as _shared

from .simpler_evaluation import resolve_oft_checkpoint


def _validate_checkpoint(checkpoint: Path, model_path: Path | None) -> None:
    resolve_oft_checkpoint(checkpoint, model_path=model_path)


OFT_BACKEND = _shared.ParallelEvaluationBackend(
    display_name="Qwen-VL OFT",
    server_module="qwen_vl_oft.server",
    evaluator_module="qwen_vl_oft.evaluate_simpler",
    failure_route="qwen-vl-oft-simpler-widowx-parallel-eval",
    supports_denoising_steps=False,
    checkpoint_validator=_validate_checkpoint,
)


def build_parser():
    return _shared.build_parser(OFT_BACKEND)


def run_parallel(arguments):
    return _shared.run_parallel(arguments, backend=OFT_BACKEND)


def main(argv: Sequence[str] | None = None) -> int:
    return _shared.main(argv, backend=OFT_BACKEND)


if __name__ == "__main__":
    raise SystemExit(main())
