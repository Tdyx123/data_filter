from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_vl_common.checkpointing import (
    CompactCheckpoint,
    CompactCheckpointError,
    inspect_compact_checkpoint as _inspect,
    load_compact_weights,
    save_compact_checkpoint as _save,
)


CHECKPOINT_FORMAT = "qwen-vl-oft-bridge-compact-v1"
CheckpointError = CompactCheckpointError


def inspect_compact_checkpoint(
    checkpoint_dir: str | Path,
    *,
    config: dict[str, Any],
) -> CompactCheckpoint:
    return _inspect(
        checkpoint_dir,
        config=config,
        expected_format=CHECKPOINT_FORMAT,
        allow_max_steps_change=True,
    )


def save_compact_checkpoint(
    engine: Any,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    model_path: str | Path,
    global_step: int,
    validation_mae: float | None,
    is_best: bool = False,
) -> Path:
    return _save(
        engine,
        output_dir,
        checkpoint_format=CHECKPOINT_FORMAT,
        config=config,
        model_path=model_path,
        global_step=global_step,
        validation_mae=validation_mae,
        is_best=is_best,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointError",
    "CompactCheckpoint",
    "inspect_compact_checkpoint",
    "load_compact_weights",
    "save_compact_checkpoint",
]
