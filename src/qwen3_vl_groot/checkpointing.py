from __future__ import annotations

import json
import re
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from .normalization import QuantileStats


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is absent or incompatible with the current run."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _read_checkpoint_pointer(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        checkpoint = str(payload["checkpoint"])
        step = int(payload["step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Invalid checkpoint pointer at {path}") from error
    if checkpoint != f"step-{step:08d}":
        raise CheckpointError(f"Invalid checkpoint pointer at {path}")
    return checkpoint


def _retain_referenced_checkpoints(checkpoint_root: Path) -> None:
    referenced = {
        checkpoint
        for pointer in ("latest.json", "best.json")
        if (checkpoint := _read_checkpoint_pointer(checkpoint_root / pointer)) is not None
    }
    for path in checkpoint_root.iterdir():
        if (
            path.is_dir()
            and re.fullmatch(r"step-\d{8}", path.name)
            and path.name not in referenced
        ):
            shutil.rmtree(path)


def _stats_from_policy(policy: Any) -> QuantileStats:
    return QuantileStats(
        state_q01=policy.state_q01.detach().float().cpu().numpy(),
        state_q99=policy.state_q99.detach().float().cpu().numpy(),
        action_q01=policy.action_q01.detach().float().cpu().numpy(),
        action_q99=policy.action_q99.detach().float().cpu().numpy(),
        epsilon=float(policy.normalization_epsilon),
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
    """Save a step containing only LoRA, action-head weights, and inference metadata."""
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if is_best and validation_mae is None:
        raise ValueError("A best checkpoint requires validation_mae")

    checkpoint_root = Path(output_dir) / "checkpoints"
    checkpoint_name = f"step-{global_step:08d}"
    target = checkpoint_root / checkpoint_name
    policy = engine.module
    compact_names = set(policy.compact_parameter_names())
    named_parameters = dict(policy.named_parameters())
    parameters = [named_parameters[name] for name in sorted(compact_names)]

    gather_context = nullcontext()
    if int(config["train"]["deepspeed_stage"]) == 3:
        try:
            import deepspeed

            gather_context = deepspeed.zero.GatheredParameters(parameters, modifier_rank=0)
        except ImportError as error:
            raise CheckpointError(
                "DeepSpeed is required to gather a compact ZeRO-3 checkpoint"
            ) from error

    with gather_context:
        if engine.global_rank == 0:
            target.mkdir(parents=True, exist_ok=True)
            state = {
                name: named_parameters[name].detach().to(device="cpu").contiguous()
                for name in sorted(compact_names)
            }
            temporary = target / "adapter_model.safetensors.tmp"
            save_file(state, str(temporary))
            temporary.replace(target / "adapter_model.safetensors")
            _stats_from_policy(policy).save(target / "normalization.json")
            clean_config = {
                key: value for key, value in config.items() if not key.startswith("_")
            }
            _atomic_json(
                target / "policy_config.json",
                {
                    "format": "qwen3-vl-groot-bridge-compact-v1",
                    "base_model": str(Path(model_path).expanduser().resolve()),
                    "global_step": global_step,
                    "validation_mae": validation_mae,
                    "config": clean_config,
                    "parameter_names": sorted(compact_names),
                },
            )
            _atomic_json(
                checkpoint_root / "latest.json",
                {"checkpoint": checkpoint_name, "step": global_step},
            )
            if is_best:
                _atomic_json(
                    checkpoint_root / "best.json",
                    {
                        "checkpoint": checkpoint_name,
                        "step": global_step,
                        "validation_action_mae": validation_mae,
                    },
                )
            _retain_referenced_checkpoints(checkpoint_root)
    return target


def load_compact_weights(policy: Any, checkpoint_dir: str | Path) -> None:
    checkpoint = Path(checkpoint_dir)
    state = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
    expected = set(policy.compact_parameter_names())
    found = set(state)
    if expected != found:
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        raise CheckpointError(
            f"Compact checkpoint parameter mismatch; missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    incompatible = policy.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise CheckpointError(f"Unexpected compact weights: {incompatible.unexpected_keys}")
