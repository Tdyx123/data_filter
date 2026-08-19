from __future__ import annotations

import json
import re
import shutil
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from .normalization import QuantileStats


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is absent or incompatible with the current run."""


@dataclass(frozen=True)
class CompactCheckpoint:
    path: Path
    global_step: int
    base_model: Path
    config: dict[str, Any]
    weights_path: Path
    normalization_path: Path


def _config_without_output(config: dict[str, Any]) -> dict[str, Any]:
    comparable = deepcopy(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )
    comparable.get("paths", {}).pop("output", None)
    return comparable


def inspect_compact_checkpoint(
    checkpoint_dir: str | Path,
    *,
    config: dict[str, Any],
) -> CompactCheckpoint:
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    manifest_path = checkpoint / "policy_config.json"
    weights_path = checkpoint / "adapter_model.safetensors"
    normalization_path = checkpoint / "normalization.json"
    missing = [
        path.name
        for path in (manifest_path, weights_path, normalization_path)
        if not path.is_file()
    ]
    if missing:
        raise CheckpointError(
            f"Compact checkpoint is missing required files: {', '.join(missing)}"
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise CheckpointError("Compact checkpoint manifest must be a JSON object")
        if manifest.get("format") != "qwen3-vl-groot-bridge-compact-v1":
            raise CheckpointError(
                f"Unsupported compact checkpoint format: {manifest.get('format')}"
            )
        global_step = manifest["global_step"]
        if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 0:
            raise CheckpointError("Compact checkpoint global_step must be a non-negative integer")
        checkpoint_config = manifest["config"]
        if not isinstance(checkpoint_config, dict):
            raise CheckpointError("Compact checkpoint config must be a mapping")
        base_model = Path(manifest["base_model"]).expanduser().resolve()
    except (KeyError, TypeError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Invalid compact checkpoint manifest: {manifest_path}") from error
    expected_name = f"step-{global_step:08d}"
    if checkpoint.name != expected_name:
        raise CheckpointError(
            f"Compact checkpoint directory {checkpoint.name!r} does not match "
            f"global_step {global_step} ({expected_name})"
        )
    current_base_model = Path(config["paths"]["model"]).expanduser().resolve()
    if base_model != current_base_model:
        raise CheckpointError(
            f"Compact checkpoint base model {base_model} differs from {current_base_model}"
        )
    if _config_without_output(checkpoint_config) != _config_without_output(config):
        raise CheckpointError(
            "Compact checkpoint configuration differs from this warm-start run; "
            "only paths.output may change"
        )
    if int(config["train"]["max_steps"]) <= global_step:
        raise CheckpointError(
            f"train.max_steps must be greater than warm-start step {global_step}"
        )
    current_output = Path(config["paths"]["output"]).expanduser().resolve()
    source_output = checkpoint.parents[1]
    if current_output == source_output:
        raise ValueError(
            "Warm-start output must differ from the source training output "
            f"{source_output}"
        )

    return CompactCheckpoint(
        path=checkpoint,
        global_step=global_step,
        base_model=base_model,
        config=checkpoint_config,
        weights_path=weights_path,
        normalization_path=normalization_path,
    )


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
