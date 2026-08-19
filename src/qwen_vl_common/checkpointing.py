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


class CompactCheckpointError(RuntimeError):
    """Raised when a compact policy checkpoint is missing or incompatible."""


@dataclass(frozen=True)
class CompactCheckpoint:
    path: Path
    global_step: int
    base_model: Path
    config: dict[str, Any]
    weights_path: Path
    normalization_path: Path


def _comparable_config(
    config: dict[str, Any],
    *,
    allow_max_steps_change: bool,
) -> dict[str, Any]:
    comparable = deepcopy(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )
    comparable.get("paths", {}).pop("output", None)
    if allow_max_steps_change:
        comparable.get("train", {}).pop("max_steps", None)
    return comparable


def inspect_compact_checkpoint(
    checkpoint_dir: str | Path,
    *,
    config: dict[str, Any],
    expected_format: str,
    allow_max_steps_change: bool = False,
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
        raise CompactCheckpointError(
            f"Compact checkpoint is missing required files: {', '.join(missing)}"
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise CompactCheckpointError("Compact checkpoint manifest must be a JSON object")
        if manifest.get("format") != expected_format:
            raise CompactCheckpointError(
                f"Unsupported compact checkpoint format: {manifest.get('format')}"
            )
        global_step = manifest["global_step"]
        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise CompactCheckpointError(
                "Compact checkpoint global_step must be a non-negative integer"
            )
        checkpoint_config = manifest["config"]
        if not isinstance(checkpoint_config, dict):
            raise CompactCheckpointError("Compact checkpoint config must be a mapping")
        base_model = Path(manifest["base_model"]).expanduser().resolve()
    except (KeyError, TypeError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompactCheckpointError(
            f"Invalid compact checkpoint manifest: {manifest_path}"
        ) from error
    expected_name = f"step-{global_step:08d}"
    if checkpoint.name != expected_name:
        raise CompactCheckpointError(
            f"Compact checkpoint directory {checkpoint.name!r} does not match "
            f"global_step {global_step} ({expected_name})"
        )
    current_base_model = Path(config["paths"]["model"]).expanduser().resolve()
    if base_model != current_base_model:
        raise CompactCheckpointError(
            f"Compact checkpoint base model {base_model} differs from {current_base_model}"
        )
    if _comparable_config(
        checkpoint_config,
        allow_max_steps_change=allow_max_steps_change,
    ) != _comparable_config(config, allow_max_steps_change=allow_max_steps_change):
        allowed = "paths.output and train.max_steps" if allow_max_steps_change else "paths.output"
        raise CompactCheckpointError(
            "Compact checkpoint configuration differs from this warm-start run; "
            f"only {allowed} may change"
        )
    if int(config["train"]["max_steps"]) <= global_step:
        raise CompactCheckpointError(
            f"train.max_steps must be greater than warm-start step {global_step}"
        )
    current_output = Path(config["paths"]["output"]).expanduser().resolve()
    if current_output == checkpoint.parents[1]:
        raise ValueError(
            "Warm-start output must differ from the source training output "
            f"{checkpoint.parents[1]}"
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
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _read_pointer(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        checkpoint = str(payload["checkpoint"])
        step = int(payload["step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CompactCheckpointError(f"Invalid checkpoint pointer at {path}") from error
    if checkpoint != f"step-{step:08d}":
        raise CompactCheckpointError(f"Invalid checkpoint pointer at {path}")
    return checkpoint


def _retain_referenced(checkpoint_root: Path) -> None:
    referenced = {
        checkpoint
        for name in ("latest.json", "best.json")
        if (checkpoint := _read_pointer(checkpoint_root / name)) is not None
    }
    for path in checkpoint_root.iterdir():
        if path.is_dir() and re.fullmatch(r"step-\d{8}", path.name) and path.name not in referenced:
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
    checkpoint_format: str,
    config: dict[str, Any],
    model_path: str | Path,
    global_step: int,
    validation_mae: float | None,
    is_best: bool = False,
) -> Path:
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
            raise CompactCheckpointError(
                "DeepSpeed is required to gather a compact ZeRO-3 checkpoint"
            ) from error
    with gather_context:
        if engine.global_rank == 0:
            target.mkdir(parents=True, exist_ok=True)
            state = {
                name: named_parameters[name].detach().cpu().contiguous()
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
                    "format": checkpoint_format,
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
            _retain_referenced(checkpoint_root)
    return target


def load_compact_weights(policy: Any, checkpoint_dir: str | Path) -> None:
    checkpoint = Path(checkpoint_dir)
    state = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
    expected = set(policy.compact_parameter_names())
    found = set(state)
    if expected != found:
        raise CompactCheckpointError(
            "Compact checkpoint parameter mismatch; "
            f"missing={sorted(expected - found)[:10]}, "
            f"unexpected={sorted(found - expected)[:10]}"
        )
    incompatible = policy.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise CompactCheckpointError(
            f"Unexpected compact weights: {incompatible.unexpected_keys}"
        )
