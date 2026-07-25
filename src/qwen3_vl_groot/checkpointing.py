from __future__ import annotations

import json
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .config import resume_config_digest
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


def capture_rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    engine: Any,
    output_dir: str | Path,
    *,
    global_step: int,
    config: dict[str, Any],
    data_fingerprint: dict[str, Any],
    best_validation_mae: float | None = None,
    tag: str | None = None,
    update_latest: bool = True,
) -> Path:
    output = Path(output_dir)
    checkpoint_root = output / "checkpoints"
    checkpoint_tag = tag or f"global_step_{global_step:08d}"
    client_state = {
        "global_step": global_step,
        "config_sha256": resume_config_digest(config),
        "data_metadata_sha256": data_fingerprint["metadata_sha256"],
        "best_validation_mae": best_validation_mae,
        "rng_state": capture_rng_state(),
    }
    success = engine.save_checkpoint(
        str(checkpoint_root),
        tag=checkpoint_tag,
        client_state=client_state,
        save_latest=False,
    )
    if success is False:
        raise CheckpointError(f"DeepSpeed failed to save {checkpoint_root / checkpoint_tag}")
    if engine.global_rank == 0 and update_latest:
        _atomic_json(
            checkpoint_root / "latest_train_checkpoint.json",
            {"tag": checkpoint_tag, "global_step": global_step},
        )
    return checkpoint_root / checkpoint_tag


def _resolve_resume(output_dir: Path, resume: str) -> tuple[Path, str]:
    if resume == "latest":
        pointer = output_dir / "checkpoints" / "latest_train_checkpoint.json"
        if not pointer.is_file():
            raise CheckpointError(f"No latest checkpoint pointer at {pointer}")
        with pointer.open("r", encoding="utf-8") as handle:
            tag = str(json.load(handle)["tag"])
        return output_dir / "checkpoints", tag

    path = Path(resume).expanduser().resolve()
    if path.is_dir() and (path / "mp_rank_00_model_states.pt").is_file():
        return path.parent, path.name
    if path.is_dir() and (path / "latest_train_checkpoint.json").is_file():
        with (path / "latest_train_checkpoint.json").open("r", encoding="utf-8") as handle:
            tag = str(json.load(handle)["tag"])
        return path, tag
    raise CheckpointError(
        f"Resume path must be a DeepSpeed tag directory or checkpoint root: {path}"
    )


def load_training_checkpoint(
    engine: Any,
    output_dir: str | Path,
    resume: str,
    *,
    config: dict[str, Any],
    data_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_root, tag = _resolve_resume(Path(output_dir), resume)
    load_path, client_state = engine.load_checkpoint(
        str(checkpoint_root),
        tag=tag,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )
    if load_path is None:
        raise CheckpointError(f"DeepSpeed could not load {checkpoint_root / tag}")
    expected_config = resume_config_digest(config)
    if client_state.get("config_sha256") != expected_config:
        raise CheckpointError(
            "Checkpoint configuration differs from this run. Strict resume was refused."
        )
    if client_state.get("data_metadata_sha256") != data_fingerprint["metadata_sha256"]:
        raise CheckpointError(
            "Dataset metadata fingerprint differs from the checkpoint. Strict resume was refused."
        )
    if "rng_state" in client_state:
        restore_rng_state(client_state["rng_state"])
    return client_state


def retain_recent_checkpoints(output_dir: str | Path, keep: int) -> None:
    root = Path(output_dir) / "checkpoints"
    checkpoints = sorted(
        (path for path in root.glob("global_step_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    for path in checkpoints[:-keep]:
        shutil.rmtree(path)


def _stats_from_policy(policy: Any) -> QuantileStats:
    return QuantileStats(
        state_q01=policy.state_q01.detach().float().cpu().numpy(),
        state_q99=policy.state_q99.detach().float().cpu().numpy(),
        action_q01=policy.action_q01.detach().float().cpu().numpy(),
        action_q99=policy.action_q99.detach().float().cpu().numpy(),
        epsilon=float(policy.normalization_epsilon),
    )


def export_compact_checkpoint(
    engine: Any,
    target_dir: str | Path,
    *,
    config: dict[str, Any],
    model_path: str | Path,
    global_step: int,
    validation_mae: float | None,
) -> None:
    """Export only LoRA, action-head weights, normalization, and configuration."""
    target = Path(target_dir)
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
