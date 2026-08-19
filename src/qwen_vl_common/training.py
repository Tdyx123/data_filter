from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader

from .checkpointing import CompactCheckpoint
from .data import (
    BridgeEpisodeDataset,
    BridgeMetadata,
    as_torch_iterable,
    bridge_collate,
    compute_quantile_stats,
)
from .normalization import QuantileStats
from .schedules import LoraUpdateSchedule


class PerformanceWindow:
    """Accumulate steady-state timing for completed optimizer steps."""

    def __init__(
        self,
        *,
        effective_batch_size: int,
        start_step: int,
        started_at: float,
    ) -> None:
        if effective_batch_size <= 0:
            raise ValueError("effective_batch_size must be positive")
        self.effective_batch_size = effective_batch_size
        self.reset(start_step=start_step, started_at=started_at)

    def add_data_wait(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("data wait cannot be negative")
        self.data_wait_seconds += seconds

    def exclude_elapsed(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("excluded elapsed time cannot be negative")
        self.started_at += seconds

    def metrics(self, *, optimizer_step: int, now: float) -> dict[str, float]:
        completed_steps = optimizer_step - self.start_step
        elapsed = now - self.started_at
        if completed_steps <= 0:
            raise ValueError("optimizer_step must advance beyond start_step")
        if elapsed <= 0:
            raise ValueError("now must be later than started_at")
        step_seconds = elapsed / completed_steps
        return {
            "performance/step_seconds": step_seconds,
            "performance/samples_per_second": self.effective_batch_size / step_seconds,
            "performance/data_wait_fraction": self.data_wait_seconds / elapsed,
        }

    def reset(self, *, start_step: int, started_at: float) -> None:
        self.start_step = start_step
        self.started_at = started_at
        self.data_wait_seconds = 0.0


def seed_everything(seed: int, rank: int) -> None:
    combined = seed + rank * 100_003
    random.seed(combined)
    np.random.seed(combined)
    torch.manual_seed(combined)
    torch.cuda.manual_seed_all(combined)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _should_save_checkpoint(*, step: int, save_every: int, improved: bool) -> bool:
    return improved or step % save_every == 0


def _should_save_final_checkpoint(*, step: int, last_checkpoint_step: int) -> bool:
    return step != last_checkpoint_step


def _lora_step_metrics(
    schedule: LoraUpdateSchedule,
    *,
    optimizer_step: int,
    learning_rate: float,
) -> dict[str, float | int]:
    return {
        "train/lora_lr": learning_rate,
        "train/lora_enabled": int(schedule.is_active(optimizer_step)),
        "train/lora_updates": schedule.active_steps_through(optimizer_step),
    }


def _initial_training_metrics(
    train_config: dict[str, Any],
    *,
    warm_start: CompactCheckpoint | None,
) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {
        "step": warm_start.global_step if warm_start is not None else 0,
        "config/action_head_learning_rate": float(train_config["head_learning_rate"]),
        "config/lora_learning_rate": float(train_config["lora_learning_rate"]),
    }
    if warm_start is not None:
        result.update(
            {
                "config/warm_start_checkpoint": str(warm_start.path),
                "config/warm_start_step": warm_start.global_step,
                "config/optimizer_state_restored": 0,
            }
        )
    return result


def _cosine_after_warmup(step: int, warmup: int, maximum: int) -> float:
    if step < warmup:
        return float(step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(maximum - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def build_optimizer_and_scheduler(
    policy: Any,
    config: dict[str, Any],
    *,
    initial_step: int = 0,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    train = config["train"]
    head_parameters = policy.action_head_parameters()
    lora_parameters = policy.lora_parameters()
    if not lora_parameters:
        raise RuntimeError("No LoRA parameters were found after full-layer injection")
    parameter_groups = [
        {
            "params": head_parameters,
            "lr": float(train["head_learning_rate"]),
            "weight_decay": float(train["weight_decay"]),
            "group_name": "action_head",
        },
        {
            "params": lora_parameters,
            "lr": float(train["lora_learning_rate"]),
            "weight_decay": float(train["weight_decay"]),
            "group_name": "qwen_lora",
        },
    ]
    optimizer_class: Any = torch.optim.AdamW
    optimizer_kwargs: dict[str, Any] = {}
    if int(train["deepspeed_stage"]) == 3 and bool(train["cpu_optimizer_offload"]):
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        optimizer_class = DeepSpeedCPUAdam
        optimizer_kwargs["adamw_mode"] = True
    optimizer = optimizer_class(
        parameter_groups,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        **optimizer_kwargs,
    )
    maximum = int(train["max_steps"])
    lora_warmup = int(train["lora_warmup_steps"])
    lora_update_schedule = LoraUpdateSchedule.from_train_config(train)
    maximum_lora_updates = lora_update_schedule.active_steps_through(maximum)

    def head_schedule(step: int) -> float:
        return _cosine_after_warmup(step, int(train["head_warmup_steps"]), maximum)

    def lora_schedule(completed_steps: int) -> float:
        upcoming_step = completed_steps + 1
        if not lora_update_schedule.is_active(upcoming_step):
            return 0.0
        active_before = lora_update_schedule.active_steps_through(completed_steps)
        return _cosine_after_warmup(
            active_before,
            lora_warmup,
            maximum_lora_updates,
        )

    if initial_step > 0:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[head_schedule, lora_schedule],
        last_epoch=initial_step - 1,
    )
    return optimizer, scheduler


def build_deepspeed_config(config: dict[str, Any], world_size: int) -> dict[str, Any]:
    train = config["train"]
    stage = int(train["deepspeed_stage"])
    result: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": int(train["micro_batch_size"]),
        "gradient_accumulation_steps": int(train["gradient_accumulation_steps"]),
        "train_batch_size": (
            int(train["micro_batch_size"])
            * int(train["gradient_accumulation_steps"])
            * world_size
        ),
        "bf16": {"enabled": bool(train["bf16"])},
        "gradient_clipping": float(train["max_grad_norm"]),
        "steps_per_print": 1_000_000,
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }
    zero = result["zero_optimization"]
    if stage == 2:
        zero.update(
            {
                "reduce_scatter": True,
                "reduce_bucket_size": 200_000_000,
                "allgather_bucket_size": 200_000_000,
            }
        )
    else:
        zero.update(
            {
                "stage3_gather_16bit_weights_on_model_save": True,
                "stage3_param_persistence_threshold": 100_000,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
            }
        )
    return result


def _make_loader(
    metadata: BridgeMetadata,
    episodes: list[Any],
    *,
    train: bool,
    rank: int,
    world_size: int,
    config: dict[str, Any],
) -> DataLoader:
    data = config["data"]
    implementation = BridgeEpisodeDataset(
        metadata,
        episodes,
        train=train,
        rank=rank,
        world_size=world_size,
        seed=int(config["train"]["seed"]),
        action_horizon=int(data["action_horizon"]),
        video_cache_size=int(data["video_cache_episodes"]),
    )
    workers = int(data["num_workers"])
    kwargs: dict[str, Any] = {
        "dataset": as_torch_iterable(implementation),
        "batch_size": int(config["train"]["micro_batch_size"]),
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": train,
        "collate_fn": bridge_collate,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(data["prefetch_factor"])
    return DataLoader(**kwargs)


def _reduce_scalar(value: torch.Tensor) -> float:
    reduced = value.detach().float()
    if distributed.is_initialized():
        distributed.all_reduce(reduced, op=distributed.ReduceOp.SUM)
        reduced /= distributed.get_world_size()
    return float(reduced.cpu())


class RankZeroLogger:
    def __init__(self, output_dir: Path, enabled: bool):
        self.enabled = enabled
        self.output_dir = output_dir
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.metrics_path = output_dir / "metrics.jsonl"
            self.writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    def log(self, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {"timestamp": time.time(), **metrics}
        serialized = json.dumps(payload, sort_keys=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
        print(serialized, flush=True)
        step = int(metrics["step"])
        for key, value in metrics.items():
            if key != "step" and isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def status(self, message: str) -> None:
        if self.enabled:
            print(message, flush=True)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _copy_checkpoint_normalization(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _prepare_stats(
    metadata: BridgeMetadata,
    train_episodes: list[Any],
    output: Path,
    config: dict[str, Any],
    rank: int,
    *,
    warm_start: CompactCheckpoint | None = None,
) -> QuantileStats:
    cache_path = output / "data_cache" / "normalization.json"
    if rank == 0:
        if warm_start is None:
            compute_quantile_stats(
                metadata,
                train_episodes,
                cache_path,
                epsilon=float(config["data"]["normalization_epsilon"]),
            )
        else:
            _copy_checkpoint_normalization(warm_start.normalization_path, cache_path)
    distributed.barrier()
    return QuantileStats.load(cache_path)


__all__ = [
    "PerformanceWindow",
    "RankZeroLogger",
    "build_deepspeed_config",
    "build_optimizer_and_scheduler",
    "seed_everything",
]
