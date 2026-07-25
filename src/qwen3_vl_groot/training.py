from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader

from .checkpointing import (
    export_compact_checkpoint,
    load_training_checkpoint,
    retain_recent_checkpoints,
    save_training_checkpoint,
)
from .config import save_resolved_config
from .data import (
    BridgeEpisodeDataset,
    BridgeMetadata,
    as_torch_iterable,
    bridge_collate,
    compute_quantile_stats,
)
from .modeling import Qwen3VLGrootPolicy
from .normalization import QuantileStats


def seed_everything(seed: int, rank: int) -> None:
    combined = seed + rank * 100_003
    random.seed(combined)
    np.random.seed(combined)
    torch.manual_seed(combined)
    torch.cuda.manual_seed_all(combined)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _cosine_after_warmup(step: int, warmup: int, maximum: int) -> float:
    if step < warmup:
        return float(step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(maximum - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def build_optimizer_and_scheduler(
    policy: Qwen3VLGrootPolicy,
    config: dict[str, Any],
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
    freeze = int(train["lora_freeze_steps"])
    lora_warmup = int(train["lora_warmup_steps"])

    def head_schedule(step: int) -> float:
        return _cosine_after_warmup(step, int(train["head_warmup_steps"]), maximum)

    def lora_schedule(step: int) -> float:
        if step < freeze:
            return 0.0
        relative_step = step - freeze
        relative_maximum = max(maximum - freeze, 1)
        return _cosine_after_warmup(relative_step, lora_warmup, relative_maximum)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[head_schedule, lora_schedule]
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
    distributed.all_reduce(reduced, op=distributed.ReduceOp.SUM)
    reduced /= distributed.get_world_size()
    return float(reduced.cpu())


@torch.no_grad()
def evaluate(
    engine: Any,
    loader: DataLoader,
    *,
    maximum_batches: int,
    denoising_steps: int,
    seed: int,
) -> float:
    engine.eval()
    absolute_error = torch.zeros((), dtype=torch.float64, device=engine.device)
    element_count = torch.zeros((), dtype=torch.float64, device=engine.device)
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        generator = torch.Generator(device=engine.device)
        generator.manual_seed(seed + batch_index)
        initial_noise = torch.randn(
            batch["state"].shape[0],
            engine.module.action_head.horizon,
            engine.module.action_head.action_dim,
            device=engine.device,
            dtype=engine.module.compute_dtype,
            generator=generator,
        )
        prediction = engine.module.predict_actions(
            batch["images"],
            batch["state"],
            batch["instructions"],
            denoising_steps=denoising_steps,
            initial_noise=initial_noise,
        )
        target = batch["actions"].to(engine.device)
        mask = batch["action_mask"].to(engine.device).unsqueeze(-1)
        absolute_error += ((prediction - target).abs() * mask).double().sum()
        element_count += mask.double().sum() * target.shape[-1]
    distributed.all_reduce(absolute_error, op=distributed.ReduceOp.SUM)
    distributed.all_reduce(element_count, op=distributed.ReduceOp.SUM)
    if element_count.item() == 0:
        raise RuntimeError("Validation loader produced no action elements")
    engine.train()
    return float((absolute_error / element_count).cpu())


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
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        step = int(metrics["step"])
        for key, value in metrics.items():
            if key != "step" and isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _prepare_stats(
    metadata: BridgeMetadata,
    train_episodes: list[Any],
    output: Path,
    config: dict[str, Any],
    rank: int,
) -> QuantileStats:
    cache_path = output / "data_cache" / "normalization.json"
    if rank == 0:
        compute_quantile_stats(
            metadata,
            train_episodes,
            cache_path,
            epsilon=float(config["data"]["normalization_epsilon"]),
        )
    distributed.barrier()
    return QuantileStats.load(cache_path)


def train(config: dict[str, Any], *, resume: str | None = None) -> None:
    try:
        import deepspeed
    except ImportError as error:
        raise RuntimeError("DeepSpeed 0.17.6 is required for distributed training") from error

    deepspeed.init_distributed()
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    expected_world_size = int(config["train"]["gpu_count"])
    if world_size != expected_world_size:
        raise RuntimeError(f"Expected world_size={expected_world_size}, got {world_size}")
    torch.cuda.set_device(local_rank)
    seed_everything(int(config["train"]["seed"]), rank)

    output = Path(config["paths"]["output"]).expanduser().resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output / "run_config.yaml")
    distributed.barrier()

    metadata = BridgeMetadata(config["paths"]["dataset"], config["data"])
    train_episodes, validation_episodes = metadata.split()
    fingerprint = metadata.fingerprint()
    if rank == 0:
        _atomic_json(output / "data_fingerprint.json", fingerprint)
    stats = _prepare_stats(metadata, train_episodes, output, config, rank)

    policy = Qwen3VLGrootPolicy.from_local_qwen(
        model_path=config["paths"]["model"],
        stats=stats,
        config=config,
    )
    policy.set_lora_trainable(False)
    optimizer, scheduler = build_optimizer_and_scheduler(policy, config)
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=policy,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=build_deepspeed_config(config, world_size),
    )

    best_validation_mae = math.inf
    global_step = 0
    if resume:
        client_state = load_training_checkpoint(
            engine,
            output,
            resume,
            config=config,
            data_fingerprint=fingerprint,
        )
        global_step = int(client_state["global_step"])
        stored_best = client_state.get("best_validation_mae")
        if stored_best is not None:
            best_validation_mae = float(stored_best)
    policy.set_lora_trainable(global_step >= int(config["train"]["lora_freeze_steps"]))

    train_loader = _make_loader(
        metadata,
        train_episodes,
        train=True,
        rank=rank,
        world_size=world_size,
        config=config,
    )
    validation_loader = _make_loader(
        metadata,
        validation_episodes,
        train=False,
        rank=rank,
        world_size=world_size,
        config=config,
    )
    train_iterator: Iterator[dict[str, Any]] = iter(train_loader)
    logger = RankZeroLogger(output, enabled=rank == 0)
    train_config = config["train"]
    flow_config = config["model"]["flow"]
    maximum_steps = int(train_config["max_steps"])
    previous_global_step = global_step

    try:
        engine.train()
        while global_step < maximum_steps:
            if (
                not any(parameter.requires_grad for parameter in policy.lora_parameters())
                and global_step >= int(train_config["lora_freeze_steps"])
            ):
                policy.set_lora_trainable(True)

            batch = next(train_iterator)
            loss = engine(
                images=batch["images"],
                state=batch["state"],
                actions=batch["actions"],
                action_mask=batch["action_mask"],
                instructions=batch["instructions"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {rank} encountered non-finite loss at step {global_step}: {loss}"
                )
            engine.backward(loss)
            engine.step()
            global_step = int(engine.global_steps)
            if global_step == previous_global_step:
                continue
            previous_global_step = global_step

            if global_step % int(train_config["log_every_steps"]) == 0:
                mean_loss = _reduce_scalar(loss)
                if rank == 0:
                    logger.log(
                        {
                            "step": global_step,
                            "train/loss": mean_loss,
                            "train/head_lr": float(optimizer.param_groups[0]["lr"]),
                            "train/lora_lr": float(optimizer.param_groups[1]["lr"]),
                            "train/lora_enabled": int(
                                global_step >= int(train_config["lora_freeze_steps"])
                            ),
                        }
                    )

            validation_mae: float | None = None
            improved = False
            if global_step % int(train_config["eval_every_steps"]) == 0:
                validation_mae = evaluate(
                    engine,
                    validation_loader,
                    maximum_batches=int(train_config["validation_batches"]),
                    denoising_steps=int(flow_config["denoising_steps"]),
                    seed=int(train_config["seed"]),
                )
                improved = validation_mae < best_validation_mae
                if improved:
                    best_validation_mae = validation_mae
                if rank == 0:
                    logger.log({"step": global_step, "validation/action_mae": validation_mae})

            should_save = global_step % int(train_config["save_every_steps"]) == 0
            if should_save:
                save_training_checkpoint(
                    engine,
                    output,
                    global_step=global_step,
                    config=config,
                    data_fingerprint=fingerprint,
                    best_validation_mae=best_validation_mae,
                )
                export_compact_checkpoint(
                    engine,
                    output / "inference",
                    config=config,
                    model_path=config["paths"]["model"],
                    global_step=global_step,
                    validation_mae=validation_mae,
                )
                distributed.barrier()
                if rank == 0:
                    retain_recent_checkpoints(
                        output, int(train_config["keep_last_checkpoints"])
                    )
                distributed.barrier()

            if improved:
                save_training_checkpoint(
                    engine,
                    output,
                    global_step=global_step,
                    config=config,
                    data_fingerprint=fingerprint,
                    best_validation_mae=best_validation_mae,
                    tag="best",
                    update_latest=False,
                )
                export_compact_checkpoint(
                    engine,
                    output / "best",
                    config=config,
                    model_path=config["paths"]["model"],
                    global_step=global_step,
                    validation_mae=best_validation_mae,
                )
                if rank == 0:
                    _atomic_json(
                        output / "best_checkpoint.json",
                        {"step": global_step, "validation_action_mae": best_validation_mae},
                    )
                distributed.barrier()

        if global_step % int(train_config["save_every_steps"]) != 0:
            save_training_checkpoint(
                engine,
                output,
                global_step=global_step,
                config=config,
                data_fingerprint=fingerprint,
                best_validation_mae=best_validation_mae,
            )
            export_compact_checkpoint(
                engine,
                output / "inference",
                config=config,
                model_path=config["paths"]["model"],
                global_step=global_step,
                validation_mae=None,
            )
    finally:
        logger.close()
