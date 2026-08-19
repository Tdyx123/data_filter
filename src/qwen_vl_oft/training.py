from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch import distributed
from torch.autograd.profiler import record_function

from qwen_vl_common.data import BridgeMetadata
from qwen_vl_common.schedules import LoraUpdateSchedule
from qwen_vl_common.training import (
    PerformanceWindow,
    RankZeroLogger,
    _atomic_json,
    _initial_training_metrics,
    _lora_step_metrics,
    _make_loader,
    _prepare_stats,
    _reduce_scalar,
    _should_save_checkpoint,
    _should_save_final_checkpoint,
    build_deepspeed_config,
    build_optimizer_and_scheduler,
    seed_everything,
)

from .checkpointing import (
    CompactCheckpoint,
    inspect_compact_checkpoint,
    load_compact_weights,
    save_compact_checkpoint,
)
from .config import save_resolved_config
from .modeling import QwenVLOFTPolicy
from .preflight import _compile_policy


@torch.no_grad()
def evaluate(
    engine: Any,
    loader: Iterable[dict[str, Any]],
    *,
    maximum_batches: int,
) -> float:
    engine.eval()
    absolute_error = torch.zeros((), dtype=torch.float64, device=engine.device)
    element_count = torch.zeros((), dtype=torch.float64, device=engine.device)
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        prediction = engine.module.predict_actions(
            batch["images"],
            batch["state"],
            batch["instructions"],
        )
        target = batch["actions"].to(engine.device)
        mask = batch["action_mask"].to(engine.device).unsqueeze(-1)
        absolute_error += ((prediction - target).abs() * mask).double().sum()
        element_count += mask.double().sum() * target.shape[-1]
    if distributed.is_initialized():
        distributed.all_reduce(absolute_error, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(element_count, op=distributed.ReduceOp.SUM)
    if element_count.item() == 0:
        raise RuntimeError("Validation loader produced no action elements")
    engine.train()
    return float((absolute_error / element_count).cpu())


def _runtime_metadata(
    config: dict[str, Any],
    *,
    world_size: int,
    warm_start: CompactCheckpoint | None = None,
) -> dict[str, Any]:
    train = config["train"]
    model = config["model"]
    result: dict[str, Any] = {
        "strategy": "starvla-compatible-causal-query-oft",
        "attention": {"requested": model["attn_implementation"], "resolved": "sdpa"},
        "action_query": {
            "token": model["action_token"],
            "horizon": int(config["data"]["action_horizon"]),
            "state_bins": int(model["state_bins"]),
        },
        "torch_compile": dict(model["torch_compile"]),
        "effective_batch_size": (
            int(train["micro_batch_size"])
            * int(train["gradient_accumulation_steps"])
            * world_size
        ),
    }
    if warm_start is not None:
        result["warm_start"] = {
            "checkpoint": str(warm_start.path),
            "initial_step": warm_start.global_step,
            "optimizer_state_restored": False,
        }
    return result


def _validate_warm_start_output(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"Warm-start output must be a directory: {output}")
    launcher_artifacts = {"run_config.yaml", "preflight.json"}
    unexpected = sorted(
        path.name
        for path in output.iterdir()
        if path.name not in launcher_artifacts or not path.is_file()
    )
    if unexpected:
        raise ValueError(
            "Warm-start output contains unexpected existing entries: "
            f"{', '.join(unexpected)}"
        )


def _build_training_policy(
    config: dict[str, Any],
    *,
    stats: Any,
    warm_start: CompactCheckpoint | None,
) -> QwenVLOFTPolicy:
    policy = QwenVLOFTPolicy.from_local_qwen(
        model_path=config["paths"]["model"],
        stats=stats,
        config=config,
    )
    if warm_start is not None:
        load_compact_weights(policy, warm_start.path)
    _compile_policy(policy, config)
    return policy


def _initialize_training_engine(
    deepspeed: Any,
    policy: QwenVLOFTPolicy,
    config: dict[str, Any],
    *,
    world_size: int,
    warm_start: CompactCheckpoint | None,
) -> tuple[Any, torch.optim.Optimizer, Any, int]:
    initial_step = warm_start.global_step if warm_start is not None else 0
    optimizer, scheduler = build_optimizer_and_scheduler(
        policy,
        config,
        initial_step=initial_step,
    )
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=policy,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=build_deepspeed_config(config, world_size),
    )
    policy.set_lora_trainable(False)
    engine.global_steps = initial_step
    return engine, optimizer, scheduler, initial_step


def train(
    config: dict[str, Any],
    *,
    warm_start_checkpoint: str | Path | None = None,
) -> None:
    try:
        import deepspeed
    except ImportError as error:
        raise RuntimeError("DeepSpeed is required for distributed OFT training") from error

    deepspeed.init_distributed()
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", rank))
    expected_world_size = int(config["train"]["gpu_count"])
    if world_size != expected_world_size:
        raise RuntimeError(f"Expected world_size={expected_world_size}, got {world_size}")
    torch.cuda.set_device(local_rank)
    seed_everything(int(config["train"]["seed"]), rank)

    output = Path(config["paths"]["output"]).expanduser().resolve()
    warm_start = (
        inspect_compact_checkpoint(warm_start_checkpoint, config=config)
        if warm_start_checkpoint is not None
        else None
    )
    if rank == 0:
        if warm_start is not None:
            _validate_warm_start_output(output)
        output.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output / "run_config.yaml")
        _atomic_json(
            output / "runtime.json",
            _runtime_metadata(config, world_size=world_size, warm_start=warm_start),
        )
    distributed.barrier()

    metadata = BridgeMetadata(config["paths"]["dataset"], config["data"])
    train_episodes, validation_episodes = metadata.split()
    if rank == 0:
        _atomic_json(output / "data_fingerprint.json", metadata.fingerprint())
    stats = _prepare_stats(
        metadata,
        train_episodes,
        output,
        config,
        rank,
        warm_start=warm_start,
    )
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
    policy = _build_training_policy(config, stats=stats, warm_start=warm_start)
    engine, optimizer, _scheduler, global_step = _initialize_training_engine(
        deepspeed,
        policy,
        config,
        world_size=world_size,
        warm_start=warm_start,
    )

    best_validation_mae = math.inf
    train_iterator: Iterator[dict[str, Any]] = iter(train_loader)
    logger = RankZeroLogger(output, enabled=rank == 0)
    train_config = config["train"]
    maximum_steps = int(train_config["max_steps"])
    schedule = LoraUpdateSchedule.from_train_config(train_config)
    previous_global_step = global_step
    last_checkpoint_step = -1
    validation_mae: float | None = None
    try:
        if rank == 0:
            logger.log(_initial_training_metrics(train_config, warm_start=warm_start))
        engine.train()
        logger.status("Starting first Qwen-VL OFT training batch.")
        performance = PerformanceWindow(
            effective_batch_size=(
                int(train_config["micro_batch_size"])
                * int(train_config["gradient_accumulation_steps"])
                * world_size
            ),
            start_step=global_step,
            started_at=time.perf_counter(),
        )
        while global_step < maximum_steps:
            upcoming_step = global_step + 1
            lora_enabled = schedule.is_active(upcoming_step)
            if lora_enabled != any(
                parameter.requires_grad for parameter in policy.lora_parameters()
            ):
                policy.set_lora_trainable(lora_enabled)
            data_started = time.perf_counter()
            batch = next(train_iterator)
            performance.add_data_wait(time.perf_counter() - data_started)
            lora_lr_used = float(optimizer.param_groups[1]["lr"])
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
            with record_function("backward"):
                engine.backward(loss)
            with record_function("optimizer"):
                engine.step()
            global_step = int(engine.global_steps)
            if global_step == previous_global_step:
                continue
            previous_global_step = global_step

            if global_step % int(train_config["log_every_steps"]) == 0:
                torch.cuda.synchronize(local_rank)
                metrics = performance.metrics(
                    optimizer_step=global_step,
                    now=time.perf_counter(),
                )
                mean_loss = _reduce_scalar(loss)
                if rank == 0:
                    logger.log(
                        {
                            "step": global_step,
                            "train/loss": mean_loss,
                            "train/head_lr": float(optimizer.param_groups[0]["lr"]),
                            **_lora_step_metrics(
                                schedule,
                                optimizer_step=global_step,
                                learning_rate=lora_lr_used,
                            ),
                            **metrics,
                        }
                    )
                performance.reset(start_step=global_step, started_at=time.perf_counter())

            validation_mae = None
            improved = False
            non_training_started: float | None = None
            if bool(train_config.get("validation_enabled", True)) and global_step % int(
                train_config["eval_every_steps"]
            ) == 0:
                torch.cuda.synchronize(local_rank)
                non_training_started = time.perf_counter()
                validation_mae = evaluate(
                    engine,
                    validation_loader,
                    maximum_batches=int(train_config["validation_batches"]),
                )
                improved = validation_mae < best_validation_mae
                if improved:
                    best_validation_mae = validation_mae
                if rank == 0:
                    logger.log(
                        {"step": global_step, "validation/action_mae": validation_mae}
                    )

            if _should_save_checkpoint(
                step=global_step,
                save_every=int(train_config["save_every_steps"]),
                improved=improved,
            ):
                if non_training_started is None:
                    torch.cuda.synchronize(local_rank)
                    non_training_started = time.perf_counter()
                save_compact_checkpoint(
                    engine,
                    output,
                    config=config,
                    model_path=config["paths"]["model"],
                    global_step=global_step,
                    validation_mae=validation_mae,
                    is_best=improved,
                )
                last_checkpoint_step = global_step
                distributed.barrier()
            if non_training_started is not None:
                torch.cuda.synchronize(local_rank)
                performance.exclude_elapsed(time.perf_counter() - non_training_started)

        if _should_save_final_checkpoint(
            step=global_step,
            last_checkpoint_step=last_checkpoint_step,
        ):
            save_compact_checkpoint(
                engine,
                output,
                config=config,
                model_path=config["paths"]["model"],
                global_step=global_step,
                validation_mae=validation_mae,
            )
            distributed.barrier()
    finally:
        logger.close()


__all__ = ["build_deepspeed_config", "evaluate", "train"]
