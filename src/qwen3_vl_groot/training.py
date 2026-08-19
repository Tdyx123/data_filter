from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as distributed
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader

from .checkpointing import (
    CompactCheckpoint,
    inspect_compact_checkpoint,
    load_compact_weights,
    save_compact_checkpoint,
)
from .config import resolved_paths, save_resolved_config
from .data import (
    BridgeEpisodeDataset,
    BridgeMetadata,
    as_torch_iterable,
    bridge_collate,
    compute_quantile_stats,
)
from .modeling import (
    ACTION_HEAD_CONTEXT_BUCKETS,
    Qwen3VLGrootPolicy,
    compile_policy_modules,
    resolve_compile_targets,
    select_attention_implementation,
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
        """Remove non-training work from the wall-clock interval."""
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
            "performance/samples_per_second": (
                self.effective_batch_size / step_seconds
            ),
            "performance/data_wait_fraction": self.data_wait_seconds / elapsed,
        }

    def reset(self, *, start_step: int, started_at: float) -> None:
        self.start_step = start_step
        self.started_at = started_at
        self.data_wait_seconds = 0.0


def _runtime_versions() -> dict[str, str | None]:
    distributions = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "deepspeed": "deepspeed",
        "flash_attn": "flash-attn",
    }
    result: dict[str, str | None] = {}
    for key, distribution_name in distributions.items():
        try:
            result[key] = package_version(distribution_name)
        except (PackageNotFoundError, ModuleNotFoundError):
            result[key] = None
    return result


def _runtime_metadata(
    config: dict[str, Any],
    *,
    world_size: int,
    warm_start: CompactCheckpoint | None = None,
) -> dict[str, Any]:
    model_config = config["model"]
    train_config = config["train"]
    compile_backbone, compile_action_head = resolve_compile_targets(model_config)
    attention_requested = str(model_config["attn_implementation"])
    compile_config = model_config.get("torch_compile", {})
    result = {
        "packages": _runtime_versions(),
        "attention": {
            "requested": attention_requested,
            "resolved": select_attention_implementation(attention_requested),
        },
        "context_forward": str(
            model_config.get("context_forward", "causal_lm")
        ),
        "torch_compile": {
            "backbone_enabled": compile_backbone,
            "action_head_enabled": compile_action_head,
            "backend": compile_config.get("backend"),
            "mode": compile_config.get("mode"),
            "dynamic": compile_config.get("dynamic"),
            "fullgraph": compile_config.get("fullgraph"),
            "action_head_context_buckets": (
                list(ACTION_HEAD_CONTEXT_BUCKETS)
                if compile_action_head and compile_config.get("dynamic") is False
                else None
            ),
        },
        "effective_batch_size": (
            int(train_config["micro_batch_size"])
            * int(train_config["gradient_accumulation_steps"])
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
    policy: Qwen3VLGrootPolicy,
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


def _prepare_libero_stats(
    normalization_root: Path,
    output: Path,
    config: dict[str, Any],
    rank: int,
    *,
    warm_start: CompactCheckpoint | None = None,
) -> tuple[QuantileStats, Path]:
    from .libero_data import compute_libero_quantile_stats

    cache_path = output / "data_cache" / "normalization.json"
    if rank == 0:
        if warm_start is None:
            compute_libero_quantile_stats(
                normalization_root,
                cache_path,
                epsilon=float(config["data"]["normalization_epsilon"]),
            )
        else:
            _copy_checkpoint_normalization(warm_start.normalization_path, cache_path)
    distributed.barrier()
    return QuantileStats.load(cache_path), cache_path


def _build_training_policy(
    config: dict[str, Any],
    *,
    stats: QuantileStats,
    warm_start: CompactCheckpoint | None,
) -> Qwen3VLGrootPolicy:
    policy = Qwen3VLGrootPolicy.from_local_qwen(
        model_path=config["paths"]["model"],
        stats=stats,
        config=config,
    )
    if warm_start is not None:
        load_compact_weights(policy, warm_start.path)
    compile_policy_modules(policy, config["model"])
    return policy


def _initialize_training_engine(
    deepspeed: Any,
    policy: Qwen3VLGrootPolicy,
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
            _runtime_metadata(
                config,
                world_size=world_size,
                warm_start=warm_start,
            ),
        )
    distributed.barrier()

    dataset_type = config["data"].get("dataset_type", "bridge")
    validation_loader: DataLoader | None
    if dataset_type == "libero":
        from .libero_data import (
            build_libero_dataset_manifest,
            make_libero_training_data,
            resolve_libero_sources,
        )

        paths = resolved_paths(config)
        sources = resolve_libero_sources(config, paths)
        stats, normalization_path = _prepare_libero_stats(
            sources.normalization_root,
            output,
            config,
            rank,
            warm_start=warm_start,
        )
        if rank == 0:
            _atomic_json(
                output / "dataset_manifest.json",
                build_libero_dataset_manifest(
                    config,
                    paths,
                    sources,
                    normalization_path=normalization_path,
                ),
            )
        training_data = make_libero_training_data(
            config,
            paths,
            rank=rank,
            world_size=world_size,
        )
        train_loader = training_data.dataloader
        validation_loader = None
    else:
        metadata = BridgeMetadata(config["paths"]["dataset"], config["data"])
        train_episodes, validation_episodes = metadata.split()
        fingerprint = metadata.fingerprint()
        if rank == 0:
            _atomic_json(output / "data_fingerprint.json", fingerprint)
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

    policy = _build_training_policy(
        config,
        stats=stats,
        warm_start=warm_start,
    )
    # ZeRO builds one internal bit16 group per optimizer parameter group and
    # filters parameters with requires_grad=False. LoRA must therefore remain
    # trainable until deepspeed.initialize() has partitioned both groups. We
    # freeze it immediately after initialization; its scheduler LR is also zero
    # during the configured LoRA freeze interval.
    engine, optimizer, scheduler, global_step = _initialize_training_engine(
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
    flow_config = config["model"]["flow"]
    validation_enabled = bool(train_config.get("validation_enabled", True))
    if validation_enabled and validation_loader is None:
        raise RuntimeError("validation is enabled but no validation loader was configured")
    maximum_steps = int(train_config["max_steps"])
    lora_update_schedule = LoraUpdateSchedule.from_train_config(train_config)
    previous_global_step = global_step
    last_checkpoint_step = -1
    validation_mae: float | None = None

    try:
        if rank == 0:
            logger.log(
                _initial_training_metrics(
                    train_config,
                    warm_start=warm_start,
                )
            )
        engine.train()
        compile_enabled = any(resolve_compile_targets(config["model"]))
        first_batch_status = "Starting first training batch."
        if compile_enabled:
            first_batch_status = (
                "Starting first training batch; TorchInductor warm-up may take several minutes."
            )
        logger.status(first_batch_status)
        effective_batch_size = (
            int(train_config["micro_batch_size"])
            * int(train_config["gradient_accumulation_steps"])
            * world_size
        )
        torch.cuda.synchronize(local_rank)
        performance = PerformanceWindow(
            effective_batch_size=effective_batch_size,
            start_step=global_step,
            started_at=time.perf_counter(),
        )
        while global_step < maximum_steps:
            upcoming_step = global_step + 1
            lora_enabled_for_update = lora_update_schedule.is_active(upcoming_step)
            lora_is_trainable = any(
                parameter.requires_grad for parameter in policy.lora_parameters()
            )
            if lora_enabled_for_update != lora_is_trainable:
                policy.set_lora_trainable(lora_enabled_for_update)

            data_wait_started = time.perf_counter()
            batch = next(train_iterator)
            performance.add_data_wait(time.perf_counter() - data_wait_started)
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
                performance_metrics = performance.metrics(
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
                                lora_update_schedule,
                                optimizer_step=global_step,
                                learning_rate=lora_lr_used,
                            ),
                            **performance_metrics,
                        }
                    )
                performance.reset(
                    start_step=global_step,
                    started_at=time.perf_counter(),
                )

            validation_mae = None
            improved = False
            non_training_started: float | None = None
            if validation_enabled and global_step % int(
                train_config["eval_every_steps"]
            ) == 0:
                torch.cuda.synchronize(local_rank)
                non_training_started = time.perf_counter()
                assert validation_loader is not None
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

            should_save = _should_save_checkpoint(
                step=global_step,
                save_every=int(train_config["save_every_steps"]),
                improved=improved,
            )
            if should_save:
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
                performance.exclude_elapsed(
                    time.perf_counter() - non_training_started
                )

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
