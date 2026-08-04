from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import inspect_octo_checkpoint, load_lerobot_statistics
from .config import normalized_sample_weights
from .lerobot_v2 import LeRobotV2Metadata


def _statistics_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_manifest(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    target_selection: Any | None = None,
    prior_selection: Any | None = None,
) -> dict[str, Any]:
    target_only = bool(config["data"].get("target_only", False))
    if target_selection is None:
        from .data import resolve_target_task_selection

        target_selection = resolve_target_task_selection(config, paths)
    if not target_only and prior_selection is None and (
        config["data"]["prior_selection"].get("top_percent") is not None
        or config["data"]["prior_selection"].get("prefiltered", False)
    ):
        from .selection import resolve_prior_selection

        prior_selection = resolve_prior_selection(config, paths)
    target_name = config["data"]["target_dataset"]
    prior_name = config["data"]["prior_dataset"]
    normalization_name = target_name if target_only else prior_name
    normalization_root = (
        paths["target_dataset"] if target_only else paths["prior_dataset"]
    )
    statistics = load_lerobot_statistics(normalization_root)
    sample_weights = (
        (1.0,)
        if target_only
        else normalized_sample_weights(config["data"]["sample_weights"])
    )
    target_metadata = LeRobotV2Metadata(paths["target_dataset"])
    conversion_manifest_path = paths["lerobot"] / "conversion_manifest.json"
    target_conversion_manifest_path = (
        paths["lerobot"] / f"{target_name}_conversion_manifest.json"
    )
    datasets = {
        target_name: {
            "sample_weight": sample_weights[0],
            "path": str(target_metadata.root),
            "metadata_sha256": target_metadata.metadata_sha256(),
            "episodes": int(target_metadata.info["total_episodes"]),
            "frames": int(target_metadata.info["total_frames"]),
            "episodes_used": target_selection.episodes,
            "frames_used": target_selection.frames,
            "selection": target_selection.as_manifest(),
        }
    }
    if not target_only:
        prior_metadata = LeRobotV2Metadata(paths["prior_dataset"])
        datasets[prior_name] = {
            "sample_weight": sample_weights[1],
            "path": str(prior_metadata.root),
            "metadata_sha256": prior_metadata.metadata_sha256(),
            "episodes": int(prior_metadata.info["total_episodes"]),
            "frames": int(prior_metadata.info["total_frames"]),
            "frames_used": (
                prior_selection.training_starts
                if prior_selection is not None
                else int(prior_metadata.info["total_frames"])
            ),
            "selection": (
                prior_selection.as_manifest()
                if prior_selection is not None
                else {"enabled": False}
            ),
        }
    statistics_sha256 = _statistics_sha256(paths["statistics"])
    manifest = {
        "training_mode": "target_only" if target_only else "mixed",
        "lerobot_root": str(paths["lerobot"]),
        "datasets": datasets,
        "normalization": {
            "source_dataset": normalization_name,
            "path": str(paths["statistics"]),
            "sha256": statistics_sha256,
            "trajectories": int(statistics["num_trajectories"]),
            "transitions": int(statistics["num_transitions"]),
        },
        "conversion_manifest": (
            str(conversion_manifest_path) if conversion_manifest_path.is_file() else None
        ),
        "target_conversion_manifest": (
            str(target_conversion_manifest_path)
            if target_conversion_manifest_path.is_file()
            else None
        ),
    }
    if not target_only:
        manifest.update(
            {
                "prior_statistics_path": str(paths["statistics"]),
                "prior_statistics_sha256": statistics_sha256,
                "normalization_trajectories": int(statistics["num_trajectories"]),
                "normalization_transitions": int(statistics["num_transitions"]),
            }
        )
    return manifest


def _move_to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _learning_rate_lambda(
    step: int,
    *,
    warmup_steps: int,
    decay_steps: int,
    end_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denominator = max(1, decay_steps - warmup_steps)
    progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end_ratio + (1.0 - end_ratio) * cosine


def _resolve_resume(output: Path, resume: str | None) -> Path | None:
    if resume is None:
        return None
    if resume == "latest":
        latest = output / "checkpoints" / "latest.json"
        if not latest.is_file():
            raise RuntimeError(f"No latest PyTorch checkpoint found at {latest}")
        with latest.open("r", encoding="utf-8") as handle:
            name = str(json.load(handle)["checkpoint"])
        return output / "checkpoints" / name
    return Path(resume).expanduser().resolve()


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.parent / f".{path.name}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _read_best_checkpoint(checkpoints: Path) -> dict[str, Any] | None:
    path = checkpoints / "best.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    try:
        checkpoint = str(value["checkpoint"])
        step = int(value["step"])
        mean_train_loss = float(value["mean_train_loss"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid best checkpoint pointer at {path}") from error
    if checkpoint != f"step-{step:08d}" or not math.isfinite(mean_train_loss):
        raise RuntimeError(f"Invalid best checkpoint pointer at {path}")
    if not (checkpoints / checkpoint).is_dir():
        raise RuntimeError(
            f"Best checkpoint pointer references a missing directory: "
            f"{checkpoints / checkpoint}"
        )
    return {
        "checkpoint": checkpoint,
        "step": step,
        "mean_train_loss": mean_train_loss,
    }


def _retain_latest_and_best_checkpoints(
    checkpoints: Path,
    *,
    latest: str,
    best: str,
) -> None:
    keep = {latest, best}
    for path in checkpoints.iterdir():
        name = path.name
        is_step_checkpoint = (
            path.is_dir()
            and name.startswith("step-")
            and len(name) == len("step-00000000")
            and name.removeprefix("step-").isdigit()
        )
        if is_step_checkpoint and name not in keep:
            shutil.rmtree(path)


def _should_save_checkpoint(*, step: int, max_steps: int, save_every: int) -> bool:
    return step % save_every == 0 or step == max_steps


def _save_checkpoint(
    *,
    output: Path,
    step: int,
    mean_train_loss: float,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    sampler: Any,
    config: dict[str, Any],
    selection_signature: str | None = None,
) -> Path:
    import torch
    from safetensors.torch import save_model

    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    mean_train_loss = float(mean_train_loss)
    if not math.isfinite(mean_train_loss):
        raise RuntimeError(
            f"Cannot save step {step} with non-finite mean training loss: "
            f"{mean_train_loss}"
        )
    previous_best = _read_best_checkpoint(checkpoints)
    name = f"step-{step:08d}"
    target = checkpoints / name
    temporary = checkpoints / f".{name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    unwrapped = model.module if hasattr(model, "module") else model
    save_model(unwrapped, str(temporary / "model.safetensors"))
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "selection_signature": selection_signature,
    }
    torch.save(state, temporary / "training_state.pt")
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)

    improved = (
        previous_best is None
        or mean_train_loss < float(previous_best["mean_train_loss"])
    )
    best = (
        {
            "checkpoint": name,
            "step": step,
            "mean_train_loss": mean_train_loss,
        }
        if improved
        else previous_best
    )
    assert best is not None
    if improved:
        _atomic_write_json(checkpoints / "best.json", best)
    _atomic_write_json(
        checkpoints / "latest.json",
        {"checkpoint": name, "step": step},
    )
    _retain_latest_and_best_checkpoints(
        checkpoints,
        latest=name,
        best=str(best["checkpoint"]),
    )
    return target


def _load_training_state(
    checkpoint: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    sampler: Any,
    device: Any,
    selection_signature: str | None = None,
) -> int:
    import torch
    from safetensors.torch import load_model

    if not checkpoint.is_dir():
        raise RuntimeError(f"Resume checkpoint does not exist: {checkpoint}")
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    saved_signature = state.get("selection_signature")
    if saved_signature != selection_signature:
        raise RuntimeError(
            "Resume checkpoint data selection differs from the current target/prior "
            "selection or sample weights"
        )
    load_model(model, str(checkpoint / "model.safetensors"), strict=True, device=str(device))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sampler.load_state_dict(state["sampler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"])


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def train(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    resume: str | None = None,
) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import tqdm

    from .data import make_training_dataset
    from .modeling import load_pytorch_model

    if not torch.cuda.is_available():
        raise RuntimeError("Octo-small production training requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = int(config["train"]["gpu_count"])
    if world_size != requested:
        raise RuntimeError(
            f"torchrun WORLD_SIZE={world_size}, but train.gpu_count={requested}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    seed = int(config["train"]["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    model, tokenizer = load_pytorch_model(paths["model"], device=device)
    model.train()
    train_data = make_training_dataset(
        config,
        paths,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
    )
    selection_signature = train_data.selection_sha256
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    peak_lr = float(config["train"]["learning_rate"]["peak_value"])
    optimizer = torch.optim.AdamW(
        trainable,
        lr=peak_lr,
        weight_decay=float(config["train"]["weight_decay"]),
    )
    learning_rate = config["train"]["learning_rate"]
    end_ratio = (
        float(learning_rate["end_value"]) / peak_lr if peak_lr > 0 else 0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_lambda(
            step,
            warmup_steps=int(learning_rate["warmup_steps"]),
            decay_steps=int(learning_rate["decay_steps"]),
            end_ratio=end_ratio,
        ),
    )

    output = paths["output"]
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        clean_config = {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        _write_json(output / "finetune_config.json", clean_config)
        _write_json(
            output / "dataset_manifest.json",
            build_dataset_manifest(
                config,
                paths,
                target_selection=train_data.target_selection,
                prior_selection=train_data.prior_selection,
            ),
        )
        checkpoint_report = inspect_octo_checkpoint(paths["model"])
        model_manifest = {
            "base_model": str(paths["model"]),
            "tensor_count": checkpoint_report["tensor_count"],
            "weights_sha256": checkpoint_report["weights_sha256"],
            "conversion_manifest": str(paths["model"] / "conversion_manifest.json"),
            "conversion_manifest_sha256": checkpoint_report[
                "conversion_manifest_sha256"
            ],
        }
        _write_json(output / "model_manifest.json", model_manifest)
    if world_size > 1:
        dist.barrier()

    start_step = 0
    resume_path = _resolve_resume(output, resume)
    if resume_path is not None:
        unwrapped = model.module if hasattr(model, "module") else model
        start_step = _load_training_state(
            resume_path,
            model=unwrapped,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_data.batch_sampler,
            device=device,
            selection_signature=selection_signature,
        )
    data_iterator = iter(train_data.dataloader)
    accumulation_steps = int(config["train"]["gradient_accumulation_steps"])
    max_steps = int(config["train"]["max_steps"])
    log_every = int(config["train"]["log_every_steps"])
    save_every = int(config["train"]["save_every_steps"])
    max_grad_norm = float(config["train"]["max_grad_norm"])
    metrics_path = output / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    checkpoint_loss_total = torch.zeros((), device=device)
    checkpoint_loss_steps = 0
    progress = tqdm(
        range(start_step + 1, max_steps + 1),
        disable=rank != 0,
        dynamic_ncols=True,
    )

    for step in progress:
        loss_total = torch.zeros((), device=device)
        mse_total = torch.zeros((), device=device)
        sampled: Counter[str] = Counter()
        for accumulation in range(accumulation_steps):
            batch = next(data_iterator)
            sampled.update(str(value) for value in batch["dataset_name"])
            batch = _move_to_device(batch, device)
            synchronizes = accumulation + 1 == accumulation_steps
            sync_context = (
                nullcontext()
                if synchronizes or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    outputs = model(batch)
                    loss = outputs["loss"] / accumulation_steps
                loss.backward()
            loss_total += outputs["loss"].detach() / accumulation_steps
            mse_total += outputs["mse"].detach() / accumulation_steps

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        reduced = torch.stack([loss_total, mse_total, gradient_norm.detach().float()])
        if world_size > 1:
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced /= world_size
        checkpoint_loss_total += reduced[0]
        checkpoint_loss_steps += 1
        if rank == 0 and (step == 1 or step % log_every == 0):
            record = {
                "step": step,
                "split": "train",
                "loss": float(reduced[0].cpu()),
                "mse": float(reduced[1].cpu()),
                "grad_norm": float(reduced[2].cpu()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                **{
                    f"source_samples/{name}": count * world_size
                    for name, count in sorted(sampled.items())
                },
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.set_postfix(loss=record["loss"], lr=record["learning_rate"])

        should_save = _should_save_checkpoint(
            step=step,
            max_steps=max_steps,
            save_every=save_every,
        )
        if should_save:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                mean_train_loss = float(
                    (checkpoint_loss_total / checkpoint_loss_steps).cpu()
                )
                _save_checkpoint(
                    output=output,
                    step=step,
                    mean_train_loss=mean_train_loss,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=train_data.batch_sampler,
                    config=config,
                    selection_signature=selection_signature,
                )
            if world_size > 1:
                dist.barrier()
            checkpoint_loss_total.zero_()
            checkpoint_loss_steps = 0

    if world_size > 1:
        dist.destroy_process_group()
