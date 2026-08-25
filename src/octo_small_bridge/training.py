"""Official-semantics Octo-small Bridge distributed training."""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from octo_small_official_pytorch.checkpoint import (
    OfficialCheckpointReport,
    sha256_file,
    validate_official_checkpoint,
)

from .checkpoint_contract import build_checkpoint_contract
from .data import BridgeTrainingData, make_training_dataset


def build_dataset_manifest(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    training_data: BridgeTrainingData | Any,
) -> dict[str, Any]:
    """Describe the exact official-semantics Bridge source used by this run."""

    adapter = training_data.dataset.adapter
    summary = adapter.dataset_summary()
    statistics_path = Path(training_data.statistics_path)
    retained_frames = sum(record.length for record in adapter.episodes())
    prior_selection = getattr(training_data, "prior_selection", None)
    selection_manifest = (
        prior_selection.as_manifest()
        if prior_selection is not None
        else {
            "enabled": False,
            "mode": "all_non_empty_episodes",
            "training_starts": retained_frames,
            "action_horizon": 4,
            "boundary_policy": "episode_tail_repeat_last_action",
        }
    )
    return {
        "dataset": config["data"]["dataset_name"],
        "path": str(Path(paths["dataset"]).resolve()),
        "metadata_sha256": adapter.fingerprint(),
        "selection_sha256": training_data.selection_sha256,
        "selection": selection_manifest,
        "source_episodes": summary["source_episodes"],
        "retained_episodes": summary["retained_episodes"],
        "excluded_episodes": summary["excluded_episodes"],
        "excluded_empty_task_episodes": summary[
            "excluded_empty_task_episodes"
        ],
        "retained_frames": retained_frames,
        "empty_task_policy": "exclude",
        "image_observations": list(adapter.image_observation_keys),
        "vector_observations": [],
        "action_key": adapter.action_key,
        "training_contract": {
            "history_horizon": 2,
            "action_horizon": 4,
            "action_dim": 7,
            "use_proprio": False,
            "action_normalization": "bridge_dataset_mean_std",
            "gripper_transform": "trajectory_backward_binarize_minus_one_to_one",
        },
        "statistics": {
            "source_dataset": "bridge_dataset",
            "path": str(statistics_path.resolve()),
            "sha256": sha256_file(statistics_path),
            "recomputed": False,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _load_training_model(
    report: OfficialCheckpointReport,
    *,
    device: Any,
) -> tuple[Any, Any]:
    """Strictly load the official artifact while keeping T5/wrist frozen."""

    try:
        from safetensors.torch import load_model
        from transformers import AutoTokenizer, T5Config, T5EncoderModel

        from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

        text_root = report.path / "text_encoder"
        text_config = T5Config.from_pretrained(text_root, local_files_only=True)
        model = OctoSmallOfficialPolicy(T5EncoderModel(text_config), report.config)
        load_model(
            model,
            str(report.weights_path),
            strict=True,
            device="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(text_root, local_files_only=True)
        model.to(device)
        model.train()
        return model, tokenizer
    except Exception as error:
        raise RuntimeError(
            f"Could not strictly load official Octo-small training artifact: {error}"
        ) from error


def train(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    resume: str | None = None,
) -> None:
    """Run 4-GPU official-semantics Bridge fine-tuning."""

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import tqdm

    from octo_small_libero.training import (
        _capture_rank_runtime_state,
        _initialize_distributed_process_group,
        _learning_rate_lambda,
        _load_training_state,
        _move_to_device,
        _resolve_resume,
        _save_checkpoint,
        _should_save_checkpoint,
    )

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
    _initialize_distributed_process_group(dist, world_size=world_size, device=device)

    seed = int(config["train"]["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    base_report = validate_official_checkpoint(paths["model"])
    model, tokenizer = _load_training_model(base_report, device=device)
    train_data = make_training_dataset(
        config,
        paths,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
    )
    selection_signature = train_data.selection_sha256
    checkpoint_contract = build_checkpoint_contract(config, paths, train_data)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Official Octo-small fine-tuning has no trainable parameters")
    peak_lr = float(config["train"]["learning_rate"]["peak_value"])
    optimizer = torch.optim.AdamW(
        trainable,
        lr=peak_lr,
        weight_decay=float(config["train"]["weight_decay"]),
    )
    learning_rate = config["train"]["learning_rate"]
    end_ratio = float(learning_rate["end_value"]) / peak_lr if peak_lr > 0 else 0.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_lambda(
            step,
            warmup_steps=int(learning_rate["warmup_steps"]),
            max_steps=int(config["train"]["max_steps"]),
            end_ratio=end_ratio,
        ),
    )

    output = paths["output"]
    dataset_manifest = build_dataset_manifest(
        config,
        paths,
        training_data=train_data,
    )
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        clean_config = {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        _write_json(output / "finetune_config.json", clean_config)
        _write_json(output / "dataset_manifest.json", dataset_manifest)
        _write_json(
            output / "model_manifest.json",
            {
                "base_model": str(base_report.path),
                "format": base_report.manifest["format"],
                "source_step": int(base_report.manifest["source_step"]),
                "source_octo_commit": base_report.manifest["source_octo_commit"],
                "weights_sha256": base_report.weights_sha256,
                "statistics_sha256": base_report.statistics_sha256,
                "conversion_manifest": str(
                    base_report.path / "conversion_manifest.json"
                ),
                "conversion_manifest_sha256": sha256_file(
                    base_report.path / "conversion_manifest.json"
                ),
            },
        )
    if world_size > 1:
        dist.barrier()

    start_step = 0
    resume_path = _resolve_resume(output, resume)
    if resume_path is not None:
        checkpoint_contract.validate(resume_path)
        unwrapped = model.module if hasattr(model, "module") else model
        start_step = _load_training_state(
            resume_path,
            model=unwrapped,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_data.batch_sampler,
            device=device,
            selection_signature=selection_signature,
            rank=rank,
            world_size=world_size,
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
            if hasattr(train_data.batch_sampler, "mark_batch_consumed"):
                train_data.batch_sampler.mark_batch_consumed()
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

        if _should_save_checkpoint(
            step=step,
            max_steps=max_steps,
            save_every=save_every,
        ):
            if world_size > 1:
                dist.barrier()
            local_runtime_state = _capture_rank_runtime_state(train_data.batch_sampler)
            if world_size > 1:
                rank_runtime_states: list[dict[str, Any] | None] = [None] * world_size
                dist.all_gather_object(rank_runtime_states, local_runtime_state)
                gathered_runtime_states = [
                    state for state in rank_runtime_states if state is not None
                ]
            else:
                gathered_runtime_states = [local_runtime_state]
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
                    rank_runtime_states=gathered_runtime_states,
                    checkpoint_contract=checkpoint_contract,
                )
            if world_size > 1:
                dist.barrier()
            checkpoint_loss_total.zero_()
            checkpoint_loss_steps = 0

    if world_size > 1:
        dist.destroy_process_group()
