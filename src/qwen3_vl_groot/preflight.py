from __future__ import annotations

import gc
import json
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from libero_lerobot.sampling import ACTION_WINDOW_POLICY, sample_counts_per_batch

from .config import resolved_paths
from .data import (
    BridgeEpisodeDataset,
    BridgeMetadata,
    bridge_collate,
    validate_episode,
)
from .normalization import QuantileStats


class PreflightError(RuntimeError):
    """Raised when the target host cannot run the requested training configuration."""


MAX_RESERVED_MEMORY_GIB = 22.0


def _build_memory_probe_batch(
    dataset: Any,
    *,
    micro_batch_size: int,
) -> dict[str, Any]:
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        if len(dataset) < micro_batch_size:
            raise PreflightError(
                f"Memory probe needs {micro_batch_size} distinct samples, "
                f"but the dataset has only {len(dataset)}"
            )
        samples = [dataset[index] for index in range(micro_batch_size)]
    else:
        samples = list(islice(iter(dataset), micro_batch_size))
        if len(samples) != micro_batch_size:
            raise PreflightError(
                f"Memory probe requested {micro_batch_size} samples, got {len(samples)}"
            )
    identities = [
        (
            sample.get("dataset_name"),
            int(sample["episode_index"]),
            int(sample["frame_index"]),
        )
        for sample in samples
    ]
    if len(set(identities)) != micro_batch_size:
        raise PreflightError("Memory probe samples must be distinct")
    return bridge_collate(samples)


def _memory_probe_result(
    *,
    loss: float,
    peak_allocated: int,
    peak_reserved: int,
    total: int,
    micro_batch_size: int,
) -> dict[str, Any]:
    limit_bytes = int(MAX_RESERVED_MEMORY_GIB * 2**30)
    if peak_reserved > limit_bytes:
        raise PreflightError(
            "Memory probe reserved "
            f"{peak_reserved / 2**30:.3f} GiB, above the "
            f"{MAX_RESERVED_MEMORY_GIB:.1f} GiB candidate limit"
        )
    if peak_reserved >= total:
        raise PreflightError("Memory probe reached or exceeded physical GPU memory")
    return {
        "loss": loss,
        "micro_batch_size": micro_batch_size,
        "lora_active": True,
        "peak_allocated_gib": round(peak_allocated / 2**30, 3),
        "peak_reserved_gib": round(peak_reserved / 2**30, 3),
        "reserved_limit_gib": MAX_RESERVED_MEMORY_GIB,
        "reserved_headroom_gib": round(
            (limit_bytes - peak_reserved) / 2**30,
            3,
        ),
        "total_gib": round(total / 2**30, 3),
    }


def _inspect_qwen_config(model_path: Path) -> dict[str, Any]:
    from .modeling import inspect_qwen_config

    return inspect_qwen_config(model_path)


def _libero_sample_report(
    dataset: Any,
    *,
    decode_samples: bool,
) -> list[dict[str, Any]]:
    positions = sorted({0, len(dataset) // 2, len(dataset) - 1})
    if not decode_samples:
        return [{"selected_position": position} for position in positions]
    result = []
    for position in positions:
        sample = dataset[position]
        image = sample["image"]
        image_shape = (
            list(image.shape)
            if hasattr(image, "shape")
            else [int(image.height), int(image.width), len(image.getbands())]
        )
        result.append(
            {
                "selected_position": position,
                "image_shape": image_shape,
                "state_shape": list(sample["state"].shape),
                "action_shape": list(sample["actions"].shape),
                "valid_action_steps": int(np.asarray(sample["action_mask"]).sum()),
                "instruction": sample["instruction"],
                "episode_index": sample["episode_index"],
                "frame_index": sample["frame_index"],
            }
        )
    return result


def _validate_libero_paths_and_data(
    config: dict[str, Any],
    *,
    decode_samples: bool,
) -> dict[str, Any]:
    from .libero_data import QwenLiberoFrameDataset, resolve_libero_sources

    paths = resolved_paths(config)
    sources = resolve_libero_sources(config, paths)
    common = {
        "action_horizon": int(config["data"]["action_horizon"]),
        "train": False,
        "seed": int(config["train"]["seed"]),
        "episode_cache_size": 1,
        "config": config["data"],
        "transform": (
            None if decode_samples else lambda image, _rng: image
        ),
    }
    target_dataset = QwenLiberoFrameDataset(
        paths["target_dataset"],
        dataset_name=str(config["data"]["target_dataset"]),
        frame_indices=sources.target_selection.frame_indices,
        **common,
    )
    target_report = {
        "dataset_path": str(paths["target_dataset"]),
        "selection": sources.target_selection.as_manifest(),
        "sampled_frames": _libero_sample_report(
            target_dataset,
            decode_samples=decode_samples,
        ),
    }
    prior_report = None
    if sources.training_mode == "mixed":
        prior_dataset = QwenLiberoFrameDataset(
            paths["prior_dataset"],
            dataset_name=str(config["data"]["prior_dataset"]),
            frame_indices=(
                None
                if sources.prior_selection is None
                else sources.prior_selection.frame_indices
            ),
            **common,
        )
        if sources.prior_selection is None:
            selection_report = {
                "enabled": True,
                "mode": "full_dataset",
                "episodes": len(prior_dataset.metadata.episodes),
                "frames": len(prior_dataset),
                "boundary_policy": ACTION_WINDOW_POLICY,
                "metadata_sha256": prior_dataset.metadata.metadata_sha256(),
            }
        else:
            selection_report = sources.prior_selection.as_manifest()
        prior_report = {
            "dataset_path": str(paths["prior_dataset"]),
            "selection": selection_report,
            "sampled_frames": _libero_sample_report(
                prior_dataset,
                decode_samples=decode_samples,
            ),
        }
    global_micro_batch = int(config["train"]["gpu_count"]) * int(
        config["train"]["micro_batch_size"]
    )
    return {
        "dataset_type": "libero",
        "action_window_policy": ACTION_WINDOW_POLICY,
        "model_path": str(paths["model"]),
        "lerobot_path": str(paths["lerobot"]),
        "qwen_text_layers": int(config["model"]["text_layers"]),
        "target": target_report,
        "prior": prior_report,
        "sample_weights": list(sources.sample_weights),
        "global_micro_batch_source_counts": list(
            sample_counts_per_batch(
                sources.sample_weights,
                global_micro_batch,
                scope="global micro-batch",
            )
        ),
        "normalization_dataset_path": str(sources.normalization_root),
        "offline_validation": False,
    }


def validate_paths_and_data(config: dict[str, Any], *, decode_samples: bool = True) -> dict[str, Any]:
    model_path = Path(config["paths"]["model"]).expanduser().resolve()
    _inspect_qwen_config(model_path)
    if config["data"].get("dataset_type", "bridge") == "libero":
        return _validate_libero_paths_and_data(
            config,
            decode_samples=decode_samples,
        )
    dataset_path = Path(config["paths"]["dataset"]).expanduser().resolve()
    metadata = BridgeMetadata(dataset_path, config["data"])
    episodes = metadata.episodes
    sample_positions = sorted({0, len(episodes) // 2, len(episodes) - 1})
    samples = [
        validate_episode(metadata, episodes[position], decode=decode_samples)
        for position in sample_positions
    ]
    train_episodes, validation_episodes = metadata.split()
    return {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "qwen_text_layers": 36,
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "sampled_episodes": samples,
        "data_fingerprint": metadata.fingerprint(),
    }


def validate_gpu_host(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise PreflightError("PyTorch 2.9.0 is not installed in this environment") from error
    required = int(config["train"]["gpu_count"])
    available = torch.cuda.device_count()
    if available < required:
        raise PreflightError(f"Requested {required} GPUs, but torch sees {available}")
    devices = []
    physical_ids = config["train"].get("gpu_ids")
    for index in range(required):
        properties = torch.cuda.get_device_properties(index)
        if properties.major < 8:
            raise PreflightError(f"GPU {index} does not provide the requested BF16 support")
        devices.append(
            {
                "index": index,
                "configured_physical_id": (
                    int(physical_ids[index]) if physical_ids is not None else None
                ),
                "name": properties.name,
                "memory_gib": round(properties.total_memory / 2**30, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "devices": devices}


def _identity_stats(config: dict[str, Any]) -> QuantileStats:
    state_dim = int(config["data"]["state_dim"])
    action_dim = int(config["data"]["action_dim"])
    return QuantileStats(
        state_q01=np.full(state_dim, -1.0, dtype=np.float32),
        state_q99=np.full(state_dim, 1.0, dtype=np.float32),
        action_q01=np.full(action_dim, -1.0, dtype=np.float32),
        action_q99=np.full(action_dim, 1.0, dtype=np.float32),
    )


def probe_single_gpu_memory(config: dict[str, Any]) -> dict[str, Any]:
    """Run one real micro-batch forward/backward before distributed launch."""
    import torch

    from .modeling import Qwen3VLGrootPolicy, compile_policy_modules

    if int(config["train"]["deepspeed_stage"]) == 3:
        return {
            "skipped": True,
            "reason": "ZeRO-3 shards parameters; the standalone single-GPU probe is not representative",
        }

    device = torch.device("cuda", 0)
    if config["data"].get("dataset_type", "bridge") == "libero":
        from .libero_data import QwenLiberoFrameDataset, resolve_libero_sources

        paths = resolved_paths(config)
        sources = resolve_libero_sources(config, paths)
        implementation = QwenLiberoFrameDataset(
            paths["target_dataset"],
            dataset_name=str(config["data"]["target_dataset"]),
            action_horizon=int(config["data"]["action_horizon"]),
            train=True,
            seed=int(config["train"]["seed"]),
            episode_cache_size=int(config["data"]["episode_cache_size"]),
            config=config["data"],
            frame_indices=sources.target_selection.frame_indices,
        )
    else:
        metadata = BridgeMetadata(config["paths"]["dataset"], config["data"])
        train_episodes, _ = metadata.split()
        implementation = BridgeEpisodeDataset(
            metadata,
            train_episodes,
            train=True,
            rank=0,
            world_size=1,
            seed=int(config["train"]["seed"]),
            action_horizon=int(config["data"]["action_horizon"]),
            video_cache_size=int(config["data"]["video_cache_episodes"]),
        )
    micro_batch_size = int(config["train"]["micro_batch_size"])
    batch = _build_memory_probe_batch(
        implementation,
        micro_batch_size=micro_batch_size,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    policy = None
    try:
        policy = Qwen3VLGrootPolicy.from_local_qwen(
            model_path=config["paths"]["model"],
            stats=_identity_stats(config),
            config=config,
        )
        policy.to(device=device, dtype=torch.bfloat16)
        compile_policy_modules(policy, config["model"])
        policy.train()
        policy.set_lora_trainable(True)
        loss = policy(
            images=batch["images"],
            state=batch["state"],
            actions=batch["actions"],
            action_mask=batch["action_mask"],
            instructions=batch["instructions"],
        )
        if not torch.isfinite(loss):
            raise PreflightError(f"Memory probe produced non-finite loss: {loss.item()}")
        loss.backward()
        if not any(parameter.grad is not None for parameter in policy.lora_parameters()):
            raise PreflightError("Memory probe did not produce LoRA gradients")
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        return _memory_probe_result(
            loss=float(loss.detach().cpu()),
            peak_allocated=peak_allocated,
            peak_reserved=peak_reserved,
            total=total,
            micro_batch_size=micro_batch_size,
        )
    except torch.OutOfMemoryError as error:
        raise PreflightError(
            "The full 36-layer Qwen + 12-layer DiT cannot complete "
            f"micro-batch {micro_batch_size} "
            "on a single 24GB GPU with ZeRO-2. Retry with --deepspeed-stage 3 "
            "(CPU optimizer offload). The model definition and layer count were not changed."
        ) from error
    finally:
        del policy
        gc.collect()
        torch.cuda.empty_cache()


def run_preflight(
    config: dict[str, Any],
    *,
    memory_probe: bool = True,
) -> dict[str, Any]:
    result = {
        "data": validate_paths_and_data(config, decode_samples=True),
        "gpu": validate_gpu_host(config),
    }
    if memory_probe:
        result["memory_probe"] = probe_single_gpu_memory(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
