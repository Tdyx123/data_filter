from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import BridgeEpisodeDataset, BridgeMetadata, validate_episode
from .modeling import Qwen3VLGrootPolicy, inspect_qwen_config
from .normalization import QuantileStats


class PreflightError(RuntimeError):
    """Raised when the target host cannot run the requested training configuration."""


def validate_paths_and_data(config: dict[str, Any], *, decode_samples: bool = True) -> dict[str, Any]:
    model_path = Path(config["paths"]["model"]).expanduser().resolve()
    dataset_path = Path(config["paths"]["dataset"]).expanduser().resolve()
    inspect_qwen_config(model_path)
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

    if int(config["train"]["deepspeed_stage"]) == 3:
        return {
            "skipped": True,
            "reason": "ZeRO-3 shards parameters; the standalone single-GPU probe is not representative",
        }

    device = torch.device("cuda", 0)
    metadata = BridgeMetadata(config["paths"]["dataset"], config["data"])
    train_episodes, _ = metadata.split()
    implementation = BridgeEpisodeDataset(
        metadata,
        [train_episodes[0]],
        train=True,
        rank=0,
        world_size=1,
        seed=int(config["train"]["seed"]),
        action_horizon=int(config["data"]["action_horizon"]),
        video_cache_size=1,
    )
    sample = next(iter(implementation))
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
        policy.train()
        loss = policy(
            images=[sample["image"]],
            state=torch.as_tensor(sample["state"]).unsqueeze(0),
            actions=torch.as_tensor(sample["actions"]).unsqueeze(0),
            action_mask=torch.as_tensor(sample["action_mask"]).unsqueeze(0),
            instructions=[sample["instruction"]],
        )
        if not torch.isfinite(loss):
            raise PreflightError(f"Memory probe produced non-finite loss: {loss.item()}")
        loss.backward()
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        peak = max(peak_allocated, peak_reserved)
        total = torch.cuda.get_device_properties(device).total_memory
        if peak >= total:
            raise PreflightError("Memory probe reached or exceeded physical GPU memory")
        return {
            "loss": float(loss.detach().cpu()),
            "peak_allocated_gib": round(peak_allocated / 2**30, 3),
            "peak_reserved_gib": round(peak_reserved / 2**30, 3),
            "total_gib": round(total / 2**30, 3),
        }
    except torch.OutOfMemoryError as error:
        raise PreflightError(
            "The full 36-layer Qwen + 12-layer DiT cannot complete micro-batch 1 "
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
