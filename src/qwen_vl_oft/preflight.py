from __future__ import annotations

import gc
import json
from typing import Any

import numpy as np

from qwen_vl_common.backbone import compile_policy_modules
from qwen_vl_common.contracts import backbone_contract
from qwen_vl_common.data import BridgeEpisodeDataset, BridgeMetadata
from qwen_vl_common.normalization import QuantileStats
from qwen_vl_common.preflight import (
    PreflightError,
    _build_memory_probe_batch,
    _memory_probe_result,
    validate_gpu_host,
    validate_paths_and_data,
)


def _identity_stats(config: dict[str, Any]) -> QuantileStats:
    return QuantileStats(
        state_q01=np.full(int(config["data"]["state_dim"]), -1.0, dtype=np.float32),
        state_q99=np.full(int(config["data"]["state_dim"]), 1.0, dtype=np.float32),
        action_q01=np.full(int(config["data"]["action_dim"]), -1.0, dtype=np.float32),
        action_q99=np.full(int(config["data"]["action_dim"]), 1.0, dtype=np.float32),
    )


def _memory_probe_model_description(config: dict[str, Any]) -> str:
    contract = backbone_contract(config["model"])
    return (
        f"{int(config['model']['text_layers'])}-layer {contract['display_name']} + "
        "StarVLA-compatible MLP OFT head"
    )


def _compile_policy(policy: Any, config: dict[str, Any]) -> None:
    compile_policy_modules(policy, config["model"])


def probe_single_gpu_memory(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from .modeling import QwenVLOFTPolicy

    if int(config["train"]["deepspeed_stage"]) == 3:
        return {
            "skipped": True,
            "reason": "ZeRO-3 shards parameters; the standalone single-GPU probe is not representative",
        }
    device = torch.device("cuda", 0)
    metadata = BridgeMetadata(config["paths"]["dataset"], config["data"])
    train_episodes, _ = metadata.split()
    dataset = BridgeEpisodeDataset(
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
    batch = _build_memory_probe_batch(dataset, micro_batch_size=micro_batch_size)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    policy = None
    try:
        policy = QwenVLOFTPolicy.from_local_qwen(
            model_path=config["paths"]["model"],
            stats=_identity_stats(config),
            config=config,
        )
        policy.to(device=device, dtype=torch.bfloat16)
        _compile_policy(policy, config)
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
        return _memory_probe_result(
            loss=float(loss.detach().cpu()),
            peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device),
            total=torch.cuda.get_device_properties(device).total_memory,
            micro_batch_size=micro_batch_size,
        )
    except torch.OutOfMemoryError as error:
        raise PreflightError(
            f"The full {_memory_probe_model_description(config)} cannot complete "
            f"micro-batch {micro_batch_size} on one GPU with ZeRO-2. "
            "Retry with --deepspeed-stage 3."
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


__all__ = [
    "PreflightError",
    "probe_single_gpu_memory",
    "run_preflight",
    "validate_gpu_host",
    "validate_paths_and_data",
]
