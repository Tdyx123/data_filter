#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy and direct Qwen context paths on one fixed LIBERO sample."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu-id", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.02)
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    arguments = parse_args()
    if arguments.gpu_id < 0:
        raise SystemExit("--gpu-id must be non-negative")
    if arguments.rtol < 0 or arguments.atol < 0:
        raise SystemExit("tolerances must be non-negative")
    output_json = arguments.output_json.expanduser().resolve()
    if output_json.exists():
        raise SystemExit(f"parity output must be new: {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu_id)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import torch

    from qwen3_vl_groot.config import apply_overrides, load_config, resolved_paths
    from qwen3_vl_groot.data import bridge_collate
    from qwen3_vl_groot.libero_data import QwenLiberoFrameDataset, resolve_libero_sources
    from qwen3_vl_groot.modeling import Qwen3VLGrootPolicy
    from qwen3_vl_groot.parity import compare_named_tensors, compare_tensors
    from qwen3_vl_groot.preflight import _identity_stats

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    config = apply_overrides(
        load_config(arguments.config),
        {"output_dir": str(output_json.parent / "unused_training_output")},
    )
    paths = resolved_paths(config)
    config["paths"]["model"] = str(paths["model"])
    config["paths"]["lerobot"] = str(paths["lerobot"])
    sources = resolve_libero_sources(config, paths)
    dataset = QwenLiberoFrameDataset(
        paths["target_dataset"],
        dataset_name=str(config["data"]["target_dataset"]),
        action_horizon=int(config["data"]["action_horizon"]),
        train=True,
        seed=int(config["train"]["seed"]),
        episode_cache_size=int(config["data"]["episode_cache_size"]),
        config=config["data"],
        frame_indices=sources.target_selection.frame_indices,
    )
    sample = dataset[0]
    batch = bridge_collate([sample])
    device = torch.device("cuda", 0)
    policy = Qwen3VLGrootPolicy.from_local_qwen(
        model_path=paths["model"],
        stats=_identity_stats(config),
        config=config,
    )
    policy.to(device=device, dtype=torch.bfloat16)
    policy.train()
    policy.set_lora_trainable(True)

    def run_path(mode: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        policy.zero_grad(set_to_none=True)
        policy.context_forward = mode
        torch.manual_seed(arguments.seed)
        torch.cuda.manual_seed_all(arguments.seed)
        context, context_mask = policy.encode_context(
            batch["images"],
            batch["instructions"],
        )
        loss = policy.flow_loss_from_context(
            context=context,
            context_attention_mask=context_mask,
            state=batch["state"],
            actions=batch["actions"],
            action_mask=batch["action_mask"],
        )
        loss.backward()
        torch.cuda.synchronize(device)
        gradients = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in policy.named_parameters()
            if parameter.grad is not None
            and (name.startswith("action_head.") or "lora_" in name)
        }
        return context.detach().cpu().clone(), loss.detach().cpu().clone(), gradients

    legacy_context, legacy_loss, legacy_gradients = run_path("causal_lm")
    direct_context, direct_loss, direct_gradients = run_path("backbone")
    context_result = compare_tensors(
        legacy_context,
        direct_context,
        rtol=arguments.rtol,
        atol=arguments.atol,
    )
    loss_result = compare_tensors(
        legacy_loss,
        direct_loss,
        rtol=arguments.rtol,
        atol=arguments.atol,
    )
    legacy_head = {
        name: value
        for name, value in legacy_gradients.items()
        if name.startswith("action_head.")
    }
    direct_head = {
        name: value
        for name, value in direct_gradients.items()
        if name.startswith("action_head.")
    }
    legacy_lora = {
        name: value for name, value in legacy_gradients.items() if "lora_" in name
    }
    direct_lora = {
        name: value for name, value in direct_gradients.items() if "lora_" in name
    }
    head_result = compare_named_tensors(
        legacy_head,
        direct_head,
        rtol=arguments.rtol,
        atol=arguments.atol,
    )
    lora_result = compare_named_tensors(
        legacy_lora,
        direct_lora,
        rtol=arguments.rtol,
        atol=arguments.atol,
    )
    passed = all(
        result["close"]
        for result in (context_result, loss_result, head_result, lora_result)
    ) and bool(legacy_head) and bool(legacy_lora)
    report = {
        "passed": passed,
        "gpu_id": arguments.gpu_id,
        "seed": arguments.seed,
        "rtol": arguments.rtol,
        "atol": arguments.atol,
        "sample": {
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "instruction": str(sample["instruction"]),
        },
        "context": context_result,
        "loss": loss_result,
        "action_head_gradients": head_result,
        "lora_gradients": lora_result,
    }
    _atomic_json(output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
