from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .checkpointing import load_compact_weights
from .config import ConfigError, require_bridge_v2_normalization_contract
from .modeling import Qwen3VLGrootPolicy
from .normalization import QuantileStats


class BridgePolicy:
    def __init__(self, policy: Qwen3VLGrootPolicy):
        self.policy = policy

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        model_path: str | Path | None = None,
        device: str | torch.device = "cuda",
    ) -> "BridgePolicy":
        checkpoint = Path(checkpoint_dir).expanduser().resolve()
        with (checkpoint / "policy_config.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != "qwen3-vl-groot-bridge-compact-v1":
            raise ValueError(f"Unsupported inference checkpoint format: {manifest.get('format')}")
        config = manifest["config"]
        try:
            data_config = config["data"]
            if data_config.get("dataset_type", "bridge") == "bridge":
                require_bridge_v2_normalization_contract(data_config)
        except (ConfigError, KeyError, TypeError) as error:
            raise ValueError(f"Unsupported Bridge normalization contract: {error}") from error
        base_model = str(model_path or manifest["base_model"])
        stats = QuantileStats.load(checkpoint / "normalization.json")
        policy = Qwen3VLGrootPolicy.from_local_qwen(
            model_path=base_model,
            stats=stats,
            config=config,
        )
        load_compact_weights(policy, checkpoint)
        target = torch.device(device)
        dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
        policy.to(device=target, dtype=dtype)
        policy.eval()
        return cls(policy)

    @torch.inference_mode()
    def predict_actions(
        self,
        image: Any | Sequence[Any],
        state: np.ndarray | torch.Tensor | Sequence[float],
        instruction: str | Sequence[str],
        denoising_steps: int = 4,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.policy.predict_actions(
            image,
            state,
            instruction,
            denoising_steps=denoising_steps,
            generator=generator,
        )
