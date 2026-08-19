from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from qwen_vl_common.normalization import QuantileStats

from .checkpointing import CHECKPOINT_FORMAT, load_compact_weights
from .modeling import QwenVLOFTPolicy


class BridgePolicy:
    def __init__(self, policy: QwenVLOFTPolicy):
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
        if manifest.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported inference checkpoint format: {manifest.get('format')}"
            )
        config = manifest["config"]
        base_model = str(model_path or manifest["base_model"])
        stats = QuantileStats.load(checkpoint / "normalization.json")
        policy = QwenVLOFTPolicy.from_local_qwen(
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
    ) -> torch.Tensor:
        return self.policy.predict_actions(image, state, instruction)
