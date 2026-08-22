from pathlib import Path

import pytest
import torch

from qwen_vl_oft.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _ValidationPolicy:
    def predict_actions(self, images, state, instructions):
        return torch.tensor([[[1.0], [3.0], [100.0]]])


class _ValidationEngine:
    device = torch.device("cpu")

    def __init__(self):
        self.module = _ValidationPolicy()
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def test_oft_validation_mae_is_deterministic_and_masks_padded_tail():
    from qwen_vl_oft.training import evaluate

    engine = _ValidationEngine()
    loader = [
        {
            "images": [object()],
            "state": torch.zeros(1, 2),
            "instructions": ["Move"],
            "actions": torch.zeros(1, 3, 1),
            "action_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        }
    ]

    assert evaluate(engine, loader, maximum_batches=1) == pytest.approx(2.0)
    assert engine.training is True


def test_oft_deepspeed_config_keeps_global_batch_64():
    from qwen_vl_oft.training import build_deepspeed_config

    config = load_config(PROJECT_ROOT / "configs" / "qwenvl_oft_bridge_4x4090.yaml")
    deepspeed_config = build_deepspeed_config(config, world_size=4)

    assert deepspeed_config["train_micro_batch_size_per_gpu"] == 1
    assert deepspeed_config["gradient_accumulation_steps"] == 16
    assert deepspeed_config["train_batch_size"] == 64
    assert deepspeed_config["zero_optimization"]["stage"] == 2


def test_oft_runtime_metadata_records_starvla_strategy():
    from qwen_vl_oft.training import _runtime_metadata

    config = load_config(PROJECT_ROOT / "configs" / "qwenvl_oft_bridge_4x4090.yaml")
    metadata = _runtime_metadata(config, world_size=4)

    assert metadata["strategy"] == "starvla-compatible-causal-query-oft"
    assert metadata["effective_batch_size"] == 64
    assert metadata["action_query"] == {"token": "🔍", "horizon": 16, "state_bins": 256}
