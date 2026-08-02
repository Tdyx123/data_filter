import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from torch import nn  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from qwen3_vl_groot.checkpointing import (  # noqa: E402
    load_compact_weights,
    save_compact_checkpoint,
)
from qwen3_vl_groot.config import load_config  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DummyCompactPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = nn.Linear(3, 2)
        self.backbone = nn.Module()
        self.backbone.register_parameter("lora_A", nn.Parameter(torch.randn(2, 2)))
        self.backbone.register_parameter("base_weight", nn.Parameter(torch.randn(2, 2)))
        self.register_buffer("state_q01", torch.zeros(8))
        self.register_buffer("state_q99", torch.ones(8))
        self.register_buffer("action_q01", torch.zeros(7))
        self.register_buffer("action_q99", torch.ones(7))
        self.normalization_epsilon = 1.0e-6

    def compact_parameter_names(self):
        return [
            name
            for name, _ in self.named_parameters()
            if name.startswith("action_head.") or "lora_" in name
        ]


class DummyEngine:
    def __init__(self, module):
        self.module = module
        self.global_rank = 0


def test_compact_safetensors_round_trip(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    policy = DummyCompactPolicy()
    original = {
        name: value.detach().clone()
        for name, value in policy.named_parameters()
    }
    target = save_compact_checkpoint(
        DummyEngine(policy),
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=7,
        validation_mae=0.25,
    )
    assert target == tmp_path / "checkpoints" / "step-00000007"
    assert (target / "adapter_model.safetensors").is_file()
    assert (target / "normalization.json").is_file()
    assert (target / "policy_config.json").is_file()
    assert set(load_file(str(target / "adapter_model.safetensors"))) == {
        "action_head.bias",
        "action_head.weight",
        "backbone.lora_A",
    }
    assert json.loads(
        (tmp_path / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    ) == {"checkpoint": "step-00000007", "step": 7}

    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(10)
    load_compact_weights(policy, target)
    for name in policy.compact_parameter_names():
        value = dict(policy.named_parameters())[name]
        torch.testing.assert_close(value, original[name])
    torch.testing.assert_close(
        policy.backbone.base_weight,
        original["backbone.base_weight"] + 10,
    )
    np.testing.assert_array_equal(policy.action_q99.cpu().numpy(), np.ones(7))


def test_latest_and_best_markers_reuse_steps_and_retain_only_references(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    engine = DummyEngine(DummyCompactPolicy())
    checkpoints = tmp_path / "checkpoints"

    save_compact_checkpoint(
        engine,
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=1,
        validation_mae=0.5,
        is_best=True,
    )
    save_compact_checkpoint(
        engine,
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=2,
        validation_mae=None,
    )
    assert sorted(path.name for path in checkpoints.glob("step-*")) == [
        "step-00000001",
        "step-00000002",
    ]

    save_compact_checkpoint(
        engine,
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=3,
        validation_mae=None,
    )
    assert sorted(path.name for path in checkpoints.glob("step-*")) == [
        "step-00000001",
        "step-00000003",
    ]
    assert json.loads((checkpoints / "latest.json").read_text(encoding="utf-8")) == {
        "checkpoint": "step-00000003",
        "step": 3,
    }
    assert json.loads((checkpoints / "best.json").read_text(encoding="utf-8")) == {
        "checkpoint": "step-00000001",
        "step": 1,
        "validation_action_mae": 0.5,
    }

    save_compact_checkpoint(
        engine,
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=3,
        validation_mae=0.25,
        is_best=True,
    )
    assert [path.name for path in checkpoints.glob("step-*")] == ["step-00000003"]
    assert json.loads((checkpoints / "best.json").read_text(encoding="utf-8")) == {
        "checkpoint": "step-00000003",
        "step": 3,
        "validation_action_mae": 0.25,
    }
