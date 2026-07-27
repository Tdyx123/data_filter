from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from qwen3_vl_groot.config import load_config  # noqa: E402
from qwen3_vl_groot.training import build_optimizer_and_scheduler  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = nn.Linear(4, 4)
        self.lora_a = nn.Parameter(torch.randn(4, 2))
        self.lora_b = nn.Parameter(torch.randn(2, 4))

    def action_head_parameters(self):
        return list(self.action_head.parameters())

    def lora_parameters(self):
        return [self.lora_a, self.lora_b]


def test_optimizer_groups_are_nonempty_and_trainable_before_zero_init():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    policy = TinyPolicy()
    optimizer, scheduler = build_optimizer_and_scheduler(policy, config)
    assert len(optimizer.param_groups) == 2
    assert all(
        any(parameter.requires_grad for parameter in group["params"])
        for group in optimizer.param_groups
    )
    assert optimizer.param_groups[0]["group_name"] == "action_head"
    assert optimizer.param_groups[1]["group_name"] == "qwen_lora"
    # LambdaLR initializes the delayed LoRA group at zero learning rate.
    assert optimizer.param_groups[1]["lr"] == 0.0
    assert scheduler is not None

