import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from qwen3_vl_groot.modeling import (  # noqa: E402
    assert_full_lora_coverage,
    assert_qwen_freeze_contract,
    lora_coverage,
)


class FakeLoraLinear(nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = FakeLoraLinear()
        self.k_proj = FakeLoraLinear()
        self.v_proj = FakeLoraLinear()
        self.o_proj = FakeLoraLinear()


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()


class FakeBackbone(nn.Module):
    def __init__(self, layers=36):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([FakeLayer() for _ in range(layers)])


def test_all_36_layers_have_all_four_lora_targets():
    model = FakeBackbone()
    coverage = lora_coverage(model)
    assert set(coverage) == set(range(36))
    assert all(value == {"q_proj", "k_proj", "v_proj", "o_proj"} for value in coverage.values())
    assert_full_lora_coverage(model)
    assert_qwen_freeze_contract(model)


def test_missing_projection_is_rejected():
    model = FakeBackbone()
    model.language_model.layers[17].self_attn.o_proj = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="layer 17"):
        assert_full_lora_coverage(model)

