import pytest

torch = pytest.importorskip("torch")

from qwen3_vl_groot.parity import compare_named_tensors, compare_tensors  # noqa: E402


def test_compare_tensors_reports_bfloat16_tolerant_differences():
    reference = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    candidate = torch.tensor([1.01, 1.99], dtype=torch.bfloat16)

    result = compare_tensors(reference, candidate, rtol=0.03, atol=0.02)

    assert result["close"] is True
    assert result["elements"] == 2
    assert result["max_abs_difference"] > 0


def test_compare_named_tensors_identifies_parameter_mismatch():
    reference = {
        "action_head.weight": torch.ones(2),
        "backbone.lora_A.weight": torch.ones(2),
    }
    candidate = {
        "action_head.weight": torch.ones(2),
        "backbone.lora_A.weight": torch.zeros(2),
    }

    result = compare_named_tensors(reference, candidate, rtol=0.01, atol=0.01)

    assert result["close"] is False
    assert result["parameters_compared"] == 2
    assert result["mismatched_parameters"] == ["backbone.lora_A.weight"]
