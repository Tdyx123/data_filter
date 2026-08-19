from pathlib import Path

import pytest

from qwen_vl_oft.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_oft_memory_probe_description_has_no_groot_components():
    from qwen_vl_oft.preflight import _memory_probe_model_description

    config = load_config(PROJECT_ROOT / "configs" / "qwenvl_oft_bridge_4x4090.yaml")

    assert _memory_probe_model_description(config) == (
        "36-layer Qwen3-VL-4B + StarVLA-compatible MLP OFT head"
    )


def test_oft_memory_probe_enforces_the_4090_reserved_memory_limit():
    from qwen_vl_oft.preflight import PreflightError, _memory_probe_result

    with pytest.raises(PreflightError, match="above the 22.0 GiB candidate limit"):
        _memory_probe_result(
            loss=0.5,
            peak_allocated=21 * 2**30,
            peak_reserved=23 * 2**30,
            total=24 * 2**30,
            micro_batch_size=1,
        )
