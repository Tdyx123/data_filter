import numpy as np
import pytest


@pytest.mark.parametrize(
    "adapter_class",
    (
        pytest.param(
            "octo_small_bridge.remote_policy:OctoRemotePolicy", id="octo-remote"
        ),
        pytest.param(
            "octo_small_bridge.simpler_evaluation:OctoBridgeSimplerPolicy",
            id="octo-local",
        ),
        pytest.param(
            "qwen3_vl_groot.remote_policy:QwenRemotePolicy", id="qwen-remote"
        ),
        pytest.param(
            "qwen3_vl_groot.simpler_evaluation:QwenPolicyAdapter", id="qwen-local"
        ),
        pytest.param(
            "qwen_vl_oft.simpler_evaluation:OFTPolicyAdapter", id="qwen-oft"
        ),
    ),
)
def test_non_starvla_adapters_keep_first_action_selection(adapter_class: str) -> None:
    module_name, class_name = adapter_class.split(":")
    module = __import__(module_name, fromlist=[class_name])
    adapter = object.__new__(getattr(module, class_name))
    actions = np.zeros((1, 8, 7), dtype=np.float32)
    actions[0, 0] = np.arange(7, dtype=np.float32)
    actions[0, 1] = 99.0

    assert adapter.begin_episode("instruction") is None
    np.testing.assert_array_equal(adapter.select_action(actions), actions[0, 0])
