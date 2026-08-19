import json

import numpy as np
import pytest
import torch
from torch import nn


class _LoadedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.call = None

    def predict_actions(self, image, state, instruction):
        self.call = (image, state, instruction)
        batch_size = 1 if np.asarray(state).ndim == 1 else len(state)
        return torch.zeros(batch_size, 8, 7)


def _checkpoint(tmp_path, *, checkpoint_format="qwen-vl-oft-bridge-compact-v1"):
    checkpoint = tmp_path / "step-00000001"
    checkpoint.mkdir()
    (checkpoint / "policy_config.json").write_text(
        json.dumps(
            {
                "format": checkpoint_format,
                "base_model": "/tmp/base-qwen",
                "config": {
                    "data": {"state_dim": 8, "action_dim": 7, "action_horizon": 8},
                    "model": {},
                },
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "normalization.json").write_text(
        json.dumps(
            {
                "state_q01": [0.0] * 8,
                "state_q99": [1.0] * 8,
                "action_q01": [0.0] * 7,
                "action_q99": [1.0] * 7,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_bridge_policy_loads_oft_checkpoint_and_preserves_batch_shape(tmp_path, monkeypatch):
    import qwen_vl_oft.inference as inference

    loaded = _LoadedPolicy()
    monkeypatch.setattr(
        inference.QwenVLOFTPolicy,
        "from_local_qwen",
        classmethod(lambda cls, **kwargs: loaded),
    )
    monkeypatch.setattr(inference, "load_compact_weights", lambda policy, checkpoint: None)

    policy = inference.BridgePolicy.from_pretrained(_checkpoint(tmp_path), device="cpu")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    actions = policy.predict_actions(image, np.zeros(8), "Move")

    assert actions.shape == (1, 8, 7)
    assert loaded.call[2] == "Move"
    assert loaded.training is False

    batch_actions = policy.predict_actions(
        [image, image],
        np.zeros((2, 8)),
        ["Move", "Lift"],
    )
    assert batch_actions.shape == (2, 8, 7)


def test_bridge_policy_rejects_groot_checkpoint_format(tmp_path):
    from qwen_vl_oft.inference import BridgePolicy

    checkpoint = _checkpoint(
        tmp_path,
        checkpoint_format="qwen3-vl-groot-bridge-compact-v1",
    )

    with pytest.raises(ValueError, match="Unsupported inference checkpoint format"):
        BridgePolicy.from_pretrained(checkpoint, device="cpu")
