import json
from pathlib import Path

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_oft_checkpoint(tmp_path: Path, *, checkpoint_format: str | None = None):
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config": {"num_hidden_layers": 36, "hidden_size": 2560},
            }
        ),
        encoding="utf-8",
    )
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "qwenvl_oft_bridge_4x4090.yaml").read_text(
            encoding="utf-8"
        )
    )
    # Simpler evaluation remains an explicit legacy 8-step protocol fixture.
    config["data"]["action_horizon"] = 8
    config["paths"]["model"] = str(base_model)
    config["paths"]["dataset"] = str(tmp_path / "dataset")
    config["paths"]["output"] = str(tmp_path / "run")
    checkpoint = tmp_path / "run" / "checkpoints" / "step-00019000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"oft-weights")
    (checkpoint / "normalization.json").write_text(
        json.dumps(
            {
                "state_q01": [0.0] * 8,
                "state_q99": [1.0] * 8,
                "action_q01": [0.0] * 7,
                "action_q99": [1.0] * 7,
                "epsilon": 1.0e-6,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "policy_config.json").write_text(
        json.dumps(
            {
                "format": checkpoint_format or "qwen-vl-oft-bridge-compact-v1",
                "base_model": str(base_model),
                "global_step": 19_000,
                "config": config,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, base_model


def test_resolve_oft_checkpoint_accepts_only_a_complete_matching_step(tmp_path):
    from qwen_vl_oft.simpler_evaluation import resolve_oft_checkpoint

    checkpoint, base_model = _write_oft_checkpoint(tmp_path)

    resolved = resolve_oft_checkpoint(checkpoint)

    assert resolved.requested_path == checkpoint.resolve()
    assert resolved.base_model_path == base_model.resolve()
    assert resolved.global_step == 19_000
    assert resolved.config["data"]["action_horizon"] == 8
    assert len(resolved.weights_sha256) == 64
    with pytest.raises(Exception, match="step-XXXXXXXX"):
        resolve_oft_checkpoint(checkpoint.parent.parent)


def test_resolve_oft_checkpoint_rejects_groot_format_before_model_loading(tmp_path):
    from qwen_vl_oft.simpler_evaluation import resolve_oft_checkpoint

    checkpoint, _ = _write_oft_checkpoint(
        tmp_path,
        checkpoint_format="qwen3-vl-groot-bridge-compact-v1",
    )

    with pytest.raises(Exception, match="Unsupported Qwen-VL OFT checkpoint format"):
        resolve_oft_checkpoint(checkpoint)


def test_resolve_oft_checkpoint_rejects_non_integer_global_step(tmp_path):
    from qwen_vl_oft.simpler_evaluation import resolve_oft_checkpoint

    checkpoint, _ = _write_oft_checkpoint(tmp_path)
    policy_config_path = checkpoint / "policy_config.json"
    manifest = json.loads(policy_config_path.read_text(encoding="utf-8"))
    manifest["global_step"] = "19000"
    policy_config_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(Exception, match="non-negative integer global_step"):
        resolve_oft_checkpoint(checkpoint)


class _DeterministicPolicy:
    def __init__(self):
        self.call = None

    def predict_actions(self, image, state, instruction):
        self.call = (image, state, instruction)
        return np.zeros((1, 8, 7), dtype=np.float32)


def test_oft_adapter_center_crops_like_validation_and_ignores_rng():
    from qwen_vl_oft.simpler_evaluation import OFTPolicyAdapter

    image = np.full((256, 256, 3), [20, 30, 40], dtype=np.uint8)
    image[:13] = [255, 0, 0]
    image[-13:] = [255, 0, 0]
    image[:, :13] = [255, 0, 0]
    image[:, -13:] = [255, 0, 0]
    policy = _DeterministicPolicy()
    adapter = OFTPolicyAdapter(
        policy=policy,
        crop_size=230,
        output_size=256,
    )

    prepared = adapter.prepare_observation(
        image,
        np.arange(8, dtype=np.float32),
        "Pick up the spoon.",
    )
    actions = adapter.predict_actions(prepared, generator=adapter.make_generator(123))

    assert prepared["image"].shape == (256, 256, 3)
    np.testing.assert_array_equal(
        prepared["image"],
        np.full((256, 256, 3), [20, 30, 40], dtype=np.uint8),
    )
    assert actions.shape == (1, 8, 7)
    assert policy.call[2] == "Pick up the spoon."
    assert adapter.protocol_metadata() == {
        "inference_strategy": "deterministic-causal-query-oft",
        "native_action_chunk_size": 8,
    }


def test_oft_adapter_moves_policy_tensor_output_to_numpy():
    import torch

    from qwen_vl_oft.simpler_evaluation import OFTPolicyAdapter

    class TensorPolicy:
        def predict_actions(self, image, state, instruction):
            del image, state, instruction
            return torch.zeros((1, 8, 7), dtype=torch.bfloat16)

    adapter = OFTPolicyAdapter(policy=TensorPolicy(), crop_size=230, output_size=256)

    actions = adapter.predict_actions(
        {
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "proprio": np.zeros(8, dtype=np.float32),
            "instruction": "Pick up the spoon.",
        },
        generator=0,
    )

    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.float32
    assert actions.shape == (1, 8, 7)
