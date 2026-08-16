import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


def _write_model_artifact(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "Qwen3VL-GR00T-Bridge-RT-1"
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "steps_20000_pytorch_model.pt"
    checkpoint.touch()
    base_model = tmp_path / "Qwen3-VL-4B-Instruct"
    base_model.mkdir()
    (base_model / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "config.yaml").write_text(
        """
framework:
  name: QwenGR00T
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
  action_model:
    action_model_type: DiT-B
    hidden_size: 1024
    add_pos_embed: true
    max_seq_len: 1024
    action_dim: 7
    state_dim: 7
    action_horizon: 16
    num_inference_timesteps: 4
    num_timestep_buckets: 1000
    num_target_vision_tokens: 32
    diffusion_model_cfg:
      cross_attention_dim: 2048
      dropout: 0.2
      final_dropout: true
      interleave_self_attention: true
      norm_type: ada_norm
      num_layers: 16
      output_dim: 1024
      positional_embeddings: null
datasets:
  vla_data:
    CoT_prompt: "Your task is {instruction}. Locate the objects."
    obs: [image_0]
    image_size: [224, 224]
""".lstrip(),
        encoding="utf-8",
    )
    statistics = {
        "oxe_bridge": {
            "action": {
                "q01": [-1, -2, -3, -4, -5, -6, 0],
                "q99": [1, 2, 3, 4, 5, 6, 1],
                "mask": [True, True, True, True, True, True, False],
            }
        }
    }
    (model_dir / "dataset_statistics.json").write_text(
        json.dumps(statistics), encoding="utf-8"
    )
    return model_dir, base_model


def test_model_spec_overrides_base_vlm_in_memory_without_mutating_artifact(tmp_path):
    from starvla_bridge.config import load_model_spec

    model_dir, base_model = _write_model_artifact(tmp_path)
    original_config = (model_dir / "config.yaml").read_text(encoding="utf-8")

    spec = load_model_spec(model_dir, base_model=base_model)

    assert spec.model_dir == model_dir.resolve()
    assert spec.base_model == base_model.resolve()
    assert spec.checkpoint_path.name == "steps_20000_pytorch_model.pt"
    assert spec.action_dim == 7
    assert spec.action_horizon == 16
    assert spec.num_inference_timesteps == 4
    assert spec.image_size == (224, 224)
    assert spec.cot_prompt == "Your task is {instruction}. Locate the objects."
    assert spec.config["framework"]["qwenvl"]["base_vlm"] == str(base_model.resolve())
    assert (model_dir / "config.yaml").read_text(encoding="utf-8") == original_config


def test_oxe_bridge_action_statistics_only_unnormalize_masked_dimensions(tmp_path):
    from starvla_bridge.config import load_model_spec

    model_dir, base_model = _write_model_artifact(tmp_path)
    statistics = load_model_spec(model_dir, base_model=base_model).action_statistics
    normalized = np.asarray([[[-1.0, 0.0, 1.0, -0.5, 0.5, 0.25, 0.25]]])

    actual = statistics.denormalize(normalized)

    np.testing.assert_allclose(
        actual,
        [[[-1.0, 0.0, 3.0, -2.0, 2.5, 1.5, 0.25]]],
        rtol=0,
        atol=1.0e-6,
    )


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 3)


def test_strict_checkpoint_loader_assigns_mmap_weights_to_meta_model(tmp_path):
    from starvla_bridge.checkpoint import load_strict_checkpoint

    source = _TinyPolicy()
    with torch.no_grad():
        source.projection.weight.copy_(torch.arange(6).reshape(3, 2))
        source.projection.bias.copy_(torch.asarray([7.0, 8.0, 9.0]))
    path = tmp_path / "model.pt"
    torch.save(source.state_dict(), path)
    with torch.device("meta"):
        target = _TinyPolicy()

    report = load_strict_checkpoint(target, path, expected_tensor_count=2)

    assert report.tensor_count == 2
    assert report.parameter_bytes == 36
    assert target.projection.weight.device.type == "cpu"
    torch.testing.assert_close(target.projection.weight, source.projection.weight)
    torch.testing.assert_close(target.projection.bias, source.projection.bias)


def test_strict_checkpoint_loader_rejects_missing_or_unexpected_tensors(tmp_path):
    from starvla_bridge.checkpoint import StarVLACheckpointError, load_strict_checkpoint

    path = tmp_path / "broken.pt"
    torch.save({"unexpected": torch.ones(1)}, path)
    with torch.device("meta"):
        target = _TinyPolicy()

    with pytest.raises(StarVLACheckpointError, match="strict checkpoint load failed"):
        load_strict_checkpoint(target, path, expected_tensor_count=1)


def test_strict_checkpoint_loader_rejects_non_bf16_release_weights(tmp_path):
    from starvla_bridge.checkpoint import StarVLACheckpointError, load_strict_checkpoint

    path = tmp_path / "float32.pt"
    torch.save(_TinyPolicy().state_dict(), path)
    with torch.device("meta"):
        target = _TinyPolicy()

    with pytest.raises(StarVLACheckpointError, match="dtype"):
        load_strict_checkpoint(
            target,
            path,
            expected_tensor_count=2,
            expected_dtype=torch.bfloat16,
        )


def test_qwen_messages_apply_checkpoint_cot_prompt_to_one_rgb_image():
    from starvla_bridge.modeling import build_qwen_messages

    image = np.zeros((224, 224, 3), dtype=np.uint8)

    messages = build_qwen_messages(
        image,
        "Put Spoon on Towel",
        cot_prompt="Your task is {instruction}. Locate the objects.",
    )

    assert len(messages) == 1
    assert messages[0][0]["role"] == "user"
    content = messages[0][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["image"] is image
    assert content[1] == {
        "type": "text",
        "text": "Your task is Put Spoon on Towel. Locate the objects.",
    }


@pytest.mark.real_data
def test_meta_policy_exactly_matches_released_checkpoint_keys_and_shapes():
    from starvla_bridge.checkpoint import load_strict_checkpoint
    from starvla_bridge.config import load_model_spec
    from starvla_bridge.modeling import build_meta_policy

    model_dir = Path("/data/dwb/models/Qwen3VL-GR00T-Bridge-RT-1")
    base_model = Path("/data/dwb/models/Qwen3-VL-4B-Instruct")
    if not model_dir.is_dir() or not base_model.is_dir():
        pytest.skip("released local StarVLA artifact is unavailable")
    spec = load_model_spec(model_dir, base_model=base_model)

    policy = build_meta_policy(spec)
    checkpoint = torch.load(
        spec.checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    report = load_strict_checkpoint(
        policy,
        spec.checkpoint_path,
        expected_tensor_count=962,
        expected_dtype=torch.bfloat16,
    )
    actual = policy.state_dict()
    assert report.tensor_count == 962
    assert report.parameter_bytes == 9_976_489_486
    assert report.dtypes == ("torch.bfloat16",)
    assert len(actual) == 962
    assert {tensor.dtype for tensor in checkpoint.values()} == {torch.bfloat16}
    assert sum(
        tensor.numel() * tensor.element_size() for tensor in checkpoint.values()
    ) == 9_976_489_486
    assert set(actual) == set(checkpoint)
    assert {
        name: tuple(tensor.shape) for name, tensor in actual.items()
    } == {
        name: tuple(tensor.shape) for name, tensor in checkpoint.items()
    }
