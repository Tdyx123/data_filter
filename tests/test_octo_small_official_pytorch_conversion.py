import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _allow_synthetic_t5_resources_in_checkpoint_fixtures(monkeypatch):
    import octo_small_official_pytorch.checkpoint as checkpoint

    monkeypatch.setattr(
        checkpoint,
        "OFFICIAL_T5_CONTENT_SHA256",
        {
            "text_encoder/tokenizer.json": hashlib.sha256(b"{}").hexdigest(),
            "text_encoder/spiece.model": hashlib.sha256(b"fixture").hexdigest(),
        },
    )


def test_conversion_cli_defaults_to_independent_official_artifact():
    from octo_small_official_pytorch.convert_checkpoint import build_parser

    arguments = build_parser().parse_args([])

    assert arguments.source == "/data/dwb/models/octo-small"
    assert arguments.output == "/data/dwb/models/octo-small-pytorch-official"
    assert arguments.step == 270000
    assert arguments.t5_source == "/data/dwb/models/t5-base"


def test_conversion_mapping_copies_full_official_head_without_proprio():
    from octo_small_official_pytorch.convert_checkpoint import conversion_mapping

    mapping = conversion_mapping()
    targets = [target for target, _source, _transform in mapping]

    assert "action_head.reverse_input.weight" in targets
    assert "action_head.reverse_output.weight" in targets
    assert "action_head.reverse_output.bias" in targets
    assert not any("proprio" in target for target in targets)
    assert len(targets) == len(set(targets))


@pytest.mark.filterwarnings("ignore:builtin type SwigPyPacked has no __module__ attribute")
@pytest.mark.filterwarnings("ignore:builtin type SwigPyObject has no __module__ attribute")
@pytest.mark.filterwarnings("ignore:builtin type swigvarlink has no __module__ attribute")
def test_encoder_construction_cannot_mutate_serialized_t5_config():
    from transformers import T5Config, T5EncoderModel

    from octo_small_official_pytorch.convert_checkpoint import (
        _build_t5_encoder_and_config_document,
    )

    config = T5Config(
        vocab_size=32,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_heads=2,
    )
    assert config.is_encoder_decoder is True

    encoder, serialized = _build_t5_encoder_and_config_document(
        config,
        encoder_factory=T5EncoderModel,
    )

    assert isinstance(encoder, T5EncoderModel)
    assert config.is_encoder_decoder is True
    assert serialized["is_encoder_decoder"] is True
    assert "_name_or_path" not in serialized
    assert "transformers_version" not in serialized


def test_official_t5_resources_are_pinned_to_released_content_hashes():
    from octo_small_official_pytorch.checkpoint import OFFICIAL_T5_CONTENT_SHA256

    # The autouse fixture substitutes synthetic hashes only for checkpoint fixtures;
    # production constants are independently asserted in the policy test module.
    assert set(OFFICIAL_T5_CONTENT_SHA256) == {
        "text_encoder/tokenizer.json",
        "text_encoder/spiece.model",
    }


def test_official_manifest_requires_every_source_tensor_to_be_mapped(tmp_path):
    from octo_small_official_pytorch.checkpoint import (
        OfficialCheckpointError,
        validate_official_checkpoint,
    )

    root = _write_checkpoint_fixture(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["skipped_source_tensors"] = ["heads_action/random"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(OfficialCheckpointError, match="skipped_source_tensors"):
        validate_official_checkpoint(root)


def test_official_checkpoint_is_self_contained_and_reports_hashes(tmp_path):
    from octo_small_official_pytorch.checkpoint import validate_official_checkpoint

    root = _write_checkpoint_fixture(tmp_path)

    report = validate_official_checkpoint(root)

    assert report.path == root.resolve()
    assert report.weights_path == root.resolve() / "model.safetensors"
    assert report.statistics_path == root.resolve() / "dataset_statistics.json"
    assert report.manifest["format"] == "octo-small-official-pytorch-v1"
    assert len(report.weights_sha256) == 64
    assert len(report.statistics_sha256) == 64


def test_official_checkpoint_rejects_incompatible_model_contract(tmp_path):
    from octo_small_official_pytorch.checkpoint import (
        OfficialCheckpointError,
        validate_official_checkpoint,
    )

    root = _write_checkpoint_fixture(tmp_path)
    config_path = root / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["action_horizon"] = 8
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(OfficialCheckpointError, match="model contract"):
        validate_official_checkpoint(root)


def test_official_checkpoint_rejects_safetensor_manifest_shape_mismatch(tmp_path):
    from octo_small_official_pytorch.checkpoint import (
        OfficialCheckpointError,
        validate_official_checkpoint,
    )

    root = _write_checkpoint_fixture(tmp_path)
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = next(iter(manifest["tensor_shapes"]))
    manifest["tensor_shapes"][first] = [2]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(OfficialCheckpointError, match="tensor_shapes"):
        validate_official_checkpoint(root)


def test_official_checkpoint_rejects_non_released_bridge_statistics(tmp_path):
    from octo_small_official_pytorch.checkpoint import (
        OfficialCheckpointError,
        validate_official_checkpoint,
    )

    root = _write_checkpoint_fixture(tmp_path)
    statistics_path = root / "dataset_statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics["bridge_dataset"]["action"]["std"][0] = 1.0
    statistics_path.write_text(json.dumps(statistics), encoding="utf-8")
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_statistics_sha256"] = hashlib.sha256(statistics_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(OfficialCheckpointError, match="released Octo-small"):
        validate_official_checkpoint(root)


def test_official_checkpoint_hashes_every_runtime_text_artifact(tmp_path):
    from octo_small_official_pytorch.checkpoint import (
        OfficialCheckpointError,
        validate_official_checkpoint,
    )

    root = _write_checkpoint_fixture(tmp_path)
    tokenizer = root / "text_encoder" / "tokenizer.json"
    tokenizer.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(OfficialCheckpointError, match="text_encoder/tokenizer.json hash"):
        validate_official_checkpoint(root)


def test_atomic_publish_restores_previous_checkpoint_when_swap_fails(tmp_path, monkeypatch):
    from octo_small_official_pytorch.convert_checkpoint import _publish_staged_checkpoint

    output = tmp_path / "official"
    output.mkdir()
    (output / "old").write_text("kept", encoding="utf-8")
    staging = tmp_path / ".official.staging"
    staging.mkdir()
    (staging / "new").write_text("replacement", encoding="utf-8")
    original_replace = Path.replace

    def fail_staging_swap(path, target):
        if path == staging:
            raise OSError("injected staging swap failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staging_swap)

    with pytest.raises(OSError, match="injected staging"):
        _publish_staged_checkpoint(staging, output)

    assert (output / "old").read_text(encoding="utf-8") == "kept"


def _write_checkpoint_fixture(tmp_path: Path) -> Path:
    import torch
    from safetensors.torch import save_file

    from octo_small_official_pytorch.convert_checkpoint import (
        _t5_mapping,
        conversion_mapping,
    )
    from octo_small_official_pytorch.checkpoint import (
        OFFICIAL_BRIDGE_ACTION_MASK,
        OFFICIAL_BRIDGE_ACTION_MEAN,
        OFFICIAL_BRIDGE_ACTION_STD,
    )

    root = tmp_path / "octo-small-pytorch-official"
    root.mkdir()
    mapping = [*conversion_mapping(), *_t5_mapping()]
    serialized_targets = {
        target for target, _source, _transform in mapping if target != "text_encoder.shared.weight"
    }
    tensors = {target: torch.zeros(1, dtype=torch.float32) for target in serialized_targets}
    save_file(tensors, root / "model.safetensors")
    (root / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "bridge_dataset": {
                    "action": {
                        "mean": list(OFFICIAL_BRIDGE_ACTION_MEAN),
                        "std": list(OFFICIAL_BRIDGE_ACTION_STD),
                        "mask": list(OFFICIAL_BRIDGE_ACTION_MASK),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "model_config.json").write_text(
        json.dumps(
            {
                "history_horizon": 2,
                "action_horizon": 4,
                "action_dim": 7,
                "use_proprio": False,
                "diffusion_steps": 20,
            }
        ),
        encoding="utf-8",
    )
    text = root / "text_encoder"
    text.mkdir()
    (text / "config.json").write_text(
        json.dumps(
            {
                "model_type": "t5",
                "d_model": 768,
                "d_kv": 64,
                "d_ff": 3072,
                "num_layers": 12,
                "num_heads": 12,
                "relative_attention_num_buckets": 32,
                "vocab_size": 32128,
                "is_encoder_decoder": True,
                "dropout_rate": 0.1,
                "layer_norm_epsilon": 1e-6,
                "dense_act_fn": "relu",
                "is_gated_act": False,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    (text / "tokenizer.json").write_text("{}", encoding="utf-8")
    (text / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "T5Tokenizer",
                "model_max_length": 16,
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "extra_ids": 100,
            }
        ),
        encoding="utf-8",
    )
    (text / "spiece.model").write_bytes(b"fixture")
    text_hashes = {
        f"text_encoder/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in text.iterdir()
    }
    (root / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "format": "octo-small-official-pytorch-v1",
                "source_path": "/models/octo-small",
                "source_step": 270000,
                "source_octo_commit": "653c54acde686fde619855f2eac0dd6edad7116b",
                "source_checkpoint_sha256": "a" * 64,
                "weights_sha256": hashlib.sha256(
                    (root / "model.safetensors").read_bytes()
                ).hexdigest(),
                "dataset_statistics_sha256": hashlib.sha256(
                    (root / "dataset_statistics.json").read_bytes()
                ).hexdigest(),
                "text_artifact_sha256": text_hashes,
                "copied_tensor_count": 367,
                "copied_tensors": [
                    {"target": target, "source": source, "transform": transform}
                    for target, source, transform in mapping
                ],
                "tensor_shapes": {target: [1] for target in serialized_targets},
                "intentionally_initialized": [],
                "skipped_source_tensors": [],
                "unexpected_source_tensors": [],
            }
        ),
        encoding="utf-8",
    )
    return root
