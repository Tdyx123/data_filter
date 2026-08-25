from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import OctoOfficialConfig


OFFICIAL_FORMAT = "octo-small-official-pytorch-v1"
OFFICIAL_FINETUNE_FORMAT = "octo-small-official-pytorch-finetune-v1"
OFFICIAL_OCTO_COMMIT = "653c54acde686fde619855f2eac0dd6edad7116b"
OFFICIAL_SOURCE_STEP = 270000
OFFICIAL_BRIDGE_ACTION_MEAN = (
    0.00021160596224945039,
    0.00012613728176802397,
    -0.00017021960229612887,
    -0.0001506186235928908,
    -0.00023830769350752234,
    0.0002564573078416288,
    0.0,
)
OFFICIAL_BRIDGE_ACTION_STD = (
    0.009637207724153996,
    0.013506603427231312,
    0.012518610805273056,
    0.028067905455827713,
    0.03016904927790165,
    0.07632624357938766,
    1.0,
)
OFFICIAL_BRIDGE_ACTION_MASK = (True, True, True, True, True, True, False)
TEXT_ARTIFACT_PATHS = (
    "text_encoder/config.json",
    "text_encoder/tokenizer_config.json",
    "text_encoder/tokenizer.json",
    "text_encoder/spiece.model",
)
OFFICIAL_T5_CONTENT_SHA256 = {
    "text_encoder/tokenizer.json": "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
    "text_encoder/spiece.model": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
}
T5_BASE_CONFIG_CONTRACT = {
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
T5_TOKENIZER_CONFIG_CONTRACT = {
    "tokenizer_class": "T5Tokenizer",
    "model_max_length": 16,
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "extra_ids": 100,
}


class OfficialCheckpointError(RuntimeError):
    """Raised when the self-contained official-parity artifact is invalid."""


@dataclass(frozen=True)
class OfficialCheckpointReport:
    path: Path
    weights_path: Path
    statistics_path: Path
    manifest: dict[str, Any]
    config: OctoOfficialConfig
    weights_sha256: str
    statistics_sha256: str
    text_artifact_sha256: dict[str, str]
    checkpoint_kind: str = "official_parity"

    def as_dict(self) -> dict[str, Any]:
        value = {
            "path": str(self.path),
            "format": self.manifest["format"],
            "checkpoint_kind": self.checkpoint_kind,
            "weights_path": str(self.weights_path),
            "statistics_path": str(self.statistics_path),
            "weights_sha256": self.weights_sha256,
            "statistics_sha256": self.statistics_sha256,
            "text_artifact_sha256": dict(self.text_artifact_sha256),
            "history_horizon": self.config.history_horizon,
            "native_action_chunk": self.config.action_horizon,
            "action_dim": self.config.action_dim,
            "use_proprio": self.config.use_proprio,
        }
        if self.checkpoint_kind == "official_parity":
            value.update(
                {
                    "source_step": int(self.manifest["source_step"]),
                    "source_octo_commit": self.manifest["source_octo_commit"],
                }
            )
        else:
            value.update(
                {
                    "training_step": int(self.manifest["training_step"]),
                    "source_step": int(self.manifest["base_checkpoint"]["source_step"]),
                    "source_octo_commit": self.manifest["base_checkpoint"][
                        "source_octo_commit"
                    ],
                }
            )
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OfficialCheckpointError(f"Unable to read {description} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise OfficialCheckpointError(f"{description} must contain a JSON object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_model_config(path: Path) -> OctoOfficialConfig:
    config_value = _read_json(path, "model config")
    try:
        config = OctoOfficialConfig.from_dict(config_value)
    except (TypeError, ValueError) as error:
        raise OfficialCheckpointError(f"Invalid model config: {error}") from error
    expected = OctoOfficialConfig().to_dict()
    mismatches = {
        key: {"actual": actual, "expected": expected[key]}
        for key, actual in config.to_dict().items()
        if actual != expected[key]
    }
    if mismatches:
        raise OfficialCheckpointError(
            f"Checkpoint does not satisfy the official model contract: {mismatches}"
        )
    return config


def _inspect_finetuned_tensor_shapes(
    weights_path: Path,
    declared_shapes: Any,
) -> None:
    if not isinstance(declared_shapes, dict) or not declared_shapes:
        raise OfficialCheckpointError("finetune manifest must contain tensor_shapes")
    try:
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            serialized_keys = set(handle.keys())
            if serialized_keys != set(declared_shapes):
                raise OfficialCheckpointError(
                    "safetensors keys do not match finetune manifest tensor_shapes"
                )
            for key in sorted(serialized_keys):
                actual = list(handle.get_slice(key).get_shape())
                if declared_shapes[key] != actual:
                    raise OfficialCheckpointError(
                        f"safetensors shape for {key!r} does not match finetune manifest"
                    )
    except OfficialCheckpointError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise OfficialCheckpointError(f"Could not inspect model.safetensors: {error}") from error


def _validate_statistics(statistics: dict[str, Any], path: Path) -> None:
    action = statistics.get("bridge_dataset", {}).get("action")
    if not isinstance(action, dict):
        raise OfficialCheckpointError(f"{path} must contain bridge_dataset.action statistics")
    for name in ("mean", "std"):
        values = action.get(name)
        if not isinstance(values, list) or len(values) != 7:
            raise OfficialCheckpointError(f"bridge_dataset.action.{name} must have length 7")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise OfficialCheckpointError(
                f"bridge_dataset.action.{name} must contain finite numbers"
            )
    if any(float(value) <= 0 for value in action["std"]):
        raise OfficialCheckpointError("bridge_dataset.action.std must be positive")
    if action.get("mask") != list(OFFICIAL_BRIDGE_ACTION_MASK):
        raise OfficialCheckpointError(
            "bridge_dataset.action.mask must match the official 6-continuous/1-gripper contract"
        )
    if action["mean"] != list(OFFICIAL_BRIDGE_ACTION_MEAN) or action["std"] != list(
        OFFICIAL_BRIDGE_ACTION_STD
    ):
        raise OfficialCheckpointError(
            "bridge_dataset.action mean/std must match the released Octo-small statistics"
        )


def validate_official_bridge_statistics(statistics: dict[str, Any], path: Path) -> None:
    """Validate the exact released Octo-small Bridge action statistics."""

    _validate_statistics(statistics, path)


def _validate_t5_base_config(config: dict[str, Any], path: Path) -> None:
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in T5_BASE_CONFIG_CONTRACT.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise OfficialCheckpointError(
            f"{path} does not satisfy the official T5-base contract: {mismatches}"
        )


def _validate_t5_tokenizer_config(config: dict[str, Any], path: Path) -> None:
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in T5_TOKENIZER_CONFIG_CONTRACT.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise OfficialCheckpointError(
            f"{path} does not satisfy the official T5 tokenizer contract: {mismatches}"
        )


def validate_official_checkpoint(
    checkpoint_path: str | Path,
) -> OfficialCheckpointReport:
    root = Path(checkpoint_path).expanduser().resolve()
    required = {
        "weights": root / "model.safetensors",
        "config": root / "model_config.json",
        "manifest": root / "conversion_manifest.json",
        "statistics": root / "dataset_statistics.json",
        "text_config": root / TEXT_ARTIFACT_PATHS[0],
        "tokenizer_config": root / TEXT_ARTIFACT_PATHS[1],
        "tokenizer_json": root / TEXT_ARTIFACT_PATHS[2],
        "sentencepiece": root / TEXT_ARTIFACT_PATHS[3],
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise OfficialCheckpointError(
            "Missing official PyTorch checkpoint files: " + ", ".join(missing)
        )

    manifest = _read_json(required["manifest"], "conversion manifest")
    if manifest.get("format") != OFFICIAL_FORMAT:
        raise OfficialCheckpointError(f"conversion manifest format must be {OFFICIAL_FORMAT!r}")
    if manifest.get("source_step") != OFFICIAL_SOURCE_STEP:
        raise OfficialCheckpointError(
            f"conversion manifest source_step must be {OFFICIAL_SOURCE_STEP}"
        )
    if manifest.get("source_octo_commit") != OFFICIAL_OCTO_COMMIT:
        raise OfficialCheckpointError(
            f"conversion manifest source_octo_commit must be {OFFICIAL_OCTO_COMMIT}"
        )
    for field in (
        "source_checkpoint_sha256",
        "weights_sha256",
        "dataset_statistics_sha256",
    ):
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise OfficialCheckpointError(
                f"conversion manifest {field} must be a lowercase SHA-256"
            )
    declared_text_hashes = manifest.get("text_artifact_sha256")
    if not isinstance(declared_text_hashes, dict) or set(declared_text_hashes) != set(
        TEXT_ARTIFACT_PATHS
    ):
        raise OfficialCheckpointError(
            "conversion manifest text_artifact_sha256 must cover every runtime text artifact"
        )
    for relative_path, value in declared_text_hashes.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise OfficialCheckpointError(
                f"conversion manifest text hash for {relative_path} must be a lowercase SHA-256"
            )
    for field in (
        "intentionally_initialized",
        "skipped_source_tensors",
        "unexpected_source_tensors",
    ):
        if manifest.get(field) != []:
            raise OfficialCheckpointError(f"conversion manifest {field} must be an empty list")
    copied = manifest.get("copied_tensors")
    if not isinstance(copied, list) or len(copied) != 367:
        raise OfficialCheckpointError(
            "conversion manifest copied_tensors must contain all 367 official mappings"
        )
    if manifest.get("copied_tensor_count") != len(copied):
        raise OfficialCheckpointError(
            "conversion manifest copied_tensor_count does not match copied_tensors"
        )
    targets: list[str] = []
    sources: list[str] = []
    for index, entry in enumerate(copied):
        if not isinstance(entry, dict) or set(entry) != {
            "target",
            "source",
            "transform",
        }:
            raise OfficialCheckpointError(f"conversion manifest copied_tensors[{index}] is invalid")
        target = entry["target"]
        source = entry["source"]
        transform = entry["transform"]
        if not isinstance(target, str) or not target:
            raise OfficialCheckpointError("conversion target names must be non-empty")
        if not isinstance(source, str) or not source:
            raise OfficialCheckpointError("conversion source names must be non-empty")
        if transform not in {
            "identity",
            "linear",
            "attention_out",
            "conv",
            "flatten",
        }:
            raise OfficialCheckpointError(
                f"conversion manifest has unknown transform {transform!r}"
            )
        targets.append(target)
        sources.append(source)
    if len(set(targets)) != 367:
        raise OfficialCheckpointError("conversion manifest target mappings must be unique")
    if len(set(sources)) != 366:
        raise OfficialCheckpointError(
            "conversion manifest must map exactly 366 official source tensors"
        )
    if any("proprio" in target for target in targets):
        raise OfficialCheckpointError("official model conversion must not contain proprio tensors")
    required_targets = {
        "primary_encoder.stem.0.weight",
        "wrist_encoder.stem.0.weight",
        "transformer.blocks.0.attention.query.weight",
        "transformer.blocks.11.mlp_out.weight",
        "action_head.reverse_input.weight",
        "action_head.reverse_output.weight",
        "action_head.reverse_output.bias",
        "text_encoder.encoder.embed_tokens.weight",
    }
    missing_targets = sorted(required_targets - set(targets))
    if missing_targets:
        raise OfficialCheckpointError(
            f"conversion manifest is missing required targets: {missing_targets}"
        )
    required_sources = {
        "heads_action/diffusion_model/reverse_network/Dense_0/kernel",
        "heads_action/diffusion_model/reverse_network/Dense_1/kernel",
        "heads_action/diffusion_model/reverse_network/Dense_1/bias",
    }
    missing_sources = sorted(required_sources - set(sources))
    if missing_sources:
        raise OfficialCheckpointError(
            f"conversion manifest is missing official source tensors: {missing_sources}"
        )

    declared_shapes = manifest.get("tensor_shapes")
    if not isinstance(declared_shapes, dict) or not declared_shapes:
        raise OfficialCheckpointError("conversion manifest must contain tensor_shapes")
    try:
        from safetensors import safe_open

        with safe_open(str(required["weights"]), framework="pt", device="cpu") as handle:
            serialized_keys = set(handle.keys())
            if serialized_keys != set(declared_shapes):
                raise OfficialCheckpointError(
                    "safetensors keys do not match manifest tensor_shapes"
                )
            mismatched_shapes = []
            for key in sorted(serialized_keys):
                actual = list(handle.get_slice(key).get_shape())
                declared = declared_shapes[key]
                if not isinstance(declared, list) or actual != declared:
                    mismatched_shapes.append({"key": key, "declared": declared, "actual": actual})
            if mismatched_shapes:
                raise OfficialCheckpointError(
                    f"safetensors do not match manifest tensor_shapes: {mismatched_shapes[:5]}"
                )
    except OfficialCheckpointError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise OfficialCheckpointError(f"Could not inspect model.safetensors: {error}") from error
    missing_serialized_targets = set(targets) - serialized_keys
    tied_embeddings = {
        "text_encoder.shared.weight",
        "text_encoder.encoder.embed_tokens.weight",
    }
    if (
        len(missing_serialized_targets) != 1
        or not missing_serialized_targets.issubset(tied_embeddings)
        or serialized_keys - set(targets)
    ):
        raise OfficialCheckpointError(
            "safetensors must contain every converted target except one tied T5 embedding"
        )

    config_value = _read_json(required["config"], "model config")
    try:
        config = OctoOfficialConfig.from_dict(config_value)
    except (TypeError, ValueError) as error:
        raise OfficialCheckpointError(f"Invalid model config: {error}") from error
    expected_config = OctoOfficialConfig()
    contract = {
        key: (actual, expected_config.to_dict()[key]) for key, actual in config.to_dict().items()
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in contract.items()
        if actual != expected
    }
    if mismatches:
        raise OfficialCheckpointError(
            f"Checkpoint does not satisfy the official model contract: {mismatches}"
        )

    text_config = _read_json(required["text_config"], "T5 config")
    _validate_t5_base_config(text_config, required["text_config"])
    tokenizer_config = _read_json(required["tokenizer_config"], "T5 tokenizer config")
    _validate_t5_tokenizer_config(tokenizer_config, required["tokenizer_config"])

    statistics = _read_json(required["statistics"], "dataset statistics")
    _validate_statistics(statistics, required["statistics"])
    weights_sha256 = sha256_file(required["weights"])
    statistics_sha256 = sha256_file(required["statistics"])
    declared_weights_hash = manifest["weights_sha256"]
    if declared_weights_hash != weights_sha256:
        raise OfficialCheckpointError("model.safetensors hash does not match manifest")
    declared_statistics_hash = manifest["dataset_statistics_sha256"]
    if declared_statistics_hash != statistics_sha256:
        raise OfficialCheckpointError("dataset_statistics.json hash does not match manifest")
    text_artifact_sha256 = {
        relative_path: sha256_file(root / relative_path) for relative_path in TEXT_ARTIFACT_PATHS
    }
    for relative_path, actual_hash in text_artifact_sha256.items():
        if declared_text_hashes[relative_path] != actual_hash:
            raise OfficialCheckpointError(f"{relative_path} hash does not match manifest")
    for relative_path, expected_hash in OFFICIAL_T5_CONTENT_SHA256.items():
        if text_artifact_sha256[relative_path] != expected_hash:
            raise OfficialCheckpointError(
                f"{relative_path} does not match the released T5-base content"
            )

    return OfficialCheckpointReport(
        path=root,
        weights_path=required["weights"],
        statistics_path=required["statistics"],
        manifest=manifest,
        config=config,
        weights_sha256=weights_sha256,
        statistics_sha256=statistics_sha256,
        text_artifact_sha256=text_artifact_sha256,
    )


def validate_official_finetuned_checkpoint(
    checkpoint_path: str | Path,
) -> OfficialCheckpointReport:
    """Validate a self-contained official-semantics fine-tuning checkpoint."""

    root = Path(checkpoint_path).expanduser().resolve()
    required = {
        "weights": root / "model.safetensors",
        "config": root / "model_config.json",
        "manifest": root / "checkpoint_manifest.json",
        "statistics": root / "dataset_statistics.json",
        "training_config": root / "finetune_config.json",
        "dataset_manifest": root / "dataset_manifest.json",
        "training_state": root / "training_state.pt",
        "text_config": root / TEXT_ARTIFACT_PATHS[0],
        "tokenizer_config": root / TEXT_ARTIFACT_PATHS[1],
        "tokenizer_json": root / TEXT_ARTIFACT_PATHS[2],
        "sentencepiece": root / TEXT_ARTIFACT_PATHS[3],
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise OfficialCheckpointError(
            "Missing official fine-tune checkpoint files: " + ", ".join(missing)
        )
    manifest = _read_json(required["manifest"], "fine-tune checkpoint manifest")
    if manifest.get("format") != OFFICIAL_FINETUNE_FORMAT:
        raise OfficialCheckpointError(
            f"fine-tune manifest format must be {OFFICIAL_FINETUNE_FORMAT!r}"
        )
    training_step = manifest.get("training_step")
    if isinstance(training_step, bool) or not isinstance(training_step, int) or training_step <= 0:
        raise OfficialCheckpointError("fine-tune manifest training_step must be positive")
    for field in (
        "weights_sha256",
        "model_config_sha256",
        "dataset_statistics_sha256",
        "finetune_config_sha256",
        "dataset_manifest_sha256",
        "data_selection_sha256",
    ):
        if not _is_sha256(manifest.get(field)):
            raise OfficialCheckpointError(
                f"fine-tune manifest {field} must be a lowercase SHA-256"
            )
    base = manifest.get("base_checkpoint")
    expected_base_fields = {
        "format",
        "source_step",
        "source_octo_commit",
        "weights_sha256",
        "conversion_manifest_sha256",
    }
    if not isinstance(base, dict) or set(base) != expected_base_fields:
        raise OfficialCheckpointError("fine-tune manifest base_checkpoint is invalid")
    if (
        base["format"] != OFFICIAL_FORMAT
        or base["source_step"] != OFFICIAL_SOURCE_STEP
        or base["source_octo_commit"] != OFFICIAL_OCTO_COMMIT
        or not _is_sha256(base["weights_sha256"])
        or not _is_sha256(base["conversion_manifest_sha256"])
    ):
        raise OfficialCheckpointError(
            "fine-tune manifest base_checkpoint does not identify the official artifact"
        )
    expected_training_contract = {
        "observation_tokenizers": ["primary"],
        "history_horizon": 2,
        "action_horizon": 4,
        "action_dim": 7,
        "use_proprio": False,
        "action_normalization": "bridge_dataset_mean_std",
        "gripper_transform": "trajectory_backward_binarize_minus_one_to_one",
        "diffusion_loss_readouts": "all_valid",
        "padded_readouts": "masked",
    }
    if manifest.get("training_contract") != expected_training_contract:
        raise OfficialCheckpointError(
            "fine-tune manifest does not satisfy the official training contract"
        )
    declared_text_hashes = manifest.get("text_artifact_sha256")
    if (
        not isinstance(declared_text_hashes, dict)
        or set(declared_text_hashes) != set(TEXT_ARTIFACT_PATHS)
        or any(not _is_sha256(value) for value in declared_text_hashes.values())
    ):
        raise OfficialCheckpointError(
            "fine-tune manifest text_artifact_sha256 must cover every runtime text artifact"
        )

    config = _validate_model_config(required["config"])
    statistics = _read_json(required["statistics"], "dataset statistics")
    _validate_statistics(statistics, required["statistics"])
    _validate_t5_base_config(
        _read_json(required["text_config"], "T5 config"),
        required["text_config"],
    )
    _validate_t5_tokenizer_config(
        _read_json(required["tokenizer_config"], "T5 tokenizer config"),
        required["tokenizer_config"],
    )
    _read_json(required["training_config"], "fine-tune config")
    dataset_manifest = _read_json(required["dataset_manifest"], "dataset manifest")
    if dataset_manifest.get("selection_sha256") != manifest["data_selection_sha256"]:
        raise OfficialCheckpointError(
            "dataset manifest selection does not match fine-tune manifest"
        )

    actual_hashes = {
        "weights_sha256": sha256_file(required["weights"]),
        "model_config_sha256": sha256_file(required["config"]),
        "dataset_statistics_sha256": sha256_file(required["statistics"]),
        "finetune_config_sha256": sha256_file(required["training_config"]),
        "dataset_manifest_sha256": sha256_file(required["dataset_manifest"]),
    }
    descriptions = {
        "weights_sha256": "model.safetensors",
        "model_config_sha256": "model_config.json",
        "dataset_statistics_sha256": "dataset_statistics.json",
        "finetune_config_sha256": "finetune_config.json",
        "dataset_manifest_sha256": "dataset_manifest.json",
    }
    for field, actual_hash in actual_hashes.items():
        if manifest[field] != actual_hash:
            raise OfficialCheckpointError(
                f"{descriptions[field]} hash does not match manifest"
            )
    text_artifact_sha256 = {
        relative_path: sha256_file(root / relative_path)
        for relative_path in TEXT_ARTIFACT_PATHS
    }
    for relative_path, actual_hash in text_artifact_sha256.items():
        if declared_text_hashes[relative_path] != actual_hash:
            raise OfficialCheckpointError(f"{relative_path} hash does not match manifest")
    for relative_path, expected_hash in OFFICIAL_T5_CONTENT_SHA256.items():
        if text_artifact_sha256[relative_path] != expected_hash:
            raise OfficialCheckpointError(
                f"{relative_path} does not match the released T5-base content"
            )
    _inspect_finetuned_tensor_shapes(required["weights"], manifest.get("tensor_shapes"))
    return OfficialCheckpointReport(
        path=root,
        weights_path=required["weights"],
        statistics_path=required["statistics"],
        manifest=manifest,
        config=config,
        weights_sha256=actual_hashes["weights_sha256"],
        statistics_sha256=actual_hashes["dataset_statistics_sha256"],
        text_artifact_sha256=text_artifact_sha256,
        checkpoint_kind="official_finetuned",
    )


def validate_official_or_finetuned_checkpoint(
    checkpoint_path: str | Path,
) -> OfficialCheckpointReport:
    """Dispatch strict validation without loading model weights."""

    root = Path(checkpoint_path).expanduser().resolve()
    has_base_manifest = (root / "conversion_manifest.json").is_file()
    has_finetune_manifest = (root / "checkpoint_manifest.json").is_file()
    if has_base_manifest and has_finetune_manifest:
        raise OfficialCheckpointError(
            "Checkpoint cannot contain both conversion and fine-tune manifests"
        )
    if has_finetune_manifest:
        return validate_official_finetuned_checkpoint(root)
    return validate_official_checkpoint(root)
