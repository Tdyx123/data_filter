from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .checkpoint import (
    OFFICIAL_FORMAT,
    OFFICIAL_OCTO_COMMIT,
    OFFICIAL_SOURCE_STEP,
    OFFICIAL_T5_CONTENT_SHA256,
    T5_BASE_CONFIG_CONTRACT,
    T5_TOKENIZER_CONFIG_CONTRACT,
    TEXT_ARTIFACT_PATHS,
    sha256_file,
    validate_official_bridge_statistics,
    validate_official_checkpoint,
)


SOURCE_PREFIX = "octo_transformer/"
TRANSFORMER_PREFIX = "octo_transformer/BlockTransformer_0/Transformer_0/"
DEFAULT_SOURCE = "/data/dwb/models/octo-small"
DEFAULT_OUTPUT = "/data/dwb/models/octo-small-pytorch-official"
DEFAULT_T5_SOURCE = "/data/dwb/models/t5-base"


def _source_checkpoint_path(source: Path, step: int) -> Path:
    return source / str(step) / "default" / "checkpoint"


def _inspect_source(source: Path, step: int) -> dict[str, Any]:
    required = {
        "config": source / "config.json",
        "statistics": source / "dataset_statistics.json",
        "checkpoint": _source_checkpoint_path(source, step),
        "step_commit": source / str(step) / "commit_success.txt",
        "item_commit": source / str(step) / "default" / "commit_success.txt",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing official Octo-small source files: " + ", ".join(missing))
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    model = config.get("model", {})
    transformer = model.get("transformer_kwargs", {})
    action = model.get("heads", {}).get("action", {}).get("kwargs", {})
    expected = {
        "token_embedding_size": (model.get("token_embedding_size"), 384),
        "num_layers": (transformer.get("num_layers"), 12),
        "num_attention_heads": (transformer.get("num_attention_heads"), 6),
        "action_dim": (action.get("action_dim"), 7),
        "pred_horizon": (action.get("pred_horizon"), 4),
    }
    mismatches = {
        key: {"actual": actual, "expected": wanted}
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    }
    tokenizers = model.get("observation_tokenizers", {})
    if not {"primary", "wrist"}.issubset(tokenizers):
        mismatches["observation_tokenizers"] = {
            "actual": sorted(tokenizers),
            "expected": ["primary", "wrist"],
        }
    if mismatches:
        raise RuntimeError(f"Source does not match official Octo-small: {mismatches}")
    try:
        statistics = json.loads(required["statistics"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read official source statistics: {error}") from error
    validate_official_bridge_statistics(statistics, required["statistics"])
    return {
        "checkpoint": required["checkpoint"],
        "statistics": required["statistics"],
        "checkpoint_sha256": sha256_file(required["checkpoint"]),
    }


def _load_source_params(source: Path, step: int) -> dict[str, Any]:
    try:
        import orbax.checkpoint
        from flax.traverse_util import flatten_dict
    except ImportError as error:
        raise RuntimeError(
            "The converter requires dependencies from requirements-octo-convert.txt"
        ) from error
    values = orbax.checkpoint.PyTreeCheckpointer().restore(str(source / str(step) / "default"))
    return flatten_dict(values, sep="/")


def _convert_tensor(
    state: dict[str, Any],
    source: dict[str, Any],
    target_key: str,
    source_key: str,
    transform: str,
) -> None:
    import numpy as np
    import torch

    if target_key not in state:
        raise RuntimeError(f"PyTorch model is missing conversion target {target_key}")
    if source_key not in source:
        raise RuntimeError(f"Source checkpoint is missing {source_key}")
    array = np.asarray(source[source_key])
    if transform == "linear":
        array = array.reshape(array.shape[0], -1).T
    elif transform == "attention_out":
        array = array.reshape(-1, array.shape[-1]).T
    elif transform == "conv":
        array = np.transpose(array, (3, 2, 0, 1))
    elif transform == "flatten":
        array = array.reshape(-1)
    elif transform != "identity":
        raise ValueError(f"Unknown conversion transform: {transform}")
    value = torch.from_numpy(array.copy()).to(dtype=state[target_key].dtype)
    if value.shape != state[target_key].shape:
        raise RuntimeError(
            f"Converted {source_key} has shape {tuple(value.shape)}, but "
            f"{target_key} requires {tuple(state[target_key].shape)}"
        )
    state[target_key] = value


def conversion_mapping() -> list[tuple[str, str, str]]:
    """Return the complete non-T5 official Octo-small inference mapping."""
    result: list[tuple[str, str, str]] = []
    for camera in ("primary", "wrist"):
        source = f"{SOURCE_PREFIX}observation_tokenizers_{camera}/SmallStem16_0"
        for layer, module_index in enumerate((0, 3, 6, 9)):
            result.extend(
                [
                    (
                        f"{camera}_encoder.stem.{module_index}.weight",
                        f"{source}/StdConv_{layer}/kernel",
                        "conv",
                    ),
                    (
                        f"{camera}_encoder.stem.{module_index}.bias",
                        f"{source}/StdConv_{layer}/bias",
                        "identity",
                    ),
                    (
                        f"{camera}_encoder.stem.{module_index + 1}.weight",
                        f"{source}/GroupNorm_{layer}/scale",
                        "identity",
                    ),
                    (
                        f"{camera}_encoder.stem.{module_index + 1}.bias",
                        f"{source}/GroupNorm_{layer}/bias",
                        "identity",
                    ),
                ]
            )
        result.extend(
            [
                (f"{camera}_encoder.embedding.weight", f"{source}/embedding/kernel", "conv"),
                (f"{camera}_encoder.embedding.bias", f"{source}/embedding/bias", "identity"),
                (
                    f"{camera}_projection.weight",
                    f"{SOURCE_PREFIX}obs_{camera}_projection/kernel",
                    "linear",
                ),
                (
                    f"{camera}_projection.bias",
                    f"{SOURCE_PREFIX}obs_{camera}_projection/bias",
                    "identity",
                ),
                (
                    f"{camera}_pos_embedding",
                    f"{SOURCE_PREFIX}obs_{camera}_pos_embedding",
                    "identity",
                ),
            ]
        )
    result.extend(
        [
            (
                "language_projection.weight",
                f"{SOURCE_PREFIX}task_language_projection/kernel",
                "linear",
            ),
            (
                "language_projection.bias",
                f"{SOURCE_PREFIX}task_language_projection/bias",
                "identity",
            ),
            ("language_pos_embedding", f"{SOURCE_PREFIX}task_language_pos_embedding", "identity"),
            ("readout_pos_embedding", f"{SOURCE_PREFIX}readout_action_pos_embedding", "identity"),
            (
                "transformer.encoder_norm.weight",
                f"{TRANSFORMER_PREFIX}encoder_norm/scale",
                "identity",
            ),
            ("transformer.encoder_norm.bias", f"{TRANSFORMER_PREFIX}encoder_norm/bias", "identity"),
        ]
    )
    for layer in range(12):
        target = f"transformer.blocks.{layer}"
        source = f"{TRANSFORMER_PREFIX}encoderblock_{layer}"
        result.extend(
            [
                (f"{target}.attention_norm.weight", f"{source}/LayerNorm_0/scale", "identity"),
                (f"{target}.attention_norm.bias", f"{source}/LayerNorm_0/bias", "identity"),
                (f"{target}.mlp_norm.weight", f"{source}/LayerNorm_1/scale", "identity"),
                (f"{target}.mlp_norm.bias", f"{source}/LayerNorm_1/bias", "identity"),
                (f"{target}.mlp_in.weight", f"{source}/MlpBlock_0/Dense_0/kernel", "linear"),
                (f"{target}.mlp_in.bias", f"{source}/MlpBlock_0/Dense_0/bias", "identity"),
                (f"{target}.mlp_out.weight", f"{source}/MlpBlock_0/Dense_1/kernel", "linear"),
                (f"{target}.mlp_out.bias", f"{source}/MlpBlock_0/Dense_1/bias", "identity"),
            ]
        )
        attention = f"{source}/MultiHeadDotProductAttention_0"
        for name in ("query", "key", "value", "out"):
            transform = "attention_out" if name == "out" else "linear"
            result.extend(
                [
                    (f"{target}.attention.{name}.weight", f"{attention}/{name}/kernel", transform),
                    (f"{target}.attention.{name}.bias", f"{attention}/{name}/bias", "flatten"),
                ]
            )
    diffusion = "heads_action/diffusion_model"
    result.extend(
        [
            ("action_head.time_features.weight", f"{diffusion}/time_preprocess/kernel", "identity"),
            (
                "action_head.time_linear1.weight",
                f"{diffusion}/cond_encoder/Dense_0/kernel",
                "linear",
            ),
            ("action_head.time_linear1.bias", f"{diffusion}/cond_encoder/Dense_0/bias", "identity"),
            (
                "action_head.time_linear2.weight",
                f"{diffusion}/cond_encoder/Dense_1/kernel",
                "linear",
            ),
            ("action_head.time_linear2.bias", f"{diffusion}/cond_encoder/Dense_1/bias", "identity"),
            (
                "action_head.reverse_input.weight",
                f"{diffusion}/reverse_network/Dense_0/kernel",
                "linear",
            ),
            (
                "action_head.reverse_input.bias",
                f"{diffusion}/reverse_network/Dense_0/bias",
                "identity",
            ),
            (
                "action_head.reverse_output.weight",
                f"{diffusion}/reverse_network/Dense_1/kernel",
                "linear",
            ),
            (
                "action_head.reverse_output.bias",
                f"{diffusion}/reverse_network/Dense_1/bias",
                "identity",
            ),
        ]
    )
    for block in range(3):
        target = f"action_head.reverse_blocks.{block}"
        source = f"{diffusion}/reverse_network/MLPResNetBlock_{block}"
        result.extend(
            [
                (f"{target}.norm.weight", f"{source}/LayerNorm_0/scale", "identity"),
                (f"{target}.norm.bias", f"{source}/LayerNorm_0/bias", "identity"),
                (f"{target}.linear1.weight", f"{source}/Dense_0/kernel", "linear"),
                (f"{target}.linear1.bias", f"{source}/Dense_0/bias", "identity"),
                (f"{target}.linear2.weight", f"{source}/Dense_1/kernel", "linear"),
                (f"{target}.linear2.bias", f"{source}/Dense_1/bias", "identity"),
            ]
        )
    return result


def _t5_mapping() -> list[tuple[str, str, str]]:
    source = f"{SOURCE_PREFIX}task_tokenizers_language/hf_model"
    result = [
        ("text_encoder.shared.weight", f"{source}/shared/embedding", "identity"),
        ("text_encoder.encoder.embed_tokens.weight", f"{source}/shared/embedding", "identity"),
        (
            "text_encoder.encoder.final_layer_norm.weight",
            f"{source}/encoder/final_layer_norm/weight",
            "identity",
        ),
    ]
    for layer in range(12):
        target = f"text_encoder.encoder.block.{layer}"
        block = f"{source}/encoder/block/{layer}"
        for projection in ("q", "k", "v", "o"):
            result.append(
                (
                    f"{target}.layer.0.SelfAttention.{projection}.weight",
                    f"{block}/layer/0/SelfAttention/{projection}/kernel",
                    "linear",
                )
            )
        if layer == 0:
            result.append(
                (
                    f"{target}.layer.0.SelfAttention.relative_attention_bias.weight",
                    f"{block}/layer/0/SelfAttention/relative_attention_bias/embedding",
                    "identity",
                )
            )
        result.extend(
            [
                (
                    f"{target}.layer.0.layer_norm.weight",
                    f"{block}/layer/0/layer_norm/weight",
                    "identity",
                ),
                (
                    f"{target}.layer.1.DenseReluDense.wi.weight",
                    f"{block}/layer/1/DenseReluDense/wi/kernel",
                    "linear",
                ),
                (
                    f"{target}.layer.1.DenseReluDense.wo.weight",
                    f"{block}/layer/1/DenseReluDense/wo/kernel",
                    "linear",
                ),
                (
                    f"{target}.layer.1.layer_norm.weight",
                    f"{block}/layer/1/layer_norm/weight",
                    "identity",
                ),
            ]
        )
    return result


def _validate_t5_source(
    source: Path,
    *,
    text_config: Any,
    tokenizer: Any,
) -> None:
    required = {
        "text_encoder/tokenizer.json": source / "tokenizer.json",
        "text_encoder/spiece.model": source / "spiece.model",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Official local T5-base source is incomplete: {missing}")
    for relative_path, path in required.items():
        actual_hash = sha256_file(path)
        expected_hash = OFFICIAL_T5_CONTENT_SHA256[relative_path]
        if actual_hash != expected_hash:
            raise RuntimeError(f"{path} does not match the released T5-base content")
    config_mismatches = {
        key: {"actual": getattr(text_config, key, None), "expected": expected}
        for key, expected in T5_BASE_CONFIG_CONTRACT.items()
        if getattr(text_config, key, None) != expected
    }
    if config_mismatches:
        raise RuntimeError(f"T5 source does not match T5-base: {config_mismatches}")
    token_contract = {
        "vocab_size": (getattr(tokenizer, "vocab_size", None), 32100),
        "pad_token_id": (getattr(tokenizer, "pad_token_id", None), 0),
        "eos_token_id": (getattr(tokenizer, "eos_token_id", None), 1),
        "unk_token_id": (getattr(tokenizer, "unk_token_id", None), 2),
        "extra_id_0": (tokenizer.convert_tokens_to_ids("<extra_id_0>"), 32099),
        "extra_id_99": (tokenizer.convert_tokens_to_ids("<extra_id_99>"), 32000),
    }
    token_mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in token_contract.items()
        if actual != expected
    }
    if token_mismatches:
        raise RuntimeError(f"T5 tokenizer semantics do not match T5-base: {token_mismatches}")


def _write_official_tokenizer_artifacts(source: Path, text_root: Path) -> None:
    shutil.copy2(source / "tokenizer.json", text_root / "tokenizer.json")
    shutil.copy2(source / "spiece.model", text_root / "spiece.model")
    (text_root / "tokenizer_config.json").write_text(
        json.dumps(T5_TOKENIZER_CONFIG_CONTRACT, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _publish_staged_checkpoint(staging: Path, output: Path) -> None:
    """Publish a sibling staging directory, restoring the old artifact on failure."""

    backup: Path | None = None
    if output.exists():
        backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
        output.replace(backup)
    try:
        staging.replace(output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            backup.replace(output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _build_t5_encoder_and_config_document(
    text_config: Any,
    *,
    encoder_factory: Any,
) -> tuple[Any, dict[str, Any]]:
    """Freeze artifact metadata before encoder implementations can mutate config."""

    document = copy.deepcopy(text_config.to_dict())
    document.pop("_name_or_path", None)
    document.pop("transformers_version", None)
    encoder = encoder_factory(copy.deepcopy(text_config))
    return encoder, document


def convert_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
    *,
    step: int = OFFICIAL_SOURCE_STEP,
    t5_source: str = DEFAULT_T5_SOURCE,
    seed: int = 42,
    overwrite: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_model
    from transformers import AutoTokenizer, T5Config, T5EncoderModel

    from .model import OctoOfficialConfig, OctoSmallOfficialPolicy

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if step != OFFICIAL_SOURCE_STEP:
        raise RuntimeError(
            f"Official baseline conversion requires step {OFFICIAL_SOURCE_STEP}, found {step}"
        )
    if output == source or output == Path(output.anchor) or output == Path.home():
        raise RuntimeError(f"Refusing unsafe conversion output path: {output}")
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"Output already exists; pass --overwrite: {output}")
        if output.is_symlink() or not output.is_dir():
            raise RuntimeError(f"Refusing to overwrite unexpected output path: {output}")
        manifest_path = output / "conversion_manifest.json"
        if any(output.iterdir()):
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"Refusing to overwrite a non-official output directory: {output}"
                ) from error
            if existing_manifest.get("format") != OFFICIAL_FORMAT:
                raise RuntimeError(
                    f"Refusing to overwrite a non-official output directory: {output}"
                )
    source_report = _inspect_source(source, step)
    torch.manual_seed(seed)
    t5_root = Path(t5_source).expanduser().resolve()
    if not t5_root.is_dir():
        raise RuntimeError(
            "Official conversion requires --t5-source to be a local T5-base directory"
        )
    text_config = T5Config.from_pretrained(t5_source, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(t5_source, local_files_only=local_files_only)
    _validate_t5_source(t5_root, text_config=text_config, tokenizer=tokenizer)
    text_encoder, text_config_document = _build_t5_encoder_and_config_document(
        text_config,
        encoder_factory=T5EncoderModel,
    )
    model = OctoSmallOfficialPolicy(text_encoder, OctoOfficialConfig())
    state = model.state_dict()
    source_params = _load_source_params(source, step)
    copied: list[dict[str, str]] = []
    mapping = [*conversion_mapping(), *_t5_mapping()]
    for target_key, source_key, transform in mapping:
        _convert_tensor(state, source_params, target_key, source_key, transform)
        copied.append({"target": target_key, "source": source_key, "transform": transform})
    used_targets = {item[0] for item in mapping}
    missing_targets = sorted(set(state) - used_targets)
    if missing_targets:
        raise RuntimeError(
            "Conversion would leave PyTorch tensors initialized: " + ", ".join(missing_targets[:20])
        )
    unexpected_source = sorted(set(source_params) - {item[1] for item in mapping})
    if unexpected_source:
        raise RuntimeError(
            "Conversion left source tensors unmapped: " + ", ".join(unexpected_source[:20])
        )
    model.load_state_dict(state, strict=True)
    source_statistics = json.loads(source_report["statistics"].read_text(encoding="utf-8"))
    if "bridge_dataset" not in source_statistics:
        raise RuntimeError("Source statistics are missing bridge_dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output.parent,
            prefix=f".{output.name}.staging-",
        )
    )
    try:
        text_root = staging / "text_encoder"
        text_root.mkdir()
        (text_root / "config.json").write_text(
            json.dumps(text_config_document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_official_tokenizer_artifacts(t5_root, text_root)
        (staging / "model_config.json").write_text(
            json.dumps(model.config.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        statistics_path = staging / "dataset_statistics.json"
        statistics_path.write_text(
            json.dumps(
                {"bridge_dataset": source_statistics["bridge_dataset"]},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        weights_path = staging / "model.safetensors"
        save_model(model, str(weights_path))
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            tensor_shapes = {key: list(handle.get_slice(key).get_shape()) for key in handle.keys()}
        text_artifact_sha256 = {
            relative_path: sha256_file(staging / relative_path)
            for relative_path in TEXT_ARTIFACT_PATHS
        }
        manifest = {
            "format": OFFICIAL_FORMAT,
            "source_path": str(source),
            "source_step": step,
            "source_octo_commit": OFFICIAL_OCTO_COMMIT,
            "source_checkpoint_sha256": source_report["checkpoint_sha256"],
            "weights_sha256": sha256_file(weights_path),
            "dataset_statistics_sha256": sha256_file(statistics_path),
            "text_artifact_sha256": text_artifact_sha256,
            "seed": seed,
            "t5_source": t5_source,
            "copied_tensor_count": len(copied),
            "copied_tensors": copied,
            "tensor_shapes": tensor_shapes,
            "intentionally_initialized": [],
            "skipped_source_tensors": [],
            "unexpected_source_tensors": [],
        }
        (staging / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        validate_official_checkpoint(staging)
        _publish_staged_checkpoint(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert official Octo-small step-270000 to standalone PyTorch"
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--step", type=int, default=OFFICIAL_SOURCE_STEP)
    parser.add_argument("--t5-source", default=DEFAULT_T5_SOURCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    manifest = convert_checkpoint(
        arguments.source,
        arguments.output,
        step=arguments.step,
        t5_source=arguments.t5_source,
        seed=arguments.seed,
        overwrite=arguments.overwrite,
        local_files_only=arguments.local_files_only,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
