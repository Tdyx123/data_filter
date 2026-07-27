from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


SOURCE_PREFIX = "octo_transformer/"
TRANSFORMER_PREFIX = (
    "octo_transformer/BlockTransformer_0/Transformer_0/"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source_params(source: Path, step: int) -> dict[str, Any]:
    try:
        import orbax.checkpoint
        from flax.traverse_util import flatten_dict
    except ImportError as error:
        raise RuntimeError(
            "The one-time converter requires requirements-octo-convert.txt"
        ) from error
    checkpoint = source / str(step) / "default"
    values = orbax.checkpoint.PyTreeCheckpointer().restore(str(checkpoint))
    return flatten_dict(values, sep="/")


def _set_tensor(
    state: dict[str, Any],
    source: dict[str, Any],
    target_key: str,
    source_key: str,
    *,
    transform: str = "identity",
) -> tuple[str, str]:
    import numpy as np
    import torch

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
        raise ValueError(f"Unknown weight transform: {transform}")
    value = torch.from_numpy(array.copy()).to(dtype=state[target_key].dtype)
    if value.shape != state[target_key].shape:
        raise RuntimeError(
            f"Converted {source_key} has shape {tuple(value.shape)}, "
            f"but {target_key} requires {tuple(state[target_key].shape)}"
        )
    state[target_key] = value
    return target_key, source_key


def _mapping() -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for camera in ("primary", "wrist"):
        source = (
            f"{SOURCE_PREFIX}observation_tokenizers_{camera}/SmallStem16_0"
        )
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
                (
                    f"{camera}_encoder.embedding.weight",
                    f"{source}/embedding/kernel",
                    "conv",
                ),
                (
                    f"{camera}_encoder.embedding.bias",
                    f"{source}/embedding/bias",
                    "identity",
                ),
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
            (
                "language_pos_embedding",
                f"{SOURCE_PREFIX}task_language_pos_embedding",
                "identity",
            ),
            (
                "readout_pos_embedding",
                f"{SOURCE_PREFIX}readout_action_pos_embedding",
                "identity",
            ),
            (
                "transformer.encoder_norm.weight",
                f"{TRANSFORMER_PREFIX}encoder_norm/scale",
                "identity",
            ),
            (
                "transformer.encoder_norm.bias",
                f"{TRANSFORMER_PREFIX}encoder_norm/bias",
                "identity",
            ),
        ]
    )
    for layer in range(12):
        target = f"transformer.blocks.{layer}"
        source = f"{TRANSFORMER_PREFIX}encoderblock_{layer}"
        result.extend(
            [
                (
                    f"{target}.attention_norm.weight",
                    f"{source}/LayerNorm_0/scale",
                    "identity",
                ),
                (
                    f"{target}.attention_norm.bias",
                    f"{source}/LayerNorm_0/bias",
                    "identity",
                ),
                (
                    f"{target}.mlp_norm.weight",
                    f"{source}/LayerNorm_1/scale",
                    "identity",
                ),
                (
                    f"{target}.mlp_norm.bias",
                    f"{source}/LayerNorm_1/bias",
                    "identity",
                ),
                (
                    f"{target}.mlp_in.weight",
                    f"{source}/MlpBlock_0/Dense_0/kernel",
                    "linear",
                ),
                (
                    f"{target}.mlp_in.bias",
                    f"{source}/MlpBlock_0/Dense_0/bias",
                    "identity",
                ),
                (
                    f"{target}.mlp_out.weight",
                    f"{source}/MlpBlock_0/Dense_1/kernel",
                    "linear",
                ),
                (
                    f"{target}.mlp_out.bias",
                    f"{source}/MlpBlock_0/Dense_1/bias",
                    "identity",
                ),
            ]
        )
        attention = f"{source}/MultiHeadDotProductAttention_0"
        for name in ("query", "key", "value", "out"):
            weight_transform = "attention_out" if name == "out" else "linear"
            result.extend(
                [
                    (
                        f"{target}.attention.{name}.weight",
                        f"{attention}/{name}/kernel",
                        weight_transform,
                    ),
                    (
                        f"{target}.attention.{name}.bias",
                        f"{attention}/{name}/bias",
                        "flatten",
                    ),
                ]
            )

    diffusion = "heads_action/diffusion_model"
    result.extend(
        [
            (
                "action_head.time_features.weight",
                f"{diffusion}/time_preprocess/kernel",
                "identity",
            ),
            (
                "action_head.time_linear1.weight",
                f"{diffusion}/cond_encoder/Dense_0/kernel",
                "linear",
            ),
            (
                "action_head.time_linear1.bias",
                f"{diffusion}/cond_encoder/Dense_0/bias",
                "identity",
            ),
            (
                "action_head.time_linear2.weight",
                f"{diffusion}/cond_encoder/Dense_1/kernel",
                "linear",
            ),
            (
                "action_head.time_linear2.bias",
                f"{diffusion}/cond_encoder/Dense_1/bias",
                "identity",
            ),
            (
                "action_head.reverse_input.bias",
                f"{diffusion}/reverse_network/Dense_0/bias",
                "identity",
            ),
        ]
    )
    for block in range(3):
        target = f"action_head.reverse_blocks.{block}"
        source = f"{diffusion}/reverse_network/MLPResNetBlock_{block}"
        result.extend(
            [
                (
                    f"{target}.norm.weight",
                    f"{source}/LayerNorm_0/scale",
                    "identity",
                ),
                (
                    f"{target}.norm.bias",
                    f"{source}/LayerNorm_0/bias",
                    "identity",
                ),
                (
                    f"{target}.linear1.weight",
                    f"{source}/Dense_0/kernel",
                    "linear",
                ),
                (
                    f"{target}.linear1.bias",
                    f"{source}/Dense_0/bias",
                    "identity",
                ),
                (
                    f"{target}.linear2.weight",
                    f"{source}/Dense_1/kernel",
                    "linear",
                ),
                (
                    f"{target}.linear2.bias",
                    f"{source}/Dense_1/bias",
                    "identity",
                ),
            ]
        )
    return result


def _t5_mapping() -> list[tuple[str, str, str]]:
    source = f"{SOURCE_PREFIX}task_tokenizers_language/hf_model"
    result = [
        ("text_encoder.shared.weight", f"{source}/shared/embedding", "identity"),
        (
            "text_encoder.encoder.embed_tokens.weight",
            f"{source}/shared/embedding",
            "identity",
        ),
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


def convert_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
    *,
    step: int = 270000,
    t5_source: str = "google-t5/t5-base",
    seed: int = 42,
    overwrite: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_model
    from transformers import AutoTokenizer, T5Config, T5EncoderModel

    from .checkpoint import inspect_flax_octo_checkpoint
    from .torch_model import OctoSmallConfig, OctoSmallPolicy, save_model_config

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"Output already exists; pass --overwrite: {output}")
        if output.is_symlink() or not output.is_dir():
            raise RuntimeError(f"Refusing to overwrite unexpected output path: {output}")
        shutil.rmtree(output)
    source_report = inspect_flax_octo_checkpoint(source, step=step)
    torch.manual_seed(seed)

    text_config = T5Config.from_pretrained(
        t5_source, local_files_only=local_files_only
    )
    tokenizer = AutoTokenizer.from_pretrained(
        t5_source, local_files_only=local_files_only
    )
    model = OctoSmallPolicy(T5EncoderModel(text_config), OctoSmallConfig())
    state = model.state_dict()
    source_params = _load_source_params(source, step)
    copied: list[dict[str, str]] = []
    for target_key, source_key, transform in [*_mapping(), *_t5_mapping()]:
        _set_tensor(
            state,
            source_params,
            target_key,
            source_key,
            transform=transform,
        )
        copied.append(
            {"target": target_key, "source": source_key, "transform": transform}
        )
    model.load_state_dict(state, strict=True)

    intentionally_initialized = [
        "proprio_pos_embedding",
        "proprio_projection.weight",
        "proprio_projection.bias",
        "action_head.reverse_input.weight",
        "action_head.reverse_output.weight",
        "action_head.reverse_output.bias",
    ]
    skipped_source = [
        "heads_action/diffusion_model/reverse_network/Dense_0/kernel",
        "heads_action/diffusion_model/reverse_network/Dense_1/kernel",
        "heads_action/diffusion_model/reverse_network/Dense_1/bias",
    ]
    used_source = {entry["source"] for entry in copied}
    unexpected_source = sorted(
        set(source_params) - used_source - set(skipped_source)
    )
    if unexpected_source:
        raise RuntimeError(
            "Checkpoint conversion left unexpected source tensors unmapped: "
            + ", ".join(unexpected_source[:20])
        )

    output.mkdir(parents=True)
    text_root = output / "text_encoder"
    text_config.save_pretrained(text_root)
    tokenizer.save_pretrained(text_root)
    save_model_config(model.config, output / "model_config.json")
    weights_path = output / "model.safetensors"
    save_model(model, str(weights_path))
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        tensor_shapes = {
            key: list(handle.get_slice(key).get_shape()) for key in handle.keys()
        }
    manifest = {
        "format": "octo-small-pytorch",
        "source_path": str(source),
        "source_step": step,
        "source_checkpoint_sha256": source_report["checkpoint_sha256"],
        "seed": seed,
        "t5_source": t5_source,
        "copied_tensor_count": len(copied),
        "copied_tensors": copied,
        "tensor_shapes": tensor_shapes,
        "intentionally_initialized": intentionally_initialized,
        "skipped_source_tensors": skipped_source,
        "unexpected_source_tensors": unexpected_source,
    }
    with (output / "conversion_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a local Octo-small Orbax checkpoint to PyTorch safetensors"
    )
    parser.add_argument("--source", default="/data/dwb/models/octo-small")
    parser.add_argument("--output", default="/data/dwb/models/octo-small-pytorch")
    parser.add_argument("--step", type=int, default=270000)
    parser.add_argument("--t5-source", default="google-t5/t5-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
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
