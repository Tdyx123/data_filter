from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class OctoCheckpointError(RuntimeError):
    """Raised when an Octo-small source or PyTorch artifact is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_flax_octo_checkpoint(
    checkpoint_path: str | Path,
    *,
    step: int = 270000,
) -> dict[str, Any]:
    """Validate the legacy source checkpoint used only by the conversion command."""
    root = Path(checkpoint_path).expanduser().resolve()
    required = {
        "config": root / "config.json",
        "example_batch": root / "example_batch.msgpack",
        "dataset_statistics": root / "dataset_statistics.json",
        "checkpoint": root / str(step) / "default" / "checkpoint",
        "step_commit": root / str(step) / "commit_success.txt",
        "item_commit": root / str(step) / "default" / "commit_success.txt",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise OctoCheckpointError(f"Missing source Octo checkpoint files: {missing}")
    if required["checkpoint"].stat().st_size < 1024 * 1024:
        raise OctoCheckpointError(
            f"Checkpoint payload is unexpectedly small: {required['checkpoint']}"
        )

    with required["config"].open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model = config.get("model", {})
    transformer = model.get("transformer_kwargs", {})
    if int(model.get("token_embedding_size", -1)) != 384:
        raise OctoCheckpointError("Expected Octo-small token_embedding_size=384")
    if int(transformer.get("num_layers", -1)) != 12:
        raise OctoCheckpointError("Expected the 12-layer Octo-small transformer")
    if int(transformer.get("num_attention_heads", -1)) != 6:
        raise OctoCheckpointError("Expected Octo-small num_attention_heads=6")
    tokenizers = model.get("observation_tokenizers", {})
    if not {"primary", "wrist"}.issubset(tokenizers):
        raise OctoCheckpointError("Expected pretrained primary/wrist tokenizers")
    action_kwargs = model.get("heads", {}).get("action", {}).get("kwargs", {})
    if int(action_kwargs.get("action_dim", -1)) != 7:
        raise OctoCheckpointError("Expected pretrained action_dim=7")
    if int(action_kwargs.get("pred_horizon", -1)) != 4:
        raise OctoCheckpointError("Expected pretrained pred_horizon=4")
    return {
        "path": str(root),
        "step": step,
        "checkpoint_bytes": required["checkpoint"].stat().st_size,
        "checkpoint_sha256": _sha256(required["checkpoint"]),
        "transformer_layers": int(transformer["num_layers"]),
        "token_embedding_size": int(model["token_embedding_size"]),
        "pretrained_action_horizon": int(action_kwargs["pred_horizon"]),
        "action_dim": int(action_kwargs["action_dim"]),
        "observation_tokenizers": sorted(tokenizers),
    }


def inspect_octo_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Validate a self-contained PyTorch Octo-small safetensors artifact."""
    root = Path(checkpoint_path).expanduser().resolve()
    required = {
        "weights": root / "model.safetensors",
        "config": root / "model_config.json",
        "manifest": root / "conversion_manifest.json",
        "text_config": root / "text_encoder" / "config.json",
        "tokenizer_config": root / "text_encoder" / "tokenizer_config.json",
        "tokenizer_model": root / "text_encoder" / "spiece.model",
        "tokenizer_json": root / "text_encoder" / "tokenizer.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise OctoCheckpointError(f"Missing PyTorch Octo-small artifact files: {missing}")
    if required["weights"].stat().st_size < 1024 * 1024:
        raise OctoCheckpointError(f"Model weights are unexpectedly small: {required['weights']}")

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise OctoCheckpointError("safetensors is required to inspect the model") from error
    with required["config"].open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "hidden_size": 384,
        "transformer_layers": 12,
        "attention_heads": 6,
        "mlp_size": 1536,
        "language_tokens": 16,
        "proprio_dim": 8,
        "action_dim": 7,
        "action_horizon": 8,
        "diffusion_steps": 20,
    }
    for key, value in expected.items():
        if int(config.get(key, -1)) != value:
            raise OctoCheckpointError(f"model_config.json requires {key}={value}")
    if float(config.get("diffusion_dropout", -1)) != 0.1:
        raise OctoCheckpointError("model_config.json requires diffusion_dropout=0.1")

    with required["manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "octo-small-pytorch":
        raise OctoCheckpointError("Invalid PyTorch Octo conversion manifest format")
    if int(manifest.get("source_step", -1)) != 270000:
        raise OctoCheckpointError("Expected conversion from Octo-small step 270000")
    expected_initialized = {
        "proprio_pos_embedding",
        "proprio_projection.weight",
        "proprio_projection.bias",
        "action_head.reverse_input.weight",
        "action_head.reverse_output.weight",
        "action_head.reverse_output.bias",
    }
    if set(manifest.get("intentionally_initialized", [])) != expected_initialized:
        raise OctoCheckpointError(
            "Conversion manifest has an unexpected randomly initialized parameter set"
        )
    if manifest.get("unexpected_source_tensors"):
        raise OctoCheckpointError(
            "Conversion manifest contains unexpected unmapped source tensors"
        )

    declared_shapes = manifest.get("tensor_shapes")
    if not isinstance(declared_shapes, dict) or not declared_shapes:
        raise OctoCheckpointError("Conversion manifest is missing tensor_shapes")
    with safe_open(str(required["weights"]), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        required_keys = {
            "primary_encoder.stem.0.weight",
            "wrist_encoder.stem.0.weight",
            "language_projection.weight",
            "proprio_projection.weight",
            "transformer.blocks.0.attention.query.weight",
            "transformer.blocks.11.mlp_out.weight",
            "action_head.reverse_input.weight",
            "action_head.reverse_output.weight",
            "text_encoder.encoder.embed_tokens.weight",
        }
        missing_keys = sorted(required_keys - keys)
        if missing_keys:
            raise OctoCheckpointError(
                f"PyTorch Octo-small model is missing required tensors: {missing_keys}"
            )
        if keys != set(declared_shapes):
            raise OctoCheckpointError(
                "Safetensors keys do not exactly match conversion manifest tensor_shapes"
            )
        mismatched_shapes = []
        for key in sorted(keys):
            actual = list(handle.get_slice(key).get_shape())
            expected_shape = declared_shapes[key]
            if not isinstance(expected_shape, list) or actual != expected_shape:
                mismatched_shapes.append(
                    {"key": key, "expected": expected_shape, "actual": actual}
                )
        if mismatched_shapes:
            raise OctoCheckpointError(
                "Safetensors shapes do not match conversion manifest: "
                f"{mismatched_shapes[:5]}"
            )
        tensor_count = len(keys)
    return {
        "path": str(root),
        "format": manifest["format"],
        "source_step": int(manifest["source_step"]),
        "weights_bytes": required["weights"].stat().st_size,
        "weights_sha256": _sha256(required["weights"]),
        "conversion_manifest_sha256": _sha256(required["manifest"]),
        "tensor_count": tensor_count,
        "transformer_layers": int(config["transformer_layers"]),
        "action_horizon": int(config["action_horizon"]),
        "action_dim": int(config["action_dim"]),
        "conversion": {
            "source_path": manifest.get("source_path"),
            "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
            "copied_tensor_count": int(manifest.get("copied_tensor_count", 0)),
            "intentionally_initialized": list(
                manifest.get("intentionally_initialized", [])
            ),
            "skipped_source_tensors": list(
                manifest.get("skipped_source_tensors", [])
            ),
            "unexpected_source_tensors": list(
                manifest.get("unexpected_source_tensors", [])
            ),
        },
    }


def load_lerobot_statistics(dataset_root: str | Path) -> dict[str, Any]:
    from .lerobot_v2 import ACTION_KEY, STATE_KEY, LeRobotV2Metadata

    try:
        metadata = LeRobotV2Metadata(dataset_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise OctoCheckpointError(str(error)) from error
    statistics = {
        "action": metadata.stats.get(ACTION_KEY, {}),
        "proprio": metadata.stats.get(STATE_KEY, {}),
        "num_trajectories": int(metadata.info["total_episodes"]),
        "num_transitions": int(metadata.info["total_frames"]),
    }
    for key, dimension in (("action", 7), ("proprio", 8)):
        entry = statistics[key]
        for statistic in ("mean", "std", "min", "max"):
            values = entry.get(statistic)
            if not isinstance(values, list) or len(values) != dimension:
                raise OctoCheckpointError(
                    f"{metadata.root}/meta/stats.json: "
                    f"{key}.{statistic} must have length {dimension}"
                )
            array = np.asarray(values, dtype=np.float32)
            if not np.all(np.isfinite(array)):
                raise OctoCheckpointError(
                    f"{metadata.root}/meta/stats.json: {key}.{statistic} contains NaN or infinity"
                )
            entry[statistic] = array
    if int(statistics.get("num_trajectories", 0)) <= 0:
        raise OctoCheckpointError(
            f"{metadata.root}/meta/info.json: total_episodes must be positive"
        )
    if int(statistics.get("num_transitions", 0)) <= 0:
        raise OctoCheckpointError(
            f"{metadata.root}/meta/info.json: total_frames must be positive"
        )
    return statistics
