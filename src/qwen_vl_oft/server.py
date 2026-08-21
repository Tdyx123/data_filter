"""Model-side entry point for split Qwen-VL OFT SimplerEnv evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from qwen3_vl_groot.ipc import serve_policy
from simpler_bridge.evaluation import SimplerEvaluationError

from .inference import BridgePolicy
from .simpler_evaluation import OFTPolicyAdapter, resolve_oft_checkpoint


class QwenOFTServerError(RuntimeError):
    """Raised when the OFT model fails its startup inference contract."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a Qwen-VL OFT Bridge checkpoint over a private Unix socket."
    )
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--auth-key-hex", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser


def load_oft_simpler_policy(
    checkpoint_path: Path,
    *,
    model_path: Path | None,
    device: str,
) -> tuple[Any, OFTPolicyAdapter]:
    checkpoint = resolve_oft_checkpoint(checkpoint_path, model_path=model_path)
    data_config = checkpoint.config["data"]
    try:
        policy = BridgePolicy.from_pretrained(
            checkpoint.requested_path,
            model_path=checkpoint.base_model_path,
            device=device,
        )
    except Exception as error:
        raise SimplerEvaluationError(
            f"Could not load Qwen-VL OFT checkpoint: {error}"
        ) from error
    return checkpoint, OFTPolicyAdapter(
        policy=policy,
        crop_size=int(data_config["train_crop_size"]),
        output_size=int(data_config["output_image_size"]),
    )


def run_startup_preflight(policy: Any) -> dict[str, Any]:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    proprio = np.zeros(8, dtype=np.float32)
    generator = policy.make_generator(0)
    prepared = policy.prepare_observation(image, proprio, "Pick up the object.")
    actions = np.asarray(
        policy.predict_actions(prepared, generator=generator),
        dtype=np.float32,
    )
    if actions.shape != (1, 8, 7):
        raise QwenOFTServerError(
            f"Startup inference returned {actions.shape}; expected (1, 8, 7)"
        )
    if not np.all(np.isfinite(actions)):
        raise QwenOFTServerError("Startup inference actions must be finite")
    return {"action_shape": [1, 8, 7], "finite": True}


def _server_metadata(
    checkpoint: Any,
    policy: Any,
    *,
    device: str,
    startup_preflight: dict[str, Any],
) -> dict[str, Any]:
    data_config = checkpoint.config["data"]
    output_size = int(data_config["output_image_size"])
    return {
        "model": "Qwen-VL OFT Bridge checkpoint",
        "native_action_chunk_size": 8,
        "action_dim": 7,
        "device": str(device),
        "checkpoint": dict(checkpoint.as_dict()),
        "model_image_shape": [output_size, output_size, 3],
        "train_crop_size": int(data_config["train_crop_size"]),
        "protocol": dict(policy.protocol_metadata()),
        "startup_preflight": dict(startup_preflight),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        authkey = bytes.fromhex(arguments.auth_key_hex)
    except ValueError as error:
        raise SystemExit("--auth-key-hex must contain valid hex bytes") from error
    if not authkey:
        raise SystemExit("--auth-key-hex must be non-empty")
    checkpoint, policy = load_oft_simpler_policy(
        arguments.checkpoint,
        model_path=arguments.model_path,
        device=arguments.device,
    )
    startup_preflight = run_startup_preflight(policy)
    serve_policy(
        socket_path=arguments.socket,
        authkey=authkey,
        policy=policy,
        metadata=_server_metadata(
            checkpoint,
            policy,
            device=arguments.device,
            startup_preflight=startup_preflight,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
