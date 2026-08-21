"""Model-side entry point for split Qwen SimplerEnv evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ipc import serve_policy
from .simpler_evaluation import (
    QwenPolicyAdapter,
    QwenSimplerPolicy,
    SimplerEvaluationError,
)


class QwenServerError(RuntimeError):
    """Raised when the Qwen model fails its startup inference contract."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a Qwen Bridge checkpoint over a private Unix socket."
    )
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--auth-key-hex", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoising-steps", type=int, default=4)
    return parser


def load_qwen_simpler_policy(
    checkpoint_path: Path,
    *,
    model_path: Path | None,
    device: str,
    denoising_steps: int,
) -> tuple[Any, QwenPolicyAdapter]:
    try:
        from .libero_evaluation import resolve_qwen_checkpoint
        from octo_small_libero.evaluation import EvaluationError

        checkpoint = resolve_qwen_checkpoint(checkpoint_path, model_path=model_path)
    except EvaluationError as error:
        raise SimplerEvaluationError(str(error)) from error
    try:
        data_config = checkpoint.config["data"]
        crop_size = int(data_config["train_crop_size"])
        output_size = int(data_config["output_image_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SimplerEvaluationError(
            "Bridge checkpoint requires integer train_crop_size and output_image_size"
        ) from error
    policy = QwenSimplerPolicy.from_checkpoint(checkpoint, device=device)
    return checkpoint, QwenPolicyAdapter(
        policy=policy,
        crop_size=crop_size,
        output_size=output_size,
        denoising_steps=denoising_steps,
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
        raise QwenServerError(
            f"Startup inference returned {actions.shape}; expected (1, 8, 7)"
        )
    if not np.all(np.isfinite(actions)):
        raise QwenServerError("Startup inference actions must be finite")
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
        "model": "Qwen Bridge checkpoint",
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
    if arguments.denoising_steps <= 0:
        raise SystemExit("--denoising-steps must be positive")
    checkpoint, policy = load_qwen_simpler_policy(
        arguments.checkpoint,
        model_path=arguments.model_path,
        device=arguments.device,
        denoising_steps=arguments.denoising_steps,
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
