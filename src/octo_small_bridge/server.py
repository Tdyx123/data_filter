"""Model-side entry point for split Octo-small SimplerEnv evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ipc import serve_policy
from .simpler_evaluation import load_octo_bridge_policy


class OctoServerError(RuntimeError):
    """Raised when the model fails the startup inference contract."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve an Octo-small Bridge checkpoint over a private Unix socket."
    )
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--auth-key-hex", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser


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
        raise OctoServerError(
            f"Startup inference returned {actions.shape}; expected (1, 8, 7)"
        )
    if not np.all(np.isfinite(actions)):
        raise OctoServerError("Startup inference actions must be finite")
    return {"action_shape": [1, 8, 7], "finite": True}


def _server_metadata(
    checkpoint: Any,
    policy: Any,
    *,
    device: str,
    precision: str,
    startup_preflight: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_report = dict(checkpoint.as_dict())
    checkpoint_report["statistics"] = policy.statistics.as_dict()
    return {
        "model": "Octo-small Bridge checkpoint",
        "native_action_chunk_size": 8,
        "action_dim": 7,
        "device": str(device),
        "precision": str(precision),
        "checkpoint": checkpoint_report,
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
    checkpoint, policy = load_octo_bridge_policy(
        arguments.checkpoint,
        base_model=arguments.base_model,
        statistics=arguments.statistics,
        device=arguments.device,
        precision=arguments.precision,
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
            precision=arguments.precision,
            startup_preflight=startup_preflight,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
