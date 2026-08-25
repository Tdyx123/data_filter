from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ipc import serve_policy
from .policy import load_official_policy


class OfficialServerError(RuntimeError):
    """Raised when startup inference violates the official policy contract."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the standalone official-semantics Octo-small PyTorch policy."
    )
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--auth-key-hex", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser


def runtime_versions() -> dict[str, str]:
    import torch

    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "not-installed"
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers_version,
    }


def run_startup_preflight(policy: Any) -> dict[str, Any]:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    instruction = "Pick up the object."
    generator = policy.make_generator(0)
    policy.begin_episode(instruction)
    first = policy.prepare_observation(image, instruction)
    actions = np.asarray(policy.predict_actions(first, generator=generator), dtype=np.float32)
    if actions.shape != (1, 4, 7):
        raise OfficialServerError(f"Startup inference returned {actions.shape}; expected (1, 4, 7)")
    if not np.all(np.isfinite(actions)):
        raise OfficialServerError("Startup inference actions must be finite")
    first_description = policy.describe_observation(first)
    second = policy.prepare_observation(image, instruction)
    second_description = policy.describe_observation(second)
    second_actions = np.asarray(
        policy.predict_actions(second, generator=generator), dtype=np.float32
    )
    if first_description.get("image_history_length") != 1:
        raise OfficialServerError("First episode inference must have one image frame")
    if second_description.get("image_history_length") != 2:
        raise OfficialServerError("Second episode inference must have two image frames")
    if second_actions.shape != (1, 4, 7) or not np.all(np.isfinite(second_actions)):
        raise OfficialServerError("Second episode inference must return finite (1, 4, 7) actions")
    if second_description.get("use_proprio") is not False:
        raise OfficialServerError("Official startup batch must not use proprio")
    policy.begin_episode(instruction)
    return {
        "action_shape": [1, 4, 7],
        "finite": True,
        "first_inference_history_length": 1,
        "second_inference_history_length": 2,
        "image_history_horizon": 2,
        "use_proprio": False,
    }


def server_metadata(
    checkpoint: Any,
    policy: Any,
    *,
    device: str,
    precision: str,
    startup_preflight: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_report = dict(checkpoint.as_dict())
    checkpoint_report["statistics"] = policy.statistics.as_dict()
    protocol = dict(policy.protocol_metadata())
    return {
        "model": "Octo-small official-semantics PyTorch baseline",
        "baseline": protocol.get(
            "baseline", checkpoint_report.get("checkpoint_kind", "official_parity")
        ),
        "native_action_chunk_size": 4,
        "action_dim": 7,
        "image_history_horizon": 2,
        "use_proprio": False,
        "device": str(device),
        "precision": str(precision),
        "checkpoint": checkpoint_report,
        "protocol": protocol,
        "rng": dict(protocol["rng"]),
        "model_runtime": runtime_versions(),
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
    checkpoint, policy = load_official_policy(
        arguments.checkpoint,
        device=arguments.device,
        precision=arguments.precision,
    )
    startup_preflight = run_startup_preflight(policy)
    serve_policy(
        socket_path=arguments.socket,
        authkey=authkey,
        policy=policy,
        metadata=server_metadata(
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
