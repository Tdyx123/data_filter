"""Model-side entry point for the managed StarVLA policy process."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import STARVLA_SOURCE_COMMIT, load_model_spec
from .ipc import serve_policy
from .modeling import load_starvla_policy
from .runtime import validate_model_runtime


class StarVLAServerError(RuntimeError):
    """Raised when the loaded policy fails its startup inference contract."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the released StarVLA policy locally.")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--auth-key-hex", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def build_server_metadata(
    spec: Any,
    report: Any,
    runtime: dict[str, str] | None = None,
    *,
    device: str,
) -> dict[str, Any]:
    return {
        "model": "Qwen3VL-GR00T-Bridge-RT-1",
        "starvla_source_commit": STARVLA_SOURCE_COMMIT,
        "model_dir": str(spec.model_dir),
        "checkpoint_path": str(spec.checkpoint_path),
        "base_model": str(spec.base_model),
        "native_action_chunk_size": int(spec.action_horizon),
        "action_dim": int(spec.action_dim),
        "device": str(device),
        "available_unnorm_keys": ["oxe_bridge"],
        "checkpoint_tensor_count": int(report.tensor_count),
        "checkpoint_parameter_bytes": int(report.parameter_bytes),
        "checkpoint_dtypes": list(report.dtypes),
        "runtime": dict(runtime or {}),
    }


def run_startup_preflight(policy: Any) -> dict[str, Any]:
    """Run one black-image inference before exposing the IPC socket."""

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    actions = np.asarray(
        policy.predict_actions(image, "Pick up the object."),
        dtype=np.float32,
    )
    if actions.shape != (1, 16, 7):
        raise StarVLAServerError(
            f"Startup inference returned {actions.shape}; expected (1, 16, 7)"
        )
    if not np.all(np.isfinite(actions)):
        raise StarVLAServerError("Startup inference actions must be finite")
    return {"action_shape": [1, 16, 7], "finite": True}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        authkey = bytes.fromhex(arguments.auth_key_hex)
    except ValueError as error:
        raise SystemExit("--auth-key-hex must contain valid hex bytes") from error
    if not authkey:
        raise SystemExit("--auth-key-hex must be non-empty")
    runtime = validate_model_runtime(device=arguments.device)
    spec = load_model_spec(arguments.model_dir, base_model=arguments.base_model)
    loaded = load_starvla_policy(spec, device=arguments.device)
    metadata = build_server_metadata(
        spec,
        loaded.checkpoint_report,
        runtime,
        device=str(loaded.policy.device),
    )
    metadata["startup_preflight"] = run_startup_preflight(loaded)
    serve_policy(
        socket_path=arguments.socket,
        authkey=authkey,
        policy=loaded,
        metadata=metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
