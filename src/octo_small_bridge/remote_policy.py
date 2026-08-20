"""SimplerEnv-side adapter for the remote Octo-small policy service."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from simpler_bridge.evaluation import SimplerEvaluationError

from .ipc import IPC_PROTOCOL_VERSION


class OctoRemotePolicy:
    policy_name = "Octo-small Bridge checkpoint"
    gripper_threshold = 0.5

    def __init__(self, client: Any) -> None:
        self.client = client
        metadata = dict(client.metadata())
        if metadata.get("protocol_version") != IPC_PROTOCOL_VERSION:
            raise SimplerEvaluationError(
                f"Octo IPC protocol_version must be {IPC_PROTOCOL_VERSION}, "
                f"found {metadata.get('protocol_version')}"
            )
        if metadata.get("native_action_chunk_size") != 8:
            raise SimplerEvaluationError(
                "Octo native_action_chunk_size must be 8, found "
                f"{metadata.get('native_action_chunk_size')}"
            )
        if metadata.get("action_dim") != 7:
            raise SimplerEvaluationError(
                f"Octo action_dim must be 7, found {metadata.get('action_dim')}"
            )
        checkpoint = metadata.get("checkpoint")
        protocol = metadata.get("protocol")
        startup_preflight = metadata.get("startup_preflight")
        if not isinstance(checkpoint, Mapping):
            raise SimplerEvaluationError("Octo server checkpoint metadata must be a mapping")
        if not isinstance(protocol, Mapping):
            raise SimplerEvaluationError("Octo server protocol metadata must be a mapping")
        if not isinstance(startup_preflight, Mapping):
            raise SimplerEvaluationError(
                "Octo server startup_preflight metadata must be a mapping"
            )
        self._metadata = metadata
        self._checkpoint_report = dict(checkpoint)
        self._protocol = dict(protocol)
        self._startup_preflight = dict(startup_preflight)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def checkpoint_report(self) -> dict[str, Any]:
        return dict(self._checkpoint_report)

    @property
    def model_device(self) -> str:
        return str(self._metadata.get("device", "unknown"))

    def make_generator(self, seed: int) -> int:
        return int(self.client.reset_rng(int(seed)))

    def prepare_observation(
        self,
        image: Any,
        proprio: Any,
        instruction: str,
    ) -> dict[str, Any]:
        image_array = np.asarray(image)
        if (
            image_array.dtype != np.uint8
            or image_array.ndim != 3
            or image_array.shape[0] <= 0
            or image_array.shape[1] <= 0
            or image_array.shape[2] != 3
        ):
            raise SimplerEvaluationError(
                "Octo expects a uint8 HWC RGB image; "
                f"found {image_array.shape} {image_array.dtype}"
            )
        proprio_array = np.asarray(proprio, dtype=np.float32)
        if proprio_array.shape != (8,):
            raise SimplerEvaluationError(
                f"Octo expects raw Bridge proprio shape (8,), found {proprio_array.shape}"
            )
        text = str(instruction).strip()
        if not text:
            raise SimplerEvaluationError("Octo instruction must be non-empty")
        return {
            "image": image_array,
            "proprio": proprio_array,
            "instruction": text,
        }

    def describe_observation(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source_image_shape": list(np.asarray(prepared["image"]).shape),
            "model_image_shape": [1, 1, 3, 256, 256],
            "model_proprio_shape": [1, 1, 8],
            "observation_tokenizers": ["primary"],
        }

    def predict_actions(self, prepared: Mapping[str, Any], *, generator: Any) -> np.ndarray:
        del generator
        actions = np.asarray(
            self.client.infer(
                prepared["image"],
                prepared["proprio"],
                prepared["instruction"],
            ),
            dtype=np.float32,
        )
        if actions.shape != (1, 8, 7):
            raise SimplerEvaluationError(
                f"Octo server returned actions with shape {actions.shape}; expected (1, 8, 7)"
            )
        if not np.all(np.isfinite(actions)):
            raise SimplerEvaluationError("Octo server returned NaN or infinite actions")
        return actions

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            **self._protocol,
            "ipc_protocol_version": IPC_PROTOCOL_VERSION,
            "startup_preflight": dict(self._startup_preflight),
        }
