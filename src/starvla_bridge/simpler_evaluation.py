"""SimplerEnv-side adapter for the remote StarVLA policy service."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from simpler_bridge.evaluation import SimplerEvaluationError


class StarVLARemotePolicy:
    policy_name = "Qwen3VL-GR00T-Bridge-RT-1"
    gripper_threshold = 0.5

    def __init__(self, client: Any):
        self.client = client
        metadata = dict(client.metadata())
        if metadata.get("protocol_version") != 2:
            raise SimplerEvaluationError(
                f"StarVLA protocol_version must be 2, found {metadata.get('protocol_version')}"
            )
        if metadata.get("native_action_chunk_size") != 16:
            raise SimplerEvaluationError(
                "StarVLA native_action_chunk_size must be 16, found "
                f"{metadata.get('native_action_chunk_size')}"
            )
        if "oxe_bridge" not in metadata.get("available_unnorm_keys", []):
            raise SimplerEvaluationError("StarVLA server does not expose oxe_bridge actions")
        self._metadata = metadata

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def make_generator(self, seed: int) -> int:
        return int(self.client.reset_rng(int(seed)))

    def prepare_observation(
        self,
        image: Any,
        proprio: Any,
        instruction: str,
    ) -> dict[str, Any]:
        del proprio
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise SimplerEvaluationError(
                f"StarVLA expects a uint8 HWC RGB image, found {array.shape} {array.dtype}"
            )
        try:
            import cv2
        except ImportError as error:
            raise SimplerEvaluationError("OpenCV is required for StarVLA image resizing") from error
        resized = cv2.resize(array, (224, 224), interpolation=cv2.INTER_AREA)
        return {"image": resized, "instruction": str(instruction)}

    def describe_observation(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "model_image_shape": list(np.asarray(prepared["image"]).shape),
            "observation_tokenizers": ["image_0"],
            "uses_proprio": False,
        }

    def predict_actions(self, prepared: Mapping[str, Any], *, generator: Any) -> np.ndarray:
        del generator
        actions = np.asarray(
            self.client.infer(prepared["image"], prepared["instruction"]),
            dtype=np.float32,
        )
        if actions.shape != (1, 16, 7) or not np.all(np.isfinite(actions)):
            raise SimplerEvaluationError(
                f"StarVLA server returned invalid actions with shape {actions.shape}"
            )
        return actions

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "native_action_chunk_size": 16,
            "model": self._metadata.get("model", self.policy_name),
            "starvla_source_commit": self._metadata.get("starvla_source_commit"),
            "checkpoint_tensor_count": self._metadata.get("checkpoint_tensor_count"),
            "unnorm_key": "oxe_bridge",
            "uses_proprio": False,
            "terminate_episode": 0,
            "model_runtime": dict(self._metadata.get("runtime", {})),
            "startup_preflight": dict(
                self._metadata.get("startup_preflight", {})
            ),
        }
