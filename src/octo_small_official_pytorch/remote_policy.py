from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from simpler_bridge.evaluation import SimplerEvaluationError, select_first_action

from .action_ensemble import OfficialTemporalActionEnsembler
from .ipc import IPC_PROTOCOL_NAME, IPC_PROTOCOL_VERSION


ACTION_POSTPROCESSING_MODES = ("octo_temporal_ensemble_v1", "first_action")


class OctoOfficialRemotePolicy:
    policy_name = "Octo-small official-semantics PyTorch baseline"
    gripper_threshold = 0.5

    def __init__(
        self,
        client: Any,
        *,
        action_postprocessing: str = "octo_temporal_ensemble_v1",
    ) -> None:
        if action_postprocessing not in ACTION_POSTPROCESSING_MODES:
            raise SimplerEvaluationError(
                "action_postprocessing must be octo_temporal_ensemble_v1 or first_action"
            )
        metadata = dict(client.metadata())
        expected = {
            "protocol_version": IPC_PROTOCOL_VERSION,
            "protocol_name": IPC_PROTOCOL_NAME,
            "native_action_chunk_size": 4,
            "action_dim": 7,
            "image_history_horizon": 2,
            "use_proprio": False,
        }
        mismatches = {
            key: {"actual": metadata.get(key), "expected": wanted}
            for key, wanted in expected.items()
            if metadata.get(key) != wanted
        }
        if mismatches:
            raise SimplerEvaluationError(
                f"Official Octo server metadata is incompatible: {mismatches}"
            )
        checkpoint = metadata.get("checkpoint")
        protocol = metadata.get("protocol")
        preflight = metadata.get("startup_preflight")
        if not isinstance(checkpoint, Mapping):
            raise SimplerEvaluationError("Server checkpoint metadata must be a mapping")
        if not isinstance(protocol, Mapping):
            raise SimplerEvaluationError("Server protocol metadata must be a mapping")
        if protocol.get("model_action_gripper") != "continuous_model_prediction":
            raise SimplerEvaluationError(
                "Server gripper semantics must be continuous_model_prediction"
            )
        if not isinstance(preflight, Mapping):
            raise SimplerEvaluationError("Server startup_preflight must be a mapping")
        self.client = client
        self.action_postprocessing = action_postprocessing
        self._ensemble = OfficialTemporalActionEnsembler()
        self._metadata = metadata
        self._checkpoint = dict(checkpoint)
        self._protocol = dict(protocol)
        self._preflight = dict(preflight)

    @property
    def checkpoint_report(self) -> dict[str, Any]:
        return dict(self._checkpoint)

    @property
    def model_device(self) -> str:
        return str(self._metadata.get("device", "unknown"))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def make_generator(self, seed: int) -> int:
        return int(self.client.reset_rng(int(seed)))

    def begin_episode(self, instruction: str) -> None:
        text = str(instruction).strip()
        if not text:
            raise SimplerEvaluationError("Official Octo instruction must be non-empty")
        self._ensemble.reset()
        self.client.begin_episode(text)

    def prepare_observation(self, image: Any, proprio: Any, instruction: str) -> dict[str, Any]:
        del proprio
        image_array = np.asarray(image)
        if (
            image_array.dtype != np.uint8
            or image_array.ndim != 3
            or image_array.shape[0] <= 0
            or image_array.shape[1] <= 0
            or image_array.shape[2] != 3
        ):
            raise SimplerEvaluationError("Official Octo expects a uint8 HWC RGB image")
        text = str(instruction).strip()
        if not text:
            raise SimplerEvaluationError("Official Octo instruction must be non-empty")
        return {"image": image_array, "instruction": text}

    def describe_observation(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source_image_shape": list(np.asarray(prepared["image"]).shape),
            "model_image_shape": [1, "1_or_2", 3, 256, 256],
            "image_history_horizon": 2,
            "use_proprio": False,
            "observation_tokenizers": ["primary"],
        }

    def predict_actions(self, prepared: Mapping[str, Any], *, generator: Any) -> np.ndarray:
        del generator
        if "proprio" in prepared:
            raise SimplerEvaluationError("Official IPC request must not contain proprio")
        actions = np.asarray(
            self.client.infer(prepared["image"], prepared["instruction"]),
            dtype=np.float32,
        )
        if actions.shape != (1, 4, 7) or not np.all(np.isfinite(actions)):
            raise SimplerEvaluationError(
                f"Official Octo server returned invalid actions {actions.shape}"
            )
        return actions

    def select_action(self, actions: np.ndarray) -> np.ndarray:
        if self.action_postprocessing == "first_action":
            return select_first_action(actions)
        return self._ensemble.select_action(actions)

    def protocol_metadata(self) -> dict[str, Any]:
        ensemble = self.action_postprocessing == "octo_temporal_ensemble_v1"
        return {
            **self._protocol,
            "model_runtime": dict(self._metadata.get("model_runtime", {})),
            "rng": dict(self._metadata.get("rng", self._protocol.get("rng", {}))),
            "action_postprocessing": self.action_postprocessing,
            "temporal_ensemble_prediction_horizon": 4 if ensemble else None,
            "temporal_ensemble_temperature": 0.0 if ensemble else None,
            "gripper_binarization": (
                "after_action_ensemble" if ensemble else "at_environment_conversion"
            ),
            "ipc_protocol_version": IPC_PROTOCOL_VERSION,
            "ipc_protocol_name": IPC_PROTOCOL_NAME,
            "startup_preflight": dict(self._preflight),
        }
