"""Self-contained official-semantics fine-tune checkpoint contract."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from octo_small_official_pytorch.checkpoint import (
    OFFICIAL_FINETUNE_FORMAT,
    OFFICIAL_FORMAT,
    TEXT_ARTIFACT_PATHS,
    OfficialCheckpointError,
    OfficialCheckpointReport,
    sha256_file,
    validate_official_checkpoint,
    validate_official_or_finetuned_checkpoint,
)


class BridgeCheckpointContractError(RuntimeError):
    """Raised when an official Bridge fine-tune checkpoint is incompatible."""


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _training_step(directory: Path) -> int:
    name = directory.name
    if name.startswith(".step-") and name.endswith(".tmp"):
        digits = name.removeprefix(".step-").removesuffix(".tmp")
    elif name.startswith("step-"):
        digits = name.removeprefix("step-")
    else:
        digits = ""
    if not digits.isdigit() or int(digits) <= 0:
        raise BridgeCheckpointContractError(
            f"Cannot derive a positive training step from checkpoint directory {directory}"
        )
    return int(digits)


def _tensor_shapes(path: Path) -> dict[str, list[int]]:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {
                key: list(handle.get_slice(key).get_shape()) for key in sorted(handle.keys())
            }
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise BridgeCheckpointContractError(
            f"Could not inspect fine-tuned model.safetensors: {error}"
        ) from error


@dataclass(frozen=True)
class OfficialFinetuneCheckpointContract:
    base_report: OfficialCheckpointReport
    training_config: dict[str, Any]
    dataset_manifest: dict[str, Any]
    selection_sha256: str

    def write(self, directory: str | Path) -> None:
        target = Path(directory)
        weights = target / "model.safetensors"
        if not weights.is_file():
            raise BridgeCheckpointContractError(
                f"Fine-tuned weights do not exist before manifest creation: {weights}"
            )
        if self.base_report.manifest.get("format") != OFFICIAL_FORMAT:
            raise BridgeCheckpointContractError(
                "Base checkpoint must be the strict official conversion artifact"
            )
        if self.dataset_manifest.get("selection_sha256") != self.selection_sha256:
            raise BridgeCheckpointContractError(
                "Dataset manifest selection differs from the training data selection"
            )

        shutil.copyfile(
            self.base_report.path / "model_config.json",
            target / "model_config.json",
        )
        shutil.copyfile(
            self.base_report.statistics_path,
            target / "dataset_statistics.json",
        )
        for relative_path in TEXT_ARTIFACT_PATHS:
            destination = target / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.base_report.path / relative_path, destination)
        _write_json(target / "finetune_config.json", _clean_config(self.training_config))
        _write_json(target / "dataset_manifest.json", self.dataset_manifest)

        base_manifest_path = self.base_report.path / "conversion_manifest.json"
        manifest = {
            "format": OFFICIAL_FINETUNE_FORMAT,
            "training_step": _training_step(target),
            "weights_sha256": sha256_file(weights),
            "model_config_sha256": sha256_file(target / "model_config.json"),
            "dataset_statistics_sha256": sha256_file(
                target / "dataset_statistics.json"
            ),
            "finetune_config_sha256": sha256_file(target / "finetune_config.json"),
            "dataset_manifest_sha256": sha256_file(target / "dataset_manifest.json"),
            "text_artifact_sha256": {
                relative_path: sha256_file(target / relative_path)
                for relative_path in TEXT_ARTIFACT_PATHS
            },
            "tensor_shapes": _tensor_shapes(weights),
            "data_selection_sha256": self.selection_sha256,
            "base_checkpoint": {
                "format": OFFICIAL_FORMAT,
                "source_step": int(self.base_report.manifest["source_step"]),
                "source_octo_commit": self.base_report.manifest["source_octo_commit"],
                "weights_sha256": self.base_report.weights_sha256,
                "conversion_manifest_sha256": sha256_file(base_manifest_path),
            },
            "training_contract": {
                "observation_tokenizers": ["primary"],
                "history_horizon": 2,
                "action_horizon": 4,
                "action_dim": 7,
                "use_proprio": False,
                "action_normalization": "bridge_dataset_mean_std",
                "gripper_transform": "trajectory_backward_binarize_minus_one_to_one",
                "diffusion_loss_readouts": "all_valid",
                "padded_readouts": "masked",
            },
        }
        _write_json(target / "checkpoint_manifest.json", manifest)

    def validate(self, directory: str | Path) -> Path:
        try:
            report = validate_official_or_finetuned_checkpoint(directory)
        except OfficialCheckpointError as error:
            raise BridgeCheckpointContractError(str(error)) from error
        if report.checkpoint_kind != "official_finetuned":
            raise BridgeCheckpointContractError(
                "Resume requires an official fine-tuned checkpoint, not base weights"
            )
        manifest = report.manifest
        if manifest.get("data_selection_sha256") != self.selection_sha256:
            raise BridgeCheckpointContractError(
                "Bridge checkpoint data selection differs from the current run"
            )
        if manifest.get("base_checkpoint", {}).get(
            "weights_sha256"
        ) != self.base_report.weights_sha256:
            raise BridgeCheckpointContractError(
                "Bridge checkpoint base weights differ from the current official artifact"
            )
        saved_config = json.loads(
            (report.path / "finetune_config.json").read_text(encoding="utf-8")
        )
        if saved_config != _clean_config(self.training_config):
            raise BridgeCheckpointContractError(
                "Bridge checkpoint training configuration differs from the current run"
            )
        return report.path


def build_checkpoint_contract(
    config: dict[str, Any],
    paths: dict[str, Path],
    training_data: Any,
) -> OfficialFinetuneCheckpointContract:
    from .training import build_dataset_manifest

    base_report = validate_official_checkpoint(paths["model"])
    statistics_path = Path(training_data.statistics_path).resolve()
    if statistics_path != base_report.statistics_path.resolve():
        raise BridgeCheckpointContractError(
            "Bridge training statistics do not come from the official base artifact"
        )
    dataset_manifest = build_dataset_manifest(
        config,
        paths,
        training_data=training_data,
    )
    return OfficialFinetuneCheckpointContract(
        base_report=base_report,
        training_config=config,
        dataset_manifest=dataset_manifest,
        selection_sha256=str(training_data.selection_sha256),
    )


def validate_bridge_checkpoint(
    checkpoint: str | Path,
    *,
    expected_selection_sha256: str | None = None,
) -> Path:
    """Compatibility entry point backed by the official fine-tune validator."""

    try:
        report = validate_official_or_finetuned_checkpoint(checkpoint)
    except OfficialCheckpointError as error:
        raise BridgeCheckpointContractError(str(error)) from error
    if report.checkpoint_kind != "official_finetuned":
        raise BridgeCheckpointContractError("Bridge resume checkpoint must be fine-tuned")
    if (
        expected_selection_sha256 is not None
        and report.manifest.get("data_selection_sha256") != expected_selection_sha256
    ):
        raise BridgeCheckpointContractError(
            "Bridge checkpoint data selection differs from the current run"
        )
    return report.path
