"""Self-contained checkpoint contract for Octo-small Bridge V2 runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BRIDGE_V2_NORMALIZATION_CONTRACT
from .normalization import BridgeV2NormalizationStatistics


CHECKPOINT_FORMAT = "octo-small-bridge-checkpoint-v2"


class BridgeCheckpointContractError(RuntimeError):
    """Raised when a Bridge checkpoint predates or violates the V2 contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BridgeCheckpointContract:
    normalization_path: Path
    selection_sha256: str

    def write(self, directory: str | Path) -> None:
        target = Path(directory)
        source = Path(self.normalization_path)
        if not source.is_file():
            raise BridgeCheckpointContractError(
                f"Bridge normalization file does not exist: {source}"
            )
        BridgeV2NormalizationStatistics.load(source)
        normalization = target / "normalization.json"
        shutil.copyfile(source, normalization)
        manifest = {
            "format": CHECKPOINT_FORMAT,
            "normalization_contract": BRIDGE_V2_NORMALIZATION_CONTRACT,
            "normalization_sha256": _sha256(normalization),
            "selection_sha256": str(self.selection_sha256),
        }
        with (target / "checkpoint_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def validate(self, directory: str | Path) -> Path:
        return validate_bridge_checkpoint(
            directory,
            expected_selection_sha256=self.selection_sha256,
            expected_normalization_path=self.normalization_path,
        )


def validate_bridge_checkpoint(
    checkpoint: str | Path,
    *,
    expected_selection_sha256: str | None = None,
    expected_normalization_path: str | Path | None = None,
) -> Path:
    root = Path(checkpoint).expanduser().resolve()
    manifest_path = root / "checkpoint_manifest.json"
    normalization_path = root / "normalization.json"
    if not manifest_path.is_file():
        raise BridgeCheckpointContractError(
            f"Bridge checkpoint is missing checkpoint_manifest.json: {root}"
        )
    if not normalization_path.is_file():
        raise BridgeCheckpointContractError(
            f"Bridge checkpoint is missing normalization.json: {root}"
        )
    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeCheckpointContractError(
            f"Could not read Bridge checkpoint manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise BridgeCheckpointContractError("Bridge checkpoint manifest must be a mapping")
    if manifest.get("format") != CHECKPOINT_FORMAT:
        raise BridgeCheckpointContractError(
            f"Bridge checkpoint format must be {CHECKPOINT_FORMAT!r}"
        )
    if manifest.get("normalization_contract") != BRIDGE_V2_NORMALIZATION_CONTRACT:
        raise BridgeCheckpointContractError(
            "Bridge checkpoint normalization contract must be "
            f"{BRIDGE_V2_NORMALIZATION_CONTRACT!r}"
        )
    expected_hash = str(manifest.get("normalization_sha256", ""))
    if not expected_hash or _sha256(normalization_path) != expected_hash:
        raise BridgeCheckpointContractError(
            "Bridge checkpoint normalization SHA-256 does not match its manifest"
        )
    try:
        BridgeV2NormalizationStatistics.load(normalization_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BridgeCheckpointContractError(
            f"Invalid Bridge checkpoint normalization: {error}"
        ) from error
    if (
        expected_selection_sha256 is not None
        and manifest.get("selection_sha256") != expected_selection_sha256
    ):
        raise BridgeCheckpointContractError(
            "Bridge checkpoint data selection differs from the current run"
        )
    if expected_normalization_path is not None:
        expected_path = Path(expected_normalization_path)
        if not expected_path.is_file() or _sha256(expected_path) != expected_hash:
            raise BridgeCheckpointContractError(
                "Bridge checkpoint normalization differs from the current run"
            )
    return normalization_path


def build_checkpoint_contract(
    config: dict[str, Any],
    paths: dict[str, Path],
    training_data: Any,
) -> BridgeCheckpointContract:
    contract = config["data"].get("normalization_contract")
    if contract != BRIDGE_V2_NORMALIZATION_CONTRACT:
        raise BridgeCheckpointContractError(
            "Cannot build a Bridge checkpoint without the V2 normalization contract"
        )
    normalization_path = Path(training_data.normalization_path).resolve()
    if normalization_path != Path(paths["normalization"]).resolve():
        raise BridgeCheckpointContractError(
            "Bridge training data normalization path differs from the configured output"
        )
    return BridgeCheckpointContract(
        normalization_path=normalization_path,
        selection_sha256=str(training_data.selection_sha256),
    )
