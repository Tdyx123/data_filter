"""Bridge-specific manifest and wrapper around the shared Octo trainer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .data import BridgeTrainingData, make_training_dataset
from .checkpoint_contract import build_checkpoint_contract


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_manifest(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    training_data: BridgeTrainingData | Any,
) -> dict[str, Any]:
    """Describe the exact filtered Bridge source used by this run."""

    adapter = training_data.dataset.adapter
    summary = adapter.dataset_summary()
    normalization_path = Path(training_data.normalization_path)
    return {
        "dataset": config["data"]["dataset_name"],
        "path": str(Path(paths["dataset"]).resolve()),
        "metadata_sha256": adapter.fingerprint(),
        "normalization": {
            "contract": config["data"]["normalization_contract"],
            "path": str(normalization_path.resolve()),
            "sha256": _file_sha256(normalization_path),
        },
        "selection_sha256": training_data.selection_sha256,
        "source_episodes": summary["source_episodes"],
        "retained_episodes": summary["retained_episodes"],
        "excluded_episodes": summary["excluded_episodes"],
        "excluded_empty_task_episodes": summary[
            "excluded_empty_task_episodes"
        ],
        "retained_frames": sum(record.length for record in adapter.episodes()),
        "empty_task_policy": "exclude",
        "image_observations": list(adapter.image_observation_keys),
        "vector_observations": list(adapter.vector_observation_keys),
        "action_key": adapter.action_key,
    }


def train(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    resume: str | None = None,
) -> None:
    """Run Bridge fine-tuning with primary-camera-only Octo tokenization."""

    from octo_small_libero import training as shared_training

    shared_training.train(
        config,
        paths,
        resume=resume,
        training_data_builder=make_training_dataset,
        dataset_manifest_builder=build_dataset_manifest,
        checkpoint_contract_builder=build_checkpoint_contract,
        observation_tokenizers=("primary",),
    )
