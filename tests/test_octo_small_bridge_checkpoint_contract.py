from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


CONTRACT = "bridge_v2_q99_binary_v1"


def _write_normalization(path: Path) -> None:
    from octo_small_bridge.normalization import BridgeV2NormalizationStatistics

    BridgeV2NormalizationStatistics(
        state_q01=np.zeros(8, dtype=np.float32),
        state_q99=np.ones(8, dtype=np.float32),
        action_q01=np.zeros(7, dtype=np.float32),
        action_q99=np.ones(7, dtype=np.float32),
        metadata_sha256="a" * 64,
        retained_episodes=3,
        retained_frames=12,
    ).save(path)


def test_bridge_checkpoint_contract_writes_and_validates_self_contained_statistics(
    tmp_path: Path,
):
    from octo_small_bridge.checkpoint_contract import (
        BridgeCheckpointContract,
        validate_bridge_checkpoint,
    )

    source = tmp_path / "run" / "normalization.json"
    _write_normalization(source)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    contract = BridgeCheckpointContract(
        normalization_path=source,
        selection_sha256="b" * 64,
    )

    contract.write(checkpoint)
    validated = validate_bridge_checkpoint(checkpoint)

    manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == "octo-small-bridge-checkpoint-v2"
    assert manifest["normalization_contract"] == CONTRACT
    assert manifest["selection_sha256"] == "b" * 64
    assert validated == checkpoint / "normalization.json"


def test_bridge_checkpoint_contract_rejects_old_and_tampered_checkpoints(tmp_path: Path):
    from octo_small_bridge.checkpoint_contract import (
        BridgeCheckpointContractError,
        BridgeCheckpointContract,
        validate_bridge_checkpoint,
    )

    old = tmp_path / "old"
    old.mkdir()
    with pytest.raises(BridgeCheckpointContractError, match="checkpoint_manifest"):
        validate_bridge_checkpoint(old)

    source = tmp_path / "normalization.json"
    _write_normalization(source)
    checkpoint = tmp_path / "new"
    checkpoint.mkdir()
    BridgeCheckpointContract(source, "c" * 64).write(checkpoint)
    (checkpoint / "normalization.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(BridgeCheckpointContractError, match="SHA-256"):
        validate_bridge_checkpoint(checkpoint)


def test_shared_checkpoint_writer_invokes_optional_bridge_contract(tmp_path: Path):
    from octo_small_libero.training import _save_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    sampler = SimpleNamespace(state_dict=lambda: {"position": 0})
    calls = []

    class ContractWriter:
        def write(self, directory):
            calls.append(Path(directory))
            (Path(directory) / "checkpoint_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )

    checkpoint = _save_checkpoint(
        output=tmp_path,
        step=1,
        mean_train_loss=1.0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config={"train": {"seed": 1}},
        checkpoint_contract=ContractWriter(),
    )

    assert calls == [tmp_path / "checkpoints" / ".step-00000001.tmp"]
    assert (checkpoint / "checkpoint_manifest.json").is_file()


def test_bridge_dataset_manifest_records_v2_normalization(tmp_path: Path):
    from octo_small_bridge.training import build_dataset_manifest

    normalization = tmp_path / "normalization.json"
    _write_normalization(normalization)
    records = [SimpleNamespace(length=4), SimpleNamespace(length=8)]
    adapter = SimpleNamespace(
        dataset_summary=lambda: {
            "source_episodes": 4,
            "retained_episodes": 2,
            "excluded_episodes": 2,
            "excluded_empty_task_episodes": 2,
        },
        fingerprint=lambda: "e" * 64,
        episodes=lambda: records,
        image_observation_keys=("observation.images.image_0",),
        vector_observation_keys=("observation.state",),
        action_key="action",
    )
    training_data = SimpleNamespace(
        dataset=SimpleNamespace(adapter=adapter),
        selection_sha256="f" * 64,
        normalization_path=normalization,
    )

    manifest = build_dataset_manifest(
        {
            "data": {
                "dataset_name": "bridge_orig_1.0.0",
                "normalization_contract": CONTRACT,
            }
        },
        {"dataset": tmp_path / "dataset", "normalization": normalization},
        training_data=training_data,
    )

    assert manifest["normalization"]["contract"] == CONTRACT
    assert manifest["normalization"]["path"] == str(normalization.resolve())
    assert len(manifest["normalization"]["sha256"]) == 64
    assert manifest["retained_frames"] == 12
