from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_base_artifact(root: Path, monkeypatch: pytest.MonkeyPatch):
    from octo_small_official_pytorch import checkpoint as checkpoint_module
    from octo_small_official_pytorch.checkpoint import (
        OFFICIAL_BRIDGE_ACTION_MASK,
        OFFICIAL_BRIDGE_ACTION_MEAN,
        OFFICIAL_BRIDGE_ACTION_STD,
        OFFICIAL_FORMAT,
        OFFICIAL_OCTO_COMMIT,
        OFFICIAL_SOURCE_STEP,
        OfficialCheckpointReport,
    )
    from octo_small_official_pytorch.model import OctoOfficialConfig

    root.mkdir()
    (root / "model_config.json").write_text(
        json.dumps(OctoOfficialConfig().to_dict()), encoding="utf-8"
    )
    (root / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "bridge_dataset": {
                    "action": {
                        "mean": list(OFFICIAL_BRIDGE_ACTION_MEAN),
                        "std": list(OFFICIAL_BRIDGE_ACTION_STD),
                        "mask": list(OFFICIAL_BRIDGE_ACTION_MASK),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    text = root / "text_encoder"
    text.mkdir()
    (text / "config.json").write_text(
        json.dumps(
            {
                "model_type": "t5",
                "d_model": 768,
                "d_kv": 64,
                "d_ff": 3072,
                "num_layers": 12,
                "num_heads": 12,
                "relative_attention_num_buckets": 32,
                "vocab_size": 32128,
                "is_encoder_decoder": True,
                "dropout_rate": 0.1,
                "layer_norm_epsilon": 1e-6,
                "dense_act_fn": "relu",
                "is_gated_act": False,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    (text / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "T5Tokenizer",
                "model_max_length": 16,
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "extra_ids": 100,
            }
        ),
        encoding="utf-8",
    )
    (text / "tokenizer.json").write_text("{}", encoding="utf-8")
    (text / "spiece.model").write_bytes(b"fixture sentencepiece")
    monkeypatch.setattr(
        checkpoint_module,
        "OFFICIAL_T5_CONTENT_SHA256",
        {
            "text_encoder/tokenizer.json": _sha256(text / "tokenizer.json"),
            "text_encoder/spiece.model": _sha256(text / "spiece.model"),
        },
    )
    (root / "conversion_manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"base-weights-not-read")
    manifest = {
        "format": OFFICIAL_FORMAT,
        "source_step": OFFICIAL_SOURCE_STEP,
        "source_octo_commit": OFFICIAL_OCTO_COMMIT,
    }
    return OfficialCheckpointReport(
        path=root,
        weights_path=root / "model.safetensors",
        statistics_path=root / "dataset_statistics.json",
        manifest=manifest,
        config=OctoOfficialConfig(),
        weights_sha256=_sha256(root / "model.safetensors"),
        statistics_sha256=_sha256(root / "dataset_statistics.json"),
        text_artifact_sha256={
            f"text_encoder/{path.name}": _sha256(path) for path in text.iterdir()
        },
    )


def _write_finetuned_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selection_sha256: str = "b" * 64,
):
    import torch
    from safetensors.torch import save_file

    from octo_small_bridge.checkpoint_contract import OfficialFinetuneCheckpointContract

    base_report = _write_base_artifact(tmp_path / "base", monkeypatch)
    checkpoint = tmp_path / "checkpoints" / ".step-00000002.tmp"
    checkpoint.mkdir(parents=True)
    save_file({"weight": torch.zeros(2, 3)}, checkpoint / "model.safetensors")
    training_config = {
        "data": {"window_size": 2, "action_horizon": 4},
        "model": {"use_proprio": False},
        "train": {"max_steps": 20_000},
    }
    dataset_manifest = {
        "dataset": "bridge_orig_1.0.0",
        "selection_sha256": selection_sha256,
    }
    contract = OfficialFinetuneCheckpointContract(
        base_report=base_report,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        selection_sha256=selection_sha256,
    )
    contract.write(checkpoint)
    torch.save({"step": 2}, checkpoint / "training_state.pt")
    return checkpoint, contract


def test_finetune_checkpoint_is_self_contained_and_accepted_by_joint_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from octo_small_official_pytorch.checkpoint import (
        OFFICIAL_FINETUNE_FORMAT,
        validate_official_or_finetuned_checkpoint,
    )

    checkpoint, _contract = _write_finetuned_checkpoint(tmp_path, monkeypatch)

    report = validate_official_or_finetuned_checkpoint(checkpoint)
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())

    assert report.checkpoint_kind == "official_finetuned"
    assert manifest["format"] == OFFICIAL_FINETUNE_FORMAT
    assert manifest["training_step"] == 2
    assert manifest["training_contract"] == {
        "observation_tokenizers": ["primary"],
        "history_horizon": 2,
        "action_horizon": 4,
        "action_dim": 7,
        "use_proprio": False,
        "action_normalization": "bridge_dataset_mean_std",
        "gripper_transform": "trajectory_backward_binarize_minus_one_to_one",
        "diffusion_loss_readouts": "all_valid",
        "padded_readouts": "masked",
    }
    assert manifest["weights_sha256"] == _sha256(checkpoint / "model.safetensors")
    assert manifest["base_checkpoint"]["weights_sha256"] == _sha256(
        tmp_path / "base" / "model.safetensors"
    )
    for relative in (
        "model_config.json",
        "dataset_statistics.json",
        "finetune_config.json",
        "dataset_manifest.json",
        "training_state.pt",
        "text_encoder/config.json",
        "text_encoder/tokenizer.json",
        "text_encoder/tokenizer_config.json",
        "text_encoder/spiece.model",
    ):
        assert (checkpoint / relative).is_file()


def test_finetune_checkpoint_rejects_weight_tampering_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from octo_small_official_pytorch.checkpoint import (
        OfficialCheckpointError,
        validate_official_or_finetuned_checkpoint,
    )

    checkpoint, _contract = _write_finetuned_checkpoint(tmp_path, monkeypatch)
    with (checkpoint / "model.safetensors").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(OfficialCheckpointError, match="model.safetensors hash"):
        validate_official_or_finetuned_checkpoint(checkpoint)


def test_resume_contract_rejects_different_data_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from octo_small_bridge.checkpoint_contract import BridgeCheckpointContractError

    checkpoint, contract = _write_finetuned_checkpoint(tmp_path, monkeypatch)

    with pytest.raises(BridgeCheckpointContractError, match="data selection"):
        replace(contract, selection_sha256="c" * 64).validate(checkpoint)


def test_joint_validator_keeps_original_checkpoint_on_official_parity_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import octo_small_official_pytorch.checkpoint as checkpoint_module

    base_report = _write_base_artifact(tmp_path / "base", monkeypatch)
    monkeypatch.setattr(
        checkpoint_module,
        "validate_official_checkpoint",
        lambda path: base_report,
    )

    report = checkpoint_module.validate_official_or_finetuned_checkpoint(base_report.path)

    assert report is base_report
    assert report.checkpoint_kind == "official_parity"


def test_atomic_step_checkpoint_restores_one_of_four_rank_runtime_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from octo_small_bridge.checkpoint_contract import OfficialFinetuneCheckpointContract
    from octo_small_libero.training import _load_training_state, _save_checkpoint

    base_report = _write_base_artifact(tmp_path / "base", monkeypatch)
    selection = "d" * 64
    config = {"train": {"seed": 42}, "model": {"use_proprio": False}}
    contract = OfficialFinetuneCheckpointContract(
        base_report=base_report,
        training_config=config,
        dataset_manifest={"dataset": "bridge_orig_1.0.0", "selection_sha256": selection},
        selection_sha256=selection,
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    sampler = type(
        "Sampler",
        (),
        {
            "state_dict": lambda self: {"rank": 0},
            "load_state_dict": lambda self, state: setattr(self, "restored", state),
        },
    )()
    rank_states = [
        {
            "sampler": {"rank": rank},
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
        }
        for rank in range(4)
    ]

    checkpoint = _save_checkpoint(
        output=tmp_path / "run",
        step=2,
        mean_train_loss=0.5,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
        selection_signature=selection,
        rank_runtime_states=rank_states,
        checkpoint_contract=contract,
    )
    contract.validate(checkpoint)
    restored_step = _load_training_state(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        device=torch.device("cpu"),
        selection_signature=selection,
        rank=2,
        world_size=4,
    )

    assert restored_step == 2
    assert sampler.restored == {"rank": 2}
    assert (tmp_path / "run" / "checkpoints" / "latest.json").is_file()
    assert (tmp_path / "run" / "checkpoints" / "best.json").is_file()
