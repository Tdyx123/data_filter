from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pytest


DATASET = Path("/data/dwb/datasets/LIBERO_lerobot/libero10_5")


def _identity_transform(image, _rng):
    return image


def test_qwen_libero_sample_reads_only_primary_image_and_raw_action_window():
    if not DATASET.is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")
    from qwen3_vl_groot.libero_data import QwenLiberoFrameDataset

    dataset = QwenLiberoFrameDataset(
        DATASET,
        dataset_name="libero10_5",
        action_horizon=8,
        train=False,
        seed=42,
        episode_cache_size=1,
        transform=_identity_transform,
    )

    sample = dataset[0]

    assert sample["image"].shape == (128, 128, 3)
    assert sample["image"].dtype == np.uint8
    assert "image_wrist" not in sample
    assert sample["state"].shape == (8,)
    assert sample["actions"].shape == (8, 7)
    np.testing.assert_array_equal(sample["action_mask"], np.ones(8, dtype=np.float32))
    assert sample["instruction"] == (
        "put both the alphabet soup and the tomato sauce in the basket"
    )


def test_qwen_libero_tail_action_window_is_zero_padded_and_masked():
    if not DATASET.is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")
    from qwen3_vl_groot.libero_data import QwenLiberoFrameDataset

    dataset = QwenLiberoFrameDataset(
        DATASET,
        dataset_name="libero10_5",
        action_horizon=8,
        train=False,
        seed=42,
        episode_cache_size=1,
        transform=_identity_transform,
    )

    sample = dataset[316]

    np.testing.assert_array_equal(
        sample["action_mask"],
        np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(sample["actions"][1:], np.zeros((7, 7), dtype=np.float32))


def test_target_only_source_resolution_never_opens_prior_dataset(tmp_path):
    if not DATASET.is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")
    from qwen3_vl_groot.libero_data import resolve_libero_sources

    config = {
        "data": {
            "target_dataset": "libero10_5",
            "prior_dataset": "libero90",
            "target_all_tasks": True,
            "target_task_index": None,
            "target_only": True,
            "sample_weights": [1.0, 1.0],
            "action_horizon": 8,
            "prior_selection": {
                "scores": str(tmp_path / "missing-scores.csv"),
                "top_percent": None,
                "prefiltered": True,
                "relcore_manifest": None,
                "quality_filter_scores": None,
            },
        }
    }
    paths = {
        "target_dataset": DATASET,
        "prior_dataset": tmp_path / "missing-prior",
        "prior_scores": tmp_path / "missing-scores.csv",
    }

    sources = resolve_libero_sources(config, paths)

    assert sources.prior_selection is None
    assert sources.training_mode == "target_only"
    assert sources.sample_weights == (1.0,)
    assert sources.normalization_root == DATASET.resolve()
    assert sources.target_selection.episodes == 50


def test_full_prior_mixed_training_builds_two_sources(tmp_path, monkeypatch):
    prior_dataset = Path("/data/dwb/datasets/LIBERO_lerobot/libero90")
    if not DATASET.is_dir() or not prior_dataset.is_dir():
        pytest.skip("LIBERO LeRobot data is not mounted")
    torch_module = ModuleType("torch")
    torch_utils = ModuleType("torch.utils")
    torch_data = ModuleType("torch.utils.data")

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

    torch_data.DataLoader = FakeDataLoader
    torch_utils.data = torch_data
    torch_module.utils = torch_utils
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data)
    from qwen3_vl_groot.libero_data import make_libero_training_data

    config = {
        "data": {
            "target_dataset": "libero10_5",
            "prior_dataset": "libero90",
            "target_all_tasks": True,
            "target_task_index": None,
            "target_only": False,
            "sample_weights": [1.0, 1.0],
            "action_horizon": 8,
            "episode_cache_size": 1,
            "num_workers": 0,
            "prefetch_factor": 2,
            "prior_selection": {
                "scores": str(tmp_path / "unused.csv"),
                "top_percent": None,
                "prefiltered": False,
                "relcore_manifest": None,
                "quality_filter_scores": None,
            },
        },
        "train": {
            "seed": 42,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 1,
        },
    }
    paths = {
        "target_dataset": DATASET,
        "prior_dataset": prior_dataset,
        "prior_scores": tmp_path / "unused.csv",
    }

    training = make_libero_training_data(
        config,
        paths,
        rank=0,
        world_size=4,
        transform=_identity_transform,
    )

    assert training.sources.training_mode == "mixed"
    assert training.sources.prior_selection is None
    assert training.dataset.source_sizes[0] == 14_144
    assert training.dataset.source_sizes[1] > training.dataset.source_sizes[0]


def test_target_only_manifest_records_selection_normalization_and_learning_rates(tmp_path):
    if not DATASET.is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")
    from qwen3_vl_groot.libero_data import (
        build_libero_dataset_manifest,
        resolve_libero_sources,
    )

    normalization = tmp_path / "normalization.json"
    normalization.write_text('{"contract": "q01_q99"}\n', encoding="utf-8")
    config = {
        "data": {
            "target_dataset": "libero10_5",
            "prior_dataset": "libero90",
            "target_all_tasks": True,
            "target_task_index": None,
            "target_only": True,
            "sample_weights": [1.0, 1.0],
            "action_horizon": 8,
            "prior_selection": {
                "scores": str(tmp_path / "missing.csv"),
                "top_percent": None,
                "prefiltered": True,
                "relcore_manifest": None,
                "quality_filter_scores": None,
            },
        },
        "train": {
            "lora_learning_rate": 5e-6,
            "head_learning_rate": 2e-4,
        },
    }
    paths = {
        "target_dataset": DATASET,
        "prior_dataset": tmp_path / "missing-prior",
        "prior_scores": tmp_path / "missing.csv",
    }
    sources = resolve_libero_sources(config, paths)

    manifest = build_libero_dataset_manifest(
        config,
        paths,
        sources,
        normalization_path=normalization,
    )

    assert manifest["training_mode"] == "target_only"
    assert manifest["sample_weights"] == [1.0]
    assert manifest["target"]["selection"]["episodes"] == 50
    assert manifest["prior"] is None
    assert manifest["normalization"]["dataset_path"] == str(DATASET.resolve())
    assert len(manifest["normalization"]["sha256"]) == 64
    assert manifest["learning_rates"] == {
        "lora": 5e-6,
        "action_head": 2e-4,
    }
    assert len(manifest["training_selection_sha256"]) == 64


@pytest.mark.real_data
def test_libero_quantiles_are_computed_from_full_normalization_dataset(tmp_path):
    if not DATASET.is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")
    from qwen3_vl_groot.libero_data import compute_libero_quantile_stats

    cache = tmp_path / "normalization.json"
    stats = compute_libero_quantile_stats(DATASET, cache, epsilon=1e-6)

    assert stats.state_q01.shape == (8,)
    assert stats.action_q99.shape == (7,)
    assert cache.is_file()
    assert '"training_frames": 14144' in cache.read_text(encoding="utf-8")


def test_target_only_training_data_builds_one_source_and_global_sampler(
    tmp_path, monkeypatch
):
    if not DATASET.is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")
    torch_module = ModuleType("torch")
    torch_utils = ModuleType("torch.utils")
    torch_data = ModuleType("torch.utils.data")

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

    torch_data.DataLoader = FakeDataLoader
    torch_utils.data = torch_data
    torch_module.utils = torch_utils
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data)
    from qwen3_vl_groot.libero_data import make_libero_training_data

    config = {
        "data": {
            "target_dataset": "libero10_5",
            "prior_dataset": "libero90",
            "target_all_tasks": True,
            "target_task_index": None,
            "target_only": True,
            "sample_weights": [1.0, 1.0],
            "action_horizon": 8,
            "episode_cache_size": 1,
            "num_workers": 0,
            "prefetch_factor": 2,
            "prior_selection": {
                "scores": str(tmp_path / "missing.csv"),
                "top_percent": None,
                "prefiltered": True,
                "relcore_manifest": None,
                "quality_filter_scores": None,
            },
        },
        "train": {
            "seed": 42,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 3,
            "max_steps": 2,
        },
    }
    paths = {
        "target_dataset": DATASET,
        "prior_dataset": tmp_path / "missing-prior",
        "prior_scores": tmp_path / "missing.csv",
    }

    training = make_libero_training_data(
        config,
        paths,
        rank=0,
        world_size=4,
        transform=_identity_transform,
    )

    assert training.dataset.source_sizes == (14_144,)
    assert len(training.batch_sampler) == 6
    assert next(iter(training.batch_sampler))[0].source == 0
    assert training.dataloader.kwargs["num_workers"] == 0
