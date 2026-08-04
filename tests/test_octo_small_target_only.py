import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import octo_small_libero.preflight as preflight
import octo_small_libero.training as training


class _TargetSelection:
    selection_sha256 = "target-selection"
    task_indices = tuple(range(10))
    episode_indices = ()
    episodes = 50
    frames = 500

    def as_manifest(self):
        return {"enabled": True, "mode": "all", "episodes": 50, "frames": 500}


class _Metadata:
    def __init__(self, root):
        self.root = Path(root)
        self.info = {
            "total_episodes": 50 if self.root.name == "libero10_5" else 90,
            "total_frames": 500 if self.root.name == "libero10_5" else 900,
        }
        self.episodes = []

    def metadata_sha256(self):
        return f"metadata-{self.root.name}"


def _config(*, target_only):
    return {
        "data": {
            "target_only": target_only,
            "target_dataset": "libero10_5",
            "prior_dataset": "libero90",
            "sample_weights": [3.0, 1.0],
            "prior_selection": {
                "scores": "/missing/prior.csv",
                "top_percent": None,
                "prefiltered": False,
            },
            "action_normalization_mask": [True] * 6 + [False],
            "window_size": 1,
            "action_horizon": 8,
        },
        "model": {"required_observation_tokenizers": ["primary", "wrist"]},
        "train": {
            "gpu_count": 4,
            "gpu_ids": [0, 1, 2, 3],
            "batch_size": 128,
            "micro_batch_size_per_gpu": 8,
            "gradient_accumulation_steps": 4,
            "precision": "bf16",
        },
    }


def _paths(tmp_path, *, target_only):
    target = tmp_path / "libero10_5"
    prior = tmp_path / "missing-libero90"
    statistics_root = target if target_only else prior
    statistics = statistics_root / "meta" / "stats.json"
    statistics.parent.mkdir(parents=True)
    statistics.write_text('{"statistics": true}\n', encoding="utf-8")
    return {
        "model": tmp_path / "model",
        "lerobot": tmp_path,
        "target_dataset": target,
        "prior_dataset": prior,
        "statistics": statistics,
        "prior_scores": tmp_path / "missing-prior.csv",
    }


def test_target_only_dataset_manifest_omits_prior_and_records_target_normalization(
    tmp_path,
    monkeypatch,
):
    config = _config(target_only=True)
    paths = _paths(tmp_path, target_only=True)
    statistics = {"num_trajectories": 50, "num_transitions": 500}
    loaded_roots = []

    monkeypatch.setattr(training, "LeRobotV2Metadata", _Metadata)
    monkeypatch.setattr(
        training,
        "load_lerobot_statistics",
        lambda root: loaded_roots.append(Path(root)) or statistics,
    )

    manifest = training.build_dataset_manifest(
        config,
        paths,
        target_selection=_TargetSelection(),
    )

    assert loaded_roots == [paths["target_dataset"]]
    assert manifest["training_mode"] == "target_only"
    assert set(manifest["datasets"]) == {"libero10_5"}
    assert manifest["datasets"]["libero10_5"]["sample_weight"] == 1.0
    assert manifest["normalization"] == {
        "source_dataset": "libero10_5",
        "path": str(paths["statistics"]),
        "sha256": hashlib.sha256(paths["statistics"].read_bytes()).hexdigest(),
        "trajectories": 50,
        "transitions": 500,
    }
    assert "prior_statistics_path" not in manifest


def test_mixed_dataset_manifest_keeps_prior_fields_and_adds_mode(tmp_path, monkeypatch):
    config = _config(target_only=False)
    paths = _paths(tmp_path, target_only=False)
    statistics = {"num_trajectories": 90, "num_transitions": 900}

    monkeypatch.setattr(training, "LeRobotV2Metadata", _Metadata)
    monkeypatch.setattr(
        training,
        "load_lerobot_statistics",
        lambda root: statistics,
    )

    manifest = training.build_dataset_manifest(
        config,
        paths,
        target_selection=_TargetSelection(),
    )

    assert manifest["training_mode"] == "mixed"
    assert set(manifest["datasets"]) == {"libero10_5", "libero90"}
    assert manifest["datasets"]["libero10_5"]["sample_weight"] == 0.75
    assert manifest["datasets"]["libero90"]["sample_weight"] == 0.25
    assert manifest["normalization"]["source_dataset"] == "libero90"
    assert manifest["prior_statistics_path"] == str(paths["statistics"])


def test_target_only_preflight_does_not_inspect_prior_or_scores(tmp_path, monkeypatch):
    config = _config(target_only=True)
    paths = _paths(tmp_path, target_only=True)
    target_report = {"episodes": 50, "tasks": 10, "path": str(paths["target_dataset"])}
    inspected = []
    loaded_roots = []

    monkeypatch.setattr(preflight, "inspect_octo_checkpoint", lambda _path: {"ok": True})
    monkeypatch.setattr(
        preflight,
        "load_lerobot_statistics",
        lambda root: loaded_roots.append(Path(root))
        or {"num_trajectories": 50, "num_transitions": 500},
    )

    def inspect_dataset(root):
        inspected.append(Path(root))
        if Path(root) != paths["target_dataset"]:
            pytest.fail("target-only inspected prior dataset")
        return target_report

    monkeypatch.setattr(preflight, "_inspect_lerobot_dataset", inspect_dataset)
    monkeypatch.setattr(
        preflight,
        "resolve_target_task_selection",
        lambda *_args, **_kwargs: _TargetSelection(),
    )
    monkeypatch.setattr(preflight, "LeRobotV2Metadata", _Metadata)

    import octo_small_libero.selection as selection

    monkeypatch.setattr(
        selection,
        "resolve_prior_selection",
        lambda *_args, **_kwargs: pytest.fail("target-only resolved prior scores"),
    )

    torch_module = ModuleType("torch")
    torch_module.__version__ = "test"
    torch_module.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 4,
        is_bf16_supported=lambda: True,
    )
    transformers_module = ModuleType("transformers")
    transformers_module.__version__ = "test"
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    report = preflight.run_preflight(config, paths)

    assert inspected == [paths["target_dataset"]]
    assert loaded_roots == [paths["target_dataset"]]
    assert "prior" not in report["lerobot"]
    assert report["lerobot"]["sample_weights"] == [1.0]
    assert report["normalization"]["source_dataset"] == "libero10_5"
