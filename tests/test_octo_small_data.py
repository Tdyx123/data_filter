import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import octo_small_libero.data as libero_data
from octo_small_libero.data import (
    BalancedDistributedBatchSampler,
    CombinedLeRobotDataset,
    DEFAULT_TARGET_TASK,
    select_target_demo_ids,
    standardize_demo_arrays,
    task_to_dataset_name,
    training_selection_sha256,
)


class _FakeGenerator:
    def __init__(self):
        self.random = random.Random()

    def manual_seed(self, seed):
        self.random.seed(seed)

    def get_state(self):
        return self.random.getstate()

    def set_state(self, state):
        self.random.setstate(state)


class _FakePermutation(list):
    def tolist(self):
        return list(self)


class _FakeTorch:
    Generator = _FakeGenerator
    cuda = SimpleNamespace(is_available=lambda: False)

    @staticmethod
    def randperm(size, *, generator):
        values = list(range(size))
        generator.random.shuffle(values)
        return _FakePermutation(values)


def test_standardization_matches_datamil_libero_contract():
    length = 2
    primary = np.zeros((length, 128, 128, 3), dtype=np.uint8)
    wrist = np.zeros_like(primary)
    primary[:, 0] = 7
    primary[:, -1] = 19
    wrist[:, 0] = 11
    wrist[:, -1] = 23
    observation = {
        "agentview_rgb": primary,
        "eye_in_hand_rgb": wrist,
        "ee_pos": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
        "ee_ori": np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.float64),
        "gripper_states": np.asarray([[0.25, 99], [0.75, 99]], dtype=np.float64),
    }
    actions = np.asarray(
        [
            [1, 2, 3, 4, 5, 6, -1],
            [7, 8, 9, 10, 11, 12, 1],
        ],
        dtype=np.float64,
    )

    result = standardize_demo_arrays(observation, actions)

    assert result["image"].shape == (2, 128, 128, 3)
    assert result["wrist_image"].shape == (2, 128, 128, 3)
    np.testing.assert_array_equal(result["image"][:, 0], 19)
    np.testing.assert_array_equal(result["wrist_image"][:, 0], 23)
    np.testing.assert_allclose(
        result["state"],
        [
            [1, 2, 3, 7, 8, 9, 0, 0.25],
            [4, 5, 6, 10, 11, 12, 0, 0.75],
        ],
    )
    np.testing.assert_allclose(result["action"][:, -1], [1, 0])
    assert result["action"].dtype == np.float32
    assert result["state"].dtype == np.float32


def test_target_demo_selection_is_datamil_deterministic():
    demo_ids = [f"demo_{index}" for index in range(50)]
    first = select_target_demo_ids(demo_ids, DEFAULT_TARGET_TASK)
    second = select_target_demo_ids(list(reversed(demo_ids)), DEFAULT_TARGET_TASK)
    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5
    assert set(first).issubset(demo_ids)


def test_dataset_name_and_lerobot_field_mapping():
    target_name = task_to_dataset_name(f"{DEFAULT_TARGET_TASK}_demo.hdf5")
    assert target_name == DEFAULT_TARGET_TASK.lower()
    from octo_small_libero.lerobot_v2 import (
        ACTION_KEY,
        PRIMARY_IMAGE_KEY,
        STATE_KEY,
        WRIST_IMAGE_KEY,
    )

    assert PRIMARY_IMAGE_KEY == "observation.images.image"
    assert WRIST_IMAGE_KEY == "observation.images.image2"
    assert STATE_KEY == "observation.state"
    assert ACTION_KEY == "action"


def test_single_source_sampler_is_disjoint_reproducible_and_resumable(monkeypatch):
    monkeypatch.setattr(libero_data, "_require_torch", lambda: _FakeTorch)

    samplers = [
        BalancedDistributedBatchSampler(
            (100,),
            local_batch_size=8,
            sample_weights=(1.0,),
            rank=rank,
            world_size=4,
            seed=17,
            num_batches=2,
        )
        for rank in range(4)
    ]
    first_batches = [next(iter(sampler)) for sampler in samplers]

    assert all({item.source for item in batch} == {0} for batch in first_batches)
    rank_frames = [{item.frame for item in batch} for batch in first_batches]
    assert len(set.union(*rank_frames)) == 32

    original = BalancedDistributedBatchSampler(
        (25,),
        local_batch_size=8,
        sample_weights=(1.0,),
        seed=5,
        num_batches=4,
    )
    iterator = iter(original)
    next(iterator)
    state = original.state_dict()
    expected = next(iterator)
    restored = BalancedDistributedBatchSampler(
        (25,),
        local_batch_size=8,
        sample_weights=(1.0,),
        seed=5,
        num_batches=4,
    )
    restored.load_state_dict(state)
    assert next(iter(restored)) == expected


def test_combined_dataset_accepts_one_target_source():
    target = [{"frame": 0}, {"frame": 1}]

    dataset = CombinedLeRobotDataset([target])

    assert dataset.source_sizes == (2,)
    assert dataset[(0, 1)] == {"frame": 1}


def test_target_only_training_uses_only_target_source_and_statistics(tmp_path, monkeypatch):
    torch_module = ModuleType("torch")
    torch_utils_module = ModuleType("torch.utils")
    torch_data_module = ModuleType("torch.utils.data")

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

    torch_data_module.DataLoader = FakeDataLoader
    torch_utils_module.data = torch_data_module
    torch_module.utils = torch_utils_module
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils_module)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data_module)
    monkeypatch.setattr(libero_data, "_require_torch", lambda: _FakeTorch)

    target_root = tmp_path / "libero10_5"
    statistics_path = target_root / "meta" / "stats.json"
    statistics_path.parent.mkdir(parents=True)
    statistics_path.write_text('{"target": true}\n', encoding="utf-8")
    prior_root = tmp_path / "libero90"
    loaded_statistics = []
    target_statistics = {"source": "target"}

    from octo_small_libero import checkpoint, selection

    def load_statistics(root):
        loaded_statistics.append(root)
        return target_statistics

    monkeypatch.setattr(checkpoint, "load_lerobot_statistics", load_statistics)
    monkeypatch.setattr(
        selection,
        "resolve_prior_selection",
        lambda *_args, **_kwargs: pytest.fail("target-only resolved prior selection"),
    )
    target_selection = SimpleNamespace(
        frame_indices=(0, 1),
        selection_sha256="target-selection",
    )
    monkeypatch.setattr(
        libero_data,
        "resolve_target_task_selection",
        lambda *_args, **_kwargs: target_selection,
    )
    created_sources = []

    class FakeFrameDataset:
        def __init__(self, root, **kwargs):
            self.root = root
            self.kwargs = kwargs
            created_sources.append(self)

        def __len__(self):
            return 2

    monkeypatch.setattr(libero_data, "LeRobotFrameDataset", FakeFrameDataset)
    config = {
        "data": {
            "target_only": True,
            "target_dataset": "libero10_5",
            "prior_dataset": "libero90",
            "sample_weights": [3.0, 1.0],
            "action_horizon": 8,
            "resize": {"primary": [256, 256], "wrist": [128, 128]},
        },
        "train": {
            "episode_cache_size": 2,
            "micro_batch_size_per_gpu": 8,
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
            "seed": 42,
            "num_workers_per_rank": 0,
            "prefetch_factor": 2,
        },
    }
    paths = {
        "target_dataset": target_root,
        "prior_dataset": prior_root,
        "statistics": statistics_path,
    }

    training_data = libero_data.make_training_dataset(
        config,
        paths,
        tokenizer=object(),
    )

    assert loaded_statistics == [target_root]
    assert len(created_sources) == 1
    assert created_sources[0].root == target_root
    assert created_sources[0].kwargs["statistics"] is target_statistics
    assert created_sources[0].kwargs["frame_indices"] == (0, 1)
    assert training_data.dataset.source_sizes == (2,)
    np.testing.assert_allclose(training_data.sample_weights, [1.0])
    assert training_data.prior_selection is None


def test_target_only_signature_covers_mode_and_normalization_statistics():
    target = SimpleNamespace(selection_sha256="target-selection")

    baseline = training_selection_sha256(
        target,
        None,
        (1.0,),
        training_mode="target_only",
        normalization_sha256="stats-a",
    )

    assert training_selection_sha256(target, None, (1.0,)) != baseline
    assert (
        training_selection_sha256(
            target,
            None,
            (1.0,),
            training_mode="target_only",
            normalization_sha256="stats-b",
        )
        != baseline
    )
