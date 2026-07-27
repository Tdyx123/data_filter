import numpy as np

from octo_small_libero.data import (
    DEFAULT_TARGET_TASK,
    select_target_demo_ids,
    standardize_demo_arrays,
    task_to_dataset_name,
)


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
