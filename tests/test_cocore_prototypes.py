from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from cocore import prototypes


def test_action_catalog_retains_counts_at_fixed_and_fractional_thresholds() -> None:
    fixed = prototypes.create_action_catalog(
        Counter({"move forward": 40, "move right": 7_960}),
        total_labels=8_000,
    )
    fractional = prototypes.create_action_catalog(
        Counter({"move forward": 42, "move backward": 41, "move right": 8_118}),
        total_labels=8_201,
    )

    fixed_by_label = {
        category.label: category for category in fixed.action_categories
    }
    fractional_by_label = {
        category.label: category for category in fractional.action_categories
    }
    assert fixed_by_label["move forward"].retained is True
    assert fractional_by_label["move forward"].retained is True
    assert fractional_by_label["move backward"].retained is False
    assert fixed.action_labels == ("move right", "move forward", "stop")
    assert fractional.action_labels == ("move right", "move forward", "stop")


def test_action_catalog_rejects_non_stop_data_without_a_retained_non_stop_action() -> None:
    with pytest.raises(ValueError, match="no non-stop action meets retention threshold"):
        prototypes.create_action_catalog(
            Counter({"move forward": 39, "stop": 61}),
            total_labels=100,
        )


def test_action_catalog_accepts_pure_stop_data_below_the_count_floor() -> None:
    catalog = prototypes.create_action_catalog(Counter({"stop": 3}), total_labels=3)

    category = catalog.action_categories[0]
    assert category.label == "stop"
    assert category.raw_count == 3
    assert category.raw_proportion == pytest.approx(1.0)
    assert category.retained is False
    assert tuple(
        (parent.label, parent.probability) for parent in category.parents
    ) == (("stop", 1.0),)
    assert catalog.action_labels == ("stop",)


def test_action_parent_distribution_uses_all_maximum_cardinality_subsets() -> None:
    distribution = prototypes.action_parent_distribution(
        "move forward right, tilt up, open gripper",
        {
            "move forward, tilt up": 40,
            "move right, tilt up": 80,
            "move forward right": 60,
            "move forward": 1_000,
            "rotate clockwise": 2_000,
        },
    )

    assert distribution == (
        ("move right, tilt up", pytest.approx(4.0 / 9.0)),
        ("move forward right", pytest.approx(3.0 / 9.0)),
        ("move forward, tilt up", pytest.approx(2.0 / 9.0)),
    )


def test_action_parent_distribution_keeps_a_retained_action_exactly() -> None:
    assert prototypes.action_parent_distribution(
        "move forward right, tilt up",
        {
            "move forward right, tilt up": 40,
            "move forward": 400,
            "move right, tilt up": 80,
        },
    ) == (("move forward right, tilt up", 1.0),)


def test_action_parent_distribution_uses_stop_only_without_a_nonempty_subset() -> None:
    assert prototypes.action_parent_distribution(
        "tilt down, close gripper",
        {"move forward": 400, "stop": 500},
    ) == (("stop", 1.0),)


def test_action_catalog_records_raw_values_and_parent_assignments() -> None:
    catalog = prototypes.create_action_catalog(
        Counter(
            {
                "move right, tilt up": 80,
                "move forward right": 60,
                "move forward, tilt up": 40,
                "move forward right, tilt up, open gripper": 10,
                "stop": 10,
            }
        ),
        total_labels=200,
    )

    by_label = {category.label: category for category in catalog.action_categories}
    rare = by_label["move forward right, tilt up, open gripper"]
    assert rare.raw_count == 10
    assert rare.raw_proportion == pytest.approx(0.05)
    assert rare.action_id is None
    assert rare.retained is False
    assert tuple(
        (parent.label, parent.probability) for parent in rare.parents
    ) == (
        ("move right, tilt up", pytest.approx(4.0 / 9.0)),
        ("move forward right", pytest.approx(3.0 / 9.0)),
        ("move forward, tilt up", pytest.approx(2.0 / 9.0)),
    )


@pytest.mark.parametrize(
    "label",
    [
        "move sideways",
        "move forward, wave gripper",
    ],
)
def test_action_helpers_reject_unknown_motion_labels(label: str) -> None:
    with pytest.raises(ValueError, match="unknown motion primitive label"):
        prototypes.action_parent_distribution(label, {"move forward": 40})
    with pytest.raises(ValueError, match="unknown motion primitive label"):
        prototypes.canonical_clip_action(label, "stop")


def test_canonical_clip_action_unions_deduplicates_and_uses_semantic_order() -> None:
    assert prototypes.canonical_clip_action(
        "open gripper, move left up, tilt down",
        "move forward left, rotate clockwise, open gripper",
    ) == "move forward left up, tilt down, rotate clockwise, open gripper"
    assert prototypes.canonical_clip_action("stop", "stop") == "stop"


@pytest.mark.parametrize(
    ("mass", "expected"),
    [
        (1.0, 1),
        (2.0, 2),
        (8.0, 4),
        (16_384.0, 15),
        (32_768.0, 16),
        (1_000_000.0, 16),
    ],
)
def test_cluster_count_for_mass_uses_capped_logarithmic_formula(
    mass: float,
    expected: int,
) -> None:
    assert prototypes.cluster_count_for_mass(mass) == expected


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan, np.inf, -np.inf, True, "2"])
def test_cluster_count_for_mass_rejects_non_positive_or_non_finite_values(
    mass: object,
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        prototypes.cluster_count_for_mass(mass)  # type: ignore[arg-type]


def test_visual_center_probabilities_use_stable_all_center_squared_distance_softmax() -> None:
    probabilities = prototypes.visual_center_probabilities(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(
        probabilities,
        np.asarray(
            [
                [0.9999546021312976, 0.0000453978687024, 4.248161389e-18],
                [0.0000453958078295, 0.9999092083843409, 0.0000453958078295],
            ],
            dtype=np.float64,
        ),
        rtol=1.0e-12,
        atol=1.0e-18,
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities > 0.0)


def test_visual_center_probabilities_return_exact_one_for_one_center() -> None:
    probabilities = prototypes.visual_center_probabilities(
        np.asarray([[0.0, 0.0], [2.0, -3.0]], dtype=np.float32),
        np.asarray([[100.0, 100.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(probabilities, np.ones((2, 1), dtype=np.float64))


@pytest.mark.parametrize(
    ("values", "centers"),
    [
        (np.asarray([1.0, 2.0]), np.asarray([[1.0, 2.0]])),
        (np.empty((0, 2)), np.asarray([[1.0, 2.0]])),
        (np.asarray([[1.0, 2.0]]), np.empty((0, 2))),
        (np.asarray([[1.0, 2.0]]), np.asarray([[1.0, 2.0, 3.0]])),
        (np.asarray([[np.nan, 2.0]]), np.asarray([[1.0, 2.0]])),
        (np.asarray([[1.0, 2.0]]), np.asarray([[np.inf, 2.0]])),
    ],
)
def test_visual_center_probabilities_reject_malformed_or_non_finite_inputs(
    values: np.ndarray,
    centers: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="finite non-empty matrices"):
        prototypes.visual_center_probabilities(values, centers)


def test_action_catalog_serializes_schema_four_strategy_and_metadata() -> None:
    catalog = prototypes.create_action_catalog(
        Counter({"move forward": 40, "move right": 60}),
        total_labels=100,
    )
    forward = next(
        category
        for category in catalog.action_categories
        if category.label == "move forward"
    )
    updated_forward = replace(
        forward,
        effective_mass=42.5,
        requested_centers=6,
        actual_centers=5,
    )
    catalog = replace(
        catalog,
        action_categories=tuple(
            updated_forward if category.label == "move forward" else category
            for category in catalog.action_categories
        ),
        leaf_prototypes=(
            prototypes.LeafPrototype(
                prototype_id=0,
                label="move forward::center_0",
                action_id=int(updated_forward.action_id),
                action_label="move forward",
                center_id=0,
            ),
        ),
    )

    payload = catalog.to_dict()

    assert payload["schema_version"] == 4
    assert payload["strategy"] == "trajectory_action_subset_then_visual_softmax"
    assert payload["constants"] == {
        "state_threshold": 0.03,
        "min_action_count": 40,
        "min_action_frequency": 0.005,
        "max_visual_centers": 16,
        "visual_softmax_temperature": 0.1,
        "cluster_count": "min(16, 1 + floor(log2(effective_mass)))",
    }
    assert payload["total_raw_actions"] == 100
    forward_payload = next(
        category
        for category in payload["action_categories"]
        if category["label"] == "move forward"
    )
    assert forward_payload["raw_count"] == 40
    assert forward_payload["raw_proportion"] == pytest.approx(0.4)
    assert forward_payload["parents"] == [
        {
            "action_id": updated_forward.action_id,
            "label": "move forward",
            "probability": 1.0,
        }
    ]
    assert forward_payload["effective_mass"] == pytest.approx(42.5)
    assert forward_payload["requested_centers"] == 6
    assert forward_payload["actual_centers"] == 5
    assert payload["leaf_prototypes"] == [
        {
            "prototype_id": 0,
            "label": "move forward::center_0",
            "action_id": updated_forward.action_id,
            "action_label": "move forward",
            "center_id": 0,
        }
    ]
