from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cocore import prototypes
from tests.test_cocore_prototypes import (
    _StopFallbackAdapter,
    _TrajectoryPrototypeAdapter,
    _candidate_clip,
    _identity_fragment_pca,
    _write_frame_caches,
)


def _candidate_halves(cache: Path) -> np.ndarray:
    frames = np.load(cache / "ep000002.npy", allow_pickle=False)
    halves = np.stack([frames[:8].mean(axis=0), frames[7:].mean(axis=0)])
    halves /= np.linalg.norm(halves, axis=1, keepdims=True)
    return halves[None, :, :]


def test_action_only_builds_one_nonvisual_leaf_per_trained_action(tmp_path: Path) -> None:
    adapter = _TrajectoryPrototypeAdapter(combined_candidate=True)
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)

    result = prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        _candidate_halves(cache),
        pca_components=_identity_fragment_pca(2),
        visual_dim=2,
        frame_cache_dir=cache,
        batch_size=32,
        max_iter=2,
        seed=23,
        max_episodes=None,
        num_workers=0,
        representation="action_only",
    )

    assert [leaf.action_label for leaf in result.catalog.leaf_prototypes] == [
        "move forward",
        "move right",
    ]
    assert all(leaf.center_id == 0 for leaf in result.catalog.leaf_prototypes)
    assert result.prototypes.centers is not None
    np.testing.assert_array_equal(result.prototypes.centers, np.zeros((2, 2), np.float32))
    assert result.half_action_labels.tolist() == [
        ["move forward right", "move forward right"]
    ]
    assert result.prototypes.indices.tolist() == [[0, -1]]
    np.testing.assert_allclose(
        result.prototypes.weights,
        np.asarray([[1.125, 0.0]], dtype=np.float32),
    )


def test_action_only_tied_parent_choice_is_independent_of_candidate_visual(
    tmp_path: Path,
) -> None:
    adapter = _TrajectoryPrototypeAdapter(combined_candidate=True)
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)
    first = _candidate_halves(cache)
    second = first[..., ::-1].copy()

    results = [
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [_candidate_clip()],
            values,
            pca_components=_identity_fragment_pca(2),
            visual_dim=2,
            frame_cache_dir=cache,
            batch_size=32,
            max_iter=2,
            seed=23,
            max_episodes=None,
            num_workers=0,
            representation="action_only",
        )
        for values in (first, second)
    ]

    assert [result.prototypes.indices.tolist() for result in results] == [
        [[0, -1]],
        [[0, -1]],
    ]


def test_action_only_preserves_stop_fallback_and_disabled_stop_filtering(
    tmp_path: Path,
) -> None:
    adapter = _StopFallbackAdapter(include_stop=True)
    visual_halves = np.asarray(
        [[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32
    )
    enabled = prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        visual_halves,
        pca_components=_identity_fragment_pca(2),
        visual_dim=2,
        frame_cache_dir=tmp_path,
        batch_size=32,
        max_iter=2,
        seed=23,
        max_episodes=None,
        num_workers=0,
        representation="action_only",
    )
    disabled = prototypes.build_hierarchical_motion_prototypes(
        adapter,
        [_candidate_clip()],
        visual_halves,
        pca_components=_identity_fragment_pca(2),
        visual_dim=2,
        frame_cache_dir=tmp_path,
        batch_size=32,
        max_iter=2,
        seed=23,
        max_episodes=None,
        num_workers=0,
        use_stop_bucket=False,
        representation="action_only",
    )

    enabled_leaf = enabled.catalog.leaf_prototypes[
        int(enabled.prototypes.indices[0, 0])
    ]
    assert enabled_leaf.action_label == "stop"
    np.testing.assert_array_equal(enabled.eligible_mask, [True])
    np.testing.assert_array_equal(disabled.eligible_mask, [False])
    assert all(
        leaf.action_label != "stop" for leaf in disabled.catalog.leaf_prototypes
    )


def test_disabling_assignment_confidence_sets_each_half_weight_to_one(
    tmp_path: Path,
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)

    results = [
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [_candidate_clip()],
            _candidate_halves(cache),
            pca_components=_identity_fragment_pca(2),
            visual_dim=2,
            frame_cache_dir=cache,
            batch_size=32,
            max_iter=2,
            seed=23,
            max_episodes=None,
            num_workers=0,
            use_assignment_confidence=enabled,
        )
        for enabled in (True, False)
    ]
    weighted, result = results

    np.testing.assert_array_equal(result.prototypes.indices, weighted.prototypes.indices)
    np.testing.assert_array_equal(result.half_action_labels, weighted.half_action_labels)
    np.testing.assert_allclose(
        result.prototypes.weights,
        np.asarray([[1.0, 1.0]], dtype=np.float32),
    )


@pytest.mark.parametrize("representation", ["visual", "", None])
def test_prototype_builder_rejects_unknown_ablation_representation(
    tmp_path: Path, representation: object
) -> None:
    adapter = _TrajectoryPrototypeAdapter()
    cache = tmp_path / "frame_embeddings"
    _write_frame_caches(cache, adapter)

    with pytest.raises(ValueError, match="representation"):
        prototypes.build_hierarchical_motion_prototypes(
            adapter,
            [_candidate_clip()],
            _candidate_halves(cache),
            pca_components=_identity_fragment_pca(2),
            visual_dim=2,
            frame_cache_dir=cache,
            batch_size=32,
            max_iter=2,
            seed=23,
            max_episodes=None,
            num_workers=0,
            representation=representation,  # type: ignore[arg-type]
        )
