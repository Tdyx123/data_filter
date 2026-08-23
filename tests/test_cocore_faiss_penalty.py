from __future__ import annotations

import builtins

import numpy as np
import pytest


def test_flat_penalty_accumulates_every_threshold_match_with_reliability_weights() -> None:
    from cocore.faiss_penalty import FaissFlatPenaltySpace

    space = FaissFlatPenaltySpace(
        np.asarray(
            [
                [1.0, 0.0],
                [0.9, np.sqrt(0.19)],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([0.5, 0.25, 0.8, 1.0], dtype=np.float32),
        similarity_threshold=0.8,
        redundancy_normalizer=2.0,
    )
    index = space.create_index((1, 0, 3))

    assert index.penalty(2) == pytest.approx(0.25, abs=1.0e-7)


def test_flat_penalty_reports_how_to_install_missing_faiss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cocore.faiss_penalty import FaissFlatPenaltySpace

    real_import = builtins.__import__

    def import_without_faiss(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("faiss intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_faiss)

    with pytest.raises(RuntimeError, match="install cocore/requirements.txt"):
        FaissFlatPenaltySpace(
            np.eye(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            similarity_threshold=0.8,
            redundancy_normalizer=1.0,
        )


def test_flat_penalty_uses_objective_epsilon_near_one_threshold() -> None:
    from cocore.faiss_penalty import FaissFlatPenaltySpace

    space = FaissFlatPenaltySpace(
        np.ones((2, 1), dtype=np.float32),
        np.ones(2, dtype=np.float32),
        similarity_threshold=1.0 - 1.0e-10,
        redundancy_normalizer=1.0,
        epsilon=1.0e-8,
    )

    assert space.create_index((0,)).penalty(1) == pytest.approx(0.01, abs=1.0e-8)
