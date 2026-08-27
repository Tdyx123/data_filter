from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from cocore_bridge_v2.action_diagnostics import (
    analyze_bridge_action_windows,
    summarize_bridge_action_counts,
    validate_reference_acceptance,
)
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord


class _DiagnosticAdapter(DatasetAdapter):
    def __init__(self) -> None:
        self.load_images_calls: list[bool] = []
        self._records = (
            EpisodeRecord(0, 1205, 0, "move"),
            EpisodeRecord(1, 1205, 1, "move and roll"),
        )

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ()

    def episodes(self) -> Sequence[EpisodeRecord]:
        return self._records

    def iter_episodes(
        self,
        *,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[EpisodeData]:
        del num_workers
        self.load_images_calls.append(load_images)
        records = self._records[:max_episodes] if max_episodes else self._records
        for record in records:
            steps = np.arange(record.length, dtype=np.float32)
            states = np.zeros((record.length, 8), dtype=np.float32)
            states[:, 0] = steps * np.float32(0.02)
            states[:, 1] = steps * np.float32(0.008)
            if record.episode_id == 1:
                states[:, 3] = steps * np.float32(0.05)
            yield EpisodeData(
                episode_id=record.episode_id,
                timestamps=steps.astype(np.float64) / 5.0,
                frame_indices=np.arange(record.length, dtype=np.int64),
                observations={"observation.state": states},
                actions=np.zeros((record.length, 7), dtype=np.float32),
                task_index=record.task_index,
                task_name=record.task_name,
            )

    def fingerprint(self) -> str:
        return "bridge-action-diagnostic-test-v1"


def test_diagnostic_scans_states_without_loading_images_and_reports_axes() -> None:
    adapter = _DiagnosticAdapter()

    report = analyze_bridge_action_windows(adapter)

    assert report["profile"] == "bridge_v2"
    assert report["episode_count"] == 2
    assert report["trajectory_window_length"] == 4
    assert report["trajectory_horizon"] == 3
    assert report["window_count"] == 804
    assert report["unique_compound_labels"] == 2
    assert report["retained_non_stop_action_buckets"] == 2
    assert report["estimated_leaf_prototypes"] == 20
    assert report["exact_non_stop_coverage"] == 1.0
    assert report["raw_stop_rate"] == 0.0
    assert report["stop_or_no_parent_fallback_rate"] == 0.0
    assert report["atomic_action_retention_quality"] == 1.0
    assert report["atomic_occurrence_retention_quality"] == 1.0
    assert report["axis_statistics"]["x"]["activation_rate"] == 1.0
    assert report["axis_statistics"]["y"]["activation_rate"] == 0.0
    assert report["axis_statistics"]["roll"]["activation_rate"] == 0.5
    assert report["axis_statistics"]["yaw"]["activation_rate"] == 0.0
    assert adapter.load_images_calls == [False]


def test_count_summary_separates_exact_parent_and_stop_fallback() -> None:
    report = summarize_bridge_action_counts(
        Counter(
            {
                "move forward": 400,
                "move forward, roll positive": 300,
                "stop": 200,
                "roll negative": 100,
            }
        )
    )

    assert report["retention_cutoff"] == 400
    assert report["retained_non_stop_action_buckets"] == 1
    assert report["exact_non_stop_coverage"] == 0.4
    assert report["parent_fallback_rate"] == 0.3
    assert report["raw_stop_rate"] == 0.2
    assert report["no_parent_fallback_rate"] == 0.1
    assert report["stop_or_no_parent_fallback_rate"] == 0.3
    assert report["atomic_action_retention_quality"] == 0.55
    assert report["atomic_occurrence_retention_quality"] == 7.0 / 11.0
    assert report["estimated_leaf_prototypes"] == 20


def test_reference_acceptance_reports_every_failed_contract() -> None:
    failures = validate_reference_acceptance(
        {
            "window_count": 1,
            "unique_compound_labels": 1,
            "retained_non_stop_action_buckets": 1,
            "estimated_leaf_prototypes": 1,
            "exact_non_stop_coverage": 0.0,
            "raw_stop_rate": 1.0,
            "atomic_action_retention_quality": 0.0,
        }
    )

    assert len(failures) == 7
    assert all(isinstance(failure, str) and failure for failure in failures)


def test_reference_acceptance_accepts_new_four_frame_production_baseline() -> None:
    failures = validate_reference_acceptance(
        {
            "window_count": 434_370,
            "unique_compound_labels": 1_399,
            "retained_non_stop_action_buckets": 83,
            "estimated_leaf_prototypes": 1_139,
            "exact_non_stop_coverage": 0.6539,
            "raw_stop_rate": 0.2236,
            "atomic_action_retention_quality": 0.730,
        }
    )

    assert failures == ()


@pytest.mark.parametrize("count", [True, -1, 1.5])
def test_count_summary_rejects_nonnegative_integer_contract_violations(count: object) -> None:
    with pytest.raises(ValueError, match="counts"):
        summarize_bridge_action_counts({"move forward": count})  # type: ignore[dict-item]
