import json

import pytest

from qwen3_vl_groot.benchmarking import (
    combine_phase_metrics,
    is_meaningfully_faster,
    summarize_metrics,
)


def _write_metrics(path, *, step_seconds, data_wait_fraction):
    rows = []
    for step in range(10, 101, 10):
        row = {
            "step": step,
            "performance/step_seconds": step_seconds + step / 1_000,
            "performance/samples_per_second": 64 / (step_seconds + step / 1_000),
            "performance/data_wait_fraction": data_wait_fraction,
        }
        rows.extend((row, row))
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_summarize_metrics_discards_first_20_steps_and_deduplicates(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_path, step_seconds=4.0, data_wait_fraction=0.125)

    summary = summarize_metrics(
        metrics_path,
        warmup_steps=20,
        expected_steps=100,
        effective_batch_size=64,
    )

    assert summary.measured_steps == 80
    assert summary.step_seconds == pytest.approx(sum(4.0 + step / 1_000 for step in range(30, 101, 10)) / 8)
    assert summary.samples_per_second == pytest.approx(64 / summary.step_seconds)
    assert summary.data_wait_fraction == pytest.approx(0.125)


def test_summarize_metrics_rejects_incomplete_candidate(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "step": 90,
                "performance/step_seconds": 4.0,
                "performance/data_wait_fraction": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="step 100"):
        summarize_metrics(
            metrics_path,
            warmup_steps=20,
            expected_steps=100,
            effective_batch_size=64,
        )


def test_phase_combination_uses_90_percent_head_and_10_percent_lora():
    combined = combine_phase_metrics(
        head_only_step_seconds=4.0,
        lora_active_step_seconds=6.0,
    )

    assert combined == pytest.approx(4.2)
    assert is_meaningfully_faster(reference_step_seconds=5.0, candidate_step_seconds=4.74)
    assert not is_meaningfully_faster(reference_step_seconds=5.0, candidate_step_seconds=4.76)
