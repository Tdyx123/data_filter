from __future__ import annotations

import pytest

from cocore.timing import emit_completed_timing, timed_step


def test_emit_completed_timing_writes_stable_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit_completed_timing("graph.prototypes", 1.25)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "cocore_timing step=graph.prototypes seconds=1.250000 status=completed\n"
    )


def test_timed_step_reports_only_successful_completion() -> None:
    events: list[tuple[str, float]] = []

    with timed_step("encode.pca_fusion", lambda step, seconds: events.append((step, seconds))):
        pass
    with pytest.raises(RuntimeError, match="boom"):
        with timed_step("failed", lambda step, seconds: events.append((step, seconds))):
            raise RuntimeError("boom")

    assert [step for step, _ in events] == ["encode.pca_fusion"]
    assert events[0][1] >= 0.0


def test_timed_step_with_no_callback_is_silent() -> None:
    with timed_step("encode.visual_cache", None):
        pass
