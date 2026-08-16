from __future__ import annotations

import math

import pytest

from cocore.index import build_clip_records, uniform_clip_windows
from trajectory_data import EpisodeRecord


@pytest.mark.parametrize(
    ("length", "expected_starts"),
    [
        (0, []),
        (14, []),
        (15, [0]),
        (16, [0, 1]),
        (31, [0, 8, 16]),
        (46, [0, 10, 20, 31]),
        (207, [0, 14, 28, 42, 57, 72, 87, 102, 117, 132, 147, 162, 177, 192]),
    ],
)
def test_uniform_clip_windows_match_fixed_examples(
    length: int,
    expected_starts: list[int],
) -> None:
    windows = uniform_clip_windows(length)

    assert [start for start, _ in windows] == expected_starts
    assert windows == [(start, start + 14) for start in expected_starts]


def test_uniform_clip_windows_preserve_count_boundaries_and_ordered_gaps() -> None:
    for length in range(15, 501):
        windows = uniform_clip_windows(length)
        starts = [start for start, _ in windows]
        gaps = [right - left for left, right in zip(starts, starts[1:], strict=False)]

        assert len(windows) == math.ceil(length / 15)
        assert windows[0] == (0, 14)
        assert windows[-1] == (length - 15, length - 1)
        assert all(end - start + 1 == 15 for start, end in windows)
        assert starts == sorted(set(starts))
        if gaps:
            assert max(gaps) - min(gaps) <= 1
            assert gaps == sorted(gaps)


def test_build_clip_records_links_ordered_candidates_even_when_they_overlap() -> None:
    clips = build_clip_records(
        [
            EpisodeRecord(2, 16, 4, "task four"),
            EpisodeRecord(3, 31, 5, "task five"),
        ]
    )

    assert [clip.sample_id for clip in clips] == [
        "ep000002_fragment_000000_000014",
        "ep000002_fragment_000001_000015",
        "ep000003_fragment_000000_000014",
        "ep000003_fragment_000008_000022",
        "ep000003_fragment_000016_000030",
    ]
    assert [(clip.previous_sample_id, clip.next_sample_id) for clip in clips] == [
        (None, clips[1].sample_id),
        (clips[0].sample_id, None),
        (None, clips[3].sample_id),
        (clips[2].sample_id, clips[4].sample_id),
        (clips[3].sample_id, None),
    ]
    assert [(clip.task_index, clip.task_name) for clip in clips] == [
        (4, "task four"),
        (4, "task four"),
        (5, "task five"),
        (5, "task five"),
        (5, "task five"),
    ]


def test_build_clip_records_requires_task_metadata() -> None:
    with pytest.raises(ValueError, match="task metadata"):
        build_clip_records([EpisodeRecord(0, 30)])
