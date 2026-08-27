"""Cocore-owned near-uniform candidate indexing."""

from __future__ import annotations

from collections.abc import Sequence

from relcore.schemas import ClipRecord
from trajectory_data import EpisodeRecord

from cocore.temporal import resolve_temporal_geometry


CLIP_LENGTH = resolve_temporal_geometry("libero").clip_length
WINDOW_POLICY = "near_uniform_full_coverage"


def uniform_clip_windows(
    episode_length: int,
    clip_length: int = CLIP_LENGTH,
) -> list[tuple[int, int]]:
    """Return complete fixed-length windows spread across the full episode."""

    if clip_length <= 0:
        raise ValueError("clip length must be positive")
    if episode_length < clip_length:
        return []
    count = (episode_length + clip_length - 1) // clip_length
    if count == 1:
        return [(0, clip_length - 1)]

    gap_count = count - 1
    short_gap, long_gap_count = divmod(episode_length - clip_length, gap_count)
    gaps = [short_gap] * (gap_count - long_gap_count) + [short_gap + 1] * long_gap_count
    starts = [0]
    for gap in gaps:
        starts.append(starts[-1] + gap)
    return [(start, start + clip_length - 1) for start in starts]


def _sample_id(episode_id: int, start: int, end: int) -> str:
    return f"ep{episode_id:06d}_fragment_{start:06d}_{end:06d}"


def build_clip_records(
    episodes: Sequence[EpisodeRecord],
    *,
    clip_length: int = CLIP_LENGTH,
) -> list[ClipRecord]:
    """Index Cocore candidates and link chronological neighbors per episode."""

    output: list[ClipRecord] = []
    for episode in episodes:
        if episode.task_index is None or episode.task_name is None:
            raise ValueError(
                f"episode {episode.episode_id} has no task metadata required for Cocore"
            )
        windows = uniform_clip_windows(episode.length, clip_length=clip_length)
        sample_ids = [_sample_id(episode.episode_id, start, end) for start, end in windows]
        for position, ((start, end), sample_id) in enumerate(zip(windows, sample_ids, strict=True)):
            output.append(
                ClipRecord(
                    sample_id=sample_id,
                    episode_id=episode.episode_id,
                    task_index=episode.task_index,
                    task_name=episode.task_name,
                    start_step=start,
                    end_step=end,
                    length=end - start + 1,
                    previous_sample_id=sample_ids[position - 1] if position > 0 else None,
                    next_sample_id=(
                        sample_ids[position + 1] if position + 1 < len(sample_ids) else None
                    ),
                )
            )
    return output
