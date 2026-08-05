"""Build deterministic SQCN-compatible clip records from episode metadata."""

from __future__ import annotations

from collections.abc import Sequence

from trajectory_data import EpisodeRecord, aligned_chunk_windows

from relcore.schemas import ClipRecord


def candidate_windows(
    episode_length: int,
    *,
    length: int = 15,
    stride: int = 15,
) -> list[tuple[int, int]]:
    """Return complete inclusive windows with a final tail-aligned window."""

    if episode_length < length:
        return []
    return aligned_chunk_windows(episode_length, length, stride)


def _sample_id(episode_id: int, start: int, end: int) -> str:
    return f"ep{episode_id:06d}_fragment_{start:06d}_{end:06d}"


def build_clip_records(
    episodes: Sequence[EpisodeRecord],
    *,
    length: int = 15,
    stride: int = 15,
) -> list[ClipRecord]:
    """Index complete clips and link only exactly contiguous neighbors."""

    output: list[ClipRecord] = []
    for episode in episodes:
        if episode.task_index is None or episode.task_name is None:
            raise ValueError(
                f"episode {episode.episode_id} has no task metadata required for quotas"
            )
        windows = candidate_windows(episode.length, length=length, stride=stride)
        by_start = {start: _sample_id(episode.episode_id, start, end) for start, end in windows}
        by_end = {end: _sample_id(episode.episode_id, start, end) for start, end in windows}
        for start, end in windows:
            output.append(
                ClipRecord(
                    sample_id=_sample_id(episode.episode_id, start, end),
                    episode_id=episode.episode_id,
                    task_index=episode.task_index,
                    task_name=episode.task_name,
                    start_step=start,
                    end_step=end,
                    length=end - start + 1,
                    previous_sample_id=by_end.get(start - 1),
                    next_sample_id=by_start.get(end + 1),
                )
            )
    return output
