"""Bridge adapters for the shared prefiltered LeRobot selection contract."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from libero_lerobot.prefiltered import (
    PREFILTERED_REQUIRED_FIELDS,
    PrefilteredPriorSelection,
    PriorSelectionError,
    load_prefiltered_selection,
)
from trajectory_data import EpisodeRecord


@dataclass(frozen=True)
class BridgePrefilteredSelection:
    """Prefiltered fragments resolved to episode-local Bridge frame positions."""

    parsed: PrefilteredPriorSelection
    frame_positions_by_episode: Mapping[int, tuple[int, ...]]

    @property
    def source_path(self) -> Path:
        return self.parsed.source_path

    @property
    def source_sha256(self) -> str:
        return self.parsed.source_sha256

    @property
    def input_format(self) -> str:
        return self.parsed.input_format

    @property
    def selected_fragments(self) -> int:
        return self.parsed.selected_fragments

    @property
    def selected_episodes(self) -> int:
        return self.parsed.selected_episodes

    @property
    def training_starts(self) -> int:
        return self.parsed.training_starts

    @property
    def selection_sha256(self) -> str:
        return self.parsed.selection_sha256

    def as_manifest(self) -> dict[str, Any]:
        return self.parsed.as_manifest()


def load_bridge_prefiltered_selection(
    source_path: str | Path,
    records: Sequence[EpisodeRecord],
    *,
    action_horizon: int,
) -> BridgePrefilteredSelection:
    """Load a shared prefiltered selection against Bridge episode metadata."""

    ordered_records = tuple(records)
    offset = 0
    offsets: list[int] = []
    metadata_episodes = []
    for record in ordered_records:
        offsets.append(offset)
        metadata_episodes.append(
            SimpleNamespace(
                episode_index=int(record.episode_id),
                length=int(record.length),
            )
        )
        offset += int(record.length)
    metadata = SimpleNamespace(
        episodes=tuple(metadata_episodes),
        global_offsets={
            int(record.episode_id): episode_offset
            for record, episode_offset in zip(
                ordered_records,
                offsets,
                strict=True,
            )
        },
    )
    parsed = load_prefiltered_selection(
        source_path,
        metadata,
        action_horizon=action_horizon,
    )
    if parsed.source_schema != "fragment_range":
        raise PriorSelectionError(
            "Bridge prefiltered JSONL requires fields "
            f"{list(PREFILTERED_REQUIRED_FIELDS)}"
        )
    grouped: dict[int, list[int]] = {}
    for global_frame in parsed.frame_indices:
        record_position = bisect_right(offsets, int(global_frame)) - 1
        record = ordered_records[record_position]
        frame_position = int(global_frame) - offsets[record_position]
        grouped.setdefault(int(record.episode_id), []).append(frame_position)
    return BridgePrefilteredSelection(
        parsed=parsed,
        frame_positions_by_episode={
            episode_id: tuple(positions) for episode_id, positions in grouped.items()
        },
    )


def resolve_bridge_prefiltered_selection(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    records: Sequence[EpisodeRecord],
) -> BridgePrefilteredSelection | None:
    source = config["data"]["prior_selection"].get("prefiltered_scores")
    if source is None:
        return None
    return load_bridge_prefiltered_selection(
        paths["prior_prefiltered_scores"],
        records,
        action_horizon=int(config["data"]["action_horizon"]),
    )


__all__ = [
    "BridgePrefilteredSelection",
    "load_bridge_prefiltered_selection",
    "resolve_bridge_prefiltered_selection",
]
