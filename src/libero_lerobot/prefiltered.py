from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import LiberoDataError
from .metadata import LeRobotV2Metadata


PREFILTERED_REQUIRED_FIELDS = (
    "episode_id",
    "start_step",
    "end_step",
)


class PriorSelectionError(LiberoDataError):
    """Raised when an input cannot safely select LIBERO prior frames."""


@dataclass(frozen=True)
class _PrefilteredFragmentRow:
    episode_id: int
    start_step: int
    end_step: int


@dataclass(frozen=True)
class PrefilteredPriorSelection:
    source_path: Path
    source_sha256: str
    input_format: str
    selected_fragments: int
    selected_episodes: int
    frame_indices: tuple[int, ...]
    action_horizon: int
    selection_sha256: str

    @property
    def training_starts(self) -> int:
        return len(self.frame_indices)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": "prefiltered_fragments",
            "input_format": self.input_format,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "required_fields": list(PREFILTERED_REQUIRED_FIELDS),
            "selected_fragments": self.selected_fragments,
            "selected_episodes": self.selected_episodes,
            "training_starts": self.training_starts,
            "action_horizon": self.action_horizon,
            "ordering": ["source row order"],
            "boundary_policy": "complete_action_window",
            "overlap_policy": "deduplicate_episode_frame_start",
            "selection_sha256": self.selection_sha256,
        }


def _strict_int(value: Any, field: str, line_number: int) -> int:
    if isinstance(value, bool):
        raise PriorSelectionError(
            f"invalid integer in {field!r} at prefiltered selection line {line_number}"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if candidate and candidate.lstrip("+-").isdigit():
            return int(candidate)
    raise PriorSelectionError(
        f"invalid integer in {field!r} at prefiltered selection line {line_number}"
    )


def _parse_prefiltered_row(
    raw: Mapping[str, Any],
    *,
    line_number: int,
    episode_lengths: Mapping[int, int],
    action_horizon: int,
    seen: set[tuple[int, int, int]],
) -> _PrefilteredFragmentRow:
    missing = [field for field in PREFILTERED_REQUIRED_FIELDS if field not in raw]
    if missing:
        raise PriorSelectionError(
            "prefiltered selection is missing required fields "
            f"{missing} at line {line_number}"
        )
    episode_id = _strict_int(raw["episode_id"], "episode_id", line_number)
    start_step = _strict_int(raw["start_step"], "start_step", line_number)
    end_step = _strict_int(raw["end_step"], "end_step", line_number)
    fragment = (episode_id, start_step, end_step)
    if fragment in seen:
        raise PriorSelectionError(
            f"duplicate fragment {fragment} at prefiltered selection line {line_number}"
        )
    seen.add(fragment)
    if episode_id not in episode_lengths:
        raise PriorSelectionError(
            f"unknown episode_id={episode_id} at prefiltered selection line {line_number}"
        )
    if start_step < 0 or end_step < start_step:
        raise PriorSelectionError(
            f"invalid frame range [{start_step}, {end_step}] "
            f"at prefiltered selection line {line_number}"
        )
    if end_step >= episode_lengths[episode_id]:
        raise PriorSelectionError(
            f"fragment end_step={end_step} exceeds episode {episode_id} "
            f"length={episode_lengths[episode_id]} at line {line_number}"
        )
    if end_step - start_step + 1 < action_horizon:
        raise PriorSelectionError(
            f"fragment [{start_step}, {end_step}] is shorter than "
            f"action_horizon={action_horizon} at line {line_number}"
        )
    return _PrefilteredFragmentRow(
        episode_id=episode_id,
        start_step=start_step,
        end_step=end_step,
    )


def _read_source(path: Path) -> tuple[str, str]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise PriorSelectionError(
            f"could not read prefiltered selection {path}: {error}"
        ) from error
    return text, hashlib.sha256(payload).hexdigest()


def _read_prefiltered_rows(
    path: Path,
    metadata: LeRobotV2Metadata,
    *,
    action_horizon: int,
) -> tuple[str, str, list[_PrefilteredFragmentRow]]:
    text, source_sha256 = _read_source(path)
    first_content = next((line.lstrip() for line in text.splitlines() if line.strip()), None)
    if first_content is None:
        raise PriorSelectionError(f"prefiltered selection is empty: {path}")

    episode_lengths = {
        episode.episode_index: episode.length for episode in metadata.episodes
    }
    seen: set[tuple[int, int, int]] = set()
    rows: list[_PrefilteredFragmentRow] = []
    if first_content.startswith("{"):
        input_format = "jsonl"
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise PriorSelectionError(
                    f"invalid JSON in prefiltered selection at line {line_number}: {error}"
                ) from error
            if not isinstance(raw, dict):
                raise PriorSelectionError(
                    "prefiltered JSONL records must be a JSON object "
                    f"at line {line_number}"
                )
            rows.append(
                _parse_prefiltered_row(
                    raw,
                    line_number=line_number,
                    episode_lengths=episode_lengths,
                    action_horizon=action_horizon,
                    seen=seen,
                )
            )
    else:
        input_format = "csv"
        reader = csv.DictReader(StringIO(text, newline=""))
        columns = set(reader.fieldnames or [])
        missing = sorted(set(PREFILTERED_REQUIRED_FIELDS) - columns)
        if missing:
            raise PriorSelectionError(
                f"prefiltered CSV is missing required fields: {missing}"
            )
        for raw in reader:
            rows.append(
                _parse_prefiltered_row(
                    raw,
                    line_number=reader.line_num,
                    episode_lengths=episode_lengths,
                    action_horizon=action_horizon,
                    seen=seen,
                )
            )
    if not rows:
        raise PriorSelectionError(f"prefiltered selection is empty: {path}")
    return input_format, source_sha256, rows


def _selection_digest(
    *,
    source_sha256: str,
    input_format: str,
    action_horizon: int,
    frame_indices: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    identity = {
        "mode": "prefiltered_fragments",
        "source_sha256": source_sha256,
        "input_format": input_format,
        "action_horizon": action_horizon,
        "boundary_policy": "complete_action_window",
        "overlap_policy": "deduplicate_episode_frame_start",
    }
    digest.update(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(np.asarray(frame_indices, dtype="<i8").tobytes())
    return digest.hexdigest()


def load_prefiltered_selection(
    source_path: str | Path,
    metadata: LeRobotV2Metadata,
    *,
    action_horizon: int,
) -> PrefilteredPriorSelection:
    """Load every fragment in a CSV or JSONL file as an already-selected prior."""

    if action_horizon <= 0:
        raise PriorSelectionError("action_horizon must be positive")
    path = Path(source_path).expanduser().resolve()
    input_format, source_sha256, rows = _read_prefiltered_rows(
        path,
        metadata,
        action_horizon=action_horizon,
    )
    frame_indices: set[int] = set()
    selected_episode_ids: set[int] = set()
    for row in rows:
        last_start = row.end_step - action_horizon + 1
        selected_episode_ids.add(row.episode_id)
        global_offset = metadata.global_offsets[row.episode_id]
        frame_indices.update(
            global_offset + frame
            for frame in range(row.start_step, last_start + 1)
        )
    ordered_indices = tuple(sorted(frame_indices))
    return PrefilteredPriorSelection(
        source_path=path,
        source_sha256=source_sha256,
        input_format=input_format,
        selected_fragments=len(rows),
        selected_episodes=len(selected_episode_ids),
        frame_indices=ordered_indices,
        action_horizon=action_horizon,
        selection_sha256=_selection_digest(
            source_sha256=source_sha256,
            input_format=input_format,
            action_horizon=action_horizon,
            frame_indices=ordered_indices,
        ),
    )


def resolve_prior_selection(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    metadata_factory: Any = LeRobotV2Metadata,
) -> PrefilteredPriorSelection | None:
    prefiltered_scores = config["data"]["prior_selection"].get(
        "prefiltered_scores"
    )
    if prefiltered_scores is None:
        return None
    metadata = metadata_factory(paths["prior_dataset"])
    return load_prefiltered_selection(
        paths["prior_prefiltered_scores"],
        metadata,
        action_horizon=int(config["data"]["action_horizon"]),
    )
