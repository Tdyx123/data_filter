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
from .sampling import ACTION_WINDOW_POLICY, expand_fragment_frame_indices


PREFILTERED_REQUIRED_FIELDS = (
    "episode_id",
    "start_step",
    "end_step",
)
DATAMIL_TRAJECTORY_REQUIRED_FIELDS = (
    "trajectory_id",
    "num_frames",
)

_FRAGMENT_SOURCE_SCHEMA = "fragment_range"
_DATAMIL_SOURCE_SCHEMA = "datamil_trajectory"


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
    source_schema: str = _FRAGMENT_SOURCE_SCHEMA

    @property
    def training_starts(self) -> int:
        return len(self.frame_indices)

    def as_manifest(self) -> dict[str, Any]:
        is_datamil = self.source_schema == _DATAMIL_SOURCE_SCHEMA
        manifest = {
            "enabled": True,
            "mode": (
                "prefiltered_trajectories" if is_datamil else "prefiltered_fragments"
            ),
            "input_format": self.input_format,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "required_fields": list(
                DATAMIL_TRAJECTORY_REQUIRED_FIELDS
                if is_datamil
                else PREFILTERED_REQUIRED_FIELDS
            ),
        }
        if is_datamil:
            manifest["source_schema"] = _DATAMIL_SOURCE_SCHEMA
            manifest["selected_trajectories"] = self.selected_fragments
        else:
            manifest["selected_fragments"] = self.selected_fragments
        manifest.update(
            {
                "selected_episodes": self.selected_episodes,
                "training_starts": self.training_starts,
                "action_horizon": self.action_horizon,
                "ordering": ["source row order"],
                "boundary_policy": ACTION_WINDOW_POLICY,
                "overlap_policy": "deduplicate_episode_frame_start",
                "selection_sha256": self.selection_sha256,
            }
        )
        return manifest


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


def _strict_json_int(value: Any, field: str, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PriorSelectionError(
            f"invalid integer in {field!r} at prefiltered selection line {line_number}"
        )
    return value


def _parse_prefiltered_row(
    raw: Mapping[str, Any],
    *,
    line_number: int,
    episode_lengths: Mapping[int, int],
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
    return _PrefilteredFragmentRow(
        episode_id=episode_id,
        start_step=start_step,
        end_step=end_step,
    )


def _detect_jsonl_schema(raw: Mapping[str, Any], *, line_number: int) -> str:
    fields = set(raw)
    fragment_fields = set(PREFILTERED_REQUIRED_FIELDS)
    datamil_fields = set(DATAMIL_TRAJECTORY_REQUIRED_FIELDS)
    if fragment_fields.issubset(fields):
        return _FRAGMENT_SOURCE_SCHEMA
    if datamil_fields.issubset(fields):
        return _DATAMIL_SOURCE_SCHEMA

    has_fragment_field = bool(fragment_fields.intersection(fields))
    has_datamil_field = bool(datamil_fields.intersection(fields))
    if has_fragment_field and has_datamil_field:
        raise PriorSelectionError(
            "prefiltered JSONL cannot mix fragment and DataMIL trajectory schemas "
            f"at line {line_number}"
        )
    if has_datamil_field:
        return _DATAMIL_SOURCE_SCHEMA
    return _FRAGMENT_SOURCE_SCHEMA


def _parse_datamil_trajectory_row(
    raw: Mapping[str, Any],
    *,
    line_number: int,
    episode_lengths: Mapping[int, int],
    seen: set[int],
) -> _PrefilteredFragmentRow:
    missing = [
        field for field in DATAMIL_TRAJECTORY_REQUIRED_FIELDS if field not in raw
    ]
    if missing:
        raise PriorSelectionError(
            "prefiltered selection is missing required fields "
            f"{missing} at line {line_number}"
        )
    trajectory_id = _strict_json_int(
        raw["trajectory_id"], "trajectory_id", line_number
    )
    num_frames = _strict_json_int(raw["num_frames"], "num_frames", line_number)
    if trajectory_id in seen:
        raise PriorSelectionError(
            f"duplicate trajectory_id={trajectory_id} at prefiltered selection "
            f"line {line_number}"
        )
    seen.add(trajectory_id)
    if trajectory_id not in episode_lengths:
        raise PriorSelectionError(
            f"unknown trajectory_id={trajectory_id} at prefiltered selection "
            f"line {line_number}"
        )
    if num_frames <= 0:
        raise PriorSelectionError(
            f"num_frames must be positive at prefiltered selection line {line_number}"
        )
    episode_length = episode_lengths[trajectory_id]
    if num_frames != episode_length:
        raise PriorSelectionError(
            f"num_frames={num_frames} does not match episode {trajectory_id} "
            f"length={episode_length} at prefiltered selection line {line_number}"
        )
    return _PrefilteredFragmentRow(
        episode_id=trajectory_id,
        start_step=0,
        end_step=num_frames - 1,
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
) -> tuple[str, str, str, list[_PrefilteredFragmentRow]]:
    text, source_sha256 = _read_source(path)
    first_content = next((line.lstrip() for line in text.splitlines() if line.strip()), None)
    if first_content is None:
        raise PriorSelectionError(f"prefiltered selection is empty: {path}")

    episode_lengths = {
        episode.episode_index: episode.length for episode in metadata.episodes
    }
    seen: set[tuple[int, int, int]] = set()
    seen_trajectories: set[int] = set()
    rows: list[_PrefilteredFragmentRow] = []
    if first_content.startswith("{"):
        input_format = "jsonl"
        source_schema: str | None = None
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
            row_schema = _detect_jsonl_schema(raw, line_number=line_number)
            if source_schema is None:
                source_schema = row_schema
            elif row_schema != source_schema:
                raise PriorSelectionError(
                    "prefiltered JSONL cannot mix fragment and DataMIL trajectory "
                    f"schemas at line {line_number}"
                )
            if source_schema == _DATAMIL_SOURCE_SCHEMA:
                row = _parse_datamil_trajectory_row(
                    raw,
                    line_number=line_number,
                    episode_lengths=episode_lengths,
                    seen=seen_trajectories,
                )
            else:
                row = _parse_prefiltered_row(
                    raw,
                    line_number=line_number,
                    episode_lengths=episode_lengths,
                    seen=seen,
                )
            rows.append(row)
        assert source_schema is not None
    else:
        input_format = "csv"
        source_schema = _FRAGMENT_SOURCE_SCHEMA
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
                    seen=seen,
                )
            )
    if not rows:
        raise PriorSelectionError(f"prefiltered selection is empty: {path}")
    return input_format, source_sha256, source_schema, rows


def _selection_digest(
    *,
    source_sha256: str,
    input_format: str,
    source_schema: str,
    action_horizon: int,
    frame_indices: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    is_datamil = source_schema == _DATAMIL_SOURCE_SCHEMA
    identity = {
        "mode": "prefiltered_trajectories" if is_datamil else "prefiltered_fragments",
        "source_sha256": source_sha256,
        "input_format": input_format,
        "action_horizon": action_horizon,
        "boundary_policy": ACTION_WINDOW_POLICY,
        "overlap_policy": "deduplicate_episode_frame_start",
    }
    if is_datamil:
        identity["source_schema"] = _DATAMIL_SOURCE_SCHEMA
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
    """Load selected fragments or DataMIL trajectories as prior training starts."""

    if action_horizon <= 0:
        raise PriorSelectionError("action_horizon must be positive")
    path = Path(source_path).expanduser().resolve()
    input_format, source_sha256, source_schema, rows = _read_prefiltered_rows(
        path,
        metadata,
    )
    selected_episode_ids: set[int] = set()
    for row in rows:
        selected_episode_ids.add(row.episode_id)
    if source_schema == _DATAMIL_SOURCE_SCHEMA:
        ordered_indices = tuple(
            int(metadata.global_offsets[row.episode_id]) + frame
            for row in rows
            for frame in range(row.start_step, row.end_step + 1)
        )
    else:
        ordered_indices = expand_fragment_frame_indices(
            tuple((row.episode_id, row.start_step, row.end_step) for row in rows),
            metadata.global_offsets,
        )
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
            source_schema=source_schema,
            action_horizon=action_horizon,
            frame_indices=ordered_indices,
        ),
        source_schema=source_schema,
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
