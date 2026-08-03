from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .data import LiberoDataError
from .lerobot_v2 import LeRobotV2Metadata


SCORE_COLUMNS = {
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
    "quality",
    "coverage",
    "diversity",
    "novelty",
    "tdus",
}
UTILITY_COLUMNS = ("quality", "coverage", "diversity", "novelty", "tdus")
SQCN_SCORE_COLUMNS = {
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
    "quality",
    "coverage",
    "novelty",
    "sqcn",
    "filter_rank",
    "adjusted_score",
    "knn_penalty",
}
SQCN_UNIT_COLUMNS = ("quality", "coverage", "novelty", "sqcn", "knn_penalty")


class PriorSelectionError(LiberoDataError):
    """Raised when TDUS scores cannot safely select LIBERO prior frames."""


@dataclass(frozen=True)
class _ScoreRow:
    sample_id: str
    episode_id: int
    start_step: int
    end_step: int
    length: int
    tdus: float


@dataclass(frozen=True)
class _PrefilteredScoreRow:
    sample_id: str
    episode_id: int
    start_step: int
    end_step: int
    length: int


@dataclass(frozen=True)
class PriorSelection:
    scores_path: Path
    scores_sha256: str
    run_manifest_path: Path
    top_percent: float
    total_chunks: int
    selected_chunks: int
    selected_sample_ids: tuple[str, ...]
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
            "scores_path": str(self.scores_path),
            "scores_sha256": self.scores_sha256,
            "run_manifest_path": str(self.run_manifest_path),
            "top_percent": self.top_percent,
            "total_chunks": self.total_chunks,
            "selected_chunks": self.selected_chunks,
            "selected_episodes": self.selected_episodes,
            "training_starts": self.training_starts,
            "action_horizon": self.action_horizon,
            "ordering": ["tdus desc", "length asc", "sample_id asc"],
            "boundary_policy": "complete_action_window",
            "overlap_policy": "deduplicate_episode_frame_start",
            "selection_sha256": self.selection_sha256,
        }


@dataclass(frozen=True)
class PrefilteredPriorSelection:
    scores_path: Path
    scores_sha256: str
    filter_manifest_path: Path
    filter_manifest_sha256: str
    run_manifest_path: Path
    run_manifest_sha256: str
    top_percent: float
    total_chunks: int
    selected_chunks: int
    selected_sample_ids: tuple[str, ...]
    selected_episodes: int
    frame_indices: tuple[int, ...]
    action_horizon: int
    ordering: tuple[str, ...]
    selection_sha256: str

    @property
    def training_starts(self) -> int:
        return len(self.frame_indices)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": "sqcn_prefiltered",
            "scores_path": str(self.scores_path),
            "scores_sha256": self.scores_sha256,
            "filter_manifest_path": str(self.filter_manifest_path),
            "filter_manifest_sha256": self.filter_manifest_sha256,
            "run_manifest_path": str(self.run_manifest_path),
            "run_manifest_sha256": self.run_manifest_sha256,
            "top_percent": self.top_percent,
            "total_fragments": self.total_chunks,
            "selected_fragments": self.selected_chunks,
            "selected_episodes": self.selected_episodes,
            "training_starts": self.training_starts,
            "action_horizon": self.action_horizon,
            "ordering": list(self.ordering),
            "boundary_policy": "complete_action_window",
            "overlap_policy": "deduplicate_episode_frame_start",
            "selection_sha256": self.selection_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run_manifest(scores_path: Path, prior_root: Path) -> Path:
    manifest_path = scores_path.parent.parent / "run_manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise PriorSelectionError(
            f"Could not read TDUS run manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise PriorSelectionError(f"TDUS run manifest must be an object: {manifest_path}")
    if manifest.get("dataset_name") != prior_root.name:
        raise PriorSelectionError(
            f"TDUS dataset_name={manifest.get('dataset_name')!r} does not match "
            f"prior dataset {prior_root.name!r}"
        )
    source_value = manifest.get("dataset_path")
    if not isinstance(source_value, str) or not source_value.strip():
        raise PriorSelectionError(f"TDUS run manifest has no dataset_path: {manifest_path}")
    source_path = Path(source_value).expanduser().resolve()
    if source_path != prior_root:
        raise PriorSelectionError(
            f"TDUS scores were computed from {source_path}, but training prior is {prior_root}"
        )
    modes = manifest.get("modes")
    if not isinstance(modes, list) or "chunk" not in modes:
        raise PriorSelectionError(
            f"TDUS run manifest does not declare chunk scores: {manifest_path}"
        )
    return manifest_path


def _parse_int(row: Mapping[str, str], column: str, line_number: int) -> int:
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise PriorSelectionError(
            f"Invalid integer in {column!r} at scores CSV line {line_number}"
        ) from error


def _parse_utility(row: Mapping[str, str], column: str, line_number: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise PriorSelectionError(
            f"Invalid number in {column!r} at scores CSV line {line_number}"
        ) from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise PriorSelectionError(
            f"{column!r} must be finite and in [0, 1] at scores CSV line {line_number}"
        )
    return value


def _parse_finite(row: Mapping[str, str], column: str, line_number: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise PriorSelectionError(
            f"Invalid number in {column!r} at scores CSV line {line_number}"
        ) from error
    if not math.isfinite(value):
        raise PriorSelectionError(
            f"{column!r} must be finite at scores CSV line {line_number}"
        )
    return value


def _read_json_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise PriorSelectionError(f"Could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PriorSelectionError(f"{label} must be an object: {path}")
    return value


def _manifest_path(value: Any, *, relative_to: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PriorSelectionError(f"SQCN filter manifest has no {label}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _read_prefiltered_sqcn_scores(
    scores_path: Path,
    metadata: LeRobotV2Metadata,
) -> list[_PrefilteredScoreRow]:
    episode_lengths = {
        episode.episode_index: episode.length for episode in metadata.episodes
    }
    rows: list[_PrefilteredScoreRow] = []
    sample_ids: set[str] = set()
    try:
        handle = scores_path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise PriorSelectionError(
            f"Could not read SQCN scores {scores_path}: {error}"
        ) from error
    with handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(SQCN_SCORE_COLUMNS - columns)
        if missing:
            raise PriorSelectionError(f"SQCN scores are missing columns: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            sample_id = str(raw.get("sample_id", "")).strip()
            if not sample_id:
                raise PriorSelectionError(
                    f"Empty sample_id at scores CSV line {line_number}"
                )
            if sample_id in sample_ids:
                raise PriorSelectionError(f"Duplicate sample_id in SQCN scores: {sample_id}")
            sample_ids.add(sample_id)
            episode_id = _parse_int(raw, "episode_id", line_number)
            start_step = _parse_int(raw, "start_step", line_number)
            end_step = _parse_int(raw, "end_step", line_number)
            length = _parse_int(raw, "length", line_number)
            filter_rank = _parse_int(raw, "filter_rank", line_number)
            if filter_rank != line_number - 1:
                raise PriorSelectionError(
                    "SQCN filter_rank must be contiguous from 1 in CSV row order"
                )
            for column in SQCN_UNIT_COLUMNS:
                _parse_utility(raw, column, line_number)
            _parse_finite(raw, "adjusted_score", line_number)
            if episode_id not in episode_lengths:
                raise PriorSelectionError(
                    f"Unknown episode_id={episode_id} at scores CSV line {line_number}"
                )
            if start_step < 0 or end_step < start_step:
                raise PriorSelectionError(
                    f"Invalid frame range [{start_step}, {end_step}] "
                    f"at scores CSV line {line_number}"
                )
            if end_step >= episode_lengths[episode_id]:
                raise PriorSelectionError(
                    f"Fragment end_step={end_step} exceeds episode {episode_id} "
                    f"length={episode_lengths[episode_id]}"
                )
            if length <= 0 or length != end_step - start_step + 1:
                raise PriorSelectionError(
                    f"Fragment length={length} does not match inclusive frame range "
                    f"at scores CSV line {line_number}"
                )
            expected_id = (
                f"ep{episode_id:06d}_fragment_{start_step:06d}_{end_step:06d}"
            )
            if sample_id != expected_id:
                raise PriorSelectionError(
                    f"sample_id={sample_id!r} does not match {expected_id!r}"
                )
            rows.append(
                _PrefilteredScoreRow(
                    sample_id=sample_id,
                    episode_id=episode_id,
                    start_step=start_step,
                    end_step=end_step,
                    length=length,
                )
            )
    if not rows:
        raise PriorSelectionError(f"SQCN scores are empty: {scores_path}")
    return rows


def _read_scores(scores_path: Path, metadata: LeRobotV2Metadata) -> list[_ScoreRow]:
    episode_lengths = {
        episode.episode_index: episode.length for episode in metadata.episodes
    }
    rows: list[_ScoreRow] = []
    sample_ids: set[str] = set()
    try:
        handle = scores_path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise PriorSelectionError(f"Could not read TDUS scores {scores_path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(SCORE_COLUMNS - columns)
        if missing:
            raise PriorSelectionError(f"TDUS scores are missing columns: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            sample_id = str(raw.get("sample_id", "")).strip()
            if not sample_id:
                raise PriorSelectionError(
                    f"Empty sample_id at scores CSV line {line_number}"
                )
            if sample_id in sample_ids:
                raise PriorSelectionError(f"Duplicate sample_id in TDUS scores: {sample_id}")
            sample_ids.add(sample_id)
            episode_id = _parse_int(raw, "episode_id", line_number)
            start_step = _parse_int(raw, "start_step", line_number)
            end_step = _parse_int(raw, "end_step", line_number)
            length = _parse_int(raw, "length", line_number)
            utilities = {
                column: _parse_utility(raw, column, line_number)
                for column in UTILITY_COLUMNS
            }
            if episode_id not in episode_lengths:
                raise PriorSelectionError(
                    f"Unknown episode_id={episode_id} at scores CSV line {line_number}"
                )
            if start_step < 0 or end_step < start_step:
                raise PriorSelectionError(
                    f"Invalid frame range [{start_step}, {end_step}] "
                    f"at scores CSV line {line_number}"
                )
            if end_step >= episode_lengths[episode_id]:
                raise PriorSelectionError(
                    f"Chunk end_step={end_step} exceeds episode {episode_id} "
                    f"length={episode_lengths[episode_id]}"
                )
            if length <= 0 or length != end_step - start_step + 1:
                raise PriorSelectionError(
                    f"Chunk length={length} does not match inclusive frame range "
                    f"at scores CSV line {line_number}"
                )
            expected_id = (
                f"ep{episode_id:06d}_chunk_{start_step:06d}_{end_step:06d}"
            )
            if sample_id != expected_id:
                raise PriorSelectionError(
                    f"sample_id={sample_id!r} does not match {expected_id!r}"
                )
            rows.append(
                _ScoreRow(
                    sample_id=sample_id,
                    episode_id=episode_id,
                    start_step=start_step,
                    end_step=end_step,
                    length=length,
                    tdus=utilities["tdus"],
                )
            )
    if not rows:
        raise PriorSelectionError(f"TDUS scores are empty: {scores_path}")
    return rows


def _selection_digest(
    *,
    scores_sha256: str,
    top_percent: float,
    action_horizon: int,
    frame_indices: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    identity = {
        "scores_sha256": scores_sha256,
        "top_percent": top_percent,
        "action_horizon": action_horizon,
        "ordering": ["tdus desc", "length asc", "sample_id asc"],
        "boundary_policy": "complete_action_window",
        "overlap_policy": "deduplicate_episode_frame_start",
    }
    digest.update(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(np.asarray(frame_indices, dtype="<i8").tobytes())
    return digest.hexdigest()


def load_prior_selection(
    scores_path: str | Path,
    top_percent: float,
    metadata: LeRobotV2Metadata,
    *,
    action_horizon: int,
) -> PriorSelection:
    """Load, validate, rank, and map TDUS chunks to unique global frame starts."""

    try:
        percent = float(top_percent)
    except (TypeError, ValueError) as error:
        raise PriorSelectionError("prior top percent must be numeric") from error
    if not math.isfinite(percent) or not 0.0 < percent <= 100.0:
        raise PriorSelectionError("prior top percent must be finite and in (0, 100]")
    if action_horizon <= 0:
        raise PriorSelectionError("action_horizon must be positive")

    path = Path(scores_path).expanduser().resolve()
    manifest_path = _load_run_manifest(path, metadata.root)
    rows = _read_scores(path, metadata)
    rows.sort(key=lambda row: (-row.tdus, row.length, row.sample_id))
    selected_count = int(
        (Decimal(len(rows)) * Decimal(str(percent)) / Decimal(100)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    selected = rows[:selected_count]

    frame_indices: set[int] = set()
    selected_episode_ids: set[int] = set()
    for row in selected:
        last_start = row.end_step - action_horizon + 1
        if last_start < row.start_step:
            continue
        global_offset = metadata.global_offsets[row.episode_id]
        selected_episode_ids.add(row.episode_id)
        frame_indices.update(
            global_offset + frame
            for frame in range(row.start_step, last_start + 1)
        )
    ordered_indices = tuple(sorted(frame_indices))
    if not ordered_indices:
        raise PriorSelectionError(
            "Selected TDUS chunks contain no complete action windows"
        )

    scores_sha256 = _sha256(path)
    return PriorSelection(
        scores_path=path,
        scores_sha256=scores_sha256,
        run_manifest_path=manifest_path,
        top_percent=percent,
        total_chunks=len(rows),
        selected_chunks=selected_count,
        selected_sample_ids=tuple(row.sample_id for row in selected),
        selected_episodes=len(selected_episode_ids),
        frame_indices=ordered_indices,
        action_horizon=action_horizon,
        selection_sha256=_selection_digest(
            scores_sha256=scores_sha256,
            top_percent=percent,
            action_horizon=action_horizon,
            frame_indices=ordered_indices,
        ),
    )


def _manifest_positive_int(
    values: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PriorSelectionError(f"{label}.{key} must be a positive integer")
    return value


def _sample_ids_sha256(rows: list[_PrefilteredScoreRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _prefiltered_selection_digest(
    *,
    scores_sha256: str,
    filter_manifest_sha256: str,
    run_manifest_sha256: str,
    action_horizon: int,
    frame_indices: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    identity = {
        "mode": "sqcn_prefiltered",
        "scores_sha256": scores_sha256,
        "filter_manifest_sha256": filter_manifest_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "action_horizon": action_horizon,
        "boundary_policy": "complete_action_window",
        "overlap_policy": "deduplicate_episode_frame_start",
    }
    digest.update(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(np.asarray(frame_indices, dtype="<i8").tobytes())
    return digest.hexdigest()


def load_prefiltered_sqcn_selection(
    scores_path: str | Path,
    metadata: LeRobotV2Metadata,
    *,
    action_horizon: int,
) -> PrefilteredPriorSelection:
    """Load an already-filtered SQCN fragment list without ranking it again."""

    if action_horizon <= 0:
        raise PriorSelectionError("action_horizon must be positive")
    path = Path(scores_path).expanduser().resolve()
    filter_manifest_path = path.parent / "filter_manifest.json"
    filter_manifest = _read_json_manifest(
        filter_manifest_path,
        "SQCN filter manifest",
    )
    if filter_manifest.get("status") != "complete":
        raise PriorSelectionError(
            f"SQCN filter manifest must have status='complete': {filter_manifest_path}"
        )

    outputs = filter_manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise PriorSelectionError("SQCN filter manifest outputs must be an object")
    declared_scores = _manifest_path(
        outputs.get("scores"),
        relative_to=filter_manifest_path.parent,
        label="outputs.scores",
    )
    if declared_scores != path:
        raise PriorSelectionError(
            f"SQCN filter manifest declares scores {declared_scores}, but loaded {path}"
        )

    source = filter_manifest.get("source")
    if not isinstance(source, dict):
        raise PriorSelectionError("SQCN filter manifest source must be an object")
    run_manifest_path = _manifest_path(
        source.get("run_manifest"),
        relative_to=filter_manifest_path.parent,
        label="source.run_manifest",
    )
    declared_run_sha256 = source.get("run_manifest_sha256")
    try:
        run_manifest_sha256 = _sha256(run_manifest_path)
    except OSError as error:
        raise PriorSelectionError(
            f"Could not read SQCN run manifest {run_manifest_path}: {error}"
        ) from error
    if declared_run_sha256 != run_manifest_sha256:
        raise PriorSelectionError(
            "SQCN source run manifest SHA256 does not match filter manifest"
        )
    run_manifest = _read_json_manifest(run_manifest_path, "SQCN run manifest")
    if run_manifest.get("status") != "complete":
        raise PriorSelectionError(
            f"SQCN run manifest must have status='complete': {run_manifest_path}"
        )
    if run_manifest.get("dataset_name") != metadata.root.name:
        raise PriorSelectionError(
            f"SQCN dataset_name={run_manifest.get('dataset_name')!r} does not match "
            f"prior dataset {metadata.root.name!r}"
        )
    source_value = run_manifest.get("dataset_path")
    source_path = _manifest_path(
        source_value,
        relative_to=run_manifest_path.parent,
        label="run_manifest.dataset_path",
    )
    if source_path != metadata.root:
        raise PriorSelectionError(
            f"SQCN scores were computed from {source_path}, but training prior is "
            f"{metadata.root}"
        )

    algorithm = filter_manifest.get("algorithm")
    counts = filter_manifest.get("counts")
    if not isinstance(algorithm, dict) or not isinstance(counts, dict):
        raise PriorSelectionError(
            "SQCN filter manifest algorithm and counts must be objects"
        )
    try:
        top_percent = float(algorithm["percent"])
    except (KeyError, TypeError, ValueError) as error:
        raise PriorSelectionError("SQCN algorithm.percent must be numeric") from error
    if not math.isfinite(top_percent) or not 0.0 < top_percent <= 100.0:
        raise PriorSelectionError("SQCN algorithm.percent must be in (0, 100]")
    input_fragments = _manifest_positive_int(
        counts,
        "input_fragments",
        label="SQCN counts",
    )
    selected_fragments = _manifest_positive_int(
        counts,
        "selected_fragments",
        label="SQCN counts",
    )
    target_size = _manifest_positive_int(
        algorithm,
        "target_size",
        label="SQCN algorithm",
    )
    expected_size = int(
        (
            Decimal(input_fragments)
            * Decimal(str(top_percent))
            / Decimal(100)
        ).to_integral_value(rounding=ROUND_CEILING)
    )
    if selected_fragments != target_size or selected_fragments != expected_size:
        raise PriorSelectionError(
            "SQCN selected fragment counts do not match algorithm percent"
        )
    ordering = algorithm.get("ordering")
    if (
        not isinstance(ordering, list)
        or not ordering
        or not all(isinstance(value, str) and value for value in ordering)
    ):
        raise PriorSelectionError("SQCN algorithm.ordering must be a non-empty string list")

    rows = _read_prefiltered_sqcn_scores(path, metadata)
    if len(rows) != selected_fragments:
        raise PriorSelectionError(
            f"SQCN scores contain {len(rows)} rows, expected {selected_fragments}"
        )
    selection_sha256 = _sample_ids_sha256(rows)
    if filter_manifest.get("selection_sha256") != selection_sha256:
        raise PriorSelectionError(
            "SQCN scores sample IDs do not match filter manifest selection_sha256"
        )

    frame_indices: set[int] = set()
    selected_episode_ids: set[int] = set()
    for row in rows:
        last_start = row.end_step - action_horizon + 1
        if last_start < row.start_step:
            continue
        global_offset = metadata.global_offsets[row.episode_id]
        selected_episode_ids.add(row.episode_id)
        frame_indices.update(
            global_offset + frame
            for frame in range(row.start_step, last_start + 1)
        )
    ordered_indices = tuple(sorted(frame_indices))
    if not ordered_indices:
        raise PriorSelectionError(
            "Selected SQCN fragments contain no complete action windows"
        )

    scores_sha256 = _sha256(path)
    filter_manifest_sha256 = _sha256(filter_manifest_path)
    return PrefilteredPriorSelection(
        scores_path=path,
        scores_sha256=scores_sha256,
        filter_manifest_path=filter_manifest_path,
        filter_manifest_sha256=filter_manifest_sha256,
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=run_manifest_sha256,
        top_percent=top_percent,
        total_chunks=input_fragments,
        selected_chunks=selected_fragments,
        selected_sample_ids=tuple(row.sample_id for row in rows),
        selected_episodes=len(selected_episode_ids),
        frame_indices=ordered_indices,
        action_horizon=action_horizon,
        ordering=tuple(ordering),
        selection_sha256=_prefiltered_selection_digest(
            scores_sha256=scores_sha256,
            filter_manifest_sha256=filter_manifest_sha256,
            run_manifest_sha256=run_manifest_sha256,
            action_horizon=action_horizon,
            frame_indices=ordered_indices,
        ),
    )


def resolve_prior_selection(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> PriorSelection | None:
    selection = config["data"]["prior_selection"]
    percent = selection.get("top_percent")
    prefiltered = selection.get("prefiltered", False)
    if percent is None and not prefiltered:
        return None
    metadata = LeRobotV2Metadata(paths["prior_dataset"])
    if prefiltered:
        return load_prefiltered_sqcn_selection(
            paths["prior_scores"],
            metadata,
            action_horizon=int(config["data"]["action_horizon"]),
        )
    return load_prior_selection(
        paths["prior_scores"],
        float(percent),
        metadata,
        action_horizon=int(config["data"]["action_horizon"]),
    )
