"""Validation and artifact publishing for SQCN diversity filtering."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .algorithm import (
    PENALTY_LAMBDA,
    WEIGHT_EPSILON,
    _AlgorithmParameters,
    select_diverse_fragments,
)


VERSION = "0.1.0"
REQUIRED_SCORE_COLUMNS = (
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
    "quality",
    "coverage",
    "novelty",
    "sqcn",
)
FILTER_COLUMNS = ("filter_rank", "adjusted_score", "max_penalty")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read completed SQCN manifest {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError(f"SQCN run manifest must have status='complete': {path}")
    return payload


def _load_scores(path: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise ValueError(f"Could not read SQCN scores {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(set(REQUIRED_SCORE_COLUMNS) - set(fieldnames))
        if missing:
            raise ValueError(f"SQCN scores are missing columns: {missing}")
        conflicts = sorted(set(FILTER_COLUMNS) & set(fieldnames))
        if conflicts:
            raise ValueError(f"SQCN source scores already contain filter columns: {conflicts}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"SQCN scores are empty: {path}")
    identifiers: list[str] = []
    scores: list[float] = []
    for line_number, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"Empty sample_id at scores CSV line {line_number}")
        try:
            score = float(row["sqcn"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid sqcn value at scores CSV line {line_number}") from error
        identifiers.append(sample_id)
        scores.append(score)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("SQCN scores contain duplicate sample_id values")
    values = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("SQCN scores must contain finite sqcn values in [0, 1]")
    return rows, fieldnames, values


def _load_embeddings(path: Path, row_count: int) -> np.ndarray:
    try:
        embeddings = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read SQCN embeddings {path}: {error}") from error
    if embeddings.ndim != 2 or embeddings.shape != (row_count, 128):
        raise ValueError(
            f"SQCN embeddings must have shape [{row_count}, 128], got {list(embeddings.shape)}"
        )
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("SQCN embeddings must contain only finite values")
    return embeddings


def _normalize_percent(percent: float) -> tuple[float, Decimal]:
    if isinstance(percent, bool):
        raise ValueError("percent must be finite and in (0, 100]")
    try:
        value = float(percent)
    except (TypeError, ValueError) as error:
        raise ValueError("percent must be finite and in (0, 100]") from error
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise ValueError("percent must be finite and in (0, 100]")
    return value, Decimal(str(percent))


def _percent_tag(percent: float) -> str:
    return format(percent, ".12g").replace(".", "p")


def _selection_sha256(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_scores(
    path: Path,
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
    selected_indices: np.ndarray,
    adjusted_scores: np.ndarray,
    max_penalties: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *FILTER_COLUMNS])
        writer.writeheader()
        for rank, (index, adjusted, penalty) in enumerate(
            zip(selected_indices, adjusted_scores, max_penalties, strict=True),
            start=1,
        ):
            writer.writerow(
                {
                    **rows[int(index)],
                    "filter_rank": rank,
                    "adjusted_score": f"{float(adjusted):.9f}",
                    "max_penalty": f"{float(penalty):.9f}",
                }
            )


def _safe_output_root(input_root: Path, output_dir: str | Path | None, percent: float) -> Path:
    if output_dir is None:
        root = input_root / "filter" / f"top{_percent_tag(percent)}pct"
    else:
        root = Path(output_dir).expanduser().resolve()
    filter_root = input_root / "filter"
    inside_input = root.is_relative_to(input_root)
    inside_filter = root.is_relative_to(filter_root)
    if (inside_input and not inside_filter) or input_root.is_relative_to(root):
        raise ValueError(f"filter output would overlap SQCN source artifacts: {root}")
    return root


def filter_sqcn_run(
    input_dir: str | Path,
    percent: float,
    *,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    force: bool = False,
) -> Path:
    """Filter one completed SQCN run and publish aligned result artifacts."""

    percent_value, percent_decimal = _normalize_percent(percent)
    input_root = Path(input_dir).expanduser().resolve()
    source_manifest_path = input_root / "run_manifest.json"
    scores_path = input_root / "fragment" / "scores.csv"
    embeddings_path = input_root / "fragment" / "embeddings.npy"
    source_manifest = _load_source_manifest(source_manifest_path)
    if source_manifest.get("embedding_dim", 128) != 128:
        raise ValueError("SQCN run manifest embedding_dim must be 128")
    rows, fieldnames, scores = _load_scores(scores_path)
    embeddings = _load_embeddings(embeddings_path, len(rows))
    target_size = int(
        (Decimal(len(rows)) * percent_decimal / Decimal(100)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    output_root = _safe_output_root(input_root, output_dir, percent_value)
    if output_root.exists() and not force:
        raise FileExistsError(
            f"SQCN filter output already exists: {output_root}; pass --force to replace it"
        )

    sample_ids = tuple(row["sample_id"].strip() for row in rows)
    result = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size,
        seed=seed,
    )
    selected_ids = [sample_ids[int(index)] for index in result.selected_indices]
    parameters = _AlgorithmParameters()
    manifest = {
        "version": VERSION,
        "status": "complete",
        "source": {
            "input_dir": str(input_root),
            "run_manifest": str(source_manifest_path),
            "run_manifest_sha256": _sha256(source_manifest_path),
            "scores": str(scores_path),
            "scores_sha256": _sha256(scores_path),
            "embeddings": str(embeddings_path),
            "embeddings_sha256": _sha256(embeddings_path),
        },
        "algorithm": {
            "percent": percent_value,
            "target_size": target_size,
            "seed": result.seed,
            "lambda": PENALTY_LAMBDA,
            "sigma": {
                "policy": "mean_pairwise_euclidean_distance_of_raw_top_100",
                "top_count": min(100, len(rows)),
                "raw": result.sigma_raw,
                "effective": result.sigma_effective,
            },
            "constants": {
                "init_select_size": parameters.init_select_size,
                "new_batch_size": parameters.new_batch_size,
                "high_ref_size": parameters.high_ref_size,
                "random_ref_size": parameters.random_ref_size,
                "candidate_threshold": parameters.candidate_threshold,
                "unseen_rank": parameters.unseen_rank,
                "weight_epsilon": WEIGHT_EPSILON,
            },
            "ordering": [
                "adjusted_score desc",
                "sqcn desc",
                "sample_id asc",
            ],
        },
        "counts": {
            "input_fragments": len(rows),
            "selected_fragments": target_size,
        },
        "score_columns": [*fieldnames, *FILTER_COLUMNS],
        "selection_sha256": _selection_sha256(selected_ids),
        "outputs": {
            "scores": str(output_root / "scores.csv"),
            "embeddings": str(output_root / "embeddings.npy"),
            "manifest": str(output_root / "filter_manifest.json"),
        },
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.sqcn-filter-", dir=output_root.parent)
    )
    try:
        _write_scores(
            temp_root / "scores.csv",
            rows,
            fieldnames,
            result.selected_indices,
            result.adjusted_scores,
            result.max_penalties,
        )
        np.save(temp_root / "embeddings.npy", embeddings[result.selected_indices])
        (temp_root / "filter_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if output_root.exists():
            if output_root.is_dir():
                shutil.rmtree(output_root)
            else:
                output_root.unlink()
        os.replace(temp_root, output_root)
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise
    return output_root
