"""Recompute only the final TDUS column in an existing score CSV."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .tdus import compute_tdus


TDUS_COMPONENTS = ("quality", "coverage", "diversity", "novelty")
REQUIRED_COLUMNS = {*TDUS_COMPONENTS, "tdus"}


def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    missing = [name for name in TDUS_COMPONENTS if name not in weights]
    if missing:
        raise ValueError(f"TDUS weights are missing: {missing}")

    validated: dict[str, float] = {}
    for name in TDUS_COMPONENTS:
        try:
            value = float(weights[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"TDUS weight {name!r} must be a number") from error
        if not math.isfinite(value):
            raise ValueError(f"TDUS weight {name!r} must be finite")
        if value < 0.0:
            raise ValueError(f"TDUS weight {name!r} must be non-negative")
        validated[name] = value

    total = sum(validated.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"TDUS weights must sum to 1, got {total:.12g}")
    return validated


def _parse_component(
    row: Mapping[str, str | None],
    column: str,
    line_number: int,
) -> float:
    raw_value = row.get(column)
    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{column!r} must be a number at CSV line {line_number}"
        ) from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{column!r} must be finite and in [0, 1] at CSV line {line_number}"
        )
    return value


def _write_rows_atomically(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str | None]],
    *,
    force: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --force to replace it"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if output_path.exists() and not force:
            raise FileExistsError(
                f"output already exists: {output_path}; pass --force to replace it"
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def convert_scores(
    input_path: str | Path,
    output_path: str | Path,
    *,
    weights: Mapping[str, float],
    force: bool = False,
) -> int:
    """Write a copy of a TDUS score CSV with only its final score recomputed."""

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise ValueError("input and output paths must be different")
    if not source.is_file():
        raise FileNotFoundError(f"input scores CSV does not exist: {source}")
    if destination.exists() and not force:
        raise FileExistsError(
            f"output already exists: {destination}; pass --force to replace it"
        )

    validated_weights = _validate_weights(weights)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"input scores CSV has no header: {source}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"input scores CSV has duplicate columns: {source}")
        missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(f"input scores CSV is missing columns: {missing}")

        rows: list[dict[str, str | None]] = []
        components: dict[str, list[float]] = {
            name: [] for name in TDUS_COMPONENTS
        }
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"input scores CSV has extra fields at line {line_number}"
                )
            if any(value is None for value in row.values()):
                raise ValueError(
                    f"input scores CSV has missing fields at line {line_number}"
                )
            rows.append(row)
            for name in TDUS_COMPONENTS:
                components[name].append(_parse_component(row, name, line_number))

    if not rows:
        raise ValueError(f"input scores CSV has no data rows: {source}")

    tdus_values = compute_tdus(
        np.asarray(components["quality"], dtype=np.float64),
        np.asarray(components["coverage"], dtype=np.float64),
        np.asarray(components["diversity"], dtype=np.float64),
        np.asarray(components["novelty"], dtype=np.float64),
        validated_weights,
    )
    for row, tdus_value in zip(rows, tdus_values, strict=True):
        row["tdus"] = f"{float(tdus_value):.9f}"

    _write_rows_atomically(
        destination,
        fieldnames,
        rows,
        force=force,
    )
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute only the final TDUS score in an existing CSV"
    )
    parser.add_argument("--input", required=True, help="existing TDUS score CSV")
    parser.add_argument("--output", required=True, help="new score CSV to write")
    parser.add_argument("--quality-weight", type=float, required=True)
    parser.add_argument("--coverage-weight", type=float, required=True)
    parser.add_argument("--diversity-weight", type=float, required=True)
    parser.add_argument("--novelty-weight", type=float, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the output path if it already exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    weights = {
        "quality": args.quality_weight,
        "coverage": args.coverage_weight,
        "diversity": args.diversity_weight,
        "novelty": args.novelty_weight,
    }
    try:
        row_count = convert_scores(
            args.input,
            args.output,
            weights=weights,
            force=args.force,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"converted={row_count} output={Path(args.output).expanduser().resolve()} "
        f"weights={weights}"
    )


if __name__ == "__main__":
    main()
