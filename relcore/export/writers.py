"""Write LeRobot/SQCN-compatible selection outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def write_selection_outputs(
    root: Path,
    *,
    selected_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    ordered = sorted(all_rows, key=lambda row: str(row["sample_id"]))
    pq.write_table(pa.Table.from_pylist(ordered), root / "all_clips.parquet")
    (root / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
