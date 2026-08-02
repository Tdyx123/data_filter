from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tdus.convert import convert_scores, main


FIELDNAMES = [
    "sample_id",
    "episode_id",
    "quality",
    "coverage",
    "diversity",
    "novelty",
    "tdus",
    "note",
]
WEIGHTS = {
    "quality": 0.4,
    "coverage": 0.3,
    "diversity": 0.2,
    "novelty": 0.1,
}


def _write_scores(
    path: Path,
    *,
    fieldnames: list[str] | None = None,
    rows: list[dict[str, str]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or FIELDNAMES
    values = rows if rows is not None else [
        {
            "sample_id": "sample-b",
            "episode_id": "2",
            "quality": "0.8000",
            "coverage": "0.400",
            "diversity": "0.20",
            "novelty": "0.1",
            "tdus": "0.999999999",
            "note": "keep, exactly",
        },
        {
            "sample_id": "sample-a",
            "episode_id": "1",
            "quality": "0.1000",
            "coverage": "0.200",
            "diversity": "0.30",
            "novelty": "0.4",
            "tdus": "0.000000000",
            "note": "unchanged",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _read_scores(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_convert_recomputes_only_tdus_and_preserves_order(tmp_path: Path):
    source = tmp_path / "scores.csv"
    output = tmp_path / "scores_custom.csv"
    _write_scores(source)
    source_fields, source_rows = _read_scores(source)

    assert convert_scores(source, output, weights=WEIGHTS) == 2

    output_fields, output_rows = _read_scores(output)
    assert output_fields == source_fields
    assert [row["sample_id"] for row in output_rows] == ["sample-b", "sample-a"]
    for source_row, output_row in zip(source_rows, output_rows, strict=True):
        assert {
            key: value for key, value in output_row.items() if key != "tdus"
        } == {
            key: value for key, value in source_row.items() if key != "tdus"
        }
        assert len(output_row["tdus"].partition(".")[2]) == 9
    assert float(output_rows[0]["tdus"]) == pytest.approx(0.49)
    assert float(output_rows[1]["tdus"]) == pytest.approx(0.20)


@pytest.mark.parametrize(
    "weights, message",
    [
        ({**WEIGHTS, "quality": -0.1, "coverage": 0.8}, "non-negative"),
        ({**WEIGHTS, "quality": float("nan")}, "finite"),
        ({**WEIGHTS, "quality": float("inf")}, "finite"),
        ({**WEIGHTS, "quality": 0.5}, "sum to 1"),
    ],
)
def test_convert_rejects_invalid_weights(
    tmp_path: Path,
    weights: dict[str, float],
    message: str,
):
    source = tmp_path / "scores.csv"
    _write_scores(source)

    with pytest.raises(ValueError, match=message):
        convert_scores(source, tmp_path / "output.csv", weights=weights)


def test_convert_rejects_missing_columns(tmp_path: Path):
    source = tmp_path / "scores.csv"
    _write_scores(
        source,
        fieldnames=[name for name in FIELDNAMES if name != "coverage"],
    )

    with pytest.raises(ValueError, match="missing columns.*coverage"):
        convert_scores(source, tmp_path / "output.csv", weights=WEIGHTS)


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("quality", "not-a-number", "must be a number"),
        ("coverage", "nan", "finite and in"),
        ("diversity", "inf", "finite and in"),
        ("novelty", "1.01", "finite and in"),
    ],
)
def test_convert_rejects_invalid_components(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
):
    source = tmp_path / "scores.csv"
    row = {
        "sample_id": "sample-a",
        "episode_id": "1",
        "quality": "0.1",
        "coverage": "0.2",
        "diversity": "0.3",
        "novelty": "0.4",
        "tdus": "0.5",
        "note": "",
    }
    row[column] = value
    _write_scores(source, rows=[row])

    with pytest.raises(ValueError, match=message):
        convert_scores(source, tmp_path / "output.csv", weights=WEIGHTS)


def test_convert_rejects_empty_csv(tmp_path: Path):
    source = tmp_path / "scores.csv"
    _write_scores(source, rows=[])

    with pytest.raises(ValueError, match="no data rows"):
        convert_scores(source, tmp_path / "output.csv", weights=WEIGHTS)


def test_convert_rejects_rows_with_missing_fields(tmp_path: Path):
    source = tmp_path / "scores.csv"
    source.write_text(
        ",".join(FIELDNAMES) + "\n"
        "sample-a,1,0.1,0.2,0.3,0.4,0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing fields"):
        convert_scores(source, tmp_path / "output.csv", weights=WEIGHTS)


def test_convert_refuses_existing_output_unless_forced(tmp_path: Path):
    source = tmp_path / "scores.csv"
    output = tmp_path / "scores_custom.csv"
    _write_scores(source)
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --force"):
        convert_scores(source, output, weights=WEIGHTS)
    assert output.read_text(encoding="utf-8") == "existing"

    assert convert_scores(source, output, weights=WEIGHTS, force=True) == 2
    assert _read_scores(output)[1][0]["sample_id"] == "sample-b"


def test_convert_always_rejects_same_input_and_output(tmp_path: Path):
    source = tmp_path / "scores.csv"
    _write_scores(source)
    original = source.read_bytes()

    with pytest.raises(ValueError, match="must be different"):
        convert_scores(source, source, weights=WEIGHTS, force=True)
    assert source.read_bytes() == original


def test_convert_cli_writes_new_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    source = tmp_path / "scores.csv"
    output = tmp_path / "custom" / "scores.csv"
    _write_scores(source)

    main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--quality-weight",
            "0.4",
            "--coverage-weight",
            "0.3",
            "--diversity-weight",
            "0.2",
            "--novelty-weight",
            "0.1",
        ]
    )

    assert output.is_file()
    assert f"converted=2 output={output.resolve()}" in capsys.readouterr().out
