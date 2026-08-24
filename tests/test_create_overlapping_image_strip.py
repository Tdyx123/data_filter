from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "create_overlapping_image_strip.py"


def _write_solid(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def _run_cli(
    input_dir: Path,
    output: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(input_dir),
            "--output",
            str(output),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_uses_natural_order_and_stacks_first_two_over_last(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    _write_solid(input_dir / "frame10.png", (0, 0, 255), (100, 80))
    _write_solid(input_dir / "frame2.png", (0, 255, 0), (100, 80))
    _write_solid(input_dir / "frame1.png", (255, 0, 0), (100, 80))
    (input_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    output = tmp_path / "nested" / "strip.png"

    completed = _run_cli(input_dir, output)

    assert completed.returncode == 0, completed.stderr
    with Image.open(output) as image:
        assert image.mode == "RGBA"
        assert image.size == (293, 88)
        assert image.getpixel((5, 5)) == (255, 0, 0, 255)
        assert image.getpixel((80, 10)) == (0, 255, 0, 255)
        assert image.getpixel((200, 10)) == (0, 0, 255, 255)
        assert image.getpixel((185, 2)) == (0, 0, 0, 0)
        assert (255, 255, 255, 255) not in set(image.getdata())


def test_cli_draws_three_white_dots_when_middle_images_are_omitted(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    _write_solid(input_dir / "frame1.png", (255, 0, 0), (100, 80))
    _write_solid(input_dir / "frame2.png", (0, 255, 0), (100, 80))
    _write_solid(input_dir / "frame3.png", (255, 255, 0), (100, 80))
    _write_solid(input_dir / "frame10.png", (0, 0, 255), (100, 80))
    output = tmp_path / "strip.png"

    completed = _run_cli(input_dir, output)

    assert completed.returncode == 0, completed.stderr
    with Image.open(output) as image:
        assert image.size == (332, 88)
        assert image.getpixel((194, 47)) == (255, 255, 255, 255)
        assert image.getpixel((204, 47)) == (255, 255, 255, 255)
        assert image.getpixel((214, 47)) == (255, 255, 255, 255)
        assert image.getpixel((240, 10)) == (0, 0, 255, 255)
        assert (255, 255, 0, 255) not in set(image.getdata())


def test_cli_center_crops_images_to_the_first_image_size(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    _write_solid(input_dir / "frame1.png", (255, 0, 0), (4, 4))
    wide = Image.new("RGB", (12, 4), (255, 0, 0))
    for x in range(4, 8):
        for y in range(4):
            wide.putpixel((x, y), (0, 255, 0))
    for x in range(8, 12):
        for y in range(4):
            wide.putpixel((x, y), (0, 0, 255))
    wide.save(input_dir / "frame2.png")
    _write_solid(input_dir / "frame3.png", (255, 255, 0), (4, 4))
    output = tmp_path / "strip.png"

    completed = _run_cli(input_dir, output)

    assert completed.returncode == 0, completed.stderr
    with Image.open(output) as image:
        assert image.size == (12, 4)
        assert {image.getpixel((x, 1)) for x in range(3, 7)} == {(0, 255, 0, 255)}


def test_cli_applies_exif_orientation_before_choosing_default_card_size(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    first = Image.new("RGB", (2, 4), (255, 0, 0))
    exif = first.getexif()
    exif[274] = 6
    first.save(input_dir / "frame1.jpg", exif=exif)
    _write_solid(input_dir / "frame2.png", (0, 255, 0), (4, 2))
    _write_solid(input_dir / "frame3.png", (0, 0, 255), (4, 2))
    output = tmp_path / "strip.png"

    completed = _run_cli(input_dir, output)

    assert completed.returncode == 0, completed.stderr
    with Image.open(output) as image:
        assert image.size == (12, 2)


def test_cli_honors_card_size_and_layout_ratio_overrides(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    _write_solid(input_dir / "frame1.png", (255, 0, 0), (4, 4))
    _write_solid(input_dir / "frame2.png", (0, 255, 0), (4, 4))
    _write_solid(input_dir / "frame3.png", (0, 0, 255), (4, 4))
    output = tmp_path / "strip.png"

    completed = _run_cli(
        input_dir,
        output,
        "--card-size",
        "100",
        "80",
        "--overlap-ratio",
        "0.5",
        "--vertical-offset-ratio",
        "0.25",
    )

    assert completed.returncode == 0, completed.stderr
    with Image.open(output) as image:
        assert image.size == (265, 100)
        assert image.getpixel((55, 25)) == (0, 255, 0, 255)
        assert image.getpixel((170, 25)) == (0, 0, 255, 255)


def test_cli_excludes_existing_output_from_input_selection_when_forced(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    _write_solid(input_dir / "frame1.png", (255, 0, 0), (100, 80))
    _write_solid(input_dir / "frame2.png", (0, 255, 0), (100, 80))
    _write_solid(input_dir / "frame10.png", (0, 0, 255), (100, 80))
    output = input_dir / "zz-result.png"
    _write_solid(output, (0, 0, 0), (1, 1))

    refused = _run_cli(input_dir, output)
    completed = _run_cli(input_dir, output, "--force")

    assert refused.returncode != 0
    assert "already exists" in refused.stderr
    assert completed.returncode == 0, completed.stderr
    with Image.open(output) as image:
        assert image.size == (293, 88)
        assert image.getpixel((200, 10)) == (0, 0, 255, 255)


@pytest.mark.parametrize(
    ("arguments", "expected_message"),
    [
        (("--overlap-ratio", "1"), "overlap-ratio"),
        (("--overlap-ratio", "-0.1"), "overlap-ratio"),
        (("--vertical-offset-ratio", "1"), "vertical-offset-ratio"),
        (("--card-size", "0", "80"), "card-size"),
    ],
)
def test_cli_rejects_invalid_layout_values(
    tmp_path: Path,
    arguments: tuple[str, ...],
    expected_message: str,
) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    for index in range(3):
        _write_solid(input_dir / f"frame{index}.png", (index, index, index), (4, 4))

    completed = _run_cli(input_dir, tmp_path / "strip.png", *arguments)

    assert completed.returncode != 0
    assert expected_message in completed.stderr


def test_cli_rejects_non_png_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    for index in range(3):
        _write_solid(input_dir / f"frame{index}.png", (index, index, index), (4, 4))

    completed = _run_cli(input_dir, tmp_path / "strip.jpg")

    assert completed.returncode != 0
    assert ".png" in completed.stderr


def test_cli_rejects_missing_directory_and_too_few_images(tmp_path: Path) -> None:
    missing = _run_cli(tmp_path / "missing", tmp_path / "missing.png")
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    _write_solid(input_dir / "frame1.png", (255, 0, 0), (4, 4))
    _write_solid(input_dir / "frame2.png", (0, 255, 0), (4, 4))

    too_few = _run_cli(input_dir, tmp_path / "few.png")

    assert missing.returncode != 0
    assert "not a directory" in missing.stderr
    assert too_few.returncode != 0
    assert "at least 3" in too_few.stderr


def test_cli_reports_corrupt_selected_image(tmp_path: Path) -> None:
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    _write_solid(input_dir / "frame1.png", (255, 0, 0), (4, 4))
    (input_dir / "frame2.png").write_bytes(b"not an image")
    _write_solid(input_dir / "frame3.png", (0, 0, 255), (4, 4))
    output = tmp_path / "strip.png"

    completed = _run_cli(input_dir, output)

    assert completed.returncode != 0
    assert "frame2.png" in completed.stderr
    assert not output.exists()
