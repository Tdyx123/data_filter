#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_OVERLAP_RATIO = 0.22
DEFAULT_VERTICAL_OFFSET_RATIO = 0.10
HORIZONTAL_GAP_RATIO = 0.15
DOT_DIAMETER_RATIO = 0.045
DOT_GAP_RATIO = 1.5


class ImageStripError(ValueError):
    """Raised when an overlapping image strip cannot be generated."""


def _unit_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number in [0, 1)") from error
    if not 0.0 <= ratio < 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return ratio


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _natural_sort_key(path: Path) -> tuple[tuple[str | int, ...], str]:
    parts = re.split(r"(\d+)", path.name.casefold())
    components = tuple(int(part) if part.isdigit() else part for part in parts)
    return components, path.name


def collect_image_paths(input_dir: Path, output: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ImageStripError(f"input path is not a directory: {input_dir}")

    output_path = output.resolve(strict=False)
    try:
        paths = [
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.casefold() in SUPPORTED_EXTENSIONS
            and path.resolve(strict=False) != output_path
        ]
    except OSError as error:
        raise ImageStripError(f"cannot read input directory {input_dir}: {error}") from error

    paths.sort(key=_natural_sort_key)
    if len(paths) < 3:
        raise ImageStripError(
            f"input directory must contain at least 3 supported images: {input_dir}"
        )
    return paths


def load_oriented_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            source.load()
            return ImageOps.exif_transpose(source).convert("RGBA")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ImageStripError(f"cannot read image {path}: {error}") from error


def prepare_card(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(
        image,
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def compose_strip(
    cards: tuple[Image.Image, Image.Image, Image.Image],
    *,
    source_count: int,
    overlap_ratio: float,
    vertical_offset_ratio: float,
) -> Image.Image:
    first, second, last = cards
    width, height = first.size
    if second.size != first.size or last.size != first.size:
        raise ImageStripError("all prepared cards must have the same size")

    second_x = round(width * (1.0 - overlap_ratio))
    second_y = round(height * vertical_offset_ratio)
    horizontal_gap = max(1, round(width * HORIZONTAL_GAP_RATIO))
    stack_right = second_x + width

    dot_positions: list[tuple[int, int, int]] = []
    if source_count > 3:
        dot_diameter = max(3, round(height * DOT_DIAMETER_RATIO))
        dot_gap = max(1, round(dot_diameter * DOT_GAP_RATIO))
        dot_group_width = 3 * dot_diameter + 2 * dot_gap
        dot_x = stack_right + horizontal_gap
        dot_y = second_y + height // 2 - dot_diameter // 2
        dot_positions = [
            (dot_x + index * (dot_diameter + dot_gap), dot_y, dot_diameter)
            for index in range(3)
        ]
        last_x = dot_x + dot_group_width + horizontal_gap
    else:
        last_x = stack_right + horizontal_gap

    canvas = Image.new("RGBA", (last_x + width, second_y + height), (0, 0, 0, 0))
    canvas.alpha_composite(first, dest=(0, 0))
    canvas.alpha_composite(second, dest=(second_x, second_y))

    if dot_positions:
        draw = ImageDraw.Draw(canvas)
        for dot_x, dot_y, dot_diameter in dot_positions:
            draw.ellipse(
                (
                    dot_x,
                    dot_y,
                    dot_x + dot_diameter - 1,
                    dot_y + dot_diameter - 1,
                ),
                fill=(255, 255, 255, 255),
            )

    canvas.alpha_composite(last, dest=(last_x, second_y))
    return canvas


def create_overlapping_image_strip(
    input_dir: Path,
    output: Path,
    *,
    card_size: tuple[int, int] | None,
    overlap_ratio: float,
    vertical_offset_ratio: float,
    force: bool,
) -> None:
    if output.suffix.casefold() != ".png":
        raise ImageStripError(f"output path must use the .png extension: {output}")
    if output.exists() and not force:
        raise ImageStripError(f"output already exists (use --force to overwrite): {output}")

    image_paths = collect_image_paths(input_dir, output)
    selected_paths = (image_paths[0], image_paths[1], image_paths[-1])
    selected_images = tuple(load_oriented_image(path) for path in selected_paths)
    target_size = card_size or selected_images[0].size
    cards = tuple(prepare_card(image, target_size) for image in selected_images)
    strip = compose_strip(
        cards,
        source_count=len(image_paths),
        overlap_ratio=overlap_ratio,
        vertical_offset_ratio=vertical_offset_ratio,
    )

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        strip.save(output, format="PNG")
    except OSError as error:
        raise ImageStripError(f"cannot write output image {output}: {error}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a transparent PNG containing the first two and last image "
            "from a naturally sorted directory."
        )
    )
    parser.add_argument("input_dir", type=Path, help="directory containing input images")
    parser.add_argument("--output", required=True, type=Path, help="output .png path")
    parser.add_argument(
        "--card-size",
        nargs=2,
        type=_positive_integer,
        metavar=("WIDTH", "HEIGHT"),
        help="override the output card size (defaults to the oriented first image size)",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=_unit_ratio,
        default=DEFAULT_OVERLAP_RATIO,
        help=f"fraction of the first card covered by the second (default: {DEFAULT_OVERLAP_RATIO})",
    )
    parser.add_argument(
        "--vertical-offset-ratio",
        type=_unit_ratio,
        default=DEFAULT_VERTICAL_OFFSET_RATIO,
        help=(
            "downward offset as a fraction of card height "
            f"(default: {DEFAULT_VERTICAL_OFFSET_RATIO})"
        ),
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    card_size = tuple(arguments.card_size) if arguments.card_size is not None else None
    try:
        create_overlapping_image_strip(
            arguments.input_dir,
            arguments.output,
            card_size=card_size,
            overlap_ratio=arguments.overlap_ratio,
            vertical_offset_ratio=arguments.vertical_offset_ratio,
            force=arguments.force,
        )
    except ImageStripError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"created overlapping image strip: {arguments.output.resolve(strict=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
