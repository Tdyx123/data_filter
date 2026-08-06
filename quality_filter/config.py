"""Configuration loading for the staged Quality Filter pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("quality_filter config root must be a mapping")
    required = {"dataset", "clip", "encoder", "quality", "filter", "runtime", "output"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"quality_filter config is missing sections: {missing}")
    if config["clip"] != {"length": 15, "stride": 15}:
        raise ValueError("quality_filter requires clip length=15 and stride=15")
    encoder = config["encoder"]
    if not isinstance(encoder, dict) or int(encoder.get("visual_dim", -1)) != 128:
        raise ValueError("quality_filter encoder.visual_dim must be 128")
    if encoder.get("local_files_only") is not True:
        raise ValueError("quality_filter encoder.local_files_only must be true")
    quality = config["quality"]
    low = float(quality.get("quantile_low", 0.01))
    high = float(quality.get("quantile_high", 0.99))
    epsilon = float(quality.get("epsilon", 1.0e-8))
    if not 0.0 <= low < high <= 1.0 or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("quality_filter quality quantiles/epsilon are invalid")
    filter_config = config["filter"]
    percent = float(filter_config.get("percent", 10.0))
    if not math.isfinite(percent) or not 0.0 < percent <= 100.0:
        raise ValueError("quality_filter filter.percent must be in (0, 100]")
    seed = filter_config.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
    ):
        raise ValueError("quality_filter filter.seed must be null or in [0, 2**64)")
    return config
