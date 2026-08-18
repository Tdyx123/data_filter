"""Fixed BridgeData V2 configuration for the Cocore adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from cocore.config import resolve_config


DEFAULT_DATASET_PATH = Path("/data/dwb/datasets/bridge_orig_1.0.0_lerobo")
DEFAULT_SELECTION_RATIO = 0.10
_CONFIG_PATH = Path(__file__).with_name("config_bridge_v2.yaml")


def _load_base_config() -> dict[str, Any]:
    payload = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("BridgeData V2 configuration root must be a mapping")
    return dict(payload)


def build_config(
    *,
    relation: str,
    relation_weight: float,
    selection_ratio: float = DEFAULT_SELECTION_RATIO,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    max_episodes: int | None = None,
    use_stop_bucket: bool = True,
) -> dict[str, Any]:
    """Build and validate the fixed BridgeData V2 Cocore configuration."""

    config = _load_base_config()
    config["dataset"]["path"] = str(Path(dataset_path).expanduser())
    config["objective"] = {
        "relation": relation,
        "relation_weight": relation_weight,
    }
    config["selection"]["ratio"] = selection_ratio
    config["selection"]["budget"] = None
    config["runtime"]["max_episodes"] = max_episodes
    config["prototypes"]["use_stop_bucket"] = use_stop_bucket
    return resolve_config(config)
