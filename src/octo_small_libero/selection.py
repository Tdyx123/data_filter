"""Compatibility API for the shared prefiltered LIBERO prior implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from libero_lerobot.prefiltered import (
    DATAMIL_TRAJECTORY_REQUIRED_FIELDS,
    PREFILTERED_REQUIRED_FIELDS,
    PrefilteredPriorSelection,
    PriorSelectionError,
    load_prefiltered_selection,
)
from libero_lerobot.prefiltered import (
    resolve_prior_selection as resolve_shared_prior_selection,
)

from .lerobot_v2 import LeRobotV2Metadata


def resolve_prior_selection(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> PrefilteredPriorSelection | None:
    return resolve_shared_prior_selection(
        config,
        paths,
        metadata_factory=LeRobotV2Metadata,
    )


__all__ = [
    "DATAMIL_TRAJECTORY_REQUIRED_FIELDS",
    "LeRobotV2Metadata",
    "PREFILTERED_REQUIRED_FIELDS",
    "PrefilteredPriorSelection",
    "PriorSelectionError",
    "load_prefiltered_selection",
    "resolve_prior_selection",
]
