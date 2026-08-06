"""Shared segment quality, encoding, and diversity-filtering primitives."""

from .encoding import (
    ClipVisionEncoder,
    NumericNormalizers,
    PCAProjector,
    fuse_fragment_features,
    l2_normalize_rows,
    temporal_pool,
    visual_fragment_feature,
)
from .quality import RawQuality, raw_quality, score_quality
from .selection import SelectionResult, select_diverse_fragments
from .windows import candidate_windows, reference_sample_count, reference_windows

__all__ = [
    "ClipVisionEncoder",
    "NumericNormalizers",
    "PCAProjector",
    "RawQuality",
    "SelectionResult",
    "candidate_windows",
    "fuse_fragment_features",
    "l2_normalize_rows",
    "raw_quality",
    "reference_sample_count",
    "reference_windows",
    "score_quality",
    "select_diverse_fragments",
    "temporal_pool",
    "visual_fragment_feature",
]

__version__ = "0.1.0"
