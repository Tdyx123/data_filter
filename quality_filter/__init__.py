"""Staged quality-only fragment filtering."""

# Import for its pre-NumPy native-thread bootstrap side effect.
import trajectory_data as _trajectory_data  # noqa: F401

__version__ = "0.1.0"

from .pipeline import (
    encode_stage,
    filter_stage,
    quality_stage,
    run_pipeline,
    validate_output,
)

__all__ = [
    "encode_stage",
    "filter_stage",
    "quality_stage",
    "run_pipeline",
    "validate_output",
]
