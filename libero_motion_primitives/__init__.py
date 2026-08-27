"""Public API for configured ECoT motion-primitive generation."""

from .motion_primitives import (
    PrimitiveConfig,
    PrimitiveThresholds,
    TailStrategy,
    classify_motion_primitive,
    compute_primitive_statistics,
    filter_frequent_primitives,
    generate_motion_primitives,
    make_bridge_v2_config,
    make_libero_config,
)

__all__ = [
    "PrimitiveConfig",
    "PrimitiveThresholds",
    "TailStrategy",
    "classify_motion_primitive",
    "compute_primitive_statistics",
    "filter_frequent_primitives",
    "generate_motion_primitives",
    "make_bridge_v2_config",
    "make_libero_config",
]
