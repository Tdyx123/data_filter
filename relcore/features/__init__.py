"""Frame and clip feature encoders."""

from .normalization import RobustNormalizer
from .relation_encoder import RelationEncoder

__all__ = ["RelationEncoder", "RobustNormalizer"]
