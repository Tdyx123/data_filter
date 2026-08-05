"""Sparse prototype and relation graph construction."""

from .build import build_graph
from .prototypes import PrototypeData, discover_prototypes

__all__ = ["PrototypeData", "build_graph", "discover_prototypes"]
