"""Motion-primitive relation coreset selection."""

# Import for its pre-NumPy native-thread bootstrap side effect.
import trajectory_data as _trajectory_data  # noqa: F401

from .objective import CocoreObjectiveContext, CocoreObjectiveState

__all__ = ["CocoreObjectiveContext", "CocoreObjectiveState"]
__version__ = "0.4.0"
