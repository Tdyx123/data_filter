"""Profile-resolved temporal geometry for Cocore data flow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TemporalGeometry:
    """Immutable clip and action-window geometry for one fixed profile."""

    profile: str
    clip_length: int
    clip_anchors: tuple[int, int, int]
    visual_half_windows: tuple[tuple[int, int], tuple[int, int]]
    trajectory_window_length: int

    @property
    def state_delta_horizon(self) -> int:
        """Return the endpoint offset used for trajectory action labels."""

        return self.trajectory_window_length - 1

    @property
    def action_window_length(self) -> int:
        """Return the trajectory action-window length."""

        return self.trajectory_window_length


_TEMPORAL_GEOMETRIES = {
    "libero": TemporalGeometry(
        profile="libero",
        clip_length=15,
        clip_anchors=(0, 7, 14),
        visual_half_windows=((0, 8), (7, 15)),
        trajectory_window_length=8,
    ),
    "bridge_v2": TemporalGeometry(
        profile="bridge_v2",
        clip_length=7,
        clip_anchors=(0, 3, 6),
        visual_half_windows=((0, 4), (3, 7)),
        trajectory_window_length=4,
    ),
}


def resolve_temporal_geometry(profile: str) -> TemporalGeometry:
    """Resolve the immutable temporal geometry for a fixed Cocore profile."""

    try:
        return _TEMPORAL_GEOMETRIES[profile]
    except KeyError as error:
        raise ValueError(f"unknown temporal geometry profile {profile!r}") from error
