"""Forbidden region construction for negative LIMP propositions."""

from __future__ import annotations

from scripts.methods.limp.grounding.scene_adapter import Candidate
from scripts.methods.limp.planning.goal_regions import traversable_subset
from scripts.methods.limp.utils.geometry import Cell, cells_within_radius
from scripts.methods.util.grid_astar import TraversableGrid


def forbidden_near_region(
    candidate: Candidate,
    traversable: TraversableGrid,
    *,
    radius: int,
    grid_size: int,
) -> set[Cell]:
    if candidate.kind == "room":
        return traversable_subset(set(candidate.cells), traversable)
    return traversable_subset(cells_within_radius(candidate.cells, radius, grid_size), traversable)

