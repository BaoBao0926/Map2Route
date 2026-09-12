"""Goal and proposition regions for grounded LIMP referents."""

from __future__ import annotations

from scripts.methods.limp.grounding.scene_adapter import Candidate
from scripts.methods.limp.utils.geometry import Cell, cells_within_radius
from scripts.methods.util.grid_astar import TraversableGrid, is_traversable


def traversable_subset(cells: set[Cell], traversable: TraversableGrid) -> set[Cell]:
    return {(row, col) for row, col in cells if is_traversable(traversable, row, col)}


def near_region(
    candidate: Candidate,
    traversable: TraversableGrid,
    *,
    radius: int,
    grid_size: int,
) -> set[Cell]:
    if candidate.kind == "room":
        return traversable_subset(set(candidate.cells), traversable)
    expanded = cells_within_radius(candidate.cells, radius, grid_size)
    object_cells = set(candidate.cells)
    return traversable_subset(expanded - object_cells, traversable)

