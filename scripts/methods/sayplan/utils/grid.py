"""Grid helpers for SayPlan graph and path planning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from scripts.methods.util.grid_astar import (
    TraversableGrid,
    is_traversable,
    nearest_traversable_cell,
)
from scripts.methods.sayplan.utils.geometry import Cell, centroid, nearest_cell


def layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    raw = layers.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return raw  # type: ignore[return-value]


def layer_value(
    grid: Sequence[Sequence[object]],
    row: int,
    col: int,
    default: int = 0,
) -> int:
    if row < 0 or row >= len(grid):
        return default
    row_values = grid[row]
    if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
        return default
    if col < 0 or col >= len(row_values):
        return default
    try:
        return int(row_values[col])
    except (TypeError, ValueError):
        return default


def cells_for_value(grid: Sequence[Sequence[object]], value: int) -> list[Cell]:
    cells: list[Cell] = []
    for row, row_values in enumerate(grid):
        if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
            continue
        for indexed_item in enumerate(row_values):
            try:
                col = int(indexed_item[0])
                item = indexed_item[1]
            except (IndexError, TypeError, ValueError):
                continue
            try:
                if int(item) == value:
                    cells.append((row, col))
            except (TypeError, ValueError):
                continue
    return cells


def approach_cells_for_mask(
    traversable: TraversableGrid,
    object_cells: Sequence[Cell],
) -> list[Cell]:
    if not object_cells:
        return []
    object_set = set(object_cells)
    candidates: set[Cell] = set()
    for row, col in object_cells:
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                candidate = (row + delta_row, col + delta_col)
                if candidate in object_set:
                    continue
                if is_traversable(traversable, candidate[0], candidate[1]):
                    candidates.add(candidate)
    center = centroid(object_cells)
    if center is None:
        return sorted(candidates)
    return sorted(
        candidates,
        key=lambda cell: (abs(cell[0] - center[0]) + abs(cell[1] - center[1]), cell),
    )


def target_for_room(
    traversable: TraversableGrid,
    room_cells: Sequence[Cell],
    fallback: Cell | None,
) -> Cell | None:
    center = centroid(room_cells) or fallback
    if center is None:
        return None
    room_traversable = [cell for cell in room_cells if is_traversable(traversable, *cell)]
    if room_traversable:
        return nearest_cell(center, room_traversable)
    return nearest_traversable_cell(traversable, center)


def assign_room_for_object(
    room_grid: Sequence[Sequence[object]],
    object_cells: Sequence[Cell],
    object_center: Cell | None,
    room_positions: Mapping[int, Cell],
) -> tuple[int | None, str]:
    counts: Counter[int] = Counter()
    for row, col in object_cells:
        room_id = layer_value(room_grid, row, col)
        if room_id > 0:
            counts[room_id] += 1
    if counts:
        return counts.most_common(1)[0][0], "footprint_overlap"
    if object_center is not None:
        room_id = layer_value(room_grid, object_center[0], object_center[1])
        if room_id > 0:
            return room_id, "centroid_cell"
        nearest_room = nearest_cell(object_center, room_positions.values())
        if nearest_room is not None:
            for room_id, position in room_positions.items():
                if position == nearest_room:
                    return room_id, "nearest_room"
    return None, "unassigned"
