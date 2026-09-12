"""Grid traversability and A* helpers shared by method implementations."""

from __future__ import annotations

import heapq
import math
import time
from typing import Callable, Mapping, Sequence


TraversableGrid = list[list[bool]]


def free_occupancy_value(map_state: Mapping[str, object]) -> int:
    legends = map_state.get("layer_legends")
    if isinstance(legends, dict):
        occupancy = legends.get("occupancy")
        if isinstance(occupancy, dict):
            free = occupancy.get("free")
            if isinstance(free, dict) and isinstance(free.get("value"), int):
                return int(free["value"])
    return 0


def build_traversable_grid(map_state: Mapping[str, object]) -> TraversableGrid:
    """Convert the map's occupancy layer into a boolean traversability grid."""
    grid_size = int(map_state["grid_size"])
    layers = map_state["layers"]
    if not isinstance(layers, dict):
        return [[False for _col in range(grid_size)] for _row in range(grid_size)]
    occupancy = layers["occupancy"]
    if not isinstance(occupancy, (list, tuple)):
        return [[False for _col in range(grid_size)] for _row in range(grid_size)]
    free_value = free_occupancy_value(map_state)
    traversable: TraversableGrid = []
    for row in range(grid_size):
        traversable_row: list[bool] = []
        try:
            row_values = occupancy[row]
        except (IndexError, KeyError, TypeError):
            row_values = ()
        for col in range(grid_size):
            try:
                raw_value: object | None = row_values[col]  # type: ignore[index]
            except (IndexError, KeyError, TypeError):
                try:
                    # JSON object keys are strings, while some in-memory test
                    # fixtures use integer keys for sparse occupancy rows.
                    raw_value = row_values[str(col)]  # type: ignore[index]
                except (IndexError, KeyError, TypeError):
                    raw_value = None
            try:
                occupancy_value = int(raw_value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                # A malformed or missing cell must never become traversable.
                occupancy_value = None
            traversable_row.append(occupancy_value == free_value)
        traversable.append(traversable_row)
    return traversable


def is_traversable(traversable: TraversableGrid, row: int, col: int) -> bool:
    if row < 0 or row >= len(traversable):
        return False
    if col < 0 or col >= len(traversable[row]):
        return False
    return traversable[row][col]


def can_traverse_between(
    traversable: TraversableGrid,
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
) -> bool:
    if not is_traversable(traversable, to_row, to_col):
        return False
    delta_row = to_row - from_row
    delta_col = to_col - from_col
    if abs(delta_row) != 1 or abs(delta_col) != 1:
        return True
    return is_traversable(traversable, from_row + delta_row, from_col) and is_traversable(
        traversable,
        from_row,
        from_col + delta_col,
    )


def octile_heuristic(first: tuple[int, int], second: tuple[int, int]) -> float:
    row_delta = abs(first[0] - second[0])
    col_delta = abs(first[1] - second[1])
    diagonal = min(row_delta, col_delta)
    straight = max(row_delta, col_delta) - diagonal
    return diagonal * math.sqrt(2) + straight


def astar_path(
    traversable: TraversableGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    edge_allowed: Callable[[tuple[int, int], tuple[int, int]], bool] | None = None,
    deadline: float | None = None,
) -> list[list[int]] | None:
    if start == goal:
        return [[start[0], start[1]]]
    if not is_traversable(traversable, *start) or not is_traversable(traversable, *goal):
        return None

    directions = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    )
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (octile_heuristic(start, goal), 0, start))
    counter = 1
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    grid_size = len(traversable)

    while open_heap:
        if deadline is not None and time.monotonic() >= deadline:
            return None
        _priority, _index, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return [[row, col] for row, col in path]
        closed.add(current)

        row, col = current
        for delta_row, delta_col, step_cost in directions:
            next_row = row + delta_row
            next_col = col + delta_col
            if (
                next_row < 0
                or next_row >= grid_size
                or next_col < 0
                or next_col >= grid_size
                or not can_traverse_between(traversable, row, col, next_row, next_col)
            ):
                continue
            neighbor = (next_row, next_col)
            if edge_allowed is not None and not edge_allowed(current, neighbor):
                continue
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(
                open_heap,
                (tentative + octile_heuristic(neighbor, goal), counter, neighbor),
            )
            counter += 1
    return None


def astar_path_to_any(
    traversable: TraversableGrid,
    start: tuple[int, int],
    goals: Sequence[tuple[int, int]],
    *,
    edge_allowed: Callable[[tuple[int, int], tuple[int, int]], bool] | None = None,
) -> list[list[int]] | None:
    """Find a shortest path from ``start`` to any traversable goal cell.

    The search runs backwards from every valid goal.  Besides making every
    target-region cell eligible, this avoids choosing a region centre and then
    travelling through an already-completed region under the next segment's
    hard constraints.
    """
    valid_goals = {
        goal for goal in goals if is_traversable(traversable, goal[0], goal[1])
    }
    if not valid_goals or not is_traversable(traversable, *start):
        return None
    if start in valid_goals:
        return [[start[0], start[1]]]

    directions = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    )
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    distance: dict[tuple[int, int], float] = {}
    next_step: dict[tuple[int, int], tuple[int, int]] = {}
    for goal in valid_goals:
        heapq.heappush(open_heap, (0.0, counter, goal))
        distance[goal] = 0.0
        counter += 1

    closed: set[tuple[int, int]] = set()
    grid_size = len(traversable)
    while open_heap:
        current_cost, _index, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == start:
            path = [start]
            while path[-1] in next_step:
                path.append(next_step[path[-1]])
            return [[row, col] for row, col in path]
        closed.add(current)
        row, col = current
        for delta_row, delta_col, step_cost in directions:
            neighbor = (row + delta_row, col + delta_col)
            if (
                neighbor[0] < 0
                or neighbor[0] >= grid_size
                or neighbor[1] < 0
                or neighbor[1] >= grid_size
                or not can_traverse_between(traversable, row, col, *neighbor)
            ):
                continue
            # The returned route travels neighbour -> current, so validate in
            # that direction.  Current geometry checks are symmetric, while
            # this also keeps the helper correct for directional callers.
            if edge_allowed is not None and not edge_allowed(neighbor, current):
                continue
            candidate = current_cost + step_cost
            if candidate >= distance.get(neighbor, math.inf):
                continue
            distance[neighbor] = candidate
            next_step[neighbor] = current
            heapq.heappush(open_heap, (candidate, counter, neighbor))
            counter += 1
    return None


def nearest_traversable_cell(
    traversable: TraversableGrid,
    seed: tuple[float, float],
    *,
    max_radius: int = 80,
) -> tuple[int, int] | None:
    grid_size = len(traversable)
    center_row = min(max(int(round(seed[0])), 0), grid_size - 1)
    center_col = min(max(int(round(seed[1])), 0), grid_size - 1)
    if is_traversable(traversable, center_row, center_col):
        return center_row, center_col
    for radius in range(1, max_radius + 1):
        candidates: list[tuple[int, int]] = []
        for row in range(max(0, center_row - radius), min(grid_size - 1, center_row + radius) + 1):
            candidates.append((row, max(0, center_col - radius)))
            candidates.append((row, min(grid_size - 1, center_col + radius)))
        for col in range(max(0, center_col - radius + 1), min(grid_size - 1, center_col + radius - 1) + 1):
            candidates.append((max(0, center_row - radius), col))
            candidates.append((min(grid_size - 1, center_row + radius), col))
        for row, col in sorted(set(candidates), key=lambda cell: math.hypot(cell[0] - seed[0], cell[1] - seed[1])):
            if is_traversable(traversable, row, col):
                return row, col
    return None
