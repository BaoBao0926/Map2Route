"""Grid A* helpers for LIMP TPSM planning."""

from __future__ import annotations

import heapq
import math

from scripts.methods.limp.utils.geometry import Cell
from scripts.methods.util.grid_astar import TraversableGrid, can_traverse_between, is_traversable, octile_heuristic


def traversable_with_forbidden(
    traversable: TraversableGrid,
    forbidden: set[Cell],
    *,
    start: Cell | None = None,
    goals: set[Cell] | None = None,
) -> TraversableGrid:
    blocked = set(forbidden)
    if start is not None:
        blocked.discard(start)
    if goals:
        blocked -= goals
    return [
        [value and (row, col) not in blocked for col, value in enumerate(values)]
        for row, values in enumerate(traversable)
    ]


def astar_path_to_any(
    traversable: TraversableGrid,
    start: Cell,
    goals: set[Cell],
    *,
    max_goal_candidates: int = 0,
) -> list[list[int]] | None:
    if not goals:
        return None
    valid_goals = {
        goal
        for goal in goals
        if is_traversable(traversable, goal[0], goal[1])
    }
    if not valid_goals:
        return None
    if start in valid_goals:
        return [[start[0], start[1]]]
    if not is_traversable(traversable, start[0], start[1]):
        return None

    if max_goal_candidates and max_goal_candidates > 0 and len(valid_goals) > max_goal_candidates:
        valid_goals = set(
            sorted(valid_goals, key=lambda goal: octile_heuristic(start, goal))[
                :max_goal_candidates
            ]
        )

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

    # Full-set multi-goal planning: search backward from every valid goal cell.
    # This keeps every semantic goal cell eligible without paying an O(|goals|)
    # heuristic cost at every forward A* expansion.
    open_heap: list[tuple[float, int, Cell]] = []
    counter = 1
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {}
    for goal in valid_goals:
        heapq.heappush(open_heap, (0.0, counter, goal))
        counter += 1
        g_score[goal] = 0.0
    closed: set[Cell] = set()
    grid_size = len(traversable)

    while open_heap:
        current_cost, _index, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == start:
            path = [start]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
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
            tentative = current_cost + step_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(open_heap, (tentative, counter, neighbor))
            counter += 1
    return None
