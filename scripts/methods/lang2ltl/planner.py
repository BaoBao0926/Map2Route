"""Product-graph planning for Lang2LTL formulas on SemPathBench grids."""

from __future__ import annotations

import heapq
import math
import time
from typing import Mapping, Sequence

from scripts.methods.lang2ltl.ltl import (
    Formula,
    atomic_propositions,
    is_accepting,
    positive_eventual_aps,
    progress,
)
from scripts.methods.lang2ltl.map_features import MapSymbols
from scripts.methods.util.grid_astar import (
    TraversableGrid,
    can_traverse_between,
    is_traversable,
    nearest_traversable_cell,
    octile_heuristic,
)


Point = tuple[int, int]
Trajectory = list[list[int]]
LabelMap = dict[Point, frozenset[str]]
PLANNING_MODES = ("vanilla", "fast", "product")


def _verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[lang2ltl:planner] {message}", flush=True)


def start_cell_from_instruction(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    instruction: Mapping[str, object],
) -> Point | None:
    start_pose = instruction.get("start_pose")
    if isinstance(start_pose, Mapping):
        row = start_pose.get("row")
        col = start_pose.get("col")
        if isinstance(row, int) and isinstance(col, int):
            if is_traversable(traversable, row, col):
                return row, col
            return nearest_traversable_cell(traversable, (row, col), max_radius=80)

    grid_size = int(map_state["grid_size"])
    return nearest_traversable_cell(
        traversable,
        (grid_size / 2, grid_size / 2),
        max_radius=grid_size,
    )


def _layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    raw = layers.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return raw  # type: ignore[return-value]


def _mark(label_sets: dict[Point, set[str]], point: Point, ap: str) -> None:
    label_sets.setdefault(point, set()).add(ap)


def build_label_map(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    symbols: MapSymbols,
    *,
    object_radius: int = 20,
) -> tuple[LabelMap, dict[str, list[Point]]]:
    """Build grid-cell labels from SemPathBench room and object layers."""
    grid_size = int(map_state["grid_size"])
    label_sets: dict[Point, set[str]] = {}

    room_by_id = symbols.rooms_by_id
    room_layer = _layer(map_state, "room")
    for row in range(min(grid_size, len(room_layer))):
        row_values = room_layer[row]
        if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
            continue
        for col in range(min(grid_size, len(row_values))):
            if not is_traversable(traversable, row, col):
                continue
            room = room_by_id.get(int(row_values[col]))
            if room is not None:
                _mark(label_sets, (row, col), room.ap)

    for entity in symbols.object_entities:
        center_row = int(round(entity.center[0]))
        center_col = int(round(entity.center[1]))
        marked_count = 0
        for row in range(
            max(0, center_row - object_radius),
            min(grid_size - 1, center_row + object_radius) + 1,
        ):
            for col in range(
                max(0, center_col - object_radius),
                min(grid_size - 1, center_col + object_radius) + 1,
            ):
                if not is_traversable(traversable, row, col):
                    continue
                if math.hypot(row - entity.center[0], col - entity.center[1]) <= object_radius:
                    _mark(label_sets, (row, col), entity.ap)
                    marked_count += 1
        if marked_count == 0:
            fallback = nearest_traversable_cell(
                traversable,
                entity.center,
                max_radius=max(80, object_radius * 4),
            )
            if fallback is not None:
                _mark(label_sets, fallback, entity.ap)

    labels = {point: frozenset(values) for point, values in label_sets.items()}
    ap_to_cells: dict[str, list[Point]] = {}
    for point, point_labels in labels.items():
        for ap_name in point_labels:
            ap_to_cells.setdefault(ap_name, []).append(point)
    return labels, ap_to_cells


def _node_heuristic(
    row: int,
    col: int,
    formula: Formula,
    ap_to_cells: Mapping[str, Sequence[Point]],
) -> float:
    targets = positive_eventual_aps(formula)
    if not targets:
        return 0.0
    best = math.inf
    for ap_name in targets:
        for target_row, target_col in ap_to_cells.get(ap_name, [])[:2000]:
            best = min(best, math.hypot(row - target_row, col - target_col))
    return 0.0 if math.isinf(best) else best


def _reconstruct_path(
    came_from: Mapping[tuple[int, int, Formula], tuple[int, int, Formula]],
    node: tuple[int, int, Formula],
) -> Trajectory:
    path = [(node[0], node[1])]
    current = node
    while current in came_from:
        current = came_from[current]
        path.append((current[0], current[1]))
    path.reverse()
    return [[row, col] for row, col in path]


def ordered_positive_eventual_aps(formula: Formula) -> list[str]:
    """Return positive APs that appear under eventual goals in formula order."""
    goals: list[str] = []

    def visit(node: Formula, *, negated: bool = False, eventual: bool = False) -> None:
        if node.op == "ap":
            if eventual and not negated and node.value:
                goals.append(node.value)
            return
        if node.op == "not":
            visit(node.args[0], negated=not negated, eventual=eventual)
            return
        if node.op == "eventually":
            visit(node.args[0], negated=negated, eventual=True)
            return
        if node.op == "until":
            left, right = node.args
            visit(left, negated=negated, eventual=eventual)
            visit(right, negated=negated, eventual=True)
            return
        for child in node.args:
            visit(child, negated=negated, eventual=eventual)

    visit(formula.simplify())
    deduped: list[str] = []
    seen: set[str] = set()
    for goal in goals:
        if goal not in seen:
            deduped.append(goal)
            seen.add(goal)
    return deduped


def _append_path(base: Trajectory, segment: Trajectory) -> None:
    if not segment:
        return
    start_index = 1 if base and base[-1] == segment[0] else 0
    base.extend(segment[start_index:])


def _reconstruct_grid_path(
    came_from: Mapping[Point, Point],
    node: Point,
) -> Trajectory:
    path = [node]
    current = node
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return [[row, col] for row, col in path]


def _astar_to_any_label(
    traversable: TraversableGrid,
    start: Point,
    target_cells: Sequence[Point],
    *,
    heuristic_sample_size: int = 256,
) -> tuple[Trajectory | None, Point | None, int]:
    if not target_cells:
        return None, None, 0
    target_set = set(target_cells)
    if start in target_set:
        return [[start[0], start[1]]], start, 1

    heuristic_targets = sorted(
        target_set,
        key=lambda cell: octile_heuristic(start, cell),
    )[:heuristic_sample_size]

    def heuristic(cell: Point) -> float:
        if not heuristic_targets:
            return 0.0
        return min(octile_heuristic(cell, target) for target in heuristic_targets)

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
    open_heap: list[tuple[float, int, Point]] = []
    heapq.heappush(open_heap, (heuristic(start), 0, start))
    counter = 1
    came_from: dict[Point, Point] = {}
    g_score = {start: 0.0}
    closed: set[Point] = set()
    grid_size = len(traversable)
    expanded = 0

    while open_heap:
        _priority, _index, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current in target_set:
            return _reconstruct_grid_path(came_from, current), current, expanded
        closed.add(current)
        expanded += 1

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
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(neighbor), counter, neighbor))
            counter += 1
    return None, None, expanded


def _astar_to_any_label_masked(
    traversable: TraversableGrid,
    start: Point,
    target_cells: Sequence[Point],
    *,
    blocked_cells: set[Point],
    heuristic_sample_size: int = 256,
) -> tuple[Trajectory | None, Point | None, int]:
    target_set = set(target_cells) - blocked_cells
    if not target_set:
        return None, None, 0
    if start in target_set:
        return [[start[0], start[1]]], start, 1

    heuristic_targets = sorted(
        target_set,
        key=lambda cell: octile_heuristic(start, cell),
    )[:heuristic_sample_size]

    def heuristic(cell: Point) -> float:
        return min(octile_heuristic(cell, target) for target in heuristic_targets)

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
    open_heap: list[tuple[float, int, Point]] = [(heuristic(start), 0, start)]
    counter = 1
    came_from: dict[Point, Point] = {}
    g_score = {start: 0.0}
    closed: set[Point] = set()
    grid_size = len(traversable)
    expanded = 0

    while open_heap:
        _priority, _index, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current in target_set:
            return _reconstruct_grid_path(came_from, current), current, expanded
        closed.add(current)
        expanded += 1
        row, col = current
        for delta_row, delta_col, step_cost in directions:
            next_row = row + delta_row
            next_col = col + delta_col
            neighbor = (next_row, next_col)
            if (
                next_row < 0
                or next_row >= grid_size
                or next_col < 0
                or next_col >= grid_size
                or neighbor in blocked_cells
                or not can_traverse_between(traversable, row, col, next_row, next_col)
            ):
                continue
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(neighbor), counter, neighbor))
            counter += 1
    return None, None, expanded


def _path_cost(path: Sequence[Sequence[int]]) -> float:
    if len(path) < 2:
        return 0.0
    return math.fsum(
        math.hypot(row - previous_row, col - previous_col)
        for (previous_row, previous_col), (row, col) in zip(path, path[1:])
    )


def _global_forbidden_aps(formula: Formula) -> set[str]:
    forbidden: set[str] = set()

    def collect_always_negated(node: Formula) -> None:
        node = node.simplify()
        if node.op == "not" and node.args[0].op == "ap" and node.args[0].value:
            forbidden.add(node.args[0].value)
            return
        if node.op == "and":
            for child in node.args:
                collect_always_negated(child)

    def visit(node: Formula) -> None:
        node = node.simplify()
        if node.op == "always":
            collect_always_negated(node.args[0])
            return
        for child in node.args:
            visit(child)

    visit(formula)
    return forbidden


def _ap_mdp_heuristic(
    current: Point,
    formula: Formula,
    ap_to_cells: Mapping[str, Sequence[Point]],
) -> float:
    targets = positive_eventual_aps(formula)
    if not targets:
        return 0.0
    best = math.inf
    for ap_name in targets:
        for target in ap_to_cells.get(ap_name, ())[:512]:
            best = min(best, octile_heuristic(current, target))
    return 0.0 if math.isinf(best) else best


def _candidate_ap_actions(formula: Formula, all_candidate_aps: Sequence[str]) -> list[str]:
    positive = positive_eventual_aps(formula)
    raw_candidates = (
        [ap_name for ap_name in all_candidate_aps if ap_name in positive]
        if positive
        else list(all_candidate_aps)
    )
    progressive: list[str] = []
    for ap_name in raw_candidates:
        next_formula = progress(formula, frozenset({ap_name}))
        if next_formula.op != "false" and next_formula != formula:
            progressive.append(ap_name)
    return progressive or raw_candidates


def ap_mdp_planner(
    traversable: TraversableGrid,
    label_map: LabelMap,
    ap_to_cells: Mapping[str, Sequence[Point]],
    start: Point,
    formula: Formula,
    *,
    verbose: bool = False,
    max_expanded_states: int = 5000,
) -> tuple[Trajectory, list[dict[str, object]]]:
    """Plan over abstract AP states, then realize AP actions with grid A*."""
    started_at = time.monotonic()
    forbidden_aps = _global_forbidden_aps(formula)
    blocked_cells: set[Point] = set()
    for ap_name in forbidden_aps:
        blocked_cells.update(ap_to_cells.get(ap_name, ()))
    blocked_cells.discard(start)

    formula_aps = atomic_propositions(formula)
    all_candidate_aps = sorted(
        ap_name
        for ap_name in formula_aps
        if ap_name not in forbidden_aps and ap_to_cells.get(ap_name)
    )
    initial_formula = progress(formula, label_map.get(start, frozenset()))
    details: list[dict[str, object]] = [
        {
            "kind": "ap_mdp",
            "status": "started",
            "initial_formula": str(formula),
            "initial_progressed_state": str(initial_formula),
            "candidate_aps": all_candidate_aps,
            "forbidden_aps": sorted(forbidden_aps),
            "blocked_cell_count": len(blocked_cells),
        }
    ]
    missing_aps = sorted(ap_name for ap_name in formula_aps if not ap_to_cells.get(ap_name))
    if missing_aps:
        details.append({"kind": "missing_ap_cells", "aps": missing_aps})

    if initial_formula.op == "false":
        details.append({"kind": "ap_mdp", "status": "false_at_start"})
        return [[start[0], start[1]]], details
    if is_accepting(initial_formula):
        details.append({"kind": "ap_mdp", "status": "accepted_at_start"})
        return [[start[0], start[1]]], details
    if not all_candidate_aps:
        details.append({"kind": "ap_mdp", "status": "no_candidate_ap_actions"})
        return [[start[0], start[1]]], details

    initial_state = (start, initial_formula)
    open_heap: list[tuple[float, int, tuple[Point, Formula]]] = [
        (_ap_mdp_heuristic(start, initial_formula, ap_to_cells), 0, initial_state)
    ]
    counter = 1
    g_score: dict[tuple[Point, Formula], float] = {initial_state: 0.0}
    came_from: dict[
        tuple[Point, Formula],
        tuple[tuple[Point, Formula], Trajectory, str],
    ] = {}
    path_cache: dict[tuple[Point, str], tuple[Trajectory | None, Point | None, int]] = {}
    closed: set[tuple[Point, Formula]] = set()
    expanded = 0
    total_grid_expanded = 0

    while open_heap and expanded < max_expanded_states:
        _priority, _index, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        current_cell, current_formula = state
        if is_accepting(current_formula):
            segments: list[Trajectory] = []
            ap_sequence: list[str] = []
            current_state = state
            while current_state in came_from:
                previous_state, segment, ap_name = came_from[current_state]
                segments.append(segment)
                ap_sequence.append(ap_name)
                current_state = previous_state
            segments.reverse()
            ap_sequence.reverse()
            trajectory: Trajectory = [[start[0], start[1]]]
            for segment in segments:
                _append_path(trajectory, segment)
            details.append(
                {
                    "kind": "ap_mdp",
                    "status": "accepted",
                    "ap_sequence": ap_sequence,
                    "expanded_ap_states": expanded,
                    "expanded_grid_states": total_grid_expanded,
                    "trajectory_length": len(trajectory),
                    "elapsed_sec": time.monotonic() - started_at,
                }
            )
            return trajectory, details

        closed.add(state)
        expanded += 1
        action_aps = _candidate_ap_actions(current_formula, all_candidate_aps)
        _verbose_print(
            verbose,
            f"ap_mdp expand={expanded} cell={current_cell} state={current_formula} actions={action_aps}",
        )
        for ap_name in action_aps:
            cache_key = (current_cell, ap_name)
            if cache_key not in path_cache:
                path_cache[cache_key] = _astar_to_any_label_masked(
                    traversable,
                    current_cell,
                    ap_to_cells.get(ap_name, ()),
                    blocked_cells=blocked_cells,
                )
            path, target_cell, grid_expanded = path_cache[cache_key]
            total_grid_expanded += grid_expanded
            if path is None or target_cell is None:
                continue
            target_labels = label_map.get(target_cell, frozenset())
            next_formula = progress(current_formula, target_labels)
            if next_formula.op == "false":
                continue
            next_state = (target_cell, next_formula)
            tentative = g_score[state] + _path_cost(path)
            if tentative >= g_score.get(next_state, math.inf):
                continue
            came_from[next_state] = (state, path, ap_name)
            g_score[next_state] = tentative
            priority = tentative + _ap_mdp_heuristic(target_cell, next_formula, ap_to_cells)
            heapq.heappush(open_heap, (priority, counter, next_state))
            counter += 1

    status = "ap_state_limit_exceeded" if expanded >= max_expanded_states else "no_accepting_ap_plan"
    details.append(
        {
            "kind": "ap_mdp",
            "status": status,
            "expanded_ap_states": expanded,
            "expanded_grid_states": total_grid_expanded,
            "max_expanded_states": max_expanded_states,
        }
    )
    return [[start[0], start[1]]], details


def _astar_to_reachable_nearest_center(
    traversable: TraversableGrid,
    start: Point,
    center: tuple[float, float],
) -> tuple[Trajectory | None, Point | None, int]:
    """Find the reachable cell closest to an object center, then return its path."""
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
    open_heap: list[tuple[float, int, Point]] = [(0.0, 0, start)]
    counter = 1
    came_from: dict[Point, Point] = {}
    g_score = {start: 0.0}
    closed: set[Point] = set()
    grid_size = len(traversable)
    best = start
    best_distance = math.hypot(start[0] - center[0], start[1] - center[1])

    while open_heap:
        _distance, _index, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        center_distance = math.hypot(current[0] - center[0], current[1] - center[1])
        if center_distance < best_distance:
            best = current
            best_distance = center_distance

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
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(open_heap, (tentative, counter, neighbor))
            counter += 1

    return _reconstruct_grid_path(came_from, best), best, len(closed)


def fast_direct_astar(
    traversable: TraversableGrid,
    ap_to_cells: Mapping[str, Sequence[Point]],
    ap_to_centers: Mapping[str, tuple[float, float]],
    start: Point,
    formula: Formula,
    *,
    verbose: bool = False,
) -> tuple[Trajectory, list[dict[str, object]]]:
    """Approximate LTL planning by visiting eventual APs with plain grid A*."""
    goals = ordered_positive_eventual_aps(formula)
    _verbose_print(verbose, f"fast_goal_sequence={goals}")
    details: list[dict[str, object]] = [
        {
            "kind": "fast_astar",
            "status": "started",
            "goal_sequence": goals,
            "formula": str(formula),
        }
    ]
    trajectory: Trajectory = [[start[0], start[1]]]
    current = start
    if not goals:
        details.append({"kind": "fast_astar", "status": "no_eventual_goals"})
        return trajectory, details

    for goal_index, ap_name in enumerate(goals, start=1):
        target_cells = ap_to_cells.get(ap_name, ())
        _verbose_print(
            verbose,
            f"fast goal {goal_index}/{len(goals)} ap={ap_name} labelled_cells={len(target_cells)}",
        )
        path, target, expanded = _astar_to_any_label(
            traversable,
            current,
            target_cells,
        )
        if path is None or target is None:
            center = ap_to_centers.get(ap_name)
            if center is not None:
                _verbose_print(
                    verbose,
                    f"fast fallback nearest reachable cell for ap={ap_name}",
                )
                path, target, expanded = _astar_to_reachable_nearest_center(
                    traversable,
                    current,
                    center,
                )
        if path is None or target is None:
            _verbose_print(
                verbose,
                f"fast failed ap={ap_name} expanded={expanded}",
            )
            details.append(
                {
                    "kind": "fast_astar",
                    "status": "failed_no_path_to_goal",
                    "goal_index": goal_index,
                    "ap": ap_name,
                    "expanded_cells": expanded,
                    "labelled_cells": len(target_cells),
                    "fallback": "nearest_reachable_center",
                }
            )
            return trajectory, details
        _append_path(trajectory, path)
        current = target
        _verbose_print(
            verbose,
            f"fast reached ap={ap_name} target={target} segment_length={len(path)}",
        )
        details.append(
            {
                "kind": "fast_astar",
                "status": "reached_goal",
                "goal_index": goal_index,
                "ap": ap_name,
                "target_cell": [target[0], target[1]],
                "segment_length": len(path),
                "expanded_cells": expanded,
                "used_fallback": target not in set(target_cells),
            }
        )

    details.append(
        {
            "kind": "fast_astar",
            "status": "accepted",
            "goal_count": len(goals),
            "trajectory_length": len(trajectory),
        }
    )
    return trajectory, details


def product_graph_astar(
    traversable: TraversableGrid,
    label_map: LabelMap,
    ap_to_cells: Mapping[str, Sequence[Point]],
    start: Point,
    formula: Formula,
    *,
    verbose: bool = False,
    log_progress: bool = False,
    progress_interval: int = 50000,
) -> tuple[Trajectory, list[dict[str, object]]]:
    """Search over product states (row, col, automaton_formula_state)."""
    initial_formula = progress(formula, label_map.get(start, frozenset()))
    _verbose_print(verbose, f"initial_formula={formula}")
    _verbose_print(verbose, f"initial_progressed_state={initial_formula}")
    _verbose_print(verbose, f"positive_eventual_aps={sorted(positive_eventual_aps(initial_formula))}")
    start_node = (start[0], start[1], initial_formula)
    details: list[dict[str, object]] = [
        {
            "kind": "automaton",
            "initial_formula": str(formula),
            "initial_progressed_state": str(initial_formula),
        }
    ]
    if is_accepting(initial_formula):
        _verbose_print(verbose, "accepted at start")
        details.append({"kind": "product_search", "status": "accepted_at_start"})
        return [[start[0], start[1]]], details

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
    open_heap: list[tuple[float, int, tuple[int, int, Formula]]] = []
    heapq.heappush(
        open_heap,
        (
            _node_heuristic(start[0], start[1], initial_formula, ap_to_cells),
            0,
            start_node,
        ),
    )
    counter = 1
    came_from: dict[tuple[int, int, Formula], tuple[int, int, Formula]] = {}
    g_score = {start_node: 0.0}
    closed: set[tuple[int, int, Formula]] = set()
    grid_size = len(traversable)
    expanded = 0
    started_at = time.monotonic()

    while open_heap:
        _priority, _index, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if is_accepting(current[2]):
            elapsed = time.monotonic() - started_at
            _verbose_print(
                verbose,
                "accepted "
                f"expanded={expanded} open={len(open_heap)} "
                f"elapsed_sec={elapsed:.1f} state={current[2]}",
            )
            details.append(
                {
                    "kind": "product_search",
                    "status": "accepted",
                    "expanded_states": expanded,
                    "accepting_formula_state": str(current[2]),
                }
            )
            return _reconstruct_path(came_from, current), details
        closed.add(current)
        expanded += 1
        if (verbose or log_progress) and progress_interval > 0 and expanded % progress_interval == 0:
            elapsed = time.monotonic() - started_at
            row, col, formula_state = current
            _verbose_print(
                True,
                "progress "
                f"expanded={expanded} open={len(open_heap)} "
                f"closed={len(closed)} g={g_score.get(current, math.inf):.1f} "
                f"cell=({row}, {col}) state={formula_state} "
                f"elapsed_sec={elapsed:.1f}",
            )

        row, col, formula_state = current
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
            labels = label_map.get((next_row, next_col), frozenset())
            next_formula = progress(formula_state, labels)
            if next_formula.op == "false":
                continue
            neighbor = (next_row, next_col, next_formula)
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            priority = tentative + _node_heuristic(
                next_row,
                next_col,
                next_formula,
                ap_to_cells,
            )
            heapq.heappush(open_heap, (priority, counter, neighbor))
            counter += 1

    elapsed = time.monotonic() - started_at
    _verbose_print(
        verbose,
        "failed_no_accepting_state "
        f"expanded={expanded} elapsed_sec={elapsed:.1f}",
    )
    details.append(
        {
            "kind": "product_search",
            "status": "failed_no_accepting_state",
            "expanded_states": expanded,
        }
    )
    return [[start[0], start[1]]], details


def _planner_needs_fast_fallback(trajectory: Trajectory, details: Sequence[Mapping[str, object]]) -> bool:
    if len(trajectory) > 1:
        return False
    if not details:
        return True
    successful_statuses = {"accepted", "accepted_at_start"}
    for item in reversed(details):
        status = item.get("status")
        if isinstance(status, str):
            return status not in successful_statuses
    return True


def build_lang2ltl_trajectory(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    symbols: MapSymbols,
    formula: Formula,
    start: Point,
    *,
    object_radius: int = 20,
    planning_mode: str = "vanilla",
    planner_progress_interval: int = 50000,
    verbose: bool = False,
) -> tuple[Trajectory, list[dict[str, object]], dict[str, object]]:
    if planning_mode not in PLANNING_MODES:
        raise ValueError(f"Unsupported planning mode: {planning_mode}")
    _verbose_print(verbose, "building label map")
    started_at = time.monotonic()
    label_map, ap_to_cells = build_label_map(
        map_state,
        traversable,
        symbols,
        object_radius=object_radius,
    )
    _verbose_print(
        verbose,
        "label map ready "
        f"labelled_cells={len(label_map)} ap_count={len(ap_to_cells)} "
        f"elapsed_sec={time.monotonic() - started_at:.1f}",
    )
    planning_formula = formula
    ap_to_centers = {
        entity.ap: entity.center
        for entity in (*symbols.room_entities, *symbols.object_entities)
    }
    for ap_name in sorted(positive_eventual_aps(planning_formula)):
        _verbose_print(
            verbose,
            f"target_ap={ap_name} labelled_cells={len(ap_to_cells.get(ap_name, []))}",
        )
    if planning_mode == "fast":
        _verbose_print(verbose, "starting fast direct A*")
        trajectory, details = fast_direct_astar(
            traversable,
            ap_to_cells,
            ap_to_centers,
            start,
            planning_formula,
            verbose=verbose,
        )
    elif planning_mode == "product":
        _verbose_print(verbose, "starting product-state A*")
        trajectory, details = product_graph_astar(
            traversable,
            label_map,
            ap_to_cells,
            start,
            planning_formula,
            verbose=verbose,
            log_progress=True,
            progress_interval=planner_progress_interval,
        )
    else:
        _verbose_print(verbose, "starting AP-MDP-style planner")
        trajectory, details = ap_mdp_planner(
            traversable,
            label_map,
            ap_to_cells,
            start,
            planning_formula,
            verbose=verbose,
        )
        if _planner_needs_fast_fallback(trajectory, details):
            _verbose_print(
                verbose,
                "AP-MDP did not produce a moving trajectory; falling back to fast direct A*",
            )
            fallback_trajectory, fallback_details = fast_direct_astar(
                traversable,
                ap_to_cells,
                ap_to_centers,
                start,
                planning_formula,
                verbose=verbose,
            )
            details.append(
                {
                    "kind": "planner_fallback",
                    "status": "used_fast_direct_astar",
                    "reason": "ap_mdp_no_moving_trajectory",
                    "fallback_trajectory_length": len(fallback_trajectory),
                }
            )
            details.extend(fallback_details)
            trajectory = fallback_trajectory
    automaton_summary = {
        "formula": str(planning_formula),
        "original_formula": str(formula),
        "planning_mode": planning_mode,
        "planner_type": (
            "sequential_astar"
            if planning_mode == "fast"
            else "product_state_astar"
            if planning_mode == "product"
            else "ap_mdp_with_fast_fallback"
            if any(item.get("kind") == "planner_fallback" for item in details)
            else "ap_mdp"
        ),
        "semantic_approximation": any(item.get("kind") == "planner_fallback" for item in details),
        "object_mode": "vanilla",
        "object_goal_expansions": [],
        "labelled_cell_count": len(label_map),
        "ap_count": len(ap_to_cells),
        "object_label_radius": object_radius,
    }
    return trajectory, details, automaton_summary
