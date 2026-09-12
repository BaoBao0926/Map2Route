"""SemPathBench planning helpers for the Lang2LTL-2 adapter."""

from __future__ import annotations

import heapq
import math
import time
from typing import Mapping, Sequence

from scripts.methods.lang2ltl.ltl import (
    Formula,
    atomic_propositions,
    is_accepting,
    ltl_and,
    ltl_eventually,
    positive_eventual_aps,
    progress,
)
from scripts.methods.lang2ltl.ltl_parser import parse_prefix_ltl
from scripts.methods.lang2ltl.map_features import MapSymbols
from scripts.methods.lang2ltl.planner import build_label_map, build_lang2ltl_trajectory
from scripts.methods.util.grid_astar import (
    TraversableGrid,
    can_traverse_between,
    octile_heuristic,
)


Point = tuple[int, int]
Trajectory = list[list[int]]
PLANNING_MODES = ("vanilla", "fast")


def _verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[lang2ltlv2:planner] {message}", flush=True)


def _formula_has_complex_semantics(formula: Formula) -> list[str]:
    complex_ops: list[str] = []

    def visit(node: Formula) -> None:
        if node.op in {"not", "always", "until", "or", "next", "imply"}:
            complex_ops.append(node.op)
        for child in node.args:
            visit(child)

    visit(formula.simplify())
    return sorted(set(complex_ops))


def _planner_needs_fast_fallback(
    trajectory: Trajectory,
    details: Sequence[Mapping[str, object]],
) -> bool:
    if len(trajectory) > 1:
        return False
    successful_statuses = {"accepted", "accepted_at_start"}
    for item in reversed(details):
        status = item.get("status")
        if isinstance(status, str):
            return status not in successful_statuses
    return True


def _sequential_eventual_formula(aps: list[str]) -> Formula:
    if not aps:
        raise ValueError("Cannot build a fallback formula for an empty target sequence.")
    result = Formula("ap", value=aps[-1])
    for ap_name in reversed(aps[:-1]):
        result = ltl_and(Formula("ap", value=ap_name), ltl_eventually(result))
    return ltl_eventually(result).simplify()


def _append_path(base: Trajectory, segment: Trajectory) -> None:
    if not segment:
        return
    start_index = 1 if base and base[-1] == segment[0] else 0
    base.extend(segment[start_index:])


def _reconstruct_grid_path(came_from: Mapping[Point, Point], node: Point) -> Trajectory:
    path = [node]
    current = node
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return [[row, col] for row, col in path]


def _path_cost(path: Sequence[Sequence[int]]) -> float:
    if len(path) < 2:
        return 0.0
    return math.fsum(
        math.hypot(row - previous_row, col - previous_col)
        for (previous_row, previous_col), (row, col) in zip(path, path[1:])
    )


def _astar_to_any_target_masked(
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
    came_from: dict[Point, Point] = {}
    g_score = {start: 0.0}
    closed: set[Point] = set()
    counter = 1
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
    candidates = positive_eventual_aps(formula)
    if not candidates:
        return 0.0
    best = math.inf
    for ap_name in candidates:
        for target in ap_to_cells.get(ap_name, ())[:512]:
            best = min(best, octile_heuristic(current, target))
    return 0.0 if math.isinf(best) else best


def _candidate_ap_actions(
    formula: Formula,
    all_candidate_aps: Sequence[str],
) -> list[str]:
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


def _plan_ap_mdp_formula(
    *,
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    symbols: MapSymbols,
    formula: Formula,
    start: Point,
    object_radius: int,
    verbose: bool = False,
    max_expanded_states: int = 5000,
) -> tuple[Trajectory, list[dict[str, object]], dict[str, object]]:
    started_at = time.monotonic()
    label_map, ap_to_cells = build_label_map(
        map_state,
        traversable,
        symbols,
        object_radius=object_radius,
    )
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
    missing_aps = sorted(ap_name for ap_name in formula_aps if not ap_to_cells.get(ap_name))
    unsupported: list[dict[str, object]] = []
    if missing_aps:
        unsupported.append({"kind": "missing_ap_cells", "aps": missing_aps})

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
            "labelled_cell_count": len(label_map),
            "ap_count": len(ap_to_cells),
        }
    ]
    details.extend(unsupported)
    _verbose_print(
        verbose,
        "AP-MDP start "
        f"candidate_aps={all_candidate_aps} forbidden_aps={sorted(forbidden_aps)}",
    )

    if initial_formula.op == "false":
        details.append({"kind": "ap_mdp", "status": "false_at_start"})
        summary = {
            "planner_mode": "vanilla",
            "planner_type": "ap_mdp",
            "status": "false_at_start",
            "semantic_approximation": True,
            "formula": str(formula),
            "candidate_aps": all_candidate_aps,
            "forbidden_aps": sorted(forbidden_aps),
        }
        return [[start[0], start[1]]], details, summary
    if is_accepting(initial_formula):
        details.append({"kind": "ap_mdp", "status": "accepted_at_start"})
        summary = {
            "planner_mode": "vanilla",
            "planner_type": "ap_mdp",
            "status": "accepted_at_start",
            "semantic_approximation": True,
            "formula": str(formula),
            "candidate_aps": all_candidate_aps,
            "forbidden_aps": sorted(forbidden_aps),
        }
        return [[start[0], start[1]]], details, summary
    if not all_candidate_aps:
        details.append({"kind": "ap_mdp", "status": "no_candidate_ap_actions"})
        summary = {
            "planner_mode": "vanilla",
            "planner_type": "ap_mdp",
            "status": "no_candidate_ap_actions",
            "semantic_approximation": True,
            "formula": str(formula),
            "candidate_aps": all_candidate_aps,
            "forbidden_aps": sorted(forbidden_aps),
        }
        return [[start[0], start[1]]], details, summary

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
            summary = {
                "planner_mode": "vanilla",
                "planner_type": "ap_mdp",
                "status": "accepted",
                "semantic_approximation": True,
                "formula": str(formula),
                "ap_sequence": ap_sequence,
                "candidate_aps": all_candidate_aps,
                "forbidden_aps": sorted(forbidden_aps),
                "expanded_ap_states": expanded,
                "expanded_grid_states": total_grid_expanded,
                "labelled_cell_count": len(label_map),
                "ap_count": len(ap_to_cells),
                "object_label_radius": object_radius,
            }
            return trajectory, details, summary

        closed.add(state)
        expanded += 1
        action_aps = _candidate_ap_actions(current_formula, all_candidate_aps)
        _verbose_print(
            verbose,
            f"AP-MDP expand={expanded} cell={current_cell} state={current_formula} actions={action_aps}",
        )
        for ap_name in action_aps:
            cache_key = (current_cell, ap_name)
            if cache_key not in path_cache:
                path_cache[cache_key] = _astar_to_any_target_masked(
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
            g_score[next_state] = tentative
            came_from[next_state] = (state, path, ap_name)
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
    summary = {
        "planner_mode": "vanilla",
        "planner_type": "ap_mdp",
        "status": status,
        "semantic_approximation": True,
        "formula": str(formula),
        "candidate_aps": all_candidate_aps,
        "forbidden_aps": sorted(forbidden_aps),
        "expanded_ap_states": expanded,
        "expanded_grid_states": total_grid_expanded,
        "labelled_cell_count": len(label_map),
        "ap_count": len(ap_to_cells),
        "object_label_radius": object_radius,
    }
    return [[start[0], start[1]]], details, summary


def plan_vanilla_ap_mdp(
    *,
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    symbols: MapSymbols,
    grounded_ltl: str | None,
    target_sequence: list[str] | None,
    start: Point,
    object_radius: int,
    verbose: bool = False,
) -> tuple[Trajectory, list[dict[str, object]], dict[str, object]]:
    """Plan over AP-level MDP states, then realize AP actions with grid A*."""
    parse_warnings: list[str] = []
    fallback: str | None = None
    if grounded_ltl:
        parse_result = parse_prefix_ltl(grounded_ltl)
        formula = parse_result.formula
        parse_warnings = list(parse_result.warnings)
    elif target_sequence:
        formula = _sequential_eventual_formula(target_sequence)
        fallback = "target_sequence_without_ltl"
    else:
        details = [
            {
                "kind": "ap_mdp",
                "status": "missing_grounded_ltl_and_target_sequence",
            }
        ]
        summary = {
            "planner_mode": "vanilla",
            "planner_type": "ap_mdp",
            "status": "missing_grounded_ltl_and_target_sequence",
            "semantic_approximation": True,
        }
        return [[start[0], start[1]]], details, summary

    trajectory, details, summary = _plan_ap_mdp_formula(
        map_state=map_state,
        traversable=traversable,
        symbols=symbols,
        formula=formula,
        start=start,
        object_radius=object_radius,
        verbose=verbose,
    )
    details.insert(
        0,
        {
            "kind": "planner_mode",
            "mode": "vanilla",
            "type": "ap_mdp",
            "fallback": fallback,
            "target_sequence": target_sequence or [],
            "grounded_ltl": grounded_ltl,
            "parse_warnings": parse_warnings,
        },
    )
    summary.update(
        {
            "planner_mode": "vanilla",
            "planner_type": "ap_mdp",
            "fallback": fallback,
            "target_sequence": target_sequence or [],
            "grounded_ltl": grounded_ltl,
            "parse_warnings": parse_warnings,
        }
    )
    return trajectory, details, summary


def plan_fast_sequential_astar(
    *,
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    symbols: MapSymbols,
    grounded_ltl: str,
    start: Point,
    object_radius: int,
    verbose: bool = False,
) -> tuple[Trajectory, list[dict[str, object]], dict[str, object]]:
    parse_result = parse_prefix_ltl(grounded_ltl)
    complex_ops = _formula_has_complex_semantics(parse_result.formula)
    trajectory, details, summary = build_lang2ltl_trajectory(
        map_state,
        traversable,
        symbols,
        parse_result.formula,
        start,
        object_radius=object_radius,
        planning_mode="fast",
        verbose=verbose,
    )
    details.insert(
        0,
        {
            "kind": "planner_mode",
            "mode": "fast",
            "type": "sequential_astar",
            "semantic_approximation": True,
            "approximated_formula_ops": complex_ops,
            "parse_warnings": list(parse_result.warnings),
        },
    )
    summary.update(
        {
            "planner_mode": "fast",
            "planner_type": "sequential_astar",
            "semantic_approximation": True,
            "approximated_formula_ops": complex_ops,
            "parse_warnings": list(parse_result.warnings),
        }
    )
    return trajectory, details, summary


def plan_fast_target_sequence(
    *,
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    symbols: MapSymbols,
    target_sequence: list[str],
    start: Point,
    object_radius: int,
    verbose: bool = False,
) -> tuple[Trajectory, list[dict[str, object]], dict[str, object]]:
    formula = _sequential_eventual_formula(target_sequence)
    trajectory, details, summary = build_lang2ltl_trajectory(
        map_state,
        traversable,
        symbols,
        formula,
        start,
        object_radius=object_radius,
        planning_mode="fast",
        verbose=verbose,
    )
    details.insert(
        0,
        {
            "kind": "planner_mode",
            "mode": "fast",
            "type": "sequential_astar",
            "semantic_approximation": True,
            "fallback": "target_sequence_without_ltl",
            "target_sequence": target_sequence,
        },
    )
    summary.update(
        {
            "planner_mode": "fast",
            "planner_type": "sequential_astar",
            "status": details[-1].get("status") if details else "unknown",
            "semantic_approximation": True,
            "fallback": "target_sequence_without_ltl",
            "target_sequence": target_sequence,
        }
    )
    return trajectory, details, summary


def build_lang2ltl2_trajectory(
    *,
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    symbols: MapSymbols,
    grounded_ltl: str | None,
    start: Point,
    target_sequence: list[str] | None = None,
    object_radius: int = 20,
    planning_mode: str = "vanilla",
    verbose: bool = False,
) -> tuple[Trajectory, list[dict[str, object]], dict[str, object]]:
    if planning_mode not in PLANNING_MODES:
        raise ValueError(f"Unsupported planning mode: {planning_mode}")
    if planning_mode == "vanilla":
        try:
            trajectory, details, summary = plan_vanilla_ap_mdp(
                map_state=map_state,
                traversable=traversable,
                symbols=symbols,
                grounded_ltl=grounded_ltl,
                target_sequence=target_sequence,
                start=start,
                object_radius=object_radius,
                verbose=verbose,
            )
            if not _planner_needs_fast_fallback(trajectory, details):
                return trajectory, details, summary

            _verbose_print(
                verbose,
                "AP-MDP did not produce a moving trajectory; falling back to sequential A*",
            )
            if grounded_ltl:
                fallback_trajectory, fallback_details, fallback_summary = plan_fast_sequential_astar(
                    map_state=map_state,
                    traversable=traversable,
                    symbols=symbols,
                    grounded_ltl=grounded_ltl,
                    start=start,
                    object_radius=object_radius,
                    verbose=verbose,
                )
                fallback_source = "grounded_ltl"
            elif target_sequence:
                fallback_trajectory, fallback_details, fallback_summary = plan_fast_target_sequence(
                    map_state=map_state,
                    traversable=traversable,
                    symbols=symbols,
                    target_sequence=target_sequence,
                    start=start,
                    object_radius=object_radius,
                    verbose=verbose,
                )
                fallback_source = "target_sequence"
            else:
                details.append(
                    {
                        "kind": "planner_fallback",
                        "status": "unavailable",
                        "reason": "no_grounded_ltl_or_target_sequence",
                    }
                )
                summary.update(
                    {
                        "planner_type": "ap_mdp",
                        "status": "ap_mdp_failed_no_fast_fallback",
                        "semantic_approximation": True,
                    }
                )
                return trajectory, details, summary

            details.append(
                {
                    "kind": "planner_fallback",
                    "status": "used_fast_sequential_astar",
                    "reason": "ap_mdp_no_moving_trajectory",
                    "fallback_source": fallback_source,
                    "fallback_trajectory_length": len(fallback_trajectory),
                }
            )
            details.extend(fallback_details)
            summary.update(
                {
                    "planner_mode": "vanilla",
                    "planner_type": "ap_mdp_with_fast_fallback",
                    "status": fallback_summary.get("status", "fast_fallback_used"),
                    "semantic_approximation": True,
                    "fallback_source": fallback_source,
                    "fallback_summary": fallback_summary,
                }
            )
            return fallback_trajectory, details, summary
        except Exception as exc:
            details = [
                {
                    "kind": "ap_mdp",
                    "status": "planning_failure",
                    "reason": str(exc),
                    "grounded_ltl": grounded_ltl,
                    "target_sequence": target_sequence or [],
                }
            ]
            if grounded_ltl:
                try:
                    fallback_trajectory, fallback_details, fallback_summary = plan_fast_sequential_astar(
                        map_state=map_state,
                        traversable=traversable,
                        symbols=symbols,
                        grounded_ltl=grounded_ltl,
                        start=start,
                        object_radius=object_radius,
                        verbose=verbose,
                    )
                    details.append(
                        {
                            "kind": "planner_fallback",
                            "status": "used_fast_sequential_astar",
                            "reason": "ap_mdp_exception",
                            "fallback_source": "grounded_ltl",
                            "fallback_trajectory_length": len(fallback_trajectory),
                        }
                    )
                    details.extend(fallback_details)
                    fallback_summary.update(
                        {
                            "planner_mode": "vanilla",
                            "planner_type": "ap_mdp_with_fast_fallback",
                            "status": fallback_summary.get("status", "fast_fallback_used"),
                            "fallback_source": "grounded_ltl",
                            "ap_mdp_error": str(exc),
                        }
                    )
                    return fallback_trajectory, details, fallback_summary
                except Exception as fallback_exc:
                    details.append(
                        {
                            "kind": "planner_fallback",
                            "status": "failed",
                            "reason": str(fallback_exc),
                        }
                    )
            elif target_sequence:
                try:
                    fallback_trajectory, fallback_details, fallback_summary = plan_fast_target_sequence(
                        map_state=map_state,
                        traversable=traversable,
                        symbols=symbols,
                        target_sequence=target_sequence,
                        start=start,
                        object_radius=object_radius,
                        verbose=verbose,
                    )
                    details.append(
                        {
                            "kind": "planner_fallback",
                            "status": "used_fast_sequential_astar",
                            "reason": "ap_mdp_exception",
                            "fallback_source": "target_sequence",
                            "fallback_trajectory_length": len(fallback_trajectory),
                        }
                    )
                    details.extend(fallback_details)
                    fallback_summary.update(
                        {
                            "planner_mode": "vanilla",
                            "planner_type": "ap_mdp_with_fast_fallback",
                            "status": fallback_summary.get("status", "fast_fallback_used"),
                            "fallback_source": "target_sequence",
                            "ap_mdp_error": str(exc),
                        }
                    )
                    return fallback_trajectory, details, fallback_summary
                except Exception as fallback_exc:
                    details.append(
                        {
                            "kind": "planner_fallback",
                            "status": "failed",
                            "reason": str(fallback_exc),
                        }
                    )
            summary = {
                "planner_mode": "vanilla",
                "planner_type": "ap_mdp",
                "status": "planning_failure",
                "reason": str(exc),
                "grounded_ltl": grounded_ltl,
                "target_sequence": target_sequence or [],
                "semantic_approximation": True,
            }
            return [[start[0], start[1]]], details, summary
    if not grounded_ltl:
        if target_sequence:
            try:
                return plan_fast_target_sequence(
                    map_state=map_state,
                    traversable=traversable,
                    symbols=symbols,
                    target_sequence=target_sequence,
                    start=start,
                    object_radius=object_radius,
                    verbose=verbose,
                )
            except Exception as exc:
                details = [
                    {
                        "kind": "fast_astar",
                        "status": "target_sequence_planning_failure",
                        "reason": str(exc),
                        "target_sequence": target_sequence,
                        "semantic_approximation": True,
                    }
                ]
                summary = {
                    "planner_mode": "fast",
                    "planner_type": "sequential_astar",
                    "status": "target_sequence_planning_failure",
                    "reason": str(exc),
                    "target_sequence": target_sequence,
                    "semantic_approximation": True,
                }
                return [[start[0], start[1]]], details, summary
        details = [
            {
                "kind": "fast_astar",
                "status": "missing_grounded_ltl",
                "semantic_approximation": True,
            }
        ]
        summary = {
            "planner_mode": "fast",
            "planner_type": "sequential_astar",
            "status": "missing_grounded_ltl",
            "semantic_approximation": True,
        }
        return [[start[0], start[1]]], details, summary
    try:
        return plan_fast_sequential_astar(
            map_state=map_state,
            traversable=traversable,
            symbols=symbols,
            grounded_ltl=grounded_ltl,
            start=start,
            object_radius=object_radius,
            verbose=verbose,
        )
    except Exception as exc:
        details = [
            {
                "kind": "fast_astar",
                "status": "planning_failure",
                "reason": str(exc),
                "grounded_ltl": grounded_ltl,
                "semantic_approximation": True,
            }
        ]
        summary = {
            "planner_mode": "fast",
            "planner_type": "sequential_astar",
            "status": "planning_failure",
            "reason": str(exc),
            "grounded_ltl": grounded_ltl,
            "semantic_approximation": True,
        }
        return [[start[0], start[1]]], details, summary
