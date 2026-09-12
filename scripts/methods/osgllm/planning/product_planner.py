"""Product-state A* planner over SemPathBench cells and LTL residuals."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
from collections.abc import Mapping, Sequence

from scripts.methods.lang2ltl.ltl import Formula, is_accepting, positive_eventual_aps, progress
from scripts.methods.osgllm.scene_graph.graph_types import Cell
from scripts.methods.osgllm.scene_graph.propositions import PropositionModel
from scripts.methods.util.grid_astar import TraversableGrid, can_traverse_between, octile_heuristic


@dataclass(frozen=True)
class PlanningResult:
    status: str
    trajectory: list[list[int]]
    details: dict[str, object]


def _heuristic(
    cell: Cell,
    formula: Formula,
    ap_to_cells: Mapping[str, Sequence[Cell]],
) -> float:
    targets = positive_eventual_aps(formula)
    if not targets:
        return 0.0
    best = math.inf
    for ap_name in targets:
        cells = ap_to_cells.get(ap_name, ())
        for target in cells:
            best = min(best, octile_heuristic(cell, target))
    return 0.0 if math.isinf(best) else best


def _sample_ap_targets(
    ap_to_cells: Mapping[str, Sequence[Cell]],
    *,
    max_samples: int = 64,
) -> dict[str, tuple[Cell, ...]]:
    sampled: dict[str, tuple[Cell, ...]] = {}
    for ap_name, cells in ap_to_cells.items():
        if not cells:
            sampled[ap_name] = ()
            continue
        if len(cells) <= max_samples:
            sampled[ap_name] = tuple(cells)
            continue
        stride = max(1, len(cells) // max_samples)
        values = list(cells[::stride][:max_samples])
        center_row = sum(row for row, _col in cells) / len(cells)
        center_col = sum(col for _row, col in cells) / len(cells)
        values.append(
            min(
                cells,
                key=lambda item: (item[0] - center_row) ** 2 + (item[1] - center_col) ** 2,
            )
        )
        sampled[ap_name] = tuple(dict.fromkeys(values))
    return sampled


def _reconstruct(
    came_from: Mapping[tuple[int, int, Formula], tuple[int, int, Formula]],
    current: tuple[int, int, Formula],
) -> list[list[int]]:
    path = [(current[0], current[1])]
    while current in came_from:
        current = came_from[current]
        path.append((current[0], current[1]))
    path.reverse()
    return [[row, col] for row, col in path]


def product_astar(
    traversable: TraversableGrid,
    proposition_model: PropositionModel,
    start: Cell,
    formula: Formula,
    *,
    max_expansions: int | None,
    max_seconds: float | None,
    verbose: bool = False,
) -> PlanningResult:
    start_time = time.monotonic()
    initial_formula = progress(formula, proposition_model.labels_for_cell(start))
    details: dict[str, object] = {
        "planner_type": "occupancy_product_anchor_astar",
        "initial_formula": str(formula),
        "initial_progressed_formula": str(initial_formula),
        "use_ltl_heuristic": True,
        "use_llm_heuristic": False,
        "expanded_states": {"anchor": 0, "abstract": 0},
        "timeout_seconds": max_seconds,
        "max_expansions": max_expansions,
    }
    if verbose:
        print(f"[osgllm:planner] initial={formula} progressed={initial_formula}", flush=True)
    if is_accepting(initial_formula):
        details["status"] = "accepted_at_start"
        details["path_cost"] = 0.0
        return PlanningResult("SUCCESS", [[start[0], start[1]]], details)
    sampled_ap_to_cells = _sample_ap_targets(proposition_model.ap_to_cells)

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
    start_node = (start[0], start[1], initial_formula)
    open_heap: list[tuple[float, int, tuple[int, int, Formula]]] = [
        (_heuristic(start, initial_formula, sampled_ap_to_cells), 0, start_node)
    ]
    came_from: dict[tuple[int, int, Formula], tuple[int, int, Formula]] = {}
    g_score = {start_node: 0.0}
    closed: set[tuple[int, int, Formula]] = set()
    counter = 1
    expanded = 0
    grid_size = len(traversable)

    while open_heap:
        if max_seconds is not None and time.monotonic() - start_time > max_seconds:
            details["status"] = "timeout"
            details["expanded_states"] = {"anchor": expanded, "abstract": 0}
            details["elapsed_seconds"] = time.monotonic() - start_time
            return PlanningResult("PLANNER_TIMEOUT", [[start[0], start[1]]], details)
        _priority, _index, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if is_accepting(current[2]):
            trajectory = _reconstruct(came_from, current)
            details["status"] = "accepted"
            details["expanded_states"] = {"anchor": expanded, "abstract": 0}
            details["elapsed_seconds"] = time.monotonic() - start_time
            details["path_cost"] = g_score[current]
            details["accepting_formula_state"] = str(current[2])
            return PlanningResult("SUCCESS", trajectory, details)
        closed.add(current)
        expanded += 1
        if max_expansions is not None and expanded >= max_expansions:
            details["status"] = "max_expansions"
            details["expanded_states"] = {"anchor": expanded, "abstract": 0}
            details["elapsed_seconds"] = time.monotonic() - start_time
            return PlanningResult("PLANNER_TIMEOUT", [[start[0], start[1]]], details)
        if verbose and expanded % 25000 == 0:
            print(
                f"[osgllm:planner] expanded={expanded} open={len(open_heap)} state={current[2]}",
                flush=True,
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
            labels = proposition_model.labels_for_cell((next_row, next_col))
            next_formula = progress(formula_state, labels)
            if next_formula.op == "false":
                continue
            neighbor = (next_row, next_col, next_formula)
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            priority = tentative + _heuristic(
                (next_row, next_col),
                next_formula,
                sampled_ap_to_cells,
            )
            heapq.heappush(open_heap, (priority, counter, neighbor))
            counter += 1

    details["status"] = "failed_no_accepting_state"
    details["expanded_states"] = {"anchor": expanded, "abstract": 0}
    details["elapsed_seconds"] = time.monotonic() - start_time
    return PlanningResult("NO_FEASIBLE_PATH", [[start[0], start[1]]], details)
