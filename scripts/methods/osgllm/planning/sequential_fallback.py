"""Explicit sequential AP fallback for simple eventual-goal formulas."""

from __future__ import annotations

from dataclasses import dataclass
import time
from collections.abc import Sequence

from scripts.methods.osgllm.scene_graph.graph_types import Cell
from scripts.methods.osgllm.scene_graph.propositions import PropositionModel
from scripts.methods.util.grid_astar import TraversableGrid, astar_path, octile_heuristic


@dataclass(frozen=True)
class SequentialFallbackResult:
    status: str
    trajectory: list[list[int]]
    details: dict[str, object]


def _candidate_cells(
    start: Cell,
    cells: Sequence[Cell],
    *,
    max_candidates: int,
) -> list[Cell]:
    return sorted(cells, key=lambda cell: octile_heuristic(start, cell))[:max_candidates]


def sequential_ap_astar(
    traversable: TraversableGrid,
    proposition_model: PropositionModel,
    start: Cell,
    goals: Sequence[str],
    *,
    forbidden_aps: Sequence[str] = (),
    max_candidates_per_goal: int = 96,
    max_seconds: float | None = None,
) -> SequentialFallbackResult:
    blocked_cells = {
        cell
        for ap_name in forbidden_aps
        for cell in proposition_model.ap_to_cells.get(ap_name, ())
    }
    if start in blocked_cells:
        return SequentialFallbackResult(
            "NO_FEASIBLE_PATH",
            [[start[0], start[1]]],
            {
                "planner_type": "sequential_ap_astar_fallback",
                "status": "start_in_forbidden_region",
                "forbidden_aps": list(forbidden_aps),
                "blocked_cell_count": len(blocked_cells),
                "segments": [],
            },
        )
    planning_grid = [
        [
            is_free and (row_index, col_index) not in blocked_cells
            for col_index, is_free in enumerate(row)
        ]
        for row_index, row in enumerate(traversable)
    ]
    trajectory: list[list[int]] = [[start[0], start[1]]]
    current = start
    details: list[dict[str, object]] = []
    deadline = time.monotonic() + max_seconds if max_seconds is not None else None

    for goal_index, ap_name in enumerate(goals, start=1):
        if deadline is not None and time.monotonic() >= deadline:
            return SequentialFallbackResult(
                "PLANNER_TIMEOUT",
                trajectory,
                {
                    "planner_type": "sequential_ap_astar_fallback",
                    "status": "timeout",
                    "timeout_seconds": max_seconds,
                    "segments": details,
                },
            )
        if ap_name in proposition_model.labels_for_cell(current):
            details.append(
                {
                    "goal_index": goal_index,
                    "ap": ap_name,
                    "status": "already_satisfied",
                    "cell": [current[0], current[1]],
                }
            )
            continue
        cells = proposition_model.ap_to_cells.get(ap_name, ())
        if not cells:
            return SequentialFallbackResult(
                "NO_FEASIBLE_PATH",
                trajectory,
                {
                    "planner_type": "sequential_ap_astar_fallback",
                    "status": "missing_ap_cells",
                    "failed_ap": ap_name,
                    "segments": details,
                },
            )
        best_path: list[list[int]] | None = None
        best_target: Cell | None = None
        for target in _candidate_cells(
            current,
            cells,
            max_candidates=max_candidates_per_goal,
        ):
            path = astar_path(planning_grid, current, target, deadline=deadline)
            if not path:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_target = target
        if deadline is not None and time.monotonic() >= deadline:
            return SequentialFallbackResult(
                "PLANNER_TIMEOUT",
                trajectory,
                {
                    "planner_type": "sequential_ap_astar_fallback",
                    "status": "timeout",
                    "timeout_seconds": max_seconds,
                    "segments": details,
                },
            )
        if best_path is None or best_target is None:
            return SequentialFallbackResult(
                "NO_FEASIBLE_PATH",
                trajectory,
                {
                    "planner_type": "sequential_ap_astar_fallback",
                    "status": "failed_segment",
                    "failed_ap": ap_name,
                    "segments": details,
                },
            )
        trajectory.extend(best_path[1:])
        current = (best_path[-1][0], best_path[-1][1])
        details.append(
            {
                "goal_index": goal_index,
                "ap": ap_name,
                "status": "connected",
                "target_cell": [best_target[0], best_target[1]],
                "segment_length": len(best_path),
            }
        )

    return SequentialFallbackResult(
        "SUCCESS",
        trajectory,
        {
            "planner_type": "sequential_ap_astar_fallback",
            "status": "accepted",
            "segments": details,
            "goal_count": len(goals),
            "trajectory_length": len(trajectory),
            "forbidden_aps": list(forbidden_aps),
            "blocked_cell_count": len(blocked_cells),
        },
    )
