"""Cost-aware A* planner for grounded GroundPlan segments."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from scripts.evaluation.evaluation_metrics import clearance_obstacle_cells
from scripts.methods.groundplan.config import (
    CLEARANCE_RADIUS_METERS,
    DEFAULT_PLANNER_MODE,
    DEFAULT_RELATIVE_COST_MODE,
    FAR_SIGMA_METERS,
    MAX_EXPANSIONS,
    NEAR_RADIUS_METERS,
    OBJECT_GOAL_RADIUS_METERS,
    PATH_SHAPE_RADIUS_METERS,
    RELATIVE_RADIUS_METERS,
    RELATIVE_COST_MODES,
    WEIGHT_CLEARANCE,
    WEIGHT_FAR,
    WEIGHT_NEAR,
    WEIGHT_PATH_SHAPE,
    WEIGHT_RELATIVE,
    WEIGHT_SMOOTHNESS,
)
from scripts.methods.groundplan.grounding.scene import SceneMap
from scripts.methods.groundplan.ir import (
    Cell,
    GPRef,
    GroundedConstraint,
    GroundedProgram,
    GroundedSegment,
    PlannedProgram,
    PlannedSegment,
    path_length,
)
from scripts.methods.util.grid_astar import can_traverse_between


Direction = tuple[int, int, float]
State = tuple[int, int, int, int]
LayerState = tuple[int, int, int]


DIRECTIONS: tuple[Direction, ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
)


@dataclass(frozen=True)
class _LayeredStage:
    mask: np.ndarray
    traversable: np.ndarray
    soft_field: np.ndarray
    smoothness_mask: np.ndarray | None
    segment_index: int
    name: str
    is_destination: bool
    track_heading: bool = False


@dataclass
class _LayerSearchResult:
    frontier: dict[LayerState, float]
    source_by_target: dict[LayerState, LayerState]
    expanded: int
    pushed: int
    peak_open: int
    complete: bool


def _all_program_constraints(grounded: GroundedProgram) -> tuple[GroundedConstraint, ...]:
    """Collect each declared constraint once, independent of its host segment.

    ``GroundedSegment.constraints`` is the declaration location produced by
    grounding code. It is deliberately not the execution scope: execution is
    determined below from ``GroundedConstraint.segment_scope``.
    """
    collected: list[GroundedConstraint] = []
    seen: set[tuple[str, object]] = set()
    for host_segment in grounded.segments:
        for constraint in host_segment.constraints:
            # Direct-CaP finalization assigns stable ids. Retain an identity
            # fallback for manually constructed, unnamed test/program objects.
            key: tuple[str, object] = (
                ("constraint_id", constraint.constraint_id)
                if constraint.constraint_id
                else ("object_identity", id(constraint))
            )
            if key not in seen:
                seen.add(key)
                collected.append(constraint)
    return tuple(collected)


def _active_constraints(
    constraints: Sequence[GroundedConstraint],
    segment_id: str,
) -> tuple[GroundedConstraint, ...]:
    return tuple(
        constraint
        for constraint in constraints
        if not constraint.segment_scope or segment_id in constraint.segment_scope
    )


def active_constraints_for_segment(
    grounded: GroundedProgram,
    segment_id: str,
) -> tuple[GroundedConstraint, ...]:
    """Return constraints activated by their explicit scope in ``segment_id``.

    An empty scope is global. A non-empty tuple is an explicit set of segment
    ids. This intentionally examines every declared constraint, so a global or
    cross-segment constraint is not limited by where it was stored.
    """
    return _active_constraints(_all_program_constraints(grounded), segment_id)



def plan_grounded_program(
    scene: SceneMap,
    grounded: GroundedProgram,
    *,
    max_expansions: int = MAX_EXPANSIONS,
    planner_mode: str = DEFAULT_PLANNER_MODE,
    heuristic_weight: float = 1.0,
    relative_cost_mode: str = DEFAULT_RELATIVE_COST_MODE,
    relative_weight: float = WEIGHT_RELATIVE,
    soft_weight_scale: float = 1.0,
    include_clearance: bool = True,
    global_min_path_improvement_m: float | None = None,
) -> PlannedProgram:
    """Plan a frozen grounded program with the selected deterministic backend."""
    if relative_cost_mode not in RELATIVE_COST_MODES:
        raise ValueError(
            f"Unsupported relative cost mode: {relative_cost_mode!r}; "
            f"expected one of {RELATIVE_COST_MODES}."
        )
    if relative_weight < 0.0:
        raise ValueError("relative_weight must be non-negative")
    if soft_weight_scale < 0.0:
        raise ValueError("soft_weight_scale must be non-negative")
    if (
        global_min_path_improvement_m is not None
        and global_min_path_improvement_m < 0.0
    ):
        raise ValueError("global_min_path_improvement_m must be non-negative")
    if planner_mode == "sequential_greedy_astar":
        return _plan_sequential_greedy(
            scene,
            grounded,
            max_expansions=max_expansions,
            relative_cost_mode=relative_cost_mode,
            relative_weight=relative_weight,
            soft_weight_scale=soft_weight_scale,
            include_clearance=include_clearance,
        )
    if planner_mode == "global_layered_dp":
        if heuristic_weight != 1.0:
            raise ValueError("global_layered_dp is exact and requires heuristic_weight=1.0")
        return _plan_global_layered_dp(
            scene,
            grounded,
            max_expansions=max_expansions,
            relative_cost_mode=relative_cost_mode,
            relative_weight=relative_weight,
            soft_weight_scale=soft_weight_scale,
            include_clearance=include_clearance,
            global_min_path_improvement_m=global_min_path_improvement_m,
        )
    if planner_mode != "global_progress_astar":
        raise ValueError(f"Unsupported GroundPlan planner mode: {planner_mode!r}")
    if heuristic_weight < 1.0:
        raise ValueError("heuristic_weight must be at least 1.0")
    return _plan_global_progress_astar(
        scene, grounded, max_expansions=max_expansions, heuristic_weight=heuristic_weight,
        relative_cost_mode=relative_cost_mode, relative_weight=relative_weight,
        soft_weight_scale=soft_weight_scale, include_clearance=include_clearance,
    )


def _plan_sequential_greedy(
    scene: SceneMap,
    grounded: GroundedProgram,
    *,
    max_expansions: int = MAX_EXPANSIONS,
    relative_cost_mode: str,
    relative_weight: float,
    soft_weight_scale: float,
    include_clearance: bool,
) -> PlannedProgram:
    if grounded.status != "success":
        return PlannedProgram([], (), "failed", grounded.failure_reason)

    current = _single_cell(scene.start)
    if current is None:
        return PlannedProgram([], (), "failed", "START_UNRESOLVED")
    full_path: list[list[int]] = [[current[0], current[1]]]
    planned_segments: list[PlannedSegment] = []
    diagnostics: list[str] = []
    all_constraints = _all_program_constraints(grounded)

    for segment in grounded.segments:
        active_constraints = _active_constraints(all_constraints, segment.id)
        planned = plan_segment(
            scene,
            segment,
            current,
            constraints=active_constraints,
            max_expansions=max_expansions,
            relative_cost_mode=relative_cost_mode,
            relative_weight=relative_weight,
            soft_weight_scale=soft_weight_scale,
            include_clearance=include_clearance,
        )
        planned_segments.append(planned)
        if planned.status != "success" or planned.endpoint is None:
            diagnostics.append(f"{segment.id}:{planned.status}")
            return PlannedProgram(
                full_path,
                tuple(planned_segments),
                "failed",
                planned.status,
                tuple(diagnostics),
            )
        if planned.path:
            full_path.extend(planned.path[1:] if full_path and full_path[-1] == planned.path[0] else planned.path)
        current = planned.endpoint

    return PlannedProgram(full_path, tuple(planned_segments), "success", diagnostics=tuple(diagnostics))


def _plan_global_layered_dp(
    scene: SceneMap,
    grounded: GroundedProgram,
    *,
    max_expansions: int,
    relative_cost_mode: str,
    relative_weight: float,
    soft_weight_scale: float,
    include_clearance: bool,
    global_min_path_improvement_m: float | None,
) -> PlannedProgram:
    """Exactly optimize ordered arrival regions one dense search layer at a time.

    Unlike the position-progress product search, this dynamic program retains
    the best cumulative cost for every legal arrival state in the current
    stage only.  A multi-source Dijkstra pass transfers those values to every
    first-entry state of the next stage.  This preserves global endpoint
    choice while keeping peak dense storage independent of the stage count.

    ``max_expansions <= 0`` means unlimited.  A sequential plan is computed as
    an executable incumbent and returned if a positive resource limit stops
    the exact layered search before it can prove a result.
    """
    planner_started = time.perf_counter()
    incumbent = _plan_sequential_greedy(
        scene,
        grounded,
        # The incumbent is the safety path. It must not inherit a deliberately
        # small global-search budget, otherwise a bounded global experiment
        # could still degrade to a start-only failure.
        max_expansions=0,
        relative_cost_mode=relative_cost_mode,
        relative_weight=relative_weight,
        soft_weight_scale=soft_weight_scale,
        include_clearance=include_clearance,
    )
    if grounded.status != "success":
        return _layered_incumbent_or_failure(
            incumbent,
            grounded.failure_reason or "GROUNDING_FAILED",
            planner_started,
            search_complete=True,
        )

    start = _single_cell(scene.start)
    if start is None:
        return _layered_incumbent_or_failure(
            incumbent, "START_UNRESOLVED", planner_started, search_complete=True
        )
    base = _traversable_mask(scene)
    if not base[start[0], start[1]]:
        return _layered_incumbent_or_failure(
            incumbent, "START_NOT_TRAVERSABLE", planner_started, search_complete=True
        )

    all_constraints = _all_program_constraints(grounded)
    stages: list[_LayeredStage] = []
    segment_data: list[dict[str, object]] = []
    for segment_index, segment in enumerate(grounded.segments):
        active = _active_constraints(all_constraints, segment.id)
        hard = _compile_hard(scene, active)
        forbidden = np.asarray(hard["forbidden"], dtype=bool)
        traversable = base & ~forbidden
        soft_field, soft_details = _compile_soft(
            scene,
            active,
            relative_cost_mode=relative_cost_mode,
            relative_weight=relative_weight,
            soft_weight_scale=soft_weight_scale,
            include_clearance=include_clearance,
        )
        goal_mask, goal_details = _goal_mask(scene, segment.target)
        required_masks = [np.asarray(mask, dtype=bool) for mask in hard["required"]]
        smoothness_masks = list(hard["required_smoothness_masks"])
        raw_masks = required_masks + [goal_mask]
        stage_smoothness = smoothness_masks + [None]
        stage_names = [f"required_{index + 1}" for index in range(len(required_masks))] + [
            "destination"
        ]
        for stage_name, raw_mask, smoothness_mask in zip(
            stage_names, raw_masks, stage_smoothness
        ):
            target_mask = np.asarray(raw_mask, dtype=bool) & traversable
            if not target_mask.any():
                reason = (
                    "EMPTY_TARGET_REGION"
                    if stage_name == "destination"
                    else "REQUIRED_REGION_BLOCKED"
                )
                return _layered_incumbent_or_failure(
                    incumbent,
                    reason,
                    planner_started,
                    search_complete=True,
                    extra={"segment_id": segment.id, "stage": stage_name},
                )
            stages.append(
                _LayeredStage(
                    target_mask,
                    traversable,
                    soft_field,
                    smoothness_mask,
                    segment_index,
                    stage_name,
                    stage_name == "destination",
                )
            )
        segment_data.append(
            {
                "segment_id": segment.id,
                "goal": goal_details,
                "hard": hard["details"],
                "soft": soft_details,
                "active_constraint_ids": [constraint.constraint_id for constraint in active],
                "stage_count": len(raw_masks),
            }
        )

    # Heading is part of a layer state only while some current/future stage
    # uses a path-shape turn cost. Ordinary object-to-object tasks therefore
    # retain one state per cell instead of paying a blanket 9x state factor.
    heading_sensitive = False
    staged_reversed: list[_LayeredStage] = []
    for stage in reversed(stages):
        heading_sensitive = heading_sensitive or stage.smoothness_mask is not None
        staged_reversed.append(
            _LayeredStage(
                stage.mask,
                stage.traversable,
                stage.soft_field,
                stage.smoothness_mask,
                stage.segment_index,
                stage.name,
                stage.is_destination,
                heading_sensitive,
            )
        )
    stages = list(reversed(staged_reversed))

    initial_progress = 0
    while initial_progress < len(stages) and stages[initial_progress].mask[start]:
        initial_progress += 1

    start_state: LayerState = (start[0], start[1], -1)
    frontier: dict[LayerState, float] = {start_state: 0.0}
    frontier_history: list[dict[LayerState, float]] = [
        {start_state: 0.0} for _stage in stages
    ]
    backrefs: list[dict[LayerState, LayerState] | None] = [None for _stage in stages]
    layer_details: list[dict[str, object]] = []
    total_expanded = 0
    total_pushed = 0
    peak_open = 0
    search_started = time.perf_counter()

    for stage_index, stage in enumerate(stages):
        if stage_index < initial_progress:
            layer_details.append(
                {
                    "stage_index": stage_index,
                    "segment_index": stage.segment_index,
                    "stage": stage.name,
                    "status": "satisfied_at_start",
                    "arrival_state_count": 1,
                    "expanded_states": 0,
                    "track_heading": stage.track_heading,
                }
            )
            continue
        remaining_limit = (
            None
            if max_expansions <= 0
            else max(0, max_expansions - total_expanded)
        )
        result = _layered_multi_source_search(
            scene,
            stage,
            frontier,
            expansion_limit=remaining_limit,
        )
        total_expanded += result.expanded
        total_pushed += result.pushed
        peak_open = max(peak_open, result.peak_open)
        layer_details.append(
            {
                "stage_index": stage_index,
                "segment_index": stage.segment_index,
                "stage": stage.name,
                "status": (
                    "success"
                    if result.complete and result.frontier
                    else "MAX_EXPANSIONS"
                    if not result.complete
                    else "NO_FEASIBLE_PATH"
                ),
                "source_state_count": len(frontier),
                "arrival_state_count": len(result.frontier),
                "expanded_states": result.expanded,
                "pushed_states": result.pushed,
                "peak_open_size": result.peak_open,
                "track_heading": stage.track_heading,
            }
        )
        if not result.complete:
            return _layered_incumbent_or_failure(
                incumbent,
                "MAX_EXPANSIONS",
                planner_started,
                search_complete=False,
                extra={
                    "initial_progress": initial_progress,
                    "completed_stage_count": stage_index,
                    "stage_count": len(stages),
                    "expanded_states": total_expanded,
                    "pushed_states": total_pushed,
                    "peak_open_size": peak_open,
                    "layer_details": layer_details,
                    "search_seconds": time.perf_counter() - search_started,
                },
            )
        if not result.frontier:
            return _layered_incumbent_or_failure(
                incumbent,
                "NO_FEASIBLE_PATH",
                planner_started,
                search_complete=True,
                extra={
                    "initial_progress": initial_progress,
                    "completed_stage_count": stage_index,
                    "stage_count": len(stages),
                    "expanded_states": total_expanded,
                    "pushed_states": total_pushed,
                    "peak_open_size": peak_open,
                    "layer_details": layer_details,
                    "search_seconds": time.perf_counter() - search_started,
                },
            )
        frontier = result.frontier
        frontier_history[stage_index] = frontier
        backrefs[stage_index] = result.source_by_target

    search_seconds = time.perf_counter() - search_started
    if not stages:
        return PlannedProgram(
            [[start[0], start[1]]],
            (),
            "success",
            details={
                "planner_mode": "global_layered_dp",
                "search_complete": True,
                "optimality_proven": True,
                "stage_count": 0,
                "expanded_states": 0,
                "planning_seconds": time.perf_counter() - planner_started,
            },
        )

    best_final_state = min(frontier, key=lambda state: (frontier[state], state))
    total_cost = frontier[best_final_state]
    selected_targets: list[LayerState] = [start_state for _stage in stages]
    selected_sources: list[LayerState] = [start_state for _stage in stages]
    current_state = best_final_state
    for stage_index in range(len(stages) - 1, initial_progress - 1, -1):
        selected_targets[stage_index] = current_state
        stage_backrefs = backrefs[stage_index]
        if stage_backrefs is None or current_state not in stage_backrefs:
            raise RuntimeError(f"Missing layered back-reference for stage {stage_index}")
        source_state = stage_backrefs[current_state]
        selected_sources[stage_index] = source_state
        current_state = source_state
    for stage_index in range(initial_progress):
        selected_sources[stage_index] = start_state
        selected_targets[stage_index] = start_state

    reconstruction_started = time.perf_counter()
    stage_paths: list[list[list[int]]] = []
    reconstruction_expanded = 0
    for stage_index, stage in enumerate(stages):
        if stage_index < initial_progress:
            stage_paths.append([[start[0], start[1]]])
            continue
        source_state = selected_sources[stage_index]
        target_state = selected_targets[stage_index]
        stage_path, stage_cost, expanded = _layered_fixed_endpoint_path(
            scene,
            stage,
            source_state,
            target_state,
            require_move=bool(stage.mask[source_state[0], source_state[1]]),
        )
        reconstruction_expanded += expanded
        if stage_path is None:
            return _layered_incumbent_or_failure(
                incumbent,
                "RECONSTRUCTION_FAILED",
                planner_started,
                search_complete=True,
                extra={
                    "optimality_proven": False,
                    "stage_index": stage_index,
                    "expanded_states": total_expanded,
                    "layer_details": layer_details,
                },
            )
        source_cost = (
            0.0
            if stage_index == 0
            else frontier_history[stage_index - 1].get(source_state, 0.0)
        )
        expected_stage_cost = frontier_history[stage_index][target_state] - source_cost
        layer_details[stage_index]["selected_source"] = list(source_state)
        layer_details[stage_index]["selected_arrival"] = list(target_state)
        layer_details[stage_index]["selected_stage_cost"] = stage_cost
        layer_details[stage_index]["expected_stage_cost"] = expected_stage_cost
        layer_details[stage_index]["reconstruction_expanded_states"] = expanded
        stage_paths.append(stage_path)

    trajectory: list[list[int]] = [[start[0], start[1]]]
    segment_paths: list[list[list[int]]] = [[] for _segment in grounded.segments]
    segment_expanded = [0 for _segment in grounded.segments]
    for stage_index, (stage, stage_path) in enumerate(zip(stages, stage_paths)):
        if stage_path:
            trajectory.extend(
                stage_path[1:]
                if trajectory and trajectory[-1] == stage_path[0]
                else stage_path
            )
        existing = segment_paths[stage.segment_index]
        if not existing:
            segment_paths[stage.segment_index] = [list(cell) for cell in stage_path]
        elif stage_path:
            existing.extend(stage_path[1:] if existing[-1] == stage_path[0] else stage_path)
        segment_expanded[stage.segment_index] += int(
            layer_details[stage_index].get("expanded_states", 0)
        )

    planned_segments: list[PlannedSegment] = []
    for segment_index, segment in enumerate(grounded.segments):
        path = segment_paths[segment_index] or [[start[0], start[1]]]
        endpoint = (int(path[-1][0]), int(path[-1][1]))
        planned_segments.append(
            PlannedSegment(
                segment.id,
                path,
                endpoint,
                "success",
                path_length(path),
                segment_expanded[segment_index],
                {
                    **segment_data[segment_index],
                    "planner_mode": "global_layered_dp",
                    "arrival_cell": list(endpoint),
                    "layer_details": [
                        detail
                        for detail in layer_details
                        if detail.get("segment_index") == segment_index
                    ],
                    "global_total_cost": total_cost,
                },
            )
        )

    sequential_path_length = path_length(incumbent.trajectory)
    global_path_length = path_length(trajectory)
    path_improvement_m = (
        sequential_path_length - global_path_length
    ) * float(scene.resolution)
    if (
        global_min_path_improvement_m is not None
        and path_improvement_m + 1e-9 < global_min_path_improvement_m
    ):
        return PlannedProgram(
            incumbent.trajectory,
            incumbent.segments,
            "success",
            diagnostics=tuple(incumbent.diagnostics)
            + ("global_layered_dp:minimum_path_improvement_not_met",),
            details={
                "planner_mode": "global_layered_dp",
                "search_complete": True,
                "optimality_proven": True,
                "fallback_used": True,
                "fallback_reason": "MINIMUM_PATH_IMPROVEMENT_NOT_MET",
                "selection": "sequential_incumbent",
                "global_min_path_improvement_m": global_min_path_improvement_m,
                "global_candidate_path_improvement_m": path_improvement_m,
                "global_candidate_path_length": global_path_length,
                "sequential_incumbent_path_length": sequential_path_length,
                "global_candidate_total_cost": total_cost,
                "expanded_states": total_expanded,
                "planning_seconds": time.perf_counter() - planner_started,
            },
        )

    return PlannedProgram(
        trajectory,
        tuple(planned_segments),
        "success",
        details={
            "planner_mode": "global_layered_dp",
            "selection": "global_candidate",
            "global_min_path_improvement_m": global_min_path_improvement_m,
            "global_candidate_path_improvement_m": path_improvement_m,
            "search_complete": True,
            "optimality_proven": True,
            "fallback_used": False,
            "initial_progress": initial_progress,
            "stage_count": len(stages),
            "state_space_upper_bound": sum(
                int(stage.traversable.sum()) * (9 if stage.track_heading else 1)
                for stage in stages
            ),
            "peak_layer_state_upper_bound": max(
                int(stage.traversable.sum()) * (9 if stage.track_heading else 1)
                for stage in stages
            ),
            "peak_dense_layers": 1,
            "expanded_states": total_expanded,
            "pushed_states": total_pushed,
            "peak_open_size": peak_open,
            "reconstruction_expanded_states": reconstruction_expanded,
            "search_seconds": search_seconds,
            "reconstruction_seconds": time.perf_counter() - reconstruction_started,
            "planning_seconds": time.perf_counter() - planner_started,
            "total_cost": total_cost,
            "sequential_incumbent_status": incumbent.status,
            "sequential_incumbent_path_length": path_length(incumbent.trajectory),
            "selected_arrival_cells": [
                list(segment.endpoint) if segment.endpoint else None
                for segment in planned_segments
            ],
            "layer_details": layer_details,
        },
    )


def _layered_incumbent_or_failure(
    incumbent: PlannedProgram,
    reason: str,
    planner_started: float,
    *,
    search_complete: bool,
    extra: Mapping[str, object] | None = None,
) -> PlannedProgram:
    details: dict[str, object] = {
        "planner_mode": "global_layered_dp",
        "search_complete": search_complete,
        "optimality_proven": False,
        "fallback_used": incumbent.status == "success",
        "fallback_reason": reason,
        "sequential_incumbent_status": incumbent.status,
        "sequential_incumbent_path_length": path_length(incumbent.trajectory),
        **dict(extra or {}),
        "planning_seconds": time.perf_counter() - planner_started,
    }
    if incumbent.status == "success":
        return PlannedProgram(
            incumbent.trajectory,
            incumbent.segments,
            "success",
            diagnostics=tuple(incumbent.diagnostics) + (f"global_layered_dp:{reason}",),
            details=details,
        )
    return PlannedProgram(
        incumbent.trajectory,
        incumbent.segments,
        "failed",
        reason,
        tuple(incumbent.diagnostics) + (f"global_layered_dp:{reason}",),
        details,
    )


def _layered_multi_source_search(
    scene: SceneMap,
    stage: _LayeredStage,
    sources: Mapping[LayerState, float],
    *,
    expansion_limit: int | None,
) -> _LayerSearchResult:
    """Transfer accumulated costs to every first-entry state of one stage."""
    heading_slots = 9 if stage.track_heading else 1
    shape = (heading_slots, scene.grid_size, scene.grid_size)
    distance = np.full(shape, np.inf, dtype=np.float64)
    origin = np.full(shape, -1, dtype=np.int32)
    source_states = list(sources)
    open_heap: list[tuple[float, int, int, int, int]] = []
    target_costs: dict[LayerState, float] = {}
    target_sources: dict[LayerState, LayerState] = {}
    counter = 0
    pushed = 0

    def heading_slot(heading: int) -> int:
        return heading + 1 if stage.track_heading else 0

    def normalized_heading(heading: int) -> int:
        return heading if stage.track_heading else -1

    def record_target(state: LayerState, cost: float, source_index: int) -> None:
        if cost < target_costs.get(state, math.inf):
            target_costs[state] = cost
            target_sources[state] = source_states[source_index]

    def push_non_target(
        row: int,
        col: int,
        heading: int,
        cost: float,
        source_index: int,
    ) -> None:
        nonlocal counter, pushed
        slot = heading_slot(heading)
        if cost >= float(distance[slot, row, col]):
            return
        distance[slot, row, col] = cost
        origin[slot, row, col] = source_index
        heapq.heappush(open_heap, (cost, counter, row, col, normalized_heading(heading)))
        counter += 1
        pushed += 1

    def seed_successor(source_index: int, source_state: LayerState, source_cost: float) -> None:
        row, col, previous_heading = source_state
        for next_heading, (dr, dc, step_cost) in enumerate(DIRECTIONS):
            next_row, next_col = row + dr, col + dc
            if not _layered_can_move(scene, stage, row, col, next_row, next_col):
                continue
            transition_cost = _layered_transition_cost(
                stage,
                row,
                col,
                next_row,
                next_col,
                previous_heading,
                next_heading,
                step_cost,
            )
            next_state = (
                next_row,
                next_col,
                normalized_heading(next_heading),
            )
            candidate = source_cost + transition_cost
            if stage.mask[next_row, next_col]:
                record_target(next_state, candidate, source_index)
            else:
                push_non_target(
                    next_row,
                    next_col,
                    next_heading,
                    candidate,
                    source_index,
                )

    for source_index, source_state in enumerate(source_states):
        row, col, heading = source_state
        if not stage.traversable[row, col]:
            continue
        source_cost = float(sources[source_state])
        # A later ordered stage requires a transition before it can be
        # completed. Seeding successors preserves the existing one-stage-per-
        # move semantics for overlapping masks, including leave-and-reenter
        # behavior for a one-cell repeated target.
        if stage.mask[row, col]:
            seed_successor(source_index, source_state, source_cost)
        else:
            push_non_target(row, col, heading, source_cost, source_index)

    expanded = 0
    peak_open = len(open_heap)
    while open_heap:
        if expansion_limit is not None and expanded >= expansion_limit:
            return _LayerSearchResult(
                target_costs,
                target_sources,
                expanded,
                pushed,
                peak_open,
                False,
            )
        cost, _tie, row, col, heading = heapq.heappop(open_heap)
        slot = heading_slot(heading)
        if cost != float(distance[slot, row, col]):
            continue
        expanded += 1
        source_index = int(origin[slot, row, col])
        for next_heading, (dr, dc, step_cost) in enumerate(DIRECTIONS):
            next_row, next_col = row + dr, col + dc
            if not _layered_can_move(scene, stage, row, col, next_row, next_col):
                continue
            transition_cost = _layered_transition_cost(
                stage,
                row,
                col,
                next_row,
                next_col,
                heading,
                next_heading,
                step_cost,
            )
            candidate = cost + transition_cost
            next_state = (
                next_row,
                next_col,
                normalized_heading(next_heading),
            )
            if stage.mask[next_row, next_col]:
                record_target(next_state, candidate, source_index)
                continue
            push_non_target(
                next_row,
                next_col,
                next_heading,
                candidate,
                source_index,
            )
        peak_open = max(peak_open, len(open_heap))

    return _LayerSearchResult(
        target_costs,
        target_sources,
        expanded,
        pushed,
        peak_open,
        True,
    )


def _layered_can_move(
    scene: SceneMap,
    stage: _LayeredStage,
    row: int,
    col: int,
    next_row: int,
    next_col: int,
) -> bool:
    return bool(
        0 <= next_row < scene.grid_size
        and 0 <= next_col < scene.grid_size
        and stage.traversable[next_row, next_col]
        and can_traverse_between(scene.traversable, row, col, next_row, next_col)
    )


def _layered_transition_cost(
    stage: _LayeredStage,
    row: int,
    col: int,
    next_row: int,
    next_col: int,
    previous_heading: int,
    next_heading: int,
    step_cost: float,
) -> float:
    smoothness = 0.0
    if (
        stage.smoothness_mask is not None
        and (stage.smoothness_mask[row, col] or stage.smoothness_mask[next_row, next_col])
    ):
        dr, dc, _cost = DIRECTIONS[next_heading]
        smoothness = _turn_smoothness(previous_heading, dr, dc)
    return (
        step_cost
        + float(stage.soft_field[next_row, next_col])
        + WEIGHT_SMOOTHNESS * smoothness
    )


def _layered_fixed_endpoint_path(
    scene: SceneMap,
    stage: _LayeredStage,
    source: LayerState,
    target: LayerState,
    *,
    require_move: bool,
) -> tuple[list[list[int]] | None, float, int]:
    """Reconstruct one selected layer edge without retaining every layer tree."""
    start = (source[0], source[1], source[2] if stage.track_heading else -1, False)

    def heuristic(row: int, col: int) -> float:
        dr = abs(row - target[0])
        dc = abs(col - target[1])
        diagonal = min(dr, dc)
        return diagonal * math.sqrt(2.0) + (max(dr, dc) - diagonal)

    open_heap: list[tuple[float, int, tuple[int, int, int, bool]]] = [
        (heuristic(start[0], start[1]), 0, start)
    ]
    parents: dict[
        tuple[int, int, int, bool], tuple[int, int, int, bool]
    ] = {}
    g_score: dict[tuple[int, int, int, bool], float] = {start: 0.0}
    closed: set[tuple[int, int, int, bool]] = set()
    counter = 1
    expanded = 0
    while open_heap:
        _priority, _tie, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        row, col, heading, moved = state
        target_heading_matches = target[2] < 0 or heading == target[2]
        if (
            row == target[0]
            and col == target[1]
            and target_heading_matches
            and (moved or not require_move)
        ):
            state_path = [state]
            while state_path[-1] in parents:
                state_path.append(parents[state_path[-1]])
            state_path.reverse()
            return [[item[0], item[1]] for item in state_path], g_score[state], expanded
        closed.add(state)
        expanded += 1
        for next_heading, (dr, dc, step_cost) in enumerate(DIRECTIONS):
            next_row, next_col = row + dr, col + dc
            if not _layered_can_move(scene, stage, row, col, next_row, next_col):
                continue
            normalized_heading = next_heading if stage.track_heading else -1
            if stage.mask[next_row, next_col] and not (
                next_row == target[0]
                and next_col == target[1]
                and (target[2] < 0 or normalized_heading == target[2])
            ):
                continue
            successor = (next_row, next_col, normalized_heading, True)
            tentative = g_score[state] + _layered_transition_cost(
                stage,
                row,
                col,
                next_row,
                next_col,
                heading,
                next_heading,
                step_cost,
            )
            if tentative >= g_score.get(successor, math.inf):
                continue
            parents[successor] = state
            g_score[successor] = tentative
            heapq.heappush(
                open_heap,
                (tentative + heuristic(next_row, next_col), counter, successor),
            )
            counter += 1
    return None, math.inf, expanded



def _plan_global_progress_astar(
    scene: SceneMap,
    grounded: GroundedProgram,
    *,
    max_expansions: int,
    heuristic_weight: float,
    relative_cost_mode: str,
    relative_weight: float,
    soft_weight_scale: float,
    include_clearance: bool,
) -> PlannedProgram:
    """Globally optimize frozen targets over position-progress A* states.

    Every goal/required region remains a full boolean mask.  Progress advances
    once on entry, so the active segment (and therefore its scope-aware hard
    constraints and soft field) is always unambiguous.
    """
    if grounded.status != "success":
        return PlannedProgram([], (), "failed", grounded.failure_reason)
    planner_started = time.perf_counter()
    start = _single_cell(scene.start)
    if start is None:
        return PlannedProgram([], (), "failed", "START_UNRESOLVED")
    base = _traversable_mask(scene)
    if not base[start[0], start[1]]:
        return PlannedProgram([], (), "failed", "START_NOT_TRAVERSABLE")

    all_constraints = _all_program_constraints(grounded)
    stage_masks: list[np.ndarray] = []
    # The legal area may change at a segment boundary. Keep it per stage so
    # the reverse-distance heuristic respects the hard constraints that apply
    # while travelling toward that particular stage.
    stage_traversable: list[np.ndarray] = []
    stage_segment_indexes: list[int] = []
    stage_is_destination: list[bool] = []
    forbidden_by_segment: list[np.ndarray] = []
    soft_by_segment: list[np.ndarray] = []
    segment_data: list[dict[str, object]] = []
    deferred_failure: tuple[str, str, str] | None = None
    for segment_index, segment in enumerate(grounded.segments):
        active = _active_constraints(all_constraints, segment.id)
        hard = _compile_hard(scene, active)
        forbidden = np.asarray(hard["forbidden"], dtype=bool)
        soft_field, soft_details = _compile_soft(
            scene,
            active,
            relative_cost_mode=relative_cost_mode,
            relative_weight=relative_weight,
            soft_weight_scale=soft_weight_scale,
            include_clearance=include_clearance,
        )
        forbidden_by_segment.append(forbidden)
        soft_by_segment.append(soft_field)
        goal_mask, goal_details = _goal_mask(scene, segment.target)
        raw_masks = list(hard["required"]) + [goal_mask]
        stage_names = [f"required_{index + 1}" for index in range(len(hard["required"]))] + ["destination"]
        for stage_name, raw_mask in zip(stage_names, raw_masks):
            target_mask = np.asarray(raw_mask, dtype=bool) & base & ~forbidden
            if not target_mask.any():
                deferred_failure = (
                    "EMPTY_TARGET_REGION" if stage_name == "destination" else "REQUIRED_REGION_BLOCKED",
                    segment.id,
                    stage_name,
                )
                break
            stage_masks.append(target_mask)
            stage_traversable.append(base & ~forbidden)
            stage_segment_indexes.append(segment_index)
            stage_is_destination.append(stage_name == "destination")
        segment_data.append(
            {
                "segment_id": segment.id,
                "goal": goal_details,
                "hard": hard["details"],
                "soft": soft_details,
                "active_constraint_ids": [constraint.constraint_id for constraint in active],
                "stage_count": len(raw_masks),
            }
        )
        if deferred_failure is not None:
            break

    if not stage_masks:
        if deferred_failure is not None:
            reason, segment_id, stage_name = deferred_failure
            return PlannedProgram(
                [[start[0], start[1]]], (), "failed", reason,
                (f"{segment_id}:{stage_name}",),
                {
                    "planner_mode": "global_progress_astar",
                    "segment_id": segment_id,
                    "stage": stage_name,
                    "partial_trajectory": False,
                    "completed_stage_count": 0,
                },
            )
        return PlannedProgram(
            [[start[0], start[1]]], (), "success",
            details={"planner_mode": "global_progress_astar", "expanded_states": 0},
        )

    # A start already inside ordered goals satisfies the largest prefix before
    # search. Later transitions advance by at most one even when masks overlap.
    initial_progress = 0
    while initial_progress < len(stage_masks) and stage_masks[initial_progress][start[0], start[1]]:
        initial_progress += 1

    heuristic_started = time.perf_counter()
    reverse_distances = [
        _reverse_distance_field(scene, traversable, mask)
        for traversable, mask in zip(stage_traversable, stage_masks)
    ]
    # transition_bounds[i] lower-bounds travelling from stage i to stage
    # i + 1, under the constraints active after stage i is completed. The
    # future bound for progress p must include only transitions still ahead
    # of p. Including earlier transitions would overestimate the remaining
    # cost and invalidate the A* optimality guarantee.
    transition_bounds: list[float] = []
    for index in range(len(stage_masks) - 1):
        values = reverse_distances[index + 1][stage_masks[index]]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return PlannedProgram(
                [[start[0], start[1]]], (), "failed", "NO_FEASIBLE_PATH",
                (f"unreachable_stage_{index + 1}",),
                {"planner_mode": "global_progress_astar", "unreachable_stage": index + 1},
            )
        transition_bounds.append(float(np.min(finite)))
    tail_bounds = [0.0] * len(stage_masks)
    for index in range(len(stage_masks) - 2, -1, -1):
        tail_bounds[index] = transition_bounds[index] + tail_bounds[index + 1]

    heuristic_precompute_seconds = time.perf_counter() - heuristic_started

    def heuristic(row: int, col: int, progress: int) -> float:
        if progress == len(stage_masks):
            return 0.0
        return float(reverse_distances[progress][row, col]) + tail_bounds[progress]

    initial = (start[0], start[1], initial_progress)
    initial_h = heuristic(*initial)
    if not math.isfinite(initial_h):
        return PlannedProgram([[start[0], start[1]]], (), "failed", "NO_FEASIBLE_PATH")
    search_started = time.perf_counter()
    open_heap: list[tuple[float, float, int, int, int, tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (heuristic_weight * initial_h, initial_h, -initial_progress, start[0] * scene.grid_size + start[1], 0, initial))
    parents: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    g_score: dict[tuple[int, int, int], float] = {initial: 0.0}
    closed: set[tuple[int, int, int]] = set()
    counter = 1
    pushed = 1
    expanded = 0
    peak_open = 1
    goal_state: tuple[int, int, int] | None = None
    best_partial_state = initial
    best_partial_rank = (initial_progress, -initial_h, 0.0)
    while open_heap:
        _f, _h, _negative_progress, _cell_index, _tie, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        row, col, progress = state
        if progress == len(stage_masks):
            goal_state = state
            break
        closed.add(state)
        expanded += 1
        if max_expansions > 0 and expanded > max_expansions:
            break
        segment_index = stage_segment_indexes[progress]
        forbidden = forbidden_by_segment[segment_index]
        soft_field = soft_by_segment[segment_index]
        for dr, dc, step_cost in DIRECTIONS:
            next_row, next_col = row + dr, col + dc
            if (
                next_row < 0 or next_row >= scene.grid_size
                or next_col < 0 or next_col >= scene.grid_size
                or not base[next_row, next_col]
                or forbidden[next_row, next_col]
                or not can_traverse_between(scene.traversable, row, col, next_row, next_col)
            ):
                continue
            next_progress = progress + 1 if stage_masks[progress][next_row, next_col] else progress
            successor = (next_row, next_col, next_progress)
            tentative = g_score[state] + step_cost + float(soft_field[next_row, next_col])
            if tentative >= g_score.get(successor, math.inf):
                continue
            successor_h = heuristic(*successor)
            parents[successor] = state
            g_score[successor] = tentative
            partial_rank = (next_progress, -successor_h, -tentative)
            if partial_rank > best_partial_rank:
                best_partial_state = successor
                best_partial_rank = partial_rank
            heapq.heappush(
                open_heap,
                (
                    tentative + heuristic_weight * successor_h,
                    successor_h,
                    -next_progress,
                    next_row * scene.grid_size + next_col,
                    counter,
                    successor,
                ),
            )
            counter += 1
            pushed += 1
        peak_open = max(peak_open, len(open_heap))

    search_seconds = time.perf_counter() - search_started
    diagnostics: dict[str, object] = {
        "planner_mode": "global_progress_astar",
        "heuristic_weight": heuristic_weight,
        "optimality": "global_discrete_augmented_graph" if heuristic_weight == 1.0 else "weighted_astar_bounded_suboptimal",
        "initial_progress": initial_progress,
        "stage_count": len(stage_masks),
        "expanded_states": expanded,
        "pushed_states": pushed,
        "peak_open_size": peak_open,
        "heuristic_precompute_seconds": heuristic_precompute_seconds,
        "heuristic_uses_stage_hard_constraints": True,
        "search_seconds": search_seconds,
        "segment_data": segment_data,
    }
    if goal_state is None:
        state_path = _reconstruct_progress_states(parents, best_partial_state)
        trajectory = [[row, col] for row, col, _progress in state_path]
        failure_reason = (
            "MAX_EXPANSIONS"
            if max_expansions > 0 and expanded > max_expansions
            else "NO_FEASIBLE_PATH"
        )
        diagnostics.update(
            {
                "partial_trajectory": len(trajectory) > 1,
                "completed_stage_count": best_partial_state[2],
                "partial_total_cost": g_score[best_partial_state],
            }
        )
        return PlannedProgram(
            trajectory, (), "failed", failure_reason,
            details=diagnostics,
        )

    reconstruction_started = time.perf_counter()
    state_path = _reconstruct_progress_states(parents, goal_state)
    trajectory = [[row, col] for row, col, _progress in state_path]
    if deferred_failure is not None:
        failure_reason, segment_id, stage_name = deferred_failure
        diagnostics.update(
            {
                "partial_trajectory": len(trajectory) > 1,
                "completed_stage_count": goal_state[2],
                "partial_total_cost": g_score[goal_state],
                "deferred_failure_segment_id": segment_id,
                "deferred_failure_stage": stage_name,
                "reconstruction_seconds": time.perf_counter() - reconstruction_started,
                "planning_seconds": time.perf_counter() - planner_started,
            }
        )
        return PlannedProgram(
            trajectory, (), "failed", failure_reason,
            (f"{segment_id}:{stage_name}",),
            details=diagnostics,
        )
    planned_segments = _reconstruct_global_segments(
        grounded, state_path, stage_is_destination, g_score[goal_state], expanded, segment_data
    )
    return PlannedProgram(
        trajectory, tuple(planned_segments), "success",
        details={
            **diagnostics,
            "total_cost": g_score[goal_state],
            "reconstruction_seconds": time.perf_counter() - reconstruction_started,
            "planning_seconds": time.perf_counter() - planner_started,
            "selected_arrival_cells": [list(segment.endpoint) if segment.endpoint else None for segment in planned_segments],
        },
    )


def _reverse_distance_field(scene: SceneMap, traversable: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """Reverse Dijkstra lower bound on the same 8-connected base graph."""
    distance = np.full((scene.grid_size, scene.grid_size), np.inf, dtype=float)
    queue: list[tuple[float, int, Cell]] = []
    counter = 0
    for row, col in zip(*np.nonzero(target_mask & traversable)):
        distance[int(row), int(col)] = 0.0
        heapq.heappush(queue, (0.0, counter, (int(row), int(col))))
        counter += 1
    while queue:
        cost, _tie, cell = heapq.heappop(queue)
        if cost != float(distance[cell[0], cell[1]]):
            continue
        for dr, dc, step_cost in DIRECTIONS:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if (
                neighbor[0] < 0 or neighbor[0] >= scene.grid_size
                or neighbor[1] < 0 or neighbor[1] >= scene.grid_size
                or not traversable[neighbor[0], neighbor[1]]
                or not can_traverse_between(scene.traversable, cell[0], cell[1], neighbor[0], neighbor[1])
            ):
                continue
            candidate = cost + step_cost
            if candidate >= float(distance[neighbor[0], neighbor[1]]):
                continue
            distance[neighbor[0], neighbor[1]] = candidate
            heapq.heappush(queue, (candidate, counter, neighbor))
            counter += 1
    return distance


def _reconstruct_progress_states(
    parents: Mapping[tuple[int, int, int], tuple[int, int, int]],
    state: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    path = [state]
    while path[-1] in parents:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def _reconstruct_global_segments(
    grounded: GroundedProgram,
    state_path: Sequence[tuple[int, int, int]],
    stage_is_destination: Sequence[bool],
    total_cost: float,
    expanded: int,
    segment_data: Sequence[Mapping[str, object]],
) -> list[PlannedSegment]:
    boundaries: list[int] = []
    stage_index = 0
    for _segment in grounded.segments:
        while stage_index < len(stage_is_destination) and not stage_is_destination[stage_index]:
            stage_index += 1
        completed = stage_index + 1
        boundaries.append(next(index for index, (_row, _col, progress) in enumerate(state_path) if progress >= completed))
        stage_index += 1
    result: list[PlannedSegment] = []
    start_index = 0
    for index, (segment, end_index) in enumerate(zip(grounded.segments, boundaries)):
        path = [[row, col] for row, col, _progress in state_path[start_index : end_index + 1]]
        endpoint = (path[-1][0], path[-1][1])
        result.append(
            PlannedSegment(
                segment.id, path, endpoint, "success", path_length(path), expanded,
                {
                    **dict(segment_data[index]),
                    "planner_mode": "global_progress_astar",
                    "arrival_cell": list(endpoint),
                    "boundary_index": end_index,
                    "global_total_cost": total_cost,
                },
            )
        )
        start_index = end_index
    return result

def plan_segment(
    scene: SceneMap,
    segment: GroundedSegment,
    start: Cell,
    *,
    constraints: Sequence[GroundedConstraint] | None = None,
    max_expansions: int,
    relative_cost_mode: str = DEFAULT_RELATIVE_COST_MODE,
    relative_weight: float = WEIGHT_RELATIVE,
    soft_weight_scale: float = 1.0,
    include_clearance: bool = True,
) -> PlannedSegment:
    # Keep standalone calls backwards compatible. Program-level planning always
    # supplies the explicitly scope-filtered constraint list above.
    active_constraints = tuple(segment.constraints if constraints is None else constraints)
    goal_mask, goal_details = _goal_mask(scene, segment.target)
    if not goal_mask.any():
        return PlannedSegment(segment.id, [], None, "EMPTY_TARGET_REGION", 0.0, 0, goal_details)

    hard = _compile_hard(scene, active_constraints)
    if hard["forbidden"][start]:
        return PlannedSegment(segment.id, [[start[0], start[1]]], start, "START_IN_FORBIDDEN_REGION", 0.0, 0, goal_details)
    traversable_mask = _traversable_mask(scene) & ~hard["forbidden"]
    goal_mask = goal_mask & traversable_mask
    if not goal_mask.any():
        return PlannedSegment(segment.id, [], None, "GOAL_BLOCKED_BY_HARD_CONSTRAINT", 0.0, 0, goal_details)

    soft_field, soft_details = _compile_soft(
        scene,
        active_constraints,
        relative_cost_mode=relative_cost_mode,
        relative_weight=relative_weight,
        soft_weight_scale=soft_weight_scale,
        include_clearance=include_clearance,
    )
    details = {
        **goal_details,
        "hard": hard["details"],
        "soft": soft_details,
        "required_region_count": len(hard["required"]),
        "smoothness_weight": WEIGHT_SMOOTHNESS,
        "smoothness_scope": "circle_required_stages",
        "active_constraint_ids": [constraint.constraint_id for constraint in active_constraints],
    }
    if hard["required"]:
        path, endpoint, expanded, stage_details = _plan_via_required_regions(
            scene,
            traversable_mask,
            hard["required"],
            hard["required_smoothness_masks"],
            goal_mask,
            soft_field,
            start,
            max_expansions=max_expansions,
        )
        details["ordered_stage_planning"] = stage_details
    else:
        goal_distance = distance_transform_edt(~goal_mask)
        path, endpoint, expanded = _astar_cell_only(
            scene,
            traversable_mask,
            goal_mask,
            goal_distance,
            soft_field,
            start,
            max_expansions=max_expansions,
        )
    if path is None or endpoint is None:
        return PlannedSegment(segment.id, [[start[0], start[1]]], start, "NO_FEASIBLE_PATH", 0.0, expanded, details)
    return PlannedSegment(segment.id, path, endpoint, "success", path_length(path), expanded, details)


def _plan_via_required_regions(
    scene: SceneMap,
    traversable_mask: np.ndarray,
    required_masks: Sequence[np.ndarray],
    required_smoothness_masks: Sequence[np.ndarray | None],
    goal_mask: np.ndarray,
    soft_field: np.ndarray,
    start: Cell,
    *,
    max_expansions: int,
) -> tuple[list[list[int]] | None, Cell | None, int, list[dict[str, object]]]:
    current = start
    current_heading = -1
    full_path: list[list[int]] = [[start[0], start[1]]]
    total_expanded = 0
    stage_details: list[dict[str, object]] = []
    stage_targets = list(required_masks) + [goal_mask]

    for stage_index, raw_target_mask in enumerate(stage_targets):
        stage_name = "goal" if stage_index == len(stage_targets) - 1 else f"required_{stage_index + 1}"
        is_required_stage = stage_index < len(required_masks)
        stage_traversable = traversable_mask
        target_mask = raw_target_mask & stage_traversable
        if not target_mask.any():
            stage_details.append({"stage": stage_name, "status": "EMPTY_TARGET_REGION", "expanded_states": 0})
            return None, None, total_expanded, stage_details
        remaining_budget = (
            0 if max_expansions <= 0 else max(1, max_expansions - total_expanded)
        )
        goal_distance = distance_transform_edt(~target_mask)
        smoothness_mask = (
            required_smoothness_masks[stage_index]
            if stage_index < len(required_smoothness_masks)
            else None
        )
        stage_path, endpoint, expanded = _astar_cell_only(
            scene,
            stage_traversable,
            target_mask,
            goal_distance,
            soft_field,
            current,
            initial_heading=current_heading,
            smoothness_mask=smoothness_mask,
            max_expansions=remaining_budget,
        )
        total_expanded += expanded
        if stage_path is None or endpoint is None:
            stage_details.append(
                {
                    "stage": stage_name,
                    "status": "NO_FEASIBLE_PATH",
                    "expanded_states": expanded,
                    "target_cell_count": int(target_mask.sum()),
                }
            )
            return None, None, total_expanded, stage_details
        full_path.extend(stage_path[1:] if full_path and full_path[-1] == stage_path[0] else stage_path)
        current = endpoint
        current_heading = _last_heading_index(stage_path, current_heading)
        stage_details.append(
            {
                "stage": stage_name,
                "status": "success",
                "expanded_states": expanded,
                "target_cell_count": int(target_mask.sum()),
                "endpoint": [endpoint[0], endpoint[1]],
                "waypoint_count": len(stage_path),
                "allowed_future_regions": is_required_stage,
                "smoothness_active": smoothness_mask is not None,
                "smoothness_scope_cell_count": int(smoothness_mask.sum()) if smoothness_mask is not None else 0,
            }
        )
    return full_path, current, total_expanded, stage_details


def _last_heading_index(path: Sequence[Sequence[int]], fallback: int = -1) -> int:
    for previous, current in zip(reversed(path[:-1]), reversed(path[1:])):
        dr = int(current[0]) - int(previous[0])
        dc = int(current[1]) - int(previous[1])
        for index, (direction_row, direction_col, _cost) in enumerate(DIRECTIONS):
            if dr == direction_row and dc == direction_col:
                return index
    return fallback


def _astar_cell_only(
    scene: SceneMap,
    traversable: np.ndarray,
    goal_mask: np.ndarray,
    goal_distance: np.ndarray,
    soft_field: np.ndarray,
    start: Cell,
    *,
    initial_heading: int = -1,
    smoothness_mask: np.ndarray | None = None,
    max_expansions: int,
) -> tuple[list[list[int]] | None, Cell | None, int]:
    def heuristic(row: int, col: int) -> float:
        return float(goal_distance[row, col])

    open_heap: list[tuple[float, int, Cell]] = []
    heapq.heappush(open_heap, (heuristic(start[0], start[1]), 0, start))
    counter = 1
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    closed: set[Cell] = set()
    expanded = 0
    grid_size = scene.grid_size

    while open_heap:
        _priority, _index, cell = heapq.heappop(open_heap)
        if cell in closed:
            continue
        row, col = cell
        if goal_mask[row, col]:
            return _reconstruct_cells(came_from, cell), cell, expanded
        closed.add(cell)
        expanded += 1
        if max_expansions > 0 and expanded >= max_expansions:
            return None, None, expanded

        heading_index = _incoming_heading_index(came_from, cell, initial_heading)
        for dr, dc, step_cost in DIRECTIONS:
            next_row = row + dr
            next_col = col + dc
            if (
                next_row < 0
                or next_row >= grid_size
                or next_col < 0
                or next_col >= grid_size
                or not traversable[next_row, next_col]
                or not can_traverse_between(scene.traversable, row, col, next_row, next_col)
            ):
                continue
            neighbor = (next_row, next_col)
            smoothness = 0.0
            if smoothness_mask is not None and (smoothness_mask[row, col] or smoothness_mask[next_row, next_col]):
                smoothness = _turn_smoothness(heading_index, dr, dc)
            transition_cost = step_cost + float(soft_field[next_row, next_col]) + WEIGHT_SMOOTHNESS * smoothness
            tentative = g_score[cell] + transition_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = cell
            g_score[neighbor] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(next_row, next_col), counter, neighbor))
            counter += 1
    return None, None, expanded


def _incoming_heading_index(came_from: Mapping[Cell, Cell], cell: Cell, initial_heading: int) -> int:
    previous = came_from.get(cell)
    if previous is None:
        return initial_heading
    dr = cell[0] - previous[0]
    dc = cell[1] - previous[1]
    for index, (direction_row, direction_col, _cost) in enumerate(DIRECTIONS):
        if dr == direction_row and dc == direction_col:
            return index
    return initial_heading


def _turn_smoothness(heading_index: int, dr: int, dc: int) -> float:
    if heading_index < 0:
        return 0.0
    prev_dr, prev_dc, _ = DIRECTIONS[heading_index]
    if prev_dr == dr and prev_dc == dc:
        return 0.0
    angle_a = math.atan2(prev_dr, prev_dc)
    angle_b = math.atan2(dr, dc)
    diff = abs((angle_b - angle_a + math.pi) % (2 * math.pi) - math.pi)
    return (diff / math.pi) ** 2


def _reconstruct_cells(came_from: Mapping[Cell, Cell], cell: Cell) -> list[list[int]]:
    path = [cell]
    current = cell
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return [[row, col] for row, col in path]


def _astar(
    scene: SceneMap,
    traversable: np.ndarray,
    goal_mask: np.ndarray,
    goal_distance: np.ndarray,
    required_masks: Sequence[np.ndarray],
    soft_field: np.ndarray,
    initial: State,
    *,
    max_expansions: int,
) -> tuple[list[list[int]] | None, Cell | None, int]:
    def heuristic(row: int, col: int) -> float:
        return float(goal_distance[row, col])

    open_heap: list[tuple[float, int, State]] = []
    heapq.heappush(open_heap, (heuristic(initial[0], initial[1]), 0, initial))
    counter = 1
    came_from: dict[State, State] = {}
    g_score: dict[State, float] = {initial: 0.0}
    closed: set[State] = set()
    expanded = 0
    grid_size = scene.grid_size

    while open_heap:
        _priority, _index, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        row, col, required_index, heading_index = state
        if goal_mask[row, col] and required_index >= len(required_masks):
            return _reconstruct(came_from, state), (row, col), expanded
        closed.add(state)
        expanded += 1
        if max_expansions > 0 and expanded >= max_expansions:
            return None, None, expanded

        for next_heading, (dr, dc, step_cost) in enumerate(DIRECTIONS):
            next_row = row + dr
            next_col = col + dc
            if (
                next_row < 0
                or next_row >= grid_size
                or next_col < 0
                or next_col >= grid_size
                or not traversable[next_row, next_col]
                or not can_traverse_between(scene.traversable, row, col, next_row, next_col)
            ):
                continue
            next_required = _advance_required_index((next_row, next_col), required_index, required_masks)
            smoothness = 0.0
            if heading_index >= 0 and heading_index != next_heading:
                prev_dr, prev_dc, _ = DIRECTIONS[heading_index]
                angle_a = math.atan2(prev_dr, prev_dc)
                angle_b = math.atan2(dr, dc)
                diff = abs((angle_b - angle_a + math.pi) % (2 * math.pi) - math.pi)
                smoothness = (diff / math.pi) ** 2
            transition_cost = step_cost + float(soft_field[next_row, next_col]) + WEIGHT_SMOOTHNESS * smoothness
            next_state = (next_row, next_col, next_required, next_heading)
            tentative = g_score[state] + transition_cost
            if tentative >= g_score.get(next_state, math.inf):
                continue
            came_from[next_state] = state
            g_score[next_state] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(next_row, next_col), counter, next_state))
            counter += 1
    return None, None, expanded


def _reconstruct(came_from: Mapping[State, State], state: State) -> list[list[int]]:
    path = [(state[0], state[1])]
    current = state
    while current in came_from:
        current = came_from[current]
        path.append((current[0], current[1]))
    path.reverse()
    return [[row, col] for row, col in path]


def _advance_required_index(cell: Cell, index: int, required_masks: Sequence[np.ndarray]) -> int:
    current = index
    while current < len(required_masks) and required_masks[current][cell[0], cell[1]]:
        current += 1
    return current


def _goal_mask(scene: SceneMap, target: GPRef) -> tuple[np.ndarray, dict[str, object]]:
    traversable = _traversable_mask(scene)
    if target.kind == "position":
        mask = np.zeros((scene.grid_size, scene.grid_size), dtype=bool)
        cell = _single_cell(target)
        if cell:
            mask[cell[0], cell[1]] = True
        return mask & traversable, {"target_type": "position"}
    if target.kind == "room":
        return _mask_from_cells(scene, target.cells) & traversable, {"target_type": "room", "target_id": target.id}
    if target.kind == "region":
        goal = _mask_from_cells(scene, target.cells) & traversable
        fallback_used = False
        if not goal.any() and target.construction.get("type") == "near" and target.center is not None:
            room_id = target.construction.get("room_id")
            room_mask = np.zeros((scene.grid_size, scene.grid_size), dtype=bool)
            if isinstance(room_id, str) and room_id in scene.rooms:
                room_mask = _mask_from_cells(scene, scene.rooms[room_id].cells)
            fallback = _nearest_traversable_cell_in_mask(scene, target.center, room_mask & traversable)
            if fallback is not None:
                goal[fallback[0], fallback[1]] = True
                fallback_used = True
        return goal, {
            "target_type": "region",
            "target_id": target.id,
            "goal_cell_count": int(goal.sum()),
            "nearest_same_room_fallback": fallback_used,
        }
    if target.kind == "entity":
        object_mask = _mask_from_cells(scene, target.cells)
        distance = distance_transform_edt(~object_mask)
        radius_cells = max(1, round(OBJECT_GOAL_RADIUS_METERS / scene.resolution))
        room_mask = np.ones((scene.grid_size, scene.grid_size), dtype=bool)
        if target.room_id and target.room_id in scene.rooms:
            room_mask = _mask_from_cells(scene, scene.rooms[target.room_id].cells)
        goal = (distance > 0) & (distance <= radius_cells) & room_mask & traversable
        fallback_used = False
        if not goal.any() and target.center is not None:
            fallback = _nearest_traversable_cell_in_mask(scene, target.center, room_mask & traversable)
            if fallback is not None:
                goal[fallback[0], fallback[1]] = True
                fallback_used = True
        return goal, {
            "target_type": "entity",
            "target_id": target.id,
            "target_category": target.category,
            "goal_radius_cells": radius_cells,
            "goal_cell_count": int(goal.sum()),
            "nearest_same_room_fallback": fallback_used,
        }
    return np.zeros((scene.grid_size, scene.grid_size), dtype=bool), {"target_type": target.kind}


def _compile_hard(scene: SceneMap, constraints: Sequence[GroundedConstraint]) -> dict[str, object]:
    forbidden = np.zeros((scene.grid_size, scene.grid_size), dtype=bool)
    required: list[np.ndarray] = []
    required_smoothness_masks: list[np.ndarray | None] = []
    details: list[dict[str, object]] = []
    for constraint in constraints:
        if constraint.kind == "forbid":
            for ref in constraint.refs:
                mask = _mask_from_cells(scene, ref.cells) & _scope_mask(scene, constraint.spatial_scope).astype(bool)
                forbidden |= mask
                details.append({"kind": "forbid", "ref": ref.to_json(), "cell_count": int(mask.sum()), "scope": _scope_to_json(constraint.spatial_scope)})
        elif constraint.kind == "require_visit":
            for ref in constraint.refs:
                mask = _mask_from_cells(scene, ref.cells) & _scope_mask(scene, constraint.spatial_scope).astype(bool)
                required.append(mask)
                required_smoothness_masks.append(None)
                details.append({"kind": "require_visit", "ref": ref.to_json(), "cell_count": int(mask.sum())})
        elif constraint.kind == "require_visit_in_order":
            for ref in constraint.refs:
                mask = _mask_from_cells(scene, ref.cells) & _scope_mask(scene, constraint.spatial_scope).astype(bool)
                required.append(mask)
                smoothness_mask = _path_shape_smoothness_mask(scene, ref)
                required_smoothness_masks.append(smoothness_mask)
                details.append(
                    {
                        "kind": "require_visit_in_order",
                        "ref": ref.to_json(),
                        "cell_count": int(mask.sum()),
                        "smoothness_active": smoothness_mask is not None,
                        "smoothness_scope_cell_count": int(smoothness_mask.sum()) if smoothness_mask is not None else 0,
                    }
                )
    return {
        "forbidden": forbidden,
        "required": required,
        "required_smoothness_masks": required_smoothness_masks,
        "details": details,
    }


def _path_shape_smoothness_mask(scene: SceneMap, ref: GPRef) -> np.ndarray | None:
    if ref.construction.get("type") not in {"circle_waypoint", "wall_waypoint"}:
        return None
    room_id = ref.construction.get("room_id") or ref.room_id
    if isinstance(room_id, str) and room_id in scene.rooms:
        return _mask_from_cells(scene, scene.rooms[room_id].cells) & _traversable_mask(scene)
    mask = _mask_from_cells(scene, ref.cells) & _traversable_mask(scene)
    return mask if mask.any() else None


def _compile_soft(
    scene: SceneMap,
    constraints: Sequence[GroundedConstraint],
    *,
    relative_cost_mode: str = DEFAULT_RELATIVE_COST_MODE,
    relative_weight: float = WEIGHT_RELATIVE,
    soft_weight_scale: float = 1.0,
    include_clearance: bool = True,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if relative_cost_mode not in RELATIVE_COST_MODES:
        raise ValueError(f"Unsupported relative cost mode: {relative_cost_mode!r}")
    fields: list[np.ndarray] = []
    details: list[dict[str, object]] = []
    for constraint in constraints:
        if constraint.kind == "prefer_near":
            ref = constraint.refs[0]
            scope = _scope_mask(scene, constraint.spatial_scope)
            field = np.clip(_distance_to_ref(scene, ref) / max(1.0, NEAR_RADIUS_METERS / scene.resolution), 0.0, 1.0)
            fields.append(field * scope * WEIGHT_NEAR * soft_weight_scale)
            details.append({"kind": "prefer_near", "ref": ref.to_json(), "weight": WEIGHT_NEAR * soft_weight_scale, "scope": _scope_to_json(constraint.spatial_scope)})
        elif constraint.kind == "prefer_far":
            ref = constraint.refs[0]
            scope = _scope_mask(scene, constraint.spatial_scope)
            sigma = max(1.0, FAR_SIGMA_METERS / scene.resolution)
            field = np.exp(-_distance_to_ref(scene, ref) / sigma)
            fields.append(field * scope * WEIGHT_FAR * soft_weight_scale)
            details.append({"kind": "prefer_far", "ref": ref.to_json(), "weight": WEIGHT_FAR * soft_weight_scale, "scope": _scope_to_json(constraint.spatial_scope)})
        elif constraint.kind == "prefer_relative" and len(constraint.refs) >= 2:
            first, second = constraint.refs[:2]
            scope = _scope_mask(scene, constraint.spatial_scope)
            radius = max(1.0, RELATIVE_RADIUS_METERS / scene.resolution)
            d_first = _distance_to_ref(scene, first)
            d_second = _distance_to_ref(scene, second)
            if constraint.relation == "farther_from":
                d_close, d_far = d_second, d_first
            else:
                d_close, d_far = d_first, d_second
            if relative_cost_mode == "difference":
                relative_cost = np.clip(np.maximum(0.0, d_close - d_far) / radius, 0.0, 1.0)
            else:
                # This is 1 / (1 + D_far / D_close), a bounded monotonic
                # transform of the benchmark metric. Unlike a clipped inverse
                # ratio, it retains a gradient on both sides of D_far=D_close.
                denominator = d_close + d_far
                relative_cost = np.divide(
                    d_close,
                    denominator,
                    out=np.full_like(d_close, 0.5),
                    where=denominator > 1e-6,
                )
            fields.append(relative_cost * scope * relative_weight * soft_weight_scale)
            details.append({"kind": "prefer_relative", "refs": [first.to_json(), second.to_json()], "relation": constraint.relation, "weight": relative_weight * soft_weight_scale, "cost_mode": relative_cost_mode})
        elif constraint.kind == "prefer_path_shape":
            scope = _scope_mask(scene, constraint.spatial_scope)
            field = _path_shape_field(scene, constraint.refs)
            fields.append(field * scope * WEIGHT_PATH_SHAPE * soft_weight_scale)
            details.append(
                {
                    "kind": "prefer_path_shape",
                    "refs": [ref.to_json() for ref in constraint.refs],
                    "weight": WEIGHT_PATH_SHAPE * soft_weight_scale,
                    "radius_meters": PATH_SHAPE_RADIUS_METERS,
                    "scope": _scope_to_json(constraint.spatial_scope),
                }
            )

    if include_clearance:
        clearance = _clearance_field(scene) * WEIGHT_CLEARANCE
        fields.append(clearance)
        details.append({"kind": "clearance", "weight": WEIGHT_CLEARANCE, "scope": "whole_segment"})
    total = np.sum(fields, axis=0) if fields else np.zeros((scene.grid_size, scene.grid_size), dtype=float)
    return total, details


def _clearance_field(scene: SceneMap) -> np.ndarray:
    obstacle_mask = np.zeros((scene.grid_size, scene.grid_size), dtype=bool)
    for row, col, _object_id in clearance_obstacle_cells(scene.map_state):
        if 0 <= row < scene.grid_size and 0 <= col < scene.grid_size:
            obstacle_mask[row, col] = True
    if not obstacle_mask.any():
        obstacle_mask = ~_traversable_mask(scene)
    distance = distance_transform_edt(~obstacle_mask)
    radius = max(1.0, CLEARANCE_RADIUS_METERS / scene.resolution)
    deficit = np.maximum(0.0, radius - distance)
    return (deficit / radius) ** 2


def _path_shape_field(scene: SceneMap, refs: Sequence[GPRef]) -> np.ndarray:
    mask = np.zeros((scene.grid_size, scene.grid_size), dtype=bool)
    for ref in refs:
        mask |= _mask_from_cells(scene, ref.cells)
    if not mask.any():
        return np.zeros((scene.grid_size, scene.grid_size), dtype=float)
    radius = max(1.0, PATH_SHAPE_RADIUS_METERS / scene.resolution)
    distance = distance_transform_edt(~mask)
    return np.clip(distance / radius, 0.0, 1.0)

def _distance_to_ref(scene: SceneMap, ref: GPRef) -> np.ndarray:
    mask = _mask_from_cells(scene, ref.cells)
    if not mask.any() and ref.center is not None:
        row = max(0, min(scene.grid_size - 1, int(round(ref.center[0]))))
        col = max(0, min(scene.grid_size - 1, int(round(ref.center[1]))))
        mask[row, col] = True
    return distance_transform_edt(~mask)


def _scope_mask(scene: SceneMap, scope: object) -> np.ndarray:
    if isinstance(scope, GPRef):
        return _mask_from_cells(scene, scope.cells).astype(float)
    if isinstance(scope, tuple):
        mask = np.zeros((scene.grid_size, scene.grid_size), dtype=bool)
        for ref in scope:
            if isinstance(ref, GPRef):
                mask |= _mask_from_cells(scene, ref.cells)
        return mask.astype(float)
    return np.ones((scene.grid_size, scene.grid_size), dtype=float)


def _scope_to_json(scope: object) -> object:
    if isinstance(scope, GPRef):
        return scope.to_json()
    if isinstance(scope, tuple):
        return [ref.to_json() for ref in scope if isinstance(ref, GPRef)]
    return None


def _traversable_mask(scene: SceneMap) -> np.ndarray:
    return np.array(scene.traversable, dtype=bool)


def _mask_from_cells(scene: SceneMap, cells: Sequence[Cell] | frozenset[Cell]) -> np.ndarray:
    mask = np.zeros((scene.grid_size, scene.grid_size), dtype=bool)
    for row, col in cells:
        if 0 <= row < scene.grid_size and 0 <= col < scene.grid_size:
            mask[row, col] = True
    return mask


def _nearest_traversable_cell_in_mask(
    scene: SceneMap,
    center: tuple[float, float],
    allowed_mask: np.ndarray,
) -> Cell | None:
    if not allowed_mask.any():
        return None
    rows, cols = np.nonzero(allowed_mask)
    distances = (rows.astype(float) - center[0]) ** 2 + (cols.astype(float) - center[1]) ** 2
    index = int(np.argmin(distances))
    return int(rows[index]), int(cols[index])


def _single_cell(ref: GPRef) -> Cell | None:
    if ref.cells:
        return sorted(ref.cells)[0]
    if ref.center is None:
        return None
    return int(round(ref.center[0])), int(round(ref.center[1]))
