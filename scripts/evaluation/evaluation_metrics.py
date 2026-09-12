#!/usr/bin/env python3
"""SemPathBench path-evaluation metrics."""

from __future__ import annotations

import heapq
import math
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

from scripts.evaluation.metric_cache import (
    load_instruction_segment_distance_field,
    load_map_metric_cache,
    warn_missing_instruction_cache,
    warn_missing_map_cache,
    write_instruction_metric_cache,
)
from scripts.evaluation.metric_geometry import (
    CLEARANCE_DISTANCE_CACHE_SUFFIX,
    CLEARANCE_DISTANCE_CACHE_VERSION,
    EPSILON,
    Point,
    REPO_ROOT,
    Trajectory,
    clearance_distance_field_from_obstacles,
    clearance_obstacle_cells,
    compute_clearance_cost,
    compute_hcs_progress,
    compute_ndtw_score,
    compute_path_length,
    compute_relative_preference_quality,
    load_clearance_distance_field,
    ordered_must_pass_constraints,
    point_inside_constraint,
    segment_intersects_constraint,
)
from scripts.methods.util.grid_astar import (
    TraversableGrid,
    build_traversable_grid,
    can_traverse_between,
    is_traversable,
)


Constraint = Mapping[str, object]
Cell = tuple[int, int]
_MAP_CACHE_KEY = "_metric_map_cache"


def grid_resolution_m(map_state: Mapping[str, object]) -> float:
    """Return the side length, in metres, of one grid cell."""
    metadata = map_state.get("metadata")
    if isinstance(metadata, Mapping):
        coordinate_frame = metadata.get("grid_coordinate_frame")
        map_info = metadata.get("map_info")
        for source in (coordinate_frame, map_info):
            if isinstance(source, Mapping):
                resolution = source.get("resolution")
                if (
                    isinstance(resolution, Real)
                    and not isinstance(resolution, bool)
                    and float(resolution) > 0
                ):
                    return float(resolution)
    return 1.0


def _point(value: object, label: str = "point") -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or not all(isinstance(item, Real) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{label} must be [row, col].")
    return float(value[0]), float(value[1])


def _constraint_id(constraint: Constraint, fallback: str) -> str:
    value = constraint.get("constraint_id")
    return str(value) if value is not None else fallback


def _constraint_hit(trajectory: Trajectory, constraint: Constraint) -> bool:
    return any(point_inside_constraint(point, constraint) for point in trajectory) or any(
        segment_intersects_constraint(start, end, constraint)
        for start, end in zip(trajectory, trajectory[1:])
    )


def _scope_is_global(constraint: Constraint) -> bool:
    scope = constraint.get("scope")
    if not isinstance(scope, Mapping):
        return True
    return scope.get("type") != "between_hard_constraints"


def _is_active_for_transition(
    constraint: Constraint,
    previous_goal: Constraint | None,
    expected_goal: Constraint,
) -> bool:
    scope = constraint.get("scope")
    if not isinstance(scope, Mapping) or scope.get("type") != "between_hard_constraints":
        return True
    from_order = scope.get("from_order")
    to_order = scope.get("to_order")
    expected_order = expected_goal.get("order")
    previous_order = previous_goal.get("order") if previous_goal is not None else None
    if to_order != expected_order:
        return False
    if from_order is None:
        return previous_goal is None
    return from_order == previous_order


def _active_avoids(
    hard_constraints: Sequence[Constraint],
    previous_goal: Constraint | None,
    expected_goal: Constraint,
) -> list[Constraint]:
    return [
        constraint
        for constraint in hard_constraints
        if constraint.get("kind") == "must_avoid"
        and _is_active_for_transition(constraint, previous_goal, expected_goal)
    ]


def compute_hcs(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
) -> tuple[float, dict[str, object]]:
    """Compute HCS, including the global must-avoid hard-zero rule."""

    global_avoids = [
        constraint
        for constraint in hard_constraints
        if constraint.get("kind") == "must_avoid" and _scope_is_global(constraint)
    ]
    violated_global_ids = [
        _constraint_id(constraint, f"global_must_avoid_{index}")
        for index, constraint in enumerate(global_avoids, start=1)
        if _constraint_hit(trajectory, constraint)
    ]

    local_only_constraints = [
        constraint
        for constraint in hard_constraints
        if constraint.get("kind") != "must_avoid" or not _scope_is_global(constraint)
    ]
    progress = compute_hcs_progress(trajectory, local_only_constraints)
    if not violated_global_ids:
        return float(progress["score"]), {
            **progress,
            "global_must_avoid_violations": [],
        }

    return 0.0, {
        **progress,
        "score": 0.0,
        "failed": True,
        "failure_type": "global_must_avoid_violation",
        "global_must_avoid_violations": violated_global_ids,
        "first_must_avoid_violation": violated_global_ids[0],
        "must_avoid_violations": [
            *list(progress.get("must_avoid_violations", [])),
            *violated_global_ids,
        ],
    }


def _first_ordered_waypoint_hits(
    trajectory: Trajectory,
    must_pass: Sequence[Constraint],
) -> dict[int, int]:
    """Return dense-waypoint hit indexes; Tutorial A* emits dense grid paths."""

    hits: dict[int, int] = {}
    cursor = 0
    for fallback_order, constraint in enumerate(must_pass, start=1):
        raw_order = constraint.get("order")
        order = int(raw_order) if isinstance(raw_order, int) else fallback_order
        for index in range(cursor, len(trajectory)):
            if point_inside_constraint(trajectory[index], constraint):
                hits[order] = index
                cursor = index
                break
    return hits


def _active_trajectory(
    trajectory: Trajectory,
    constraint: Constraint,
    hard_hits: Mapping[int, int],
) -> Trajectory:
    scope = constraint.get("scope")
    if not isinstance(scope, Mapping) or scope.get("type") != "between_hard_constraints":
        return trajectory
    from_order = scope.get("from_order")
    to_order = scope.get("to_order")
    if not isinstance(from_order, int) or not isinstance(to_order, int):
        return trajectory
    from_index = hard_hits.get(from_order)
    to_index = hard_hits.get(to_order)
    if from_index is not None and to_index is not None and to_index >= from_index:
        return trajectory[from_index : to_index + 1]
    if from_index is not None:
        return trajectory[from_index:]
    if to_index is not None:
        return trajectory[: to_index + 1]
    return trajectory


def _completed_segment_to_orders(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
) -> set[int]:
    """Return ``to_order`` values for the valid completed hard prefix."""

    must_pass = ordered_must_pass_constraints(hard_constraints)
    if len(must_pass) < 2:
        return set()
    progress = compute_hcs_progress(trajectory, hard_constraints)
    completed_count = int(progress["valid_completed_count"])
    completed_orders: set[int] = set()
    for fallback_order, constraint in enumerate(
        must_pass[1 : completed_count + 1],
        start=2,
    ):
        raw_order = constraint.get("order")
        if isinstance(raw_order, Real) and not isinstance(raw_order, bool):
            completed_orders.add(int(raw_order))
        else:
            completed_orders.add(fallback_order)
    return completed_orders


def _segment_gate(
    constraint: Constraint,
    completed_to_orders: set[int],
) -> tuple[bool, bool, int | None]:
    """Return whether a soft constraint is scoped and its segment succeeded."""

    scope = constraint.get("scope")
    if not isinstance(scope, Mapping) or scope.get("type") != "between_hard_constraints":
        return False, True, None
    raw_to_order = scope.get("to_order")
    if not isinstance(raw_to_order, Real) or isinstance(raw_to_order, bool):
        return True, False, None
    to_order = int(raw_to_order)
    return True, to_order in completed_to_orders, to_order


def _region_cells(constraint: Constraint, grid_size: int) -> list[Cell]:
    shape = constraint.get("shape")
    if shape == "freeform":
        cells = constraint.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            return []
        result: list[Cell] = []
        for cell in cells:
            row, col = _point(cell, "constraint cell")
            parsed = (int(round(row)), int(round(col)))
            if 0 <= parsed[0] < grid_size and 0 <= parsed[1] < grid_size:
                result.append(parsed)
        return sorted(set(result))

    center_row, center_col = _point(constraint.get("center"), "constraint center")
    if shape == "circle":
        radius = float(constraint.get("radius", 0.0))
        row_min, row_max = math.floor(center_row - radius), math.ceil(center_row + radius)
        col_min, col_max = math.floor(center_col - radius), math.ceil(center_col + radius)
    elif shape == "rectangle":
        width = float(constraint.get("width", 0.0))
        height = float(constraint.get("height", 0.0))
        row_min, row_max = math.floor(center_row - height / 2), math.ceil(center_row + height / 2)
        col_min, col_max = math.floor(center_col - width / 2), math.ceil(center_col + width / 2)
    else:
        return []
    result = []
    for row in range(max(0, row_min), min(grid_size - 1, row_max) + 1):
        for col in range(max(0, col_min), min(grid_size - 1, col_max) + 1):
            if point_inside_constraint((row, col), constraint):
                result.append((row, col))
    return result


def _reference_cells(constraint: Constraint, key: str) -> list[tuple[float, float]]:
    raw = constraint.get(key)
    if not isinstance(raw, Mapping):
        return []
    cells = raw.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        return []
    return [_point(cell, "reference cell") for cell in cells]


def _distance_to_cells(point: Point, cells: Sequence[tuple[float, float]]) -> float:
    row, col = _point(point, "trajectory point")
    if not cells:
        raise ValueError("Object reference must contain at least one footprint cell.")
    return min(math.hypot(row - cell_row, col - cell_col) for cell_row, cell_col in cells)


def _region_waypoints(trajectory: Trajectory, constraint: Constraint) -> list[Point]:
    return [point for point in trajectory if point_inside_constraint(point, constraint)]


def _near_or_far_detail(
    trajectory: Trajectory,
    constraint: Constraint,
    *,
    grid_size: int,
    grid_resolution: float,
) -> dict[str, object]:
    preference_type = str(constraint.get("preference_type"))
    reference_cells = _reference_cells(constraint, "reference_region")
    region_waypoints = _region_waypoints(trajectory, constraint)
    if not reference_cells:
        return {
            "status": "missing_object_footprint_cells",
            "raw_value": None,
            "region_waypoint_count": len(region_waypoints),
        }
    if region_waypoints:
        distance = sum(_distance_to_cells(point, reference_cells) for point in region_waypoints) / len(region_waypoints)
        status = "computed"
    elif preference_type == "near_preference":
        cells = _region_cells(constraint, grid_size)
        distance = max((_distance_to_cells(cell, reference_cells) for cell in cells), default=None)
        status = "empty_region_max_distance" if distance is not None else "empty_region_without_cells"
    else:
        distance = 0.0
        status = "empty_region_zero_distance"
    return {
        "status": status,
        "raw_value": distance * grid_resolution if distance is not None else None,
        "region_waypoint_count": len(region_waypoints),
        "direction": "smaller_is_better" if preference_type == "near_preference" else "larger_is_better",
        "unit": "m",
    }


def _relative_detail(
    trajectory: Trajectory,
    constraint: Constraint,
    *,
    grid_resolution: float,
) -> dict[str, object]:
    raw_regions = constraint.get("reference_regions")
    if not isinstance(raw_regions, Sequence) or isinstance(raw_regions, (str, bytes)) or len(raw_regions) != 2:
        return {"status": "missing_relative_object_footprints", "raw_value": None, "region_waypoint_count": 0}
    regions = [item for item in raw_regions if isinstance(item, Mapping)]
    if len(regions) != 2:
        return {"status": "missing_relative_object_footprints", "raw_value": None, "region_waypoint_count": 0}
    first_cells = _reference_cells({"reference_region": regions[0]}, "reference_region")
    second_cells = _reference_cells({"reference_region": regions[1]}, "reference_region")
    region_waypoints = _region_waypoints(trajectory, constraint)
    if not first_cells or not second_cells:
        return {"status": "missing_relative_object_footprints", "raw_value": None, "region_waypoint_count": len(region_waypoints)}
    if not region_waypoints:
        return {
            "status": "empty_region_zero_score",
            "raw_value": 0.0,
            "region_waypoint_count": 0,
            "direction": "larger_is_better",
            "unit": "ratio",
        }
    distance_a = sum(_distance_to_cells(point, first_cells) for point in region_waypoints) / len(region_waypoints)
    distance_b = sum(_distance_to_cells(point, second_cells) for point in region_waypoints) / len(region_waypoints)
    if distance_a <= EPSILON:
        return {
            "status": "zero_distance_to_preferred_object_pending_policy",
            "raw_value": None,
            "D_A": distance_a * grid_resolution,
            "D_B": distance_b * grid_resolution,
            "region_waypoint_count": len(region_waypoints),
            "direction": "larger_is_better",
            "distance_unit": "m",
        }
    return {
        "status": "computed",
        "raw_value": distance_b / distance_a,
        "D_A": distance_a * grid_resolution,
        "D_B": distance_b * grid_resolution,
        "region_waypoint_count": len(region_waypoints),
        "direction": "larger_is_better",
        "distance_unit": "m",
    }


def _geometric_path_detail(trajectory: Trajectory, constraint: Constraint) -> dict[str, object]:
    reference = constraint.get("reference_trajectory")
    reference_source = constraint.get("_path_shape_reference_source")
    if not isinstance(reference_source, str) or not reference_source:
        reference_source = "instruction_reference_trajectory"
    reference_file = constraint.get("_path_shape_reference_file")
    if not isinstance(reference, Sequence) or isinstance(reference, (str, bytes)):
        return {
            "status": "missing_reference_trajectory",
            "raw_value": None,
            "reference_trajectory_source": reference_source,
            "reference_waypoint_count": 0,
        }
    provenance = {
        "reference_trajectory_source": reference_source,
        "reference_waypoint_count": len(reference),
    }
    if isinstance(reference_file, str) and reference_file:
        provenance["reference_trajectory_file"] = reference_file
    local_trajectory = _region_waypoints(trajectory, constraint)
    if not local_trajectory:
        return {
            "status": "empty_region_zero_score",
            "raw_value": 0.0,
            "nDTW": 0.0,
            "region_waypoint_count": 0,
            "direction": "larger_is_better",
            **provenance,
        }
    score, dtw_cost, mean_distance, steps = compute_ndtw_score(local_trajectory, reference)  # type: ignore[arg-type]
    return {
        "status": "computed",
        "raw_value": score,
        "nDTW": score,
        "dtw_cost": dtw_cost,
        "mean_dtw_distance": mean_distance,
        "dtw_alignment_steps": steps,
        "region_waypoint_count": len(local_trajectory),
        "direction": "larger_is_better",
        **provenance,
    }


def _map_cache(map_state: Mapping[str, object]) -> Mapping[str, np.ndarray] | None:
    cached = map_state.get(_MAP_CACHE_KEY)
    if isinstance(cached, Mapping):
        traversable = cached.get("traversable")
        clearance = cached.get("clearance_distance")
        if isinstance(traversable, np.ndarray) and isinstance(clearance, np.ndarray):
            return cached  # type: ignore[return-value]
    loaded = load_map_metric_cache(map_state)
    if loaded is not None and isinstance(map_state, dict):
        map_state[_MAP_CACHE_KEY] = loaded
    return loaded


def _traversable_grid(map_state: Mapping[str, object]) -> TraversableGrid:
    cache = _map_cache(map_state)
    if cache is not None:
        return cache["traversable"]  # type: ignore[return-value]
    return build_traversable_grid(map_state)


def _clearance_detail(trajectory: Trajectory, map_state: Mapping[str, object]) -> dict[str, object]:
    points = [_point(point, "trajectory point") for point in trajectory]
    if not points:
        return {"status": "empty_trajectory", "raw_value": None, "waypoint_count": 0}
    cache = _map_cache(map_state)
    field = cache["clearance_distance"] if cache is not None else load_clearance_distance_field(map_state)
    # When a distance field is available, obstacle enumeration is unnecessary.
    # This avoids repeatedly scanning a large map's object-instance layer.
    obstacles = () if field is not None else clearance_obstacle_cells(map_state)
    if field is None and not obstacles:
        return {"status": "no_nontraversable_cells", "raw_value": None, "waypoint_count": len(points)}
    distances: list[float] = []
    for point in points:
        row, col = point
        row_index, col_index = int(round(row)), int(round(col))
        if (
            field is not None
            and math.isclose(row, row_index, abs_tol=EPSILON)
            and math.isclose(col, col_index, abs_tol=EPSILON)
            and 0 <= row_index < field.shape[0]
            and 0 <= col_index < field.shape[1]
        ):
            distances.append(float(field[row_index, col_index]))
            continue
        distances.append(min(math.hypot(row - obstacle_row, col - obstacle_col) for obstacle_row, obstacle_col, _object_id in obstacles))
    return {
        "status": "computed",
        "raw_value": (sum(distances) / len(distances)) * grid_resolution_m(map_state),
        "waypoint_count": len(points),
        "direction": "larger_is_better",
        "unit": "m",
    }


def compute_soft_details(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
    soft_constraints: Sequence[Constraint],
    map_state: Mapping[str, object],
) -> list[dict[str, object]]:
    must_pass = ordered_must_pass_constraints(hard_constraints)
    hard_hits = _first_ordered_waypoint_hits(trajectory, must_pass)
    completed_to_orders = _completed_segment_to_orders(
        trajectory,
        hard_constraints,
    )
    grid_size = int(map_state["grid_size"])
    resolution = grid_resolution_m(map_state)
    details: list[dict[str, object]] = []
    for index, constraint in enumerate(soft_constraints, start=1):
        preference_type = constraint.get("preference_type")
        candidate_active = _active_trajectory(trajectory, constraint, hard_hits)
        is_segment_scoped, segment_succeeded, required_to_order = _segment_gate(
            constraint,
            completed_to_orders,
        )
        active = candidate_active if segment_succeeded else []
        normalized_type = (
            "geometric_path_preference"
            if preference_type == "path_shape_preference"
            else preference_type
        )
        base: dict[str, object] = {
            "constraint_id": _constraint_id(constraint, f"soft_{index}"),
            "preference_type": normalized_type,
            "source_preference_type": preference_type,
            "scope": constraint.get("scope"),
            "active_waypoint_count": len(active),
            "score": None,
            "segment_gate_applied": is_segment_scoped,
            "segment_succeeded": segment_succeeded if is_segment_scoped else None,
        }
        if is_segment_scoped:
            base["required_to_order"] = required_to_order
            if not segment_succeeded:
                base["ungated_active_waypoint_count"] = len(candidate_active)
        if preference_type in {"near_preference", "far_preference"}:
            detail = _near_or_far_detail(
                active,
                constraint,
                grid_size=grid_size,
                grid_resolution=resolution,
            )
        elif preference_type == "relative_preference":
            detail = _relative_detail(
                active,
                constraint,
                grid_resolution=resolution,
            )
        elif preference_type in {"path_shape_preference", "geometric_path_preference"}:
            detail = _geometric_path_detail(active, constraint)
        elif preference_type == "clearance":
            detail = _clearance_detail(active, map_state)
        else:
            detail = {"status": "unsupported_preference", "raw_value": None}
        if is_segment_scoped and not segment_succeeded:
            ungated_status = detail.get("status")
            if preference_type == "clearance":
                detail = {
                    **detail,
                    "raw_value": 0.0,
                    "waypoint_count": 0,
                    "direction": "larger_is_better",
                    "unit": "m",
                }
            detail = {
                **detail,
                "status": "segment_not_completed_worst_case",
                "worst_case_measurement_status": ungated_status,
            }
        details.append({**base, **detail})
    return details


def worst_case_metrics_for_evaluation_error(
    trajectory: Trajectory,
    instruction: Mapping[str, object],
    map_state: Mapping[str, object],
    *,
    error: BaseException | str,
) -> dict[str, object]:
    """Build auditable worst-case metrics after trajectory evaluation raises."""

    raw_hard = instruction.get("hard_constraints", [])
    raw_soft = instruction.get("soft_constraints", [])
    hard = (
        [constraint for constraint in raw_hard if isinstance(constraint, Mapping)]
        if isinstance(raw_hard, Sequence) and not isinstance(raw_hard, (str, bytes))
        else []
    )
    soft = (
        [constraint for constraint in raw_soft if isinstance(constraint, Mapping)]
        if isinstance(raw_soft, Sequence) and not isinstance(raw_soft, (str, bytes))
        else []
    )
    grid_size = int(map_state["grid_size"])
    resolution = grid_resolution_m(map_state)
    error_text = (
        error
        if isinstance(error, str)
        else f"{type(error).__name__}: {error}"
    )

    soft_details: list[dict[str, object]] = []
    for index, constraint in enumerate(soft, start=1):
        preference_type = constraint.get("preference_type")
        normalized_type = (
            "geometric_path_preference"
            if preference_type == "path_shape_preference"
            else preference_type
        )
        base: dict[str, object] = {
            "constraint_id": _constraint_id(constraint, f"soft_{index}"),
            "preference_type": normalized_type,
            "source_preference_type": preference_type,
            "scope": constraint.get("scope"),
            "active_waypoint_count": 0,
            "score": None,
            "evaluation_fallback": "worst_case",
        }

        if preference_type == "near_preference":
            try:
                detail = _near_or_far_detail(
                    [],
                    constraint,
                    grid_size=grid_size,
                    grid_resolution=resolution,
                )
            except Exception as near_error:
                detail = {
                    "raw_value": None,
                    "region_waypoint_count": 0,
                    "fallback_error": (
                        f"{type(near_error).__name__}: {near_error}"
                    ),
                }
            if detail.get("raw_value") is None:
                detail = {
                    **detail,
                    "status": "evaluation_error_worst_case_map_diagonal",
                    "raw_value": (
                        math.sqrt(2.0)
                        * max(0, grid_size - 1)
                        * resolution
                    ),
                    "direction": "smaller_is_better",
                    "unit": "m",
                }
            else:
                detail = {
                    **detail,
                    "status": "evaluation_error_worst_case_max_distance",
                }
        elif preference_type == "far_preference":
            detail = {
                "status": "evaluation_error_worst_case_zero",
                "raw_value": 0.0,
                "region_waypoint_count": 0,
                "direction": "larger_is_better",
                "unit": "m",
            }
        elif preference_type == "relative_preference":
            detail = {
                "status": "evaluation_error_worst_case_zero",
                "raw_value": 0.0,
                "region_waypoint_count": 0,
                "direction": "larger_is_better",
                "unit": "ratio",
            }
        elif preference_type in {
            "path_shape_preference",
            "geometric_path_preference",
        }:
            detail = {
                "status": "evaluation_error_worst_case_zero",
                "raw_value": 0.0,
                "nDTW": 0.0,
                "region_waypoint_count": 0,
                "direction": "larger_is_better",
            }
        elif preference_type == "clearance":
            detail = {
                "status": "evaluation_error_worst_case_zero",
                "raw_value": 0.0,
                "waypoint_count": 0,
                "direction": "larger_is_better",
                "unit": "m",
            }
        else:
            detail = {
                "status": "evaluation_error_unsupported_preference",
                "raw_value": 0.0,
            }
        soft_details.append({**base, **detail})

    must_pass = ordered_must_pass_constraints(hard)
    evaluated_target_count = max(0, len(must_pass) - 1)
    try:
        path_length = compute_path_length(trajectory) * resolution
    except Exception:
        path_length = 0.0

    return {
        "HCS": 0.0,
        "SCS": None,
        "PL": path_length,
        "segment_wise_SPL": 0.0,
        "details": {
            "hcs": {
                "score": 0.0,
                "valid_completed_count": 0,
                "must_pass_count": evaluated_target_count,
                "ignored_record_must_pass": (
                    {
                        "constraint_id": must_pass[0].get("constraint_id"),
                        "order": must_pass[0].get("order"),
                    }
                    if must_pass
                    else None
                ),
                "failed": True,
                "first_failed_transition": (
                    1 if evaluated_target_count else None
                ),
                "failure_type": "metric_evaluation_error_worst_case",
                "first_must_avoid_violation": None,
                "must_avoid_violations": [],
                "global_must_avoid_violations": [],
            },
            "segments": [],
            "soft_constraints": soft_details,
            "scs_status": "evaluation_error_worst_case_raw_values",
            "segment_reference_status": "evaluation_error_worst_case_zero",
            "evaluation_fallback": {
                "policy": "worst_case",
                "error": error_text,
            },
            "distance_unit": "m",
            "grid_resolution_m": resolution,
        },
    }


def _path_violates_constraints(path: Trajectory, constraints: Sequence[Constraint]) -> bool:
    return any(_constraint_hit(path, constraint) for constraint in constraints)


def _shortest_legal_path_length(
    traversable: TraversableGrid,
    start: Cell,
    target: Constraint,
    avoids: Sequence[Constraint],
) -> float | None:
    """Return the shortest legal path from ``start`` to any target-region cell."""

    if not is_traversable(traversable, *start):
        return None
    if any(point_inside_constraint(start, avoid) for avoid in avoids):
        return None
    if point_inside_constraint(start, target):
        return 0.0
    directions = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    )
    queue: list[tuple[float, int, Cell]] = [(0.0, 0, start)]
    counter = 1
    costs = {start: 0.0}
    closed: set[Cell] = set()
    grid_size = len(traversable)
    while queue:
        _priority, _tie, current = heapq.heappop(queue)
        if current in closed:
            continue
        if point_inside_constraint(current, target):
            return costs[current]
        closed.add(current)
        for row_delta, col_delta, step_cost in directions:
            neighbor = (current[0] + row_delta, current[1] + col_delta)
            if (
                neighbor[0] < 0 or neighbor[0] >= grid_size
                or neighbor[1] < 0 or neighbor[1] >= grid_size
                or not can_traverse_between(traversable, current[0], current[1], neighbor[0], neighbor[1])
                or any(segment_intersects_constraint(current, neighbor, avoid) for avoid in avoids)
            ):
                continue
            candidate = costs[current] + step_cost
            if candidate >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = candidate
            heapq.heappush(queue, (candidate, counter, neighbor))
            counter += 1
    return None


_DIRECTIONS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
)


def _rectangle_intersects_edges(
    start_rows: np.ndarray,
    start_cols: np.ndarray,
    row_delta: int,
    col_delta: int,
    rectangle: Constraint,
) -> np.ndarray:
    center_row, center_col = _point(rectangle.get("center"), "rectangle center")
    width = float(rectangle.get("width", 0.0))
    height = float(rectangle.get("height", 0.0))
    row_min, row_max = center_row - height / 2, center_row + height / 2
    col_min, col_max = center_col - width / 2, center_col + width / 2

    def interval(start: np.ndarray, delta: int, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
        if delta == 0:
            inside = (start >= lower) & (start <= upper)
            return (
                np.where(inside, -np.inf, np.inf),
                np.where(inside, np.inf, -np.inf),
            )
        first = (lower - start) / float(delta)
        second = (upper - start) / float(delta)
        return np.minimum(first, second), np.maximum(first, second)

    row_enter, row_exit = interval(start_rows, row_delta, row_min, row_max)
    col_enter, col_exit = interval(start_cols, col_delta, col_min, col_max)
    enter = np.maximum(np.maximum(row_enter, col_enter), 0.0)
    exit_ = np.minimum(np.minimum(row_exit, col_exit), 1.0)
    return enter <= exit_ + EPSILON


def build_traversable_sparse_graph(
    traversable: TraversableGrid,
    *,
    rectangle_avoids: Sequence[Constraint] = (),
) -> csr_matrix:
    """Build the undirected 8-neighbour graph for a static traversability map.

    This is used only while precomputing instruction caches.  Building it once
    per map lets SciPy compute the many no-avoid segment fields in native code.
    """

    grid = np.asarray(traversable, dtype=np.bool_)
    grid_size = grid.shape[0]
    if grid.ndim != 2 or grid.shape[1] != grid_size:
        raise ValueError("Traversable grid must be square.")
    cell_ids = np.arange(grid_size * grid_size, dtype=np.int32).reshape(grid_size, grid_size)
    source_ids: list[np.ndarray] = []
    target_ids: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    def add_edges(
        valid: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
        source_rows: np.ndarray,
        source_cols: np.ndarray,
        row_delta: int,
        col_delta: int,
        weight: float,
    ) -> None:
        for rectangle in rectangle_avoids:
            valid &= ~_rectangle_intersects_edges(
                source_rows,
                source_cols,
                row_delta,
                col_delta,
                rectangle,
            )
        if not np.any(valid):
            return
        count = int(np.count_nonzero(valid))
        source_ids.append(source[valid])
        target_ids.append(target[valid])
        weights.append(np.full(count, weight, dtype=np.float64))

    row_coordinates, col_coordinates = np.indices((grid_size, grid_size), dtype=np.float64)
    add_edges(
        grid[:, :-1] & grid[:, 1:], cell_ids[:, :-1], cell_ids[:, 1:],
        row_coordinates[:, :-1], col_coordinates[:, :-1], 0, 1, 1.0,
    )
    add_edges(
        grid[:-1, :] & grid[1:, :], cell_ids[:-1, :], cell_ids[1:, :],
        row_coordinates[:-1, :], col_coordinates[:-1, :], 1, 0, 1.0,
    )
    add_edges(
        grid[:-1, :-1] & grid[1:, 1:] & grid[1:, :-1] & grid[:-1, 1:],
        cell_ids[:-1, :-1],
        cell_ids[1:, 1:],
        row_coordinates[:-1, :-1], col_coordinates[:-1, :-1], 1, 1,
        math.sqrt(2),
    )
    add_edges(
        grid[:-1, 1:] & grid[1:, :-1] & grid[1:, 1:] & grid[:-1, :-1],
        cell_ids[:-1, 1:],
        cell_ids[1:, :-1],
        row_coordinates[:-1, 1:], col_coordinates[:-1, 1:], 1, -1,
        math.sqrt(2),
    )
    if not source_ids:
        return csr_matrix((grid_size * grid_size, grid_size * grid_size), dtype=np.float64)
    return csr_matrix(
        (np.concatenate(weights), (np.concatenate(source_ids), np.concatenate(target_ids))),
        shape=(grid_size * grid_size, grid_size * grid_size),
        dtype=np.float64,
    )


def _freeform_avoid_mask(avoids: Sequence[Constraint], grid_size: int) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=np.bool_)
    for avoid in avoids:
        cells = avoid.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            continue
        for cell in cells:
            row, col = _point(cell, "must_avoid cell")
            row_index, col_index = int(round(row)), int(round(col))
            if 0 <= row_index < grid_size and 0 <= col_index < grid_size:
                mask[row_index, col_index] = True
    return mask


def _sparse_distance_field_to_target(
    traversable: TraversableGrid,
    target: Constraint,
    *,
    sparse_graph: csr_matrix,
) -> np.ndarray:
    grid_size = len(traversable)
    target_cells = _region_cells(target, grid_size)
    source_ids = np.asarray(
        [
            row * grid_size + col
            for row, col in target_cells
            if is_traversable(traversable, row, col)
        ],
        dtype=np.int32,
    )
    if source_ids.size == 0:
        return np.full((grid_size, grid_size), np.inf, dtype=np.float64)
    return np.asarray(
        sparse_dijkstra(sparse_graph, directed=False, indices=source_ids, min_only=True),
        dtype=np.float64,
    ).reshape((grid_size, grid_size))


def _segment_descriptor(
    index: int,
    from_order: object,
    to_order: object,
    target: Constraint,
    avoids: Sequence[Constraint],
) -> dict[str, object]:
    return {
        "index": index,
        "from_order": from_order,
        "to_order": to_order,
        "goal_region_constraint_id": _constraint_id(target, f"must_pass_{index + 1}"),
        "active_must_avoid_ids": [
            _constraint_id(avoid, f"must_avoid_{avoid_index}")
            for avoid_index, avoid in enumerate(avoids, start=1)
        ],
    }


def build_segment_distance_field(
    traversable: TraversableGrid,
    target: Constraint,
    avoids: Sequence[Constraint],
    *,
    sparse_graph: csr_matrix | None = None,
) -> np.ndarray:
    """Build reverse legal shortest-path distances to a target region.

    The resulting field is independent of a method's actual segment start, so
    it can be reused by every evaluated trajectory for this instruction.
    """

    grid_size = len(traversable)
    if not avoids:
        graph = sparse_graph if sparse_graph is not None else build_traversable_sparse_graph(traversable)
        return _sparse_distance_field_to_target(traversable, target, sparse_graph=graph)
    if all(avoid.get("shape") in {"freeform", "rectangle"} for avoid in avoids):
        # A freeform region is a union of unit grid-cell squares. Removing its
        # cells before graph construction exactly matches the evaluator's
        # endpoint and diagonal corner-intersection rules, without scanning all
        # freeform cells for every Dijkstra edge expansion.
        freeform_avoids = [avoid for avoid in avoids if avoid.get("shape") == "freeform"]
        rectangle_avoids = [avoid for avoid in avoids if avoid.get("shape") == "rectangle"]
        allowed = np.asarray(traversable, dtype=np.bool_) & ~_freeform_avoid_mask(freeform_avoids, grid_size)
        graph = build_traversable_sparse_graph(allowed, rectangle_avoids=rectangle_avoids)
        return _sparse_distance_field_to_target(allowed, target, sparse_graph=graph)

    distance = np.full((grid_size, grid_size), np.inf, dtype=np.float64)
    queue: list[tuple[float, int, Cell]] = []
    counter = 0
    for row in range(grid_size):
        for col in range(grid_size):
            cell = (row, col)
            if (
                is_traversable(traversable, row, col)
                and point_inside_constraint(cell, target)
                and not any(point_inside_constraint(cell, avoid) for avoid in avoids)
            ):
                distance[row, col] = 0.0
                heapq.heappush(queue, (0.0, counter, cell))
                counter += 1

    while queue:
        current_cost, _tie, current = heapq.heappop(queue)
        if current_cost != float(distance[current[0], current[1]]):
            continue
        for row_delta, col_delta, step_cost in _DIRECTIONS:
            neighbor = (current[0] + row_delta, current[1] + col_delta)
            if (
                neighbor[0] < 0
                or neighbor[0] >= grid_size
                or neighbor[1] < 0
                or neighbor[1] >= grid_size
                or not can_traverse_between(
                    traversable,
                    neighbor[0],
                    neighbor[1],
                    current[0],
                    current[1],
                )
                or any(segment_intersects_constraint(neighbor, current, avoid) for avoid in avoids)
            ):
                continue
            candidate = current_cost + step_cost
            if candidate >= float(distance[neighbor[0], neighbor[1]]):
                continue
            distance[neighbor[0], neighbor[1]] = candidate
            heapq.heappush(queue, (candidate, counter, neighbor))
            counter += 1
    return distance


def build_instruction_metric_cache(
    instruction_path: Path | str,
    instruction: Mapping[str, object],
    map_state: Mapping[str, object],
) -> dict[str, object]:
    """Precompute all source-independent SPL fields for one instruction."""

    hard_constraints = instruction.get("hard_constraints", [])
    if not isinstance(hard_constraints, Sequence) or isinstance(hard_constraints, (str, bytes)):
        raise ValueError("hard_constraints must be a list.")
    constraints = [constraint for constraint in hard_constraints if isinstance(constraint, Mapping)]
    must_pass = ordered_must_pass_constraints(constraints)
    traversable = _traversable_grid(map_state)
    sparse_graph: csr_matrix | None = None
    previous_goal: Constraint | None = must_pass[0] if must_pass else None
    segments: list[tuple[Mapping[str, object], np.ndarray]] = []
    for index, target in enumerate(must_pass[1:], start=1):
        raw_from_order = must_pass[index - 1].get("order", index)
        raw_to_order = target.get("order", index + 1)
        avoids = _active_avoids(constraints, previous_goal, target)
        descriptor = _segment_descriptor(index, raw_from_order, raw_to_order, target, avoids)
        if not avoids and sparse_graph is None:
            sparse_graph = build_traversable_sparse_graph(traversable)
        segments.append(
            (
                descriptor,
                build_segment_distance_field(
                    traversable,
                    target,
                    avoids,
                    sparse_graph=sparse_graph,
                ),
            )
        )
        previous_goal = target
    return write_instruction_metric_cache(instruction_path, map_state, segments)


def compute_segment_wise_spl(
    trajectory: Trajectory,
    instruction: Mapping[str, object],
    map_state: Mapping[str, object],
) -> tuple[float | None, list[dict[str, object]]]:
    hard_constraints = instruction.get("hard_constraints", [])
    if not isinstance(hard_constraints, Sequence) or isinstance(hard_constraints, (str, bytes)):
        raise ValueError("hard_constraints must be a list.")
    constraints = [constraint for constraint in hard_constraints if isinstance(constraint, Mapping)]
    must_pass = ordered_must_pass_constraints(constraints)
    if len(must_pass) < 2:
        raise ValueError("Segment-wise SPL requires a record region and at least one target.")
    traversable = _traversable_grid(map_state)
    resolution = grid_resolution_m(map_state)
    prediction = list(trajectory)
    cursor = 0
    start_index = 0
    previous_goal: Constraint | None = must_pass[0]
    complete = True
    details: list[dict[str, object]] = []
    scores: list[float] = []
    for index, target in enumerate(must_pass[1:], start=1):
        raw_from_order = must_pass[index - 1].get("order", index)
        raw_to_order = target.get("order", index + 1)
        avoids = _active_avoids(constraints, previous_goal, target)
        descriptor = _segment_descriptor(index, raw_from_order, raw_to_order, target, avoids)
        oracle_length = None
        segment_start: Cell | None = None
        if complete and start_index < len(prediction):
            row, col = _point(prediction[start_index], "trajectory segment start")
            segment_start = (int(round(row)), int(round(col)))
            cached_distance = load_instruction_segment_distance_field(
                instruction,
                map_state,
                descriptor,
            )
            if cached_distance is not None and 0 <= segment_start[0] < cached_distance.shape[0] and 0 <= segment_start[1] < cached_distance.shape[1]:
                cached_value = float(cached_distance[segment_start[0], segment_start[1]])
                oracle_length = cached_value if math.isfinite(cached_value) else None
            else:
                warn_missing_instruction_cache(instruction)
                oracle_length = _shortest_legal_path_length(
                    traversable,
                    segment_start,
                    target,
                    avoids,
                )
        hit_index: int | None = None
        if complete:
            for candidate_index in range(cursor, len(prediction)):
                if point_inside_constraint(prediction[candidate_index], target):
                    hit_index = candidate_index
                    break
        actual_segment = prediction[start_index : hit_index + 1] if hit_index is not None else []
        actual_length = compute_path_length(actual_segment) if actual_segment else None
        violation = bool(actual_segment) and _path_violates_constraints(actual_segment, avoids)
        success = hit_index is not None and not violation and oracle_length is not None
        if success:
            denominator = max(float(oracle_length), float(actual_length or 0.0))
            score = 1.0 if denominator <= EPSILON else float(oracle_length) / denominator
            status = "completed"
            cursor = hit_index
            start_index = hit_index
        else:
            score = 0.0
            complete = False
            if oracle_length is None:
                status = "missing_or_invalid_oracle_segment"
            elif hit_index is None:
                status = "target_not_completed"
            else:
                status = "active_must_avoid_violation"
        scores.append(score)
        details.append(
            {
                "index": index,
                "from_order": raw_from_order,
                "to_order": raw_to_order,
                "start_anchor": list(segment_start) if segment_start is not None else None,
                "goal_region_constraint_id": descriptor["goal_region_constraint_id"],
                "oracle_definition": "precomputed_reverse_shortest_legal_distance_field_from_actual_segment_start_to_any_target_region_cell",
                "active_must_avoid_ids": descriptor["active_must_avoid_ids"],
                "L_star": oracle_length * resolution if oracle_length is not None else None,
                "L": actual_length * resolution if actual_length is not None else None,
                "length_unit": "m",
                "S": int(success),
                "SPL": score,
                "status": status,
            }
        )
        previous_goal = target
    return (sum(scores) / len(scores) if scores else None), details


def evaluate_trajectory(
    trajectory: Trajectory,
    instruction: Mapping[str, object],
    map_state: Mapping[str, object],
) -> dict[str, object]:
    hard_constraints = instruction.get("hard_constraints", [])
    soft_constraints = instruction.get("soft_constraints", [])
    if not isinstance(hard_constraints, Sequence) or isinstance(hard_constraints, (str, bytes)):
        raise ValueError("hard_constraints must be a list.")
    if not isinstance(soft_constraints, Sequence) or isinstance(soft_constraints, (str, bytes)):
        raise ValueError("soft_constraints must be a list.")
    hard = [constraint for constraint in hard_constraints if isinstance(constraint, Mapping)]
    soft = [constraint for constraint in soft_constraints if isinstance(constraint, Mapping)]
    if _map_cache(map_state) is None:
        warn_missing_map_cache(map_state)
    resolution = grid_resolution_m(map_state)
    hcs, hcs_details = compute_hcs(trajectory, hard)
    segment_spl, segment_details = compute_segment_wise_spl(trajectory, instruction, map_state)
    soft_details = compute_soft_details(trajectory, hard, soft, map_state)
    return {
        "HCS": hcs,
        "SCS": None,
        "PL": compute_path_length(trajectory) * resolution,
        "segment_wise_SPL": segment_spl,
        "details": {
            "hcs": hcs_details,
            "segments": segment_details,
            "soft_constraints": soft_details,
            "scs_status": "pending_score_normalization",
            "segment_reference_status": "computed_on_demand_from_actual_segment_starts_to_target_regions",
            "distance_unit": "m",
            "grid_resolution_m": resolution,
        },
    }
