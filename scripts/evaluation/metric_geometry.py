#!/usr/bin/env python3
"""Core trajectory evaluation metrics for SemPathBench."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.hyparameter import (
    CLEARANCE_DISTANCE_CACHE_SUFFIX,
    CLEARANCE_DISTANCE_CACHE_VERSION,
    CLEARANCE_DISTANCE_FIELD_KEY,
    CLEARANCE_EXCLUDED_DISTANCE_FIELDS_KEY,
    EVALUATION_EPSILON,
    PATH_SHAPE_NDTW_SUCCESS_DISTANCE,
    PATH_SHAPE_TOLERANCE_GRID,
    PLR_MIN_EFFECTIVE_PATH_LENGTH_GRID,
    RELATIVE_PREFERENCE_HUMAN_BETA,
    RELATIVE_PREFERENCE_MIN_MARGIN,
    RELATIVE_PREFERENCE_MIN_VALID_LENGTH_GRID,
    TRAVERSABLE_OBJECT_CATEGORIES,
    move_smoothness_enabled,
    score_larger_is_better,
    score_smaller_is_better,
    scs_config_details,
)


Point = Sequence[Real]
Trajectory = Sequence[Point]
Constraint = Mapping[str, object]
EPSILON = EVALUATION_EPSILON
REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_ROOT = REPO_ROOT / "resources" / "maps"
GLOBAL_SCOPE = {"type": "global", "from_order": None, "to_order": None}
GLOBAL_CLEARANCE_CONSTRAINT_ID = "__global_clearance__"
REFERENCE_UPDATE_COMMAND = (
    "python scripts/evaluation/update_reference_metrics.py --set all"
)


def _point(point: object, label: str = "point") -> tuple[float, float]:
    if (
        not isinstance(point, Sequence)
        or isinstance(point, (str, bytes))
        or len(point) != 2
        or not all(isinstance(value, Real) and not isinstance(value, bool) for value in point)
    ):
        raise ValueError(f"{label} must be [row, col].")
    return float(point[0]), float(point[1])


def compute_path_length(trajectory: Trajectory) -> float:
    """Return the Euclidean length between consecutive grid points."""

    points = [_point(point, "trajectory point") for point in trajectory]
    return math.fsum(
        math.hypot(row - previous_row, col - previous_col)
        for (previous_row, previous_col), (row, col) in zip(points, points[1:])
    )


def compute_path_length_ratio(
    prediction_trajectory: Trajectory,
    reference_trajectory: Trajectory | None,
) -> float | None:
    """Return capped GT/prediction path-length ratio, or None without GT."""

    if reference_trajectory is None:
        return None

    prediction_length = effective_plr_path_length(prediction_trajectory)
    reference_length = effective_plr_path_length(reference_trajectory)
    if prediction_length <= EPSILON:
        return 1.0 if reference_length <= EPSILON else 0.0
    return min(reference_length / prediction_length, 1.0)


def effective_plr_path_length(trajectory: Trajectory) -> float:
    """Return path length with a one-grid floor for non-empty trajectories."""

    raw_length = compute_path_length(trajectory)
    return max(raw_length, float(PLR_MIN_EFFECTIVE_PATH_LENGTH_GRID)) if trajectory else raw_length


def compute_mean_squared_turning_angle(trajectory: Trajectory) -> tuple[float, int]:
    """Return mean squared heading-change angle and valid turn count."""

    points = [_point(point, "trajectory point") for point in trajectory]
    if len(points) < 3:
        return 0.0, 0

    squared_angles: list[float] = []
    for previous, current, following in zip(points, points[1:], points[2:]):
        first_vector = (
            current[0] - previous[0],
            current[1] - previous[1],
        )
        second_vector = (
            following[0] - current[0],
            following[1] - current[1],
        )
        first_norm = math.hypot(first_vector[0], first_vector[1])
        second_norm = math.hypot(second_vector[0], second_vector[1])
        if first_norm == 0 or second_norm == 0:
            continue
        cosine = (
            first_vector[0] * second_vector[0]
            + first_vector[1] * second_vector[1]
        ) / (first_norm * second_norm)
        angle = math.acos(max(-1.0, min(1.0, cosine)))
        squared_angles.append(angle * angle)

    if not squared_angles:
        return 0.0, 0
    return math.fsum(squared_angles) / len(squared_angles), len(squared_angles)


def _map_free_occupancy_value(map_state: Mapping[str, object]) -> int:
    legends = map_state.get("layer_legends")
    if isinstance(legends, Mapping):
        occupancy_legends = legends.get("occupancy")
        if isinstance(occupancy_legends, Mapping):
            free_tile = occupancy_legends.get("free")
            if isinstance(free_tile, Mapping):
                value = free_tile.get("value")
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
    return 0


def _object_categories_by_id(map_state: Mapping[str, object]) -> dict[int, str]:
    object_instances = map_state.get("object_instances", [])
    if not isinstance(object_instances, Sequence) or isinstance(
        object_instances, (str, bytes)
    ):
        return {}
    categories: dict[int, str] = {}
    for instance in object_instances:
        if not isinstance(instance, Mapping):
            continue
        object_id = instance.get("id")
        category = instance.get("category")
        if isinstance(object_id, int) and not isinstance(object_id, bool) and isinstance(category, str):
            categories[object_id] = category
    return categories


def clearance_obstacle_cells(
    map_state: Mapping[str, object],
) -> list[tuple[int, int, int]]:
    """Return clearance obstacle source cells as (row, col, object_id)."""

    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    occupancy = layers.get("occupancy")
    object_grid = layers.get("object_instance")
    if (
        not isinstance(occupancy, Sequence)
        or isinstance(occupancy, (str, bytes))
        or not isinstance(object_grid, Sequence)
        or isinstance(object_grid, (str, bytes))
    ):
        return []

    free_value = _map_free_occupancy_value(map_state)
    object_categories = _object_categories_by_id(map_state)
    obstacles: list[tuple[int, int, int]] = []
    for row_index, row in enumerate(occupancy):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            continue
        object_row = object_grid[row_index] if row_index < len(object_grid) else None
        for col_index, value in enumerate(row):
            try:
                occupancy_value = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                occupancy_value = free_value + 1
            object_id = 0
            if (
                isinstance(object_row, Sequence)
                and not isinstance(object_row, (str, bytes))
                and col_index < len(object_row)
            ):
                try:
                    object_id = int(object_row[col_index])  # type: ignore[index]
                except (TypeError, ValueError):
                    object_id = 0
            object_category = object_categories.get(object_id)
            object_is_obstacle = (
                object_id != 0 and object_category not in TRAVERSABLE_OBJECT_CATEGORIES
            )
            if occupancy_value != free_value or object_is_obstacle:
                obstacles.append((row_index, col_index, object_id))
    return obstacles


def point_inside_constraint(point: Point, constraint: Constraint) -> bool:
    """Match the circle, rectangle, and freeform semantics used by the annotator."""

    row, col = _point(point)
    shape = constraint.get("shape")

    if shape == "freeform":
        cells = constraint.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            raise ValueError("freeform constraint cells must be a list.")
        return any(_point(cell, "constraint cell") == (row, col) for cell in cells)

    center_row, center_col = _point(constraint.get("center"), "constraint center")
    delta_row = row - center_row
    delta_col = col - center_col

    if shape == "circle":
        radius = constraint.get("radius")
        if not isinstance(radius, Real) or isinstance(radius, bool) or radius <= 0:
            raise ValueError("circle constraint radius must be positive.")
        return math.hypot(delta_row, delta_col) <= float(radius)

    if shape == "rectangle":
        width = constraint.get("width")
        height = constraint.get("height")
        if (
            not isinstance(width, Real)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, Real)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ValueError("rectangle constraint width and height must be positive.")
        return (
            abs(delta_row) <= float(height) / 2
            and abs(delta_col) <= float(width) / 2
        )

    raise ValueError(f"Unknown constraint shape: {shape!r}")


def near_preference_excluded_object_ids(
    point: Point,
    soft_constraints: Sequence[Constraint],
) -> set[int]:
    """Objects ignored for clearance when point is inside a near-preference region."""

    excluded: set[int] = set()
    for constraint in soft_constraints:
        if constraint.get("preference_type") != "near_preference":
            continue
        reference_region = constraint.get("reference_region")
        if not isinstance(reference_region, Mapping):
            continue
        object_id = reference_region.get("object_id")
        if (
            reference_region.get("mode") == "object"
            and isinstance(object_id, int)
            and not isinstance(object_id, bool)
            and point_inside_constraint(point, constraint)
        ):
            excluded.add(object_id)
    return excluded


def clearance_distance_cache_path(map_state: Mapping[str, object]) -> Path | None:
    raw_map_path = map_state.get("map_path")
    if isinstance(raw_map_path, str) and raw_map_path.strip():
        map_path = Path(raw_map_path)
        if not map_path.is_absolute():
            map_path = REPO_ROOT / map_path
        return map_path.with_name(
            f"{map_path.stem}{CLEARANCE_DISTANCE_CACHE_SUFFIX}"
        )

    metadata = map_state.get("metadata")
    raw_map_id = metadata.get("map_id") if isinstance(metadata, Mapping) else None
    if not isinstance(raw_map_id, str) or not raw_map_id.strip():
        raw_map_id = map_state.get("map_key")
    map_json_path = _resolve_map_json_path(raw_map_id)
    if map_json_path is None:
        return None
    return map_json_path.with_name(
        f"{map_json_path.stem}{CLEARANCE_DISTANCE_CACHE_SUFFIX}"
    )


def load_clearance_distance_field(
    map_state: Mapping[str, object],
) -> np.ndarray | None:
    cached = map_state.get(CLEARANCE_DISTANCE_FIELD_KEY)
    if isinstance(cached, np.ndarray):
        return cached

    cache_path = clearance_distance_cache_path(map_state)
    if cache_path is None or not cache_path.exists():
        return None

    with np.load(cache_path) as payload:
        version = int(payload["version"]) if "version" in payload else 0
        if version != CLEARANCE_DISTANCE_CACHE_VERSION:
            return None
        distance = payload["distance"].astype(np.float64, copy=False)

    if isinstance(map_state, dict):
        map_state[CLEARANCE_DISTANCE_FIELD_KEY] = distance
    return distance


def clearance_occupancy_shape(map_state: Mapping[str, object]) -> tuple[int, int] | None:
    layers = map_state.get("layers")
    occupancy = layers.get("occupancy") if isinstance(layers, Mapping) else None
    if not isinstance(occupancy, Sequence) or isinstance(occupancy, (str, bytes)):
        return None
    row_count = len(occupancy)
    if row_count == 0:
        return None
    col_count = max(
        len(row)
        for row in occupancy
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
    )
    return row_count, col_count


def clearance_distance_field_from_obstacles(
    map_state: Mapping[str, object],
    obstacles: Sequence[tuple[int, int, int]],
    excluded_ids: frozenset[int] = frozenset(),
) -> np.ndarray | None:
    shape = clearance_occupancy_shape(map_state)
    if shape is None:
        return None

    obstacle_mask = np.zeros(shape, dtype=bool)
    for row, col, object_id in obstacles:
        if object_id in excluded_ids:
            continue
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            obstacle_mask[row, col] = True

    if not obstacle_mask.any():
        return np.full(shape, np.inf, dtype=np.float64)
    return distance_transform_edt(~obstacle_mask)


def clearance_distance_field_for_excluded_ids(
    map_state: Mapping[str, object],
    obstacles: Sequence[tuple[int, int, int]],
    excluded_ids: set[int],
) -> np.ndarray | None:
    key = frozenset(excluded_ids)
    if not key:
        return load_clearance_distance_field(map_state)

    cache = map_state.get(CLEARANCE_EXCLUDED_DISTANCE_FIELDS_KEY)
    if not isinstance(cache, dict):
        cache = {}
        if isinstance(map_state, dict):
            map_state[CLEARANCE_EXCLUDED_DISTANCE_FIELDS_KEY] = cache

    cached = cache.get(key)
    if isinstance(cached, np.ndarray):
        return cached

    distance_field = clearance_distance_field_from_obstacles(
        map_state,
        obstacles,
        key,
    )
    if distance_field is not None:
        cache[key] = distance_field
    return distance_field


def clearance_cost_from_distance(distance: float, epsilon: float) -> float:
    if math.isinf(distance):
        return 0.0
    return 1.0 / ((distance + epsilon) ** 2)


def clearance_cost_from_obstacles(
    point: tuple[float, float],
    obstacles: Sequence[tuple[int, int, int]],
    excluded_ids: set[int],
    epsilon: float,
) -> float:
    row, col = point
    min_distance = math.inf
    for obstacle_row, obstacle_col, object_id in obstacles:
        if object_id in excluded_ids:
            continue
        distance = math.hypot(row - obstacle_row, col - obstacle_col)
        if distance < min_distance:
            min_distance = distance
    return clearance_cost_from_distance(min_distance, epsilon)


def clearance_cost_from_distance_field(
    point: tuple[float, float],
    distance_field: np.ndarray,
    epsilon: float,
) -> float | None:
    row, col = point
    row_index = int(round(row))
    col_index = int(round(col))
    if (
        not math.isclose(row, row_index, rel_tol=0.0, abs_tol=EPSILON)
        or not math.isclose(col, col_index, rel_tol=0.0, abs_tol=EPSILON)
        or row_index < 0
        or col_index < 0
        or row_index >= distance_field.shape[0]
        or col_index >= distance_field.shape[1]
    ):
        return None
    return clearance_cost_from_distance(
        float(distance_field[row_index, col_index]),
        epsilon,
    )


def compute_clearance_cost(
    trajectory: Trajectory,
    map_state: Mapping[str, object],
    soft_constraints: Sequence[Constraint] = (),
    *,
    epsilon: float = EPSILON,
) -> tuple[float, int]:
    """Return mean inverse-square distance to non-traversable cells."""

    if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    points = [_point(point, "trajectory point") for point in trajectory]
    if not points:
        return 0.0, 0

    obstacles = clearance_obstacle_cells(map_state)
    if not obstacles:
        return 0.0, len(points)

    distance_field = load_clearance_distance_field(map_state)
    costs: list[float] = []
    for point in points:
        excluded_ids = near_preference_excluded_object_ids(point, soft_constraints)
        active_distance_field = distance_field
        if excluded_ids:
            active_distance_field = clearance_distance_field_for_excluded_ids(
                map_state,
                obstacles,
                excluded_ids,
            )
        cached_cost = None
        if active_distance_field is not None:
            cached_cost = clearance_cost_from_distance_field(
                point,
                active_distance_field,
                float(epsilon),
            )
        if cached_cost is not None:
            costs.append(cached_cost)
            continue
        costs.append(
            clearance_cost_from_obstacles(
                point,
                obstacles,
                excluded_ids,
                float(epsilon),
            )
        )
    return math.fsum(costs) / len(costs), len(costs)


def compute_clearance_score(
    trajectory: Trajectory,
    constraint: Constraint,
    map_state: Mapping[str, object],
    soft_constraints: Sequence[Constraint] = (),
    *,
    epsilon: float = EPSILON,
) -> tuple[float, float, int]:
    """Score clearance against the human reference clearance cost."""

    c_pred, waypoint_count = compute_clearance_cost(
        trajectory,
        map_state,
        soft_constraints,
        epsilon=epsilon,
    )
    c_ref = constraint.get("C_clear_ref")
    if c_ref is None:
        if constraint.get("reference_metric_status") != "invalid_human_reference":
            warn_missing_reference_metric(constraint, "C_clear_ref")
        return 1.0, c_pred, waypoint_count
    if not isinstance(c_ref, Real) or isinstance(c_ref, bool) or c_ref < 0:
        raise ValueError("clearance requires a non-negative C_clear_ref.")
    score, _ratio = score_smaller_is_better(c_pred, float(c_ref))
    return score, c_pred, waypoint_count


def warn_missing_reference_metric(
    constraint: Constraint,
    field_name: str,
    *,
    preference_type: object | None = None,
) -> None:
    resolved_type = preference_type or constraint.get("preference_type")
    constraint_id = constraint.get("constraint_id")
    warnings.warn(
        f"Missing GT reference metric {field_name!r} for "
        f"{resolved_type!r} constraint {constraint_id!r}; computing it on the "
        f"fly. To persist references, run: {REFERENCE_UPDATE_COMMAND}",
        RuntimeWarning,
        stacklevel=3,
    )


def default_global_clearance_constraint() -> dict[str, object]:
    return {
        "constraint_id": GLOBAL_CLEARANCE_CONSTRAINT_ID,
        "preference_type": "clearance",
        "label": "Clearance",
        "scope": dict(GLOBAL_SCOPE),
        "C_clear_ref": None,
        "C_smooth_ref": None,
        "D_ref": None,
        "reference_trajectory": [],
        "reference_waypoint_count": 0,
    }


def normalize_soft_constraints_for_evaluation(
    soft_constraints: Sequence[Constraint],
    *,
    hard_constraints: Sequence[Constraint] = (),
    map_state: Mapping[str, object] | None = None,
    reference_trajectory: Trajectory | None = None,
) -> list[dict[str, object]]:
    """Ensure every episode has a global clearance soft score."""

    normalized = [
        dict(constraint)
        for constraint in soft_constraints
        if isinstance(constraint, Mapping)
    ]
    if not any(
        constraint.get("preference_type") == "clearance"
        for constraint in normalized
    ):
        normalized.append(default_global_clearance_constraint())

    for constraint in normalized:
        if constraint.get("preference_type") != "clearance":
            continue
        constraint["scope"] = dict(GLOBAL_SCOPE)
        if (
            constraint.get("C_clear_ref") is None
            and map_state is not None
            and reference_trajectory is not None
            and constraint.get("reference_metric_status") != "invalid_human_reference"
        ):
            warn_missing_reference_metric(constraint, "C_clear_ref")
            c_clear_ref, waypoint_count = compute_clearance_cost(
                reference_trajectory,
                map_state,
                normalized,
            )
            constraint["C_clear_ref"] = c_clear_ref
            constraint["reference_waypoint_count"] = waypoint_count
            constraint["reference_metric_status"] = "computed"
        constraint["C_smooth_ref"] = None
        constraint["D_ref"] = None
        constraint["reference_trajectory"] = []
    if reference_trajectory is None:
        return normalized

    reference_hit_indexes = ordered_must_pass_hit_indexes(
        reference_trajectory,
        hard_constraints,
    )
    for constraint in normalized:
        preference_type = constraint.get("preference_type")
        active_reference_trajectory = scoped_trajectory_for_constraint(
            reference_trajectory,
            constraint,
            reference_hit_indexes,
        )

        if preference_type in {"near_preference", "far_preference"}:
            if constraint.get("D_ref") is not None:
                continue
            if constraint.get("reference_metric_status") == "invalid_human_reference":
                continue
            reference_region = constraint.get("reference_region")
            if not isinstance(reference_region, Mapping):
                continue
            reference_points = reference_region.get("cells")
            if not isinstance(reference_points, Sequence) or isinstance(
                reference_points,
                (str, bytes),
            ):
                continue
            warn_missing_reference_metric(constraint, "D_ref")
            d_ref = mean_region_distance_to_reference(
                active_reference_trajectory,
                constraint,
                reference_points,  # type: ignore[arg-type]
            )
            constraint["D_ref"] = d_ref
            constraint["reference_trajectory"] = []
            constraint["reference_waypoint_count"] = len(
                trajectory_inside_constraint_region(
                    active_reference_trajectory,
                    constraint,
                )
            )
            constraint["reference_metric_status"] = (
                "computed" if d_ref is not None else "invalid_human_reference"
            )
            continue

        if preference_type == "move_smoothness" and move_smoothness_enabled():
            if constraint.get("C_smooth_ref") is not None:
                continue
            if constraint.get("reference_metric_status") == "invalid_human_reference":
                continue
            warn_missing_reference_metric(constraint, "C_smooth_ref")
            c_smooth_ref, waypoint_count = compute_mean_squared_turning_angle(
                active_reference_trajectory,
            )
            constraint["C_smooth_ref"] = c_smooth_ref
            constraint["C_clear_ref"] = None
            constraint["D_ref"] = None
            constraint["reference_trajectory"] = []
            constraint["reference_waypoint_count"] = waypoint_count
            constraint["reference_metric_status"] = "computed"
            continue

        if preference_type == "path_shape_preference":
            if (
                "reference_trajectory" in constraint
                and constraint.get("reference_trajectory") is not None
            ):
                continue
            if constraint.get("reference_metric_status") == "invalid_human_reference":
                continue
            warn_missing_reference_metric(constraint, "reference_trajectory")
            reference_region_trajectory = trajectory_inside_constraint_region(
                active_reference_trajectory,
                constraint,
            )
            constraint["reference_trajectory"] = reference_region_trajectory
            constraint["C_smooth_ref"] = None
            constraint["C_clear_ref"] = None
            constraint["D_ref"] = None
            constraint["reference_waypoint_count"] = len(reference_region_trajectory)
            constraint["reference_metric_status"] = "computed"
            continue

        if preference_type != "relative_preference":
            continue
        has_stored_reference = constraint.get("Q_relative_ref") is not None or (
            "Q_relative_ref" in constraint
            and constraint.get("relative_reference_status")
            in {"computed", "invalid_human_reference"}
        ) or (
            "Q_relative_ref" in constraint
            and constraint.get("reference_metric_status")
            == "invalid_human_reference"
        )
        if has_stored_reference:
            continue
        warn_missing_reference_metric(constraint, "Q_relative_ref")
        q_ref, q_details = compute_relative_preference_quality(
            active_reference_trajectory,
            constraint,
        )
        constraint["Q_relative_ref"] = q_ref
        constraint["reference_waypoint_count"] = q_details.get(
            "valid_segment_count",
            0,
        )
        constraint["reference_valid_path_length"] = q_details.get(
            "valid_path_length",
            0.0,
        )
        constraint["relative_reference_status"] = (
            "computed" if q_ref is not None else "invalid_human_reference"
        )
        constraint["reference_metric_status"] = constraint["relative_reference_status"]
        constraint["D_ref"] = None
    return normalized


def ordered_must_pass_constraints(
    hard_constraints: Sequence[Constraint],
) -> list[Constraint]:
    """Return must-pass constraints in their annotated order."""

    indexed_constraints = [
        (index, constraint)
        for index, constraint in enumerate(hard_constraints)
        if constraint.get("kind") == "must_pass"
    ]

    def order_key(item: tuple[int, Constraint]) -> tuple[float, int]:
        index, constraint = item
        order = constraint.get("order")
        if isinstance(order, Real) and not isinstance(order, bool):
            return float(order), index
        return float(index + 1), index

    return [constraint for _, constraint in sorted(indexed_constraints, key=order_key)]


def _segment_rectangle_interval(
    start: tuple[float, float],
    end: tuple[float, float],
    row_min: float,
    row_max: float,
    col_min: float,
    col_max: float,
) -> tuple[float, float] | None:
    t_min = 0.0
    t_max = 1.0
    for start_value, delta, low, high in (
        (start[0], end[0] - start[0], row_min, row_max),
        (start[1], end[1] - start[1], col_min, col_max),
    ):
        if abs(delta) <= EPSILON:
            if start_value < low or start_value > high:
                return None
            continue
        first = (low - start_value) / delta
        second = (high - start_value) / delta
        enter = min(first, second)
        exit_ = max(first, second)
        t_min = max(t_min, enter)
        t_max = min(t_max, exit_)
        if t_min - t_max > EPSILON:
            return None
    return max(0.0, t_min), min(1.0, t_max)


def _segment_circle_interval(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float] | None:
    delta_row = end[0] - start[0]
    delta_col = end[1] - start[1]
    start_row = start[0] - center[0]
    start_col = start[1] - center[1]
    a = delta_row * delta_row + delta_col * delta_col
    c = start_row * start_row + start_col * start_col - radius * radius
    if a <= EPSILON:
        return (0.0, 1.0) if c <= EPSILON else None
    b = 2 * (start_row * delta_row + start_col * delta_col)
    discriminant = b * b - 4 * a * c
    if discriminant < -EPSILON:
        return None
    if discriminant < 0:
        discriminant = 0.0
    root = math.sqrt(discriminant)
    first = (-b - root) / (2 * a)
    second = (-b + root) / (2 * a)
    enter = max(0.0, min(first, second))
    exit_ = min(1.0, max(first, second))
    if enter - exit_ > EPSILON or exit_ < -EPSILON or enter > 1 + EPSILON:
        return None
    return enter, exit_


def segment_constraint_interval(
    start: Point,
    end: Point,
    constraint: Constraint,
) -> tuple[float, float] | None:
    """Return the path-parameter interval where a segment intersects a region."""

    parsed_start = _point(start, "segment start")
    parsed_end = _point(end, "segment end")
    shape = constraint.get("shape")

    if shape == "freeform":
        cells = constraint.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            raise ValueError("freeform constraint cells must be a list.")
        intervals: list[tuple[float, float]] = []
        for cell in cells:
            row, col = _point(cell, "constraint cell")
            interval = _segment_rectangle_interval(
                parsed_start,
                parsed_end,
                row - 0.5,
                row + 0.5,
                col - 0.5,
                col + 0.5,
            )
            if interval is not None:
                intervals.append(interval)
        return min(intervals, key=lambda item: item[0]) if intervals else None

    center = _point(constraint.get("center"), "constraint center")
    if shape == "circle":
        radius = constraint.get("radius")
        if not isinstance(radius, Real) or isinstance(radius, bool) or radius <= 0:
            raise ValueError("circle constraint radius must be positive.")
        return _segment_circle_interval(parsed_start, parsed_end, center, float(radius))

    if shape == "rectangle":
        width = constraint.get("width")
        height = constraint.get("height")
        if (
            not isinstance(width, Real)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, Real)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ValueError("rectangle constraint width and height must be positive.")
        return _segment_rectangle_interval(
            parsed_start,
            parsed_end,
            center[0] - float(height) / 2,
            center[0] + float(height) / 2,
            center[1] - float(width) / 2,
            center[1] + float(width) / 2,
        )

    raise ValueError(f"Unknown constraint shape: {shape!r}")


def _merge_segment_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    sorted_intervals = sorted(
        (
            (max(0.0, float(start)), min(1.0, float(end)))
            for start, end in intervals
            if end >= start - EPSILON
        ),
        key=lambda item: item[0],
    )
    merged: list[tuple[float, float]] = []
    for start, end in sorted_intervals:
        if not merged or start > merged[-1][1] + EPSILON:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def segment_constraint_intervals(
    start: Point,
    end: Point,
    constraint: Constraint,
) -> list[tuple[float, float]]:
    """Return all path-parameter intervals where a segment intersects a region."""

    parsed_start = _point(start, "segment start")
    parsed_end = _point(end, "segment end")
    shape = constraint.get("shape")

    if shape == "freeform":
        cells = constraint.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            raise ValueError("freeform constraint cells must be a list.")
        intervals: list[tuple[float, float]] = []
        for cell in cells:
            row, col = _point(cell, "constraint cell")
            interval = _segment_rectangle_interval(
                parsed_start,
                parsed_end,
                row - 0.5,
                row + 0.5,
                col - 0.5,
                col + 0.5,
            )
            if interval is not None:
                intervals.append(interval)
        return _merge_segment_intervals(intervals)

    interval = segment_constraint_interval(parsed_start, parsed_end, constraint)
    return [] if interval is None else [interval]


def segment_intersects_constraint(
    start: Point,
    end: Point,
    constraint: Constraint,
) -> bool:
    return segment_constraint_interval(start, end, constraint) is not None


def _hard_constraint_identifier(constraint: Constraint, fallback: str) -> str:
    constraint_id = constraint.get("constraint_id")
    return str(constraint_id) if constraint_id is not None else fallback


def _must_avoid_constraints(
    hard_constraints: Sequence[Constraint],
) -> list[Constraint]:
    return [
        constraint
        for constraint in hard_constraints
        if constraint.get("kind") == "must_avoid"
    ]


def _constraint_scope_is_active_for_transition(
    constraint: Constraint,
    previous_goal: Constraint | None,
    expected_goal: Constraint,
) -> bool:
    scope = constraint.get("scope")
    if not isinstance(scope, Mapping):
        return True
    scope_type = scope.get("type")
    if scope_type == "global":
        return True
    if scope_type != "between_hard_constraints":
        return True

    from_order = scope.get("from_order")
    to_order = scope.get("to_order")
    expected_order = expected_goal.get("order")
    previous_order = None if previous_goal is None else previous_goal.get("order")
    if isinstance(to_order, Real) and not isinstance(to_order, bool):
        if not (
            isinstance(expected_order, Real)
            and not isinstance(expected_order, bool)
            and float(to_order) == float(expected_order)
        ):
            return False
    if from_order is None:
        return previous_goal is None
    if isinstance(from_order, Real) and not isinstance(from_order, bool):
        return (
            isinstance(previous_order, Real)
            and not isinstance(previous_order, bool)
            and float(from_order) == float(previous_order)
        )
    return True


def _active_must_avoid_constraints(
    hard_constraints: Sequence[Constraint],
    must_pass: Sequence[Constraint],
    next_goal_index: int,
) -> list[Constraint]:
    previous_goal = must_pass[next_goal_index - 1] if next_goal_index > 0 else None
    expected_goal = must_pass[next_goal_index]
    return [
        constraint
        for constraint in _must_avoid_constraints(hard_constraints)
        if _constraint_scope_is_active_for_transition(
            constraint,
            previous_goal,
            expected_goal,
        )
    ]


def _earliest_intersection(
    start: Point,
    end: Point,
    constraints: Sequence[Constraint],
    *,
    after_t: float = 0.0,
) -> tuple[float, Constraint] | None:
    intersections: list[tuple[float, Constraint]] = []
    for constraint in constraints:
        interval = segment_constraint_interval(start, end, constraint)
        if interval is None or interval[1] < after_t - EPSILON:
            continue
        intersections.append((max(interval[0], after_t), constraint))
    return min(intersections, key=lambda item: item[0]) if intersections else None


def _first_constraint_hit_after(
    start: Point,
    end: Point,
    constraint: Constraint,
    after_t: float,
) -> float | None:
    interval = segment_constraint_interval(start, end, constraint)
    if interval is None or interval[1] < after_t - EPSILON:
        return None
    return max(interval[0], after_t)


def compute_hcs_progress(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
) -> dict[str, object]:
    """Return constraint-aware ordered hard-constraint progress."""

    ordered_must_pass = ordered_must_pass_constraints(hard_constraints)
    if len(ordered_must_pass) < 2:
        raise ValueError(
            "HCS requires at least one scored must-pass region after the first "
            "record region."
        )

    record_region = ordered_must_pass[0]
    must_pass = ordered_must_pass[1:]
    scoped_must_pass = [record_region, *must_pass]

    points = [_point(point, "trajectory point") for point in trajectory]
    completed_count = 0
    failed = False
    failure_type: str | None = None
    failed_transition: int | None = None
    first_violation_id: str | None = None
    violations: list[str] = []

    if points:
        while completed_count < len(must_pass) and point_inside_constraint(
            points[0],
            must_pass[completed_count],
        ):
            completed_count += 1

    for start, end in zip(points, points[1:]):
        if failed or completed_count == len(must_pass):
            break

        segment_t = 0.0
        while completed_count < len(must_pass):
            active_avoid = _active_must_avoid_constraints(
                hard_constraints,
                scoped_must_pass,
                completed_count + 1,
            )
            avoid_hit = _earliest_intersection(
                start,
                end,
                active_avoid,
                after_t=segment_t,
            )
            expected_goal = must_pass[completed_count]
            goal_hit = _first_constraint_hit_after(
                start,
                end,
                expected_goal,
                segment_t,
            )

            if avoid_hit is not None and (
                goal_hit is None or avoid_hit[0] <= goal_hit + EPSILON
            ):
                failed = True
                failure_type = "must_avoid_violation"
                failed_transition = completed_count + 1
                identifier = _hard_constraint_identifier(
                    avoid_hit[1],
                    f"must_avoid_{len(violations) + 1}",
                )
                first_violation_id = identifier
                violations.append(identifier)
                break

            if goal_hit is None:
                break

            completed_count += 1
            segment_t = goal_hit
            if completed_count == len(must_pass):
                break

    total = len(must_pass)
    return {
        "score": completed_count / total,
        "valid_completed_count": completed_count,
        "must_pass_count": total,
        "ignored_record_must_pass": {
            "constraint_id": record_region.get("constraint_id"),
            "order": record_region.get("order"),
        },
        "failed": failed,
        "first_failed_transition": failed_transition,
        "failure_type": failure_type,
        "first_must_avoid_violation": first_violation_id,
        "must_avoid_violations": violations,
    }


def count_ordered_must_passes(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
) -> tuple[int, int]:
    """Count the ordered must-pass prefix without applying avoid constraints."""

    must_pass = ordered_must_pass_constraints(hard_constraints)
    if len(must_pass) < 2:
        return 0, 0
    sanitized_constraints = [
        constraint
        for constraint in hard_constraints
        if constraint.get("kind") != "must_avoid"
    ]
    progress = compute_hcs_progress(trajectory, sanitized_constraints)
    return int(progress["valid_completed_count"]), int(progress["must_pass_count"])


def find_must_avoid_violations(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
) -> list[str]:
    """Return identifiers for every must-avoid region touched by the trajectory."""

    violations: list[str] = []
    must_avoid = _must_avoid_constraints(hard_constraints)
    for index, constraint in enumerate(must_avoid, start=1):
        point_hit = any(point_inside_constraint(point, constraint) for point in trajectory)
        segment_hit = any(
            segment_intersects_constraint(start, end, constraint)
            for start, end in zip(trajectory, trajectory[1:])
        )
        if point_hit or segment_hit:
            violations.append(_hard_constraint_identifier(constraint, f"must_avoid_{index}"))
    return violations


def compute_hcs(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
) -> float:
    """Return valid ordered hard-constraint progress under active avoids."""

    return float(compute_hcs_progress(trajectory, hard_constraints)["score"])


def compute_hcs_details(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
) -> list[dict[str, object]]:
    """Return per-hard-constraint diagnostics used by HCS."""

    progress = compute_hcs_progress(trajectory, hard_constraints)
    matched = int(progress["valid_completed_count"])
    must_pass = ordered_must_pass_constraints(hard_constraints)[1:]
    details: list[dict[str, object]] = []
    for index, constraint in enumerate(must_pass, start=1):
        score = 1.0 if index <= matched else 0.0
        details.append(
            {
                "constraint_id": constraint.get("constraint_id"),
                "kind": "must_pass",
                "order": constraint.get("order"),
                "score": score,
                "status": "matched_in_order" if score == 1.0 else "missed",
            }
        )

    first_violation = progress["first_must_avoid_violation"]
    must_avoid = _must_avoid_constraints(hard_constraints)
    for index, constraint in enumerate(must_avoid, start=1):
        identifier = _hard_constraint_identifier(constraint, f"must_avoid_{index}")
        violated = identifier == first_violation
        details.append(
            {
                "constraint_id": constraint.get("constraint_id"),
                "kind": "must_avoid",
                "score": 0.0 if violated else 1.0,
                "status": "violated" if violated else "avoided",
            }
        )

    return details


def compute_oss(
    *,
    hcs: float,
    scs: float | None,
    plr: float | None,
) -> float | None:
    """Return episode score as SCS * HCS * PLR."""

    if plr is None:
        return None
    effective_scs = 1.0 if scs is None else scs
    return effective_scs * hcs * plr


def path_length_details(
    trajectory: Trajectory,
    reference_trajectory: Trajectory | None,
) -> dict[str, object]:
    prediction_length = compute_path_length(trajectory)
    reference_length = (
        None
        if reference_trajectory is None
        else compute_path_length(reference_trajectory)
    )
    effective_prediction_length = effective_plr_path_length(trajectory)
    effective_reference_length = (
        None
        if reference_trajectory is None
        else effective_plr_path_length(reference_trajectory)
    )
    return {
        "prediction_path_length": prediction_length,
        "expert_path_length": reference_length,
        "effective_prediction_path_length": effective_prediction_length,
        "effective_expert_path_length": effective_reference_length,
        "plr_min_effective_path_length_grid": float(PLR_MIN_EFFECTIVE_PATH_LENGTH_GRID),
        "PLR": compute_path_length_ratio(trajectory, reference_trajectory),
    }


def compute_scs(constraint_scores: Iterable[Real]) -> float:
    """Average per-soft-constraint scores; an empty set is neutral and scores 1."""

    scores = list(constraint_scores)
    for score in scores:
        if (
            not isinstance(score, Real)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
        ):
            raise ValueError("Each soft constraint score must be between 0 and 1.")
    return 1.0 if not scores else math.fsum(float(score) for score in scores) / len(scores)


def soft_score_normalization_details(
    *,
    preference_type: object,
    algorithm_value: object,
    human_value: object,
    comparison_direction: str,
) -> dict[str, object]:
    """Return raw-value diagnostics for human-relative SCS normalization."""

    details: dict[str, object] = {
        "score_normalization": "human_relative_linear_bad_ratio",
        "comparison_direction": comparison_direction,
        "algorithm_value": algorithm_value,
        "human_value": human_value,
        "scs_config": scs_config_details(),
    }
    if (
        algorithm_value is None
        or human_value is None
        or not isinstance(algorithm_value, Real)
        or isinstance(algorithm_value, bool)
        or not isinstance(human_value, Real)
        or isinstance(human_value, bool)
    ):
        details["degradation_ratio"] = None
        return details

    if comparison_direction == "smaller_is_better":
        _score, ratio = score_smaller_is_better(
            float(algorithm_value),
            float(human_value),
        )
    elif comparison_direction == "larger_is_better":
        _score, ratio = score_larger_is_better(
            float(algorithm_value),
            float(human_value),
        )
    else:
        details["degradation_ratio"] = None
        return details

    details["degradation_ratio"] = ratio if math.isfinite(ratio) else None
    return details


def unique_trajectory_points(trajectory: Trajectory) -> list[tuple[float, float]]:
    """Deduplicate waypoint coordinates while preserving their first occurrence."""

    unique: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for point in trajectory:
        parsed = _point(point, "trajectory point")
        if parsed not in seen:
            unique.append(parsed)
            seen.add(parsed)
    return unique


def mean_region_distance_to_reference(
    trajectory: Trajectory,
    constraint: Constraint,
    reference_points: Sequence[Point],
) -> float | None:
    """Average nearest-reference distance for unique waypoints inside a region."""

    references = [_point(point, "reference point") for point in reference_points]
    if not references:
        raise ValueError("reference_points must contain at least one point.")

    region_points = [
        point
        for point in unique_trajectory_points(trajectory)
        if point_inside_constraint(point, constraint)
    ]
    if not region_points:
        return None

    distances = [
        min(
            math.hypot(row - reference_row, col - reference_col)
            for reference_row, reference_col in references
        )
        for row, col in region_points
    ]
    return math.fsum(distances) / len(distances)


def compute_distance_preference_score(
    trajectory: Trajectory,
    constraint: Constraint,
    *,
    epsilon: float = EPSILON,
) -> tuple[float | None, float | None]:
    """Score one near/far preference and return ``(score, D_pred)``."""

    preference_type = constraint.get("preference_type")
    if preference_type not in {"near_preference", "far_preference"}:
        raise ValueError("Distance scoring only supports near_preference and far_preference.")

    reference_region = constraint.get("reference_region")
    if not isinstance(reference_region, Mapping):
        raise ValueError("near/far preference requires reference_region.")
    reference_points = reference_region.get("cells")
    if not isinstance(reference_points, Sequence) or isinstance(
        reference_points, (str, bytes)
    ):
        raise ValueError("reference_region.cells must be a list.")

    d_pred = mean_region_distance_to_reference(
        trajectory,
        constraint,
        reference_points,
    )
    if d_pred is None:
        return None, None

    d_ref = constraint.get("D_ref")
    if d_ref is None:
        if constraint.get("reference_metric_status") != "invalid_human_reference":
            warn_missing_reference_metric(constraint, "D_ref")
        return 1.0, d_pred
    if not isinstance(d_ref, Real) or isinstance(d_ref, bool) or d_ref < 0:
        raise ValueError("near/far preference requires a non-negative D_ref.")
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    if preference_type == "near_preference":
        score, _ratio = score_smaller_is_better(d_pred, float(d_ref))
    else:
        score, _ratio = score_larger_is_better(d_pred, float(d_ref))
    return score, d_pred


def compute_relative_preference_score(
    trajectory: Trajectory,
    constraint: Constraint,
) -> tuple[float | None, float | None, dict[str, object]]:
    """Score a relative preference against the stored human reference margin."""

    q_pred, quality_details = compute_relative_preference_quality(
        trajectory,
        constraint,
    )
    if q_pred is None:
        return None, None, quality_details

    q_ref = constraint.get("Q_relative_ref")
    if q_ref is None:
        if constraint.get("relative_reference_status") == "invalid_human_reference":
            return 0.0, q_pred, {
                **quality_details,
                "Q_relative_ref": None,
                "score_normalization": "invalid_human_reference_zero",
                "zero_threshold": None,
                "status": "invalid_human_reference",
            }
        warn_missing_reference_metric(constraint, "Q_relative_ref")
        return 1.0, q_pred, {
            **quality_details,
            "Q_relative_ref": None,
            "score_normalization": "missing_reference_neutral",
            "zero_threshold": None,
            "status": "computed",
        }
    if not isinstance(q_ref, Real) or isinstance(q_ref, bool):
        raise ValueError("relative_preference requires numeric Q_relative_ref.")

    score, zero_threshold = normalize_relative_preference_quality(
        predicted_quality=q_pred,
        human_quality=float(q_ref),
    )
    return score, q_pred, {
        **quality_details,
        "Q_relative_ref": float(q_ref),
        "score_normalization": "human_relative_margin",
        "zero_threshold": zero_threshold,
        "status": "computed",
    }


def compute_relative_preference_quality(
    trajectory: Trajectory,
    constraint: Constraint,
    *,
    epsilon: float = EPSILON,
    min_valid_length: float = RELATIVE_PREFERENCE_MIN_VALID_LENGTH_GRID,
) -> tuple[float | None, dict[str, object]]:
    """Return length-weighted continuous margin for ordered object A over B."""

    if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if (
        not isinstance(min_valid_length, Real)
        or isinstance(min_valid_length, bool)
        or float(min_valid_length) < 0
    ):
        raise ValueError("min_valid_length must be non-negative.")

    reference_regions = constraint.get("reference_regions")
    if (
        not isinstance(reference_regions, Sequence)
        or isinstance(reference_regions, (str, bytes))
        or len(reference_regions) != 2
    ):
        raise ValueError(
            "relative_preference requires two ordered reference_regions."
        )

    parsed_references: list[list[tuple[float, float]]] = []
    for index, reference_region in enumerate(reference_regions):
        if not isinstance(reference_region, Mapping):
            raise ValueError("Each relative reference region must be an object.")
        cells = reference_region.get("cells")
        if (
            not isinstance(cells, Sequence)
            or isinstance(cells, (str, bytes))
            or not cells
        ):
            raise ValueError(
                f"relative reference object {index + 1} must contain cells."
            )
        parsed_references.append([
            _point(cell, "relative reference cell") for cell in cells
        ])

    points = [_point(point, "trajectory point") for point in trajectory]
    weighted_margin_sum = 0.0
    valid_length = 0.0
    valid_segment_count = 0
    positive_segment_count = 0
    if len(points) >= 2:
        for start, end in zip(points, points[1:]):
            segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
            if segment_length <= EPSILON:
                continue
            for enter_t, exit_t in segment_constraint_intervals(start, end, constraint):
                if exit_t - enter_t <= EPSILON:
                    continue
                midpoint_t = (enter_t + exit_t) / 2
                midpoint = (
                    start[0] + (end[0] - start[0]) * midpoint_t,
                    start[1] + (end[1] - start[1]) * midpoint_t,
                )
                active_length = segment_length * (exit_t - enter_t)
                margin = relative_margin_for_point(
                    midpoint,
                    parsed_references[0],
                    parsed_references[1],
                    epsilon=float(epsilon),
                )
                weighted_margin_sum += active_length * margin
                valid_length += active_length
                valid_segment_count += 1
                if margin > 0:
                    positive_segment_count += 1

    details = {
        "valid_path_length": valid_length,
        "valid_segment_count": valid_segment_count,
        "positive_segment_count": positive_segment_count,
        "min_valid_length": float(min_valid_length),
    }
    if valid_length < float(min_valid_length):
        return None, details
    return weighted_margin_sum / valid_length, details


def relative_margin_for_point(
    point: Point,
    preferred_points: Sequence[Point],
    reference_points: Sequence[Point],
    *,
    epsilon: float = EPSILON,
) -> float:
    """Return normalized margin: positive means closer to preferred_points."""

    row, col = _point(point, "relative point")
    parsed_preferred = [
        _point(cell, "preferred reference cell") for cell in preferred_points
    ]
    parsed_reference = [
        _point(cell, "reference reference cell") for cell in reference_points
    ]
    if not parsed_preferred or not parsed_reference:
        raise ValueError("relative margin requires non-empty reference cells.")
    d_preferred = min(
        math.hypot(row - reference_row, col - reference_col)
        for reference_row, reference_col in parsed_preferred
    )
    d_reference = min(
        math.hypot(row - reference_row, col - reference_col)
        for reference_row, reference_col in parsed_reference
    )
    return (d_reference - d_preferred) / (
        d_preferred + d_reference + float(epsilon)
    )


def normalize_relative_preference_quality(
    *,
    predicted_quality: float | None,
    human_quality: float,
    beta: float = RELATIVE_PREFERENCE_HUMAN_BETA,
    min_margin: float = RELATIVE_PREFERENCE_MIN_MARGIN,
    epsilon: float = EPSILON,
) -> tuple[float, float]:
    """Convert raw relative margin into a bounded score and zero threshold."""

    if predicted_quality is None:
        zero_threshold = human_quality - max(
            float(beta) * abs(human_quality),
            float(min_margin),
        )
        return 0.0, zero_threshold
    if (
        not isinstance(human_quality, Real)
        or isinstance(human_quality, bool)
        or not math.isfinite(float(human_quality))
    ):
        raise ValueError("human_quality must be finite.")
    if not isinstance(beta, Real) or isinstance(beta, bool) or float(beta) < 0:
        raise ValueError("beta must be non-negative.")
    if (
        not isinstance(min_margin, Real)
        or isinstance(min_margin, bool)
        or float(min_margin) <= 0
    ):
        raise ValueError("min_margin must be positive.")
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or float(epsilon) <= 0:
        raise ValueError("epsilon must be positive.")
    zero_threshold = float(human_quality) - max(
        float(beta) * abs(float(human_quality)),
        float(min_margin),
    )
    denominator = float(human_quality) - zero_threshold
    if denominator <= float(epsilon):
        raise ValueError("Invalid normalization interval for relative preference.")
    score = (float(predicted_quality) - zero_threshold) / denominator
    return max(0.0, min(1.0, score)), zero_threshold


def compute_move_smoothness_score(
    trajectory: Trajectory,
    constraint: Constraint,
    *,
    epsilon: float = EPSILON,
) -> tuple[float, float, int]:
    """Score trajectory smoothness against the human reference smoothness cost."""

    c_pred, turn_count = compute_mean_squared_turning_angle(trajectory)
    c_ref = constraint.get("C_smooth_ref")
    if c_ref is None:
        return 1.0, c_pred, turn_count
    if not isinstance(c_ref, Real) or isinstance(c_ref, bool) or c_ref < 0:
        raise ValueError("move_smoothness requires a non-negative C_smooth_ref.")
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    score, _ratio = score_smaller_is_better(c_pred, float(c_ref))
    return score, c_pred, turn_count


def _dtw_cost_and_steps(
    trajectory: Trajectory,
    reference_trajectory: Trajectory,
    *,
    tolerance_grid: float = 0.0,
) -> tuple[float, int]:
    """Return cumulative DTW cost and alignment steps."""

    if (
        not isinstance(tolerance_grid, Real)
        or isinstance(tolerance_grid, bool)
        or float(tolerance_grid) < 0.0
    ):
        raise ValueError("tolerance_grid must be non-negative.")
    tolerance = float(tolerance_grid)
    predicted_points = [_point(point, "trajectory point") for point in trajectory]
    reference_points = [
        _point(point, "reference trajectory point")
        for point in reference_trajectory
    ]
    if not predicted_points or not reference_points:
        return 0.0 if not predicted_points and not reference_points else math.inf, 0

    pred_count = len(predicted_points)
    ref_count = len(reference_points)
    costs = [[math.inf] * (ref_count + 1) for _ in range(pred_count + 1)]
    steps = [[0] * (ref_count + 1) for _ in range(pred_count + 1)]
    costs[0][0] = 0.0

    for pred_index, predicted in enumerate(predicted_points, start=1):
        for ref_index, reference in enumerate(reference_points, start=1):
            previous_options = (
                (costs[pred_index - 1][ref_index], steps[pred_index - 1][ref_index]),
                (costs[pred_index][ref_index - 1], steps[pred_index][ref_index - 1]),
                (
                    costs[pred_index - 1][ref_index - 1],
                    steps[pred_index - 1][ref_index - 1],
                ),
            )
            previous_cost, previous_steps = min(
                previous_options,
                key=lambda item: (item[0], item[1]),
            )
            local_cost = math.hypot(
                predicted[0] - reference[0],
                predicted[1] - reference[1],
            )
            costs[pred_index][ref_index] = previous_cost + max(0.0, local_cost - tolerance)
            steps[pred_index][ref_index] = previous_steps + 1

    alignment_steps = steps[pred_count][ref_count]
    if alignment_steps == 0:
        return math.inf, 0
    return costs[pred_count][ref_count], alignment_steps


def compute_dtw_distance(
    trajectory: Trajectory,
    reference_trajectory: Trajectory,
) -> tuple[float, int]:
    """Return mean per-alignment-step DTW distance between two trajectories."""

    cost, alignment_steps = _dtw_cost_and_steps(trajectory, reference_trajectory)
    if math.isinf(cost) or alignment_steps == 0:
        return cost, alignment_steps
    return cost / alignment_steps, alignment_steps


def compute_ndtw_score(
    trajectory: Trajectory,
    reference_trajectory: Trajectory,
    *,
    success_distance: float = PATH_SHAPE_NDTW_SUCCESS_DISTANCE,
    tolerance_grid: float = PATH_SHAPE_TOLERANCE_GRID,
) -> tuple[float, float, float, int]:
    """Return nDTW score, cumulative DTW cost, mean DTW distance, and steps."""

    if (
        not isinstance(success_distance, Real)
        or isinstance(success_distance, bool)
        or float(success_distance) <= 0
    ):
        raise ValueError("PATH_SHAPE_NDTW_SUCCESS_DISTANCE must be positive.")

    reference_points = [
        _point(point, "reference trajectory point")
        for point in reference_trajectory
    ]
    dtw_cost, alignment_steps = _dtw_cost_and_steps(
        trajectory,
        reference_trajectory,
        tolerance_grid=tolerance_grid,
    )
    if math.isinf(dtw_cost):
        return 0.0, dtw_cost, dtw_cost, alignment_steps
    if alignment_steps == 0:
        return 1.0, 0.0, 0.0, 0

    mean_dtw_distance = dtw_cost / alignment_steps
    denominator = float(success_distance) * max(len(reference_points), 1)
    return math.exp(-dtw_cost / denominator), dtw_cost, mean_dtw_distance, alignment_steps


def compute_path_shape_preference_score(
    trajectory: Trajectory,
    constraint: Constraint,
    *,
    epsilon: float = EPSILON,
) -> tuple[float, float, int]:
    """Score path-shape similarity against a human reference trajectory via DTW."""

    reference_trajectory = constraint.get("reference_trajectory")
    if reference_trajectory is None:
        if constraint.get("reference_metric_status") != "invalid_human_reference":
            warn_missing_reference_metric(constraint, "reference_trajectory")
        return 1.0, 0.0, 0
    if not isinstance(reference_trajectory, Sequence) or isinstance(
        reference_trajectory, (str, bytes)
    ):
        raise ValueError("path_shape_preference requires reference_trajectory.")
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    score, _dtw_cost, c_pred, alignment_steps = compute_ndtw_score(
        trajectory,
        reference_trajectory,  # type: ignore[arg-type]
    )
    if math.isinf(c_pred):
        return 0.0, c_pred, alignment_steps
    return score, c_pred, alignment_steps


def ordered_must_pass_hit_indexes(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
) -> dict[int, int]:
    """Return first ordered hit index for each must-pass order."""

    hit_indexes: dict[int, int] = {}
    start_index = 0
    for constraint in ordered_must_pass_constraints(hard_constraints):
        order = constraint.get("order")
        if not isinstance(order, Real) or isinstance(order, bool):
            continue
        parsed_order = int(order)
        for index in range(start_index, len(trajectory)):
            if point_inside_constraint(trajectory[index], constraint):
                hit_indexes[parsed_order] = index
                start_index = index + 1
                break
    return hit_indexes


def scoped_trajectory_for_constraint(
    trajectory: Trajectory,
    constraint: Constraint,
    hit_indexes: Mapping[int, int],
) -> Trajectory:
    """Return the soft-constraint trajectory segment.

    Soft scores are independent from hard-constraint success. When a local
    scope boundary is missing, keep the observable part of the path instead of
    returning an empty segment that would make SCS depend on HCS completion.
    """

    scope = constraint.get("scope")
    if not isinstance(scope, Mapping) or scope.get("type") != "between_hard_constraints":
        return trajectory
    from_order = scope.get("from_order")
    to_order = scope.get("to_order")
    if not isinstance(from_order, int) or not isinstance(to_order, int):
        return trajectory
    from_index = hit_indexes.get(from_order)
    to_index = hit_indexes.get(to_order)
    if from_index is not None and to_index is not None and to_index > from_index:
        return trajectory[from_index : to_index + 1]
    if from_index is not None:
        return trajectory[from_index:]
    if to_index is not None:
        return trajectory[: to_index + 1]
    return trajectory


def trajectory_inside_constraint_region(
    trajectory: Trajectory,
    constraint: Constraint,
) -> list[list[float]]:
    """Return trajectory waypoints inside the constraint's own region."""

    return [
        [float(row), float(col)]
        for row, col in (_point(point, "trajectory point") for point in trajectory)
        if point_inside_constraint((row, col), constraint)
    ]


def constraint_has_region(constraint: Constraint) -> bool:
    if constraint.get("preference_type") in {"clearance", "move_smoothness"}:
        return False
    return constraint.get("shape") in {"circle", "rectangle", "freeform"}


def no_region_soft_constraint_detail(
    constraint: Constraint,
    preference_type: object,
    active_trajectory: Trajectory,
    *,
    region_waypoint_count: int,
) -> dict[str, object]:
    return {
        "constraint_id": constraint.get("constraint_id"),
        "preference_type": preference_type,
        "scope": constraint.get("scope"),
        "active_waypoint_count": len(active_trajectory),
        "region_waypoint_count": region_waypoint_count,
        "score": 0.0,
        "score_normalization": "no_region_waypoints_zero",
        "status": "no_region_waypoints_scored_zero",
    }


def compute_soft_constraint_scores(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
    soft_constraints: Sequence[Constraint],
    map_state: Mapping[str, object] | None = None,
) -> tuple[list[float] | None, list[dict[str, object]]]:
    """Compute supported soft scores; return pending when a rule is undefined."""

    scores: list[float] = []
    details: list[dict[str, object]] = []
    hit_indexes = ordered_must_pass_hit_indexes(trajectory, hard_constraints)
    for constraint in soft_constraints:
        preference_type = constraint.get("preference_type")
        active_trajectory = scoped_trajectory_for_constraint(
            trajectory,
            constraint,
            hit_indexes,
        )
        region_trajectory = (
            trajectory_inside_constraint_region(active_trajectory, constraint)
            if constraint_has_region(constraint)
            else None
        )
        if preference_type == "relative_preference":
            score, q_pred, relative_details = (
                compute_relative_preference_score(active_trajectory, constraint)
            )
            if score is None:
                scores.append(0.0)
                details.append(
                    no_region_soft_constraint_detail(
                        constraint,
                        preference_type,
                        active_trajectory,
                        region_waypoint_count=0,
                    )
                    | relative_details
                )
                continue
            scores.append(score)
            valid_segment_count = relative_details.get("valid_segment_count")
            positive_segment_count = relative_details.get("positive_segment_count")
            details.append(
                {
                    "constraint_id": constraint.get("constraint_id"),
                    "preference_type": preference_type,
                    "scope": constraint.get("scope"),
                    "active_waypoint_count": len(active_trajectory),
                    "Q_relative_ref": relative_details.get("Q_relative_ref"),
                    "Q_relative_pred": q_pred,
                    "score": score,
                    "score_normalization": relative_details.get(
                        "score_normalization"
                    ),
                    "zero_threshold": relative_details.get("zero_threshold"),
                    "valid_path_length": relative_details.get("valid_path_length"),
                    "valid_segment_count": valid_segment_count,
                    "positive_segment_count": positive_segment_count,
                    "region_waypoints": valid_segment_count,
                    "satisfying_waypoints": positive_segment_count,
                    "status": relative_details.get("status", "computed"),
                }
            )
            continue
        if region_trajectory is not None and not region_trajectory:
            scores.append(0.0)
            details.append(
                no_region_soft_constraint_detail(
                    constraint,
                    preference_type,
                    active_trajectory,
                    region_waypoint_count=0,
                )
            )
            continue
        if preference_type == "clearance":
            if map_state is None:
                scores.append(1.0)
                details.append(
                    {
                        "constraint_id": constraint.get("constraint_id"),
                        "preference_type": preference_type,
                        "scope": constraint.get("scope"),
                        "active_waypoint_count": len(active_trajectory),
                        "C_clear_ref": constraint.get("C_clear_ref"),
                        "C_clear_pred": None,
                        "waypoint_count": 0,
                        "score": 1.0,
                        "status": "assumed_without_map_state",
                    }
                )
                continue
            score, c_pred, waypoint_count = compute_clearance_score(
                active_trajectory,
                constraint,
                map_state,
                soft_constraints,
            )
            scores.append(score)
            details.append(
                {
                    "constraint_id": constraint.get("constraint_id"),
                    "preference_type": preference_type,
                    "scope": constraint.get("scope"),
                    "active_waypoint_count": len(active_trajectory),
                    "C_clear_ref": constraint.get("C_clear_ref"),
                    "C_clear_pred": c_pred,
                    "waypoint_count": waypoint_count,
                    "score": score,
                    **soft_score_normalization_details(
                        preference_type=preference_type,
                        algorithm_value=c_pred,
                        human_value=constraint.get("C_clear_ref"),
                        comparison_direction="smaller_is_better",
                    ),
                    "status": "computed",
                }
            )
            continue
        if preference_type == "move_smoothness":
            if not move_smoothness_enabled():
                details.append(
                    {
                        "constraint_id": constraint.get("constraint_id"),
                        "preference_type": preference_type,
                        "scope": constraint.get("scope"),
                        "active_waypoint_count": len(active_trajectory),
                        "score": None,
                        "status": "disabled_by_hyperparameter",
                    }
                )
                continue
            score, c_pred, turn_count = compute_move_smoothness_score(
                active_trajectory,
                constraint,
            )
            scores.append(score)
            details.append(
                {
                    "constraint_id": constraint.get("constraint_id"),
                    "preference_type": preference_type,
                    "scope": constraint.get("scope"),
                    "active_waypoint_count": len(active_trajectory),
                    "C_smooth_ref": constraint.get("C_smooth_ref"),
                    "C_smooth_pred": c_pred,
                    "turn_count": turn_count,
                    "score": score,
                    **soft_score_normalization_details(
                        preference_type=preference_type,
                        algorithm_value=c_pred,
                        human_value=constraint.get("C_smooth_ref"),
                        comparison_direction="smaller_is_better",
                    ),
                    "status": "computed",
                }
            )
            continue
        if preference_type == "path_shape_preference":
            region_trajectory = (
                region_trajectory
                if region_trajectory is not None
                else list(active_trajectory)  # type: ignore[arg-type]
            )
            reference_trajectory = constraint.get("reference_trajectory", [])
            score, c_pred, alignment_steps = compute_path_shape_preference_score(
                region_trajectory,
                constraint,
            )
            dtw_cost = None
            if isinstance(reference_trajectory, Sequence) and not isinstance(
                reference_trajectory,
                (str, bytes),
            ):
                _ndtw, dtw_cost, _mean_dtw, _steps = compute_ndtw_score(
                    region_trajectory,
                    reference_trajectory,  # type: ignore[arg-type]
                )
            scores.append(score)
            details.append(
                {
                    "constraint_id": constraint.get("constraint_id"),
                    "preference_type": preference_type,
                    "scope": constraint.get("scope"),
                    "active_waypoint_count": len(active_trajectory),
                    "region_waypoint_count": len(region_trajectory),
                    "C_shape_pred": c_pred,
                    "dtw_alignment_steps": alignment_steps,
                    "dtw_cost": dtw_cost,
                    "nDTW": score,
                    "ndtw_success_distance": PATH_SHAPE_NDTW_SUCCESS_DISTANCE,
                    "path_shape_tolerance_grid": PATH_SHAPE_TOLERANCE_GRID,
                    "reference_waypoint_count": len(
                        constraint.get("reference_trajectory", [])  # type: ignore[arg-type]
                    )
                    if isinstance(constraint.get("reference_trajectory"), Sequence)
                    and not isinstance(constraint.get("reference_trajectory"), (str, bytes))
                    else 0,
                    "score": score,
                    "score_normalization": "nDTW",
                    "status": "computed",
                }
            )
            continue
        if preference_type not in {"near_preference", "far_preference"}:
            details.append(
                {
                    "constraint_id": constraint.get("constraint_id"),
                    "preference_type": preference_type,
                    "status": "pending_scoring_rule",
                }
            )
            continue

        score, d_pred = compute_distance_preference_score(active_trajectory, constraint)
        if score is None:
            scores.append(0.0)
            details.append(
                no_region_soft_constraint_detail(
                    constraint,
                    preference_type,
                    active_trajectory,
                    region_waypoint_count=0,
                )
            )
            continue
        scores.append(score)
        details.append(
            {
                "constraint_id": constraint.get("constraint_id"),
                "preference_type": preference_type,
                "scope": constraint.get("scope"),
                "active_waypoint_count": len(active_trajectory),
                "D_ref": constraint.get("D_ref"),
                "D_pred": d_pred,
                "score": score,
                **soft_score_normalization_details(
                    preference_type=preference_type,
                    algorithm_value=d_pred,
                    human_value=constraint.get("D_ref"),
                    comparison_direction=(
                        "smaller_is_better"
                        if preference_type == "near_preference"
                        else "larger_is_better"
                    ),
                ),
                "status": "computed",
            }
        )
    return scores, details


def evaluate_trajectory(
    trajectory: Trajectory,
    hard_constraints: Sequence[Constraint],
    soft_constraints: Sequence[Constraint] = (),
    soft_constraint_scores: Sequence[Real] | None = None,
    map_state: Mapping[str, object] | None = None,
    reference_trajectory: Trajectory | None = None,
) -> dict[str, object]:
    """Compute all currently defined metrics and useful hard-constraint details."""

    hcs_progress = compute_hcs_progress(trajectory, hard_constraints)
    matched = int(hcs_progress["valid_completed_count"])
    total = int(hcs_progress["must_pass_count"])
    hcs_details = compute_hcs_details(trajectory, hard_constraints)
    hcs = float(hcs_progress["score"])
    pl_details = path_length_details(trajectory, reference_trajectory)
    raw_plr = pl_details["PLR"]
    plr = (
        float(raw_plr)
        if isinstance(raw_plr, Real) and not isinstance(raw_plr, bool)
        else None
    )
    evaluated_soft_constraints = normalize_soft_constraints_for_evaluation(
        soft_constraints,
        hard_constraints=hard_constraints,
        map_state=map_state,
        reference_trajectory=reference_trajectory,
    )

    soft_details: list[dict[str, object]] = []
    scs_status = "computed"
    if soft_constraint_scores is None:
        computed_scores, soft_details = compute_soft_constraint_scores(
            trajectory,
            hard_constraints,
            evaluated_soft_constraints,
            map_state,
        )
        if computed_scores is None:
            scs = None
            scs_status = "pending_soft_constraint_scoring_rules"
        elif computed_scores:
            scs = compute_scs(computed_scores)
        else:
            scs = None
            scs_status = "no_active_soft_constraints"
    else:
        if len(soft_constraint_scores) != len(evaluated_soft_constraints):
            raise ValueError(
                "soft_constraint_scores must contain one score per soft constraint."
            )
        scs = compute_scs(soft_constraint_scores)
        soft_details = [
            {
                "constraint_id": constraint.get("constraint_id"),
                "preference_type": constraint.get("preference_type"),
                "scope": constraint.get("scope"),
                "score": float(score),
                "status": "provided",
            }
            for constraint, score in zip(evaluated_soft_constraints, soft_constraint_scores)
        ]

    return {
        "PLR": pl_details["PLR"],
        "HCS": hcs,
        "SCS": scs,
        "OSS": compute_oss(hcs=hcs, scs=scs, plr=plr),
        "scs_config": scs_config_details(),
        "details": {
            **pl_details,
            "matched_must_pass": matched,
            "valid_completed_count": matched,
            "total_must_pass": total,
            "must_pass_count": total,
            "ignored_record_must_pass": hcs_progress["ignored_record_must_pass"],
            "first_failed_transition": hcs_progress["first_failed_transition"],
            "failure_type": hcs_progress["failure_type"],
            "must_avoid_violations": hcs_progress["must_avoid_violations"],
            "hard_constraints": hcs_details,
            "soft_constraint_count": len(evaluated_soft_constraints),
            "soft_constraints": soft_details,
            "scs_config": scs_config_details(),
            "scs_status": scs_status,
        },
    }


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_map_json_path(map_id: object) -> Path | None:
    if not isinstance(map_id, str) or not map_id.strip():
        return None
    normalized = map_id.strip().replace("\\", "/").strip("/")
    leaf = normalized.rsplit("/", 1)[-1]
    candidates = [
        MAP_ROOT / normalized / f"{leaf}.json",
        MAP_ROOT / f"{normalized}.json",
    ]
    if leaf.endswith("_train"):
        candidates.append(MAP_ROOT / "procthor" / "train" / leaf / f"{leaf}.json")
    elif leaf.endswith("_valunseen"):
        candidates.append(MAP_ROOT / "procthor" / "valunseen" / leaf / f"{leaf}.json")
    if "/" not in normalized:
        candidates.extend(MAP_ROOT.glob(f"*/{leaf}/{leaf}.json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(MAP_ROOT.rglob(f"{leaf}.json"))
    return matches[0] if matches else None


def _load_map_state_for_instruction(
    instruction: Mapping[str, object],
) -> Mapping[str, object] | None:
    map_path = _resolve_map_json_path(instruction.get("map_id"))
    if map_path is None:
        return None
    payload = _load_json(map_path)
    return payload if isinstance(payload, Mapping) else None


def _trajectory_from_payload(payload: object) -> Sequence[Point]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("trajectory", "predicted_trajectory", "human_expert_trajectory"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(
        "Trajectory JSON must be a point list or contain trajectory, "
        "predicted_trajectory, or human_expert_trajectory."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SemPathBench trajectory against one instruction JSON."
    )
    parser.add_argument("instruction_file", type=Path)
    parser.add_argument(
        "--trajectory-file",
        type=Path,
        help="Optional prediction JSON. Defaults to human_expert_trajectory.",
    )
    parser.add_argument(
        "--soft-score",
        type=float,
        action="append",
        dest="soft_scores",
        help="Per-soft-constraint score in annotation order. Repeat for each constraint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instruction = _load_json(args.instruction_file)
    if not isinstance(instruction, dict):
        raise ValueError("Instruction file must contain a JSON object.")

    trajectory_payload = (
        _load_json(args.trajectory_file)
        if args.trajectory_file
        else instruction.get("human_expert_trajectory", [])
    )
    trajectory = _trajectory_from_payload(trajectory_payload)
    hard_constraints = instruction.get("hard_constraints", [])
    soft_constraints = instruction.get("soft_constraints", [])
    if not isinstance(hard_constraints, list) or not isinstance(
        soft_constraints, list
    ):
        raise ValueError("hard_constraints and soft_constraints must be lists.")
    map_state = _load_map_state_for_instruction(instruction)

    metrics = evaluate_trajectory(
        trajectory,
        hard_constraints,
        soft_constraints,
        soft_constraint_scores=args.soft_scores,
        map_state=map_state,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
