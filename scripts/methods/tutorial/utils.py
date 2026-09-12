"""Helper functions for the tutorial baseline method."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from scripts.evaluation.evaluation_metrics import (
    ordered_must_pass_constraints,
    point_inside_constraint,
    segment_intersects_constraint,
)
from scripts.make_instruction.make_instruction import REPO_ROOT, normalize_map_key
from scripts.methods.util.grid_astar import (
    TraversableGrid,
    astar_path_to_any,
    is_traversable,
)


DIFFICULTY_LEVELS = ("easy", "hard", "extreme", "unknown")
SCS_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("near preference", ("near_preference",)),
    ("far preference", ("far_preference",)),
    ("relative preference", ("relative_preference",)),
    (
        "path-shape preference",
        ("geometric_path_preference", "path_shape_preference"),
    ),
    ("clearance", ("clearance",)),
)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def display_path(path: Path) -> str:
    if path.is_relative_to(REPO_ROOT):
        return path.relative_to(REPO_ROOT).as_posix()
    return str(path)


def output_path_for_prediction(
    output_root: Path,
    map_id: str,
    instruction_id: str,
) -> Path:
    return output_root / normalize_map_key(map_id) / f"{instruction_id}.json"


def difficulty_from_instruction(instruction: Mapping[str, object]) -> str:
    difficulty = instruction.get("difficulty_level")
    if isinstance(difficulty, str) and difficulty.strip():
        return difficulty.strip().lower()
    return "unknown"


def constraint_center(constraint: Mapping[str, object]) -> tuple[float, float]:
    center = constraint.get("center")
    if (
        isinstance(center, Sequence)
        and not isinstance(center, (str, bytes))
        and len(center) == 2
    ):
        return float(center[0]), float(center[1])

    cells = constraint.get("cells")
    if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes)):
        rows = [float(cell[0]) for cell in cells if isinstance(cell, Sequence)]
        cols = [float(cell[1]) for cell in cells if isinstance(cell, Sequence)]
        if rows and cols:
            return sum(rows) / len(rows), sum(cols) / len(cols)
    return 0.0, 0.0


def candidate_cells_for_constraint(
    constraint: Mapping[str, object],
    grid_size: int,
) -> list[tuple[int, int]]:
    shape = constraint.get("shape")
    center_row, center_col = constraint_center(constraint)
    if shape == "freeform":
        cells = constraint.get("cells")
        if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes)):
            return sorted(
                {
                    (int(cell[0]), int(cell[1]))
                    for cell in cells
                    if isinstance(cell, Sequence)
                    and not isinstance(cell, (str, bytes))
                    and len(cell) == 2
                },
                key=lambda cell: math.hypot(
                    cell[0] - center_row,
                    cell[1] - center_col,
                ),
            )

    radius = float(constraint.get("radius", 1.0) or 1.0)
    width = float(constraint.get("width", radius * 2) or radius * 2)
    height = float(constraint.get("height", radius * 2) or radius * 2)
    if shape == "circle":
        row_min = max(0, math.floor(center_row - radius))
        row_max = min(grid_size - 1, math.ceil(center_row + radius))
        col_min = max(0, math.floor(center_col - radius))
        col_max = min(grid_size - 1, math.ceil(center_col + radius))
        cells = [
            (row, col)
            for row in range(row_min, row_max + 1)
            for col in range(col_min, col_max + 1)
            if math.hypot(row - center_row, col - center_col) <= radius
        ]
    else:
        row_min = max(0, math.floor(center_row - height / 2))
        row_max = min(grid_size - 1, math.ceil(center_row + height / 2))
        col_min = max(0, math.floor(center_col - width / 2))
        col_max = min(grid_size - 1, math.ceil(center_col + width / 2))
        cells = [
            (row, col)
            for row in range(row_min, row_max + 1)
            for col in range(col_min, col_max + 1)
        ]

    return sorted(
        cells,
        key=lambda cell: math.hypot(cell[0] - center_row, cell[1] - center_col),
    )


def nearest_reachable_cell(
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
        for row in range(
            max(0, center_row - radius),
            min(grid_size - 1, center_row + radius) + 1,
        ):
            candidates.append((row, max(0, center_col - radius)))
            candidates.append((row, min(grid_size - 1, center_col + radius)))
        for col in range(
            max(0, center_col - radius + 1),
            min(grid_size - 1, center_col + radius - 1) + 1,
        ):
            candidates.append((max(0, center_row - radius), col))
            candidates.append((min(grid_size - 1, center_row + radius), col))

        for row, col in sorted(
            set(candidates),
            key=lambda cell: math.hypot(cell[0] - seed[0], cell[1] - seed[1]),
        ):
            if is_traversable(traversable, row, col):
                return row, col
    return None


def target_for_constraint(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    constraint: Mapping[str, object],
) -> tuple[int, int] | None:
    grid_size = int(map_state["grid_size"])
    for row, col in candidate_cells_for_constraint(constraint, grid_size):
        if is_traversable(traversable, row, col):
            return row, col
    return nearest_reachable_cell(traversable, constraint_center(constraint))


def target_cells_for_constraint(
    traversable: TraversableGrid,
    constraint: Mapping[str, object],
) -> list[tuple[int, int]]:
    """Return every legal cell in a hard target region, not one centre point."""
    grid_size = len(traversable)
    return [
        cell
        for cell in candidate_cells_for_constraint(constraint, grid_size)
        if is_traversable(traversable, *cell)
        and point_inside_constraint(cell, constraint)
    ]


def active_must_avoids(
    hard_constraints: Sequence[Mapping[str, object]],
    previous_goal: Mapping[str, object] | None,
    expected_goal: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """Return must-avoid regions active for one ordered hard transition."""
    expected_order = expected_goal.get("order")
    previous_order = previous_goal.get("order") if previous_goal is not None else None
    active: list[Mapping[str, object]] = []
    for constraint in hard_constraints:
        if constraint.get("kind") != "must_avoid":
            continue
        scope = constraint.get("scope")
        if not isinstance(scope, Mapping) or scope.get("type") != "between_hard_constraints":
            active.append(constraint)
            continue
        to_order = scope.get("to_order")
        from_order = scope.get("from_order")
        if isinstance(to_order, Real) and not isinstance(to_order, bool):
            if not (
                isinstance(expected_order, Real)
                and not isinstance(expected_order, bool)
                and float(to_order) == float(expected_order)
            ):
                continue
        if from_order is None:
            if previous_goal is not None:
                continue
        elif isinstance(from_order, Real) and not isinstance(from_order, bool):
            if not (
                isinstance(previous_order, Real)
                and not isinstance(previous_order, bool)
                and float(from_order) == float(previous_order)
            ):
                continue
        active.append(constraint)
    return active


def block_must_avoid_cells(
    traversable: TraversableGrid,
    avoids: Sequence[Mapping[str, object]],
) -> TraversableGrid:
    """Mask avoid-region cell centers before selecting a segment target.

    Freeform regions can contain thousands of grid cells.  Looking through that
    list for every map cell makes a single A* setup quadratic in map size and
    region size, so they are compiled into a set for O(1) membership checks.
    """
    freeform_cells = freeform_avoid_cells(avoids)
    geometric_avoids = [avoid for avoid in avoids if avoid.get("shape") != "freeform"]
    return [
        [
            traversable[row][col]
            and (row, col) not in freeform_cells
            and not any(
                point_inside_constraint((row, col), avoid)
                for avoid in geometric_avoids
            )
            for col in range(len(traversable[row]))
        ]
        for row in range(len(traversable))
    ]


def freeform_avoid_cells(
    avoids: Sequence[Mapping[str, object]],
) -> set[tuple[int, int]]:
    """Return the union of freeform avoid cells as a constant-time lookup set."""
    cells: set[tuple[int, int]] = set()
    for avoid in avoids:
        if avoid.get("shape") != "freeform":
            continue
        raw_cells = avoid.get("cells")
        if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
            continue
        for cell in raw_cells:
            if (
                isinstance(cell, Sequence)
                and not isinstance(cell, (str, bytes))
                and len(cell) == 2
            ):
                cells.add((int(cell[0]), int(cell[1])))
    return cells


def segment_intersects_freeform_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    cells: set[tuple[int, int]],
) -> bool:
    """Exactly test only freeform cells geometrically near one A* edge."""
    if not cells:
        return False
    row_min = math.ceil(min(start[0], end[0]) - 0.5)
    row_max = math.floor(max(start[0], end[0]) + 0.5)
    col_min = math.ceil(min(start[1], end[1]) - 0.5)
    col_max = math.floor(max(start[1], end[1]) + 0.5)
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            if (row, col) not in cells:
                continue
            # Reuse the evaluator's exact grid-cell intersection semantics;
            # only the candidate list is reduced from every region cell to a
            # handful adjacent to this unit-length A* edge.
            if segment_intersects_constraint(
                start,
                end,
                {"shape": "freeform", "cells": [[row, col]]},
            ):
                return True
    return False


def start_cell_for_instruction(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    instruction: Mapping[str, object],
) -> tuple[int, int] | None:
    start_pose = instruction.get("start_pose")
    if isinstance(start_pose, Mapping):
        row = start_pose.get("row")
        col = start_pose.get("col")
        if isinstance(row, int) and isinstance(col, int):
            if is_traversable(traversable, row, col):
                return row, col
            return nearest_reachable_cell(traversable, (row, col))

    hard_constraints = instruction.get("hard_constraints", [])
    if isinstance(hard_constraints, Sequence) and not isinstance(
        hard_constraints,
        (str, bytes),
    ):
        must_pass = ordered_must_pass_constraints(hard_constraints)  # type: ignore[arg-type]
        if must_pass:
            return target_for_constraint(map_state, traversable, must_pass[0])
    return None


def tutorial_waypoints(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    instruction: Mapping[str, object],
) -> tuple[
    list[tuple[int, int]],
    list[dict[str, object]],
    list[tuple[Mapping[str, object] | None, Mapping[str, object]]],
]:
    hard_constraints = instruction.get("hard_constraints", [])
    if not isinstance(hard_constraints, Sequence) or isinstance(
        hard_constraints,
        (str, bytes),
    ):
        hard_constraints = []
    must_pass = ordered_must_pass_constraints(hard_constraints)  # type: ignore[arg-type]

    start = start_cell_for_instruction(map_state, traversable, instruction)
    if start is None:
        return [], [{"status": "failed", "reason": "no_traversable_start"}], []

    # HCS treats the first must-pass region as the record/start region and
    # scores transitions beginning with record -> order 2.  Do not move to an
    # arbitrary point inside that record region: the start can already satisfy
    # it, and such a detour can violate the next transition's must-avoid area.
    waypoints = [start]
    waypoint_details: list[dict[str, object]] = [
        {"kind": "start", "row": start[0], "col": start[1]}
    ]
    segment_transitions: list[
        tuple[Mapping[str, object] | None, Mapping[str, object]]
    ] = []
    if not must_pass:
        return waypoints, waypoint_details, segment_transitions

    record_goal = must_pass[0]
    waypoint_details.append(
        {
            "kind": "must_pass",
            "constraint_id": record_goal.get("constraint_id"),
            "order": record_goal.get("order"),
            "status": "record_region",
        }
    )
    previous_goal: Mapping[str, object] = record_goal
    for constraint in must_pass[1:]:
        avoids = active_must_avoids(hard_constraints, previous_goal, constraint)
        segment_transitions.append((previous_goal, constraint))
        waypoint_details.append(
            {
                "kind": "must_pass",
                "constraint_id": constraint.get("constraint_id"),
                "order": constraint.get("order"),
                "status": "planned_to_any_legal_region_cell",
                "active_must_avoid_ids": [
                    avoid.get("constraint_id") for avoid in avoids
                ],
            }
        )
        previous_goal = constraint
    return waypoints, waypoint_details, segment_transitions


def connect_waypoints_with_astar(
    traversable: TraversableGrid,
    waypoints: Sequence[tuple[int, int]],
    hard_constraints: Sequence[Mapping[str, object]],
    segment_transitions: Sequence[
        tuple[Mapping[str, object] | None, Mapping[str, object]]
    ],
) -> tuple[list[list[int]], list[dict[str, object]]]:
    if not waypoints:
        return [], []

    trajectory = [[waypoints[0][0], waypoints[0][1]]]
    segment_details: list[dict[str, object]] = []
    current_cell = waypoints[0]
    for index, (previous_goal, expected_goal) in enumerate(segment_transitions):
        avoids = active_must_avoids(hard_constraints, previous_goal, expected_goal)
        segment_traversable = block_must_avoid_cells(traversable, avoids)
        freeform_cells = freeform_avoid_cells(avoids)
        geometric_avoids = [
            avoid for avoid in avoids if avoid.get("shape") != "freeform"
        ]

        def edge_allowed(
            start: tuple[int, int],
            end: tuple[int, int],
        ) -> bool:
            if segment_intersects_freeform_cells(start, end, freeform_cells):
                return False
            return not any(
                segment_intersects_constraint(start, end, avoid)
                for avoid in geometric_avoids
            )

        target_cells = target_cells_for_constraint(segment_traversable, expected_goal)
        segment = astar_path_to_any(
            segment_traversable,
            current_cell,
            target_cells,
            edge_allowed=edge_allowed,
        )
        if not segment:
            segment_details.append(
                {
                    "kind": "segment",
                    "from": list(current_cell),
                    "to_constraint_id": expected_goal.get("constraint_id"),
                    "status": "failed_no_path",
                    "active_must_avoid_ids": [
                        avoid.get("constraint_id") for avoid in avoids
                    ],
                }
            )
            # A later segment would start at a waypoint the trajectory never
            # reached, creating a non-physical jump that can cross must-avoid.
            break
        trajectory.extend(segment[1:])
        current_cell = (segment[-1][0], segment[-1][1])
        segment_details.append(
            {
                "kind": "segment",
                "from": list(segment[0]),
                "to": list(current_cell),
                "status": "connected",
                "waypoint_count": len(segment),
                "active_must_avoid_ids": [
                    avoid.get("constraint_id") for avoid in avoids
                ],
            }
        )
    return trajectory, segment_details


def metric_value(metrics: Mapping[str, object] | None, key: str) -> float | None:
    if metrics is None:
        return None
    value = metrics.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def metric_summary(metrics: Mapping[str, object] | None) -> dict[str, float | None]:
    return {
        "HCS": metric_value(metrics, "HCS"),
        "SCS": metric_value(metrics, "SCS"),
        "PL": metric_value(metrics, "PL"),
        "segment_wise_SPL": metric_value(metrics, "segment_wise_SPL"),
    }


def metrics_from_record(record: Mapping[str, object]) -> Mapping[str, object] | None:
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics
    return None


def difficulty_from_record(record: Mapping[str, object]) -> str:
    difficulty = record.get("difficulty_level")
    if isinstance(difficulty, str) and difficulty.strip():
        return difficulty.strip().lower()

    instruction = record.get("instruction")
    if isinstance(instruction, Mapping):
        return difficulty_from_instruction(instruction)
    return "unknown"


def load_existing_result(
    output_path: Path,
    map_id: str,
    instruction_id: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "instruction_id": instruction_id,
        "map_id": map_id,
        "status": "skipped_exists",
        "path": display_path(output_path),
    }
    try:
        record = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    if not isinstance(record, Mapping):
        return result

    metrics = metrics_from_record(record)
    trajectory = record.get("trajectory")
    result["difficulty_level"] = difficulty_from_record(record)
    if isinstance(trajectory, list):
        result["trajectory_length"] = len(trajectory)
    if metrics is not None:
        result["metrics"] = metrics
        result["metrics_summary"] = metric_summary(metrics)
    runtime_value = runtime_seconds(record)
    if runtime_value is not None:
        result["runtime_seconds"] = runtime_value
    runtime = record.get("runtime")
    if isinstance(runtime, Mapping):
        result["runtime"] = dict(runtime)
    return result


def collect_prediction_results(
    output_root: Path,
    *,
    method_name: str | None = None,
) -> list[dict[str, object]]:
    """Load every saved prediction below one method output directory.

    It lets an aggregate summary include results produced by earlier partial
    runs, while resume skips remain inexpensive during the run itself.
    """
    if not output_root.exists():
        return []

    results: list[dict[str, object]] = []
    for path in sorted(output_root.rglob("instruction_*.json")):
        # Intermediate artifacts use names such as instruction_000001.steps.json
        # and are not prediction records. In keyed running summaries they can
        # otherwise overwrite the real prediction for the same episode.
        if path.name.endswith(".steps.json"):
            continue
        relative_parts = path.relative_to(output_root).parts
        if any(part.startswith("_") for part in relative_parts):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, Mapping):
            continue
        saved_method = record.get("method")
        if (
            method_name is not None
            and isinstance(saved_method, str)
            and saved_method != method_name
        ):
            continue
        map_id = record.get("map_id")
        instruction_id = record.get("instruction_id")
        if not isinstance(map_id, str) or not isinstance(instruction_id, str):
            continue
        # Prediction records are stored at <output-root>/<map-id>/<instruction>.json.
        # Do not descend into sibling ablation outputs or cache directories when
        # the caller intentionally uses a method root as its output root.
        if relative_parts[:-1] != Path(map_id).parts:
            continue
        result = load_existing_result(path, map_id, instruction_id)
        result["scene_id"] = record.get("scene_id", map_id)
        results.append(result)
    return results


def write_prediction_record(
    output_path: Path,
    record: Mapping[str, object],
) -> None:
    payload = record_with_trajectory_last(record)
    write_json_atomic(output_path, payload)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write one JSON object without exposing a partial destination file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def record_with_trajectory_last(record: Mapping[str, object]) -> dict[str, object]:
    if "trajectory" not in record:
        return dict(record)
    payload = {key: value for key, value in record.items() if key != "trajectory"}
    payload["trajectory"] = record["trajectory"]
    return payload


def result_from_prediction_record(
    record: Mapping[str, object],
    output_path: Path,
) -> dict[str, object]:
    trajectory = record.get("trajectory")
    return {
        "instruction_id": record["instruction_id"],
        "map_id": record["map_id"],
        "scene_id": record.get("scene_id", record["map_id"]),
        "status": "written",
        "path": display_path(output_path),
        "difficulty_level": record["difficulty_level"],
        "trajectory_length": len(trajectory) if isinstance(trajectory, list) else 0,
        "metrics": record.get("metrics"),
        "metrics_summary": record.get("metrics_summary"),
        "trajectory_image": record.get("trajectory_image"),
        "trajectory_with_gt_image": record.get("trajectory_with_gt_image"),
        "runtime_seconds": record.get("runtime_seconds"),
        "runtime": record.get("runtime"),
    }


def format_metric(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return "N/A"


def format_metres(value: object) -> str:
    formatted = format_metric(value)
    return f"{formatted}m" if formatted != "N/A" else formatted


def print_compact_metric(
    label: str,
    metric: Mapping[str, object],
    *,
    runtime: float | None = None,
) -> None:
    """Print one metric block in two concise, stable lines."""
    path_efficiency = metric.get("path efficiency")
    if not isinstance(path_efficiency, Mapping):
        path_efficiency = {}
    scs = metric.get("SCS")
    if not isinstance(scs, Mapping):
        scs = {}
    runtime_text = f" time={format_metric(runtime)}s" if runtime is not None else ""
    print(
        f"{label} HCS={format_metric(metric.get('HCS'))} "
        f"PL={format_metres(path_efficiency.get('PL'))} "
        f"H-SPL={format_metric(path_efficiency.get('H-SPL'))}{runtime_text}",
        flush=True,
    )
    print(
        f"  soft: near={format_metres(scs.get('near preference'))} "
        f"far={format_metres(scs.get('far preference'))} "
        f"relative={format_metric(scs.get('relative preference'))} "
        f"path-shape={format_metric(scs.get('path-shape preference'))} "
        f"clearance={format_metres(scs.get('clearance'))}",
        flush=True,
    )


def print_episode_result(index: int, total: int, result: Mapping[str, object]) -> None:
    instruction_id = result.get("instruction_id", "unknown")
    scene_id = result.get("scene_id", result.get("map_id", "unknown"))
    status = result.get("status", "unknown")
    if status == "failed":
        print(
            f"[{index}/{total}] scene_id={scene_id} instruction_id={instruction_id} failed",
            flush=True,
        )
        error = result.get("error")
        if isinstance(error, str) and error.strip():
            print(f"  error: {error.strip()}", flush=True)
        return

    metric = metric_block([dict(result)])
    # ``load_existing_result`` preserves the original run time for aggregate
    # runtime reporting. It is not the time spent by the current skip.
    runtime = None if status == "skipped_exists" else runtime_seconds(result)
    print_compact_metric(
        f"[{index}/{total}] scene_id={scene_id} instruction_id={instruction_id} {status}",
        metric,
        runtime=runtime,
    )


def mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    if not collected:
        return None
    return sum(collected) / len(collected)


def result_metrics(result: Mapping[str, object]) -> Mapping[str, object] | None:
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics
    return None


def runtime_seconds(result: Mapping[str, object]) -> float | None:
    value = result.get("runtime_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    runtime = result.get("runtime")
    if isinstance(runtime, Mapping):
        nested_value = runtime.get("seconds")
        if isinstance(nested_value, (int, float)) and not isinstance(
            nested_value,
            bool,
        ):
            return float(nested_value)
    return None


def runtime_seconds_from_prediction_file(result: Mapping[str, object]) -> float | None:
    raw_path = result.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, Mapping):
        return None
    value = record.get("runtime_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def runtime_breakdown_from_prediction_file(
    result: Mapping[str, object],
) -> dict[str, float]:
    raw_path = result.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {}
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(record, Mapping):
        return {}
    runtime = record.get("runtime")
    if not isinstance(runtime, Mapping):
        return {}
    raw_breakdown = runtime.get("breakdown_seconds")
    if not isinstance(raw_breakdown, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in raw_breakdown.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def runtime_method_seconds_from_prediction_file(
    result: Mapping[str, object],
) -> float | None:
    raw_path = result.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, Mapping):
        return None
    runtime = record.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    value = runtime.get("method_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def runtime_seconds_for_summary(result: Mapping[str, object]) -> float | None:
    value = runtime_seconds(result)
    return value if value is not None else runtime_seconds_from_prediction_file(result)


def runtime_method_seconds_for_summary(result: Mapping[str, object]) -> float | None:
    runtime = result.get("runtime")
    if isinstance(runtime, Mapping):
        value = runtime.get("method_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return runtime_method_seconds_from_prediction_file(result)


def runtime_breakdown_for_summary(result: Mapping[str, object]) -> dict[str, float]:
    runtime = result.get("runtime")
    if isinstance(runtime, Mapping):
        raw_breakdown = runtime.get("breakdown_seconds")
        if isinstance(raw_breakdown, Mapping):
            return {
                str(key): float(value)
                for key, value in raw_breakdown.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
    return runtime_breakdown_from_prediction_file(result)


def runtime_summary(results: Iterable[Mapping[str, object]]) -> dict[str, object]:
    materialized_results = list(results)
    values = [
        value
        for result in materialized_results
        if (value := runtime_seconds_for_summary(result)) is not None
    ]
    method_values = [
        value
        for result in materialized_results
        if (value := runtime_method_seconds_for_summary(result)) is not None
    ]
    breakdown_values: dict[str, list[float]] = {}
    for result in materialized_results:
        for key, value in runtime_breakdown_for_summary(result).items():
            breakdown_values.setdefault(key, []).append(value)
    if breakdown_values:
        if not method_values:
            method_values = breakdown_values.get("trajectory_construction_seconds", [])
        metric_values = breakdown_values.get("metric_seconds", [])
        other_values = [
            total - method - metric
            for total, method, metric in zip(values, method_values, metric_values)
        ]
        return {
            "average_total_seconds": mean(values),
            "average_method_seconds": mean(method_values),
            "average_metric_seconds": mean(metric_values),
            "average_other_seconds": mean(other_values),
        }
    return {
        "average_total_seconds": mean(values),
        "average_method_seconds": mean(method_values),
    }


def metric_values(results: Iterable[Mapping[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for result in results:
        metrics = result_metrics(result)
        value = metric_value(metrics, key)
        if value is None:
            summary = result.get("metrics_summary")
            if isinstance(summary, Mapping):
                raw_value = summary.get(key)
                if isinstance(raw_value, (int, float)) and not isinstance(
                    raw_value,
                    bool,
                ):
                    value = float(raw_value)
        if value is not None:
            values.append(value)
    return values


def raw_values_for_preference(
    metrics: Mapping[str, object],
    preference_types: Sequence[str],
) -> list[float]:
    """Return numeric raw measurements for one displayed SCS preference."""
    details = metrics.get("details")
    if not isinstance(details, Mapping):
        return []
    soft_constraints = details.get("soft_constraints")
    if not isinstance(soft_constraints, list):
        return []

    values: list[float] = []
    accepted = set(preference_types)
    for detail in soft_constraints:
        if not isinstance(detail, Mapping):
            continue
        preference_type = detail.get("preference_type")
        source_preference_type = detail.get("source_preference_type")
        if (
            preference_type not in accepted
            and source_preference_type not in accepted
        ):
            continue
        raw_value = detail.get("raw_value")
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            raw_value = float(raw_value)
            if math.isfinite(raw_value):
                values.append(raw_value)
    return values


def scs_detail_averages(
    results: Iterable[Mapping[str, object]],
) -> dict[str, float | None]:
    """Average each raw preference value once per episode, then globally."""
    averages: dict[str, float | None] = {}
    materialized_results = list(results)
    for display_name, preference_types in SCS_FIELDS:
        episode_values: list[float] = []
        for result in materialized_results:
            metrics = result_metrics(result)
            if metrics is None:
                continue
            values = raw_values_for_preference(metrics, preference_types)
            if values:
                episode_values.append(sum(values) / len(values))
        averages[display_name] = mean(episode_values)
    return averages


def metric_block(results: list[Mapping[str, object]]) -> dict[str, object]:
    """Format metrics using the stable Tutorial summary contract."""
    return {
        "HCS": mean(metric_values(results, "HCS")),
        "SCS": scs_detail_averages(results),
        "path efficiency": {
            "PL": mean(metric_values(results, "PL")),
            "H-SPL": mean(metric_values(results, "segment_wise_SPL")),
        },
    }


def score_block(
    results: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "record_count": len(results),
        "finished_instruction_count": len(
            {
                (result.get("map_id"), result.get("instruction_id"))
                for result in results
            }
        ),
        "metric": metric_block(results),
    }


def build_summary(
    results: list[dict[str, object]],
    input_root: Path,
    output_root: Path,
    *,
    method_name: str,
    evaluate: bool,
) -> dict[str, object]:
    scored_results = [
        result
        for result in results
        if result.get("status") in {"written", "skipped_exists"}
        and isinstance(result.get("metrics_summary"), Mapping)
    ]
    by_difficulty = {
        difficulty: [
            result
            for result in scored_results
            if str(result.get("difficulty_level") or "unknown").lower() == difficulty
        ]
        for difficulty in DIFFICULTY_LEVELS
    }
    return {
        "version": 1,
        "method": method_name,
        "overall": score_block(scored_results),
        **{
            difficulty: score_block(by_difficulty[difficulty])
            for difficulty in ("easy", "hard")
        },
        "runtime_summary": runtime_summary(results),
    }


def write_summary(
    results: list[dict[str, object]],
    input_root: Path,
    output_root: Path,
    *,
    method_name: str,
    evaluate: bool,
) -> Path:
    summary = build_summary(
        results,
        input_root,
        output_root,
        method_name=method_name,
        evaluate=evaluate,
    )
    summary_path = output_root / "summary.json"
    write_json_atomic(summary_path, summary)
    return summary_path


def print_summary(summary_path: Path) -> None:
    """Print saved aggregate metric blocks in the same compact two-line form."""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = None
    if isinstance(summary, Mapping):
        for scope in ("overall", "easy", "hard"):
            block = summary.get(scope)
            if not isinstance(block, Mapping):
                continue
            metric = block.get("metric")
            if not isinstance(metric, Mapping):
                continue
            count = block.get("record_count")
            print_compact_metric(f"{scope} n={count}", metric)
    print(display_path(summary_path), flush=True)
