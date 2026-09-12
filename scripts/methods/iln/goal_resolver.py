"""Resolve grounded ILN stage targets to traversable endpoint cells."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from scripts.methods.iln.graph_builder import _layer, _value_at
from scripts.methods.iln.graph_types import AreaPassageGraph, Cell, GroundedStage, StageGoal
from scripts.methods.util.grid_astar import TraversableGrid, astar_path, is_traversable, nearest_traversable_cell
from scripts.methods.util.map_objects import (
    object_cells as rich_object_cells,
    object_room_id as rich_object_room_id,
)


def _object_cells(map_state: Mapping[str, object], object_id: int) -> list[Cell]:
    return list(rich_object_cells(map_state, object_id))


def _object_room_from_cells(map_state: Mapping[str, object], object_id: int) -> int | None:
    rich_room_id = rich_object_room_id(map_state, object_id)
    if rich_room_id is not None:
        return rich_room_id
    cells = _object_cells(map_state, object_id)
    if not cells:
        return None
    room_layer = _layer(map_state, "room")
    center = (
        round(sum(row for row, _col in cells) / len(cells)),
        round(sum(col for _row, col in cells) / len(cells)),
    )
    room_id = _value_at(room_layer, int(center[0]), int(center[1]))
    return room_id if room_id > 0 else None


def _approach_candidates(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    *,
    object_id: int,
    room_id: int,
    radius: int = 8,
) -> list[Cell]:
    object_cells = _object_cells(map_state, object_id)
    if not object_cells:
        return []
    room_layer = _layer(map_state, "room")
    candidates: set[Cell] = set()
    for row, col in object_cells:
        for rr in range(max(0, row - radius), min(len(traversable) - 1, row + radius) + 1):
            for cc in range(max(0, col - radius), min(len(traversable) - 1, col + radius) + 1):
                if not is_traversable(traversable, rr, cc):
                    continue
                if _value_at(room_layer, rr, cc) != room_id:
                    continue
                candidates.add((rr, cc))
    center = (
        sum(row for row, _col in object_cells) / len(object_cells),
        sum(col for _row, col in object_cells) / len(object_cells),
    )
    return sorted(candidates, key=lambda cell: math.hypot(cell[0] - center[0], cell[1] - center[1]))


def _reachable_candidate(
    traversable: TraversableGrid,
    start: Cell | None,
    candidates: Sequence[Cell],
) -> Cell | None:
    if not candidates:
        return None
    if start is None:
        return candidates[0]
    for candidate in sorted(candidates, key=lambda cell: math.hypot(cell[0] - start[0], cell[1] - start[1]))[:80]:
        path = astar_path(traversable, start, candidate)
        if path:
            return candidate
    return candidates[0]


def resolve_stage_goal(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    graph: AreaPassageGraph,
    stage: GroundedStage,
    *,
    current_cell: Cell | None = None,
    allow_grid_fallback: bool = False,
) -> StageGoal:
    resolved_area_id = stage.area_id
    area_correction: dict[str, object] = {}
    if stage.target_type == "object" and stage.object_id is not None:
        object_room = _object_room_from_cells(map_state, stage.object_id)
        if object_room is not None and object_room in graph.areas and object_room != stage.area_id:
            area_correction = {
                "corrected_area_id": f"room_{object_room}",
                "original_area_id": f"room_{stage.area_id}",
                "reason": "object_instance_layer_room_lookup",
            }
            resolved_area_id = object_room
    area = graph.areas.get(resolved_area_id)
    if area is None:
        return StageGoal(stage, None, 0, "NO_GOAL_REGION", {"reason": "unknown_area"})

    if stage.target_type == "object" and stage.object_id is not None:
        candidates = _approach_candidates(
            map_state,
            traversable,
            object_id=stage.object_id,
            room_id=resolved_area_id,
        )
        endpoint = _reachable_candidate(traversable, current_cell, candidates)
        if endpoint is not None:
            return StageGoal(
                stage,
                endpoint,
                len(candidates),
                "resolved_object_approach",
                {"target_area_medoid": list(area.medoid), **area_correction},
            )
        if allow_grid_fallback:
            fallback = nearest_traversable_cell(traversable, area.medoid, max_radius=80)
            return StageGoal(
                stage,
                fallback,
                0,
                "OBJECT_APPROACH_FALLBACK_USED" if fallback else "NO_GOAL_REGION",
                {"fallback": "target_area_medoid"},
            )
        return StageGoal(stage, None, 0, "NO_GOAL_REGION", {"reason": "no_object_approach_cell"})

    endpoint = nearest_traversable_cell(traversable, area.medoid, max_radius=80)
    return StageGoal(
        stage,
        endpoint,
        1 if endpoint else 0,
        "resolved_area_endpoint" if endpoint else "NO_GOAL_REGION",
        {"target_area_medoid": list(area.medoid)},
    )
