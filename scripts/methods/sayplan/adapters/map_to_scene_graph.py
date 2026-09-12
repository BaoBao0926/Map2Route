"""Convert SemPathBench layered maps into SayPlan scene graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scripts.methods.util.map_objects import (
    object_cells as rich_object_cells,
    object_room_id as rich_object_room_id,
)
from scripts.methods.util.grid_astar import TraversableGrid
from scripts.methods.sayplan.graph.scene_graph import GraphNode, SceneGraph
from scripts.methods.sayplan.utils.geometry import Cell, centroid
from scripts.methods.sayplan.utils.grid import (
    approach_cells_for_mask,
    assign_room_for_object,
    cells_for_value,
    layer,
    layer_value,
    target_for_room,
)


def _as_instance_list(raw: object) -> list[Mapping[str, object]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _attributes(item: Mapping[str, object]) -> list[str]:
    raw = item.get("attributes", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [str(value) for value in raw if isinstance(value, str) and value.strip()]


def _name(item: Mapping[str, object], fallback: str) -> str:
    raw = item.get("name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return fallback


def _category(item: Mapping[str, object], fallback: str) -> str:
    raw = item.get("category")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return fallback


def _instance_id(item: Mapping[str, object]) -> int | None:
    raw = item.get("id")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    return None


def build_scene_graph(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    start_cell: Cell | None,
) -> SceneGraph:
    graph = SceneGraph()
    graph.add_node(
        GraphNode(
            id="scene_0",
            type="scene",
            name=str(map_state.get("map_key") or "scene_0"),
            metadata={"child_count": 0},
        )
    )
    graph.add_node(
        GraphNode(
            id="agent_0",
            type="agent",
            name="agent_0",
            position=start_cell,
            metadata={"start_cell": list(start_cell) if start_cell else None},
        )
    )
    graph.add_edge("scene_0", "agent_0", "contains")

    room_grid = layer(map_state, "room")
    room_instances = _as_instance_list(map_state.get("room_instances"))
    object_instances = _as_instance_list(map_state.get("object_instances"))

    room_cells_by_id: dict[int, list[Cell]] = {}
    room_position_by_raw_id: dict[int, Cell] = {}
    for item in room_instances:
        raw_id = _instance_id(item)
        if raw_id is None:
            continue
        node_id = f"room_{raw_id}"
        cells = cells_for_value(room_grid, raw_id)
        room_cells_by_id[raw_id] = cells
        position = centroid(cells)
        if position is not None:
            room_position_by_raw_id[raw_id] = position
        target = target_for_room(traversable, cells, position)
        node = GraphNode(
            id=node_id,
            type="room",
            category=_category(item, "room"),
            name=_name(item, node_id),
            position=position,
            attributes=_attributes(item),
            metadata={
                "raw_id": raw_id,
                "cells": cells,
                "navigation_target": target,
                "child_count": 0,
            },
        )
        graph.add_node(node)
        graph.add_edge("scene_0", node_id, "contains")

    for item in object_instances:
        raw_id = _instance_id(item)
        if raw_id is None:
            continue
        node_id = f"object_{raw_id}"
        cells = list(rich_object_cells(map_state, raw_id))
        position = centroid(cells)
        approach_cells = approach_cells_for_mask(traversable, cells)
        room_raw_id = rich_object_room_id(map_state, raw_id)
        assignment_method = "rich_object_localization"
        if room_raw_id is None:
            room_raw_id, assignment_method = assign_room_for_object(
                room_grid,
                cells,
                position,
                room_position_by_raw_id,
            )
        room_id = f"room_{room_raw_id}" if room_raw_id is not None else None
        node = GraphNode(
            id=node_id,
            type="object",
            category=_category(item, "object"),
            name=_name(item, node_id),
            position=position,
            room_id=room_id,
            attributes=_attributes(item),
            metadata={
                "raw_id": raw_id,
                "cells": cells,
                "approach_cells": approach_cells,
                "room_assignment_method": assignment_method,
            },
        )
        graph.add_node(node)
        parent = room_id if room_id and graph.has_node(room_id) else "scene_0"
        graph.add_edge(parent, node_id, "contains")

    _add_room_adjacency(graph, room_grid)

    for node_id, node in graph.nodes.items():
        node.metadata["child_count"] = len(graph.children(node_id))
    return graph


def _add_room_adjacency(graph: SceneGraph, room_grid: Sequence[Sequence[object]]) -> None:
    seen: set[tuple[int, int]] = set()
    for row, row_values in enumerate(room_grid):
        if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
            continue
        for col, _value in enumerate(row_values):
            first = layer_value(room_grid, row, col)
            if first <= 0:
                continue
            for delta_row, delta_col in ((1, 0), (0, 1)):
                second = layer_value(room_grid, row + delta_row, col + delta_col)
                if second <= 0 or second == first:
                    continue
                pair = tuple(sorted((first, second)))
                if pair in seen:
                    continue
                first_id = f"room_{pair[0]}"
                second_id = f"room_{pair[1]}"
                if graph.has_node(first_id) and graph.has_node(second_id):
                    graph.add_edge(first_id, second_id, "adjacent_to")
                    graph.add_edge(second_id, first_id, "adjacent_to")
                    seen.add(pair)
