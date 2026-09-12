"""Build an ILN-style area-passage graph from SemPathBench map layers."""

from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from scripts.methods.iln.graph_types import Area, AreaPassageGraph, Cell, Passage
from scripts.methods.util.grid_astar import (
    TraversableGrid,
    build_traversable_grid,
    is_traversable,
    nearest_traversable_cell,
)


def _layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    raw = layers.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return raw  # type: ignore[return-value]


def _instances(map_state: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    raw = map_state.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _clean_category(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().lower().replace(" ", "_")
    return fallback


def _value_at(layer: Sequence[Sequence[object]], row: int, col: int) -> int:
    if row < 0 or row >= len(layer):
        return 0
    row_values = layer[row]
    if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
        return 0
    if col < 0 or col >= len(row_values):
        return 0
    try:
        return int(row_values[col])
    except (TypeError, ValueError):
        return 0


def _start_cell(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    instruction: Mapping[str, object],
) -> Cell | None:
    start_pose = instruction.get("start_pose")
    if isinstance(start_pose, Mapping):
        row = start_pose.get("row")
        col = start_pose.get("col")
        if isinstance(row, int) and isinstance(col, int):
            if is_traversable(traversable, row, col):
                return row, col
            return nearest_traversable_cell(traversable, (row, col), max_radius=80)
    grid_size = int(map_state.get("grid_size", len(traversable)))
    return nearest_traversable_cell(traversable, (grid_size / 2, grid_size / 2), max_radius=grid_size)


def _nearest_room_for_cell(
    room_layer: Sequence[Sequence[object]],
    traversable: TraversableGrid,
    cell: Cell,
    *,
    max_radius: int = 120,
) -> tuple[int | None, dict[str, object]]:
    row, col = cell
    direct = _value_at(room_layer, row, col)
    if direct > 0:
        return direct, {"mode": "direct_room_label", "cell": [row, col]}
    grid_size = len(traversable)
    for radius in range(1, max_radius + 1):
        candidates: list[Cell] = []
        for rr in range(max(0, row - radius), min(grid_size - 1, row + radius) + 1):
            candidates.append((rr, max(0, col - radius)))
            candidates.append((rr, min(grid_size - 1, col + radius)))
        for cc in range(max(0, col - radius + 1), min(grid_size - 1, col + radius - 1) + 1):
            candidates.append((max(0, row - radius), cc))
            candidates.append((min(grid_size - 1, row + radius), cc))
        for candidate in sorted(set(candidates), key=lambda item: math.hypot(item[0] - row, item[1] - col)):
            room_id = _value_at(room_layer, candidate[0], candidate[1])
            if room_id > 0 and is_traversable(traversable, candidate[0], candidate[1]):
                return room_id, {
                    "mode": "nearest_traversable_room_label",
                    "cell": [candidate[0], candidate[1]],
                    "radius": radius,
                }
    return None, {"mode": "unresolved"}


def _room_cells(
    room_layer: Sequence[Sequence[object]],
    traversable: TraversableGrid,
    grid_size: int,
) -> dict[int, list[Cell]]:
    cells: dict[int, list[Cell]] = defaultdict(list)
    for row in range(grid_size):
        for col in range(grid_size):
            room_id = _value_at(room_layer, row, col)
            if room_id > 0 and is_traversable(traversable, row, col):
                cells[room_id].append((row, col))
    return dict(cells)


def _medoid(cells: Sequence[Cell]) -> Cell:
    avg_row = sum(row for row, _col in cells) / len(cells)
    avg_col = sum(col for _row, col in cells) / len(cells)
    return min(cells, key=lambda cell: math.hypot(cell[0] - avg_row, cell[1] - avg_col))


def _areas(map_state: Mapping[str, object], room_cells: Mapping[int, list[Cell]]) -> dict[int, Area]:
    instances = {int(item["id"]): item for item in _instances(map_state, "room_instances") if isinstance(item.get("id"), int)}
    areas: dict[int, Area] = {}
    for room_id, cells in sorted(room_cells.items()):
        if not cells:
            continue
        item = instances.get(room_id, {})
        centroid = (
            sum(row for row, _col in cells) / len(cells),
            sum(col for _row, col in cells) / len(cells),
        )
        category = _clean_category(item.get("category"), "room")
        areas[room_id] = Area(
            id=room_id,
            key=f"room_{room_id}",
            category=category,
            name=str(item.get("name") or f"{category}_{room_id}"),
            centroid=centroid,
            medoid=_medoid(cells),
            cell_count=len(cells),
        )
    return areas


def _transition_components(transitions: list[tuple[Cell, Cell]]) -> list[list[tuple[Cell, Cell]]]:
    if not transitions:
        return []
    by_endpoint: dict[Cell, list[int]] = defaultdict(list)
    for index, transition in enumerate(transitions):
        by_endpoint[transition[0]].append(index)
        by_endpoint[transition[1]].append(index)
    visited: set[int] = set()
    components: list[list[tuple[Cell, Cell]]] = []
    for start_index in range(len(transitions)):
        if start_index in visited:
            continue
        queue: deque[int] = deque([start_index])
        visited.add(start_index)
        component: list[tuple[Cell, Cell]] = []
        while queue:
            index = queue.popleft()
            transition = transitions[index]
            component.append(transition)
            endpoints = {transition[0], transition[1]}
            for endpoint in transition:
                row, col = endpoint
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        endpoints.add((row + delta_row, col + delta_col))
            for endpoint in endpoints:
                for neighbor_index in by_endpoint.get(endpoint, []):
                    if neighbor_index not in visited:
                        visited.add(neighbor_index)
                        queue.append(neighbor_index)
        components.append(component)
    return components


def _passage_anchor(
    component: Sequence[tuple[Cell, Cell]],
    traversable: TraversableGrid,
) -> Cell | None:
    points = [cell for transition in component for cell in transition if is_traversable(traversable, cell[0], cell[1])]
    if points:
        avg_row = sum(row for row, _col in points) / len(points)
        avg_col = sum(col for _row, col in points) / len(points)
        return min(points, key=lambda cell: math.hypot(cell[0] - avg_row, cell[1] - avg_col))
    midpoint = (
        sum((a[0] + b[0]) / 2 for a, b in component) / len(component),
        sum((a[1] + b[1]) / 2 for a, b in component) / len(component),
    )
    return nearest_traversable_cell(traversable, midpoint, max_radius=20)


def _passages(
    room_layer: Sequence[Sequence[object]],
    traversable: TraversableGrid,
    areas: Mapping[int, Area],
    grid_size: int,
) -> tuple[dict[str, Passage], dict[int, list[str]], dict[str, object]]:
    transitions_by_pair: dict[tuple[int, int], list[tuple[Cell, Cell]]] = defaultdict(list)
    for row in range(grid_size):
        for col in range(grid_size):
            room_id = _value_at(room_layer, row, col)
            if room_id <= 0 or room_id not in areas:
                continue
            for delta_row, delta_col in ((1, 0), (0, 1)):
                next_row = row + delta_row
                next_col = col + delta_col
                other_room = _value_at(room_layer, next_row, next_col)
                if other_room <= 0 or other_room == room_id or other_room not in areas:
                    continue
                if not (
                    is_traversable(traversable, row, col)
                    or is_traversable(traversable, next_row, next_col)
                ):
                    continue
                pair = tuple(sorted((room_id, other_room)))
                transitions_by_pair[pair].append(((row, col), (next_row, next_col)))

    passages: dict[str, Passage] = {}
    room_passages: dict[int, list[str]] = defaultdict(list)
    long_components: list[dict[str, object]] = []
    passage_index = 1
    for pair, transitions in sorted(transitions_by_pair.items()):
        for component in _transition_components(transitions):
            anchor = _passage_anchor(component, traversable)
            if anchor is None:
                continue
            passage_id = f"passage_{passage_index}"
            passage_index += 1
            passage = Passage(
                id=passage_id,
                rooms=pair,
                anchor=anchor,
                width=len({cell for transition in component for cell in transition}),
                component_size=len(component),
            )
            passages[passage.id] = passage
            room_passages[pair[0]].append(passage.id)
            room_passages[pair[1]].append(passage.id)
            if passage.component_size > 30:
                long_components.append(
                    {
                        "passage_id": passage.id,
                        "rooms": [pair[0], pair[1]],
                        "component_size": passage.component_size,
                    }
                )
    metadata = {
        "transition_pair_count": len(transitions_by_pair),
        "long_boundary_components": long_components,
    }
    return passages, {key: sorted(value) for key, value in room_passages.items()}, metadata


def graph_fingerprint(map_state: Mapping[str, object], passage_extraction: str) -> str:
    payload = {
        "map_key": map_state.get("map_key"),
        "grid_size": map_state.get("grid_size"),
        "passage_extraction": passage_extraction,
        "room_instances": map_state.get("room_instances"),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_area_passage_graph(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    *,
    passage_extraction: str = "boundary",
    graph_cache_root: Path | None = None,
) -> AreaPassageGraph:
    traversable = build_traversable_grid(map_state)
    grid_size = int(map_state["grid_size"])
    room_layer = _layer(map_state, "room")
    start = _start_cell(map_state, traversable, instruction)
    room_cells = _room_cells(room_layer, traversable, grid_size)
    areas = _areas(map_state, room_cells)
    passages, room_passages, passage_metadata = _passages(room_layer, traversable, areas, grid_size)
    start_area = None
    start_resolution: dict[str, object] = {"mode": "no_start_cell"}
    if start is not None:
        start_area, start_resolution = _nearest_room_for_cell(room_layer, traversable, start)
    metadata = {
        "passage_extraction": passage_extraction,
        "fingerprint": graph_fingerprint(map_state, passage_extraction),
        "start_area_resolution": start_resolution,
        "cache_root": str(graph_cache_root) if graph_cache_root else None,
        **passage_metadata,
    }
    return AreaPassageGraph(
        areas=areas,
        passages=passages,
        room_passages=room_passages,
        start_cell=start,
        start_area=start_area if start_area in areas else None,
        metadata=metadata,
    )

