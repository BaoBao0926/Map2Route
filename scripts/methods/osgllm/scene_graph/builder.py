"""Build OSG-LLM attribute regions from SemPathBench map_state objects."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import math

from scripts.methods.limp.utils.geometry import bbox, centroid, normalize_name
from scripts.methods.osgllm.scene_graph.graph_types import AttributeRegion, Cell, SceneGraph
from scripts.methods.util.grid_astar import TraversableGrid, is_traversable, nearest_traversable_cell
from scripts.methods.util.map_objects import (
    object_cells_by_id as rich_object_cells_by_id,
    object_room_id as rich_object_room_id,
)


def _layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    layer = layers.get(name)
    if not isinstance(layer, Sequence) or isinstance(layer, (str, bytes)):
        return []
    return layer  # type: ignore[return-value]


def _metadata_by_id(items: object) -> dict[int, Mapping[str, object]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return {}
    result: dict[int, Mapping[str, object]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("id")
        if isinstance(raw_id, int) and not isinstance(raw_id, bool):
            result[raw_id] = item
    return result


def _attributes(item: Mapping[str, object]) -> tuple[str, ...]:
    attrs = item.get("attributes")
    if not isinstance(attrs, Sequence) or isinstance(attrs, (str, bytes)):
        return ()
    return tuple(str(attr) for attr in attrs)


def _scan_positive_cells(layer: Sequence[Sequence[object]]) -> dict[int, list[Cell]]:
    cells_by_id: dict[int, list[Cell]] = defaultdict(list)
    for row, values in enumerate(layer):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for col, raw_value in enumerate(values):
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                cells_by_id[value].append((row, col))
    return cells_by_id


def _room_value(room_layer: Sequence[Sequence[object]], cell: Cell) -> int:
    row, col = cell
    try:
        return int(room_layer[row][col])
    except (IndexError, TypeError, ValueError):
        return 0


def _distance_to_bbox(row: int, col: int, box: tuple[int, int, int, int]) -> float:
    min_row, min_col, max_row, max_col = box
    delta_row = max(min_row - row, 0, row - max_row)
    delta_col = max(min_col - col, 0, col - max_col)
    return math.hypot(delta_row, delta_col)


def _approach_cells_from_bbox(
    traversable: TraversableGrid,
    source_cells: Sequence[Cell],
    *,
    radius: int,
) -> tuple[Cell, ...]:
    if not source_cells:
        return ()
    grid_size = len(traversable)
    min_row, min_col, max_row, max_col = bbox(source_cells)
    row_min = max(0, min_row - radius)
    row_max = min(grid_size - 1, max_row + radius)
    col_min = max(0, min_col - radius)
    col_max = min(grid_size - 1, max_col + radius)
    source_bbox = (min_row, min_col, max_row, max_col)
    cells: list[Cell] = []
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            if not is_traversable(traversable, row, col):
                continue
            if _distance_to_bbox(row, col, source_bbox) <= radius:
                cells.append((row, col))
    return tuple(cells)


def _nearest_room_for_cells(
    room_layer: Sequence[Sequence[object]],
    cells: Sequence[Cell],
) -> int | None:
    votes: Counter[int] = Counter()
    for cell in cells:
        room_id = _room_value(room_layer, cell)
        if room_id > 0:
            votes[room_id] += 1
    return votes.most_common(1)[0][0] if votes else None


def build_scene_graph(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    *,
    object_reach_radius: int = 20,
) -> SceneGraph:
    grid_size = int(map_state["grid_size"])
    room_layer = _layer(map_state, "room")
    room_cells_raw = _scan_positive_cells(room_layer)
    object_cells_raw = rich_object_cells_by_id(map_state)
    room_meta = _metadata_by_id(map_state.get("room_instances"))
    object_meta = _metadata_by_id(map_state.get("object_instances"))

    base_cells = tuple(
        (row, col)
        for row in range(grid_size)
        for col in range(grid_size)
        if is_traversable(traversable, row, col)
    )
    regions: dict[str, AttributeRegion] = {
        "floor_0": AttributeRegion(
            node_id="floor_0",
            kind="floor",
            instance_id=0,
            category="floor",
            name="floor_0",
            cells=base_cells,
            center=centroid(base_cells),
            parent_id=None,
        )
    }
    warnings: list[str] = []

    for room_id, cells in sorted(room_cells_raw.items()):
        traversable_cells = tuple(cell for cell in cells if is_traversable(traversable, *cell))
        if not traversable_cells:
            warnings.append(f"room_{room_id}_has_no_traversable_cells")
            continue
        meta = room_meta.get(room_id, {})
        category = normalize_name(meta.get("category", f"room_{room_id}"))
        name = str(meta.get("name") or f"{category}_{room_id}")
        node_id = f"room_{room_id}"
        regions[node_id] = AttributeRegion(
            node_id=node_id,
            kind="room",
            instance_id=room_id,
            category=category,
            name=name,
            cells=traversable_cells,
            center=centroid(traversable_cells),
            parent_id="floor_0",
            source_cells=tuple(cells),
            attributes=_attributes(meta),
        )

    object_room: dict[str, str] = {}
    for object_id, source_cells in sorted(object_cells_raw.items()):
        meta = object_meta.get(object_id, {})
        category = normalize_name(meta.get("category", f"object_{object_id}"))
        name = str(meta.get("name") or f"{category}_{object_id}")
        approach_cells = _approach_cells_from_bbox(
            traversable,
            source_cells,
            radius=object_reach_radius,
        )
        if not approach_cells:
            fallback = nearest_traversable_cell(
                traversable,
                centroid(source_cells),
                max_radius=max(80, object_reach_radius * 4),
            )
            approach_cells = (fallback,) if fallback is not None else ()
        room_id = rich_object_room_id(map_state, object_id)
        if room_id is None:
            room_id = _nearest_room_for_cells(room_layer, source_cells)
        if room_id is None:
            room_id = _nearest_room_for_cells(room_layer, approach_cells)
        parent_id = f"room_{room_id}" if room_id is not None and f"room_{room_id}" in regions else "floor_0"
        node_id = f"object_{object_id}"
        if parent_id.startswith("room_"):
            object_room[node_id] = parent_id
        else:
            warnings.append(f"{node_id}_has_no_room_parent")
        regions[node_id] = AttributeRegion(
            node_id=node_id,
            kind="object",
            instance_id=object_id,
            category=category,
            name=name,
            cells=approach_cells,
            center=centroid(source_cells),
            parent_id=parent_id,
            source_cells=tuple(source_cells),
            attributes=_attributes(meta),
        )

    adjacency_sets: dict[str, set[str]] = defaultdict(set)
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for row, col in base_cells:
        room_a = _room_value(room_layer, (row, col))
        if room_a <= 0:
            continue
        node_a = f"room_{room_a}"
        if node_a not in regions:
            continue
        for delta_row, delta_col in directions:
            next_row = row + delta_row
            next_col = col + delta_col
            if not is_traversable(traversable, next_row, next_col):
                continue
            room_b = _room_value(room_layer, (next_row, next_col))
            if room_b <= 0 or room_b == room_a:
                continue
            node_b = f"room_{room_b}"
            if node_b not in regions:
                continue
            adjacency_sets[node_a].add(node_b)
            adjacency_sets[node_b].add(node_a)

    room_adjacency = {
        node_id: tuple(sorted(neighbors))
        for node_id, neighbors in sorted(adjacency_sets.items())
    }
    return SceneGraph(
        grid_size=grid_size,
        traversable=traversable,
        base_cells=base_cells,
        regions=regions,
        room_adjacency=room_adjacency,
        object_room=object_room,
        warnings=tuple(warnings),
        metadata={"object_reach_radius": object_reach_radius},
    )
