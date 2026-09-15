"""SceneMap construction and restricted map queries for Grounding2Route."""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Mapping, Sequence

from scripts.methods.grounding2route.alias_utils import merged_entity_aliases, merged_room_aliases
from scripts.methods.grounding2route.config import ENTITY_ALIASES, ROOM_ALIASES
from scripts.methods.grounding2route.ir import Cell, GPRef
from scripts.methods.util.grid_astar import TraversableGrid, build_traversable_grid, is_traversable, nearest_traversable_cell
from scripts.methods.util.map_objects import (
    object_cells_by_id as rich_object_cells_by_id,
)


def canonical_category(value: object, *, aliases: Mapping[str, str] | None = None) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        key.strip().lower().replace("-", "_").replace(" ", "_"): str(val).strip().lower().replace("-", "_").replace(" ", "_")
        for key, val in (aliases or {}).items()
    }
    return alias_map.get(normalized, normalized)


def _layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    raw = layers.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return raw  # type: ignore[return-value]


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


def _instances(map_state: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    raw = map_state.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _center(cells: Sequence[Cell]) -> tuple[float, float]:
    if not cells:
        return 0.0, 0.0
    return (
        sum(row for row, _col in cells) / len(cells),
        sum(col for _row, col in cells) / len(cells),
    )


class SceneMap:
    """Restricted semantic-map view used by Grounding2Route's deterministic grounder."""

    def __init__(self, map_state: Mapping[str, object], instruction: Mapping[str, object]) -> None:
        self.map_state = map_state
        self.grid_size = int(map_state["grid_size"])
        self.traversable = build_traversable_grid(map_state)
        self.room_layer = _layer(map_state, "room")
        self.object_layer = _layer(map_state, "object_instance")
        self.resolution = self._map_resolution()
        self.rooms = self._build_rooms()
        self.entities = self._build_entities()
        self.object_to_room = {ref.id: ref.room_id for ref in self.entities.values()}
        self.start = self._start_ref(instruction)
        self.passages = self._build_passages()

    def _map_resolution(self) -> float:
        metadata = self.map_state.get("metadata")
        if isinstance(metadata, Mapping):
            map_info = metadata.get("map_info")
            if isinstance(map_info, Mapping):
                value = map_info.get("resolution")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                    return float(value)
        return 1.0

    def _build_rooms(self) -> dict[str, GPRef]:
        cells_by_room: dict[int, list[Cell]] = defaultdict(list)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                room_id = _value_at(self.room_layer, row, col)
                if room_id > 0:
                    cells_by_room[room_id].append((row, col))
        meta_by_id = {
            int(item["id"]): item
            for item in _instances(self.map_state, "room_instances")
            if isinstance(item.get("id"), int)
        }
        refs: dict[str, GPRef] = {}
        for room_id, cells in sorted(cells_by_room.items()):
            meta = meta_by_id.get(room_id, {})
            category = canonical_category(meta.get("category"), aliases=merged_room_aliases(ROOM_ALIASES)) or "room"
            key = f"room_{room_id}"
            refs[key] = GPRef(
                kind="room",
                id=key,
                category=category,
                center=_center(cells),
                cells=frozenset(cells),
            )
        return refs

    def _room_for_cells(self, cells: Sequence[Cell], center: tuple[float, float]) -> str | None:
        counts: dict[int, int] = defaultdict(int)
        for row, col in cells:
            room_id = _value_at(self.room_layer, row, col)
            if room_id > 0:
                counts[room_id] += 1
        if counts:
            return f"room_{max(counts.items(), key=lambda item: (item[1], -item[0]))[0]}"
        nearest = nearest_traversable_cell(self.traversable, center, max_radius=80)
        if nearest is None:
            return None
        room_id = _value_at(self.room_layer, nearest[0], nearest[1])
        return f"room_{room_id}" if room_id > 0 else None

    def _build_entities(self) -> dict[str, GPRef]:
        cells_by_object = rich_object_cells_by_id(self.map_state)
        meta_by_id = {
            int(item["id"]): item
            for item in _instances(self.map_state, "object_instances")
            if isinstance(item.get("id"), int)
        }
        refs: dict[str, GPRef] = {}
        for object_id, cells in sorted(cells_by_object.items()):
            meta = meta_by_id.get(object_id, {})
            category = canonical_category(meta.get("category"), aliases=merged_entity_aliases(ENTITY_ALIASES)) or "object"
            center = _center(cells)
            raw_room_id = meta.get("room_id")
            if isinstance(raw_room_id, int) and not isinstance(raw_room_id, bool) and raw_room_id > 0:
                room_id = f"room_{raw_room_id}"
            elif isinstance(raw_room_id, float) and raw_room_id > 0:
                room_id = f"room_{int(raw_room_id)}"
            else:
                room_id = self._room_for_cells(cells, center)
            key = f"object_{object_id}"
            refs[key] = GPRef(
                kind="entity",
                id=key,
                category=category,
                room_id=room_id,
                center=center,
                cells=frozenset(cells),
            )
        return refs

    def _start_ref(self, instruction: Mapping[str, object]) -> GPRef:
        start_pose = instruction.get("start_pose")
        if isinstance(start_pose, Mapping):
            row = start_pose.get("row")
            col = start_pose.get("col")
            if isinstance(row, int) and isinstance(col, int):
                cell = (row, col)
                if not is_traversable(self.traversable, row, col):
                    cell = nearest_traversable_cell(self.traversable, (row, col), max_radius=80) or cell
                room_id = _value_at(self.room_layer, cell[0], cell[1])
                return GPRef(
                    kind="position",
                    id="task_start",
                    room_id=f"room_{room_id}" if room_id > 0 else None,
                    center=(float(cell[0]), float(cell[1])),
                    cells=frozenset({cell}),
                )
        fallback = nearest_traversable_cell(self.traversable, (self.grid_size / 2, self.grid_size / 2), max_radius=self.grid_size)
        fallback = fallback or (0, 0)
        return GPRef(kind="position", id="task_start", center=(float(fallback[0]), float(fallback[1])), cells=frozenset({fallback}))

    def _build_passages(self) -> dict[tuple[str, str], list[GPRef]]:
        transitions: dict[tuple[int, int], set[Cell]] = defaultdict(set)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                room_id = _value_at(self.room_layer, row, col)
                if room_id <= 0:
                    continue
                for dr, dc in ((1, 0), (0, 1)):
                    other = _value_at(self.room_layer, row + dr, col + dc)
                    if other > 0 and other != room_id:
                        pair = tuple(sorted((room_id, other)))
                        if is_traversable(self.traversable, row, col):
                            transitions[pair].add((row, col))
                        if is_traversable(self.traversable, row + dr, col + dc):
                            transitions[pair].add((row + dr, col + dc))
        result: dict[tuple[str, str], list[GPRef]] = {}
        for (first, second), cells in transitions.items():
            components = self._components(cells)
            refs: list[GPRef] = []
            for index, component in enumerate(components, start=1):
                key = f"passage_room_{first}_room_{second}_{index}"
                refs.append(
                    GPRef(
                        kind="region",
                        id=key,
                        category="passage",
                        center=_center(list(component)),
                        cells=frozenset(component),
                        construction={"type": "passage", "rooms": [f"room_{first}", f"room_{second}"]},
                    )
                )
            result[(f"room_{first}", f"room_{second}")] = refs
        return result

    def _components(self, cells: set[Cell]) -> list[set[Cell]]:
        remaining = set(cells)
        components: list[set[Cell]] = []
        while remaining:
            start = remaining.pop()
            component = {start}
            queue: deque[Cell] = deque([start])
            while queue:
                row, col = queue.popleft()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        neighbor = (row + dr, col + dc)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)
            components.append(component)
        return components

    def refs_by_category(self, kind: str, category: str | None = None) -> list[GPRef]:
        refs = self.rooms.values() if kind == "room" else self.entities.values()
        normalized = (
            canonical_category(
                category,
                aliases=merged_room_aliases(ROOM_ALIASES)
                if kind == "room"
                else merged_entity_aliases(ENTITY_ALIASES),
            )
            if category
            else None
        )
        return sorted(
            [ref for ref in refs if normalized is None or ref.category == normalized],
            key=lambda ref: ref.id,
        )

    def room_of(self, ref: GPRef) -> GPRef:
        if ref.kind == "room":
            return ref
        if ref.room_id and ref.room_id in self.rooms:
            return self.rooms[ref.room_id]
        if ref.center is not None:
            row, col = int(round(ref.center[0])), int(round(ref.center[1]))
            room_id = _value_at(self.room_layer, row, col)
            if room_id > 0 and f"room_{room_id}" in self.rooms:
                return self.rooms[f"room_{room_id}"]
        raise ValueError("INVALID_REFERENCE: reference has no containing room.")

    def distance(self, first: GPRef, second: GPRef, metric: str = "geodesic") -> float:
        if metric == "euclidean":
            return _euclidean_ref_distance(first, second)
        return _euclidean_ref_distance(first, second)

    def passage_regions(self, first: GPRef, second: GPRef) -> list[GPRef]:
        if first.kind != "room" or second.kind != "room":
            raise ValueError("INVALID_TOPOLOGY_QUERY: passage_regions requires rooms.")
        key = tuple(sorted((first.id, second.id)))
        return list(self.passages.get(key, ()))

    def region_from_cells(self, region_id: str, cells: set[Cell], construction: Mapping[str, object]) -> GPRef:
        if not cells:
            raise ValueError("INVALID_REGION: constructed region is empty.")
        return GPRef(
            kind="region",
            id=region_id,
            category=str(construction.get("type", "region")),
            center=_center(list(cells)),
            cells=frozenset(cells),
            construction=dict(construction),
        )


def _euclidean_ref_distance(first: GPRef, second: GPRef) -> float:
    if first.cells and second.cells:
        # Use centers for speed; static relation operators can refine this later.
        pass
    if first.center is None or second.center is None:
        return math.inf
    return math.hypot(first.center[0] - second.center[0], first.center[1] - second.center[1])
