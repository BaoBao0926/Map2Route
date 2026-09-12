"""Map-derived symbols for the SemPathBench Lang2LTL adapter."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from scripts.methods.util.map_objects import (
    object_cells_by_id as rich_object_cells_by_id,
    object_room_id as rich_object_room_id,
)

@dataclass(frozen=True)
class SemanticEntity:
    kind: str
    id: int
    ap: str
    category: str
    label: str
    center: tuple[float, float]
    cell_count: int
    room_id: int | None = None
    room_category: str | None = None
    attributes: tuple[str, ...] = field(default_factory=tuple)

    def prompt_line(self) -> str:
        context = f", room={self.room_category}" if self.room_category else ""
        return f"{self.ap}: {self.category}{context}"


@dataclass(frozen=True)
class MapSymbols:
    grid_size: int
    object_entities: tuple[SemanticEntity, ...]
    room_entities: tuple[SemanticEntity, ...]

    @property
    def objects_by_id(self) -> dict[int, SemanticEntity]:
        return {entity.id: entity for entity in self.object_entities}

    @property
    def rooms_by_id(self) -> dict[int, SemanticEntity]:
        return {entity.id: entity for entity in self.room_entities}


def _clean_token(value: object, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def _instance_list(map_state: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    raw = map_state.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    raw = layers.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return raw  # type: ignore[return-value]


def _centers_for_layer(
    layer: Sequence[Sequence[object]],
    grid_size: int,
) -> dict[int, tuple[float, float, int]]:
    sums: dict[int, list[float]] = {}
    for row in range(grid_size):
        if row >= len(layer):
            break
        row_values = layer[row]
        if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
            continue
        for col in range(min(grid_size, len(row_values))):
            value = int(row_values[col])
            if value <= 0:
                continue
            current = sums.setdefault(value, [0.0, 0.0, 0.0])
            current[0] += row
            current[1] += col
            current[2] += 1
    return {
        item_id: (values[0] / values[2], values[1] / values[2], int(values[2]))
        for item_id, values in sums.items()
        if values[2] > 0
    }


def _room_at_center(
    room_layer: Sequence[Sequence[object]],
    room_by_id: Mapping[int, SemanticEntity],
    center: tuple[float, float],
) -> tuple[int | None, str | None]:
    row = int(round(center[0]))
    col = int(round(center[1]))
    if row < 0 or row >= len(room_layer):
        return None, None
    row_values = room_layer[row]
    if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
        return None, None
    if col < 0 or col >= len(row_values):
        return None, None
    room_id = int(row_values[col])
    room = room_by_id.get(room_id)
    if room is None:
        return None, None
    return room.id, room.category


def build_map_symbols(map_state: Mapping[str, object]) -> MapSymbols:
    grid_size = int(map_state["grid_size"])
    object_cells_by_id = rich_object_cells_by_id(map_state)
    object_centers = {
        object_id: (
            sum(row for row, _col in cells) / len(cells),
            sum(col for _row, col in cells) / len(cells),
            len(cells),
        )
        for object_id, cells in object_cells_by_id.items()
        if cells
    }
    room_centers = _centers_for_layer(_layer(map_state, "room"), grid_size)

    rooms: list[SemanticEntity] = []
    for item in _instance_list(map_state, "room_instances"):
        raw_id = item.get("id")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            continue
        category = _clean_token(item.get("category"), "room")
        center_row, center_col, cell_count = room_centers.get(raw_id, (0.0, 0.0, 0))
        if cell_count == 0:
            continue
        rooms.append(
            SemanticEntity(
                kind="room",
                id=raw_id,
                ap=f"room_{raw_id}",
                category=category,
                label=str(item.get("name") or f"{category}_{raw_id}"),
                center=(center_row, center_col),
                cell_count=cell_count,
                attributes=tuple(str(value) for value in item.get("attributes", [])[:4])
                if isinstance(item.get("attributes"), list)
                else (),
            )
        )
    room_by_id = {room.id: room for room in rooms}
    room_layer = _layer(map_state, "room")

    objects: list[SemanticEntity] = []
    for item in _instance_list(map_state, "object_instances"):
        raw_id = item.get("id")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            continue
        category = _clean_token(item.get("category"), "object")
        center_row, center_col, cell_count = object_centers.get(raw_id, (0.0, 0.0, 0))
        if cell_count == 0:
            continue
        room_id = rich_object_room_id(map_state, raw_id)
        room_category = room_by_id.get(room_id).category if room_id in room_by_id else None
        if room_id is None:
            room_id, room_category = _room_at_center(
                room_layer,
                room_by_id,
                (center_row, center_col),
            )
        objects.append(
            SemanticEntity(
                kind="object",
                id=raw_id,
                ap=f"object_{raw_id}",
                category=category,
                label=str(item.get("name") or f"{category}_{raw_id}"),
                center=(center_row, center_col),
                cell_count=cell_count,
                room_id=room_id,
                room_category=room_category,
                attributes=tuple(str(value) for value in item.get("attributes", [])[:4])
                if isinstance(item.get("attributes"), list)
                else (),
            )
        )

    return MapSymbols(
        grid_size=grid_size,
        object_entities=tuple(sorted(objects, key=lambda item: (item.category, item.id))),
        room_entities=tuple(sorted(rooms, key=lambda item: (item.category, item.id))),
    )


def prompt_inventory(
    symbols: MapSymbols,
    *,
    max_objects: int = 120,
) -> dict[str, object]:
    object_counts = Counter(entity.category for entity in symbols.object_entities)
    room_counts = Counter(entity.category for entity in symbols.room_entities)
    objects_by_id = sorted(symbols.object_entities, key=lambda entity: entity.id)
    selected_by_ap: dict[str, SemanticEntity] = {}
    for entity in sorted(symbols.object_entities, key=lambda item: (item.category, item.id)):
        if entity.category not in {item.category for item in selected_by_ap.values()}:
            selected_by_ap[entity.ap] = entity
            if len(selected_by_ap) >= max_objects:
                break
    for entity in objects_by_id:
        if len(selected_by_ap) >= max_objects:
            break
        selected_by_ap.setdefault(entity.ap, entity)
    selected = sorted(selected_by_ap.values(), key=lambda entity: entity.id)

    return {
        "room_category_counts": dict(sorted(room_counts.items())),
        "object_category_counts": dict(sorted(object_counts.items())),
        "rooms": [
            entity.prompt_line()
            for entity in sorted(symbols.room_entities, key=lambda entity: entity.id)
        ],
        "objects": [entity.prompt_line() for entity in selected],
        "object_prompt_count": len(selected),
        "object_total_count": len(symbols.object_entities),
        "duplicate_category_rule": (
            "If multiple APs share a category, use the first AP in this listed "
            "deterministic instance-id order."
        ),
        "ap_usage_rule": "Use only AP names explicitly listed in rooms or objects.",
    }


def obj2sem(symbols: MapSymbols) -> dict[str, dict[str, object]]:
    """Build the semantic dictionary consumed by the Lang2LTL-style grounder."""
    result: dict[str, dict[str, object]] = {}
    for entity in (*symbols.room_entities, *symbols.object_entities):
        result[entity.ap] = {
            "kind": entity.kind,
            "id": entity.id,
            "category": entity.category,
            "label": entity.label,
            "room_id": entity.room_id,
            "room_category": entity.room_category,
            "attributes": list(entity.attributes),
        }
    return result
