"""Prompt serialization for ILN-style area-passage graphs."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from scripts.methods.iln.graph_types import AreaPassageGraph
from scripts.methods.util.map_objects import object_cells_by_id as rich_object_cells_by_id


def _layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    raw = layers.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return raw  # type: ignore[return-value]


def _localizable_object_ids(map_state: Mapping[str, object]) -> set[int]:
    return set(rich_object_cells_by_id(map_state))


def _tokenize(text: str) -> set[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {part for part in cleaned.split() if part}


def _instance_lines(
    instances: Sequence[Mapping[str, object]],
    *,
    kind: str,
    instruction_text: str,
    max_count: int,
) -> tuple[list[str], int]:
    query = _tokenize(instruction_text)

    def score(item: Mapping[str, object]) -> tuple[int, int]:
        category = str(item.get("category") or "")
        name = str(item.get("name") or "")
        overlap = len(query & _tokenize(f"{category} {name}"))
        raw_id = item.get("id")
        item_id = raw_id if isinstance(raw_id, int) and not isinstance(raw_id, bool) else 0
        return (-overlap, item_id)

    selected = sorted(instances, key=score)[:max_count]
    lines: list[str] = []
    for item in selected:
        raw_id = item.get("id")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            continue
        category = str(item.get("category") or kind)
        name = str(item.get("name") or f"{kind}_{raw_id}")
        lines.append(f"{kind}_{raw_id}: category={category}, name={name}")
    return lines, max(0, len(instances) - len(selected))


def map_inventory(
    map_state: Mapping[str, object],
    graph: AreaPassageGraph,
    *,
    instruction_text: str,
    max_prompt_objects: int = 120,
) -> dict[str, object]:
    object_instances_raw = map_state.get("object_instances")
    object_instances = (
        [item for item in object_instances_raw if isinstance(item, Mapping)]
        if isinstance(object_instances_raw, Sequence)
        and not isinstance(object_instances_raw, (str, bytes))
        else []
    )
    localizable_ids = _localizable_object_ids(map_state)
    localizable_objects = [
        item
        for item in object_instances
        if isinstance(item.get("id"), int) and int(item["id"]) in localizable_ids
    ]
    room_counts = Counter(area.category for area in graph.areas.values())
    object_counts = Counter(str(item.get("category") or "object") for item in localizable_objects)
    object_lines, truncated_objects = _instance_lines(
        localizable_objects,
        kind="object",
        instruction_text=instruction_text,
        max_count=max_prompt_objects,
    )
    return {
        "room_category_counts": dict(sorted(room_counts.items())),
        "object_category_counts": dict(sorted(object_counts.items())),
        "objects": object_lines,
        "object_prompt_count": len(object_lines),
        "object_total_count": len(localizable_objects),
        "object_unlocalizable_count": max(0, len(object_instances) - len(localizable_objects)),
        "truncated_object_count": truncated_objects,
    }


def serialize_graph_for_prompt(
    graph: AreaPassageGraph,
    *,
    map_state: Mapping[str, object],
    instruction_text: str,
    max_prompt_objects: int = 120,
) -> dict[str, object]:
    area_lines = [
        f"{area.key}: category={area.category}, name={area.name}, passages={graph.room_passages.get(area.id, [])}"
        for area in sorted(graph.areas.values(), key=lambda item: item.id)
    ]
    passage_lines = []
    for passage in sorted(graph.passages.values(), key=lambda item: item.id):
        passage_lines.append(
            f"{passage.id}: rooms=room_{passage.rooms[0]}<->room_{passage.rooms[1]}, "
            f"anchor={list(passage.anchor)}, width={passage.width}"
        )
    inventory = map_inventory(
        map_state,
        graph,
        instruction_text=instruction_text,
        max_prompt_objects=max_prompt_objects,
    )
    return {
        "current_area": graph.area_key(graph.start_area),
        "areas": area_lines,
        "passages": passage_lines,
        "inventory": inventory,
        "truncation": {
            "object_total_count": inventory["object_total_count"],
            "object_prompt_count": inventory["object_prompt_count"],
            "truncated_object_count": inventory["truncated_object_count"],
        },
    }
