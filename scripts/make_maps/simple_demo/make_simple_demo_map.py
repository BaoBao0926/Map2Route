#!/usr/bin/env python3
"""Browser-based editor for quickly hand-making layered 2D grid maps."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import webbrowser
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.evaluation.metric_cache import build_map_metric_cache

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is optional at runtime.
    Image = None


GRID_SIZE = 128
DEFAULT_MAP_ID = "simple_demo_map"

OCCUPANCY_TILES = {
    "free": {"value": 0, "label": "Free Space", "color": "#FFFFFF"},
    "obstacle": {"value": 1, "label": "Obstacle / Wall", "color": "#111111"},
    "unknown": {"value": 2, "label": "Unknown", "color": "#9E9E9E"},
}

ROOM_CATEGORIES = {
    "hallway": {"label": "Hallway", "color": "#F2E8CF"},
    "kitchen": {"label": "Kitchen", "color": "#CDECCF"},
    "bedroom": {"label": "Bedroom", "color": "#D6E8FF"},
    "living_room": {"label": "Living Room", "color": "#FFE0B5"},
}

OBJECT_CATEGORIES = {
    "table": {"label": "Table", "color": "#F4D03F"},
    "chair": {"label": "Chair", "color": "#EFA8A0"},
    "sofa": {"label": "Sofa", "color": "#D96C5F"},
    "desk": {"label": "Desk", "color": "#8EC5A4"},
    "cabinet": {"label": "Cabinet", "color": "#7C9A6D"},
}

LAYER_MODES = [
    ("occupancy", "Free / Obstacle / Unknown"),
    ("room", "Room Space"),
    ("object_instance", "Object Instance"),
]

OCCUPANCY_VALUE_TO_NAME = {
    tile["value"]: name for name, tile in OCCUPANCY_TILES.items()
}
VALID_OCCUPANCY_VALUES = set(OCCUPANCY_VALUE_TO_NAME)
VALID_ROOM_CATEGORIES = set(ROOM_CATEGORIES)
VALID_OBJECT_CATEGORIES = set(OBJECT_CATEGORIES)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_DIR = REPO_ROOT / "resources" / "maps" / "simple_demo"
TEMPLATE_OBJECT_COUNTS = (2, 3, 4, 5, 10)
MIN_TEMPLATE_DISTANCE = 40.0
MAX_TEMPLATE_PER_OBJECT_COUNT = 200
MAX_TEMPLATE_OBJECTS = 80
MAX_TOTAL_TEMPLATES = MAX_TEMPLATE_PER_OBJECT_COUNT * len(TEMPLATE_OBJECT_COUNTS)


def sanitize_map_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return cleaned or DEFAULT_MAP_ID


def map_paths(map_id: str) -> dict[str, Path]:
    safe_id = sanitize_map_id(map_id)
    map_dir = RESOURCE_DIR / safe_id
    return {
        "directory": map_dir,
        "json": map_dir / f"{safe_id}.json",
        "png": map_dir / f"{safe_id}.png",
        "ppm": map_dir / f"{safe_id}.ppm",
        "template_instruction": map_dir / "template_instruction.json",
    }


def template_instruction_path(map_id: str) -> Path:
    return map_paths(map_id)["template_instruction"]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_blank_grid(fill_value: int = 0) -> list[list[int]]:
    return [[fill_value for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def fill_rect(
    grid: list[list[int]], top: int, left: int, height: int, width: int, value: int
) -> None:
    for row in range(top, min(GRID_SIZE, top + height)):
        for col in range(left, min(GRID_SIZE, left + width)):
            grid[row][col] = value


def next_instance_name(category: str, existing: list[dict[str, object]]) -> str:
    prefix = f"{category}_"
    used = {
        item["name"]
        for item in existing
        if isinstance(item.get("name"), str) and item["name"].startswith(prefix)
    }
    index = 1
    while f"{category}_{index}" in used:
        index += 1
    return f"{category}_{index}"


def add_instance(
    instances: list[dict[str, object]], category: str, name: str | None = None
) -> int:
    next_id = max((int(item["id"]) for item in instances), default=0) + 1
    instances.append(
        {
            "id": next_id,
            "category": category,
            "name": name or next_instance_name(category, instances),
            "attributes": [],
        }
    )
    return next_id


def make_default_metadata(map_id: str = DEFAULT_MAP_ID) -> dict[str, str]:
    safe_id = sanitize_map_id(map_id)
    title = " ".join(part.capitalize() for part in safe_id.split("_")) or "Simple Demo Map"
    return {
        "map_id": safe_id,
        "name": title,
        "description": "Hand-made prototype map for fast SemPathBench validation.",
    }


def make_blank_state(map_id: str = DEFAULT_MAP_ID) -> dict[str, object]:
    metadata = make_default_metadata(map_id)
    return {
        "metadata": metadata,
        "layers": {
            "occupancy": make_blank_grid(OCCUPANCY_TILES["free"]["value"]),
            "room": make_blank_grid(0),
            "object_instance": make_blank_grid(0),
        },
        "room_instances": [],
        "object_instances": [],
    }


def make_sample_state(map_id: str = DEFAULT_MAP_ID) -> dict[str, object]:
    state = make_blank_state(map_id)
    occupancy = state["layers"]["occupancy"]  # type: ignore[index]
    room = state["layers"]["room"]  # type: ignore[index]
    object_grid = state["layers"]["object_instance"]  # type: ignore[index]
    room_instances = state["room_instances"]  # type: ignore[assignment]
    instances = state["object_instances"]  # type: ignore[assignment]

    for index in range(GRID_SIZE):
        occupancy[0][index] = OCCUPANCY_TILES["obstacle"]["value"]
        occupancy[GRID_SIZE - 1][index] = OCCUPANCY_TILES["obstacle"]["value"]
        occupancy[index][0] = OCCUPANCY_TILES["obstacle"]["value"]
        occupancy[index][GRID_SIZE - 1] = OCCUPANCY_TILES["obstacle"]["value"]

    for row in range(16, 112):
        occupancy[row][62] = OCCUPANCY_TILES["obstacle"]["value"]
    for col in range(16, 112):
        occupancy[60][col] = OCCUPANCY_TILES["obstacle"]["value"]

    for row in range(34, 42):
        occupancy[row][62] = OCCUPANCY_TILES["free"]["value"]
    for row in range(84, 92):
        occupancy[row][62] = OCCUPANCY_TILES["free"]["value"]
    for col in range(30, 38):
        occupancy[60][col] = OCCUPANCY_TILES["free"]["value"]
    for col in range(88, 96):
        occupancy[60][col] = OCCUPANCY_TILES["free"]["value"]

    living_room_1 = add_instance(room_instances, "living_room", "living_room_1")
    kitchen_1 = add_instance(room_instances, "kitchen", "kitchen_1")
    bedroom_1 = add_instance(room_instances, "bedroom", "bedroom_1")
    hallway_1 = add_instance(room_instances, "hallway", "hallway_1")
    hallway_2 = add_instance(room_instances, "hallway", "hallway_2")
    hallway_3 = add_instance(room_instances, "hallway", "hallway_3")

    fill_rect(room, 8, 8, 50, 50, living_room_1)
    fill_rect(room, 8, 68, 42, 44, kitchen_1)
    fill_rect(room, 68, 10, 42, 42, bedroom_1)
    fill_rect(room, 68, 68, 42, 42, hallway_1)
    fill_rect(room, 8, 58, 104, 10, hallway_2)
    fill_rect(room, 58, 8, 104, 10, hallway_3)

    fill_rect(occupancy, 3, 20, 5, 88, OCCUPANCY_TILES["unknown"]["value"])

    sofa_1 = add_instance(instances, "sofa", "sofa_1")
    table_1 = add_instance(instances, "table", "table_1")
    chair_1 = add_instance(instances, "chair", "chair_1")
    table_2 = add_instance(instances, "table", "table_2")
    cabinet_1 = add_instance(instances, "cabinet", "cabinet_1")
    chair_2 = add_instance(instances, "chair", "chair_2")
    desk_1 = add_instance(instances, "desk", "desk_1")
    table_3 = add_instance(instances, "table", "table_3")
    sofa_2 = add_instance(instances, "sofa", "sofa_2")
    chair_3 = add_instance(instances, "chair", "chair_3")

    fill_rect(object_grid, 24, 22, 8, 12, sofa_1)
    fill_rect(object_grid, 38, 18, 6, 10, table_1)
    fill_rect(object_grid, 38, 31, 5, 5, chair_1)

    fill_rect(object_grid, 22, 82, 8, 12, table_2)
    fill_rect(object_grid, 34, 95, 12, 6, cabinet_1)
    fill_rect(object_grid, 41, 77, 5, 10, chair_2)

    fill_rect(object_grid, 82, 18, 10, 10, desk_1)
    fill_rect(object_grid, 95, 28, 7, 14, table_3)

    fill_rect(object_grid, 76, 84, 7, 18, table_2)
    fill_rect(object_grid, 92, 76, 8, 18, sofa_2)
    fill_rect(object_grid, 74, 72, 6, 8, chair_3)

    state["metadata"] = {
        "map_id": sanitize_map_id(map_id),
        "name": "Simple Demo Map",
        "description": (
            "Layered sample map with occupancy, room labels, and object instances."
        ),
    }
    return state


def validate_grid(grid: object, valid_values: set[int], label: str) -> list[list[int]]:
    if not isinstance(grid, list) or len(grid) != GRID_SIZE:
        raise ValueError(f"{label} grid must have {GRID_SIZE} rows.")

    validated: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != GRID_SIZE:
            raise ValueError(f"Each {label} grid row must have {GRID_SIZE} columns.")
        validated_row: list[int] = []
        for value in row:
            if not isinstance(value, int) or value not in valid_values:
                raise ValueError(f"Unexpected {label} cell value: {value!r}")
            validated_row.append(value)
        validated.append(validated_row)
    return validated


def validate_metadata(metadata: object) -> dict[str, str]:
    default = make_default_metadata()
    if not isinstance(metadata, dict):
        return default

    map_id = metadata.get("map_id", default["map_id"])
    name = metadata.get("name", default["name"])
    description = metadata.get("description", default["description"])

    if not isinstance(map_id, str) or not map_id.strip():
        map_id = default["map_id"]
    if not isinstance(name, str) or not name.strip():
        name = default["name"]
    if not isinstance(description, str):
        description = default["description"]

    safe_id = sanitize_map_id(map_id)
    return {
        "map_id": safe_id,
        "name": name.strip(),
        "description": description,
    }


def validate_object_instances(instances: object) -> list[dict[str, object]]:
    if instances is None:
        return []
    if not isinstance(instances, list):
        raise ValueError("object_instances must be a list.")

    validated: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(instances, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each object instance must be an object.")

        raw_id = item.get("id")
        category = item.get("category")
        name = item.get("name")
        attributes = item.get("attributes", [])

        if not isinstance(raw_id, int) or raw_id <= 0:
            raise ValueError("Object instance id must be a positive integer.")
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate object instance id: {raw_id}")
        if not isinstance(category, str) or category not in VALID_OBJECT_CATEGORIES:
            raise ValueError(f"Unknown object category: {category!r}")
        if not isinstance(name, str) or not name.strip():
            name = f"{category}_{index}"
        if not isinstance(attributes, list):
            raise ValueError("Object instance attributes must be a list.")

        validated_attributes: list[str] = []
        for attribute in attributes:
            if not isinstance(attribute, str):
                raise ValueError("Each object instance attribute must be a string.")
            cleaned = attribute.strip()
            if cleaned:
                validated_attributes.append(cleaned)

        seen_ids.add(raw_id)
        validated.append(
            {
                "id": raw_id,
                "category": category,
                "name": name.strip(),
                "attributes": validated_attributes,
            }
        )

    return validated


def validate_room_instances(instances: object) -> list[dict[str, object]]:
    if instances is None:
        return []
    if not isinstance(instances, list):
        raise ValueError("room_instances must be a list.")

    validated: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(instances, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each room instance must be an object.")

        raw_id = item.get("id")
        category = item.get("category")
        name = item.get("name")
        attributes = item.get("attributes", [])

        if not isinstance(raw_id, int) or raw_id <= 0:
            raise ValueError("Room instance id must be a positive integer.")
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate room instance id: {raw_id}")
        if not isinstance(category, str) or category not in VALID_ROOM_CATEGORIES:
            raise ValueError(f"Unknown room category: {category!r}")
        if not isinstance(name, str) or not name.strip():
            name = f"{category}_{index}"
        if not isinstance(attributes, list):
            raise ValueError("Room instance attributes must be a list.")

        validated_attributes: list[str] = []
        for attribute in attributes:
            if not isinstance(attribute, str):
                raise ValueError("Each room instance attribute must be a string.")
            cleaned = attribute.strip()
            if cleaned:
                validated_attributes.append(cleaned)

        seen_ids.add(raw_id)
        validated.append(
            {
                "id": raw_id,
                "category": category,
                "name": name.strip(),
                "attributes": validated_attributes,
            }
        )

    return validated


def validate_object_grid(
    grid: object, instance_ids: set[int], allow_any_non_negative: bool = False
) -> list[list[int]]:
    if not isinstance(grid, list) or len(grid) != GRID_SIZE:
        raise ValueError(f"object_instance grid must have {GRID_SIZE} rows.")

    validated: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != GRID_SIZE:
            raise ValueError(
                f"Each object_instance grid row must have {GRID_SIZE} columns."
            )
        validated_row: list[int] = []
        for value in row:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Unexpected object_instance cell value: {value!r}")
            if not allow_any_non_negative and value not in instance_ids and value != 0:
                raise ValueError(f"Unknown object instance id in grid: {value}")
            validated_row.append(value)
        validated.append(validated_row)
    return validated


def validate_instance_grid(
    grid: object, instance_ids: set[int], label: str
) -> list[list[int]]:
    if not isinstance(grid, list) or len(grid) != GRID_SIZE:
        raise ValueError(f"{label} grid must have {GRID_SIZE} rows.")

    validated: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != GRID_SIZE:
            raise ValueError(f"Each {label} grid row must have {GRID_SIZE} columns.")
        validated_row: list[int] = []
        for value in row:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Unexpected {label} cell value: {value!r}")
            if value not in instance_ids and value != 0:
                raise ValueError(f"Unknown {label} id in grid: {value}")
            validated_row.append(value)
        validated.append(validated_row)
    return validated


def make_payload(state: dict[str, object]) -> dict[str, object]:
    return {
        "grid_size": GRID_SIZE,
        "layer_legends": {
            "occupancy": OCCUPANCY_TILES,
            "room_categories": ROOM_CATEGORIES,
            "object_categories": OBJECT_CATEGORIES,
        },
        "metadata": state["metadata"],
        "layers": state["layers"],
        "room_instances": state["room_instances"],
        "object_instances": state["object_instances"],
    }


def legacy_extract_object_instances(
    legacy_grid: list[list[int]],
) -> tuple[list[dict[str, object]], list[list[int]]]:
    object_value_to_category = {
        30: "table",
        31: "chair",
        32: "sofa",
        33: "desk",
        34: "cabinet",
    }
    object_grid = make_blank_grid(0)
    instances: list[dict[str, object]] = []
    visited = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    category_counts = {name: 0 for name in OBJECT_CATEGORIES}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            value = legacy_grid[row][col]
            category = object_value_to_category.get(value)
            if category is None or visited[row][col]:
                continue

            category_counts[category] += 1
            instance_id = len(instances) + 1
            instance_name = f"{category}_{category_counts[category]}"
            instances.append(
                {
                    "id": instance_id,
                    "category": category,
                    "name": instance_name,
                    "attributes": [],
                }
            )

            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row][col] = True
            while queue:
                current_row, current_col = queue.popleft()
                object_grid[current_row][current_col] = instance_id
                for d_row, d_col in directions:
                    next_row = current_row + d_row
                    next_col = current_col + d_col
                    if (
                        next_row < 0
                        or next_row >= GRID_SIZE
                        or next_col < 0
                        or next_col >= GRID_SIZE
                        or visited[next_row][next_col]
                        or legacy_grid[next_row][next_col] != value
                    ):
                        continue
                    visited[next_row][next_col] = True
                    queue.append((next_row, next_col))

    return instances, object_grid


def extract_room_instances_from_category_grid(
    category_grid: list[list[int]],
) -> tuple[list[dict[str, object]], list[list[int]]]:
    value_to_category = {
        1: "hallway",
        2: "kitchen",
        3: "bedroom",
        4: "living_room",
    }
    room_grid = make_blank_grid(0)
    instances: list[dict[str, object]] = []
    visited = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    category_counts = {name: 0 for name in ROOM_CATEGORIES}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            value = category_grid[row][col]
            category = value_to_category.get(value)
            if category is None or visited[row][col]:
                continue

            category_counts[category] += 1
            instance_id = len(instances) + 1
            instance_name = f"{category}_{category_counts[category]}"
            instances.append(
                {
                    "id": instance_id,
                    "category": category,
                    "name": instance_name,
                    "attributes": [],
                }
            )

            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row][col] = True
            while queue:
                current_row, current_col = queue.popleft()
                room_grid[current_row][current_col] = instance_id
                for d_row, d_col in directions:
                    next_row = current_row + d_row
                    next_col = current_col + d_col
                    if (
                        next_row < 0
                        or next_row >= GRID_SIZE
                        or next_col < 0
                        or next_col >= GRID_SIZE
                        or visited[next_row][next_col]
                        or category_grid[next_row][next_col] != value
                    ):
                        continue
                    visited[next_row][next_col] = True
                    queue.append((next_row, next_col))

    return instances, room_grid


def load_legacy_state(payload: dict[str, object]) -> dict[str, object]:
    legacy_grid = validate_grid(payload.get("grid"), set(range(0, 35)), "legacy")

    occupancy = make_blank_grid(OCCUPANCY_TILES["free"]["value"])
    room_category_grid = make_blank_grid(0)
    room_value_map = {
        10: 1,
        11: 2,
        12: 3,
        13: 4,
    }

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            value = legacy_grid[row][col]
            if value == 1:
                occupancy[row][col] = OCCUPANCY_TILES["obstacle"]["value"]
            elif value == 2:
                occupancy[row][col] = OCCUPANCY_TILES["unknown"]["value"]
            else:
                occupancy[row][col] = OCCUPANCY_TILES["free"]["value"]

            if value in room_value_map:
                room_category_grid[row][col] = room_value_map[value]

    room_instances, room = extract_room_instances_from_category_grid(room_category_grid)
    object_instances, object_grid = legacy_extract_object_instances(legacy_grid)
    metadata = validate_metadata(payload.get("metadata"))
    if metadata["map_id"] == DEFAULT_MAP_ID and isinstance(payload.get("metadata"), dict):
        raw_name = payload["metadata"].get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            metadata["name"] = "Simple Demo Map"

    return {
        "metadata": metadata,
        "layers": {
            "occupancy": occupancy,
            "room": room,
            "object_instance": object_grid,
        },
        "room_instances": room_instances,
        "object_instances": object_instances,
    }


def validate_state_payload(payload: dict[str, object]) -> dict[str, object]:
    if "layers" not in payload:
        return load_legacy_state(payload)

    metadata = validate_metadata(payload.get("metadata"))
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("layers must be an object.")

    room_instances = validate_room_instances(payload.get("room_instances"))
    object_instances = validate_object_instances(payload.get("object_instances"))
    room_instance_ids = {int(item["id"]) for item in room_instances}
    object_instance_ids = {int(item["id"]) for item in object_instances}

    occupancy = validate_grid(
        layers.get("occupancy"), VALID_OCCUPANCY_VALUES, "occupancy"
    )
    raw_room_layer = layers.get("room")
    if room_instances:
        room = validate_instance_grid(raw_room_layer, room_instance_ids, "room")
    else:
        # Backward compatibility for layered maps saved before room instances existed.
        legacy_room = validate_grid(raw_room_layer, {0, 1, 2, 3, 4}, "room")
        room_instances, room = extract_room_instances_from_category_grid(legacy_room)
        room_instance_ids = {int(item["id"]) for item in room_instances}
    object_grid = validate_object_grid(layers.get("object_instance"), object_instance_ids)

    return {
        "metadata": metadata,
        "layers": {
            "occupancy": occupancy,
            "room": room,
            "object_instance": object_grid,
        },
        "room_instances": room_instances,
        "object_instances": object_instances,
    }


def list_map_ids() -> list[str]:
    migrate_legacy_map_layout()
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path.name
        for path in RESOURCE_DIR.iterdir()
        if path.is_dir() and map_paths(path.name)["json"].is_file()
    )


def migrate_legacy_map_layout() -> None:
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    legacy_template_dir = RESOURCE_DIR / "template_instruction"
    for legacy_json in list(RESOURCE_DIR.glob("*.json")):
        map_id = sanitize_map_id(legacy_json.stem)
        paths = map_paths(map_id)
        paths["directory"].mkdir(parents=True, exist_ok=True)
        if not paths["json"].exists():
            legacy_json.replace(paths["json"])
        for suffix in ("png", "ppm"):
            legacy_asset = RESOURCE_DIR / f"{map_id}.{suffix}"
            if legacy_asset.exists() and not paths[suffix].exists():
                legacy_asset.replace(paths[suffix])
        legacy_template = legacy_template_dir / f"{map_id}.json"
        if legacy_template.exists() and not paths["template_instruction"].exists():
            legacy_template.replace(paths["template_instruction"])
    if legacy_template_dir.exists() and not any(legacy_template_dir.iterdir()):
        legacy_template_dir.rmdir()


def list_map_summaries() -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for map_id in list_map_ids():
        path = map_paths(map_id)["json"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = validate_state_payload(payload)["metadata"]  # type: ignore[index]
        except Exception:
            metadata = make_default_metadata(map_id)
        summaries.append(
            {
                "map_id": metadata["map_id"],  # type: ignore[index]
                "name": metadata["name"],  # type: ignore[index]
            }
        )
    summaries.sort(key=lambda item: (item["name"].lower(), item["map_id"]))
    return summaries


def load_map_state(map_id: str | None = None) -> dict[str, object]:
    ensure_default_map()
    available = list_map_ids()
    if not available:
        raise RuntimeError("No maps available.")

    safe_id = sanitize_map_id(map_id or DEFAULT_MAP_ID)
    if safe_id not in available:
        safe_id = DEFAULT_MAP_ID if DEFAULT_MAP_ID in available else available[0]

    payload = json.loads(map_paths(safe_id)["json"].read_text(encoding="utf-8"))
    state = validate_state_payload(payload)
    state["metadata"]["map_id"] = safe_id  # type: ignore[index]
    ensure_template_instructions(safe_id, state)
    return state


def resolve_save_target(
    original_map_id: str | None, metadata: dict[str, str]
) -> tuple[str, dict[str, str]]:
    target_id = sanitize_map_id(metadata["map_id"])
    metadata = dict(metadata)
    metadata["map_id"] = target_id

    paths = map_paths(target_id)
    if paths["json"].exists() and sanitize_map_id(original_map_id or "") != target_id:
        raise ValueError(f"Map '{target_id}' already exists. Open it instead of creating it again.")

    return target_id, metadata


def render_preview_rgb(
    occupancy: list[list[int]],
    room: list[list[int]],
    object_grid: list[list[int]],
    room_instance_map: dict[int, dict[str, object]],
    instance_map: dict[int, dict[str, object]],
) -> list[tuple[int, int, int]]:
    pixels: list[tuple[int, int, int]] = []
    for row_idx in range(GRID_SIZE):
        for col_idx in range(GRID_SIZE):
            occupancy_name = OCCUPANCY_VALUE_TO_NAME[occupancy[row_idx][col_idx]]
            color = OCCUPANCY_TILES[occupancy_name]["color"]

            room_value = room[row_idx][col_idx]
            if room_value != 0 and occupancy_name == "free" and room_value in room_instance_map:
                category = room_instance_map[room_value]["category"]
                if isinstance(category, str) and category in ROOM_CATEGORIES:
                    color = ROOM_CATEGORIES[category]["color"]

            object_id = object_grid[row_idx][col_idx]
            if object_id != 0 and object_id in instance_map:
                category = instance_map[object_id]["category"]
                if isinstance(category, str) and category in OBJECT_CATEGORIES:
                    color = OBJECT_CATEGORIES[category]["color"]

            pixels.append(tuple(bytes.fromhex(color.lstrip("#"))))
    return pixels


def export_ppm(state: dict[str, object], target_id: str) -> None:
    paths = map_paths(target_id)
    occupancy = state["layers"]["occupancy"]  # type: ignore[index]
    room = state["layers"]["room"]  # type: ignore[index]
    object_grid = state["layers"]["object_instance"]  # type: ignore[index]
    room_instances = state["room_instances"]  # type: ignore[assignment]
    instances = state["object_instances"]  # type: ignore[assignment]
    room_instance_map = {int(item["id"]): item for item in room_instances}
    instance_map = {int(item["id"]): item for item in instances}

    pixels = render_preview_rgb(occupancy, room, object_grid, room_instance_map, instance_map)
    with paths["ppm"].open("wb") as handle:
        handle.write(f"P6\n{GRID_SIZE} {GRID_SIZE}\n255\n".encode("ascii"))
        for pixel in pixels:
            handle.write(bytes(pixel))


def export_png(state: dict[str, object], target_id: str) -> None:
    if Image is None:
        return

    paths = map_paths(target_id)
    occupancy = state["layers"]["occupancy"]  # type: ignore[index]
    room = state["layers"]["room"]  # type: ignore[index]
    object_grid = state["layers"]["object_instance"]  # type: ignore[index]
    room_instances = state["room_instances"]  # type: ignore[assignment]
    instances = state["object_instances"]  # type: ignore[assignment]
    room_instance_map = {int(item["id"]): item for item in room_instances}
    instance_map = {int(item["id"]): item for item in instances}

    image = Image.new("RGB", (GRID_SIZE, GRID_SIZE))
    image.putdata(render_preview_rgb(occupancy, room, object_grid, room_instance_map, instance_map))
    image.save(paths["png"])


def save_map_state(original_map_id: str | None, state: dict[str, object]) -> str:
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    target_id, metadata = resolve_save_target(
        original_map_id,
        state["metadata"],  # type: ignore[arg-type]
    )
    state = {
        "metadata": metadata,
        "layers": state["layers"],
        "room_instances": state["room_instances"],
        "object_instances": state["object_instances"],
    }
    paths = map_paths(target_id)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    payload = make_payload(state)
    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    build_map_metric_cache(paths["json"], payload)
    export_ppm(state, target_id)
    export_png(state, target_id)
    ensure_template_instructions(target_id, state)
    return target_id


def collect_object_centers(state: dict[str, object]) -> list[dict[str, object]]:
    object_grid = state["layers"]["object_instance"]  # type: ignore[index]
    object_cells: dict[int, list[tuple[int, int]]] = {}
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            object_id = int(object_grid[row][col])
            if object_id != 0:
                object_cells.setdefault(object_id, []).append((row, col))

    objects: list[dict[str, object]] = []
    for instance in state["object_instances"]:  # type: ignore[union-attr]
        object_id = int(instance["id"])
        cells = object_cells.get(object_id, [])
        if not cells:
            continue
        center_row = sum(row for row, _ in cells) / len(cells)
        center_col = sum(col for _, col in cells) / len(cells)
        objects.append(
            {
                "object_id": object_id,
                "name": instance["name"],
                "category": instance["category"],
                "center": [round(center_row, 3), round(center_col, 3)],
            }
        )
    return objects


def object_center_distance(first: dict[str, object], second: dict[str, object]) -> float:
    first_center = first["center"]
    second_center = second["center"]
    return math.hypot(
        float(first_center[0]) - float(second_center[0]),  # type: ignore[index]
        float(first_center[1]) - float(second_center[1]),  # type: ignore[index]
    )


def template_diversity_signature(
    path: list[dict[str, object]], grid_size: int
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    bin_size = max(1, grid_size // 6)
    categories = tuple(str(item["category"]) for item in path)
    spatial_bins = tuple(
        (
            int(float(item["center"][0]) // bin_size),  # type: ignore[index]
            int(float(item["center"][1]) // bin_size),  # type: ignore[index]
        )
        for item in path
    )
    return categories, spatial_bins


def sample_diverse_template_paths(
    objects: list[dict[str, object]],
    distances: dict[tuple[int, int], float],
    target_count: int,
) -> list[list[dict[str, object]]]:
    rng = random.SystemRandom()
    paths: list[list[dict[str, object]]] = []
    seen_object_sets: set[tuple[int, ...]] = set()
    seen_signatures: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
    attempts = max(2500, MAX_TEMPLATE_PER_OBJECT_COUNT * target_count * 80)

    def candidate_is_valid(
        path: list[dict[str, object]], candidate: dict[str, object]
    ) -> bool:
        candidate_id = int(candidate["object_id"])
        return all(
            distances[(int(item["object_id"]), candidate_id)] > MIN_TEMPLATE_DISTANCE
            for item in path
        )

    for _attempt in range(attempts):
        if len(paths) >= MAX_TEMPLATE_PER_OBJECT_COUNT:
            break
        if len(objects) < target_count:
            break
        path = [rng.choice(objects)]
        used_ids = {int(path[0]["object_id"])}
        while len(path) < target_count:
            candidates = [
                item
                for item in objects
                if int(item["object_id"]) not in used_ids
                and candidate_is_valid(path, item)
            ]
            if not candidates:
                break
            used_categories = {str(item["category"]) for item in path}
            centroid_row = sum(float(item["center"][0]) for item in path) / len(path)  # type: ignore[index]
            centroid_col = sum(float(item["center"][1]) for item in path) / len(path)  # type: ignore[index]
            scored = []
            for candidate in candidates:
                category_bonus = 25.0 if str(candidate["category"]) not in used_categories else 0.0
                spread_bonus = math.hypot(
                    float(candidate["center"][0]) - centroid_row,  # type: ignore[index]
                    float(candidate["center"][1]) - centroid_col,  # type: ignore[index]
                )
                scored.append((category_bonus + spread_bonus + rng.random() * 10.0, candidate))
            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[: min(12, len(scored))]
            total_weight = sum(max(score, 0.001) for score, _candidate in top)
            pick = rng.random() * total_weight
            running = 0.0
            chosen = top[-1][1]
            for score, candidate in top:
                running += max(score, 0.001)
                if running >= pick:
                    chosen = candidate
                    break
            path.append(chosen)
            used_ids.add(int(chosen["object_id"]))
        if len(path) != target_count:
            continue
        object_set = tuple(sorted(int(item["object_id"]) for item in path))
        if object_set in seen_object_sets:
            continue
        signature = template_diversity_signature(path, GRID_SIZE)
        if signature in seen_signatures:
            continue
        seen_object_sets.add(object_set)
        seen_signatures.add(signature)
        paths.append(path)
    return paths


def generate_template_instructions(
    map_id: str, state: dict[str, object]
) -> dict[str, object]:
    safe_map_id = sanitize_map_id(map_id)
    objects = collect_object_centers(state)[:MAX_TEMPLATE_OBJECTS]
    distances: dict[tuple[int, int], float] = {}
    for first in objects:
        for second in objects:
            first_id = int(first["object_id"])
            second_id = int(second["object_id"])
            if first_id != second_id:
                distances[(first_id, second_id)] = object_center_distance(first, second)

    templates: list[dict[str, object]] = []
    created_at = iso_now()

    for target_count in TEMPLATE_OBJECT_COUNTS:
        sampled_paths = sample_diverse_template_paths(
            objects,
            distances,
            target_count,
        )
        for count_index, path in enumerate(sampled_paths, start=1):
            segment_distances = [
                round(
                    distances[
                        (int(path[index]["object_id"]), int(path[index + 1]["object_id"]))
                    ],
                    3,
                )
                for index in range(len(path) - 1)
            ]
            templates.append(
                {
                    "template_instruction_id": (
                        f"template_{target_count}_{count_index:06d}"
                    ),
                    "map_id": safe_map_id,
                    "object_count": target_count,
                    "objects": [
                        {
                            "order": index,
                            **item,
                        }
                        for index, item in enumerate(path, start=1)
                    ],
                    "segment_distances": segment_distances,
                    "status": "unlabeled",
                    "labeled_instruction_id": None,
                    "created_at": created_at,
                    "updated_at": created_at,
                }
            )
    random.SystemRandom().shuffle(templates)

    return {
        "version": 1,
        "map_id": safe_map_id,
        "distance_rule": {
            "metric": "object_centroid_euclidean_grid",
            "minimum_exclusive": MIN_TEMPLATE_DISTANCE,
        },
        "object_counts": list(TEMPLATE_OBJECT_COUNTS),
        "randomized_order": True,
        "template_limits": {
            "max_objects_considered": MAX_TEMPLATE_OBJECTS,
            "max_per_object_count": MAX_TEMPLATE_PER_OBJECT_COUNT,
            "max_total_templates": MAX_TOTAL_TEMPLATES,
            "spacing_scope": "all_object_pairs",
            "diversity_strategy": "random_spatial_category_sampling",
        },
        "templates": templates,
        "created_at": created_at,
        "updated_at": created_at,
    }


def ensure_template_instructions(
    map_id: str, state: dict[str, object]
) -> tuple[Path, bool]:
    path = template_instruction_path(map_id)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("randomized_order"):
            templates = payload.get("templates")
            if isinstance(templates, list):
                random.SystemRandom().shuffle(templates)
                payload["randomized_order"] = True
                payload["updated_at"] = iso_now()
                path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = generate_template_instructions(map_id, state)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path, True


def ensure_default_map() -> None:
    migrate_legacy_map_layout()
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if list_map_ids():
        return
    save_map_state(None, make_sample_state(DEFAULT_MAP_ID))


def build_client_state(map_id: str, state: dict[str, object]) -> dict[str, object]:
    paths = map_paths(map_id)
    template_path, _ = ensure_template_instructions(map_id, state)
    return {
        "grid_size": GRID_SIZE,
        "legends": {
            "occupancy": OCCUPANCY_TILES,
            "room_categories": ROOM_CATEGORIES,
            "object_categories": OBJECT_CATEGORIES,
        },
        "layer_modes": LAYER_MODES,
        "map_summaries": list_map_summaries(),
        "current_map_id": map_id,
        "paths": {
            "json": str(paths["json"].relative_to(REPO_ROOT)),
            "png": str(paths["png"].relative_to(REPO_ROOT)),
            "ppm": str(paths["ppm"].relative_to(REPO_ROOT)),
            "template_instruction": str(template_path.relative_to(REPO_ROOT)),
        },
        "metadata": state["metadata"],
        "layers": state["layers"],
        "room_instances": state["room_instances"],
        "object_instances": state["object_instances"],
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Simple Demo Layered Map Editor</title>
  <style>
    :root {
      --bg: #f5f2ea;
      --panel: rgba(252, 249, 242, 0.96);
      --line: #d5ccbb;
      --line-strong: #b3a48a;
      --ink: #2e2a23;
      --accent: #8a6a2a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #faf7f1 0%, #ede5d7 100%);
      color: var(--ink);
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr) 360px;
      gap: 16px;
      padding: 16px;
    }
    .panel, .canvas-wrap {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(70, 56, 34, 0.08);
    }
    .panel-scroll {
      max-height: calc(100vh - 64px);
      overflow-y: auto;
      padding-right: 6px;
    }
    .tab-list {
      display: flex;
      gap: 8px;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    .tab-btn {
      border: 1px solid #d4c9b6;
      border-radius: 999px;
      padding: 8px 14px;
      background: #fffaf2;
      color: #6a5d47;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .tab-btn.active {
      background: #2f2921;
      color: #fffaf2;
      border-color: #2f2921;
    }
    .tab-panel {
      display: none;
      margin-top: 18px;
    }
    .tab-panel.active {
      display: block;
    }
    .canvas-wrap {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    h1, h2, h3 {
      margin: 0 0 8px;
    }
    h1 { font-size: 24px; }
    h2 { font-size: 20px; }
    h3 {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #6a5d47;
    }
    p, .small {
      margin: 0;
      line-height: 1.45;
      color: #655945;
      font-size: 14px;
    }
    .section {
      margin-top: 18px;
      display: grid;
      gap: 8px;
    }
    .buttons, .mode-list, .palette-list, .instance-list {
      display: grid;
      gap: 8px;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .swatch, .mode-item, .instance-item {
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid #ddd3c2;
      border-radius: 10px;
      padding: 10px;
      background: #fffaf2;
      cursor: pointer;
      user-select: none;
    }
    .swatch.active, .mode-item.active, .instance-item.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(138, 106, 42, 0.18);
    }
    .instance-item {
      display: block;
    }
    .instance-title {
      font-weight: 700;
      color: #2f2921;
    }
    .instance-meta {
      margin-top: 4px;
      font-size: 12px;
      color: #7a6c57;
    }
    .chip {
      width: 22px;
      height: 22px;
      border-radius: 6px;
      border: 1px solid rgba(0, 0, 0, 0.18);
      flex: 0 0 auto;
    }
    button, select, input, textarea {
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 12px;
      background: #2f2921;
      color: white;
      cursor: pointer;
    }
    button.alt {
      background: #d9ccb3;
      color: #2f2921;
    }
    .status {
      padding: 10px 12px;
      background: #f2eadc;
      border-radius: 10px;
      border: 1px solid #ddd1bc;
      font-size: 14px;
    }
    .canvas-box {
      overflow: auto;
      border: 1px solid #d6cdbe;
      border-radius: 12px;
      background: white;
      max-height: calc(100vh - 150px);
    }
    canvas {
      display: block;
      image-rendering: pixelated;
      cursor: crosshair;
    }
    textarea, input[type="text"], select {
      width: 100%;
      border: 1px solid #d4c9b6;
      border-radius: 10px;
      padding: 10px 12px;
      resize: vertical;
      background: #fffaf2;
      color: #2f2921;
      line-height: 1.45;
    }
    textarea {
      min-height: 120px;
    }
    .inline {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .file-list {
      font-size: 13px;
      line-height: 1.5;
      color: #6a5d47;
      word-break: break-word;
    }
    .hint-box {
      padding: 10px 12px;
      border: 1px dashed #ccbfa8;
      border-radius: 10px;
      background: #fbf5e9;
      color: #6b5f49;
      font-size: 13px;
      line-height: 1.45;
    }
    .attribute-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
    }
    .instance-scroll {
      max-height: 260px;
      overflow-y: auto;
      padding-right: 6px;
    }
    @media (max-width: 1180px) {
      .shell {
        grid-template-columns: 1fr;
      }
      .canvas-box {
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel">
      <h1>Map Manager</h1>
      <p>Open an existing map or create a new one. Editing happens on three separate layers: occupancy, room space, and object instances.</p>

      <div class="section">
        <h3>Existing Maps</h3>
        <select id="mapSelect"></select>
        <div class="buttons">
          <button id="openMapBtn">Open Selected Map</button>
        </div>
      </div>

      <div class="section">
        <h3>Create New Map</h3>
        <input id="newMapIdInput" type="text" placeholder="new_map_1">
        <input id="newMapNameInput" type="text" placeholder="New Map 1">
        <textarea id="newMapDescriptionInput" placeholder="Optional description for the new map."></textarea>
        <div class="buttons">
          <button id="newBlankBtn" class="alt">Create Blank Map</button>
          <button id="newSampleBtn" class="alt">Create Sample Map</button>
        </div>
      </div>

      <div class="section">
        <h3>Layer Mode</h3>
        <div id="layerModeList" class="mode-list"></div>
      </div>

      <div class="section">
        <h3>Palette</h3>
        <div id="palette" class="palette-list"></div>
      </div>

      <div class="section">
        <h3>Actions</h3>
        <div class="buttons">
          <button id="saveBtn">Save Map</button>
        </div>
      </div>

      <div class="section">
        <h3>Files</h3>
        <div class="file-list" id="paths"></div>
      </div>
    </aside>

    <main class="canvas-wrap">
      <div class="toolbar">
        <label class="inline">
          Zoom
          <input id="zoomSlider" type="range" min="3" max="10" step="1" value="6">
          <span id="zoomValue">6x</span>
        </label>
        <label class="inline">
          Brush
          <select id="brushSizeSelect">
            <option value="1">1x1</option>
            <option value="2">2x2</option>
            <option value="4">4x4</option>
            <option value="8">8x8</option>
            <option value="16">16x16</option>
          </select>
        </label>
        <label class="inline">
          Tool
          <select id="paintToolSelect">
            <option value="brush">Brush</option>
            <option value="rectangle">Rectangle</option>
          </select>
        </label>
        <label class="inline">
          <input id="showRoomsToggle" type="checkbox" checked>
          Show rooms
        </label>
        <label class="inline">
          <input id="showObjectsToggle" type="checkbox" checked>
          Show objects
        </label>
        <label class="inline">
          <input id="gridToggle" type="checkbox" checked>
          Show grid
        </label>
      </div>

      <div id="status" class="status">Loading...</div>

      <div class="canvas-box">
        <canvas id="gridCanvas" width="768" height="768"></canvas>
      </div>
    </main>

    <aside class="panel">
      <div class="panel-scroll">
        <h2>Metadata & Instances</h2>
        <p>Use the metadata fields to define the map file identity. In the object layer, you paint instance IDs rather than generic object categories.</p>
        <div class="tab-list" id="metadataTabList">
          <button class="tab-btn active" data-tab="map" type="button">Map</button>
          <button class="tab-btn" data-tab="room" type="button">Room</button>
          <button class="tab-btn" data-tab="object" type="button">Object</button>
        </div>

        <div id="metadataMapPanel" class="tab-panel active">
          <div class="section">
            <h3>Map ID</h3>
            <input id="mapIdInput" type="text" placeholder="simple_demo_map">
          </div>

          <div class="section">
            <h3>Map Name</h3>
            <input id="mapNameInput" type="text" placeholder="Simple Demo Map">
          </div>

          <div class="section">
            <h3>Description</h3>
            <textarea id="mapDescriptionInput" placeholder="Describe the map and intended semantics."></textarea>
          </div>
        </div>

        <div id="metadataRoomPanel" class="tab-panel">
          <div class="section">
            <h3>New Room Instance</h3>
            <select id="roomCategorySelect"></select>
            <input id="roomNameInput" type="text" placeholder="Optional room instance name">
            <div class="buttons">
              <button id="addRoomBtn">Add Room Instance</button>
              <button id="deleteRoomBtn" class="alt">Delete Selected Room</button>
            </div>
          </div>

          <div class="section">
            <h3>Room Instances</h3>
            <div class="hint-box" id="selectedRoomInfo">
              No room selected.
            </div>
            <div class="hint-box" id="selectedRoomEditorHint">
              Select a room instance from the list to edit its attributes.
            </div>
            <div id="roomAttributeList" class="instance-list"></div>
            <div class="buttons">
              <button id="addRoomAttributeBtn" class="alt">Add Room Attribute</button>
            </div>
            <div class="instance-scroll">
              <div id="roomInstanceList" class="instance-list"></div>
            </div>
          </div>
        </div>

        <div id="metadataObjectPanel" class="tab-panel">
          <div class="section">
            <h3>New Object Instance</h3>
            <select id="objectCategorySelect"></select>
            <input id="objectNameInput" type="text" placeholder="Optional instance name">
            <div class="buttons">
              <button id="addObjectBtn">Add Object Instance</button>
              <button id="deleteObjectBtn" class="alt">Delete Selected Instance</button>
            </div>
          </div>

          <div class="section">
            <h3>Object Instances</h3>
            <div class="hint-box" id="selectedObjectInfo">
              No object selected.
            </div>
            <div class="hint-box" id="selectedObjectEditorHint">
              Select an object instance from the list to edit its attributes.
            </div>
            <div id="attributeList" class="instance-list"></div>
            <div class="buttons">
              <button id="addAttributeBtn" class="alt">Add Attribute</button>
            </div>
            <div class="hint-box" id="objectLayerHint">
              Switch to the object layer, select an instance below, and paint its footprint onto the grid. Choose “Erase object cells” to clear object cells.
            </div>
            <div class="hint-box">
              Rectangle mode:
              click once to set one corner, then click again to fill the whole rectangle with the currently selected occupancy tile, room label, or object instance.
            </div>
            <div class="instance-scroll">
              <div id="instanceList" class="instance-list"></div>
            </div>
          </div>
        </div>
      </div>
    </aside>
  </div>

  <script>
    const INITIAL_STATE = __INITIAL_STATE__;
    const GRID_SIZE = INITIAL_STATE.grid_size;

    let currentMapId = INITIAL_STATE.current_map_id;
    let mapSummaries = INITIAL_STATE.map_summaries.slice();
    let currentPaths = { ...INITIAL_STATE.paths };

    let occupancyGrid = INITIAL_STATE.layers.occupancy.map(row => row.slice());
    let roomGrid = INITIAL_STATE.layers.room.map(row => row.slice());
    let objectGrid = INITIAL_STATE.layers.object_instance.map(row => row.slice());
    let roomInstances = INITIAL_STATE.room_instances.map(item => ({ ...item }));
    let objectInstances = INITIAL_STATE.object_instances.map(item => ({ ...item }));

    let activeLayer = "occupancy";
    let currentOccupancyTile = "obstacle";
    let currentRoomInstanceId = 0;
    let currentObjectInstanceId = 0;
    let inspectedRoomInstanceId = 0;
    let inspectedObjectInstanceId = 0;
    let brushSize = 1;
    let paintTool = "brush";
    let zoom = 6;
    let roomOverlayEnabled = true;
    let objectOverlayEnabled = true;
    let showGrid = true;
    let isPainting = false;
    let lastCellKey = null;
    let rectangleStartCell = null;

    const canvas = document.getElementById("gridCanvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const zoomSlider = document.getElementById("zoomSlider");
    const zoomValue = document.getElementById("zoomValue");
    const brushSizeSelect = document.getElementById("brushSizeSelect");
    const paintToolSelect = document.getElementById("paintToolSelect");
    const showRoomsToggle = document.getElementById("showRoomsToggle");
    const showObjectsToggle = document.getElementById("showObjectsToggle");
    const gridToggle = document.getElementById("gridToggle");
    const mapSelect = document.getElementById("mapSelect");
    const newMapIdInput = document.getElementById("newMapIdInput");
    const newMapNameInput = document.getElementById("newMapNameInput");
    const newMapDescriptionInput = document.getElementById("newMapDescriptionInput");
    const layerModeList = document.getElementById("layerModeList");
    const paletteEl = document.getElementById("palette");
    const pathsEl = document.getElementById("paths");
    const metadataTabList = document.getElementById("metadataTabList");
    const metadataMapPanel = document.getElementById("metadataMapPanel");
    const metadataRoomPanel = document.getElementById("metadataRoomPanel");
    const metadataObjectPanel = document.getElementById("metadataObjectPanel");
    const mapIdInput = document.getElementById("mapIdInput");
    const mapNameInput = document.getElementById("mapNameInput");
    const mapDescriptionInput = document.getElementById("mapDescriptionInput");
    const roomCategorySelect = document.getElementById("roomCategorySelect");
    const roomNameInput = document.getElementById("roomNameInput");
    const roomInstanceListEl = document.getElementById("roomInstanceList");
    const selectedRoomInfoEl = document.getElementById("selectedRoomInfo");
    const selectedRoomEditorHintEl = document.getElementById("selectedRoomEditorHint");
    const roomAttributeListEl = document.getElementById("roomAttributeList");
    const objectCategorySelect = document.getElementById("objectCategorySelect");
    const objectNameInput = document.getElementById("objectNameInput");
    const instanceListEl = document.getElementById("instanceList");
    const selectedObjectInfoEl = document.getElementById("selectedObjectInfo");
    const selectedObjectEditorHintEl = document.getElementById("selectedObjectEditorHint");
    const attributeListEl = document.getElementById("attributeList");
    const objectLayerHint = document.getElementById("objectLayerHint");
    let currentMetadataTab = "map";

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function setMetadataTab(tabName, silent = false) {
      currentMetadataTab = tabName;
      for (const button of metadataTabList.querySelectorAll(".tab-btn")) {
        button.classList.toggle("active", button.dataset.tab === tabName);
      }
      metadataMapPanel.classList.toggle("active", tabName === "map");
      metadataRoomPanel.classList.toggle("active", tabName === "room");
      metadataObjectPanel.classList.toggle("active", tabName === "object");
      if (!silent) {
        setStatus(`Switched metadata panel to ${tabName}.`);
      }
    }

    function occupancyNameFromValue(value) {
      for (const [name, tile] of Object.entries(INITIAL_STATE.legends.occupancy)) {
        if (tile.value === value) return name;
      }
      return "free";
    }

    function selectedRoomInstance() {
      return roomInstances.find(item => item.id === currentRoomInstanceId) || null;
    }

    function inspectedRoomInstance() {
      return roomInstances.find(item => item.id === inspectedRoomInstanceId) || null;
    }

    function infoRoomInstance() {
      return inspectedRoomInstance() || selectedRoomInstance();
    }

    function objectInstanceMap() {
      return new Map(objectInstances.map(item => [item.id, item]));
    }

    function roomInstanceMap() {
      return new Map(roomInstances.map(item => [item.id, item]));
    }

    function selectedObjectInstance() {
      return objectInstances.find(item => item.id === currentObjectInstanceId) || null;
    }

    function inspectedObjectInstance() {
      return objectInstances.find(item => item.id === inspectedObjectInstanceId) || null;
    }

    function infoObjectInstance() {
      return inspectedObjectInstance() || selectedObjectInstance();
    }

    function nonEmptyAttributes(attributes) {
      if (!Array.isArray(attributes)) {
        return [];
      }
      return attributes
        .map(attribute => typeof attribute === "string" ? attribute.trim() : "")
        .filter(attribute => attribute.length > 0);
    }

    function renderSelectedObjectInfo() {
      const item = infoObjectInstance();
      if (!item) {
        selectedObjectInfoEl.innerHTML = "No object selected.";
        return;
      }
      const meta = INITIAL_STATE.legends.object_categories[item.category];
      const attributes = nonEmptyAttributes(item.attributes)
        .map(attribute => `<li>${attribute}</li>`)
        .join("");
      selectedObjectInfoEl.innerHTML = `
        <div><strong>Object:</strong> object_${item.id}</div>
        <div><strong>Category:</strong> ${meta.label}</div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
          <strong>Color:</strong>
          <span class="chip" style="background:${meta.color}"></span>
          <span>${meta.color}</span>
        </div>
        ${attributes ? `<div style="margin-top:8px;"><strong>Attributes:</strong></div><ul style="margin:6px 0 0 18px; padding:0;">${attributes}</ul>` : ""}
      `;
    }

    function renderSelectedRoomInfo() {
      const item = infoRoomInstance();
      if (!item) {
        selectedRoomInfoEl.innerHTML = "No room selected.";
        return;
      }
      const meta = INITIAL_STATE.legends.room_categories[item.category];
      const attributes = nonEmptyAttributes(item.attributes)
        .map(attribute => `<li>${attribute}</li>`)
        .join("");
      selectedRoomInfoEl.innerHTML = `
        <div><strong>Room:</strong> room_${item.id}</div>
        <div><strong>Category:</strong> ${meta.label}</div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
          <strong>Color:</strong>
          <span class="chip" style="background:${meta.color}"></span>
          <span>${meta.color}</span>
        </div>
        ${attributes ? `<div style="margin-top:8px;"><strong>Attributes:</strong></div><ul style="margin:6px 0 0 18px; padding:0;">${attributes}</ul>` : ""}
      `;
    }

    function renderRoomAttributeEditor() {
      const item = selectedRoomInstance();
      roomAttributeListEl.innerHTML = "";

      if (!item) {
        selectedRoomEditorHintEl.textContent = "Select a room instance from the list to edit its attributes.";
        return;
      }

      if (!Array.isArray(item.attributes)) {
        item.attributes = [];
      }

      selectedRoomEditorHintEl.textContent = `Editing attributes for ${item.name}.`;

      item.attributes.forEach((attribute, index) => {
        const row = document.createElement("div");
        row.className = "attribute-row";
        const input = document.createElement("input");
        input.type = "text";
        input.value = attribute;
        input.placeholder = "e.g. room beside the kitchen";
        input.addEventListener("input", () => {
          item.attributes[index] = input.value;
          renderSelectedRoomInfo();
          setStatus(`Updated room attribute ${index + 1} for ${item.name}. Save the map to persist it.`);
        });

        const removeBtn = document.createElement("button");
        removeBtn.className = "alt";
        removeBtn.textContent = "Remove";
        removeBtn.onclick = () => {
          item.attributes.splice(index, 1);
          renderRoomAttributeEditor();
          renderSelectedRoomInfo();
          setStatus(`Removed a room attribute from ${item.name}. Save the map to persist it.`);
        };

        row.appendChild(input);
        row.appendChild(removeBtn);
        roomAttributeListEl.appendChild(row);
      });
    }

    function renderAttributeEditor() {
      const item = selectedObjectInstance();
      attributeListEl.innerHTML = "";

      if (!item) {
        selectedObjectEditorHintEl.textContent = "Select an object instance from the list to edit its attributes.";
        return;
      }

      if (!Array.isArray(item.attributes)) {
        item.attributes = [];
      }

      selectedObjectEditorHintEl.textContent = `Editing attributes for ${item.name}.`;

      item.attributes.forEach((attribute, index) => {
        const row = document.createElement("div");
        row.className = "attribute-row";
        const input = document.createElement("input");
        input.type = "text";
        input.value = attribute;
        input.placeholder = "e.g. chair beside the largest table";
        input.addEventListener("input", () => {
          item.attributes[index] = input.value;
          renderSelectedObjectInfo();
          setStatus(`Updated attribute ${index + 1} for ${item.name}. Save the map to persist it.`);
        });

        const removeBtn = document.createElement("button");
        removeBtn.className = "alt";
        removeBtn.textContent = "Remove";
        removeBtn.onclick = () => {
          item.attributes.splice(index, 1);
          renderAttributeEditor();
          renderSelectedObjectInfo();
          setStatus(`Removed an attribute from ${item.name}. Save the map to persist it.`);
        };

        row.appendChild(input);
        row.appendChild(removeBtn);
        attributeListEl.appendChild(row);
      });
    }

    function clearInspectedObject(shouldRedraw = true) {
      inspectedObjectInstanceId = 0;
      renderSelectedObjectInfo();
      if (shouldRedraw) {
        drawGrid();
      }
    }

    function clearInspectedRoom(shouldRedraw = true) {
      inspectedRoomInstanceId = 0;
      renderSelectedRoomInfo();
      if (shouldRedraw) {
        drawGrid();
      }
    }

    function effectiveShowRooms() {
      return activeLayer !== "occupancy" && roomOverlayEnabled;
    }

    function effectiveShowObjects() {
      return activeLayer !== "occupancy" && objectOverlayEnabled;
    }

    function syncOverlayControls() {
      const disableSemanticOverlays = activeLayer === "occupancy";
      showRoomsToggle.checked = disableSemanticOverlays ? false : roomOverlayEnabled;
      showObjectsToggle.checked = disableSemanticOverlays ? false : objectOverlayEnabled;
      showRoomsToggle.disabled = disableSemanticOverlays;
      showObjectsToggle.disabled = disableSemanticOverlays;
    }

    function clearRectangleSelection(shouldRedraw = true) {
      rectangleStartCell = null;
      if (shouldRedraw) {
        drawGrid();
      }
    }

    function cellKey(row, col) {
      return `${row}:${col}`;
    }

    function cloneGrid(source) {
      return source.map(row => row.slice());
    }

    function blankGrid(fillValue = 0) {
      return Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(fillValue));
    }

    function sanitizeMapId(value) {
      const cleaned = value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
      return cleaned || "new_map";
    }

    function suggestNextMapId(base = "new_map") {
      const normalizedBase = sanitizeMapId(base);
      const existing = new Set(mapSummaries.map(item => item.map_id));
      if (!existing.has(normalizedBase)) {
        return normalizedBase;
      }
      let index = 1;
      while (existing.has(`${normalizedBase}_${index}`)) {
        index += 1;
      }
      return `${normalizedBase}_${index}`;
    }

    function titleFromMapId(mapId) {
      return mapId
        .split("_")
        .filter(Boolean)
        .map(part => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
    }

    function syncNewMapForm(base = "new_map", force = false) {
      const suggestedId = suggestNextMapId(base);
      newMapIdInput.value = suggestedId;
      if (force || !newMapNameInput.value.trim() || newMapNameInput.dataset.autofill !== "manual") {
        newMapNameInput.value = titleFromMapId(suggestedId) || "New Map";
        newMapNameInput.dataset.autofill = "auto";
      }
      if (force || !newMapDescriptionInput.value.trim() || newMapDescriptionInput.dataset.autofill !== "manual") {
        newMapDescriptionInput.value = "";
        newMapDescriptionInput.dataset.autofill = "auto";
      }
    }

    function refreshMapSelect() {
      const selected = mapSelect.value || currentMapId;
      mapSelect.innerHTML = "";
      for (const item of mapSummaries) {
        const option = document.createElement("option");
        option.value = item.map_id;
        option.textContent = `${item.name} (${item.map_id})`;
        if (item.map_id === selected || item.map_id === currentMapId) {
          option.selected = true;
        }
        mapSelect.appendChild(option);
      }
    }

    function renderLayerModes() {
      layerModeList.innerHTML = "";
      for (const [name, label] of INITIAL_STATE.layer_modes) {
        const item = document.createElement("div");
        item.className = "mode-item" + (activeLayer === name ? " active" : "");
        item.innerHTML = `
          <div class="instance-title">${label}</div>
          <div class="instance-meta">${name}</div>
        `;
        item.onclick = () => {
          activeLayer = name;
          clearRectangleSelection(false);
          renderLayerModes();
          syncOverlayControls();
          renderPalette();
          renderRoomInstanceList();
          renderRoomAttributeEditor();
          renderInstanceList();
          drawGrid();
          setStatus(`Active layer set to ${label}.`);
        };
        layerModeList.appendChild(item);
      }
    }

    function renderPalette() {
      paletteEl.innerHTML = "";

      if (activeLayer === "occupancy") {
        for (const [name, tile] of Object.entries(INITIAL_STATE.legends.occupancy)) {
          const item = document.createElement("label");
          item.className = "swatch" + (currentOccupancyTile === name ? " active" : "");
          item.innerHTML = `
            <span class="chip" style="background:${tile.color}"></span>
            <span>${tile.label}</span>
          `;
          item.onclick = () => {
            currentOccupancyTile = name;
            renderPalette();
            setStatus(`Occupancy brush set to ${tile.label}.`);
          };
          paletteEl.appendChild(item);
        }
        return;
      }

      if (activeLayer === "room") {
        const hint = document.createElement("div");
        hint.className = "hint-box";
        const selected = selectedRoomInstance();
        hint.textContent = selected
          ? `Painting room instance ${selected.name} (${selected.category}).`
          : "Select a room instance on the right, or use the room eraser entry to clear room cells.";
        paletteEl.appendChild(hint);
        return;
      }

      const hint = document.createElement("div");
      hint.className = "hint-box";
      const selected = selectedObjectInstance();
      hint.textContent = selected
        ? `Painting object instance ${selected.name} (${selected.category}).`
        : "Select an object instance on the right, or use the erase entry to clear object cells.";
      paletteEl.appendChild(hint);
    }

    function renderObjectCategorySelect() {
      objectCategorySelect.innerHTML = "";
      for (const [name, meta] of Object.entries(INITIAL_STATE.legends.object_categories)) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = meta.label;
        objectCategorySelect.appendChild(option);
      }
    }

    function renderRoomCategorySelect() {
      roomCategorySelect.innerHTML = "";
      for (const [name, meta] of Object.entries(INITIAL_STATE.legends.room_categories)) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = meta.label;
        roomCategorySelect.appendChild(option);
      }
    }

    function renderRoomInstanceList() {
      roomInstanceListEl.innerHTML = "";

      const clearItem = document.createElement("div");
      clearItem.className = "instance-item" + (currentRoomInstanceId === 0 ? " active" : "");
      clearItem.innerHTML = `
        <div class="instance-title">Erase room cells</div>
        <div class="instance-meta">Paint 0 into the room layer.</div>
      `;
      clearItem.onclick = () => {
        currentRoomInstanceId = 0;
        inspectedRoomInstanceId = 0;
        renderRoomInstanceList();
        renderSelectedRoomInfo();
        renderRoomAttributeEditor();
        renderPalette();
        drawGrid();
        setStatus("Room eraser selected.");
      };
      roomInstanceListEl.appendChild(clearItem);

      for (const item of roomInstances) {
        const meta = INITIAL_STATE.legends.room_categories[item.category];
        const div = document.createElement("div");
        div.className = "instance-item" + (item.id === currentRoomInstanceId ? " active" : "");
        div.innerHTML = `
          <div style="display:flex; align-items:center; gap:10px;">
            <span class="chip" style="background:${meta.color}"></span>
            <div>
              <div class="instance-title">${item.name}</div>
              <div class="instance-meta">${meta.label} | room_id=${item.id}</div>
            </div>
          </div>
        `;
        div.onclick = () => {
          setMetadataTab("room", true);
          currentRoomInstanceId = item.id;
          inspectedRoomInstanceId = item.id;
          renderRoomInstanceList();
          renderSelectedRoomInfo();
          renderRoomAttributeEditor();
          renderPalette();
          drawGrid();
          setStatus(`Selected room instance ${item.name}.`);
        };
        roomInstanceListEl.appendChild(div);
      }
    }

    function renderInstanceList() {
      instanceListEl.innerHTML = "";

      const clearItem = document.createElement("div");
      clearItem.className = "instance-item" + (currentObjectInstanceId === 0 ? " active" : "");
      clearItem.innerHTML = `
        <div class="instance-title">Erase object cells</div>
        <div class="instance-meta">Paint 0 into the object-instance layer.</div>
      `;
      clearItem.onclick = () => {
        currentObjectInstanceId = 0;
        inspectedObjectInstanceId = 0;
        renderInstanceList();
        renderSelectedObjectInfo();
        renderAttributeEditor();
        renderPalette();
        drawGrid();
        setStatus("Object eraser selected.");
      };
      instanceListEl.appendChild(clearItem);

      for (const item of objectInstances) {
        const meta = INITIAL_STATE.legends.object_categories[item.category];
        const div = document.createElement("div");
        div.className = "instance-item" + (item.id === currentObjectInstanceId ? " active" : "");
        div.innerHTML = `
          <div style="display:flex; align-items:center; gap:10px;">
            <span class="chip" style="background:${meta.color}"></span>
            <div>
              <div class="instance-title">${item.name}</div>
              <div class="instance-meta">${meta.label} | instance_id=${item.id}</div>
            </div>
          </div>
        `;
        div.onclick = () => {
          setMetadataTab("object", true);
          currentObjectInstanceId = item.id;
          inspectedObjectInstanceId = item.id;
          renderInstanceList();
          renderSelectedObjectInfo();
          renderAttributeEditor();
          renderPalette();
          drawGrid();
          setStatus(`Selected object instance ${item.name}.`);
        };
        instanceListEl.appendChild(div);
      }

      objectLayerHint.style.opacity = activeLayer === "object_instance" ? "1" : "0.75";
    }

    function renderPaths() {
      pathsEl.innerHTML = `${currentPaths.json}<br>${currentPaths.png}<br>${currentPaths.ppm}`;
    }

    function syncMetadataFromState(metadata) {
      mapIdInput.value = metadata.map_id || "simple_demo_map";
      mapNameInput.value = metadata.name || "Simple Demo Map";
      mapDescriptionInput.value = metadata.description || "";
    }

    function buildMetadataPayload() {
      return {
        map_id: mapIdInput.value.trim() || "simple_demo_map",
        name: mapNameInput.value.trim() || "Simple Demo Map",
        description: mapDescriptionInput.value,
      };
    }

    function buildNewMapMetadataPayload() {
      const mapId = sanitizeMapId(newMapIdInput.value || "new_map");
      const name = newMapNameInput.value.trim() || titleFromMapId(mapId) || "New Map";
      return {
        map_id: mapId,
        name,
        description: newMapDescriptionInput.value,
      };
    }

    function resizeCanvas() {
      const size = GRID_SIZE * zoom;
      canvas.width = size;
      canvas.height = size;
      zoomValue.textContent = `${zoom}x`;
      drawGrid();
    }

    function drawGrid() {
      ctx.globalAlpha = 1;
      const roomMap = roomInstanceMap();
      const instanceMap = objectInstanceMap();

      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          const occupancyName = occupancyNameFromValue(occupancyGrid[row][col]);
          ctx.fillStyle = INITIAL_STATE.legends.occupancy[occupancyName].color;
          ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
        }
      }

      if (effectiveShowRooms()) {
        ctx.globalAlpha = 0.62;
        for (let row = 0; row < GRID_SIZE; row += 1) {
          for (let col = 0; col < GRID_SIZE; col += 1) {
            const roomId = roomGrid[row][col];
            if (!roomId) continue;
            const roomItem = roomMap.get(roomId);
            if (!roomItem) continue;
            ctx.fillStyle = INITIAL_STATE.legends.room_categories[roomItem.category].color;
            ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
          }
        }
        ctx.globalAlpha = 1;
      }

      const highlightedRoomInstanceId = inspectedRoomInstanceId || currentRoomInstanceId;
      if (activeLayer !== "occupancy" && highlightedRoomInstanceId !== 0) {
        const highlightStroke = "#111111";
        const highlightFill = "#111111";
        for (let row = 0; row < GRID_SIZE; row += 1) {
          for (let col = 0; col < GRID_SIZE; col += 1) {
            if (roomGrid[row][col] !== highlightedRoomInstanceId) {
              continue;
            }

            ctx.globalAlpha = 0.82;
            ctx.fillStyle = highlightFill;
            ctx.fillRect(col * zoom, row * zoom, zoom, zoom);

            ctx.globalAlpha = 1;
            ctx.strokeStyle = highlightStroke;
            ctx.lineWidth = Math.max(1, Math.floor(zoom / 4));
            ctx.strokeRect(
              col * zoom + ctx.lineWidth / 2,
              row * zoom + ctx.lineWidth / 2,
              zoom - ctx.lineWidth,
              zoom - ctx.lineWidth
            );
          }
        }
        ctx.globalAlpha = 1;
      }

      if (effectiveShowObjects()) {
        for (let row = 0; row < GRID_SIZE; row += 1) {
          for (let col = 0; col < GRID_SIZE; col += 1) {
            const objectId = objectGrid[row][col];
            if (!objectId) continue;
            const item = instanceMap.get(objectId);
            if (!item) continue;
            ctx.globalAlpha = 0.96;
            ctx.fillStyle = INITIAL_STATE.legends.object_categories[item.category].color;
            ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
          }
        }
        ctx.globalAlpha = 1;
      }

      const highlightedObjectInstanceId = inspectedObjectInstanceId || currentObjectInstanceId;
      if (activeLayer !== "occupancy" && highlightedObjectInstanceId !== 0) {
        const highlightStroke = "#111111";
        const highlightFill = "#111111";
        for (let row = 0; row < GRID_SIZE; row += 1) {
          for (let col = 0; col < GRID_SIZE; col += 1) {
            if (objectGrid[row][col] !== highlightedObjectInstanceId) {
              continue;
            }

            ctx.globalAlpha = 0.92;
            ctx.fillStyle = highlightFill;
            ctx.fillRect(col * zoom, row * zoom, zoom, zoom);

            ctx.globalAlpha = 1;
            ctx.strokeStyle = highlightStroke;
            ctx.lineWidth = Math.max(1, Math.floor(zoom / 4));
            ctx.strokeRect(
              col * zoom + ctx.lineWidth / 2,
              row * zoom + ctx.lineWidth / 2,
              zoom - ctx.lineWidth,
              zoom - ctx.lineWidth
            );
          }
        }
        ctx.globalAlpha = 1;
      }

      if (paintTool === "rectangle" && rectangleStartCell) {
        const [startRow, startCol] = rectangleStartCell;
        ctx.strokeStyle = "#2E6F95";
        ctx.lineWidth = Math.max(2, Math.floor(zoom / 3));
        ctx.strokeRect(
          startCol * zoom + ctx.lineWidth / 2,
          startRow * zoom + ctx.lineWidth / 2,
          zoom - ctx.lineWidth,
          zoom - ctx.lineWidth
        );
      }

      if (!showGrid) return;

      for (let i = 0; i <= GRID_SIZE; i += 1) {
        const color = i % 16 === 0 ? "#B3A48A" : "#D5CCBB";
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, i * zoom + 0.5);
        ctx.lineTo(canvas.width, i * zoom + 0.5);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(i * zoom + 0.5, 0);
        ctx.lineTo(i * zoom + 0.5, canvas.height);
        ctx.stroke();
      }
    }

    function eventToCell(event) {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const col = Math.floor(x / zoom);
      const row = Math.floor(y / zoom);
      if (row < 0 || row >= GRID_SIZE || col < 0 || col >= GRID_SIZE) return null;
      return [row, col];
    }

    function hoverDescription(row, col) {
      const occName = occupancyNameFromValue(occupancyGrid[row][col]);
      const roomId = roomGrid[row][col];
      const roomItem = roomInstances.find(item => item.id === roomId);
      const roomText = roomItem
        ? `${roomItem.name} (${roomItem.category})`
        : "none";
      const objectId = objectGrid[row][col];
      const objectItem = objectInstances.find(item => item.id === objectId);
      const objectText = objectItem
        ? `${objectItem.name} (${objectItem.category})`
        : "none";
      return `Cell (${row}, ${col}) | occupancy=${occName} | room=${roomText} | object=${objectText}`;
    }

    function paintCell(row, col) {
      let changed = false;
      const maxRow = Math.min(GRID_SIZE, row + brushSize);
      const maxCol = Math.min(GRID_SIZE, col + brushSize);

      for (let r = row; r < maxRow; r += 1) {
        for (let c = col; c < maxCol; c += 1) {
          if (activeLayer === "occupancy") {
            const nextValue = INITIAL_STATE.legends.occupancy[currentOccupancyTile].value;
            if (occupancyGrid[r][c] !== nextValue) {
              occupancyGrid[r][c] = nextValue;
              changed = true;
            }
          } else if (activeLayer === "room") {
            const nextValue = currentRoomInstanceId;
            if (roomGrid[r][c] !== nextValue) {
              roomGrid[r][c] = nextValue;
              changed = true;
            }
          } else {
            if (objectGrid[r][c] !== currentObjectInstanceId) {
              objectGrid[r][c] = currentObjectInstanceId;
              changed = true;
            }
          }
        }
      }

      if (!changed) return;

      drawGrid();
      if (activeLayer === "occupancy") {
        setStatus(`Painted occupancy layer with ${currentOccupancyTile} from (${row}, ${col}).`);
      } else if (activeLayer === "room") {
        const item = selectedRoomInstance();
        setStatus(`Painted room instance ${item ? item.name : currentRoomInstanceId} from (${row}, ${col}).`);
      } else if (currentObjectInstanceId === 0) {
        setStatus(`Erased object cells from (${row}, ${col}).`);
      } else {
        const item = selectedObjectInstance();
        setStatus(`Painted object instance ${item ? item.name : currentObjectInstanceId} from (${row}, ${col}).`);
      }
    }

    function fillRectangleBetween(startCell, endCell) {
      const startRow = Math.min(startCell[0], endCell[0]);
      const endRow = Math.max(startCell[0], endCell[0]);
      const startCol = Math.min(startCell[1], endCell[1]);
      const endCol = Math.max(startCell[1], endCell[1]);

      let changed = false;
      for (let row = startRow; row <= endRow; row += 1) {
        for (let col = startCol; col <= endCol; col += 1) {
          if (activeLayer === "occupancy") {
            const nextValue = INITIAL_STATE.legends.occupancy[currentOccupancyTile].value;
            if (occupancyGrid[row][col] !== nextValue) {
              occupancyGrid[row][col] = nextValue;
              changed = true;
            }
          } else if (activeLayer === "room") {
            const nextValue = currentRoomInstanceId;
            if (roomGrid[row][col] !== nextValue) {
              roomGrid[row][col] = nextValue;
              changed = true;
            }
          } else if (objectGrid[row][col] !== currentObjectInstanceId) {
            objectGrid[row][col] = currentObjectInstanceId;
            changed = true;
          }
        }
      }

      if (!changed) {
        return false;
      }

      drawGrid();
      if (activeLayer === "occupancy") {
        setStatus(
          `Filled occupancy rectangle (${startRow}, ${startCol}) -> (${endRow}, ${endCol}) with ${currentOccupancyTile}.`
        );
      } else if (activeLayer === "room") {
        const item = selectedRoomInstance();
        setStatus(
          `Filled room rectangle (${startRow}, ${startCol}) -> (${endRow}, ${endCol}) with ${item ? item.name : currentRoomInstanceId}.`
        );
      } else if (currentObjectInstanceId === 0) {
        setStatus(
          `Erased object rectangle (${startRow}, ${startCol}) -> (${endRow}, ${endCol}).`
        );
      } else {
        const item = selectedObjectInstance();
        setStatus(
          `Filled object rectangle (${startRow}, ${startCol}) -> (${endRow}, ${endCol}) with ${item ? item.name : currentObjectInstanceId}.`
        );
      }
      return true;
    }

    function handleRectangleClick(row, col) {
      if (!rectangleStartCell) {
        rectangleStartCell = [row, col];
        drawGrid();
        setStatus(`Rectangle start set at (${row}, ${col}). Click the opposite corner to fill the rectangle.`);
        return;
      }

      const startCell = rectangleStartCell;
      clearRectangleSelection(false);
      const changed = fillRectangleBetween(startCell, [row, col]);
      if (!changed) {
        setStatus(
          `Rectangle (${Math.min(startCell[0], row)}, ${Math.min(startCell[1], col)}) -> (${Math.max(startCell[0], row)}, ${Math.max(startCell[1], col)}) already had the selected value.`
        );
      }
    }

    function buildSavePayload() {
      return {
        original_map_id: currentMapId,
        metadata: buildMetadataPayload(),
        layers: {
          occupancy: occupancyGrid,
          room: roomGrid,
          object_instance: objectGrid,
        },
        room_instances: roomInstances,
        object_instances: objectInstances,
      };
    }

    function applyServerState(payload) {
      currentMapId = payload.current_map_id;
      mapSummaries = payload.map_summaries.slice();
      currentPaths = { ...payload.paths };
      occupancyGrid = payload.layers.occupancy.map(row => row.slice());
      roomGrid = payload.layers.room.map(row => row.slice());
      objectGrid = payload.layers.object_instance.map(row => row.slice());
      roomInstances = payload.room_instances.map(item => ({ ...item }));
      objectInstances = payload.object_instances.map(item => ({ ...item }));

      if (!roomInstances.some(item => item.id === currentRoomInstanceId)) {
        currentRoomInstanceId = 0;
      }
      if (!objectInstances.some(item => item.id === currentObjectInstanceId)) {
        currentObjectInstanceId = 0;
      }
      if (!roomInstances.some(item => item.id === inspectedRoomInstanceId)) {
        inspectedRoomInstanceId = 0;
      }
      if (!objectInstances.some(item => item.id === inspectedObjectInstanceId)) {
        inspectedObjectInstanceId = 0;
      }

      refreshMapSelect();
      syncMetadataFromState(payload.metadata);
      syncOverlayControls();
      syncNewMapForm("new_map", true);
      renderPaths();
      renderSelectedRoomInfo();
      renderRoomAttributeEditor();
      renderRoomInstanceList();
      renderSelectedObjectInfo();
      renderAttributeEditor();
      renderPalette();
      renderInstanceList();
      drawGrid();
    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed for ${path}`);
      }
      return await response.json();
    }

    async function saveCurrentMap() {
      setStatus("Saving...");
      const payload = await postJson("/save", buildSavePayload());
      applyServerState(payload.state);
      setStatus(payload.message);
    }

    async function openSelectedMap() {
      const targetId = mapSelect.value;
      if (!targetId) return;
      const payload = await postJson("/open", { map_id: targetId });
      applyServerState(payload.state);
      setStatus(payload.message);
    }

    async function createMap(template) {
      setStatus("Creating map...");
      const payload = await postJson("/new", {
        metadata: buildNewMapMetadataPayload(),
        template,
      });
      applyServerState(payload.state);
      setStatus(payload.message);
    }

    function nextLocalInstanceName(category) {
      let index = 1;
      const existing = new Set(objectInstances.map(item => item.name));
      while (existing.has(`${category}_${index}`)) {
        index += 1;
      }
      return `${category}_${index}`;
    }

    function addLocalObjectInstance() {
      const category = objectCategorySelect.value;
      const nextId = objectInstances.reduce((maxValue, item) => Math.max(maxValue, item.id), 0) + 1;
      const name = objectNameInput.value.trim() || nextLocalInstanceName(category);
      objectInstances.push({ id: nextId, category, name, attributes: [] });
      setMetadataTab("object", true);
      currentObjectInstanceId = nextId;
      inspectedObjectInstanceId = 0;
      objectNameInput.value = "";
      renderInstanceList();
      renderSelectedObjectInfo();
      renderAttributeEditor();
      renderPalette();
      setStatus(`Created local object instance ${name}. Save the map to persist it.`);
    }

    function nextLocalRoomName(category) {
      let index = 1;
      const existing = new Set(roomInstances.map(item => item.name));
      while (existing.has(`${category}_${index}`)) {
        index += 1;
      }
      return `${category}_${index}`;
    }

    function addLocalRoomInstance() {
      const category = roomCategorySelect.value;
      const nextId = roomInstances.reduce((maxValue, item) => Math.max(maxValue, item.id), 0) + 1;
      const name = roomNameInput.value.trim() || nextLocalRoomName(category);
      roomInstances.push({ id: nextId, category, name, attributes: [] });
      setMetadataTab("room", true);
      currentRoomInstanceId = nextId;
      inspectedRoomInstanceId = 0;
      roomNameInput.value = "";
      renderRoomInstanceList();
      renderSelectedRoomInfo();
      renderRoomAttributeEditor();
      renderPalette();
      drawGrid();
      setStatus(`Created local room instance ${name}. Save the map to persist it.`);
    }

    function deleteSelectedRoomInstance() {
      if (currentRoomInstanceId === 0) {
        setStatus("Select a concrete room instance before deleting it.");
        return;
      }
      const instance = selectedRoomInstance();
      roomInstances = roomInstances.filter(item => item.id !== currentRoomInstanceId);
      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          if (roomGrid[row][col] === currentRoomInstanceId) {
            roomGrid[row][col] = 0;
          }
        }
      }
      currentRoomInstanceId = 0;
      inspectedRoomInstanceId = 0;
      renderRoomInstanceList();
      renderSelectedRoomInfo();
      renderRoomAttributeEditor();
      renderPalette();
      drawGrid();
      setStatus(`Deleted ${instance ? instance.name : "the selected room instance"} locally. Save the map to persist it.`);
    }

    function deleteSelectedObjectInstance() {
      if (currentObjectInstanceId === 0) {
        setStatus("Select a concrete object instance before deleting it.");
        return;
      }
      const instance = selectedObjectInstance();
      objectInstances = objectInstances.filter(item => item.id !== currentObjectInstanceId);
      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          if (objectGrid[row][col] === currentObjectInstanceId) {
            objectGrid[row][col] = 0;
          }
        }
      }
      currentObjectInstanceId = 0;
      inspectedObjectInstanceId = 0;
      renderInstanceList();
      renderSelectedObjectInfo();
      renderAttributeEditor();
      renderPalette();
      drawGrid();
      setStatus(`Deleted ${instance ? instance.name : "the selected instance"} locally. Save the map to persist the change.`);
    }

    canvas.addEventListener("mousedown", (event) => {
      if (event.button !== 0) return;
      const cell = eventToCell(event);
      if (!cell) return;
      if (paintTool === "rectangle") {
        handleRectangleClick(cell[0], cell[1]);
        return;
      }
      isPainting = true;
      lastCellKey = null;
      paintCell(cell[0], cell[1]);
      lastCellKey = cellKey(cell[0], cell[1]);
    });

    canvas.addEventListener("mousemove", (event) => {
      const cell = eventToCell(event);
      if (!cell) return;

      if (!isPainting) {
        if (paintTool === "rectangle" && rectangleStartCell) {
          setStatus(
            `${hoverDescription(cell[0], cell[1])} | Rectangle start = (${rectangleStartCell[0]}, ${rectangleStartCell[1]})`
          );
        } else {
          setStatus(hoverDescription(cell[0], cell[1]));
        }
        return;
      }

      const key = cellKey(cell[0], cell[1]);
      if (key === lastCellKey) return;
      paintCell(cell[0], cell[1]);
      lastCellKey = key;
    });

    window.addEventListener("mouseup", () => {
      isPainting = false;
      lastCellKey = null;
    });

    canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      const cell = eventToCell(event);
      if (!cell) return;

      if (paintTool === "rectangle" && rectangleStartCell) {
        clearRectangleSelection();
        setStatus("Cleared the pending rectangle start point.");
        return;
      }

      if (activeLayer === "occupancy") {
        currentOccupancyTile = occupancyNameFromValue(occupancyGrid[cell[0]][cell[1]]);
        renderPalette();
        setStatus(`Picked occupancy tile ${currentOccupancyTile} from (${cell[0]}, ${cell[1]}).`);
        return;
      }

      if (activeLayer === "room") {
        setMetadataTab("room", true);
        inspectedRoomInstanceId = roomGrid[cell[0]][cell[1]];
        renderSelectedRoomInfo();
        drawGrid();
        if (inspectedRoomInstanceId === 0) {
          setStatus("Picked room eraser from an empty room cell.");
        } else {
          const item = inspectedRoomInstance();
          setStatus(`Inspecting room instance ${item ? item.name : inspectedRoomInstanceId} from (${cell[0]}, ${cell[1]}). Click outside the map to clear the preview.`);
        }
        return;
      }

      setMetadataTab("object", true);
      inspectedObjectInstanceId = objectGrid[cell[0]][cell[1]];
      renderSelectedObjectInfo();
      drawGrid();
      if (inspectedObjectInstanceId === 0) {
        setStatus("Picked object eraser from an empty object cell.");
      } else {
        const item = inspectedObjectInstance();
        setStatus(`Inspecting object instance ${item ? item.name : inspectedObjectInstanceId} from (${cell[0]}, ${cell[1]}). Click outside the map to clear the preview.`);
      }
    });

    document.addEventListener("mousedown", (event) => {
      if (inspectedObjectInstanceId === 0) {
        return;
      }
      if (event.target === canvas) {
        return;
      }
      clearInspectedObject();
      setStatus("Cleared temporary object preview.");
    });

    document.addEventListener("mousedown", (event) => {
      if (inspectedRoomInstanceId === 0) {
        return;
      }
      if (event.target === canvas) {
        return;
      }
      clearInspectedRoom();
      setStatus("Cleared temporary room preview.");
    });

    document.getElementById("saveBtn").addEventListener("click", async () => {
      try {
        await saveCurrentMap();
      } catch (error) {
        setStatus(error.message);
      }
    });

    document.getElementById("openMapBtn").addEventListener("click", async () => {
      try {
        await openSelectedMap();
      } catch (error) {
        setStatus(error.message);
      }
    });

    document.getElementById("newBlankBtn").addEventListener("click", async () => {
      try {
        await createMap("blank");
      } catch (error) {
        setStatus(error.message);
      }
    });

    document.getElementById("newSampleBtn").addEventListener("click", async () => {
      try {
        await createMap("sample");
      } catch (error) {
        setStatus(error.message);
      }
    });

    document.getElementById("addObjectBtn").addEventListener("click", () => {
      addLocalObjectInstance();
    });

    document.getElementById("deleteObjectBtn").addEventListener("click", () => {
      deleteSelectedObjectInstance();
    });

    document.getElementById("addRoomBtn").addEventListener("click", () => {
      addLocalRoomInstance();
    });

    metadataTabList.addEventListener("click", (event) => {
      const button = event.target.closest(".tab-btn");
      if (!button) {
        return;
      }
      setMetadataTab(button.dataset.tab || "map");
    });

    document.getElementById("deleteRoomBtn").addEventListener("click", () => {
      deleteSelectedRoomInstance();
    });

    document.getElementById("addRoomAttributeBtn").addEventListener("click", () => {
      const item = selectedRoomInstance();
      if (!item) {
        setStatus("Select a room instance before adding attributes.");
        return;
      }
      if (!Array.isArray(item.attributes)) {
        item.attributes = [];
      }
      item.attributes.push("");
      renderRoomAttributeEditor();
      renderSelectedRoomInfo();
      setStatus(`Added a new room attribute slot for ${item.name}.`);
    });

    document.getElementById("addAttributeBtn").addEventListener("click", () => {
      const item = selectedObjectInstance();
      if (!item) {
        setStatus("Select an object instance before adding attributes.");
        return;
      }
      if (!Array.isArray(item.attributes)) {
        item.attributes = [];
      }
      item.attributes.push("");
      renderAttributeEditor();
      renderSelectedObjectInfo();
      setStatus(`Added a new attribute slot for ${item.name}.`);
    });

    zoomSlider.addEventListener("input", () => {
      zoom = Number(zoomSlider.value);
      resizeCanvas();
    });

    brushSizeSelect.addEventListener("change", () => {
      brushSize = Number(brushSizeSelect.value);
      setStatus(`Brush size set to ${brushSize}x${brushSize}.`);
    });

    paintToolSelect.addEventListener("change", () => {
      paintTool = paintToolSelect.value;
      clearRectangleSelection(false);
      drawGrid();
      if (paintTool === "rectangle") {
        setStatus("Paint tool set to Rectangle. Click two corners to fill a rectangle.");
      } else {
        setStatus("Paint tool set to Brush.");
      }
    });

    showRoomsToggle.addEventListener("change", () => {
      roomOverlayEnabled = showRoomsToggle.checked;
      drawGrid();
      setStatus(`Room overlay ${roomOverlayEnabled ? "on" : "off"}.`);
    });

    showObjectsToggle.addEventListener("change", () => {
      objectOverlayEnabled = showObjectsToggle.checked;
      drawGrid();
      setStatus(`Object overlay ${objectOverlayEnabled ? "on" : "off"}.`);
    });

    gridToggle.addEventListener("change", () => {
      showGrid = gridToggle.checked;
      drawGrid();
      setStatus(`Grid overlay ${showGrid ? "on" : "off"}.`);
    });

    mapIdInput.addEventListener("input", () => {
      setStatus("Updated map ID. Save or create a new map to persist this filename target.");
    });
    mapNameInput.addEventListener("input", () => {
      setStatus("Updated map name. Save to persist the change.");
    });
    mapDescriptionInput.addEventListener("input", () => {
      setStatus("Updated map description. Save to persist the change.");
    });

    newMapIdInput.addEventListener("input", () => {
      if (newMapNameInput.dataset.autofill !== "manual") {
        const sanitized = sanitizeMapId(newMapIdInput.value || "new_map");
        newMapNameInput.value = titleFromMapId(sanitized) || "New Map";
        newMapNameInput.dataset.autofill = "auto";
      }
      setStatus("Updated new map ID. Use Create Blank Map or Create Sample Map to create it.");
    });

    newMapNameInput.addEventListener("input", () => {
      newMapNameInput.dataset.autofill = "manual";
      setStatus("Updated new map name.");
    });

    newMapDescriptionInput.addEventListener("input", () => {
      newMapDescriptionInput.dataset.autofill = "manual";
      setStatus("Updated new map description.");
    });

    document.addEventListener("keydown", async (event) => {
      if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA" || event.target.tagName === "SELECT") {
        return;
      }

      if (event.key === "1") {
        activeLayer = "occupancy";
        clearRectangleSelection(false);
        renderLayerModes();
        syncOverlayControls();
        renderPalette();
        renderRoomInstanceList();
        renderRoomAttributeEditor();
        renderInstanceList();
        drawGrid();
        setStatus("Active layer set to occupancy.");
      }
      if (event.key === "2") {
        activeLayer = "room";
        clearRectangleSelection(false);
        renderLayerModes();
        syncOverlayControls();
        renderPalette();
        renderRoomInstanceList();
        renderRoomAttributeEditor();
        renderInstanceList();
        drawGrid();
        setStatus("Active layer set to room.");
      }
      if (event.key === "3") {
        activeLayer = "object_instance";
        clearRectangleSelection(false);
        renderLayerModes();
        syncOverlayControls();
        renderPalette();
        renderRoomInstanceList();
        renderRoomAttributeEditor();
        renderInstanceList();
        drawGrid();
        setStatus("Active layer set to object_instance.");
      }
      if (event.key === "b" || event.key === "B") {
        paintTool = "brush";
        paintToolSelect.value = "brush";
        clearRectangleSelection(false);
        drawGrid();
        setStatus("Paint tool set to Brush.");
      }
      if (event.key === "r" || event.key === "R") {
        paintTool = "rectangle";
        paintToolSelect.value = "rectangle";
        clearRectangleSelection(false);
        drawGrid();
        setStatus("Paint tool set to Rectangle. Click two corners to fill a rectangle.");
      }
      if (event.key === "Escape" && paintTool === "rectangle" && rectangleStartCell) {
        clearRectangleSelection();
        setStatus("Cleared the pending rectangle start point.");
      }
      if (event.key === "s" || event.key === "S") {
        event.preventDefault();
        try {
          await saveCurrentMap();
        } catch (error) {
          setStatus(error.message);
        }
      }
    });

    renderRoomCategorySelect();
    renderObjectCategorySelect();
    refreshMapSelect();
    renderLayerModes();
    syncOverlayControls();
    renderSelectedRoomInfo();
    renderRoomAttributeEditor();
    renderRoomInstanceList();
    renderSelectedObjectInfo();
    renderAttributeEditor();
    renderPalette();
    renderInstanceList();
    renderPaths();
    syncMetadataFromState(INITIAL_STATE.metadata);
    syncNewMapForm("new_map", true);
    setMetadataTab("map", true);
    brushSizeSelect.value = "1";
    paintToolSelect.value = "brush";
    resizeCanvas();
    setStatus("Loaded layered map editor. Open a map or create a new one, then edit occupancy, room, and object-instance layers separately.");
  </script>
</body>
</html>
"""


class SimpleMapHandler(BaseHTTPRequestHandler):
    def _build_index_html(self) -> bytes:
        state = load_map_state(DEFAULT_MAP_ID)
        map_id = state["metadata"]["map_id"]  # type: ignore[index]
        html = HTML_PAGE.replace(
            "__INITIAL_STATE__",
            json.dumps(build_client_state(map_id, state), ensure_ascii=False),
        )
        return html.encode("utf-8")

    def _send_bytes(
        self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(
        self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send_bytes(self._build_index_html(), "text/html; charset=utf-8")
            return

        if self.path == "/favicon.ico":
            self.send_error(HTTPStatus.NO_CONTENT)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            content = self._build_index_html()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            return

        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:  # noqa: N802
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as exc:  # pragma: no cover - defensive runtime validation.
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        try:
            if self.path == "/save":
                original_map_id = payload.get("original_map_id")
                if original_map_id is not None and not isinstance(original_map_id, str):
                    raise ValueError("original_map_id must be a string when provided.")
                state = validate_state_payload(payload)
                target_id = save_map_state(original_map_id, state)
                current_state = load_map_state(target_id)
                self._send_json(
                    {
                        "message": f"Saved map '{target_id}' and refreshed its preview assets.",
                        "state": build_client_state(target_id, current_state),
                    }
                )
                return

            if self.path == "/open":
                map_id = payload.get("map_id")
                if not isinstance(map_id, str) or not map_id.strip():
                    raise ValueError("map_id must be a non-empty string.")
                current_state = load_map_state(map_id)
                target_id = current_state["metadata"]["map_id"]  # type: ignore[index]
                self._send_json(
                    {
                        "message": f"Opened map '{target_id}'.",
                        "state": build_client_state(target_id, current_state),
                    }
                )
                return

            if self.path == "/new":
                metadata = validate_metadata(payload.get("metadata"))
                template = payload.get("template", "blank")
                if template not in {"blank", "sample"}:
                    raise ValueError("template must be 'blank' or 'sample'.")
                target_id = sanitize_map_id(metadata["map_id"])
                if map_paths(target_id)["json"].exists():
                    raise ValueError(
                        f"Map '{target_id}' already exists. Pick a new map_id or open it."
                    )
                state = (
                    make_sample_state(target_id) if template == "sample" else make_blank_state(target_id)
                )
                state["metadata"] = {
                    "map_id": target_id,
                    "name": metadata["name"],
                    "description": metadata["description"],
                }
                save_map_state(None, state)
                current_state = load_map_state(target_id)
                self._send_json(
                    {
                        "message": f"Created new {template} map '{target_id}'.",
                        "state": build_client_state(target_id, current_state),
                    }
                )
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found.")
        except Exception as exc:  # pragma: no cover - defensive runtime validation.
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        return


def initialize_assets() -> None:
    ensure_default_map()
    default_state = load_map_state(DEFAULT_MAP_ID)
    save_map_state(DEFAULT_MAP_ID, default_state)


def run_server(host: str, port: int, no_browser: bool) -> None:
    initialize_assets()

    with ThreadingHTTPServer((host, port), SimpleMapHandler) as server:
        actual_host, actual_port = server.server_address[:2]
        display_host = "127.0.0.1" if actual_host == "0.0.0.0" else actual_host
        url = f"http://{display_host}:{actual_port}"

        print("Layered simple map editor is ready.")
        print(f"Open this URL in your browser: {url}")
        print(f"Map directory: {RESOURCE_DIR}")
        print("Press Ctrl+C to stop the server.")

        if not no_browser:
            webbrowser.open(url)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\\nServer stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the layered simple map editor.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the local editor server to. Use 0.0.0.0 if needed.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8876,
        help="Port to bind the local editor server to.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not try to open a browser automatically.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Ensure the default sample map exists, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_only:
        initialize_assets()
        print(f"Initialized map assets in {RESOURCE_DIR}")
        return
    run_server(args.host, args.port, args.no_browser)


if __name__ == "__main__":
    main()
