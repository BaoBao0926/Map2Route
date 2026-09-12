#!/usr/bin/env python3
"""Browser-based instruction and expert-route annotator for SemPathBench prototypes."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import random
import sys
import uuid
import webbrowser
import warnings
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .argument import load_arguments, save_arguments
except ImportError:
    from argument import load_arguments, save_arguments

from scripts.annotation.path_shape_schema import (
    PATH_SHAPE_ANNOTATION_FIELD,
    normalize_path_shape_annotation,
    path_shape_reference_trajectory,
)

from scripts.evaluation.evaluation_metrics import (
    clearance_distance_field_from_obstacles as cached_clearance_distance_field_from_obstacles,
)
from scripts.evaluation.evaluation_metrics import (
    clearance_obstacle_cells as cached_clearance_obstacle_cells,
)
from scripts.evaluation.evaluation_metrics import compute_clearance_cost
from scripts.evaluation.evaluation_metrics import compute_relative_preference_quality
from scripts.evaluation.evaluation_metrics import build_instruction_metric_cache
from scripts.evaluation.metric_cache import MetricCacheWarning, build_map_metric_cache
from scripts.evaluation.hyparameter import (
    CLEARANCE_DISTANCE_FIELD_KEY,
    CLEARANCE_EXCLUDED_DISTANCE_FIELDS_KEY,
)
from scripts.methods.util.map_objects import object_cells as shared_object_cells


GRID_SIZE = 128
DEFAULT_MAP_ID = "simple_demo_map"
DEFAULT_MAP_KEY = "simple_demo/simple_demo_map"
ROUTE_COLOR = "#D84C3F"
START_COLOR = "#2D6CDF"
ASTAR_COLOR = "#1E8F62"

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

OCCUPANCY_VALUE_TO_NAME = {
    tile["value"]: name for name, tile in OCCUPANCY_TILES.items()
}
VALID_OCCUPANCY_VALUES = set(OCCUPANCY_VALUE_TO_NAME)
VALID_ROOM_CATEGORIES = set(ROOM_CATEGORIES)
VALID_OBJECT_CATEGORIES = set(OBJECT_CATEGORIES)
OBJECT_CATEGORY_ALIASES = {
    "shelving_unit": "shelf",
}

MAP_ROOT = REPO_ROOT / "resources" / "maps"
MAP_DIR = MAP_ROOT / "simple_demo"
OBJECT_COLOR_REGISTRY_PATH = MAP_ROOT / "object_color_registry.json"
INSTRUCTION_ROOT = REPO_ROOT / "resources" / "instructions"
INSTRUCTION_DIR = INSTRUCTION_ROOT / "simple_demo"
TEMPLATE_STATUSES = {"unlabeled", "labeled", "abandoned"}
TEMPLATE_OBJECT_COUNTS = {2, 3, 4, 5, 10}
DIFFICULTY_LEVELS = {"easy", "hard", "extreme"}
GLOBAL_DIFFICULTY_SPLITS = ("train", "valunseen")
GLOBAL_DIFFICULTY_SPLIT_LABELS = {
    "train": "Training",
    "valunseen": "Val-Unseen",
}
DISPLAYED_CONSTRAINT_USAGE_TYPES = (
    "must_avoid",
    "near_preference",
    "far_preference",
    "relative_preference",
    "path_shape_preference",
)
PERCENTAGE_CONSTRAINT_USAGE_TYPES = (
    "near_preference",
    "far_preference",
    "relative_preference",
    "path_shape_preference",
)
SOFT_CONSTRAINT_TYPES = (
    "near_preference",
    "far_preference",
    "relative_preference",
    "move_smoothness",
    "clearance",
    "path_shape_preference",
)
CONSTRAINT_USAGE_LABELS = {
    "near_preference": "Near preference",
    "far_preference": "Far preference",
    "relative_preference": "Relative preference",
    "move_smoothness": "Move smoothness",
    "clearance": "Clearance",
    "path_shape_preference": "Path-shape preference",
    "must_avoid": "Must avoid",
}
GLOBAL_CONSTRAINT_SCOPE = {"type": "global", "from_order": None, "to_order": None}
GLOBAL_CLEARANCE_CONSTRAINT_ID = "__global_clearance__"
TEMPLATE_MIN_DISTANCE = 40.0
TEMPLATE_HARD_CONSTRAINT_PADDING_METERS = 1
TEMPLATE_MAX_PER_OBJECT_COUNT = 200
TEMPLATE_MAX_OBJECTS = 80
TEMPLATE_MAX_TOTAL = TEMPLATE_MAX_PER_OBJECT_COUNT * len(TEMPLATE_OBJECT_COUNTS)
TEMPLATE_GRID_METRIC = "object_centroid_euclidean_grid"
TRAVERSABLE_OBJECT_CATEGORIES = {"doorframe", "doorway"}


def default_global_clearance_constraint() -> dict[str, object]:
    return {
        "constraint_id": GLOBAL_CLEARANCE_CONSTRAINT_ID,
        "preference_type": "clearance",
        "shape": "freeform",
        "center": [0, 0],
        "radius": 1.0,
        "width": 1.0,
        "height": 1.0,
        "object_ids": [],
        "reference_regions": [],
        "reference_region": None,
        "scope": dict(GLOBAL_CONSTRAINT_SCOPE),
        "label": "Clearance",
        "cells": [],
        "D_ref": None,
        "C_smooth_ref": None,
        "C_clear_ref": None,
        "reference_trajectory": [],
        "reference_waypoint_count": 0,
        "initialized": True,
    }


def has_clearance_constraint(soft_constraints: object) -> bool:
    return isinstance(soft_constraints, list) and any(
        isinstance(constraint, dict)
        and constraint.get("preference_type") == "clearance"
        for constraint in soft_constraints
    )


def ensure_global_clearance_constraint(
    soft_constraints: list[dict[str, object]],
) -> list[dict[str, object]]:
    if has_clearance_constraint(soft_constraints):
        return soft_constraints
    return [*soft_constraints, default_global_clearance_constraint()]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_map_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    cleaned = cleaned.strip("_").lower()
    return cleaned or DEFAULT_MAP_ID


def normalize_map_key(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_MAP_KEY
    parts = [
        sanitize_map_id(part)
        for part in value.replace("\\", "/").split("/")
        if part.strip()
    ]
    return "/".join(parts) or DEFAULT_MAP_KEY


def canonicalize_object_category(category: str) -> str:
    return OBJECT_CATEGORY_ALIASES.get(category, category)


def normalize_difficulty_level(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    return normalized if normalized in DIFFICULTY_LEVELS else ""


def map_split_sort_rank(map_key: object) -> int:
    leaf = normalize_map_key(str(map_key)).rsplit("/", 1)[-1]
    if leaf.endswith("_valunseen") or leaf.startswith("valunseen_"):
        return 0
    if leaf.endswith("_train") or leaf.startswith("train_"):
        return 1
    return 2


def _map_name_from_key(map_key: str) -> str:
    leaf = normalize_map_key(map_key).rsplit("/", 1)[-1]
    return " ".join(part.capitalize() for part in leaf.split("_")) or leaf


def _map_key_for_json_path(path: Path) -> str:
    try:
        parts = path.parent.relative_to(MAP_ROOT).parts
    except ValueError:
        return path.stem
    # ProcTHOR maps are physically grouped by split:
    # ``procthor/<train|valunseen>/<map_id>/<map_id>.json``.  The stable map
    # id intentionally remains ``procthor/<map_id>`` so instructions and
    # method outputs do not encode storage layout.
    if len(parts) >= 3 and parts[-2] in {"train", "valunseen"}:
        return Path(*parts[:-2], parts[-1]).as_posix()
    return Path(*parts).as_posix()


def list_map_records() -> list[dict[str, object]]:
    migrate_legacy_map_layout()
    MAP_ROOT.mkdir(parents=True, exist_ok=True)
    by_key: dict[str, dict[str, object]] = {}
    for json_path in sorted(MAP_ROOT.rglob("*.json")):
        if any(part.endswith("_metric_cache") for part in json_path.relative_to(MAP_ROOT).parts):
            continue
        if json_path == OBJECT_COLOR_REGISTRY_PATH:
            continue
        if json_path.name == "template_instruction.json":
            continue
        if json_path.name.endswith("_metadata.json"):
            continue

        map_key = normalize_map_key(_map_key_for_json_path(json_path))
        name = _map_name_from_key(map_key)
        record = {
            "map_id": map_key,
            "map_key": map_key,
            "name": name,
            "path": json_path,
            "path_display": json_path.relative_to(REPO_ROOT).as_posix(),
            "source": map_key.split("/", 1)[0],
        }
        existing = by_key.get(map_key)
        if existing is None or json_path.stem == json_path.parent.name:
            by_key[map_key] = record

    return sorted(
        by_key.values(),
        key=lambda item: (
            map_split_sort_rank(item["map_key"]),
            str(item["source"]),
            str(item["path_display"]),
        ),
    )


def resolve_map_record(map_id: str | None = None) -> dict[str, object]:
    records = list_map_records()
    if not records:
        raise RuntimeError(
            f"No layered map JSON files were found under {MAP_ROOT}."
        )

    def has_nonempty_template(record: dict[str, object]) -> bool:
        template_path = Path(str(record["path"])).parent / "template_instruction.json"
        if not template_path.exists():
            return False
        try:
            payload = json.loads(template_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        templates = payload.get("templates") if isinstance(payload, dict) else None
        return isinstance(templates, list) and len(templates) > 0

    requested = normalize_map_key(map_id)
    by_key = {str(record["map_key"]): record for record in records}
    if not isinstance(map_id, str) or not map_id.strip():
        default_record = by_key.get(DEFAULT_MAP_KEY)
        if default_record is not None and has_nonempty_template(default_record):
            return default_record
        templated_record = next(
            (record for record in records if has_nonempty_template(record)),
            None,
        )
        if templated_record is not None:
            return templated_record
        return default_record if default_record is not None else records[0]

    if requested in by_key:
        return by_key[requested]

    short_id = sanitize_map_id(map_id or DEFAULT_MAP_ID)
    if short_id == DEFAULT_MAP_ID and DEFAULT_MAP_KEY in by_key:
        return by_key[DEFAULT_MAP_KEY]

    matching = [
        record
        for record in records
        if str(record["map_key"]).rsplit("/", 1)[-1] == short_id
        or Path(str(record["path"])).stem == short_id
    ]
    if matching:
        return matching[0]

    return by_key[DEFAULT_MAP_KEY] if DEFAULT_MAP_KEY in by_key else records[0]


def map_path(map_id: str) -> Path:
    return Path(resolve_map_record(map_id)["path"])


def map_directory(map_id: str) -> Path:
    return map_path(map_id).parent


def template_instruction_path(map_id: str) -> Path:
    return map_directory(map_id) / "template_instruction.json"


def instruction_files_directory(map_id: str) -> Path:
    return INSTRUCTION_ROOT / normalize_map_key(map_id) / "instruction_files"


def instruction_directory(map_id: str) -> Path:
    return INSTRUCTION_ROOT / normalize_map_key(map_id)


def instruction_file_path(map_id: str, instruction_id: int) -> Path:
    return instruction_files_directory(map_id) / f"instruction_{instruction_id:06d}.json"


def saved_instruction_paths(map_id: str) -> list[Path]:
    """Return saved instruction files for a map, including legacy root files."""
    map_key = normalize_map_key(map_id)
    canonical_dir = instruction_files_directory(map_key)
    legacy_dir = instruction_directory(map_key)
    candidates = [
        *sorted(canonical_dir.glob("instruction_*.json")),
        *sorted(legacy_dir.glob("instruction_*.json")),
    ]
    paths: list[Path] = []
    seen_instruction_ids: set[str] = set()
    for path in candidates:
        instruction_id = path.stem.rsplit("_", 1)[-1]
        if instruction_id in seen_instruction_ids:
            continue
        seen_instruction_ids.add(instruction_id)
        paths.append(path)
    return paths


def iter_saved_instruction_payloads() -> Iterable[dict[str, object]]:
    """Yield saved instruction JSON payloads across every map."""
    for payload, _map_key in iter_saved_instruction_records():
        yield payload


def iter_saved_instruction_records() -> Iterable[tuple[dict[str, object], str]]:
    """Yield saved instruction payloads and their map keys across every map."""
    if not INSTRUCTION_ROOT.exists():
        return

    seen_keys: set[tuple[str, ...]] = set()
    paths = sorted(
        INSTRUCTION_ROOT.rglob("instruction_*.json"),
        key=lambda path: (
            0 if path.parent.name == "instruction_files" else 1,
            path.relative_to(INSTRUCTION_ROOT).as_posix(),
        ),
    )
    for path in paths:
        relative_parts = path.relative_to(INSTRUCTION_ROOT).parts
        semantic_key = (
            (*relative_parts[:-2], relative_parts[-1])
            if len(relative_parts) >= 2 and relative_parts[-2] == "instruction_files"
            else relative_parts
        )
        if semantic_key in seen_keys:
            continue
        seen_keys.add(semantic_key)
        map_key = normalize_map_key("/".join(semantic_key[:-1]))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        yield payload, map_key


def split_from_map_key(map_key: str) -> str:
    leaf = normalize_map_key(map_key).rsplit("/", 1)[-1]
    if leaf.endswith("_valunseen") or leaf.startswith("valunseen_"):
        return "valunseen"
    if leaf.endswith("_train") or leaf.startswith("train_"):
        return "train"
    return ""


def blank_difficulty_counts() -> dict[str, int]:
    return {level: 0 for level in sorted(DIFFICULTY_LEVELS)}


def blank_constraint_usage_counts() -> dict[str, int]:
    return {key: 0 for key in [*SOFT_CONSTRAINT_TYPES, "must_avoid"]}


def constraint_usage_items(counts: dict[str, int]) -> tuple[int, list[dict[str, object]]]:
    total = sum(counts[key] for key in DISPLAYED_CONSTRAINT_USAGE_TYPES)
    percentage_total = sum(counts[key] for key in PERCENTAGE_CONSTRAINT_USAGE_TYPES)
    return total, [
        {
            "key": key,
            "label": CONSTRAINT_USAGE_LABELS[key],
            "count": counts[key],
            "percentage": (
                None
                if key not in PERCENTAGE_CONSTRAINT_USAGE_TYPES
                else 0.0
                if percentage_total == 0
                else counts[key] * 100.0 / percentage_total
            ),
        }
        for key in DISPLAYED_CONSTRAINT_USAGE_TYPES
    ]


def add_constraint_usage_counts(
    counts: dict[str, int],
    payload: dict[str, object],
) -> None:
    soft_constraints = payload.get("soft_constraints", [])
    saw_clearance = False
    if isinstance(soft_constraints, list):
        for constraint in soft_constraints:
            if not isinstance(constraint, dict):
                continue
            preference_type = constraint.get("preference_type")
            if isinstance(preference_type, str) and preference_type in counts:
                counts[preference_type] += 1
                if preference_type == "clearance":
                    saw_clearance = True
    if not saw_clearance:
        counts["clearance"] += 1

    hard_constraints = payload.get("hard_constraints", [])
    if isinstance(hard_constraints, list):
        for constraint in hard_constraints:
            if (
                isinstance(constraint, dict)
                and constraint.get("kind") == "must_avoid"
            ):
                counts["must_avoid"] += 1


def global_difficulty_totals() -> dict[str, object]:
    """Count saved instruction difficulty labels across every map."""
    by_split = {
        split: blank_difficulty_counts()
        for split in GLOBAL_DIFFICULTY_SPLITS
    }
    total = blank_difficulty_counts()
    for payload, map_key in iter_saved_instruction_records():
        difficulty = normalize_difficulty_level(payload.get("difficulty_level"))
        split = split_from_map_key(map_key)
        if difficulty and split in by_split:
            total[difficulty] += 1
            by_split[split][difficulty] += 1
    return {
        "total": total,
        "by_split": by_split,
        "split_labels": GLOBAL_DIFFICULTY_SPLIT_LABELS,
    }


def map_difficulty_totals(map_id: str) -> dict[str, int]:
    """Count saved instruction difficulty labels for one map."""

    counts = blank_difficulty_counts()
    for path in saved_instruction_paths(map_id):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        difficulty = normalize_difficulty_level(payload.get("difficulty_level"))
        if difficulty:
            counts[difficulty] += 1
    return counts


def global_constraint_usage_summary() -> dict[str, object]:
    """Count soft-constraint and must-avoid usage across saved instructions."""
    counts = blank_constraint_usage_counts()
    by_split_counts = {
        split: blank_constraint_usage_counts()
        for split in GLOBAL_DIFFICULTY_SPLITS
    }

    for payload, map_key in iter_saved_instruction_records():
        split = split_from_map_key(map_key)
        if split not in by_split_counts:
            continue
        add_constraint_usage_counts(counts, payload)
        add_constraint_usage_counts(by_split_counts[split], payload)

    total, items = constraint_usage_items(counts)
    return {
        "total": total,
        "items": items,
        "by_split": {
            split: {
                "total": split_total,
                "items": split_items,
            }
            for split, (split_total, split_items) in (
                (split, constraint_usage_items(split_counts))
                for split, split_counts in by_split_counts.items()
            )
        },
    }


def migrate_legacy_map_layout() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    legacy_template_dir = MAP_DIR / "template_instruction"
    for legacy_json in list(MAP_DIR.glob("*.json")):
        map_id = sanitize_map_id(legacy_json.stem)
        target_dir = MAP_DIR / map_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_json = target_dir / f"{map_id}.json"
        if not target_json.exists():
            legacy_json.replace(target_json)
        for suffix in ("png", "ppm"):
            legacy_asset = MAP_DIR / f"{map_id}.{suffix}"
            target_asset = target_dir / f"{map_id}.{suffix}"
            if legacy_asset.exists() and not target_asset.exists():
                legacy_asset.replace(target_asset)
        legacy_template = legacy_template_dir / f"{map_id}.json"
        target_template = target_dir / "template_instruction.json"
        if legacy_template.exists() and not target_template.exists():
            legacy_template.replace(target_template)
    if legacy_template_dir.exists() and not any(legacy_template_dir.iterdir()):
        legacy_template_dir.rmdir()


def make_blank_grid(fill_value: int = 0, grid_size: int = GRID_SIZE) -> list[list[int]]:
    return [[fill_value for _ in range(grid_size)] for _ in range(grid_size)]


def validate_grid(
    grid: object,
    valid_values: set[int],
    label: str,
    grid_size: int = GRID_SIZE,
) -> list[list[int]]:
    if not isinstance(grid, list) or len(grid) != grid_size:
        raise ValueError(f"{label} grid must have {grid_size} rows.")

    validated: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != grid_size:
            raise ValueError(f"Each {label} row must have {grid_size} columns.")
        validated_row: list[int] = []
        for value in row:
            if not isinstance(value, int) or value not in valid_values:
                raise ValueError(f"Unexpected {label} cell value: {value!r}")
            validated_row.append(value)
        validated.append(validated_row)
    return validated


def validate_metadata(metadata: object, fallback_map_id: str) -> dict[str, object]:
    safe_map_id = sanitize_map_id(fallback_map_id)
    default = {
        "map_id": safe_map_id,
        "name": " ".join(part.capitalize() for part in safe_map_id.split("_")) or safe_map_id,
        "description": "",
    }
    if not isinstance(metadata, dict):
        return default

    map_id = metadata.get("map_id", default["map_id"])
    name = metadata.get("name", default["name"])
    description = metadata.get("description", default["description"])
    source = metadata.get("source")
    scene_size = metadata.get("scene_size")
    original_grid_shape = metadata.get("original_grid_shape")
    padded_grid_size = metadata.get("padded_grid_size")
    map_info = metadata.get("map_info")

    if not isinstance(map_id, str) or not map_id.strip():
        map_id = default["map_id"]
    if not isinstance(name, str) or not name.strip():
        name = default["name"]
    if not isinstance(description, str):
        description = default["description"]

    validated: dict[str, object] = {
        "map_id": sanitize_map_id(map_id),
        "name": name.strip(),
        "description": description,
    }
    if isinstance(source, str) and source.strip():
        validated["source"] = source.strip()
    if isinstance(scene_size, (int, float)):
        validated["scene_size"] = float(scene_size)
    if (
        isinstance(original_grid_shape, list)
        and len(original_grid_shape) == 2
        and all(isinstance(value, int) for value in original_grid_shape)
    ):
        validated["original_grid_shape"] = original_grid_shape
    if isinstance(padded_grid_size, int):
        validated["padded_grid_size"] = padded_grid_size
    if isinstance(map_info, dict):
        validated_map_info: dict[str, object] = {}
        for key in ("x_min", "x_max", "z_min", "z_max", "resolution"):
            value = map_info.get(key)
            if isinstance(value, (int, float)):
                validated_map_info[key] = float(value)
        for key in ("H", "W"):
            value = map_info.get(key)
            if isinstance(value, int):
                validated_map_info[key] = value
        if validated_map_info:
            validated["map_info"] = validated_map_info
    grid_coordinate_frame = metadata.get("grid_coordinate_frame")
    if isinstance(grid_coordinate_frame, dict):
        validated_frame: dict[str, object] = {}
        for key in ("index_order", "row_axis", "col_axis"):
            value = grid_coordinate_frame.get(key)
            if isinstance(value, str) and value.strip():
                validated_frame[key] = value.strip()
        for key in ("resolution", "x_min", "z_min"):
            value = grid_coordinate_frame.get(key)
            if isinstance(value, (int, float)):
                validated_frame[key] = float(value)
        if validated_frame:
            validated["grid_coordinate_frame"] = validated_frame
    return validated


def validate_instance_list(
    instances: object, valid_categories: set[str] | None, label: str
) -> list[dict[str, object]]:
    if instances is None:
        return []
    if not isinstance(instances, list):
        raise ValueError(f"{label} must be a list.")

    validated: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(instances, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Each {label[:-1]} must be an object.")
        raw_id = item.get("id")
        category = item.get("category")
        name = item.get("name")
        attributes = item.get("attributes", [])

        if not isinstance(raw_id, int) or raw_id <= 0:
            raise ValueError(f"{label[:-1].capitalize()} id must be a positive integer.")
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate {label[:-1]} id: {raw_id}")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Unknown {label[:-1]} category: {category!r}")
        category = canonicalize_object_category(category.strip())
        if valid_categories is not None and category not in valid_categories:
            raise ValueError(f"Unknown {label[:-1]} category: {category!r}")
        if not isinstance(name, str) or not name.strip():
            name = f"{category}_{index}"
        if not isinstance(attributes, list):
            raise ValueError(f"{label[:-1].capitalize()} attributes must be a list.")

        cleaned_attributes: list[str] = []
        for attribute in attributes:
            if not isinstance(attribute, str):
                raise ValueError(f"Each {label[:-1]} attribute must be a string.")
            cleaned = attribute.strip()
            if cleaned:
                cleaned_attributes.append(cleaned)

        cleaned_instance: dict[str, object] = {
            "id": raw_id,
            "category": category,
            "name": name.strip(),
            "attributes": cleaned_attributes,
        }
        for key in ("center_grid", "metadata_grid_cell"):
            point = item.get(key)
            if (
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(value, (int, float)) for value in point)
            ):
                cleaned_instance[key] = [float(point[0]), float(point[1])]
        for key in ("room_id", "room_overlap_grid_cells", "num_grid_cells"):
            value = item.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                cleaned_instance[key] = value
            elif isinstance(value, float) and not isinstance(value, bool):
                cleaned_instance[key] = value
        for key in ("assignment_method", "footprint_source"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                cleaned_instance[key] = value.strip()

        validated.append(cleaned_instance)
        seen_ids.add(raw_id)

    return validated


def validate_instance_grid(
    grid: object,
    instance_ids: set[int],
    label: str,
    grid_size: int = GRID_SIZE,
) -> list[list[int]]:
    if not isinstance(grid, list) or len(grid) != grid_size:
        raise ValueError(f"{label} grid must have {grid_size} rows.")

    validated: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != grid_size:
            raise ValueError(f"Each {label} row must have {grid_size} columns.")
        validated_row: list[int] = []
        for value in row:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Unexpected {label} cell value: {value!r}")
            if value not in instance_ids and value != 0:
                raise ValueError(f"Unknown {label} id in grid: {value}")
            validated_row.append(value)
        validated.append(validated_row)
    return validated


def object_category_label(category: str) -> str:
    return " ".join(part.capitalize() for part in category.split("_")) or category


def generated_object_color(category: str) -> str:
    digest = hashlib.sha1(category.encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.52, 0.86)
    return f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"


def normalize_color(value: object, category: str) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if (
            len(cleaned) == 7
            and cleaned.startswith("#")
            and all(char in "0123456789abcdefABCDEF" for char in cleaned[1:])
        ):
            return cleaned.upper()
    return generated_object_color(category)


def default_object_legend_entry(category: str) -> dict[str, str]:
    if category in OBJECT_CATEGORIES:
        entry = OBJECT_CATEGORIES[category]
        return {
            "label": str(entry["label"]),
            "color": normalize_color(entry["color"], category),
        }
    return {
        "label": object_category_label(category),
        "color": generated_object_color(category),
    }


def load_object_color_registry() -> dict[str, dict[str, str]]:
    registry: dict[str, dict[str, str]] = {}
    if OBJECT_COLOR_REGISTRY_PATH.exists():
        try:
            payload = json.loads(OBJECT_COLOR_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        raw_categories = payload.get("object_categories") if isinstance(payload, dict) else {}
        if isinstance(raw_categories, dict):
            for category, raw_entry in raw_categories.items():
                if not isinstance(category, str) or not category.strip():
                    continue
                category = canonicalize_object_category(category.strip())
                entry = raw_entry if isinstance(raw_entry, dict) else {}
                label = entry.get("label")
                if category in registry:
                    continue
                registry[category] = {
                    "label": label.strip() if isinstance(label, str) and label.strip() else object_category_label(category),
                    "color": normalize_color(entry.get("color"), category),
                }

    for category in OBJECT_CATEGORIES:
        registry.setdefault(category, default_object_legend_entry(category))
    return registry


def save_object_color_registry(registry: dict[str, dict[str, str]]) -> None:
    OBJECT_COLOR_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "description": (
            "Global object category color registry shared by SemPathBench maps. "
            "New object categories are appended automatically when maps are opened."
        ),
        "object_categories": {
            category: {
                "label": entry["label"],
                "color": entry["color"],
            }
            for category, entry in sorted(registry.items())
        },
        "updated_at": iso_now(),
    }
    OBJECT_COLOR_REGISTRY_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_object_color_registry(
    categories: set[str],
    preferred_legends: object = None,
) -> dict[str, dict[str, str]]:
    registry = load_object_color_registry()
    preferred = preferred_legends if isinstance(preferred_legends, dict) else {}
    changed = not OBJECT_COLOR_REGISTRY_PATH.exists()
    if OBJECT_COLOR_REGISTRY_PATH.exists():
        try:
            payload = json.loads(OBJECT_COLOR_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        raw_categories = payload.get("object_categories") if isinstance(payload, dict) else {}
        if isinstance(raw_categories, dict):
            changed = changed or any(
                canonicalize_object_category(category) != category
                for category in raw_categories
                if isinstance(category, str)
            )

    for category in sorted({canonicalize_object_category(category) for category in categories}):
        if not category:
            continue
        if category in registry:
            continue

        preferred_entry = preferred.get(category)
        if isinstance(preferred_entry, dict):
            label = preferred_entry.get("label")
            color = preferred_entry.get("color")
            registry[category] = {
                "label": label.strip() if isinstance(label, str) and label.strip() else object_category_label(category),
                "color": normalize_color(color, category),
            }
        else:
            registry[category] = default_object_legend_entry(category)
        changed = True

    if changed:
        save_object_color_registry(registry)
    return {
        category: registry[category]
        for category in sorted({canonicalize_object_category(category) for category in categories})
        if category in registry
    }


def validate_map_payload(payload: object, fallback_map_id: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Map payload must be a JSON object.")

    layers = payload.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Map payload must include a layers object.")

    raw_grid_size = payload.get("grid_size")
    if isinstance(raw_grid_size, int) and raw_grid_size > 0:
        grid_size = raw_grid_size
    else:
        occupancy_layer = layers.get("occupancy")
        if not isinstance(occupancy_layer, list) or not occupancy_layer:
            raise ValueError("Map grid_size is missing and occupancy cannot be inspected.")
        grid_size = len(occupancy_layer)

    layer_legends = payload.get("layer_legends")
    if not isinstance(layer_legends, dict):
        layer_legends = {}
    room_legends = layer_legends.get("room_categories")
    object_legends = layer_legends.get("object_categories")
    occupancy_legends = layer_legends.get("occupancy")
    valid_room_categories = (
        set(room_legends)
        if isinstance(room_legends, dict) and room_legends
        else None
    )
    normalized_legends = {
        "occupancy": occupancy_legends if isinstance(occupancy_legends, dict) else OCCUPANCY_TILES,
        "room_categories": room_legends if isinstance(room_legends, dict) else ROOM_CATEGORIES,
        "object_categories": object_legends if isinstance(object_legends, dict) else OBJECT_CATEGORIES,
    }

    metadata = validate_metadata(payload.get("metadata"), fallback_map_id)
    room_instances = validate_instance_list(
        payload.get("room_instances"), valid_room_categories, "room_instances"
    )
    object_instances = validate_instance_list(
        payload.get("object_instances"), None, "object_instances"
    )
    object_categories = {
        canonicalize_object_category(str(instance["category"]))
        for instance in object_instances
        if isinstance(instance.get("category"), str)
    }
    if isinstance(object_legends, dict):
        object_categories.update(
            canonicalize_object_category(category)
            for category in object_legends
            if isinstance(category, str) and category.strip()
        )
    normalized_legends["object_categories"] = ensure_object_color_registry(
        object_categories,
        object_legends,
    )
    room_instance_ids = {int(item["id"]) for item in room_instances}
    object_instance_ids = {int(item["id"]) for item in object_instances}

    occupancy = validate_grid(
        layers.get("occupancy"), VALID_OCCUPANCY_VALUES, "occupancy", grid_size
    )
    room = validate_instance_grid(
        layers.get("room"), room_instance_ids, "room", grid_size
    )
    object_grid = validate_instance_grid(
        layers.get("object_instance"),
        object_instance_ids,
        "object_instance",
        grid_size,
    )

    validated_payload: dict[str, object] = {
        "metadata": metadata,
        "grid_size": grid_size,
        "layer_legends": normalized_legends,
        "layers": {
            "occupancy": occupancy,
            "room": room,
            "object_instance": object_grid,
        },
        "room_instances": room_instances,
        "object_instances": object_instances,
    }
    object_footprints = payload.get("object_footprints")
    if isinstance(object_footprints, dict):
        validated_payload["object_footprints"] = object_footprints
    cell_object_ids = payload.get("cell_object_ids")
    if isinstance(cell_object_ids, dict):
        validated_payload["cell_object_ids"] = cell_object_ids
    return validated_payload


def list_map_ids() -> list[str]:
    return [str(record["map_key"]) for record in list_map_records()]


def list_map_summaries() -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for record in list_map_records():
        map_id = str(record["map_key"])
        summaries.append(
            {
                "map_id": map_id,
                "name": str(record["name"]),
                "path": str(record["path_display"]),
                "source": str(record["source"]),
            }
        )
    summaries.sort(
        key=lambda item: (
            map_split_sort_rank(item["map_id"]),
            item["source"],
            item["path"],
        )
    )
    return summaries


def load_map_state(map_id: str | None = None) -> dict[str, object]:
    record = resolve_map_record(map_id)
    map_key = str(record["map_key"])
    record_path = Path(str(record["path"]))
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    state = validate_map_payload(payload, map_key)
    enrich_object_instances_from_thinggraph(state, record_path)
    state["map_key"] = map_key
    state["map_path"] = str(record["path_display"])
    state["metadata"]["map_id"] = map_key  # type: ignore[index]
    return state


def valid_grid_point(value: object) -> list[float] | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return [float(value[0]), float(value[1])]
    return None


def object_cells_for_map(
    map_state: dict[str, object],
    object_id: int,
) -> list[list[int]]:
    """Return raw object cells, falling back across rich schema, legacy layer, and centers."""
    return [[row, col] for row, col in shared_object_cells(map_state, object_id)]


def enrich_object_instances_from_thinggraph(
    state: dict[str, object],
    map_json_path: Path,
) -> None:
    """Attach object center hints from ProcTHOR thinggraph exports when present."""

    thinggraph_path = map_json_path.with_name(f"{map_json_path.stem}_thinggraph.json")
    if not thinggraph_path.exists():
        return
    try:
        payload = json.loads(thinggraph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return

    object_records: dict[int, dict[str, object]] = {}
    for room in payload.get("rooms", []):
        if not isinstance(room, dict):
            continue
        for item in room.get("objects", []):
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                object_records[int(item["id"])] = item
    for item in payload.get("unassigned_objects", []):
        if isinstance(item, dict) and isinstance(item.get("id"), int):
            object_records[int(item["id"])] = item

    instances = state.get("object_instances")
    if not isinstance(instances, list):
        return
    for instance in instances:
        if not isinstance(instance, dict) or not isinstance(instance.get("id"), int):
            continue
        record = object_records.get(int(instance["id"]))
        if not record:
            continue
        for key in ("center_grid", "metadata_grid_cell"):
            point = valid_grid_point(record.get(key))
            if point is not None:
                instance[key] = point
        for key in (
            "assignment_method",
            "room_id",
            "room_overlap_grid_cells",
            "num_grid_cells",
        ):
            value = record.get(key)
            if isinstance(value, (int, float, str)):
                instance[key] = value


def map_state_for_client(map_state: dict[str, object]) -> dict[str, object]:
    """Return a JSON-serializable map state without runtime metric caches."""

    payload = dict(map_state)
    payload.pop(CLEARANCE_DISTANCE_FIELD_KEY, None)
    payload.pop(CLEARANCE_EXCLUDED_DISTANCE_FIELDS_KEY, None)
    return payload


def make_sample(
    map_id: str,
    index: int = 1,
    template_instruction_id: str | None = None,
) -> dict[str, object]:
    now = iso_now()
    return {
        "sample_id": f"sample_{uuid.uuid4().hex[:8]}",
        "map_id": normalize_map_key(map_id),
        "template_instruction_id": template_instruction_id,
        "name": f"Sample {index}",
        "instruction": "",
        "start_pose": {"row": None, "col": None},
        "expert_route": [],
        "feasible": True,
        "notes": "",
        "created_by": "",
        "created_at": now,
        "updated_at": now,
        "events": [],
        "constraints": [],
        "hard_constraints": [],
        "soft_constraints": [default_global_clearance_constraint()],
        "ordering": "",
        "difficulty_level": "",
        "valid_routes": [],
    }


def make_annotation_bundle(map_id: str) -> dict[str, object]:
    sample = make_sample(map_id)
    return {
        "version": 1,
        "map_id": normalize_map_key(map_id),
        "active_sample_id": sample["sample_id"],
        "samples": [sample],
        "updated_at": iso_now(),
    }


def validate_start_pose(raw_pose: object, grid_size: int = GRID_SIZE) -> dict[str, object]:
    default = {"row": None, "col": None}
    if raw_pose is None:
        return default
    if not isinstance(raw_pose, dict):
        raise ValueError("start_pose must be an object.")

    row = raw_pose.get("row")
    col = raw_pose.get("col")

    if row is not None and (not isinstance(row, int) or row < 0 or row >= grid_size):
        raise ValueError("start_pose.row must be an integer grid index or null.")
    if col is not None and (not isinstance(col, int) or col < 0 or col >= grid_size):
        raise ValueError("start_pose.col must be an integer grid index or null.")
    if (row is None) != (col is None):
        raise ValueError("start_pose.row and start_pose.col must both be set or both be null.")
    return {"row": row, "col": col}


def validate_route(
    route: object,
    grid_size: int = GRID_SIZE,
    *,
    label: str = "expert_route",
) -> list[list[int]]:
    if route is None:
        return []
    if not isinstance(route, list):
        raise ValueError(f"{label} must be a list.")

    validated: list[list[int]] = []
    for point in route:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not isinstance(point[0], int)
            or not isinstance(point[1], int)
        ):
            raise ValueError(f"Each {label} point must be [row, col].")
        row, col = point
        if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
            raise ValueError(f"Route point out of bounds: {point!r}")
        validated.append([row, col])
    return validated


def validate_string_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    cleaned_values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Each {label[:-1]} must be a string.")
        cleaned = item.strip()
        if cleaned:
            cleaned_values.append(cleaned)
    return cleaned_values


def validate_region_cells(
    value: object,
    label: str,
    grid_size: int = GRID_SIZE,
) -> list[list[int]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    cells: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in value:
        if (
            not isinstance(cell, list)
            or len(cell) != 2
            or not isinstance(cell[0], int)
            or not isinstance(cell[1], int)
        ):
            raise ValueError(f"Each {label} entry must be [row, col].")
        row, col = cell
        if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
            raise ValueError(f"{label} contains an out-of-bounds cell.")
        key = (row, col)
        if key not in seen:
            cells.append([row, col])
            seen.add(key)
    return cells


def validate_constraint_scope(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {"type": "global", "from_order": None, "to_order": None}
    scope_type = value.get("type", "global")
    if scope_type != "between_hard_constraints":
        return {"type": "global", "from_order": None, "to_order": None}
    from_order = value.get("from_order")
    to_order = value.get("to_order")
    if (
        not isinstance(from_order, int)
        or not isinstance(to_order, int)
        or from_order <= 0
        or to_order <= from_order
    ):
        return {"type": "global", "from_order": None, "to_order": None}
    return {
        "type": "between_hard_constraints",
        "from_order": from_order,
        "to_order": to_order,
    }


def validate_constraint_label(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def validate_hard_constraints(
    value: object,
    grid_size: int = GRID_SIZE,
) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("hard_constraints must be a list.")

    validated: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each hard constraint must be an object.")
        constraint_id = item.get("constraint_id")
        kind = item.get("kind")
        shape = item.get("shape")
        center = item.get("center")
        radius = item.get("radius", 6)
        width = item.get("width", 12)
        height = item.get("height", 12)
        order = item.get("order")
        source = item.get("source", "custom")
        object_id = item.get("object_id")
        cells = item.get("cells", [])
        scope = validate_constraint_scope(item.get("scope"))
        label = validate_constraint_label(
            item.get("label"),
            "Must Pass" if kind == "must_pass" else "Must Avoid",
        )

        if not isinstance(constraint_id, str) or not constraint_id.strip():
            constraint_id = f"constraint_{uuid.uuid4().hex[:8]}"
        if constraint_id in seen_ids:
            raise ValueError(f"Duplicate hard constraint id: {constraint_id}")
        if kind not in {"must_pass", "must_avoid"}:
            raise ValueError("hard constraint kind must be must_pass or must_avoid.")
        if shape not in {"circle", "rectangle", "freeform"}:
            raise ValueError(
                "hard constraint shape must be circle, rectangle, or freeform."
            )
        if (
            not isinstance(center, list)
            or len(center) != 2
            or not all(isinstance(number, (int, float)) for number in center)
        ):
            raise ValueError("hard constraint center must be [row, col].")
        center_row = float(center[0])
        center_col = float(center[1])
        if not (0 <= center_row < grid_size and 0 <= center_col < grid_size):
            raise ValueError("hard constraint center must be inside the map.")
        if not isinstance(radius, (int, float)) or float(radius) <= 0:
            raise ValueError("hard constraint radius must be positive.")
        if not isinstance(width, (int, float)) or float(width) <= 0:
            raise ValueError("hard constraint width must be positive.")
        if not isinstance(height, (int, float)) or float(height) <= 0:
            raise ValueError("hard constraint height must be positive.")
        if kind == "must_pass":
            if not isinstance(order, int) or order <= 0:
                order = index
        else:
            order = None
        if source not in {"template_object", "custom"}:
            source = "custom"
        if object_id is not None and (
            not isinstance(object_id, int) or object_id <= 0
        ):
            raise ValueError("hard constraint object_id must be positive or null.")
        validated_cells = validate_region_cells(
            cells,
            "hard constraint cells",
            grid_size,
        )
        if shape == "freeform" and not validated_cells:
            raise ValueError("freeform hard constraints must contain at least one cell.")

        validated.append(
            {
                "constraint_id": constraint_id.strip(),
                "kind": kind,
                "shape": shape,
                "center": [center_row, center_col],
                "radius": float(radius),
                "width": float(width),
                "height": float(height),
                "order": order,
                "source": source,
                "object_id": object_id,
                "scope": (
                    {"type": "global", "from_order": None, "to_order": None}
                    if kind == "must_pass"
                    else scope
                ),
                "label": label,
                "cells": validated_cells,
            }
        )
        seen_ids.add(constraint_id)
    return validated


def validate_soft_constraints(
    value: object,
    grid_size: int = GRID_SIZE,
) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("soft_constraints must be a list.")

    validated: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each soft constraint must be an object.")
        constraint_id = item.get("constraint_id")
        preference_type = item.get("preference_type")
        shape = item.get("shape")
        center = item.get("center")
        radius = item.get("radius", 6)
        width = item.get("width", 12)
        height = item.get("height", 12)
        object_ids = item.get("object_ids", [])
        reference_regions = item.get("reference_regions", [])
        reference_region = item.get("reference_region")
        legacy_reference_point = item.get("reference_point")
        d_ref = item.get("D_ref")
        q_relative_ref = item.get("Q_relative_ref")
        c_smooth_ref = item.get("C_smooth_ref")
        c_clear_ref = item.get("C_clear_ref")
        reference_trajectory = item.get("reference_trajectory", [])
        reference_waypoint_count = item.get("reference_waypoint_count", 0)
        reference_valid_path_length = item.get("reference_valid_path_length", 0.0)
        reference_metric_status = item.get("reference_metric_status")
        relative_reference_status = item.get("relative_reference_status")
        path_shape_annotation = item.get(PATH_SHAPE_ANNOTATION_FIELD)
        cells = item.get("cells", [])
        scope = validate_constraint_scope(item.get("scope"))
        label = validate_constraint_label(item.get("label"), "Soft Constraint")

        if not isinstance(constraint_id, str) or not constraint_id.strip():
            constraint_id = f"soft_constraint_{uuid.uuid4().hex[:8]}"
        if constraint_id in seen_ids:
            raise ValueError(f"Duplicate soft constraint id: {constraint_id}")
        if preference_type not in {
            "clearance",
            "far_preference",
            "move_smoothness",
            "near_preference",
            "path_shape_preference",
            "relative_preference",
        }:
            raise ValueError("Unknown soft constraint preference_type.")
        if preference_type in {"clearance", "move_smoothness"}:
            shape = "freeform" if shape is None else shape
            center = [0, 0] if center is None else center
            scope = dict(GLOBAL_CONSTRAINT_SCOPE)
        if (
            path_shape_annotation is not None
            and preference_type != "path_shape_preference"
        ):
            raise ValueError(
                "path_shape_annotation is only valid for path_shape_preference."
            )
        if reference_metric_status not in {
            None,
            "computed",
            "invalid_human_reference",
        }:
            raise ValueError(
                "reference_metric_status must be computed, "
                "invalid_human_reference, or null."
            )
        if shape not in {"circle", "rectangle", "freeform"}:
            raise ValueError(
                "soft constraint shape must be circle, rectangle, or freeform."
            )
        if (
            not isinstance(center, list)
            or len(center) != 2
            or not all(isinstance(number, (int, float)) for number in center)
        ):
            raise ValueError("soft constraint center must be [row, col].")
        center_row = float(center[0])
        center_col = float(center[1])
        if not (0 <= center_row < grid_size and 0 <= center_col < grid_size):
            raise ValueError("soft constraint center must be inside the map.")
        for number, dimension_label in (
            (radius, "radius"),
            (width, "width"),
            (height, "height"),
        ):
            if not isinstance(number, (int, float)) or float(number) <= 0:
                raise ValueError(f"soft constraint {dimension_label} must be positive.")
        if not isinstance(object_ids, list) or not all(
            isinstance(object_id, int) and object_id > 0 for object_id in object_ids
        ):
            raise ValueError("soft constraint object_ids must contain positive integers.")
        if preference_type == "relative_preference":
            if len(object_ids) != 2 or object_ids[0] == object_ids[1]:
                raise ValueError(
                    "relative_preference must reference two different objects."
                )
            if reference_regions is None:
                reference_regions = []
            if not isinstance(reference_regions, list):
                raise ValueError("relative reference_regions must be a list.")
            validated_reference_regions = []
            for index, relative_region in enumerate(reference_regions):
                if not isinstance(relative_region, dict):
                    raise ValueError(
                        "Each relative reference region must be an object."
                    )
                relative_object_id = relative_region.get("object_id")
                if (
                    not isinstance(relative_object_id, int)
                    or relative_object_id <= 0
                ):
                    raise ValueError(
                        "Relative reference object_id must be positive."
                    )
                validated_reference_regions.append(
                    {
                        "object_id": relative_object_id,
                        "cells": validate_region_cells(
                            relative_region.get("cells"),
                            f"relative reference region {index + 1} cells",
                            grid_size,
                        ),
                    }
                )
            reference_regions = validated_reference_regions
            reference_region = None
            d_ref = None
            if q_relative_ref is not None and (
                not isinstance(q_relative_ref, (int, float))
                or isinstance(q_relative_ref, bool)
            ):
                raise ValueError("relative_preference Q_relative_ref must be numeric or null.")
            if (
                not isinstance(reference_valid_path_length, (int, float))
                or isinstance(reference_valid_path_length, bool)
                or float(reference_valid_path_length) < 0
            ):
                raise ValueError(
                    "reference_valid_path_length must be non-negative."
                )
            if relative_reference_status not in {
                None,
                "computed",
                "invalid_human_reference",
            }:
                raise ValueError(
                    "relative_reference_status must be computed, "
                    "invalid_human_reference, or null."
                )
            c_smooth_ref = None
            c_clear_ref = None
            reference_waypoint_count = 0
        elif preference_type in {
            "clearance",
            "move_smoothness",
        }:
            object_ids = []
            reference_regions = []
            reference_region = None
            d_ref = None
            q_relative_ref = None
            reference_valid_path_length = 0.0
            relative_reference_status = None
            if preference_type == "move_smoothness":
                c_clear_ref = None
                if c_smooth_ref is not None and (
                    not isinstance(c_smooth_ref, (int, float))
                    or isinstance(c_smooth_ref, bool)
                    or float(c_smooth_ref) < 0
                ):
                    raise ValueError("move_smoothness C_smooth_ref must be non-negative or null.")
                reference_trajectory = []
            elif preference_type == "clearance":
                c_smooth_ref = None
                if c_clear_ref is not None and (
                    not isinstance(c_clear_ref, (int, float))
                    or isinstance(c_clear_ref, bool)
                    or float(c_clear_ref) < 0
                ):
                    raise ValueError("clearance C_clear_ref must be non-negative or null.")
                reference_trajectory = []
            else:
                c_smooth_ref = None
                c_clear_ref = None
                reference_trajectory = validate_route(
                    reference_trajectory,
                    grid_size,
                    label="reference_trajectory",
                )
            if (
                not isinstance(reference_waypoint_count, int)
                or isinstance(reference_waypoint_count, bool)
                or reference_waypoint_count < 0
            ):
                raise ValueError(
                    "reference_waypoint_count must be a non-negative integer."
                )
        elif preference_type == "path_shape_preference":
            object_ids = []
            reference_regions = []
            reference_region = None
            d_ref = None
            q_relative_ref = None
            reference_valid_path_length = 0.0
            relative_reference_status = None
            c_smooth_ref = None
            c_clear_ref = None
            if path_shape_annotation is not None:
                path_shape_annotation = normalize_path_shape_annotation(
                    path_shape_annotation,
                    grid_size=grid_size,
                )
                reference_trajectory = validate_route(
                    path_shape_annotation["shape_reference_trajectory"],
                    grid_size,
                    label=(
                        f"{PATH_SHAPE_ANNOTATION_FIELD}."
                        "shape_reference_trajectory"
                    ),
                )
                reference_waypoint_count = len(reference_trajectory)
            else:
                reference_trajectory = validate_route(
                    reference_trajectory,
                    grid_size,
                    label="reference_trajectory",
                )
            if (
                not isinstance(reference_waypoint_count, int)
                or isinstance(reference_waypoint_count, bool)
                or reference_waypoint_count < 0
            ):
                raise ValueError(
                    "reference_waypoint_count must be a non-negative integer."
                )
        else:
            object_ids = []
            reference_regions = []
            q_relative_ref = None
            reference_valid_path_length = 0.0
            relative_reference_status = None
            c_smooth_ref = None
            c_clear_ref = None
            reference_trajectory = []
            if reference_region is None:
                legacy_row = int(round(center_row))
                legacy_col = int(round(center_col))
                if (
                    isinstance(legacy_reference_point, list)
                    and len(legacy_reference_point) == 2
                    and all(
                        isinstance(number, (int, float))
                        for number in legacy_reference_point
                    )
                ):
                    legacy_row = int(round(float(legacy_reference_point[0])))
                    legacy_col = int(round(float(legacy_reference_point[1])))
                legacy_row = min(max(legacy_row, 0), grid_size - 1)
                legacy_col = min(max(legacy_col, 0), grid_size - 1)
                reference_region = {
                    "mode": "brush",
                    "object_id": None,
                    "cells": [[legacy_row, legacy_col]],
                }
            if not isinstance(reference_region, dict):
                raise ValueError("near/far reference_region must be an object.")
            reference_mode = reference_region.get("mode")
            reference_object_id = reference_region.get("object_id")
            if reference_mode not in {"object", "brush"}:
                raise ValueError(
                    "near/far reference_region mode must be object or brush."
                )
            if reference_mode == "object":
                if (
                    not isinstance(reference_object_id, int)
                    or reference_object_id <= 0
                ):
                    raise ValueError(
                        "object reference_region must include a positive object_id."
                    )
            else:
                reference_object_id = None
            reference_region = {
                "mode": reference_mode,
                "object_id": reference_object_id,
                "cells": validate_region_cells(
                    reference_region.get("cells"),
                    "soft reference region cells",
                    grid_size,
                ),
            }
            if d_ref is not None and (
                not isinstance(d_ref, (int, float))
                or isinstance(d_ref, bool)
                or float(d_ref) < 0
            ):
                raise ValueError("near/far D_ref must be non-negative or null.")
            if (
                not isinstance(reference_waypoint_count, int)
                or isinstance(reference_waypoint_count, bool)
                or reference_waypoint_count < 0
            ):
                raise ValueError(
                    "reference_waypoint_count must be a non-negative integer."
                )
        validated_cells = validate_region_cells(
            cells,
            "soft constraint cells",
            grid_size,
        )
        if shape == "freeform" and not validated_cells:
            if preference_type not in {
                "clearance",
                "move_smoothness",
            }:
                raise ValueError(
                    "freeform soft constraints must contain at least one cell "
                    f"(soft #{index}, id={constraint_id}, type={preference_type}, "
                    f"label={label})."
                )

        validated_constraint = {
            "constraint_id": constraint_id.strip(),
            "preference_type": preference_type,
            "shape": shape,
            "center": [center_row, center_col],
            "radius": float(radius),
            "width": float(width),
            "height": float(height),
            "object_ids": object_ids,
            "reference_regions": reference_regions,
            "reference_region": reference_region,
            "D_ref": float(d_ref) if d_ref is not None else None,
            "Q_relative_ref": (
                float(q_relative_ref) if q_relative_ref is not None else None
            ),
            "C_smooth_ref": (
                float(c_smooth_ref) if c_smooth_ref is not None else None
            ),
            "C_clear_ref": (
                float(c_clear_ref) if c_clear_ref is not None else None
            ),
            "reference_trajectory": reference_trajectory,
            "reference_waypoint_count": reference_waypoint_count,
            "reference_valid_path_length": float(reference_valid_path_length),
            "reference_metric_status": reference_metric_status,
            "relative_reference_status": relative_reference_status,
            "scope": (
                dict(GLOBAL_CONSTRAINT_SCOPE)
                if preference_type in {"clearance", "move_smoothness"}
                else scope
            ),
            "label": label,
            "cells": validated_cells,
        }
        if path_shape_annotation is not None:
            validated_constraint[PATH_SHAPE_ANNOTATION_FIELD] = path_shape_annotation
        validated.append(validated_constraint)
        seen_ids.add(constraint_id)
    return validated


def validate_sample(
    sample: object,
    map_id: str,
    index: int,
    grid_size: int = GRID_SIZE,
) -> dict[str, object]:
    if not isinstance(sample, dict):
        raise ValueError("Each sample must be an object.")

    now = iso_now()
    sample_id = sample.get("sample_id")
    name = sample.get("name")
    instruction = sample.get("instruction", "")
    feasible = sample.get("feasible", True)
    notes = sample.get("notes", "")
    created_by = sample.get("created_by", "")
    created_at = sample.get("created_at", now)
    updated_at = sample.get("updated_at", now)
    ordering = sample.get("ordering", "")
    difficulty_level = sample.get("difficulty_level", "")
    template_instruction_id = sample.get("template_instruction_id")

    if not isinstance(sample_id, str) or not sample_id.strip():
        sample_id = f"sample_{uuid.uuid4().hex[:8]}"
    if not isinstance(name, str) or not name.strip():
        name = f"Sample {index}"
    if not isinstance(instruction, str):
        instruction = ""
    if not isinstance(feasible, bool):
        raise ValueError("sample.feasible must be a boolean.")
    if not isinstance(notes, str):
        notes = ""
    if not isinstance(created_by, str):
        created_by = ""
    if not isinstance(created_at, str):
        created_at = now
    if not isinstance(updated_at, str):
        updated_at = now
    if not isinstance(ordering, str):
        ordering = ""
    difficulty_level = normalize_difficulty_level(difficulty_level)
    if template_instruction_id is not None and not isinstance(
        template_instruction_id, str
    ):
        raise ValueError("sample.template_instruction_id must be a string or null.")

    return {
        "sample_id": sample_id.strip(),
        "map_id": normalize_map_key(map_id),
        "template_instruction_id": (
            template_instruction_id.strip() if template_instruction_id else None
        ),
        "name": name.strip(),
        "instruction": instruction,
        "start_pose": validate_start_pose(sample.get("start_pose"), grid_size),
        "expert_route": validate_route(sample.get("expert_route"), grid_size),
        "feasible": feasible,
        "notes": notes,
        "created_by": created_by,
        "created_at": created_at,
        "updated_at": updated_at,
        "events": validate_string_list(sample.get("events"), "events"),
        "constraints": validate_string_list(sample.get("constraints"), "constraints"),
        "hard_constraints": validate_hard_constraints(
            sample.get("hard_constraints"),
            grid_size,
        ),
        "soft_constraints": ensure_global_clearance_constraint(
            validate_soft_constraints(
                sample.get("soft_constraints"),
                grid_size,
            )
        ),
        "ordering": ordering.strip(),
        "difficulty_level": difficulty_level,
        "valid_routes": validate_string_list(sample.get("valid_routes"), "valid_routes"),
    }


def validate_annotation_bundle(
    payload: object,
    map_id: str,
    grid_size: int = GRID_SIZE,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Annotation payload must be a JSON object.")

    map_key = normalize_map_key(map_id)
    raw_samples = payload.get("samples")
    if raw_samples is None:
        return make_annotation_bundle(map_key)
    if not isinstance(raw_samples, list):
        raise ValueError("samples must be a list.")

    validated_samples = [
        validate_sample(sample, map_key, index, grid_size)
        for index, sample in enumerate(raw_samples, start=1)
    ]
    if not validated_samples:
        validated_samples = [make_sample(map_key, 1)]

    active_sample_id = payload.get("active_sample_id")
    valid_ids = {sample["sample_id"] for sample in validated_samples}
    if not isinstance(active_sample_id, str) or active_sample_id not in valid_ids:
        active_sample_id = validated_samples[0]["sample_id"]

    updated_at = payload.get("updated_at", iso_now())
    if not isinstance(updated_at, str):
        updated_at = iso_now()

    return {
        "version": 1,
        "map_id": map_key,
        "active_sample_id": active_sample_id,
        "samples": validated_samples,
        "updated_at": updated_at,
    }


def validate_active_sample_from_payload(
    payload: object,
    map_id: str,
    grid_size: int = GRID_SIZE,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Annotation payload must be a JSON object.")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("samples must be a list.")
    active_sample_id = payload.get("active_sample_id")
    if not isinstance(active_sample_id, str) or not active_sample_id.strip():
        raise ValueError("active_sample_id must be a non-empty string.")
    for index, sample in enumerate(raw_samples, start=1):
        if isinstance(sample, dict) and sample.get("sample_id") == active_sample_id:
            return validate_sample(sample, normalize_map_key(map_id), index, grid_size)
    raise ValueError("The active annotation sample is missing.")


def response_annotation_state_from_payload(
    payload: dict[str, object],
    map_id: str,
    saved_sample: dict[str, object],
    instruction_id: int,
) -> dict[str, object]:
    saved_sample_id = f"instruction_{instruction_id:06d}"
    raw_samples = payload.get("samples")
    samples: list[dict[str, object]] = []
    if isinstance(raw_samples, list):
        samples = [dict(sample) for sample in raw_samples if isinstance(sample, dict)]
    previous_sample_id = saved_sample.get("sample_id")
    response_sample = dict(saved_sample)
    response_sample["sample_id"] = saved_sample_id
    replaced = False
    for index, sample in enumerate(samples):
        if sample.get("sample_id") == previous_sample_id:
            samples[index] = response_sample
            replaced = True
            break
    if not replaced:
        samples.append(response_sample)
    return {
        "version": 1,
        "map_id": normalize_map_key(map_id),
        "active_sample_id": saved_sample_id,
        "samples": samples,
        "updated_at": iso_now(),
    }


def load_annotation_bundle(
    map_id: str,
    grid_size: int = GRID_SIZE,
) -> dict[str, object]:
    map_key = normalize_map_key(map_id)
    directory = instruction_files_directory(map_key)
    directory.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    for index, path in enumerate(saved_instruction_paths(map_key), start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Instruction file must contain an object: {path}")
        instruction_id = payload.get("id")
        if not isinstance(instruction_id, int) or instruction_id <= 0:
            raise ValueError(f"Instruction id must be a positive integer: {path}")
        sample_payload = {
            "sample_id": f"instruction_{instruction_id:06d}",
            "map_id": map_key,
            "template_instruction_id": payload.get("template_instruction_id"),
            "name": payload.get("name", f"Instruction {instruction_id}"),
            "instruction": payload.get("instruction", ""),
            "start_pose": payload.get("start_pose"),
            "expert_route": payload.get("human_expert_trajectory", []),
            "feasible": payload.get("feasible", True),
            "notes": payload.get("notes", ""),
            "created_by": payload.get("created_by", ""),
            "created_at": payload.get("created_at", ""),
            "updated_at": payload.get("updated_at", ""),
            "events": payload.get("events", []),
            "constraints": payload.get("constraints", []),
            "hard_constraints": payload.get("hard_constraints", []),
            "soft_constraints": payload.get("soft_constraints", []),
            "ordering": payload.get("ordering", ""),
            "difficulty_level": payload.get("difficulty_level", ""),
            "valid_routes": payload.get("valid_routes", []),
        }
        samples.append(validate_sample(sample_payload, map_key, index, grid_size))

    if not samples:
        return make_annotation_bundle(map_key)
    return {
        "version": 1,
        "map_id": map_key,
        "active_sample_id": samples[0]["sample_id"],
        "samples": samples,
        "updated_at": iso_now(),
    }


def next_instruction_id(map_id: str) -> int:
    ids: list[int] = []
    for path in saved_instruction_paths(map_id):
        try:
            ids.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(ids, default=0) + 1


def point_inside_region(
    point: list[int], constraint: dict[str, object]
) -> bool:
    row, col = point
    shape = constraint["shape"]
    if shape == "freeform":
        return point in constraint["cells"]
    center_row, center_col = constraint["center"]  # type: ignore[misc]
    delta_row = row - float(center_row)
    delta_col = col - float(center_col)
    if shape == "circle":
        return math.hypot(delta_row, delta_col) <= float(constraint["radius"])
    return (
        abs(delta_row) <= float(constraint["height"]) / 2
        and abs(delta_col) <= float(constraint["width"]) / 2
    )


def mean_unique_waypoint_distance(
    trajectory: list[list[int]],
    constraint: dict[str, object],
    reference_points: list[list[int]],
) -> tuple[float | None, int]:
    if not reference_points:
        raise ValueError("Soft constraint reference region cannot be empty.")

    unique_region_points: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for point in trajectory:
        key = (point[0], point[1])
        if key in seen:
            continue
        seen.add(key)
        if point_inside_region(point, constraint):
            unique_region_points.append(point)

    if not unique_region_points:
        return None, 0

    distances = [
        min(
            math.hypot(
                point[0] - reference_point[0],
                point[1] - reference_point[1],
            )
            for reference_point in reference_points
        )
        for point in unique_region_points
    ]
    return math.fsum(distances) / len(distances), len(unique_region_points)


def mean_squared_turning_angle(trajectory: list[list[int]]) -> tuple[float, int]:
    """Return mean squared turn angle and number of valid turns in a route."""
    if len(trajectory) < 3:
        return 0.0, 0

    squared_angles: list[float] = []
    for index in range(1, len(trajectory) - 1):
        previous = trajectory[index - 1]
        current = trajectory[index]
        following = trajectory[index + 1]
        first_vector = [
            float(current[0]) - float(previous[0]),
            float(current[1]) - float(previous[1]),
        ]
        second_vector = [
            float(following[0]) - float(current[0]),
            float(following[1]) - float(current[1]),
        ]
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


def map_free_occupancy_value(map_state: dict[str, object]) -> int:
    legends = map_state.get("layer_legends")
    if isinstance(legends, dict):
        occupancy_legends = legends.get("occupancy")
        if isinstance(occupancy_legends, dict):
            free_tile = occupancy_legends.get("free")
            if isinstance(free_tile, dict) and isinstance(free_tile.get("value"), int):
                return int(free_tile["value"])
    return 0


def map_object_categories_by_id(map_state: dict[str, object]) -> dict[int, str]:
    object_instances = map_state.get("object_instances", [])
    if not isinstance(object_instances, list):
        return {}
    categories: dict[int, str] = {}
    for instance in object_instances:
        if not isinstance(instance, dict):
            continue
        object_id = instance.get("id")
        category = instance.get("category")
        if isinstance(object_id, int) and isinstance(category, str):
            categories[object_id] = category
    return categories


def clearance_obstacle_cells(
    map_state: dict[str, object],
) -> list[tuple[int, int, int]]:
    """Return obstacle source cells as (row, col, object_id)."""
    layers = map_state.get("layers")
    if not isinstance(layers, dict):
        return []
    occupancy = layers.get("occupancy")
    object_grid = layers.get("object_instance")
    if not isinstance(occupancy, list) or not isinstance(object_grid, list):
        return []

    free_value = map_free_occupancy_value(map_state)
    object_categories = map_object_categories_by_id(map_state)
    obstacles: list[tuple[int, int, int]] = []
    for row_index, row in enumerate(occupancy):
        if not isinstance(row, list):
            continue
        object_row = object_grid[row_index] if row_index < len(object_grid) else None
        for col_index, value in enumerate(row):
            try:
                occupancy_value = int(value)
            except (TypeError, ValueError):
                occupancy_value = free_value + 1
            object_id = 0
            if isinstance(object_row, list) and col_index < len(object_row):
                try:
                    object_id = int(object_row[col_index])
                except (TypeError, ValueError):
                    object_id = 0
            object_category = object_categories.get(object_id)
            object_is_obstacle = (
                object_id != 0 and object_category not in TRAVERSABLE_OBJECT_CATEGORIES
            )
            if occupancy_value != free_value or object_is_obstacle:
                obstacles.append((row_index, col_index, object_id))
    return obstacles


def near_preference_excluded_object_ids(
    point: list[int],
    soft_constraints: list[dict[str, object]],
) -> set[int]:
    excluded: set[int] = set()
    for constraint in soft_constraints:
        if constraint.get("preference_type") != "near_preference":
            continue
        reference_region = constraint.get("reference_region")
        if not isinstance(reference_region, dict):
            continue
        object_id = reference_region.get("object_id")
        if (
            reference_region.get("mode") == "object"
            and isinstance(object_id, int)
            and point_inside_region(point, constraint)
        ):
            excluded.add(object_id)
    return excluded


def clearance_cost_for_map(
    trajectory: list[list[int]],
    soft_constraints: list[dict[str, object]],
    map_state: dict[str, object],
    *,
    epsilon: float = 1e-6,
) -> tuple[float, int]:
    """Compute mean inverse-square obstacle clearance cost for a route."""
    if not trajectory:
        return 0.0, 0
    if CLEARANCE_DISTANCE_FIELD_KEY not in map_state:
        obstacles = cached_clearance_obstacle_cells(map_state)
        distance_field = cached_clearance_distance_field_from_obstacles(
            map_state,
            obstacles,
        )
        if distance_field is not None:
            map_state[CLEARANCE_DISTANCE_FIELD_KEY] = distance_field
    return compute_clearance_cost(
        trajectory,
        map_state,
        soft_constraints,
        epsilon=epsilon,
    )


def first_region_hit_index(
    trajectory: list[list[int]],
    constraint: dict[str, object],
    start_index: int = 0,
) -> int:
    for index in range(start_index, len(trajectory)):
        if point_inside_region(trajectory[index], constraint):
            return index
    return -1


def hard_constraint_hit_indexes(sample: dict[str, object]) -> dict[int, int]:
    trajectory = sample["expert_route"]  # type: ignore[assignment]
    ordered = sorted(
        [
            constraint
            for constraint in sample.get("hard_constraints", [])  # type: ignore[union-attr]
            if constraint["kind"] == "must_pass"
        ],
        key=lambda item: int(item["order"]),
    )
    hit_indexes: dict[int, int] = {}
    start_index = 0
    for constraint in ordered:
        index = first_region_hit_index(trajectory, constraint, start_index)
        if index >= 0:
            hit_indexes[int(constraint["order"])] = index
            start_index = index + 1
    return hit_indexes


def scoped_trajectory(
    sample: dict[str, object],
    constraint: dict[str, object],
    hit_indexes: dict[int, int],
) -> list[list[int]]:
    trajectory = sample["expert_route"]  # type: ignore[assignment]
    scope = constraint.get("scope")
    if not isinstance(scope, dict) or scope.get("type") != "between_hard_constraints":
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
    trajectory: list[list[int]],
    constraint: dict[str, object],
) -> list[list[int]]:
    """Return waypoints that fall inside a constraint's own region."""
    return [
        [point[0], point[1]]
        for point in trajectory
        if point_inside_region(point, constraint)
    ]


def annotate_soft_reference_distances(
    sample: dict[str, object],
    map_state: dict[str, object] | None = None,
) -> None:
    hit_indexes = hard_constraint_hit_indexes(sample)
    for constraint in sample["soft_constraints"]:  # type: ignore[union-attr]
        if constraint["preference_type"] == "clearance":
            if map_state is None:
                raise ValueError("clearance requires map_state to compute C_clear_ref.")
            c_clear_ref, waypoint_count = clearance_cost_for_map(
                scoped_trajectory(sample, constraint, hit_indexes),
                sample["soft_constraints"],  # type: ignore[arg-type]
                map_state,
            )
            constraint["C_clear_ref"] = c_clear_ref
            constraint["C_smooth_ref"] = None
            constraint["D_ref"] = None
            constraint["reference_waypoint_count"] = waypoint_count
            constraint["reference_metric_status"] = "computed"
            continue
        if constraint["preference_type"] == "move_smoothness":
            c_smooth_ref, waypoint_count = mean_squared_turning_angle(
                scoped_trajectory(sample, constraint, hit_indexes)
            )
            constraint["C_smooth_ref"] = c_smooth_ref
            constraint["C_clear_ref"] = None
            constraint["D_ref"] = None
            constraint["reference_trajectory"] = []
            constraint["reference_waypoint_count"] = waypoint_count
            constraint["reference_metric_status"] = "computed"
            continue
        if constraint["preference_type"] == "path_shape_preference":
            if constraint.get(PATH_SHAPE_ANNOTATION_FIELD) is not None:
                raw_reference = path_shape_reference_trajectory(constraint)
                if not isinstance(raw_reference, list):
                    raise ValueError(
                        "Embedded path-shape annotation has no reference trajectory."
                    )
                reference_trajectory = [
                    [int(point[0]), int(point[1])]
                    for point in raw_reference
                ]
            else:
                reference_trajectory = trajectory_inside_constraint_region(
                    scoped_trajectory(sample, constraint, hit_indexes),
                    constraint,
                )
            constraint["reference_trajectory"] = reference_trajectory
            constraint["C_smooth_ref"] = None
            constraint["C_clear_ref"] = None
            constraint["D_ref"] = None
            constraint["reference_waypoint_count"] = len(reference_trajectory)
            constraint["reference_metric_status"] = "computed"
            continue
        if constraint["preference_type"] == "relative_preference":
            q_ref, q_details = compute_relative_preference_quality(
                scoped_trajectory(sample, constraint, hit_indexes),
                constraint,
            )
            constraint["Q_relative_ref"] = q_ref
            constraint["D_ref"] = None
            constraint["reference_trajectory"] = []
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
            constraint["reference_metric_status"] = constraint[
                "relative_reference_status"
            ]
            continue
        if constraint["preference_type"] not in {
            "near_preference",
            "far_preference",
        }:
            continue
        reference_region = constraint["reference_region"]
        if not isinstance(reference_region, dict):
            raise ValueError("Near/far preference requires a reference region.")
        d_ref, waypoint_count = mean_unique_waypoint_distance(
            scoped_trajectory(sample, constraint, hit_indexes),
            constraint,
            reference_region["cells"],  # type: ignore[arg-type]
        )
        constraint["D_ref"] = d_ref
        constraint["reference_trajectory"] = []
        constraint["reference_waypoint_count"] = waypoint_count
        constraint["reference_metric_status"] = (
            "computed" if d_ref is not None else "invalid_human_reference"
        )


def route_collision_violations_for_map(
    sample: dict[str, object],
    map_state: dict[str, object],
) -> list[list[int]]:
    """Return route cells that are not traversable in the map occupancy layer."""
    layers = map_state.get("layers")
    if not isinstance(layers, dict):
        return []
    occupancy = layers.get("occupancy")
    if not isinstance(occupancy, list):
        return []

    legends = map_state.get("layer_legends")
    occupancy_legends = legends.get("occupancy") if isinstance(legends, dict) else None
    free_value = 0
    if isinstance(occupancy_legends, dict):
        free_tile = occupancy_legends.get("free")
        if isinstance(free_tile, dict) and isinstance(free_tile.get("value"), int):
            free_value = int(free_tile["value"])

    violations: list[list[int]] = []
    for point in sample.get("expert_route", []):
        if not isinstance(point, list) or len(point) != 2:
            continue
        row, col = int(point[0]), int(point[1])
        if row < 0 or row >= len(occupancy):
            violations.append([row, col])
            continue
        occupancy_row = occupancy[row]
        if (
            not isinstance(occupancy_row, list)
            or col < 0
            or col >= len(occupancy_row)
        ):
            violations.append([row, col])
            continue
        try:
            occupancy_value = int(occupancy_row[col])
        except (TypeError, ValueError):
            violations.append([row, col])
            continue
        if occupancy_value != free_value:
            violations.append([row, col])
    return violations


def save_instruction_file(
    map_id: str,
    template: dict[str, object],
    sample: dict[str, object],
    *,
    map_state: dict[str, object] | None = None,
) -> dict[str, object]:
    existing_id = template.get("labeled_instruction_id")
    instruction_id = (
        existing_id
        if isinstance(existing_id, int) and existing_id > 0
        else next_instruction_id(map_id)
    )
    now = iso_now()
    map_key = normalize_map_key(map_id)
    payload = {
        "id": instruction_id,
        "map_id": map_key,
        "template_instruction_id": template["template_instruction_id"],
        "object_count": template["object_count"],
        "objects": template["objects"],
        "human_expert_trajectory": sample["expert_route"],
        "start_pose": sample["start_pose"],
        "feasible": sample["feasible"],
        "name": sample["name"],
        "notes": sample["notes"],
        "created_by": sample["created_by"],
        "hard_constraints": sample["hard_constraints"],
        "soft_constraints": sample["soft_constraints"],
        "ordering": sample["ordering"],
        "difficulty_level": sample["difficulty_level"],
        "created_at": sample["created_at"] or now,
        "updated_at": now,
        "instruction": sample["instruction"],
    }
    path = instruction_file_path(map_key, instruction_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        cache_map_state = map_state if map_state is not None else load_map_state(map_key)
        build_map_metric_cache(Path(str(cache_map_state["map_path"])), cache_map_state)
        build_instruction_metric_cache(path, payload, cache_map_state)
    except Exception as exc:  # Keep a completed annotation usable if cache generation fails.
        warnings.warn(
            f"Instruction was saved but metric cache generation failed for {path}: {exc}",
            MetricCacheWarning,
            stacklevel=2,
        )
    return payload


def template_object_center_distance(
    first: dict[str, object],
    second: dict[str, object],
) -> float:
    first_center = first["center"]
    second_center = second["center"]
    return math.hypot(
        float(first_center[0]) - float(second_center[0]),  # type: ignore[index]
        float(first_center[1]) - float(second_center[1]),  # type: ignore[index]
    )


def collect_template_object_centers(map_state: dict[str, object]) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for instance in map_state["object_instances"]:  # type: ignore[union-attr]
        object_id = int(instance["id"])
        cells = shared_object_cells(map_state, object_id)
        if not cells:
            continue
        center_row = sum(row for row, _col in cells) / len(cells)
        center_col = sum(col for _row, col in cells) / len(cells)
        objects.append(
            {
                "object_id": object_id,
                "name": instance["name"],
                "category": instance["category"],
                "center": [round(center_row, 3), round(center_col, 3)],
            }
        )
    return sorted(objects, key=lambda item: int(item["object_id"]))


def template_pair_distances(
    objects: list[dict[str, object]],
) -> dict[tuple[int, int], float]:
    distances: dict[tuple[int, int], float] = {}
    for first in objects:
        for second in objects:
            first_id = int(first["object_id"])
            second_id = int(second["object_id"])
            if first_id != second_id:
                distances[(first_id, second_id)] = template_object_center_distance(
                    first, second
                )
    return distances


def template_path_is_spaced(
    path: list[dict[str, object]],
    distances: dict[tuple[int, int], float],
) -> bool:
    for first_index, first in enumerate(path):
        first_id = int(first["object_id"])
        for second in path[first_index + 1 :]:
            if distances[(first_id, int(second["object_id"]))] <= TEMPLATE_MIN_DISTANCE:
                return False
    return True


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
    grid_size: int,
) -> list[list[dict[str, object]]]:
    rng = random.SystemRandom()
    templates: list[list[dict[str, object]]] = []
    seen_object_sets: set[tuple[int, ...]] = set()
    seen_signatures: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
    attempts = max(2500, TEMPLATE_MAX_PER_OBJECT_COUNT * target_count * 80)

    def candidate_is_valid(
        path: list[dict[str, object]], candidate: dict[str, object]
    ) -> bool:
        candidate_id = int(candidate["object_id"])
        return all(
            distances[(int(item["object_id"]), candidate_id)] > TEMPLATE_MIN_DISTANCE
            for item in path
        )

    for _attempt in range(attempts):
        if len(templates) >= TEMPLATE_MAX_PER_OBJECT_COUNT:
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
                jitter = rng.random() * 10.0
                scored.append((category_bonus + spread_bonus + jitter, candidate))
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
        signature = template_diversity_signature(path, grid_size)
        if signature in seen_signatures:
            continue
        seen_object_sets.add(object_set)
        seen_signatures.add(signature)
        templates.append(path)

    return templates


def generate_template_instruction_bundle(
    map_id: str, map_state: dict[str, object]
) -> dict[str, object]:
    map_key = normalize_map_key(map_id)
    grid_size = int(map_state["grid_size"])
    objects = collect_template_object_centers(map_state)[:TEMPLATE_MAX_OBJECTS]
    distances = template_pair_distances(objects)
    templates: list[dict[str, object]] = []
    created_at = iso_now()

    for target_count in sorted(TEMPLATE_OBJECT_COUNTS):
        sampled_paths = sample_diverse_template_paths(
            objects,
            distances,
            target_count,
            grid_size,
        )
        for count_index, path in enumerate(sampled_paths, start=1):
            segment_distances = [
                round(
                    distances[
                        (
                            int(path[index]["object_id"]),
                            int(path[index + 1]["object_id"]),
                        )
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
                    "map_id": map_key,
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
        "map_id": map_key,
        "distance_rule": {
            "metric": TEMPLATE_GRID_METRIC,
            "minimum_exclusive": TEMPLATE_MIN_DISTANCE,
        },
        "object_counts": sorted(TEMPLATE_OBJECT_COUNTS),
        "randomized_order": True,
        "template_limits": {
            "max_objects_considered": TEMPLATE_MAX_OBJECTS,
            "max_per_object_count": TEMPLATE_MAX_PER_OBJECT_COUNT,
            "max_total_templates": TEMPLATE_MAX_TOTAL,
            "spacing_scope": "all_object_pairs",
            "diversity_strategy": "random_spatial_category_sampling",
        },
        "templates": templates,
        "created_at": created_at,
        "updated_at": created_at,
    }


def validate_template_instruction_bundle(
    payload: object, map_id: str
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Template instruction payload must be a JSON object.")

    map_key = normalize_map_key(map_id)
    raw_templates = payload.get("templates")
    if not isinstance(raw_templates, list):
        raise ValueError("Template instruction payload must include a templates list.")

    templates: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw_template in raw_templates:
        if not isinstance(raw_template, dict):
            raise ValueError("Each template instruction must be an object.")
        template_id = raw_template.get("template_instruction_id")
        object_count = raw_template.get("object_count")
        objects = raw_template.get("objects")
        status = raw_template.get("status", "unlabeled")
        labeled_instruction_id = raw_template.get("labeled_instruction_id")

        if not isinstance(template_id, str) or not template_id.strip():
            raise ValueError("template_instruction_id must be a non-empty string.")
        if template_id in seen_ids:
            raise ValueError(f"Duplicate template_instruction_id: {template_id}")
        if object_count not in TEMPLATE_OBJECT_COUNTS:
            raise ValueError("template object_count must be one of 2, 3, 4, 5, or 10.")
        if not isinstance(objects, list) or len(objects) != object_count:
            raise ValueError("Template objects must match object_count.")
        if status not in TEMPLATE_STATUSES:
            raise ValueError(f"Unknown template status: {status!r}")
        if labeled_instruction_id is not None and (
            not isinstance(labeled_instruction_id, int)
            or labeled_instruction_id <= 0
        ):
            raise ValueError("labeled_instruction_id must be a positive integer or null.")

        cleaned_objects: list[dict[str, object]] = []
        for order, raw_object in enumerate(objects, start=1):
            if not isinstance(raw_object, dict):
                raise ValueError("Each template object must be an object.")
            object_id = raw_object.get("object_id")
            name = raw_object.get("name")
            category = raw_object.get("category")
            center = raw_object.get("center")
            if not isinstance(object_id, int) or object_id <= 0:
                raise ValueError("Template object_id must be a positive integer.")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Template object name must be a non-empty string.")
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Template object category must be a non-empty string.")
            if (
                not isinstance(center, list)
                or len(center) != 2
                or not all(isinstance(value, (int, float)) for value in center)
            ):
                raise ValueError("Template object center must be [row, col].")
            cleaned_objects.append(
                {
                    "order": order,
                    "object_id": object_id,
                    "name": name.strip(),
                    "category": category.strip(),
                    "center": [float(center[0]), float(center[1])],
                }
            )

        segment_distances = raw_template.get("segment_distances", [])
        if (
            not isinstance(segment_distances, list)
            or len(segment_distances) != object_count - 1
            or not all(isinstance(value, (int, float)) for value in segment_distances)
        ):
            raise ValueError("segment_distances must contain one value per connection.")
        cleaned_segment_distances = [
            round(
                template_object_center_distance(
                    cleaned_objects[index],
                    cleaned_objects[index + 1],
                ),
                3,
            )
            for index in range(object_count - 1)
        ]
        if any(distance <= TEMPLATE_MIN_DISTANCE for distance in cleaned_segment_distances):
            seen_ids.add(template_id)
            continue
        if not template_path_is_spaced(cleaned_objects, template_pair_distances(cleaned_objects)):
            seen_ids.add(template_id)
            continue

        templates.append(
            {
                "template_instruction_id": template_id.strip(),
                "map_id": map_key,
                "object_count": object_count,
                "objects": cleaned_objects,
                "segment_distances": cleaned_segment_distances,
                "status": status,
                "labeled_instruction_id": labeled_instruction_id,
                "created_at": raw_template.get("created_at", ""),
                "updated_at": raw_template.get("updated_at", ""),
            }
        )
        seen_ids.add(template_id)

    distance_rule: dict[str, object] = {
        **(
            payload.get("distance_rule", {})
            if isinstance(payload.get("distance_rule"), dict)
            else {}
        ),
        "metric": TEMPLATE_GRID_METRIC,
        "minimum_exclusive": TEMPLATE_MIN_DISTANCE,
    }

    return {
        "version": 1,
        "map_id": map_key,
        "distance_rule": distance_rule,
        "object_counts": sorted(TEMPLATE_OBJECT_COUNTS),
        "randomized_order": bool(payload.get("randomized_order")),
        "template_limits": (
            payload.get("template_limits")
            if isinstance(payload.get("template_limits"), dict)
            else {
                "max_objects_considered": TEMPLATE_MAX_OBJECTS,
                "max_per_object_count": TEMPLATE_MAX_PER_OBJECT_COUNT,
                "max_total_templates": TEMPLATE_MAX_TOTAL,
                "spacing_scope": "all_object_pairs",
                "diversity_strategy": "random_spatial_category_sampling",
            }
        ),
        "templates": templates,
        "created_at": payload.get("created_at", ""),
        "updated_at": payload.get("updated_at", ""),
    }


def template_payload_has_labeled_work(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    templates = payload.get("templates")
    if not isinstance(templates, list):
        return False
    return any(
        isinstance(template, dict)
        and (
            template.get("status") == "labeled"
            or isinstance(template.get("labeled_instruction_id"), int)
        )
        for template in templates
    )


def template_payload_needs_regeneration(payload: object) -> bool:
    if not isinstance(payload, dict):
        return True
    templates = payload.get("templates")
    if not isinstance(templates, list) or len(templates) == 0:
        return True
    object_counts = payload.get("object_counts")
    if not isinstance(object_counts, list) or set(object_counts) != TEMPLATE_OBJECT_COUNTS:
        return True
    template_limits = payload.get("template_limits")
    if not isinstance(template_limits, dict):
        return True
    if template_limits.get("max_per_object_count") != TEMPLATE_MAX_PER_OBJECT_COUNT:
        return True
    if template_limits.get("spacing_scope") != "all_object_pairs":
        return True
    per_count: dict[int, int] = {}
    for template in templates:
        if not isinstance(template, dict):
            return True
        object_count = template.get("object_count")
        if not isinstance(object_count, int):
            return True
        per_count[object_count] = per_count.get(object_count, 0) + 1
        if per_count[object_count] > TEMPLATE_MAX_PER_OBJECT_COUNT:
            return True
    return False


def load_template_instruction_bundle(map_id: str) -> dict[str, object]:
    path = template_instruction_path(map_id)
    if not path.exists():
        payload = generate_template_instruction_bundle(
            map_id,
            load_map_state(map_id),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return validate_template_instruction_bundle(payload, map_id)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if template_payload_needs_regeneration(payload) and not template_payload_has_labeled_work(payload):
        payload = generate_template_instruction_bundle(
            map_id,
            load_map_state(map_id),
        )
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return validate_template_instruction_bundle(payload, map_id)

    if isinstance(payload, dict) and not payload.get("randomized_order"):
        templates = payload.get("templates")
        if isinstance(templates, list):
            random.SystemRandom().shuffle(templates)
            payload["randomized_order"] = True
            payload["updated_at"] = iso_now()
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    return validate_template_instruction_bundle(payload, map_id)


def sync_labeled_instructions_with_templates(
    map_id: str,
    template_bundle: dict[str, object] | None = None,
) -> dict[str, object]:
    """Reconcile saved instruction files with template labeled status."""
    map_key = normalize_map_key(map_id)
    bundle = (
        load_template_instruction_bundle(map_key)
        if template_bundle is None
        else template_bundle
    )
    templates = bundle.get("templates", [])
    if not isinstance(templates, list):
        return bundle

    templates_by_id = {
        str(template["template_instruction_id"]): template
        for template in templates
        if isinstance(template, dict) and template.get("template_instruction_id")
    }
    changed = False
    canonical_dir = instruction_files_directory(map_key)
    canonical_dir.mkdir(parents=True, exist_ok=True)

    for path in saved_instruction_paths(map_key):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        instruction_id = payload.get("id")
        template_id = payload.get("template_instruction_id")
        if (
            not isinstance(instruction_id, int)
            or instruction_id <= 0
            or not isinstance(template_id, str)
            or not template_id.strip()
        ):
            continue

        canonical_path = instruction_file_path(map_key, instruction_id)
        if path != canonical_path and not canonical_path.exists():
            canonical_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        template = templates_by_id.get(template_id.strip())
        if template is None:
            continue
        if (
            template.get("status") != "labeled"
            or template.get("labeled_instruction_id") != instruction_id
        ):
            template["status"] = "labeled"
            template["labeled_instruction_id"] = instruction_id
            template["updated_at"] = payload.get("updated_at") or iso_now()
            changed = True

    if changed:
        bundle["updated_at"] = iso_now()
        path = template_instruction_path(map_key)
        path.write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return validate_template_instruction_bundle(bundle, map_key)
    return bundle


def update_template_instruction_status(
    map_id: str,
    template_instruction_id: str,
    status: str,
    labeled_instruction_id: int | None = None,
) -> dict[str, object]:
    if status not in TEMPLATE_STATUSES:
        raise ValueError(f"Unknown template status: {status!r}")

    bundle = load_template_instruction_bundle(map_id)
    matched = False
    for template in bundle["templates"]:  # type: ignore[union-attr]
        if template["template_instruction_id"] != template_instruction_id:
            continue
        template["status"] = status
        template["labeled_instruction_id"] = (
            labeled_instruction_id if status == "labeled" else None
        )
        template["updated_at"] = iso_now()
        matched = True
        break
    if not matched:
        raise ValueError(
            f"Unknown template_instruction_id: {template_instruction_id!r}"
        )

    bundle["updated_at"] = iso_now()
    path = template_instruction_path(map_id)
    path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return bundle


def next_unlabeled_template_id(
    bundle: dict[str, object], current_template_id: str
) -> str | None:
    templates = bundle["templates"]  # type: ignore[assignment]
    current = next(
        (
            template
            for template in templates
            if template["template_instruction_id"] == current_template_id
        ),
        None,
    )
    if current is None:
        return None
    object_count = current["object_count"]
    for template in templates:
        if (
            template["object_count"] == object_count
            and template["status"] == "unlabeled"
        ):
            return str(template["template_instruction_id"])
    return None


def build_client_state(
    map_id: str,
    *,
    map_state: dict[str, object] | None = None,
    annotation_state: dict[str, object] | None = None,
    template_instruction_state: dict[str, object] | None = None,
) -> dict[str, object]:
    map_state = load_map_state(map_id) if map_state is None else map_state
    resolved_map_id = str(map_state["map_key"])
    grid_size = int(map_state["grid_size"])
    if template_instruction_state is None:
        template_instruction_state = sync_labeled_instructions_with_templates(
            resolved_map_id,
            load_template_instruction_bundle(resolved_map_id),
        )
    if annotation_state is None:
        annotation_state = load_annotation_bundle(resolved_map_id, grid_size)
    legends = map_state.get(
        "layer_legends",
        {
            "occupancy": OCCUPANCY_TILES,
            "room_categories": ROOM_CATEGORIES,
            "object_categories": OBJECT_CATEGORIES,
        },
    )
    return {
        "grid_size": grid_size,
        "route_color": ROUTE_COLOR,
        "start_color": START_COLOR,
        "astar_color": ASTAR_COLOR,
        "legends": legends,
        "map_summaries": list_map_summaries(),
        "global_difficulty_totals": global_difficulty_totals(),
        "map_difficulty_totals": map_difficulty_totals(resolved_map_id),
        "global_constraint_usage": global_constraint_usage_summary(),
        "current_map_id": resolved_map_id,
        "paths": {
            "map_json": str(map_path(resolved_map_id).relative_to(REPO_ROOT)),
            "template_instruction_json": str(
                template_instruction_path(resolved_map_id).relative_to(REPO_ROOT)
            ),
            "instruction_directory": str(
                instruction_files_directory(resolved_map_id).relative_to(REPO_ROOT)
            ),
        },
        "map_state": map_state_for_client(map_state),
        "annotation_state": annotation_state,
        "template_instruction_state": template_instruction_state,
        "arguments": load_arguments(),
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SemPathBench Instruction + Expert Route Annotator</title>
  <style>
    :root {
      --bg: #f4efe7;
      --panel: rgba(252, 249, 243, 0.96);
      --line: #d4cab8;
      --line-strong: #b9aa8d;
      --ink: #2d2923;
      --muted: #6b604f;
      --accent: #8b6631;
      --accent-soft: #efe3cf;
      --danger: #b54840;
      --ok: #2f7a58;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #faf7f1 0%, #ece2d2 100%);
      color: var(--ink);
    }
    button, input, select, textarea { font: inherit; }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr) 380px;
      gap: 16px;
      padding: 16px;
    }
    .panel, .canvas-wrap {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 12px 28px rgba(65, 52, 33, 0.08);
    }
    .panel-scroll {
      max-height: calc(100vh - 64px);
      overflow-y: auto;
      padding-right: 4px;
    }
    h1, h2, h3, p {
      margin: 0;
    }
    h1 {
      font-size: 24px;
      margin-bottom: 8px;
    }
    h2 {
      font-size: 20px;
      margin-bottom: 8px;
    }
    .hint, .small {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    .section {
      margin-top: 16px;
      display: grid;
      gap: 8px;
    }
    .section-title {
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #736652;
      font-weight: 700;
    }
    .stack {
      display: grid;
      gap: 8px;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .grow {
      flex: 1 1 auto;
      min-width: 0;
    }
    .pill-list, .sample-list, .check-list {
      display: grid;
      gap: 8px;
    }
    .item, .sample-item, .mode-btn, .chip-row {
      border: 1px solid #ddd3c1;
      border-radius: 12px;
      background: #fffaf3;
    }
    .sample-item, .item, .chip-row {
      padding: 10px 12px;
    }
    .sample-item {
      cursor: pointer;
    }
    .template-scroll {
      height: 112px;
      overflow-y: auto;
      overscroll-behavior: contain;
      scroll-snap-type: y mandatory;
      padding-right: 4px;
    }
    .template-scroll .sample-item {
      min-height: 104px;
      scroll-snap-align: start;
      scroll-snap-stop: always;
    }
    .sample-item.active, .mode-btn.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(139, 102, 49, 0.14);
    }
    .template-object {
      width: 100%;
      padding: 8px 10px;
      border: 1px solid #ddd3c1;
      border-radius: 9px;
      background: #fffaf3;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
    }
    .template-object + .template-object {
      margin-top: 6px;
    }
    .template-object:hover {
      border-color: #8f8068;
      background: #f7efe2;
    }
    .template-object.active {
      background: #191919;
      border-color: #191919;
      color: #fff;
    }
    .template-object-card + .template-object-card {
      margin-top: 8px;
    }
    .selected-object-scroll {
      max-height: 116px;
      overflow-y: auto;
      overscroll-behavior: contain;
      padding-right: 4px;
      scrollbar-gutter: stable;
    }
    .template-attribute-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }
    .attribute-chip-btn {
      padding: 6px 8px;
      border-radius: 999px;
      border: 1px solid #d5c7af;
      background: #f8f0e2;
      color: #5d503d;
      font-size: 12px;
      line-height: 1.25;
    }
    .attribute-chip-btn:hover {
      border-color: #8f8068;
      background: #efe3cf;
    }
    .constraint-list {
      display: grid;
      gap: 6px;
      max-height: 210px;
      overflow-y: auto;
    }
    .constraint-create-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
    }
    .constraint-drop-zone {
      min-height: 76px;
      padding: 6px;
      border: 1px dashed #c9bda8;
      border-radius: 12px;
      background: rgba(255, 250, 243, 0.65);
    }
    .constraint-drop-zone.drop-target {
      border-color: var(--accent);
      background: #f1e3cb;
    }
    .hidden-legacy-list {
      display: none !important;
    }
    .constraint-item {
      padding: 8px 10px;
      border: 1px solid #ddd3c1;
      border-radius: 9px;
      background: #fffaf3;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
    }
    .constraint-item.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(139, 102, 49, 0.14);
    }
    .constraint-item.must-pass {
      border-left: 5px solid #000000;
    }
    .constraint-item.must-avoid {
      border-left: 5px solid #d14f45;
    }
    .constraint-item.soft {
      border-left: 5px solid #3478b8;
    }
    .constraint-item.dragging {
      opacity: 0.45;
    }
    .constraint-item.drop-target {
      outline: 2px dashed var(--accent);
      outline-offset: 2px;
    }
    .tool-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
    }
    .tool-grid button {
      padding: 8px;
      font-size: 12px;
    }
    .inspector {
      display: none;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fffaf3;
      line-height: 1.5;
    }
    .inspector.visible {
      display: block;
    }
    .sample-title {
      font-weight: 700;
      color: #2d2923;
    }
    .sample-meta {
      margin-top: 4px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .status-badge {
      display: inline-flex;
      margin-top: 7px;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #6d5128;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .status-badge.labeled {
      background: #dfeee5;
      color: #286447;
    }
    .status-badge.abandoned {
      background: #eee7e0;
      color: #776a5e;
    }
    .mode-list {
      display: grid;
      gap: 8px;
    }
    .mode-btn {
      text-align: left;
      padding: 10px 12px;
      cursor: pointer;
      background: #fffaf3;
      border-color: #111111;
      color: #111111;
    }
    .mode-btn:hover {
      background: #f0e8dc;
    }
    .mode-btn.active {
      background: #111111;
      border-color: #111111;
      color: #ffffff;
    }
    .mode-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
    }
    .mode-shortcut {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 22px;
      height: 22px;
      border: 1px solid currentColor;
      border-radius: 999px;
      font-size: 12px;
      line-height: 1;
    }
    .mode-detail {
      margin-top: 4px;
      font-size: 12px;
      color: var(--muted);
    }
    .mode-btn.active .mode-detail {
      color: rgba(255, 255, 255, 0.76);
    }
    .canvas-wrap {
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-width: 0;
    }
    .toolbar, .toolbar-secondary {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .status {
      padding: 10px 12px;
      border-radius: 12px;
      background: #f2e9da;
      border: 1px solid #dfd1ba;
      font-size: 14px;
      line-height: 1.4;
      height: calc(2.8em + 22px);
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }
    .canvas-box {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: white;
      overflow: auto;
      max-height: calc(100vh - 190px);
    }
    canvas {
      display: block;
      image-rendering: pixelated;
      cursor: crosshair;
    }
    input[type="text"], input[type="number"], select, textarea {
      width: 100%;
      border: 1px solid #d6cab7;
      border-radius: 10px;
      padding: 10px 12px;
      background: #fffaf3;
      color: var(--ink);
    }
    textarea {
      min-height: 88px;
      resize: vertical;
      line-height: 1.45;
    }
    #instructionInput {
      min-height: 220px;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 12px;
      background: #2d2923;
      color: white;
      cursor: pointer;
    }
    button.alt {
      background: #ddcfb4;
      color: #2d2923;
    }
    button.warn {
      background: var(--danger);
    }
    .mono {
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      color: #5d5446;
      line-height: 1.5;
      word-break: break-word;
    }
    .check-ok, .check-warn {
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 13px;
      line-height: 1.45;
    }
    .check-ok {
      background: #e5f2ea;
      color: #255d44;
      border: 1px solid #b9dcc8;
    }
    .check-warn {
      background: #fbebe8;
      color: #7f342f;
      border: 1px solid #e5b9b3;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .stat-card {
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid #ddd3c1;
      background: #fffaf3;
    }
    .stat-label {
      font-size: 12px;
      color: var(--muted);
    }
    .stat-value {
      margin-top: 4px;
      font-size: 18px;
      font-weight: 700;
    }
    .difficulty-summary {
      margin-top: 10px;
    }
    .difficulty-summary-title {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .difficulty-summary-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
    }
    .difficulty-split-list {
      display: grid;
      gap: 6px;
    }
    .difficulty-split {
      border: 1px solid #ddd3c1;
      border-radius: 10px;
      background: #fffaf3;
      overflow: hidden;
    }
    .difficulty-split summary {
      cursor: pointer;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 8px;
      color: #111111;
      font-size: 12px;
      font-weight: 800;
      list-style-position: inside;
    }
    .difficulty-split-count {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }
    .difficulty-split .difficulty-summary-grid {
      padding: 0 8px 8px;
    }
    .difficulty-split-subtitle {
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
      padding: 2px 8px 6px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .difficulty-split .constraint-usage-list {
      padding: 0 8px 8px;
    }
    .difficulty-pill {
      border: 1px solid #ddd3c1;
      border-radius: 10px;
      background: #fffaf3;
      padding: 8px;
    }
    .difficulty-pill-label {
      color: #111111;
      font-size: 12px;
      font-weight: 700;
    }
    .difficulty-pill-count {
      color: #111111;
      font-size: 18px;
      font-weight: 800;
      margin-top: 3px;
    }
    .constraint-usage {
      margin-top: 10px;
    }
    .constraint-usage-list {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 6px;
    }
    .constraint-usage-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 4px;
      border: 1px solid #ddd3c1;
      border-radius: 10px;
      background: #fffaf3;
      padding: 7px;
    }
    .constraint-usage-label {
      color: #111111;
      font-size: 10px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .constraint-usage-count,
    .constraint-usage-percent {
      color: #111111;
      font-size: 10px;
      font-weight: 800;
      white-space: nowrap;
    }
    .constraint-usage-count {
      text-align: right;
    }
    .constraint-usage-percent {
      grid-column: 1 / -1;
      color: var(--muted);
      font-weight: 700;
    }
    @media (max-width: 1200px) {
      .shell {
        grid-template-columns: 1fr;
      }
      .canvas-box {
        max-height: none;
      }
      .panel-scroll {
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel">
      <div class="panel-scroll">
        <h1>Annotation Editor</h1>
        <p class="hint">Annotate instruction and expert route on top of an existing layered map.</p>

        <div class="section">
          <div class="section-title">All Maps Difficulty</div>
          <div id="globalDifficultySummary" class="item difficulty-summary"></div>
          <div id="globalConstraintUsage" class="item constraint-usage"></div>
        </div>

        <div class="section">
          <div class="section-title">Map Source</div>
          <div class="row">
            <select id="mapSelect" class="grow"></select>
            <button id="openMapBtn" type="button">Open Map</button>
          </div>
          <div id="mapInfo" class="item"></div>
          <div id="difficultySummary" class="item difficulty-summary"></div>
        </div>

        <div class="section">
          <div class="section-title">Annotation Actions</div>
          <div class="stack">
            <button id="saveBtn" type="button">Save Instruction</button>
            <button id="abandonBtn" type="button" class="warn">Abandon Template</button>
          </div>
        </div>

        <div class="section">
          <div class="section-title">Mode</div>
          <div id="modeList" class="mode-list"></div>
        </div>

        <div class="section">
          <div class="section-title">Route Checks</div>
          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-label">HCS</div>
              <div id="hcsScore" class="stat-value">1.000</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">SCS</div>
              <div id="scsScore" class="stat-value">1.000</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">PL</div>
              <div id="pathLengthScore" class="stat-value">0.000</div>
            </div>
          </div>
        </div>

        <div class="section">
          <div class="section-title">Files</div>
          <div id="paths" class="mono"></div>
        </div>
      </div>
    </aside>

    <main class="canvas-wrap">
      <div class="toolbar">
        <label class="row">
          <span>Zoom</span>
          <input id="zoomSlider" type="range" min="1" max="12" step="1">
          <span id="zoomValue"></span>
        </label>
        <label class="row">
          <input id="gridToggle" type="checkbox" checked>
          <span>Show grid</span>
        </label>
        <label class="row">
          <input id="roomToggle" type="checkbox" checked>
          <span>Show rooms</span>
        </label>
        <label class="row">
          <input id="objectToggle" type="checkbox" checked>
          <span>Show objects</span>
        </label>
        <label class="row">
          <input id="smallObjectNameToggle" type="checkbox">
          <span>Show small object names</span>
        </label>
        <label class="row">
          <input id="hardConstraintToggle" type="checkbox" checked>
          <span>Show hard constraints</span>
        </label>
        <label class="row">
          <input id="softConstraintToggle" type="checkbox" checked>
          <span>Show soft constraints</span>
        </label>
      </div>

      <div class="toolbar-secondary">
        <button id="focusAnnotationBtn" type="button" class="alt">Focus Annotation</button>
        <button id="clearRouteBtn" type="button" class="alt">Clear Route</button>
            <button id="undoLastAStarBtn" type="button" class="alt">Undo Last Route Segment</button>
            <button id="resetAStarBtn" type="button" class="alt hidden-legacy-list">Reset A*</button>
      </div>

      <div id="status" class="status">Loading…</div>

      <div id="canvasBox" class="canvas-box">
        <canvas id="gridCanvas" width="768" height="768"></canvas>
      </div>
      <div id="mapInspector" class="inspector"></div>
    </main>

    <aside class="panel">
      <div class="panel-scroll">
        <h2>Samples</h2>
        <p class="hint">Open generated object-sequence templates and label them one by one.</p>

        <div class="section">
          <div class="section-title">Annotator</div>
          <input id="createdByInput" type="text" placeholder="Your name or annotator ID">
          <p class="small">This name is reused for every instruction and remembered in this browser.</p>
        </div>

        <div class="section">
          <div class="section-title">Template Filters</div>
          <div class="row">
            <select id="objectCountFilter" class="grow">
              <option value="">All objects</option>
              <option value="2" selected>2 objects</option>
              <option value="3">3 objects</option>
              <option value="4">4 objects</option>
              <option value="5">5 objects</option>
              <option value="10">10 objects</option>
            </select>
            <select id="statusFilter" class="grow">
              <option value="unlabeled">Unlabeled</option>
              <option value="labeled">Labeled</option>
              <option value="abandoned">Abandoned</option>
            </select>
          </div>
          <select id="difficultyFilter">
            <option value="">All difficulties</option>
            <option value="easy">Easy</option>
            <option value="hard">Hard</option>
            <option value="extreme">Extreme hard</option>
          </select>
          <select id="labeledInstructionSelect" style="display: none;"></select>
        </div>

        <div class="section">
          <div class="section-title">Template Instructions</div>
          <div id="sampleList" class="sample-list template-scroll"></div>
        </div>

        <div class="section">
          <div class="section-title">Selected Objects</div>
          <div id="templateDetails" class="item selected-object-scroll"></div>
        </div>

        <div class="section">
          <div class="section-title">Basic Fields</div>
          <input id="sampleNameInput" type="text" placeholder="Sample name">
          <textarea id="instructionInput" placeholder="Write the natural-language instruction here..."></textarea>
          <label class="small">
            Difficulty
            <select id="difficultyLevelInput">
              <option value="">Select difficulty...</option>
              <option value="easy">Easy</option>
              <option value="hard">Hard</option>
              <option value="extreme">Extreme hard</option>
            </select>
          </label>
          <label class="row">
            <input id="feasibleInput" type="checkbox" checked>
            <span>Feasible sample</span>
          </label>
        </div>

        <div class="section">
          <div class="section-title">Route Start</div>
          <div id="startPoseInfo" class="item"></div>
          <p class="small">The first expert-route cell is saved as the start pose automatically.</p>
        </div>

        <div class="section">
          <div class="section-title">Create Constraints</div>
          <p class="small">Create a bar first, then choose Circle / Rectangle / Brush / Add Box below.</p>
          <div class="constraint-create-grid">
            <button id="createMustPassBtn" type="button">Must Pass</button>
            <button id="createMustAvoidBtn" type="button">Must Avoid</button>
            <button id="createSoftConstraintBtn" type="button">Soft</button>
          </div>
        </div>

        <div class="section">
          <div class="section-title">Sequence Constraints</div>
          <p class="small">Ordered bars. Black hard bars define the sequence; red/blue bars can be scoped to sequence segments.</p>
          <div id="sequenceConstraintList" class="constraint-list constraint-drop-zone"></div>
          <div id="mustPassList" class="constraint-list hidden-legacy-list"></div>
          <div id="mustAvoidList" class="constraint-list hidden-legacy-list"></div>
        </div>

        <div class="section">
          <div class="section-title">Global Constraints</div>
          <p class="small">Unordered bars. These constraints apply to the whole route.</p>
          <div id="globalConstraintList" class="constraint-list constraint-drop-zone"></div>
          <div id="softConstraintList" class="constraint-list hidden-legacy-list"></div>
        </div>

        <div id="hardEditorSection" class="section" style="display: none;">
          <div class="section-title">Selected Hard Region</div>
          <div id="constraintEditor" class="stack">
            <div id="hardCreationTools" class="tool-grid">
              <button id="addPassCircleBtn" type="button" class="alt">Circle</button>
              <button id="addPassRectangleBtn" type="button" class="alt">Rectangle</button>
              <button id="paintPassBrushBtn" type="button" class="alt">Add Brush</button>
              <button id="addPassBoxBtn" type="button" class="alt">Add Box</button>
              <button id="addAvoidCircleBtn" type="button" class="alt hidden-legacy-list">Legacy Avoid Circle</button>
              <button id="addAvoidRectangleBtn" type="button" class="alt hidden-legacy-list">Legacy Avoid Rect</button>
              <button id="paintAvoidBrushBtn" type="button" class="alt hidden-legacy-list">Legacy Avoid Brush</button>
              <button id="addAvoidBoxBtn" type="button" class="alt hidden-legacy-list">Legacy Avoid Box</button>
            </div>
            <div id="passCreationTools" class="tool-grid hidden-legacy-list"></div>
            <div id="avoidCreationTools" class="tool-grid hidden-legacy-list"></div>
            <select id="constraintShapeInput">
              <option value="circle">Circle</option>
              <option value="rectangle">Rectangle</option>
              <option value="freeform">Freeform</option>
            </select>
            <input id="constraintRadiusInput" type="number" min="1" step="1" placeholder="Circle radius">
            <div class="row">
              <input id="constraintWidthInput" class="grow" type="number" min="1" step="1" placeholder="Rectangle width">
              <input id="constraintHeightInput" class="grow" type="number" min="1" step="1" placeholder="Rectangle height">
            </div>
            <input id="constraintLabelInput" type="text" placeholder="Constraint label">
            <select id="constraintScopeInput"></select>
            <div id="constraintOrderInfo" class="small"></div>
            <div class="row">
              <button id="eraseHardFreeformBtn" type="button" class="alt grow">Erase Brush</button>
              <button id="eraseHardBoxBtn" type="button" class="alt grow">Erase Box</button>
            </div>
            <button id="deleteConstraintBtn" type="button" class="warn">Delete Selected Constraint</button>
          </div>
        </div>

        <div id="softEditorSection" class="section" style="display: none;">
          <div class="section-title">Selected Soft Region</div>
          <p class="small">Blue preferences can be global or active only between two hard constraints.</p>
          <div class="stack">
            <div id="softCreationTools" class="tool-grid">
              <button id="addSoftCircleBtn" type="button" class="alt">Circle</button>
              <button id="addSoftRectangleBtn" type="button" class="alt">Rectangle</button>
              <button id="paintSoftBrushBtn" type="button" class="alt">Add Brush</button>
              <button id="addSoftBoxBtn" type="button" class="alt">Add Box</button>
            </div>
            <select id="softPreferenceTypeInput">
              <option value="near_preference">Near preference</option>
              <option value="far_preference">Far preference</option>
              <option value="relative_preference">Relative preference</option>
              <option value="path_shape_preference">Path-shape preference</option>
            </select>
            <select id="softConstraintShapeInput">
              <option value="circle">Circle</option>
              <option value="rectangle">Rectangle</option>
              <option value="freeform">Freeform</option>
            </select>
            <input id="softConstraintRadiusInput" type="number" min="1" step="1" placeholder="Circle radius">
            <div class="row">
              <input id="softConstraintWidthInput" class="grow" type="number" min="1" step="1" placeholder="Rectangle width">
              <input id="softConstraintHeightInput" class="grow" type="number" min="1" step="1" placeholder="Rectangle height">
            </div>
            <input id="softConstraintLabelInput" type="text" placeholder="Soft constraint label">
            <select id="softConstraintScopeInput"></select>
            <div id="relativeObjectInputs" class="row">
              <label class="grow small">Object A (closer)<select id="relativeObjectAInput"></select></label>
              <button id="pickRelativeObjectABtn" type="button" class="alt">Pick A on Map</button>
              <label class="grow small">Object B (farther)<select id="relativeObjectBInput"></select></label>
              <button id="pickRelativeObjectBBtn" type="button" class="alt">Pick B on Map</button>
            </div>
            <div id="softReferenceRegionInputs" class="stack">
              <label id="softReferenceObjectLabel" class="small">Reference object
                <select id="softReferenceObjectInput"></select>
              </label>
              <button id="pickSoftReferenceBtn" type="button" class="alt">Pick Reference Object on Map</button>
              <div id="softReferenceRegionInfo" class="small">Reference region: not set.</div>
            </div>
            <div id="softSmoothnessInfo" class="small" style="display: none;">
              Trajectory preferences use the scoped trajectory only. No region or reference object is needed.
            </div>
            <div class="row">
              <button id="eraseSoftFreeformBtn" type="button" class="alt grow">Erase Brush</button>
              <button id="eraseSoftBoxBtn" type="button" class="alt grow">Erase Box</button>
            </div>
            <button id="deleteSoftConstraintBtn" type="button" class="warn">Delete Selected Soft Constraint</button>
          </div>
        </div>
      </div>
    </aside>
  </div>

  <script>
    const INITIAL_STATE = __INITIAL_STATE__;
    let GRID_SIZE = INITIAL_STATE.grid_size;
    let currentLegends = INITIAL_STATE.legends;
    const MODES = {
      route_draw: {
        label: "Route Draw",
        detail: "Drag to hand-draw; click to A* connect from the route tail."
      },
      route_erase: {
        label: "Route Erase",
        detail: "Drag or click to remove route cells."
      },
      route_adjust: {
        label: "Route Adjust",
        detail: "Drag a route waypoint to a new cell; A* reconnects both sides."
      },
      must_pass_point: {
        label: "Hard Region Circle",
        detail: "Click once to create an editable hard-region circle."
      },
      must_pass_rectangle: {
        label: "Hard Region Rectangle",
        detail: "Click two opposite corners to create a hard-region rectangle."
      },
      must_pass_brush: {
        label: "Hard Region Add Brush",
        detail: "Drag to add cells to the selected hard region."
      },
      must_pass_free_rectangle: {
        label: "Hard Region Add Box",
        detail: "Click two corners to add a box to the selected hard region."
      },
      must_avoid_point: {
        label: "Hard Region Circle",
        detail: "Click once to create an editable hard-region circle."
      },
      must_avoid_rectangle: {
        label: "Hard Region Rectangle",
        detail: "Click two opposite corners to create a hard-region rectangle."
      },
      must_avoid_brush: {
        label: "Hard Region Add Brush",
        detail: "Drag to add cells to the selected hard region."
      },
      must_avoid_free_rectangle: {
        label: "Hard Region Add Box",
        detail: "Click two corners to add a box to the selected hard region."
      },
      soft_point: {
        label: "Soft Region Point",
        detail: "Click once to create an editable soft-preference circle."
      },
      soft_rectangle: {
        label: "Soft Region Rectangle",
        detail: "Click two opposite corners to create a soft-preference rectangle."
      },
      soft_brush: {
        label: "Soft Region Brush",
        detail: "Drag to paint cells into a freeform soft region."
      },
      soft_free_rectangle: {
        label: "Soft Region Add Box",
        detail: "Click two corners to append a box to a freeform soft region."
      },
      must_pass_free_erase: {
        label: "Hard Region Erase Brush",
        detail: "Drag over the selected hard region to remove cells."
      },
      must_pass_free_erase_rectangle: {
        label: "Hard Region Erase Box",
        detail: "Click two corners to remove a box from the selected hard region."
      },
      must_avoid_free_erase: {
        label: "Hard Region Erase Brush",
        detail: "Drag over the selected hard region to remove cells."
      },
      must_avoid_free_erase_rectangle: {
        label: "Hard Region Erase Box",
        detail: "Click two corners to remove a box from the selected hard region."
      },
      soft_free_erase: {
        label: "Soft Region Erase Brush",
        detail: "Drag over the selected soft region to remove cells."
      },
      soft_free_erase_rectangle: {
        label: "Soft Region Erase Box",
        detail: "Click two corners to remove a box from the selected soft region."
      }
    };
    const MODE_ORDER = [
      "route_draw",
      "route_erase",
      "route_adjust",
    ];
    const TEMPLATE_HARD_CONSTRAINT_PADDING_METERS = 1;
    const TRAVERSABLE_OBJECT_CATEGORIES = new Set(["doorframe", "doorway"]);
    const DIFFICULTY_LEVELS = ["easy", "hard", "extreme"];
    const GLOBAL_DIFFICULTY_SPLITS = ["train", "valunseen"];
    const GLOBAL_DIFFICULTY_SPLIT_LABELS = {
      train: "Training",
      valunseen: "Val-Unseen",
    };
    const CONSTRAINT_USAGE_KEYS = [
      "must_avoid",
      "near_preference",
      "far_preference",
      "relative_preference",
      "path_shape_preference",
    ];
    const CONSTRAINT_PERCENTAGE_KEYS = new Set([
      "near_preference",
      "far_preference",
      "relative_preference",
      "path_shape_preference",
    ]);
    const CONSTRAINT_USAGE_LABELS = {
      must_avoid: "Must-avoid",
      near_preference: "Near preference",
      far_preference: "Far preference",
      relative_preference: "Relative preference",
      path_shape_preference: "Path-shape",
    };
    const MIN_OBJECT_VISUAL_SIDE = 3;
    const SMALL_OBJECT_NAME_MAX_CELLS = MIN_OBJECT_VISUAL_SIDE * MIN_OBJECT_VISUAL_SIDE;
    const OBJECT_INSPECTION_RING_PADDING_GRIDS = 5;
    const HIDDEN_OBJECT_CATEGORIES = new Set(["large_ring", "wall"]);
    const FORCE_SMALL_OBJECT_NAME_CATEGORIES = new Set([
      "pillow",
      "bowl",
      "sink_basin",
      "teddy_bear",
      "remote_control",
      "vase",
      "coffee_machine",
    ]);
    const METRIC_EPSILON = 1e-6;
    const RELATIVE_PREFERENCE_HUMAN_BETA = 0.5;
    const RELATIVE_PREFERENCE_MIN_MARGIN = 0.2;
    const RELATIVE_PREFERENCE_MIN_VALID_LENGTH_GRID = 0.05;

    let mapState = cloneMapState(INITIAL_STATE.map_state);
    let mapSummaries = INITIAL_STATE.map_summaries.slice();
    let globalDifficultyTotals = normalizeGlobalDifficultyTotals(INITIAL_STATE.global_difficulty_totals);
    let mapDifficultyTotals = normalizeDifficultyCounts(INITIAL_STATE.map_difficulty_totals);
    let globalConstraintUsage = normalizeGlobalConstraintUsage(INITIAL_STATE.global_constraint_usage);
    let currentMapId = INITIAL_STATE.current_map_id;
    let paths = { ...INITIAL_STATE.paths };
    let samples = cloneSamples(INITIAL_STATE.annotation_state.samples);
    let activeSampleId = INITIAL_STATE.annotation_state.active_sample_id;
    let templates = cloneTemplates(INITIAL_STATE.template_instruction_state.templates);
    let templateDistanceRule = { ...INITIAL_STATE.template_instruction_state.distance_rule };
    let activeTemplateId = null;
    let selectedTemplateObjectId = null;
    let currentMode = "route_draw";
    let zoom = Math.min(12, Math.max(1, Number(INITIAL_STATE.arguments?.zoom ?? 6)));
    let showGrid = true;
    let showRooms = true;
    let showObjects = true;
    let showSmallObjectNames = false;
    let showHardConstraints = true;
    let showSoftConstraints = true;
    let isPainting = false;
    let lastPaintCell = null;
    let astarStartCell = null;
    let routeGestureStartCell = null;
    let routeGestureDragged = false;
    let activeRouteSegment = null;
    let routeAdjustIndex = null;
    let routeAdjustTargetCell = null;
    let pendingConstraintCorner = null;
    let selectedConstraintId = null;
    let selectedSoftConstraintId = null;
    let inspectorSelection = null;
    let inspectedMapEntity = null;
    let draggedMustPassId = null;
    let draggedConstraintRef = null;
    let referencePickTarget = null;
    let undoHistory = [];
    let gestureUndoRecorded = false;
    let drawGridFrame = null;
    let sessionCreatedBy = localStorage.getItem("sempathbench_created_by") || "";
    let objectCategoryCountsCache = null;
    let expandedObjectGridCache = null;
    let expandedObjectStackGridCache = null;

    const canvas = document.getElementById("gridCanvas");
    const ctx = canvas.getContext("2d");
    const canvasBox = document.getElementById("canvasBox");
    const statusEl = document.getElementById("status");
    const mapSelect = document.getElementById("mapSelect");
    const mapInfo = document.getElementById("mapInfo");
    const globalDifficultySummaryEl = document.getElementById("globalDifficultySummary");
    const globalConstraintUsageEl = document.getElementById("globalConstraintUsage");
    const difficultySummaryEl = document.getElementById("difficultySummary");
    const modeList = document.getElementById("modeList");
    const pathsEl = document.getElementById("paths");
    const sampleListEl = document.getElementById("sampleList");
    const templateDetailsEl = document.getElementById("templateDetails");
    const objectCountFilter = document.getElementById("objectCountFilter");
    const statusFilter = document.getElementById("statusFilter");
    const difficultyFilter = document.getElementById("difficultyFilter");
    const labeledInstructionSelect = document.getElementById("labeledInstructionSelect");
    const zoomSlider = document.getElementById("zoomSlider");
    const zoomValue = document.getElementById("zoomValue");
    const gridToggle = document.getElementById("gridToggle");
    const roomToggle = document.getElementById("roomToggle");
    const objectToggle = document.getElementById("objectToggle");
    const smallObjectNameToggle = document.getElementById("smallObjectNameToggle");
    const hardConstraintToggle = document.getElementById("hardConstraintToggle");
    const softConstraintToggle = document.getElementById("softConstraintToggle");
    const mapInspector = document.getElementById("mapInspector");
    const hcsScoreEl = document.getElementById("hcsScore");
    const scsScoreEl = document.getElementById("scsScore");
    const pathLengthScoreEl = document.getElementById("pathLengthScore");
    const sampleNameInput = document.getElementById("sampleNameInput");
    const instructionInput = document.getElementById("instructionInput");
    const difficultyLevelInput = document.getElementById("difficultyLevelInput");
    const feasibleInput = document.getElementById("feasibleInput");
    const createdByInput = document.getElementById("createdByInput");
    const startPoseInfo = document.getElementById("startPoseInfo");
    const sequenceConstraintListEl = document.getElementById("sequenceConstraintList");
    const globalConstraintListEl = document.getElementById("globalConstraintList");
    const hardEditorSection = document.getElementById("hardEditorSection");
    const softEditorSection = document.getElementById("softEditorSection");
    const mustPassListEl = document.getElementById("mustPassList");
    const mustAvoidListEl = document.getElementById("mustAvoidList");
    const constraintShapeInput = document.getElementById("constraintShapeInput");
    const constraintRadiusInput = document.getElementById("constraintRadiusInput");
    const constraintWidthInput = document.getElementById("constraintWidthInput");
    const constraintHeightInput = document.getElementById("constraintHeightInput");
    const constraintLabelInput = document.getElementById("constraintLabelInput");
    const constraintScopeInput = document.getElementById("constraintScopeInput");
    const constraintOrderInfo = document.getElementById("constraintOrderInfo");
    const deleteConstraintBtn = document.getElementById("deleteConstraintBtn");
    const softConstraintListEl = document.getElementById("softConstraintList");
    const softPreferenceTypeInput = document.getElementById("softPreferenceTypeInput");
    const softConstraintShapeInput = document.getElementById("softConstraintShapeInput");
    const softConstraintRadiusInput = document.getElementById("softConstraintRadiusInput");
    const softConstraintWidthInput = document.getElementById("softConstraintWidthInput");
    const softConstraintHeightInput = document.getElementById("softConstraintHeightInput");
    const softConstraintLabelInput = document.getElementById("softConstraintLabelInput");
    const softConstraintScopeInput = document.getElementById("softConstraintScopeInput");
    const relativeObjectInputs = document.getElementById("relativeObjectInputs");
    const relativeObjectAInput = document.getElementById("relativeObjectAInput");
    const relativeObjectBInput = document.getElementById("relativeObjectBInput");
    const softReferenceRegionInputs = document.getElementById("softReferenceRegionInputs");
    const softReferenceObjectLabel = document.getElementById("softReferenceObjectLabel");
    const softReferenceObjectInput = document.getElementById("softReferenceObjectInput");
    const softReferenceRegionInfo = document.getElementById("softReferenceRegionInfo");
    const softSmoothnessInfo = document.getElementById("softSmoothnessInfo");
    const pickSoftReferenceBtn = document.getElementById("pickSoftReferenceBtn");
    const pickRelativeObjectABtn = document.getElementById("pickRelativeObjectABtn");
    const pickRelativeObjectBBtn = document.getElementById("pickRelativeObjectBBtn");
    const eraseSoftFreeformBtn = document.getElementById("eraseSoftFreeformBtn");
    const eraseHardFreeformBtn = document.getElementById("eraseHardFreeformBtn");
    const eraseSoftBoxBtn = document.getElementById("eraseSoftBoxBtn");
    const eraseHardBoxBtn = document.getElementById("eraseHardBoxBtn");
    const deleteSoftConstraintBtn = document.getElementById("deleteSoftConstraintBtn");
    const createMustPassBtn = document.getElementById("createMustPassBtn");
    const createMustAvoidBtn = document.getElementById("createMustAvoidBtn");
    const createSoftConstraintBtn = document.getElementById("createSoftConstraintBtn");
    const passCreationTools = document.getElementById("passCreationTools");
    const avoidCreationTools = document.getElementById("avoidCreationTools");
    const softCreationTools = document.getElementById("softCreationTools");
    const addPassCircleBtn = document.getElementById("addPassCircleBtn");
    const addPassRectangleBtn = document.getElementById("addPassRectangleBtn");
    const paintPassBrushBtn = document.getElementById("paintPassBrushBtn");
    const addPassBoxBtn = document.getElementById("addPassBoxBtn");
    const addAvoidCircleBtn = document.getElementById("addAvoidCircleBtn");
    const addAvoidRectangleBtn = document.getElementById("addAvoidRectangleBtn");
    const paintAvoidBrushBtn = document.getElementById("paintAvoidBrushBtn");
    const addAvoidBoxBtn = document.getElementById("addAvoidBoxBtn");
    const addSoftCircleBtn = document.getElementById("addSoftCircleBtn");
    const addSoftRectangleBtn = document.getElementById("addSoftRectangleBtn");
    const paintSoftBrushBtn = document.getElementById("paintSoftBrushBtn");
    const addSoftBoxBtn = document.getElementById("addSoftBoxBtn");
    const undoLastAStarBtn = document.getElementById("undoLastAStarBtn");
    const focusAnnotationBtn = document.getElementById("focusAnnotationBtn");

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function recordUndoSnapshot(label) {
      const sample = getActiveSample();
      undoHistory.push({
        label,
        activeSampleId,
        activeTemplateId,
        sample: JSON.parse(JSON.stringify(sample)),
        selectedConstraintId,
        selectedSoftConstraintId,
        astarStartCell: astarStartCell ? astarStartCell.slice() : null,
      });
      if (undoHistory.length > 100) {
        undoHistory.shift();
      }
    }

    function recordGestureUndo(label) {
      if (gestureUndoRecorded) {
        return;
      }
      recordUndoSnapshot(label);
      gestureUndoRecorded = true;
    }

    function undoLastOperation() {
      const snapshot = undoHistory.pop();
      if (!snapshot) {
        setStatus("Nothing is available to undo.");
        return;
      }
      const sampleIndex = samples.findIndex(
        sample => sample.sample_id === snapshot.activeSampleId
      );
      if (sampleIndex < 0) {
        setStatus("The previous sample is no longer available.");
        return;
      }
      samples[sampleIndex] = cloneSamples([snapshot.sample])[0];
      activeSampleId = snapshot.activeSampleId;
      activeTemplateId = snapshot.activeTemplateId;
      selectedConstraintId = snapshot.selectedConstraintId;
      selectedSoftConstraintId = snapshot.selectedSoftConstraintId;
      astarStartCell = snapshot.astarStartCell;
      pendingConstraintCorner = null;
      referencePickTarget = null;
      syncFormFromActiveSample();
      renderSampleList();
      drawGrid();
      setStatus(`Undid: ${snapshot.label}.`);
    }

    function formatGridPoint(point) {
      if (!Array.isArray(point) || point.length !== 2) {
        return "not set";
      }
      return `(${Number(point[0])}, ${Number(point[1])})`;
    }

	    function cloneMapState(state) {
	      return {
        metadata: {
          ...state.metadata,
          map_info: state.metadata?.map_info ? { ...state.metadata.map_info } : undefined,
        },
        layers: {
          occupancy: state.layers.occupancy.map(row => row.slice()),
          room: state.layers.room.map(row => row.slice()),
          object_instance: state.layers.object_instance.map(row => row.slice()),
        },
        room_instances: state.room_instances.map(item => ({
          id: item.id,
          category: item.category,
          name: item.name,
          attributes: Array.isArray(item.attributes) ? item.attributes.slice() : [],
        })),
        object_instances: state.object_instances.map(item => ({
          id: item.id,
          category: item.category,
          name: item.name,
          attributes: Array.isArray(item.attributes) ? item.attributes.slice() : [],
          center_grid: Array.isArray(item.center_grid) && item.center_grid.length === 2
            ? [Number(item.center_grid[0]), Number(item.center_grid[1])]
            : null,
          metadata_grid_cell: Array.isArray(item.metadata_grid_cell) && item.metadata_grid_cell.length === 2
            ? [Number(item.metadata_grid_cell[0]), Number(item.metadata_grid_cell[1])]
            : null,
          assignment_method: item.assignment_method || "",
          room_overlap_grid_cells: Number(item.room_overlap_grid_cells || 0),
          num_grid_cells: Number(item.num_grid_cells || 0),
        })),
        object_footprints: state.object_footprints && typeof state.object_footprints === "object"
          ? Object.fromEntries(Object.entries(state.object_footprints).map(([key, value]) => [
              key,
              {
                ...value,
                cells: Array.isArray(value?.cells)
                  ? value.cells.map(cell => [Number(cell[0]), Number(cell[1])])
                  : [],
              },
            ]))
          : {},
        cell_object_ids: state.cell_object_ids && typeof state.cell_object_ids === "object"
          ? Object.fromEntries(Object.entries(state.cell_object_ids).map(([key, value]) => [
              key,
              Array.isArray(value) ? value.map(Number).filter(Number.isInteger) : [],
            ]))
          : {},
	      };
	    }

	    function defaultGlobalClearanceConstraint() {
	      return {
	        constraint_id: "__global_clearance__",
	        preference_type: "clearance",
	        shape: "freeform",
	        center: [0, 0],
	        radius: 1,
	        width: 1,
	        height: 1,
	        object_ids: [],
	        reference_regions: [],
	        reference_region: null,
	        D_ref: null,
	        C_smooth_ref: null,
	        C_clear_ref: null,
	        reference_trajectory: [],
	        reference_waypoint_count: 0,
	        scope: { type: "global", from_order: null, to_order: null },
	        label: "Clearance",
	        initialized: true,
	        cells: [],
	      };
	    }

	    function ensureGlobalClearanceConstraint(softConstraints) {
	      const constraints = Array.isArray(softConstraints) ? softConstraints : [];
	      if (constraints.some(constraint => constraint.preference_type === "clearance")) {
	        return constraints;
	      }
	      return [...constraints, defaultGlobalClearanceConstraint()];
	    }

	    function cloneSamples(rawSamples) {
	      return rawSamples.map(sample => ({
        sample_id: sample.sample_id,
        map_id: sample.map_id,
        template_instruction_id: sample.template_instruction_id || null,
        name: sample.name,
        instruction: sample.instruction,
        start_pose: {
          row: sample.start_pose?.row ?? null,
          col: sample.start_pose?.col ?? null,
        },
        expert_route: Array.isArray(sample.expert_route)
          ? sample.expert_route.map(point => [point[0], point[1]])
          : [],
        astar_segments: Array.isArray(sample.astar_segments)
          ? sample.astar_segments.map(segment =>
              segment.map(point => [point[0], point[1]])
            )
          : [],
        route_segments: Array.isArray(sample.route_segments)
          ? sample.route_segments.map(segment => ({
              kind: segment.kind || "manual",
              cells: Array.isArray(segment.cells)
                ? segment.cells.map(point => [point[0], point[1]])
                : [],
            }))
          : [],
        feasible: Boolean(sample.feasible),
        notes: sample.notes || "",
        created_by: sample.created_by || "",
        created_at: sample.created_at || "",
        updated_at: sample.updated_at || "",
        events: Array.isArray(sample.events) ? sample.events.slice() : [],
        constraints: Array.isArray(sample.constraints) ? sample.constraints.slice() : [],
        hard_constraints: Array.isArray(sample.hard_constraints)
          ? sample.hard_constraints.map(cloneHardConstraint)
          : [],
        soft_constraints: ensureGlobalClearanceConstraint(
          Array.isArray(sample.soft_constraints)
            ? sample.soft_constraints.map(cloneSoftConstraint)
            : []
        ),
        ordering: sample.ordering || "",
        difficulty_level: sample.difficulty_level || "",
        valid_routes: Array.isArray(sample.valid_routes) ? sample.valid_routes.slice() : [],
      }));
    }

    function cloneHardConstraint(constraint) {
      return {
        constraint_id: constraint.constraint_id,
        kind: constraint.kind,
        shape: constraint.shape,
        center: [Number(constraint.center[0]), Number(constraint.center[1])],
        radius: Number(constraint.radius ?? 6),
        width: Number(constraint.width ?? 12),
        height: Number(constraint.height ?? 12),
        order: constraint.order ?? null,
        source: constraint.source || "custom",
        object_id: constraint.object_id ?? null,
        scope: cloneConstraintScope(constraint.scope),
        label: constraint.label || (constraint.kind === "must_avoid" ? "Must Avoid" : "Must Pass"),
        initialized: constraint.initialized !== false,
        cells: Array.isArray(constraint.cells)
          ? constraint.cells.map(cell => [cell[0], cell[1]])
          : [],
      };
    }

    function cloneConstraintScope(scope) {
      if (scope?.type === "between_hard_constraints") {
        const fromOrder = Number(scope.from_order);
        const toOrder = Number(scope.to_order);
        if (Number.isInteger(fromOrder) && Number.isInteger(toOrder) && toOrder > fromOrder) {
          return {
            type: "between_hard_constraints",
            from_order: fromOrder,
            to_order: toOrder,
          };
        }
      }
      return { type: "global", from_order: null, to_order: null };
    }

	    function cloneSoftConstraint(constraint) {
	      const regionless = isRegionlessTrajectoryMetricPreference(constraint.preference_type);
	      const center = Array.isArray(constraint.center) && constraint.center.length === 2
	        ? [Number(constraint.center[0]), Number(constraint.center[1])]
	        : [0, 0];
	      let referenceRegion = null;
      if (constraint.reference_region && typeof constraint.reference_region === "object") {
        referenceRegion = {
          mode: constraint.reference_region.mode === "object" ? "object" : "brush",
          object_id: Number.isInteger(constraint.reference_region.object_id)
            ? constraint.reference_region.object_id
            : null,
          cells: Array.isArray(constraint.reference_region.cells)
            ? constraint.reference_region.cells.map(cell => [cell[0], cell[1]])
            : [],
        };
      } else if (
        Array.isArray(constraint.reference_point) &&
        constraint.reference_point.length === 2
      ) {
        referenceRegion = {
          mode: "brush",
          object_id: null,
          cells: [[
            Math.round(Number(constraint.reference_point[0])),
            Math.round(Number(constraint.reference_point[1])),
          ]],
        };
      }
      return {
	        constraint_id: constraint.constraint_id,
	        preference_type: constraint.preference_type,
	        shape: constraint.shape || (regionless ? "freeform" : "circle"),
	        center,
	        radius: Number(constraint.radius ?? (regionless ? 1 : 6)),
	        width: Number(constraint.width ?? (regionless ? 1 : 12)),
	        height: Number(constraint.height ?? (regionless ? 1 : 12)),
        object_ids: Array.isArray(constraint.object_ids)
          ? constraint.object_ids.slice()
          : [],
        reference_regions: Array.isArray(constraint.reference_regions)
          ? constraint.reference_regions.map(region => ({
              object_id: region.object_id,
              cells: Array.isArray(region.cells)
                ? region.cells.map(cell => [cell[0], cell[1]])
                : [],
            }))
          : [],
        reference_region: referenceRegion,
        D_ref: Number.isFinite(constraint.D_ref) ? Number(constraint.D_ref) : null,
        C_smooth_ref: Number.isFinite(constraint.C_smooth_ref)
          ? Number(constraint.C_smooth_ref)
          : null,
        C_clear_ref: Number.isFinite(constraint.C_clear_ref)
          ? Number(constraint.C_clear_ref)
          : null,
        reference_trajectory: Array.isArray(constraint.reference_trajectory)
          ? constraint.reference_trajectory.map(point => [point[0], point[1]])
          : [],
        reference_waypoint_count: Number(constraint.reference_waypoint_count || 0),
        path_shape_annotation:
          constraint.path_shape_annotation &&
          typeof constraint.path_shape_annotation === "object"
            ? JSON.parse(JSON.stringify(constraint.path_shape_annotation))
            : null,
        scope: cloneConstraintScope(constraint.scope),
        label: constraint.label || "Soft Constraint",
        initialized: constraint.initialized !== false,
        cells: Array.isArray(constraint.cells)
          ? constraint.cells.map(cell => [cell[0], cell[1]])
          : [],
      };
    }

    function cloneTemplates(rawTemplates) {
      return rawTemplates.map(template => ({
        template_instruction_id: template.template_instruction_id,
        map_id: template.map_id,
        object_count: template.object_count,
        objects: template.objects.map(item => ({
          order: item.order,
          object_id: item.object_id,
          name: item.name,
          category: item.category,
          center: [Number(item.center[0]), Number(item.center[1])],
        })),
        segment_distances: template.segment_distances.map(Number),
        status: template.status,
        labeled_instruction_id: template.labeled_instruction_id || null,
      }));
    }

    function templateDistanceUnit() {
      if (templateDistanceRule?.metric === "object_centroid_euclidean_meters") {
        return "m";
      }
      if (templateDistanceRule?.metric === "object_centroid_euclidean_grid") {
        return "grid";
      }
      return "";
    }

    function formatTemplateDistance(value) {
      const unit = templateDistanceUnit();
      const formatted = Number(value).toFixed(1);
      return unit ? `${formatted} ${unit}` : formatted;
    }

    function roomInstanceMap() {
      return new Map(mapState.room_instances.map(item => [item.id, item]));
    }

    function objectInstanceMap() {
      return new Map(mapState.object_instances.map(item => [item.id, item]));
    }

    function resetObjectVisualizationCaches() {
      objectCategoryCountsCache = null;
      expandedObjectGridCache = null;
      expandedObjectStackGridCache = null;
    }

    function objectCategoryCounts() {
      if (objectCategoryCountsCache) {
        return objectCategoryCountsCache;
      }
      const counts = new Map();
      for (const object of mapState.object_instances) {
        counts.set(object.category, (counts.get(object.category) || 0) + 1);
      }
      objectCategoryCountsCache = counts;
      return counts;
    }

    function objectCategoryCount(category) {
      return objectCategoryCounts().get(category) || 0;
    }

    function normalizedObjectCategory(category) {
      return String(category || "").trim().toLowerCase().replace(/[\\s-]+/g, "_");
    }

    function shouldHideObjectCategory(category) {
      return HIDDEN_OBJECT_CATEGORIES.has(normalizedObjectCategory(category));
    }

    function normalizedGridCell(cell) {
      if (!Array.isArray(cell) || cell.length !== 2) {
        return null;
      }
      const row = Math.round(Number(cell[0]));
      const col = Math.round(Number(cell[1]));
      if (
        !Number.isInteger(row) ||
        !Number.isInteger(col) ||
        row < 0 ||
        row >= GRID_SIZE ||
        col < 0 ||
        col >= GRID_SIZE
      ) {
        return null;
      }
      return [row, col];
    }

    function objectFootprintRecord(objectId) {
      const footprints = mapState.object_footprints;
      if (!footprints || typeof footprints !== "object") {
        return null;
      }
      const record = footprints[String(objectId)];
      return record && typeof record === "object" ? record : null;
    }

    function objectCellsFromFootprint(objectId) {
      const record = objectFootprintRecord(objectId);
      if (!record || !Array.isArray(record.cells)) {
        return [];
      }
      const cells = [];
      const seen = new Set();
      for (const rawCell of record.cells) {
        const cell = normalizedGridCell(rawCell);
        if (!cell) {
          continue;
        }
        const key = `${cell[0]},${cell[1]}`;
        if (!seen.has(key)) {
          cells.push(cell);
          seen.add(key);
        }
      }
      return cells;
    }

    function objectIdsFromCellObjectIndex(row, col) {
      const cellIndex = mapState.cell_object_ids;
      if (!cellIndex || typeof cellIndex !== "object") {
        return [];
      }
      const rawIds = cellIndex[`${row},${col}`];
      if (!Array.isArray(rawIds)) {
        return [];
      }
      const ids = [];
      for (const rawId of rawIds) {
        const objectId = Number(rawId);
        if (Number.isInteger(objectId) && objectId > 0 && !ids.includes(objectId)) {
          ids.push(objectId);
        }
      }
      return ids;
    }

    function objectCellCounts() {
      const counts = new Map();
      for (const object of mapState.object_instances) {
        counts.set(object.id, cellsForObject(object.id).length);
      }
      return counts;
    }

    function objectDisplaySize(objectId, objectMap, visibleCounts = null) {
      const object = objectMap.get(objectId);
      const authoredSize = Number(object?.num_grid_cells || 0);
      if (Number.isFinite(authoredSize) && authoredSize > 0) {
        return authoredSize;
      }
      const visibleSize = Number(visibleCounts?.get(objectId) || 0);
      return visibleSize > 0 ? visibleSize : Number.POSITIVE_INFINITY;
    }

    function shouldDisplayObjectReplace(existingId, candidateId, objectMap, visibleCounts) {
      if (existingId === 0 || existingId === candidateId) {
        return true;
      }
      const candidateSize = objectDisplaySize(candidateId, objectMap, visibleCounts);
      const existingSize = objectDisplaySize(existingId, objectMap, visibleCounts);
      if (candidateSize !== existingSize) {
        return candidateSize < existingSize;
      }
      return candidateId < existingId;
    }

    function centerHintCellForObject(object) {
      const center = Array.isArray(object?.center_grid) && object.center_grid.length === 2
        ? object.center_grid
        : Array.isArray(object?.metadata_grid_cell) && object.metadata_grid_cell.length === 2
          ? object.metadata_grid_cell
          : null;
      if (!center) {
        return null;
      }
      const row = Math.round(Number(center[0]));
      const col = Math.round(Number(center[1]));
      if (
        !Number.isInteger(row) ||
        !Number.isInteger(col) ||
        row < 0 ||
        row >= GRID_SIZE ||
        col < 0 ||
        col >= GRID_SIZE
      ) {
        return null;
      }
      return [row, col];
    }

    function topDisplayObjectIdFromIds(objectIds, objectMap, visibleCounts) {
      if (!Array.isArray(objectIds) || objectIds.length === 0) {
        return 0;
      }
      return objectIds.slice().sort((firstId, secondId) => {
        const firstSize = objectDisplaySize(firstId, objectMap, visibleCounts);
        const secondSize = objectDisplaySize(secondId, objectMap, visibleCounts);
        if (firstSize !== secondSize) {
          return firstSize - secondSize;
        }
        return firstId - secondId;
      })[0] || 0;
    }

    function expandedObjectStackGrid() {
      if (expandedObjectStackGridCache) {
        return expandedObjectStackGridCache;
      }
      const objectGrid = mapState.layers.object_instance;
      const counts = objectCellCounts();
      const objectMap = objectInstanceMap();
      const visualBounds = new Map();
      const stackGrid = objectGrid.map(row => row.map(() => []));

      function recordVisualCell(objectId, row, col) {
        if (objectId === 0) {
          return;
        }
        const object = objectMap.get(objectId);
        if (!object || shouldHideObjectCategory(object.category)) {
          return;
        }
        if (!visualBounds.has(objectId)) {
          visualBounds.set(objectId, {
            minRow: row,
            maxRow: row,
            minCol: col,
            maxCol: col,
            count: 0,
          });
        }
        const bounds = visualBounds.get(objectId);
        bounds.minRow = Math.min(bounds.minRow, row);
        bounds.maxRow = Math.max(bounds.maxRow, row);
        bounds.minCol = Math.min(bounds.minCol, col);
        bounds.maxCol = Math.max(bounds.maxCol, col);
        bounds.count += 1;
      }

      function addObjectToDisplayCell(row, col, objectId) {
        if (objectId === 0) {
          return;
        }
        const object = objectMap.get(objectId);
        if (!object || shouldHideObjectCategory(object.category)) {
          return;
        }
        const stack = stackGrid[row][col];
        if (!stack.includes(objectId)) {
          stack.push(objectId);
          recordVisualCell(objectId, row, col);
        }
      }

      const indexedCells = mapState.cell_object_ids && typeof mapState.cell_object_ids === "object"
        ? Object.entries(mapState.cell_object_ids)
        : [];
      for (const [key, objectIds] of indexedCells) {
        const [rowText, colText] = key.split(",");
        const row = Number(rowText);
        const col = Number(colText);
        if (
          !Number.isInteger(row) ||
          !Number.isInteger(col) ||
          row < 0 ||
          row >= GRID_SIZE ||
          col < 0 ||
          col >= GRID_SIZE ||
          !Array.isArray(objectIds)
        ) {
          continue;
        }
        for (const rawObjectId of objectIds) {
          const objectId = Number(rawObjectId);
          if (Number.isInteger(objectId)) {
            addObjectToDisplayCell(row, col, objectId);
          }
        }
      }

      if (indexedCells.length === 0) {
        for (const object of mapState.object_instances) {
          for (const [row, col] of objectCellsFromFootprint(object.id)) {
            addObjectToDisplayCell(row, col, object.id);
          }
        }
      }

      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          addObjectToDisplayCell(row, col, objectGrid[row][col]);
        }
      }

      for (const object of mapState.object_instances) {
        if (shouldHideObjectCategory(object.category)) {
          continue;
        }
        if (visualBounds.has(object.id)) {
          continue;
        }
        const cell = centerHintCellForObject(object);
        if (!cell) {
          continue;
        }
        const [row, col] = cell;
        addObjectToDisplayCell(row, col, object.id);
      }

      for (const [objectId, bounds] of visualBounds.entries()) {
        const width = bounds.maxCol - bounds.minCol + 1;
        const height = bounds.maxRow - bounds.minRow + 1;
        if (
          width >= MIN_OBJECT_VISUAL_SIDE &&
          height >= MIN_OBJECT_VISUAL_SIDE &&
          bounds.count >= MIN_OBJECT_VISUAL_SIDE * MIN_OBJECT_VISUAL_SIDE
        ) {
          continue;
        }
        const centerRow = Math.round((bounds.minRow + bounds.maxRow) / 2);
        const centerCol = Math.round((bounds.minCol + bounds.maxCol) / 2);
        const startRow = Math.max(
          0,
          Math.min(GRID_SIZE - MIN_OBJECT_VISUAL_SIDE, centerRow - 1)
        );
        const startCol = Math.max(
          0,
          Math.min(GRID_SIZE - MIN_OBJECT_VISUAL_SIDE, centerCol - 1)
        );
        for (
          let expandedRow = startRow;
          expandedRow < startRow + MIN_OBJECT_VISUAL_SIDE;
          expandedRow += 1
        ) {
          for (
            let expandedCol = startCol;
            expandedCol < startCol + MIN_OBJECT_VISUAL_SIDE;
            expandedCol += 1
          ) {
            addObjectToDisplayCell(expandedRow, expandedCol, objectId);
          }
        }
      }

      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          stackGrid[row][col].sort((firstId, secondId) => {
            const firstSize = objectDisplaySize(firstId, objectMap, counts);
            const secondSize = objectDisplaySize(secondId, objectMap, counts);
            if (firstSize !== secondSize) {
              return secondSize - firstSize;
            }
            return firstId - secondId;
          });
        }
      }

      expandedObjectStackGridCache = stackGrid;
      return stackGrid;
    }

    function expandedObjectGrid() {
      if (expandedObjectGridCache) {
        return expandedObjectGridCache;
      }
      const objectMap = objectInstanceMap();
      const counts = objectCellCounts();
      const stackGrid = expandedObjectStackGrid();
      const displayGrid = stackGrid.map(row => row.map(
        objectIds => topDisplayObjectIdFromIds(objectIds, objectMap, counts)
      ));
      expandedObjectGridCache = displayGrid;
      return displayGrid;
    }

    function displayedObjectIdsAt(row, col) {
      return expandedObjectStackGrid()[row][col] || [];
    }

    function displayedObjectIdAt(row, col) {
      return topDisplayObjectIdFromIds(
        displayedObjectIdsAt(row, col),
        objectInstanceMap(),
        objectCellCounts()
      );
    }

    function occupancyNameFromValue(value) {
      for (const [name, meta] of Object.entries(currentLegends.occupancy)) {
        if (meta.value === value) {
          return name;
        }
      }
      return "free";
    }

    function hexToRgba(hex, alpha) {
      const value = hex.replace("#", "");
      const r = parseInt(value.slice(0, 2), 16);
      const g = parseInt(value.slice(2, 4), 16);
      const b = parseInt(value.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function textColorForHex(hex) {
      const value = String(hex || "").replace("#", "");
      const r = parseInt(value.slice(0, 2), 16);
      const g = parseInt(value.slice(2, 4), 16);
      const b = parseInt(value.slice(4, 6), 16);
      if (![r, g, b].every(Number.isFinite)) {
        return "#111111";
      }
      const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      return luminance < 0.58 ? "#ffffff" : "#111111";
    }

    function routeColorAt(index, total) {
      if (total <= 1) {
        return INITIAL_STATE.route_color;
      }
      const progress = index / (total - 1);
      const cycles = Math.max(1, Math.ceil(total / 300));
      const hue = (18 + progress * 320 * cycles) % 360;
      const saturation = 86;
      const lightness = 42 + progress * 20;
      return `hsl(${hue.toFixed(2)}, ${saturation}%, ${lightness.toFixed(2)}%)`;
    }

    function nowIso() {
      return new Date().toISOString();
    }

    function normalizeDifficultyLevel(value) {
      const normalized = String(value || "").trim().toLowerCase();
      return DIFFICULTY_LEVELS.includes(normalized) ? normalized : "";
    }

    function difficultyLabel(value) {
      const labels = {
        easy: "Easy",
        hard: "Hard",
        extreme: "Extreme hard",
      };
      return labels[normalizeDifficultyLevel(value)] || "Unset";
    }

    function cellKey(row, col) {
      return `${row}:${col}`;
    }

    function sameCell(first, second) {
      return Boolean(first && second && first[0] === second[0] && first[1] === second[1]);
    }

    function getActiveTemplate() {
      return templates.find(
        item => item.template_instruction_id === activeTemplateId
      ) || null;
    }

    function findSampleForTemplate(template) {
      if (template.labeled_instruction_id) {
        const sampleId = `instruction_${String(template.labeled_instruction_id).padStart(6, "0")}`;
        const sample = samples.find(item => item.sample_id === sampleId);
        if (sample) {
          return sample;
        }
      }
      return samples.find(
        item => item.template_instruction_id === template.template_instruction_id
      ) || null;
    }

    function templateDifficultyLevel(template) {
      const sample = findSampleForTemplate(template);
      return normalizeDifficultyLevel(sample?.difficulty_level);
    }

    function templateInstructionNumber(template) {
      const labeledId = Number(template.labeled_instruction_id);
      if (Number.isInteger(labeledId) && labeledId > 0) {
        return labeledId;
      }
      const sample = findSampleForTemplate(template);
      const match = String(sample?.sample_id || "").match(/^instruction_(\\d{6})$/);
      return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
    }

    function filteredTemplates() {
      const objectCount = Number(objectCountFilter.value);
      const status = statusFilter.value;
      const difficulty = normalizeDifficultyLevel(difficultyFilter.value);
      return templates.filter(
        template => (
          (!objectCount || template.object_count === objectCount) &&
          template.status === status &&
          (!difficulty || templateDifficultyLevel(template) === difficulty)
        )
      );
    }

    function orderedFilteredTemplates() {
      const visibleTemplates = filteredTemplates();
      if (statusFilter.value !== "labeled") {
        return visibleTemplates;
      }
      return visibleTemplates.sort((first, second) => {
        const firstNumber = templateInstructionNumber(first);
        const secondNumber = templateInstructionNumber(second);
        if (firstNumber !== secondNumber) {
          return firstNumber - secondNumber;
        }
        return first.template_instruction_id.localeCompare(second.template_instruction_id);
      });
    }

    function instructionSelectLabel(template) {
      const instructionNumber = templateInstructionNumber(template);
      const difficulty = templateDifficultyLevel(template);
      const parts = [
        Number.isFinite(instructionNumber)
          ? `#${instructionNumber} instruction_${String(instructionNumber).padStart(6, "0")}`
          : "Unknown instruction",
      ];
      if (difficulty) {
        parts.push(difficultyLabel(difficulty));
      }
      parts.push(`${template.object_count} objects`);
      return parts.join(" · ");
    }

    function createLocalSample(template) {
      const objectNames = template.objects.map(item => item.name);
      return {
        sample_id: `sample_${Math.random().toString(36).slice(2, 10)}`,
        map_id: currentMapId,
        template_instruction_id: template.template_instruction_id,
        name: objectNames.join(" -> "),
        instruction: "",
        start_pose: { row: null, col: null },
        expert_route: [],
        astar_segments: [],
        route_segments: [],
        feasible: true,
        notes: "",
        created_by: "",
        created_at: nowIso(),
        updated_at: nowIso(),
        events: [],
        constraints: [],
        hard_constraints: template.objects.map(object => {
          const radius = templateObjectHardConstraintRadius(object);
          return {
            constraint_id: `constraint_${Math.random().toString(36).slice(2, 10)}`,
            kind: "must_pass",
            shape: "circle",
            center: [object.center[0], object.center[1]],
            radius,
            width: radius * 2,
            height: radius * 2,
            order: object.order,
            source: "template_object",
            object_id: object.object_id,
            scope: { type: "global", from_order: null, to_order: null },
            label: "Must Pass",
            cells: [],
          };
        }),
	        soft_constraints: [defaultGlobalClearanceConstraint()],
        ordering: objectNames.join(" -> "),
        difficulty_level: "",
        valid_routes: [],
      };
    }

    function ensureSampleForTemplate(template) {
      let sample = findSampleForTemplate(template);
      if (!sample) {
        sample = createLocalSample(template);
        samples.push(sample);
      }
      activeSampleId = sample.sample_id;
      return sample;
    }

    function getActiveSample() {
      let sample = samples.find(item => item.sample_id === activeSampleId);
      if (!sample) {
        const template = getActiveTemplate() || templates[0];
        if (!template) {
          throw new Error("This map has no template instructions.");
        }
        activeTemplateId = template.template_instruction_id;
        sample = ensureSampleForTemplate(template);
      }
      return sample;
    }

    function persistFormIntoActiveSample() {
      if (!activeSampleId) {
        return;
      }
      const sample = getActiveSample();
      syncStartPoseFromRoute(sample);
      sample.name = sampleNameInput.value.trim() || "Untitled Sample";
      sample.instruction = instructionInput.value;
      sample.difficulty_level = normalizeDifficultyLevel(difficultyLevelInput.value);
      sample.feasible = feasibleInput.checked;
      sample.created_by = sessionCreatedBy;
      sample.updated_at = nowIso();
    }

    function clearAStarHistory(sample = getActiveSample()) {
      sample.astar_segments = [];
      sample.route_segments = [];
    }

    function syncFormFromActiveSample() {
      const sample = getActiveSample();
      syncStartPoseFromRoute(sample);
      sampleNameInput.value = sample.name || "";
      instructionInput.value = sample.instruction || "";
      difficultyLevelInput.value = normalizeDifficultyLevel(sample.difficulty_level);
      feasibleInput.checked = Boolean(sample.feasible);
      renderStartPoseInfo();
      renderChecks();
      renderTemplateDetails();
      renderConstraintEditor();
      renderSoftConstraintEditor();
    }

    function multilineToList(value) {
      return value
        .split("\\n")
        .map(item => item.trim())
        .filter(Boolean);
    }

    function listToMultiline(items) {
      return (items || []).join("\\n");
    }

    function renderMapSelect() {
      mapSelect.innerHTML = "";
      for (const summary of mapSummaries) {
        const option = document.createElement("option");
        option.value = summary.map_id;
        option.textContent = `${summary.name} [${summary.path || summary.map_id}]`;
        option.selected = summary.map_id === currentMapId;
        mapSelect.appendChild(option);
      }
    }

    function blankDifficultyCounts() {
      return Object.fromEntries(DIFFICULTY_LEVELS.map(level => [level, 0]));
    }

    function difficultyCountsTotal(counts) {
      return DIFFICULTY_LEVELS.reduce(
        (sum, level) => sum + Number(counts?.[level] || 0),
        0
      );
    }

    function normalizeDifficultyCounts(counts) {
      const normalized = blankDifficultyCounts();
      if (!counts || typeof counts !== "object") {
        return normalized;
      }
      for (const level of DIFFICULTY_LEVELS) {
        normalized[level] = Number(counts[level] || 0);
      }
      return normalized;
    }

    function normalizeGlobalDifficultyTotals(value) {
      const hasSplitShape = Boolean(value && typeof value === "object" && value.by_split);
      const total = normalizeDifficultyCounts(
        hasSplitShape ? value.total : value
      );
      const bySplit = {};
      for (const split of GLOBAL_DIFFICULTY_SPLITS) {
        bySplit[split] = normalizeDifficultyCounts(
          hasSplitShape ? value.by_split?.[split] : null
        );
      }
      const labels = {
        ...GLOBAL_DIFFICULTY_SPLIT_LABELS,
        ...(hasSplitShape ? value.split_labels || {} : {}),
      };
      return { total, by_split: bySplit, split_labels: labels };
    }

    function normalizeConstraintUsageItems(items, total = null) {
      const byKey = new Map(
        Array.isArray(items)
          ? items.map(item => [item.key, item])
          : []
      );
      const normalized = CONSTRAINT_USAGE_KEYS.map(key => {
        const item = byKey.get(key) || {};
        const count = Number(item.count || 0);
        return {
          key,
          label: item.label || CONSTRAINT_USAGE_LABELS[key] || key,
          count,
          percentage: CONSTRAINT_PERCENTAGE_KEYS.has(key) ? 0 : null,
        };
      });
      const normalizedTotal = total === null
        ? normalized.reduce((sum, item) => sum + item.count, 0)
        : Number(total || 0);
      const percentageTotal = normalized
        .filter(item => CONSTRAINT_PERCENTAGE_KEYS.has(item.key))
        .reduce((sum, item) => sum + item.count, 0);
      if (percentageTotal > 0) {
        for (const item of normalized) {
          item.percentage = CONSTRAINT_PERCENTAGE_KEYS.has(item.key)
            ? item.count * 100 / percentageTotal
            : null;
        }
      }
      return {
        total: normalizedTotal,
        items: normalized,
      };
    }

    function normalizeGlobalConstraintUsage(value) {
      const normalizedTotal = normalizeConstraintUsageItems(
        value?.items,
        value?.total ?? null
      );
      const bySplit = {};
      for (const split of GLOBAL_DIFFICULTY_SPLITS) {
        bySplit[split] = normalizeConstraintUsageItems(
          value?.by_split?.[split]?.items,
          value?.by_split?.[split]?.total ?? null
        );
      }
      return {
        total: normalizedTotal.total,
        items: normalizedTotal.items,
        by_split: bySplit,
      };
    }

    function renderDifficultySummaryBlock(container, counts, titleText) {
      container.innerHTML = "";
      const total = difficultyCountsTotal(counts);

      const title = document.createElement("div");
      title.className = "difficulty-summary-title";
      title.textContent = `${titleText} (${total})`;
      container.appendChild(title);

      const grid = document.createElement("div");
      grid.className = "difficulty-summary-grid";
      for (const level of DIFFICULTY_LEVELS) {
        const item = document.createElement("div");
        item.className = "difficulty-pill";
        const label = document.createElement("div");
        label.className = "difficulty-pill-label";
        label.textContent = difficultyLabel(level);
        const count = document.createElement("div");
        count.className = "difficulty-pill-count";
        count.textContent = String(Number(counts[level] || 0));
        item.appendChild(label);
        item.appendChild(count);
        grid.appendChild(item);
      }
      container.appendChild(grid);
    }

    function renderDifficultyPillGrid(counts) {
      const grid = document.createElement("div");
      grid.className = "difficulty-summary-grid";
      for (const level of DIFFICULTY_LEVELS) {
        const item = document.createElement("div");
        item.className = "difficulty-pill";
        const label = document.createElement("div");
        label.className = "difficulty-pill-label";
        label.textContent = difficultyLabel(level);
        const count = document.createElement("div");
        count.className = "difficulty-pill-count";
        count.textContent = String(Number(counts[level] || 0));
        item.appendChild(label);
        item.appendChild(count);
        grid.appendChild(item);
      }
      return grid;
    }

    function renderConstraintUsageList(usage) {
      const list = document.createElement("div");
      list.className = "constraint-usage-list";
      const items = Array.isArray(usage?.items) ? usage.items : [];
      for (const item of items) {
        const row = document.createElement("div");
        row.className = "constraint-usage-row";

        const label = document.createElement("div");
        label.className = "constraint-usage-label";
        label.textContent = item.label || item.key || "Unknown";

        const count = document.createElement("div");
        count.className = "constraint-usage-count";
        count.textContent = String(Number(item.count || 0));

        const percent = document.createElement("div");
        percent.className = "constraint-usage-percent";

        row.appendChild(label);
        row.appendChild(count);
        if (Number.isFinite(item.percentage)) {
          percent.textContent = `${Number(item.percentage).toFixed(1)}%`;
          row.appendChild(percent);
        }
        list.appendChild(row);
      }
      return list;
    }

    function renderGlobalDifficultySummary() {
      const totals = normalizeGlobalDifficultyTotals(globalDifficultyTotals);
      const usageTotals = normalizeGlobalConstraintUsage(globalConstraintUsage);
      globalDifficultyTotals = totals;
      globalConstraintUsage = usageTotals;
      globalDifficultySummaryEl.innerHTML = "";

      const total = difficultyCountsTotal(totals.total);
      const constraintTotal = Number(usageTotals.total || 0);
      const title = document.createElement("div");
      title.className = "difficulty-summary-title";
      title.textContent = `Saved totals across all maps (${total} instructions, ${constraintTotal} constraints)`;
      globalDifficultySummaryEl.appendChild(title);

      const list = document.createElement("div");
      list.className = "difficulty-split-list";
      for (const split of GLOBAL_DIFFICULTY_SPLITS) {
        const counts = totals.by_split[split] || blankDifficultyCounts();
        const splitTotal = difficultyCountsTotal(counts);
        const details = document.createElement("details");
        details.className = "difficulty-split";

        const summary = document.createElement("summary");
        const label = document.createElement("span");
        label.textContent = totals.split_labels[split] || split;
        const count = document.createElement("span");
        count.className = "difficulty-split-count";
        const splitConstraintTotal = Number(usageTotals.by_split[split]?.total || 0);
        count.textContent = `${splitTotal} instr / ${splitConstraintTotal} cons`;
        summary.appendChild(label);
        summary.appendChild(count);

        details.appendChild(summary);
        const difficultySubtitle = document.createElement("div");
        difficultySubtitle.className = "difficulty-split-subtitle";
        difficultySubtitle.textContent = "Difficulty";
        details.appendChild(difficultySubtitle);
        details.appendChild(renderDifficultyPillGrid(counts));
        const constraintSubtitle = document.createElement("div");
        constraintSubtitle.className = "difficulty-split-subtitle";
        constraintSubtitle.textContent = "Constraint usage";
        details.appendChild(constraintSubtitle);
        details.appendChild(renderConstraintUsageList(usageTotals.by_split[split]));
        list.appendChild(details);
      }
      globalDifficultySummaryEl.appendChild(list);
    }

    function renderGlobalConstraintUsage() {
      globalConstraintUsageEl.innerHTML = "";
      globalConstraintUsageEl.style.display = "none";
    }

    function renderDifficultySummary() {
      renderDifficultySummaryBlock(
        difficultySummaryEl,
        mapDifficultyTotals,
        "Saved difficulty totals for this map"
      );
    }

    function isContinuousConstraintPaintMode(modeName = currentMode) {
      return (
        modeName === "soft_brush" ||
        modeName === "soft_free_erase" ||
        modeName.endsWith("_brush") ||
        modeName.endsWith("_free_erase")
      );
    }

    function renderAfterConstraintPaint({ soft = false, hard = false } = {}) {
      if (isPainting && isContinuousConstraintPaintMode()) {
        scheduleDrawGrid();
        return;
      }
      if (soft) {
        renderSoftConstraintEditor();
      }
      if (hard) {
        renderConstraintEditor();
      }
      renderChecks();
      drawGrid();
    }

    function finishContinuousConstraintPaint() {
      renderConstraintEditor();
      renderSoftConstraintEditor();
      renderChecks();
      flushDrawGrid();
    }

    function renderMapInfo() {
      const metadata = mapState.metadata;
      mapInfo.innerHTML = "";
      const title = document.createElement("div");
      title.className = "sample-title";
      title.textContent = metadata.name || metadata.map_id;
      const meta = document.createElement("div");
      meta.className = "sample-meta";
      meta.textContent = `${metadata.map_id} | rooms=${mapState.room_instances.length} | objects=${mapState.object_instances.length}`;
      mapInfo.appendChild(title);
      mapInfo.appendChild(meta);
      if (metadata.map_split) {
        const splitInfo = document.createElement("div");
        splitInfo.className = "small";
        splitInfo.style.marginTop = "8px";
        const positionText = metadata.map_index
          ? ` | map_index=${metadata.map_index}`
          : "";
        const basisText = metadata.map_assignment_basis
          ? ` | basis=${metadata.map_assignment_basis}`
          : "";
        splitInfo.textContent = (
          `Map split: ${metadata.map_split}${positionText}${basisText}`
        );
        mapInfo.appendChild(splitInfo);
      }
      if (metadata.description) {
        const desc = document.createElement("div");
        desc.className = "small";
        desc.style.marginTop = "8px";
        desc.textContent = metadata.description;
        mapInfo.appendChild(desc);
      }
      renderDifficultySummary();
    }

    function renderPaths() {
      pathsEl.textContent = (
        `${paths.map_json}\\n${paths.template_instruction_json}\\n${paths.instruction_directory}`
      );
    }

    function renderModeList() {
      modeList.innerHTML = "";
      MODE_ORDER.forEach((modeName, index) => {
        const meta = MODES[modeName];
        const button = document.createElement("button");
        button.type = "button";
        button.className = `mode-btn${currentMode === modeName ? " active" : ""}`;
        button.addEventListener("click", () => setMode(modeName));

        const title = document.createElement("div");
        title.className = "mode-title";
        const shortcut = document.createElement("span");
        shortcut.className = "mode-shortcut";
        shortcut.textContent = String(index + 1);
        const label = document.createElement("span");
        label.textContent = meta.label;
        title.appendChild(shortcut);
        title.appendChild(label);
        const detail = document.createElement("div");
        detail.className = "mode-detail";
        detail.textContent = meta.detail;

        button.appendChild(title);
        button.appendChild(detail);
        modeList.appendChild(button);
      });
    }

    function selectTemplate(template, statusMessage = null) {
      if (!template || template.template_instruction_id === activeTemplateId) {
        return;
      }
      persistFormIntoActiveSample();
      activeTemplateId = template.template_instruction_id;
      selectedTemplateObjectId = null;
      selectedConstraintId = null;
      selectedSoftConstraintId = null;
      pendingConstraintCorner = null;
      undoHistory = [];
      ensureSampleForTemplate(template);
      astarStartCell = null;
      syncFormFromActiveSample();
      renderSampleList();
      drawGrid();
      focusActiveAnnotationSoon();
      setStatus(statusMessage || `Opened ${template.objects.map(object => object.name).join(" -> ")}.`);
    }

    function renderLabeledInstructionSelect(visibleTemplates) {
      labeledInstructionSelect.innerHTML = "";
      if (statusFilter.value !== "labeled") {
        labeledInstructionSelect.style.display = "none";
        return;
      }
      labeledInstructionSelect.style.display = "";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Jump to instruction...";
      labeledInstructionSelect.appendChild(placeholder);
      for (const template of visibleTemplates) {
        const option = document.createElement("option");
        option.value = template.template_instruction_id;
        option.textContent = instructionSelectLabel(template);
        labeledInstructionSelect.appendChild(option);
      }
      labeledInstructionSelect.value = visibleTemplates.some(
        template => template.template_instruction_id === activeTemplateId
      )
        ? activeTemplateId
        : "";
    }

    function renderSampleList() {
      const previousScrollTop = sampleListEl.scrollTop;
      sampleListEl.innerHTML = "";
      const visibleTemplates = orderedFilteredTemplates();
      renderLabeledInstructionSelect(visibleTemplates);
      if (visibleTemplates.length === 0) {
        const empty = document.createElement("div");
        empty.className = "item small";
        empty.textContent = "No templates match these filters.";
        sampleListEl.appendChild(empty);
        return;
      }

      for (const template of visibleTemplates) {
        const item = document.createElement("div");
        item.className = (
          `sample-item${template.template_instruction_id === activeTemplateId ? " active" : ""}`
        );
        item.addEventListener("click", () => {
          selectTemplate(template);
        });

        const title = document.createElement("div");
        title.className = "sample-title";
        const instructionNumber = templateInstructionNumber(template);
        const prefix = statusFilter.value === "labeled" && Number.isFinite(instructionNumber)
          ? `#${instructionNumber} · `
          : "";
        title.textContent = prefix + template.objects
          .map(object => `${object.order}. ${object.name}`)
          .join(" -> ");
        const meta = document.createElement("div");
        meta.className = "sample-meta";
        meta.textContent = template.segment_distances
          .map(formatTemplateDistance)
          .join(" / ");
        const badge = document.createElement("div");
        badge.className = `status-badge ${template.status}`;
        const difficulty = templateDifficultyLevel(template);
        badge.textContent = difficulty
          ? `${template.status} · ${difficultyLabel(difficulty)}`
          : template.status;

        item.appendChild(title);
        item.appendChild(meta);
        item.appendChild(badge);
        sampleListEl.appendChild(item);
      }
      sampleListEl.scrollTop = previousScrollTop;
    }

    function renderTemplateDetails() {
      const template = getActiveTemplate();
      templateDetailsEl.innerHTML = "";
      if (!template) {
        templateDetailsEl.textContent = "No template selected.";
        return;
      }
      for (const object of template.objects) {
        const card = document.createElement("div");
        card.className = "template-object-card";

        const row = document.createElement("button");
        row.type = "button";
        row.className = (
          `template-object${object.object_id === selectedTemplateObjectId ? " active" : ""}`
        );
        row.textContent = (
          `${object.order}. ${object.name} (${object.category}, id=${object.object_id})`
        );
        row.addEventListener("click", () => {
          selectedTemplateObjectId = (
            selectedTemplateObjectId === object.object_id ? null : object.object_id
          );
          renderTemplateDetails();
          drawGrid();
          setStatus(
            selectedTemplateObjectId === null
              ? `Cleared object highlight for ${object.name}.`
              : `Highlighted ${object.name} in black on the map.`
          );
        });
        card.appendChild(row);

        const objectItem = objectInstanceMap().get(object.object_id);
        const attributes = Array.isArray(objectItem?.attributes)
          ? objectItem.attributes.filter(attribute => typeof attribute === "string" && attribute.trim())
          : [];
        if (attributes.length > 0) {
          const attrList = document.createElement("div");
          attrList.className = "template-attribute-list";
          for (const attribute of attributes) {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "attribute-chip-btn";
            chip.textContent = attribute;
            chip.addEventListener("click", () => {
              insertInstructionSnippet(attribute);
              setStatus(`Inserted attribute from ${object.name} into the instruction.`);
            });
            attrList.appendChild(chip);
          }
          card.appendChild(attrList);
        }
        templateDetailsEl.appendChild(card);
      }
      const distances = document.createElement("div");
      distances.className = "sample-meta";
      distances.style.marginTop = "8px";
      distances.textContent = (
        `Distances: ${template.segment_distances.map(formatTemplateDistance).join(" -> ")}`
      );
      templateDetailsEl.appendChild(distances);
    }

    function insertInstructionSnippet(snippet) {
      const text = typeof snippet === "string" ? snippet.trim() : "";
      if (!text) {
        return;
      }
      const start = instructionInput.selectionStart ?? instructionInput.value.length;
      const end = instructionInput.selectionEnd ?? instructionInput.value.length;
      const before = instructionInput.value.slice(0, start);
      const after = instructionInput.value.slice(end);
      const needsLeadingSpace = before.length > 0 && !/[\\s(]$/.test(before);
      const needsTrailingSpace = after.length > 0 && !/^[\\s),.;!?]/.test(after);
      const insertion = `${needsLeadingSpace ? " " : ""}${text}${needsTrailingSpace ? " " : ""}`;
      instructionInput.value = `${before}${insertion}${after}`;
      const caret = before.length + insertion.length;
      instructionInput.focus();
      instructionInput.setSelectionRange(caret, caret);
      persistFormIntoActiveSample();
      renderSampleList();
      renderChecks();
    }

    function getSelectedConstraint() {
      return getActiveSample().hard_constraints.find(
        constraint => constraint.constraint_id === selectedConstraintId
      ) || null;
    }

    function orderedMustPassConstraints(sample = getActiveSample()) {
      return sample.hard_constraints
        .filter(constraint => constraint.kind === "must_pass")
        .sort((first, second) => first.order - second.order);
    }

    function scopeToInputValue(scope) {
      const normalized = cloneConstraintScope(scope);
      if (normalized.type !== "between_hard_constraints") {
        return "global";
      }
      return `${normalized.from_order}:${normalized.to_order}`;
    }

    function inputValueToScope(value) {
      if (value === "global") {
        return { type: "global", from_order: null, to_order: null };
      }
      const [fromOrder, toOrder] = value.split(":").map(Number);
      if (Number.isInteger(fromOrder) && Number.isInteger(toOrder) && toOrder > fromOrder) {
        return {
          type: "between_hard_constraints",
          from_order: fromOrder,
          to_order: toOrder,
        };
      }
      return { type: "global", from_order: null, to_order: null };
    }

    function populateScopeSelect(select, currentScope) {
      const previousValue = scopeToInputValue(currentScope);
      select.innerHTML = "";
      const globalOption = document.createElement("option");
      globalOption.value = "global";
      globalOption.textContent = "Global constraint";
      select.appendChild(globalOption);
      const ordered = orderedMustPassConstraints();
      for (let index = 0; index < ordered.length - 1; index += 1) {
        const fromOrder = ordered[index].order;
        const toOrder = ordered[index + 1].order;
        const option = document.createElement("option");
        option.value = `${fromOrder}:${toOrder}`;
        option.textContent = `Between hard #${fromOrder} -> #${toOrder}`;
        select.appendChild(option);
      }
      select.value = [...select.options].some(option => option.value === previousValue)
        ? previousValue
        : "global";
    }

    function scopeLabel(scope) {
      const normalized = cloneConstraintScope(scope);
      if (normalized.type === "between_hard_constraints") {
        return `between #${normalized.from_order}->#${normalized.to_order}`;
      }
      return "global";
    }

    function reorderMustPass(sourceId, targetId) {
      if (!sourceId || sourceId === targetId) {
        return;
      }
      const sample = getActiveSample();
      const ordered = sample.hard_constraints
        .filter(constraint => constraint.kind === "must_pass")
        .sort((first, second) => first.order - second.order);
      const sourceIndex = ordered.findIndex(
        constraint => constraint.constraint_id === sourceId
      );
      const targetIndex = ordered.findIndex(
        constraint => constraint.constraint_id === targetId
      );
      if (sourceIndex < 0 || targetIndex < 0) {
        return;
      }
      recordUndoSnapshot("reorder must-pass constraints");
      const [moved] = ordered.splice(sourceIndex, 1);
      ordered.splice(targetIndex, 0, moved);
      ordered.forEach((constraint, index) => {
        constraint.order = index + 1;
      });
      refreshScopedConstraintBounds(sample);
      sample.updated_at = nowIso();
      selectedConstraintId = sourceId;
      renderConstraintEditor();
      renderChecks();
      drawGrid();
    }

    function refreshScopedConstraintBounds(sample = getActiveSample()) {
      const orders = orderedMustPassConstraints(sample).map(constraint => constraint.order);
      for (const constraint of [
        ...sample.hard_constraints.filter(item => item.kind === "must_avoid"),
        ...sample.soft_constraints,
      ]) {
        const scope = cloneConstraintScope(constraint.scope);
        if (
          scope.type === "between_hard_constraints" &&
          (!orders.includes(scope.from_order) || !orders.includes(scope.to_order))
        ) {
          constraint.scope = { type: "global", from_order: null, to_order: null };
        }
      }
    }

    function allConstraintRefs(sample = getActiveSample()) {
      return [
        ...sample.hard_constraints.map(constraint => ({
          collection: "hard",
          id: constraint.constraint_id,
          kind: constraint.kind,
          constraint,
        })),
        ...sample.soft_constraints.map(constraint => ({
          collection: "soft",
          id: constraint.constraint_id,
          kind: "soft",
          constraint,
        })),
      ];
    }

    function isSequenceConstraint(ref) {
      if (ref.kind === "must_pass") {
        return true;
      }
      return cloneConstraintScope(ref.constraint.scope).type === "between_hard_constraints";
    }

    function constraintRefColorClass(ref) {
      if (ref.kind === "must_pass") {
        return "must-pass";
      }
      if (ref.kind === "must_avoid") {
        return "must-avoid";
      }
      return "soft";
    }

    function isTrajectoryMetricPreference(preferenceType) {
      return [
        "clearance",
        "move_smoothness",
        "path_shape_preference",
      ].includes(preferenceType);
    }

    function isRegionlessTrajectoryMetricPreference(preferenceType) {
      return [
        "clearance",
        "move_smoothness",
      ].includes(preferenceType);
    }

    function usesObjectReferenceRegion(preferenceType) {
      return [
        "near_preference",
        "far_preference",
      ].includes(preferenceType);
    }

    function trajectoryMetricLabel(preferenceType) {
      if (preferenceType === "clearance") {
        return "clearance";
      }
      if (preferenceType === "move_smoothness") {
        return "trajectory smoothness";
      }
      if (preferenceType === "path_shape_preference") {
        return "path shape";
      }
      return "trajectory metric";
    }

    function constraintRefTitle(ref) {
      const constraint = ref.constraint;
      if (ref.kind === "must_pass") {
        return `#${constraint.order} ${constraint.label || "Must Pass"}`;
      }
      const label = constraint.label || (
        ref.kind === "must_avoid" ? "Must Avoid" : "Soft Constraint"
      );
      const shape = isRegionlessTrajectoryMetricPreference(constraint.preference_type)
        ? trajectoryMetricLabel(constraint.preference_type)
        : constraint.initialized === false ? "choose drawing method" : constraint.shape;
      return `${label} · ${scopeLabel(constraint.scope)} · ${shape}`;
    }

    function selectConstraintRef(ref) {
      if (ref.collection === "hard") {
        selectedConstraintId = ref.id;
        selectedSoftConstraintId = null;
      } else {
        selectedSoftConstraintId = ref.id;
        selectedConstraintId = null;
      }
      referencePickTarget = null;
      renderConstraintEditor();
      renderSoftConstraintEditor();
      drawGrid();
    }

    function defaultSequenceScope() {
      const ordered = orderedMustPassConstraints();
      if (ordered.length >= 2) {
        return {
          type: "between_hard_constraints",
          from_order: ordered[0].order,
          to_order: ordered[1].order,
        };
      }
      setStatus("Need at least two must-pass hard constraints before moving avoid/soft bars into Sequence.");
      return { type: "global", from_order: null, to_order: null };
    }

    function moveConstraintRefToBucket(ref, bucket) {
      const sample = getActiveSample();
      recordUndoSnapshot(`move constraint to ${bucket}`);
      if (bucket === "global") {
        if (ref.kind === "must_pass") {
          setStatus("Must-pass bars stay in the sequence because they define route order.");
          return;
        }
        ref.constraint.scope = { type: "global", from_order: null, to_order: null };
      } else if (ref.kind !== "must_pass") {
        ref.constraint.scope = defaultSequenceScope();
      }
      sample.updated_at = nowIso();
      renderConstraintEditor();
      renderSoftConstraintEditor();
      renderChecks();
      drawGrid();
    }

    function renderUnifiedConstraintBar(ref) {
      const button = document.createElement("button");
      button.type = "button";
      button.draggable = true;
      button.className = (
        `constraint-item ${constraintRefColorClass(ref)}` +
        `${ref.id === selectedConstraintId || ref.id === selectedSoftConstraintId ? " active" : ""}`
      );
      button.textContent = `☰ ${constraintRefTitle(ref)}`;
      button.addEventListener("click", () => selectConstraintRef(ref));
      button.addEventListener("dragstart", () => {
        draggedConstraintRef = ref;
        draggedMustPassId = ref.kind === "must_pass" ? ref.id : null;
        button.classList.add("dragging");
      });
      button.addEventListener("dragend", () => {
        draggedConstraintRef = null;
        draggedMustPassId = null;
        button.classList.remove("dragging");
        document.querySelectorAll(".drop-target").forEach(
          element => element.classList.remove("drop-target")
        );
      });
      button.addEventListener("dragover", event => {
        event.preventDefault();
        button.classList.add("drop-target");
      });
      button.addEventListener("dragleave", () => {
        button.classList.remove("drop-target");
      });
      button.addEventListener("drop", event => {
        event.preventDefault();
        button.classList.remove("drop-target");
        if (draggedConstraintRef?.kind === "must_pass" && ref.kind === "must_pass") {
          reorderMustPass(draggedConstraintRef.id, ref.id);
        }
      });
      return button;
    }

    function renderUnifiedConstraintLists() {
      sequenceConstraintListEl.innerHTML = "";
      globalConstraintListEl.innerHTML = "";
      const refs = allConstraintRefs();
      const sequenceRefs = refs
        .filter(isSequenceConstraint)
        .sort((first, second) => {
          const firstScope = cloneConstraintScope(first.constraint.scope);
          const secondScope = cloneConstraintScope(second.constraint.scope);
          const firstOrder = first.kind === "must_pass" ? first.constraint.order : firstScope.from_order + 0.5;
          const secondOrder = second.kind === "must_pass" ? second.constraint.order : secondScope.from_order + 0.5;
          return Number(firstOrder || 0) - Number(secondOrder || 0);
        });
      const globalRefs = refs.filter(ref => !isSequenceConstraint(ref));

      if (sequenceRefs.length === 0) {
        const empty = document.createElement("div");
        empty.className = "item small";
        empty.textContent = "No sequence constraints yet.";
        sequenceConstraintListEl.appendChild(empty);
      }
      if (globalRefs.length === 0) {
        const empty = document.createElement("div");
        empty.className = "item small";
        empty.textContent = "No global constraints yet.";
        globalConstraintListEl.appendChild(empty);
      }
      for (const ref of sequenceRefs) {
        sequenceConstraintListEl.appendChild(renderUnifiedConstraintBar(ref));
      }
      for (const ref of globalRefs) {
        globalConstraintListEl.appendChild(renderUnifiedConstraintBar(ref));
      }
    }

    function renderConstraintEditor() {
      const sample = getActiveSample();
      renderUnifiedConstraintLists();
      mustPassListEl.innerHTML = "";
      mustAvoidListEl.innerHTML = "";
      const selected = getSelectedConstraint();
      hardEditorSection.style.display = selected ? "grid" : "none";
      const mustPassConstraints = orderedMustPassConstraints(sample);
      const mustAvoidConstraints = sample.hard_constraints.filter(
        constraint => constraint.kind === "must_avoid"
      );

      if (mustPassConstraints.length === 0) {
        const empty = document.createElement("div");
        empty.className = "item small";
        empty.textContent = "No must-pass regions.";
        mustPassListEl.appendChild(empty);
      }
      if (mustAvoidConstraints.length === 0) {
        const empty = document.createElement("div");
        empty.className = "item small";
        empty.textContent = "No must-avoid regions.";
        mustAvoidListEl.appendChild(empty);
      }

      for (const constraint of mustPassConstraints) {
        const button = document.createElement("button");
        button.type = "button";
        button.draggable = true;
        button.className = (
          "constraint-item must-pass" +
          `${constraint.constraint_id === selectedConstraintId ? " active" : ""}`
        );
        button.textContent = `☰ #${constraint.order}`;
        button.addEventListener("click", () => {
          selectedConstraintId = constraint.constraint_id;
          selectedSoftConstraintId = null;
          referencePickTarget = null;
          renderConstraintEditor();
          renderSoftConstraintEditor();
          drawGrid();
        });
        button.addEventListener("dragstart", () => {
          draggedMustPassId = constraint.constraint_id;
          button.classList.add("dragging");
        });
        button.addEventListener("dragend", () => {
          draggedMustPassId = null;
          button.classList.remove("dragging");
          document.querySelectorAll(".drop-target").forEach(
            element => element.classList.remove("drop-target")
          );
        });
        button.addEventListener("dragover", event => {
          event.preventDefault();
          button.classList.add("drop-target");
        });
        button.addEventListener("dragleave", () => {
          button.classList.remove("drop-target");
        });
        button.addEventListener("drop", event => {
          event.preventDefault();
          button.classList.remove("drop-target");
          reorderMustPass(draggedMustPassId, constraint.constraint_id);
        });
        mustPassListEl.appendChild(button);
      }

      for (const constraint of mustAvoidConstraints) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = (
          "constraint-item must-avoid" +
          `${constraint.constraint_id === selectedConstraintId ? " active" : ""}`
        );
        button.textContent = constraint.initialized === false
          ? `${constraint.label || "Must Avoid"} · ${scopeLabel(constraint.scope)} · choose drawing method`
          : `${constraint.label || "Must Avoid"} · ${scopeLabel(constraint.scope)} · ${constraint.shape}`;
        button.addEventListener("click", () => {
          selectedConstraintId = constraint.constraint_id;
          selectedSoftConstraintId = null;
          referencePickTarget = null;
          renderConstraintEditor();
          renderSoftConstraintEditor();
          drawGrid();
        });
        mustAvoidListEl.appendChild(button);
      }

      const disabled = selected === null;
      passCreationTools.style.display = (
        selected?.kind === "must_pass" ? "grid" : "none"
      );
      avoidCreationTools.style.display = (
        selected?.kind === "must_avoid" ? "grid" : "none"
      );
        constraintShapeInput.disabled = disabled;
        constraintRadiusInput.disabled = disabled;
        constraintWidthInput.disabled = disabled;
        constraintHeightInput.disabled = disabled;
        constraintLabelInput.disabled = disabled;
        constraintScopeInput.disabled = disabled;
      eraseHardFreeformBtn.disabled = disabled;
      eraseHardBoxBtn.disabled = disabled;
      deleteConstraintBtn.disabled = disabled;
      if (!selected) {
        constraintShapeInput.value = "circle";
        constraintRadiusInput.value = "";
        constraintWidthInput.value = "";
        constraintHeightInput.value = "";
        constraintLabelInput.value = "";
        populateScopeSelect(constraintScopeInput, null);
        constraintRadiusInput.style.display = "block";
        constraintWidthInput.parentElement.style.display = "none";
        constraintLabelInput.style.display = "block";
        constraintScopeInput.style.display = "block";
        eraseHardFreeformBtn.disabled = true;
        eraseHardBoxBtn.disabled = true;
        constraintOrderInfo.textContent = "Select a constraint to edit its shape and size.";
        return;
      }

      constraintShapeInput.value = selected.shape;
      constraintRadiusInput.value = String(selected.radius);
      constraintWidthInput.value = String(selected.width);
      constraintHeightInput.value = String(selected.height);
      constraintLabelInput.value = selected.label || "";
      populateScopeSelect(constraintScopeInput, selected.scope);
      constraintRadiusInput.style.display = selected.shape === "circle" ? "block" : "none";
      constraintWidthInput.parentElement.style.display = (
        selected.shape === "rectangle" ? "flex" : "none"
      );
      constraintLabelInput.style.display = "block";
      constraintScopeInput.style.display = selected.kind === "must_avoid" ? "block" : "none";
      eraseHardFreeformBtn.disabled = selected.initialized === false;
      eraseHardBoxBtn.disabled = selected.initialized === false;
      constraintOrderInfo.textContent = selected.kind === "must_pass"
        ? `Visit order: ${selected.order}. Drag this item in the Must Pass list to reorder it.${
            selected.initialized === false ? " Choose a drawing method above." : ""
          }`
        : "Must-avoid regions have no ordering requirement.";
    }

    function getSelectedSoftConstraint() {
      return getActiveSample().soft_constraints.find(
        constraint => constraint.constraint_id === selectedSoftConstraintId
      ) || null;
    }

    function populateRelativeObjectSelects() {
      const objectMap = objectInstanceMap();
      for (const select of [relativeObjectAInput, relativeObjectBInput]) {
        const previousValue = select.value;
        select.innerHTML = "";
        for (const [objectId, object] of objectMap.entries()) {
          const option = document.createElement("option");
          option.value = String(objectId);
          option.textContent = `${object.name} (${object.category}, id=${objectId})`;
          select.appendChild(option);
        }
        if ([...select.options].some(option => option.value === previousValue)) {
          select.value = previousValue;
        }
      }
    }

    function populateSoftReferenceObjectSelect() {
      const previousValue = softReferenceObjectInput.value;
      softReferenceObjectInput.innerHTML = "";
      for (const [objectId, object] of objectInstanceMap().entries()) {
        const option = document.createElement("option");
        option.value = String(objectId);
        option.textContent = `${object.name} (${object.category}, id=${objectId})`;
        softReferenceObjectInput.appendChild(option);
      }
      if (
        [...softReferenceObjectInput.options].some(
          option => option.value === previousValue
        )
      ) {
        softReferenceObjectInput.value = previousValue;
      }
    }

    function cellsForObject(objectId) {
      const footprintCells = objectCellsFromFootprint(objectId);
      if (footprintCells.length > 0) {
        return footprintCells;
      }
      const cells = [];
      const objectGrid = mapState.layers.object_instance;
      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          if (objectGrid[row][col] === objectId) {
            cells.push([row, col]);
          }
        }
      }
      if (cells.length === 0) {
        const centerCell = centerHintCellForObject(objectInstanceMap().get(objectId));
        if (centerCell) {
          cells.push(centerCell);
        }
      }
      return cells;
    }

    function mapGridResolutionMeters() {
      const resolution = Number(mapState.metadata?.map_info?.resolution);
      return Number.isFinite(resolution) && resolution > 0 ? resolution : 1;
    }

    function metersToGrid(meters) {
      return Number(meters) / mapGridResolutionMeters();
    }

    function templateObjectHardConstraintRadius(object) {
      const cells = cellsForObject(object.object_id);
      const paddingGrids = metersToGrid(TEMPLATE_HARD_CONSTRAINT_PADDING_METERS);
      if (cells.length === 0) {
        return Math.max(1, Math.ceil(paddingGrids));
      }
      const objectRadius = Math.max(
        ...cells.map(cell => Math.hypot(
          cell[0] - object.center[0],
          cell[1] - object.center[1]
        ))
      );
      return Math.max(
        1,
        Math.ceil(objectRadius + paddingGrids)
      );
    }

    function syncObjectReferenceRegion(constraint) {
      if (
        !constraint?.reference_region ||
        constraint.reference_region.mode !== "object"
      ) {
        return;
      }
      constraint.reference_region.cells = cellsForObject(
        constraint.reference_region.object_id
      );
    }

    function renderSoftConstraintEditor() {
      const sample = getActiveSample();
      renderUnifiedConstraintLists();
      softConstraintListEl.innerHTML = "";
      populateRelativeObjectSelects();
      populateSoftReferenceObjectSelect();
      const selected = getSelectedSoftConstraint();
      softEditorSection.style.display = selected ? "grid" : "none";
      if (sample.soft_constraints.length === 0) {
        const empty = document.createElement("div");
        empty.className = "item small";
        empty.textContent = "No soft constraints yet.";
        softConstraintListEl.appendChild(empty);
      }

      for (const constraint of sample.soft_constraints) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = (
          `constraint-item${constraint.constraint_id === selectedSoftConstraintId ? " active" : ""}`
        );
        button.style.borderLeft = "5px solid #3478b8";
        const softShapeLabel = isRegionlessTrajectoryMetricPreference(constraint.preference_type)
          ? trajectoryMetricLabel(constraint.preference_type)
          : constraint.shape;
        button.textContent = constraint.initialized === false
          ? `${constraint.label || constraint.preference_type.replaceAll("_", " ")} · ${scopeLabel(constraint.scope)} · choose drawing method`
          : `${constraint.label || constraint.preference_type.replaceAll("_", " ")} · ${scopeLabel(constraint.scope)} · ${softShapeLabel}`;
        button.addEventListener("click", () => {
          selectedSoftConstraintId = constraint.constraint_id;
          selectedConstraintId = null;
          referencePickTarget = null;
          renderConstraintEditor();
          renderSoftConstraintEditor();
          drawGrid();
        });
        softConstraintListEl.appendChild(button);
      }

      const disabled = selected === null;
      softCreationTools.style.display = selected ? "grid" : "none";
      for (const input of [
        softPreferenceTypeInput,
        softConstraintShapeInput,
        softConstraintRadiusInput,
        softConstraintWidthInput,
        softConstraintHeightInput,
        softConstraintLabelInput,
        softConstraintScopeInput,
        relativeObjectAInput,
        relativeObjectBInput,
        softReferenceObjectInput,
        pickSoftReferenceBtn,
        pickRelativeObjectABtn,
        pickRelativeObjectBBtn,
        eraseSoftFreeformBtn,
        eraseSoftBoxBtn,
        deleteSoftConstraintBtn,
      ]) {
        input.disabled = disabled;
      }
      if (!selected) {
        softPreferenceTypeInput.value = "near_preference";
        softConstraintShapeInput.value = "circle";
        softConstraintRadiusInput.value = "";
        softConstraintWidthInput.value = "";
        softConstraintHeightInput.value = "";
        softConstraintLabelInput.value = "";
        populateScopeSelect(softConstraintScopeInput, null);
        softPreferenceTypeInput.style.display = "block";
        softConstraintRadiusInput.style.display = "block";
        softConstraintWidthInput.parentElement.style.display = "none";
        relativeObjectInputs.style.display = "none";
        softReferenceRegionInputs.style.display = "none";
        softSmoothnessInfo.style.display = "none";
        softReferenceRegionInfo.textContent = "Reference region: not set.";
        eraseSoftFreeformBtn.style.display = "block";
        eraseSoftBoxBtn.style.display = "block";
        eraseSoftFreeformBtn.disabled = true;
        eraseSoftBoxBtn.disabled = true;
        return;
      }

      softPreferenceTypeInput.value = selected.preference_type;
      softConstraintShapeInput.value = selected.shape;
      softConstraintRadiusInput.value = String(selected.radius);
      softConstraintWidthInput.value = String(selected.width);
      softConstraintHeightInput.value = String(selected.height);
      softConstraintLabelInput.value = selected.label || "";
      populateScopeSelect(softConstraintScopeInput, selected.scope);
      const isRegionlessTrajectoryMetric = isRegionlessTrajectoryMetricPreference(
        selected.preference_type
      );
      const usesReferenceRegion = usesObjectReferenceRegion(selected.preference_type);
      softCreationTools.style.display = isRegionlessTrajectoryMetric ? "none" : "grid";
      softPreferenceTypeInput.style.display = isRegionlessTrajectoryMetric ? "none" : "block";
      softConstraintShapeInput.style.display = isRegionlessTrajectoryMetric ? "none" : "block";
      softConstraintRadiusInput.style.display = (
        !isRegionlessTrajectoryMetric && selected.shape === "circle" ? "block" : "none"
      );
      softConstraintWidthInput.parentElement.style.display = (
        !isRegionlessTrajectoryMetric && selected.shape === "rectangle" ? "flex" : "none"
      );
      relativeObjectInputs.style.display = (
        selected.preference_type === "relative_preference" ? "flex" : "none"
      );
      softReferenceRegionInputs.style.display = usesReferenceRegion ? "flex" : "none";
      softSmoothnessInfo.style.display = (
        isRegionlessTrajectoryMetric ||
        selected.preference_type === "path_shape_preference"
      ) ? "block" : "none";
      if (selected.preference_type === "move_smoothness") {
        const metricCost = Number.isFinite(selected.C_smooth_ref)
          ? selected.C_smooth_ref.toFixed(4)
          : "not computed yet";
        softSmoothnessInfo.textContent = (
          `Move smoothness uses the scoped trajectory only. C_smooth_ref=${metricCost}.`
        );
      } else if (selected.preference_type === "clearance") {
        const metricCost = Number.isFinite(selected.C_clear_ref)
          ? selected.C_clear_ref.toFixed(4)
          : "not computed yet";
        softSmoothnessInfo.textContent = (
          `Clearance uses the scoped trajectory and map obstacles only. C_clear_ref=${metricCost}.`
        );
      } else if (selected.preference_type === "path_shape_preference") {
        const waypointCount = Number(selected.reference_waypoint_count || 0);
        softSmoothnessInfo.textContent = (
          `Path shape stores the expert trajectory inside this soft region for DTW comparison. reference_waypoint_count=${waypointCount}.`
        );
      }
      eraseSoftFreeformBtn.style.display = isRegionlessTrajectoryMetric ? "none" : "block";
      eraseSoftBoxBtn.style.display = isRegionlessTrajectoryMetric ? "none" : "block";
      if (
        usesReferenceRegion &&
        (!selected.reference_region || selected.reference_region.mode !== "object")
      ) {
        const firstObjectId = [...objectInstanceMap().keys()][0] ?? null;
        selected.reference_region = {
          mode: "object",
          object_id: firstObjectId,
          cells: firstObjectId === null ? [] : cellsForObject(firstObjectId),
        };
      }
      if (usesReferenceRegion) {
        syncObjectReferenceRegion(selected);
        const referenceRegion = selected.reference_region;
        softReferenceObjectLabel.style.display = "block";
        if (
          referenceRegion.object_id !== null &&
          [...softReferenceObjectInput.options].some(
            option => Number(option.value) === referenceRegion.object_id
          )
        ) {
          softReferenceObjectInput.value = String(referenceRegion.object_id);
        }
        softReferenceRegionInfo.textContent = (
          `Reference object: ${referenceRegion.object_id ?? "not selected"}, ` +
          `${referenceRegion.cells.length} cells.`
        );
      }
      eraseSoftFreeformBtn.disabled = selected.initialized === false;
      eraseSoftBoxBtn.disabled = selected.initialized === false;
      if (selected.object_ids.length === 2) {
        selected.reference_regions = selected.object_ids.map(objectId => ({
          object_id: objectId,
          cells: cellsForObject(objectId),
        }));
        relativeObjectAInput.value = String(selected.object_ids[0]);
        relativeObjectBInput.value = String(selected.object_ids[1]);
      }
    }

    function renderStartPoseInfo() {
      const pose = getActiveSample().start_pose;
      if (pose.row === null || pose.col === null) {
        startPoseInfo.textContent = "No expert route yet.";
        return;
      }
      startPoseInfo.textContent = `first route cell: row=${pose.row}, col=${pose.col}`;
    }

    function syncStartPoseFromRoute(sample = getActiveSample(), updateTimestamp = false) {
      const firstPoint = sample.expert_route[0] || null;
      const nextPose = firstPoint
        ? { row: firstPoint[0], col: firstPoint[1] }
        : { row: null, col: null };
      const changed = (
        sample.start_pose.row !== nextPose.row ||
        sample.start_pose.col !== nextPose.col
      );
      sample.start_pose = nextPose;
      if (changed && updateTimestamp) {
        sample.updated_at = nowIso();
      }
      return changed;
    }

    function computePathLength(trajectory) {
      let length = 0;
      for (let index = 1; index < trajectory.length; index += 1) {
        const [previousRow, previousCol] = trajectory[index - 1];
        const [row, col] = trajectory[index];
        length += Math.hypot(row - previousRow, col - previousCol);
      }
      return length;
    }

    function routePointInsideConstraint(point, constraint) {
      const row = point[0];
      const col = point[1];
      const dRow = row - constraint.center[0];
      const dCol = col - constraint.center[1];
      if (constraint.shape === "freeform") {
        return constraint.cells.some(cell => cell[0] === row && cell[1] === col);
      }
      if (constraint.shape === "circle") {
        return Math.hypot(dRow, dCol) <= constraint.radius;
      }
      return (
        Math.abs(dRow) <= constraint.height / 2 &&
        Math.abs(dCol) <= constraint.width / 2
      );
    }

    function coveredCellsForConstraint(constraint) {
      if (constraint.shape === "freeform") {
        return constraint.cells.map(cell => [cell[0], cell[1]]);
      }
      const cells = [];
      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          if (routePointInsideConstraint([row, col], constraint)) {
            cells.push([row, col]);
          }
        }
      }
      return cells;
    }

    function trajectoryInsideConstraintRegion(trajectory, constraint) {
      return trajectory
        .filter(point => routePointInsideConstraint(point, constraint))
        .map(point => [point[0], point[1]]);
    }

    function meanUniqueWaypointDistance(trajectory, constraint, referencePoints) {
      if (!Array.isArray(referencePoints) || referencePoints.length === 0) {
        return { distance: null, waypointCount: 0 };
      }
      const seen = new Set();
      const regionPoints = [];
      for (const point of trajectory) {
        const key = cellKey(point[0], point[1]);
        if (seen.has(key)) {
          continue;
        }
        seen.add(key);
        if (routePointInsideConstraint(point, constraint)) {
          regionPoints.push(point);
        }
      }
      if (regionPoints.length === 0) {
        return { distance: null, waypointCount: 0 };
      }
      const totalDistance = regionPoints.reduce((sum, point) => {
        const nearestDistance = Math.min(...referencePoints.map(referencePoint =>
          Math.hypot(
            point[0] - referencePoint[0],
            point[1] - referencePoint[1]
          )
        ));
        return sum + nearestDistance;
      }, 0);
      return {
        distance: totalDistance / regionPoints.length,
        waypointCount: regionPoints.length,
      };
    }

    function segmentRectangleInterval(start, end, rowMin, rowMax, colMin, colMax) {
      let tMin = 0;
      let tMax = 1;
      const axes = [
        [start[0], end[0] - start[0], rowMin, rowMax],
        [start[1], end[1] - start[1], colMin, colMax],
      ];
      for (const [startValue, delta, low, high] of axes) {
        if (Math.abs(delta) <= METRIC_EPSILON) {
          if (startValue < low || startValue > high) {
            return null;
          }
          continue;
        }
        const first = (low - startValue) / delta;
        const second = (high - startValue) / delta;
        const enter = Math.min(first, second);
        const exit = Math.max(first, second);
        tMin = Math.max(tMin, enter);
        tMax = Math.min(tMax, exit);
        if (tMin - tMax > METRIC_EPSILON) {
          return null;
        }
      }
      return [Math.max(0, tMin), Math.min(1, tMax)];
    }

    function segmentCircleInterval(start, end, center, radius) {
      const deltaRow = end[0] - start[0];
      const deltaCol = end[1] - start[1];
      const startRow = start[0] - center[0];
      const startCol = start[1] - center[1];
      const a = deltaRow * deltaRow + deltaCol * deltaCol;
      const c = startRow * startRow + startCol * startCol - radius * radius;
      if (a <= METRIC_EPSILON) {
        return c <= METRIC_EPSILON ? [0, 1] : null;
      }
      const b = 2 * (startRow * deltaRow + startCol * deltaCol);
      let discriminant = b * b - 4 * a * c;
      if (discriminant < -METRIC_EPSILON) {
        return null;
      }
      if (discriminant < 0) {
        discriminant = 0;
      }
      const root = Math.sqrt(discriminant);
      const first = (-b - root) / (2 * a);
      const second = (-b + root) / (2 * a);
      const enter = Math.max(0, Math.min(first, second));
      const exit = Math.min(1, Math.max(first, second));
      if (enter - exit > METRIC_EPSILON || exit < -METRIC_EPSILON || enter > 1 + METRIC_EPSILON) {
        return null;
      }
      return [enter, exit];
    }

    function mergeSegmentIntervals(intervals) {
      const sortedIntervals = intervals
        .filter(interval => interval && interval[1] >= interval[0] - METRIC_EPSILON)
        .map(interval => [Math.max(0, interval[0]), Math.min(1, interval[1])])
        .sort((first, second) => first[0] - second[0]);
      const merged = [];
      for (const [start, end] of sortedIntervals) {
        if (merged.length === 0 || start > merged[merged.length - 1][1] + METRIC_EPSILON) {
          merged.push([start, end]);
          continue;
        }
        merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], end);
      }
      return merged;
    }

    function segmentConstraintIntervals(start, end, constraint) {
      if (constraint.shape === "freeform") {
        const cells = Array.isArray(constraint.cells) ? constraint.cells : [];
        const intervals = [];
        for (const cell of cells) {
          intervals.push(segmentRectangleInterval(
            start,
            end,
            cell[0] - 0.5,
            cell[0] + 0.5,
            cell[1] - 0.5,
            cell[1] + 0.5
          ));
        }
        return mergeSegmentIntervals(intervals);
      }
      if (constraint.shape === "circle") {
        const interval = segmentCircleInterval(start, end, constraint.center, Number(constraint.radius));
        return interval ? [interval] : [];
      }
      if (constraint.shape === "rectangle") {
        const interval = segmentRectangleInterval(
          start,
          end,
          constraint.center[0] - Number(constraint.height) / 2,
          constraint.center[0] + Number(constraint.height) / 2,
          constraint.center[1] - Number(constraint.width) / 2,
          constraint.center[1] + Number(constraint.width) / 2
        );
        return interval ? [interval] : [];
      }
      return [];
    }

    function relativeMarginForPoint(point, preferredPoints, referencePoints) {
      const preferredDistance = Math.min(...preferredPoints.map(referencePoint =>
        Math.hypot(point[0] - referencePoint[0], point[1] - referencePoint[1])
      ));
      const referenceDistance = Math.min(...referencePoints.map(referencePoint =>
        Math.hypot(point[0] - referencePoint[0], point[1] - referencePoint[1])
      ));
      return (referenceDistance - preferredDistance) / (
        preferredDistance + referenceDistance + METRIC_EPSILON
      );
    }

    function computeRelativePreferenceQuality(trajectory, constraint) {
      const referenceRegions = Array.isArray(constraint.reference_regions)
        ? constraint.reference_regions
        : [];
      const preferredCells = Array.isArray(referenceRegions[0]?.cells)
        ? referenceRegions[0].cells
        : [];
      const referenceCells = Array.isArray(referenceRegions[1]?.cells)
        ? referenceRegions[1].cells
        : [];
      if (preferredCells.length === 0 || referenceCells.length === 0) {
        return { quality: null, incomplete: true, validPathLength: 0, validSegmentCount: 0 };
      }

      let weightedMarginSum = 0;
      let validPathLength = 0;
      let validSegmentCount = 0;
      let positiveSegmentCount = 0;
      for (let index = 1; index < trajectory.length; index += 1) {
        const start = trajectory[index - 1];
        const end = trajectory[index];
        const segmentLength = Math.hypot(end[0] - start[0], end[1] - start[1]);
        if (segmentLength <= METRIC_EPSILON) {
          continue;
        }
        for (const [enterT, exitT] of segmentConstraintIntervals(start, end, constraint)) {
          if (exitT - enterT <= METRIC_EPSILON) {
            continue;
          }
          const midpointT = (enterT + exitT) / 2;
          const midpoint = [
            start[0] + (end[0] - start[0]) * midpointT,
            start[1] + (end[1] - start[1]) * midpointT,
          ];
          const activeLength = segmentLength * (exitT - enterT);
          const margin = relativeMarginForPoint(midpoint, preferredCells, referenceCells);
          weightedMarginSum += activeLength * margin;
          validPathLength += activeLength;
          validSegmentCount += 1;
          if (margin > 0) {
            positiveSegmentCount += 1;
          }
        }
      }
      if (validPathLength < RELATIVE_PREFERENCE_MIN_VALID_LENGTH_GRID) {
        return {
          quality: null,
          incomplete: false,
          validPathLength,
          validSegmentCount,
          positiveSegmentCount,
        };
      }
      return {
        quality: weightedMarginSum / validPathLength,
        incomplete: false,
        validPathLength,
        validSegmentCount,
        positiveSegmentCount,
      };
    }

    function normalizeRelativePreferenceQuality(predictedQuality, humanQuality) {
      const zeroThreshold = humanQuality - Math.max(
        RELATIVE_PREFERENCE_HUMAN_BETA * Math.abs(humanQuality),
        RELATIVE_PREFERENCE_MIN_MARGIN
      );
      const denominator = humanQuality - zeroThreshold;
      if (denominator <= METRIC_EPSILON) {
        return { score: 0, zeroThreshold };
      }
      const score = (predictedQuality - zeroThreshold) / denominator;
      return { score: Math.max(0, Math.min(1, score)), zeroThreshold };
    }

    function computeRelativePreferenceScore(trajectory, constraint, options = {}) {
      const quality = computeRelativePreferenceQuality(trajectory, constraint);
      if (quality.quality === null) {
        return { score: null, ...quality };
      }
      const storedReference = Number(constraint.Q_relative_ref);
      const humanQuality = options.refreshHumanReference || !Number.isFinite(storedReference)
        ? quality.quality
        : storedReference;
      const normalized = normalizeRelativePreferenceQuality(quality.quality, humanQuality);
      return {
        score: normalized.score,
        quality: quality.quality,
        humanQuality,
        zeroThreshold: normalized.zeroThreshold,
        incomplete: false,
        validPathLength: quality.validPathLength,
        validSegmentCount: quality.validSegmentCount,
        positiveSegmentCount: quality.positiveSegmentCount,
      };
    }

    function meanSquaredTurningAngle(trajectory) {
      if (!Array.isArray(trajectory) || trajectory.length < 3) {
        return { cost: 0, waypointCount: 0 };
      }
      const squaredAngles = [];
      for (let index = 1; index < trajectory.length - 1; index += 1) {
        const previous = trajectory[index - 1];
        const current = trajectory[index];
        const following = trajectory[index + 1];
        const firstVector = [
          current[0] - previous[0],
          current[1] - previous[1],
        ];
        const secondVector = [
          following[0] - current[0],
          following[1] - current[1],
        ];
        const firstNorm = Math.hypot(firstVector[0], firstVector[1]);
        const secondNorm = Math.hypot(secondVector[0], secondVector[1]);
        if (firstNorm === 0 || secondNorm === 0) {
          continue;
        }
        const cosine = Math.max(
          -1,
          Math.min(
            1,
            (firstVector[0] * secondVector[0] + firstVector[1] * secondVector[1]) /
              (firstNorm * secondNorm)
          )
        );
        const angle = Math.acos(cosine);
        squaredAngles.push(angle * angle);
      }
      if (squaredAngles.length === 0) {
        return { cost: 0, waypointCount: 0 };
      }
      return {
        cost: squaredAngles.reduce((sum, value) => sum + value, 0) / squaredAngles.length,
        waypointCount: squaredAngles.length,
      };
    }

    function clearanceObstacleCells() {
      const obstacles = [];
      const occupancy = mapState.layers.occupancy;
      const objectGrid = mapState.layers.object_instance;
      const objectMap = objectInstanceMap();
      const freeValue = currentLegends.occupancy.free.value;
      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          const objectId = objectGrid[row][col];
          const objectItem = objectId === 0 ? null : objectMap.get(objectId);
          const objectIsObstacle = (
            objectId !== 0 &&
            !TRAVERSABLE_OBJECT_CATEGORIES.has(objectItem?.category)
          );
          if (occupancy[row][col] !== freeValue || objectIsObstacle) {
            obstacles.push([row, col, objectId]);
          }
        }
      }
      return obstacles;
    }

    function nearPreferenceExcludedObjectIds(point, softConstraints) {
      const excluded = new Set();
      for (const constraint of softConstraints) {
        if (constraint.preference_type !== "near_preference") {
          continue;
        }
        const referenceRegion = constraint.reference_region;
        if (
          referenceRegion?.mode === "object" &&
          Number.isInteger(referenceRegion.object_id) &&
          routePointInsideConstraint(point, constraint)
        ) {
          excluded.add(referenceRegion.object_id);
        }
      }
      return excluded;
    }

    function computeClearanceCost(trajectory, softConstraints) {
      if (!Array.isArray(trajectory) || trajectory.length === 0) {
        return { cost: 0, waypointCount: 0 };
      }
      const obstacles = clearanceObstacleCells();
      if (obstacles.length === 0) {
        return { cost: 0, waypointCount: trajectory.length };
      }
      const epsilon = 1e-6;
      let totalCost = 0;
      for (const point of trajectory) {
        const excludedIds = nearPreferenceExcludedObjectIds(point, softConstraints);
        let minDistance = Number.POSITIVE_INFINITY;
        for (const [obstacleRow, obstacleCol, objectId] of obstacles) {
          if (excludedIds.has(objectId)) {
            continue;
          }
          const distance = Math.hypot(point[0] - obstacleRow, point[1] - obstacleCol);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        if (Number.isFinite(minDistance)) {
          totalCost += 1 / ((minDistance + epsilon) ** 2);
        }
      }
      return {
        cost: totalCost / trajectory.length,
        waypointCount: trajectory.length,
      };
    }

    function firstHitIndex(trajectory, constraint, startIndex = 0) {
      for (let index = startIndex; index < trajectory.length; index += 1) {
        if (routePointInsideConstraint(trajectory[index], constraint)) {
          return index;
        }
      }
      return -1;
    }

    function hardConstraintHitIndexes(sample, orderedHardConstraints) {
      const hitIndexes = new Map();
      let startIndex = 0;
      for (const constraint of orderedHardConstraints) {
        const index = firstHitIndex(sample.expert_route, constraint, startIndex);
        if (index >= 0) {
          hitIndexes.set(Number(constraint.order), index);
          startIndex = index + 1;
        }
      }
      return hitIndexes;
    }

    function scopedTrajectory(sample, constraint, hardHitIndexes) {
      const scope = cloneConstraintScope(constraint.scope);
      if (scope.type !== "between_hard_constraints") {
        return sample.expert_route;
      }
      const fromIndex = hardHitIndexes.get(scope.from_order);
      const toIndex = hardHitIndexes.get(scope.to_order);
      if (Number.isInteger(fromIndex) && Number.isInteger(toIndex) && toIndex > fromIndex) {
        return sample.expert_route.slice(fromIndex, toIndex + 1);
      }
      if (Number.isInteger(fromIndex)) {
        return sample.expert_route.slice(fromIndex);
      }
      if (Number.isInteger(toIndex)) {
        return sample.expert_route.slice(0, toIndex + 1);
      }
      return sample.expert_route;
    }

    function routeCollisionViolations(trajectory) {
      return trajectory.filter(([row, col]) => {
        if (
          row < 0 ||
          row >= GRID_SIZE ||
          col < 0 ||
          col >= GRID_SIZE
        ) {
          return true;
        }
        return !isTraversableCell(row, col);
      });
    }

    function computeEvaluationMetrics(sample) {
      const allMustPass = sample.hard_constraints
        .filter(
          constraint => constraint.kind === "must_pass" && constraint.initialized !== false
        )
        .sort((first, second) => first.order - second.order);
      const scoredMustPass = allMustPass.slice(1);
      const hardHitIndexes = hardConstraintHitIndexes(sample, allMustPass);
      const collisionViolations = routeCollisionViolations(sample.expert_route);
      let matchedMustPass = 0;
      let hardFailed = false;
      const recordOrder = allMustPass.length > 0 ? Number(allMustPass[0].order) : null;
      let previousHitIndex = Number.isInteger(recordOrder) && hardHitIndexes.has(recordOrder)
        ? hardHitIndexes.get(recordOrder)
        : 0;
      let previousOrder = Number.isInteger(recordOrder) ? recordOrder : null;
      for (const constraint of scoredMustPass) {
        if (hardFailed) {
          break;
        }
        const currentOrder = Number(constraint.order);
        const currentHitIndex = hardHitIndexes.get(currentOrder);
        const transitionEndIndex = Number.isInteger(currentHitIndex)
          ? currentHitIndex
          : sample.expert_route.length - 1;
        const transitionTrajectory = sample.expert_route.slice(
          previousHitIndex,
          transitionEndIndex + 1
        );
        const violatesActiveAvoid = sample.hard_constraints
          .filter(item => item.kind === "must_avoid" && item.initialized !== false)
          .filter(item => {
            const scope = cloneConstraintScope(item.scope);
            if (scope.type !== "between_hard_constraints") {
              return true;
            }
            const toMatches = scope.to_order === currentOrder;
            const fromMatches = previousOrder === null
              ? scope.from_order === null
              : scope.from_order === previousOrder;
            return toMatches && fromMatches;
          })
          .some(item => transitionTrajectory.some(
            point => routePointInsideConstraint(point, item)
          ));
        if (violatesActiveAvoid) {
          hardFailed = true;
          break;
        }
        if (!Number.isInteger(currentHitIndex)) {
          break;
        }
        matchedMustPass += 1;
        previousHitIndex = currentHitIndex;
        previousOrder = currentOrder;
      }
      const hcs = scoredMustPass.length === 0
        ? null
        : matchedMustPass / scoredMustPass.length;
      const softScores = [];
      let softRulesComplete = true;
      for (const constraint of sample.soft_constraints) {
        if (constraint.initialized === false) {
          continue;
        }
        const activeTrajectory = scopedTrajectory(sample, constraint, hardHitIndexes);
        if (constraint.preference_type === "clearance") {
          const result = computeClearanceCost(activeTrajectory, sample.soft_constraints);
          constraint.C_clear_ref = result.cost;
          constraint.C_smooth_ref = null;
          constraint.D_ref = null;
          constraint.reference_trajectory = [];
          constraint.reference_waypoint_count = result.waypointCount;
          softScores.push(1);
          continue;
        }
        if (constraint.preference_type === "move_smoothness") {
          const result = meanSquaredTurningAngle(activeTrajectory);
          constraint.C_smooth_ref = result.cost;
          constraint.C_clear_ref = null;
          constraint.D_ref = null;
          constraint.reference_trajectory = [];
          constraint.reference_waypoint_count = result.waypointCount;
          softScores.push(1);
          continue;
        }
        if (constraint.preference_type === "path_shape_preference") {
          const embeddedTrajectory =
            constraint.path_shape_annotation?.shape_reference_trajectory;
          const regionTrajectory = Array.isArray(embeddedTrajectory)
            ? embeddedTrajectory.map(point => [point[0], point[1]])
            : trajectoryInsideConstraintRegion(activeTrajectory, constraint);
          constraint.reference_trajectory = regionTrajectory;
          constraint.C_smooth_ref = null;
          constraint.C_clear_ref = null;
          constraint.D_ref = null;
          constraint.reference_waypoint_count = regionTrajectory.length;
          softScores.push(1);
          continue;
        }
        if (constraint.preference_type === "relative_preference") {
          const result = computeRelativePreferenceScore(
            activeTrajectory,
            constraint,
            { refreshHumanReference: true }
          );
          if (result.incomplete) {
            softRulesComplete = false;
            break;
          }
          constraint.Q_relative_ref = result.quality;
          constraint.D_ref = null;
          constraint.C_smooth_ref = null;
          constraint.C_clear_ref = null;
          constraint.reference_trajectory = [];
          constraint.reference_waypoint_count = result.validSegmentCount || 0;
          constraint.reference_valid_path_length = result.validPathLength || 0;
          constraint.relative_reference_status = result.quality === null
            ? "invalid_human_reference"
            : "computed";
          constraint.reference_metric_status = constraint.relative_reference_status;
          if (result.score === null) {
            continue;
          }
          softScores.push(result.score);
          continue;
        }
        if (!["near_preference", "far_preference"].includes(
          constraint.preference_type
        )) {
          softRulesComplete = false;
          break;
        }
        const result = meanUniqueWaypointDistance(
          activeTrajectory,
          constraint,
          constraint.reference_region?.cells || []
        );
        constraint.D_ref = result.distance;
        constraint.reference_trajectory = [];
	        constraint.reference_waypoint_count = result.waypointCount;
	        if (result.distance === null) {
	          continue;
	        }
	        softScores.push(1);
	      }
	      const scs = softScores.length > 0 && softRulesComplete
	          ? softScores.reduce((sum, score) => sum + score, 0) / softScores.length
	          : null;

      return {
        HCS: hcs,
        collisionViolations,
        SCS: scs,
        PL: computePathLength(sample.expert_route),
      };
    }

	    function renderChecks() {
	      const metrics = computeEvaluationMetrics(getActiveSample());
	      hcsScoreEl.textContent = metrics.HCS === null
        ? "N/A"
        : metrics.HCS.toFixed(3);
      hcsScoreEl.title = metrics.collisionViolations.length === 0
        ? "HCS is ordered hard-constraint progress under must-avoid checks."
        : `${metrics.collisionViolations.length} route cell(s) are not traversable.`;
      scsScoreEl.textContent = metrics.SCS === null
        ? "N/A"
        : metrics.SCS.toFixed(3);
      scsScoreEl.title = metrics.SCS === null
        ? "Soft-preference scoring is pending."
        : "Annotation preview uses the expert trajectory with current reference metrics.";
	      pathLengthScoreEl.textContent = metrics.PL.toFixed(3);
	    }

	    function scheduleDrawGrid() {
	      if (drawGridFrame !== null) {
	        return;
	      }
	      drawGridFrame = window.requestAnimationFrame(() => {
	        drawGridFrame = null;
	        drawGrid();
	      });
	    }

	    function flushDrawGrid() {
	      if (drawGridFrame !== null) {
	        window.cancelAnimationFrame(drawGridFrame);
	        drawGridFrame = null;
	      }
	      drawGrid();
	    }

    function resizeCanvas() {
	      const size = GRID_SIZE * zoom;
	      canvas.width = size;
      canvas.height = size;
      zoomValue.textContent = `${zoom}x`;
      drawGrid();
    }

    function makeEmptyBounds() {
      return {
        minRow: Number.POSITIVE_INFINITY,
        minCol: Number.POSITIVE_INFINITY,
        maxRow: Number.NEGATIVE_INFINITY,
        maxCol: Number.NEGATIVE_INFINITY,
      };
    }

    function expandBoundsWithPoint(bounds, point, padding = 0) {
      if (!Array.isArray(point) || point.length !== 2) {
        return;
      }
      const row = Number(point[0]);
      const col = Number(point[1]);
      if (!Number.isFinite(row) || !Number.isFinite(col)) {
        return;
      }
      bounds.minRow = Math.min(bounds.minRow, row - padding);
      bounds.maxRow = Math.max(bounds.maxRow, row + padding);
      bounds.minCol = Math.min(bounds.minCol, col - padding);
      bounds.maxCol = Math.max(bounds.maxCol, col + padding);
    }

    function expandBoundsWithCells(bounds, cells) {
      if (!Array.isArray(cells)) {
        return;
      }
      for (const cell of cells) {
        expandBoundsWithPoint(bounds, cell);
      }
    }

    function expandBoundsWithConstraint(bounds, constraint) {
      if (!constraint || constraint.initialized === false) {
        return;
      }
      if (isRegionlessTrajectoryMetricPreference(constraint.preference_type)) {
        return;
      }
      if (constraint.shape === "freeform") {
        expandBoundsWithCells(bounds, constraint.cells);
      } else if (constraint.shape === "circle") {
        expandBoundsWithPoint(bounds, constraint.center, Number(constraint.radius || 0));
      } else if (constraint.shape === "rectangle") {
        expandBoundsWithPoint(
          bounds,
          constraint.center,
          Math.max(Number(constraint.width || 0), Number(constraint.height || 0)) / 2
        );
      } else {
        expandBoundsWithPoint(bounds, constraint.center);
      }
      if (constraint.preference_type === "relative_preference") {
        const referenceRegions = Array.isArray(constraint.reference_regions)
          ? constraint.reference_regions
          : [];
        for (const region of referenceRegions) {
          expandBoundsWithCells(bounds, region.cells);
        }
      } else if (constraint.reference_region) {
        expandBoundsWithCells(bounds, constraint.reference_region.cells);
      }
    }

    function activeAnnotationBounds() {
      const sample = getActiveSample();
      const bounds = makeEmptyBounds();
      expandBoundsWithCells(bounds, sample.expert_route);
      for (const constraint of sample.hard_constraints) {
        expandBoundsWithConstraint(bounds, constraint);
      }
      for (const constraint of sample.soft_constraints) {
        expandBoundsWithConstraint(bounds, constraint);
      }
      if (!Number.isFinite(bounds.minRow)) {
        return null;
      }
      return {
        minRow: Math.max(0, Math.floor(bounds.minRow)),
        minCol: Math.max(0, Math.floor(bounds.minCol)),
        maxRow: Math.min(GRID_SIZE - 1, Math.ceil(bounds.maxRow)),
        maxCol: Math.min(GRID_SIZE - 1, Math.ceil(bounds.maxCol)),
      };
    }

    function focusActiveAnnotation(options = {}) {
      const bounds = activeAnnotationBounds();
      if (!bounds) {
        if (options.report) {
          setStatus("No trajectory or constraint region is available to focus.");
        }
        return;
      }
      const centerRow = (bounds.minRow + bounds.maxRow + 1) / 2;
      const centerCol = (bounds.minCol + bounds.maxCol + 1) / 2;
      canvasBox.scrollTo({
        left: Math.max(0, centerCol * zoom - canvasBox.clientWidth / 2),
        top: Math.max(0, centerRow * zoom - canvasBox.clientHeight / 2),
        behavior: options.smooth ? "smooth" : "auto",
      });
      if (options.report) {
        setStatus(
          `Focused annotation rows ${bounds.minRow}-${bounds.maxRow}, ` +
          `cols ${bounds.minCol}-${bounds.maxCol}.`
        );
      }
    }

    function focusActiveAnnotationSoon() {
      window.requestAnimationFrame(() => focusActiveAnnotation());
    }

    function drawConstraintRegion(constraint, fillColor, strokeColor, selected) {
      ctx.save();
      ctx.fillStyle = fillColor;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = selected ? 4 : 2;
      if (constraint.shape === "freeform") {
        const cells = Array.isArray(constraint.cells) ? constraint.cells : [];
        if (cells.length === 0) {
          ctx.restore();
          return;
        }
        const cellSet = new Set(
          cells.map(cell => cellKey(cell[0], cell[1]))
        );

        for (const [row, col] of cells) {
          ctx.fillRect(
            col * zoom + 1,
            row * zoom + 1,
            Math.max(1, zoom - 2),
            Math.max(1, zoom - 2)
          );
        }

        const drawBoundary = (color, width) => {
          ctx.strokeStyle = color;
          ctx.lineWidth = width;
          ctx.lineJoin = "round";
          ctx.lineCap = "round";
          ctx.beginPath();
          for (const [row, col] of cells) {
            const left = col * zoom + 0.5;
            const right = (col + 1) * zoom + 0.5;
            const top = row * zoom + 0.5;
            const bottom = (row + 1) * zoom + 0.5;
            if (!cellSet.has(cellKey(row - 1, col))) {
              ctx.moveTo(left, top);
              ctx.lineTo(right, top);
            }
            if (!cellSet.has(cellKey(row + 1, col))) {
              ctx.moveTo(left, bottom);
              ctx.lineTo(right, bottom);
            }
            if (!cellSet.has(cellKey(row, col - 1))) {
              ctx.moveTo(left, top);
              ctx.lineTo(left, bottom);
            }
            if (!cellSet.has(cellKey(row, col + 1))) {
              ctx.moveTo(right, top);
              ctx.lineTo(right, bottom);
            }
          }
          ctx.stroke();
        };

        drawBoundary(
          selected ? "rgba(255, 255, 255, 0.92)" : "rgba(255, 255, 255, 0.65)",
          selected ? Math.max(5, zoom * 0.55) : Math.max(3, zoom * 0.35)
        );
        drawBoundary(
          strokeColor,
          selected ? Math.max(3, zoom * 0.35) : Math.max(2, zoom * 0.22)
        );
        ctx.restore();
        return;
      }

      if (constraint.shape !== "circle" && constraint.shape !== "rectangle") {
        ctx.restore();
        return;
      }

      const centerX = (constraint.center[1] + 0.5) * zoom;
      const centerY = (constraint.center[0] + 0.5) * zoom;
      ctx.beginPath();
      if (constraint.shape === "circle") {
        ctx.arc(centerX, centerY, constraint.radius * zoom, 0, Math.PI * 2);
      } else {
        ctx.rect(
          centerX - constraint.width * zoom / 2,
          centerY - constraint.height * zoom / 2,
          constraint.width * zoom,
          constraint.height * zoom
        );
      }
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function regionCellsCenter(cells) {
      if (!Array.isArray(cells) || cells.length === 0) {
        return null;
      }
      return [
        cells.reduce((sum, cell) => sum + cell[0], 0) / cells.length,
        cells.reduce((sum, cell) => sum + cell[1], 0) / cells.length,
      ];
    }

    function drawSmallRegionLabel(cells, text, fillColor) {
      const center = regionCellsCenter(cells);
      if (!center) {
        return;
      }
      const [centerRow, centerCol] = center;
      const centerX = (centerCol + 0.5) * zoom;
      const centerY = (centerRow + 0.5) * zoom;
      const markerRadius = Math.max(6, zoom * 1.1);
      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.arc(centerX, centerY, markerRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.font = `bold ${Math.max(9, zoom * 1.1)}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, centerX, centerY);
    }

    function drawReferenceCells(cells, fillColor, strokeColor, selected, labelText = null) {
      if (!Array.isArray(cells) || cells.length === 0) {
        return;
      }
      drawConstraintRegion(
        {
          shape: "freeform",
          cells,
        },
        fillColor,
        strokeColor,
        selected
      );
      if (labelText) {
        drawSmallRegionLabel(cells, labelText, strokeColor);
      }
    }

    function drawSoftReferenceRegion(constraint, selected) {
      if (constraint.preference_type === "relative_preference") {
        const referenceRegions = Array.isArray(constraint.reference_regions)
          ? constraint.reference_regions
          : [];
        const styles = [
          ["rgba(34, 139, 92, 0.18)", "#1f7a52", "A"],
          ["rgba(141, 83, 194, 0.18)", "#6d3ea0", "B"],
        ];
        referenceRegions.slice(0, 2).forEach((region, index) => {
          const [fillColor, strokeColor, labelText] = styles[index];
          drawReferenceCells(
            region.cells,
            fillColor,
            selected ? strokeColor : strokeColor,
            selected,
            labelText
          );
        });
        return;
      }
      if (!constraint.reference_region) {
        return;
      }
      drawReferenceCells(
        constraint.reference_region.cells,
        "rgba(213, 142, 31, 0.2)",
        selected ? "#7a4100" : "#d58e1f",
        selected
      );
    }

    function constraintDisplayCenter(constraint) {
      if (constraint.shape !== "freeform" || constraint.cells.length === 0) {
        return constraint.center;
      }
      return [
        constraint.cells.reduce((sum, cell) => sum + cell[0], 0) / constraint.cells.length,
        constraint.cells.reduce((sum, cell) => sum + cell[1], 0) / constraint.cells.length,
      ];
    }

    function softConstraintMapLabel(constraint) {
      const customLabel = String(constraint.label || "").trim();
      if (customLabel && customLabel !== "Soft Constraint") {
        return customLabel;
      }
      const labels = {
        near_preference: "Near",
        far_preference: "Far",
        relative_preference: "Relative",
        move_smoothness: "Smoothness",
        clearance: "Clearance",
        path_shape_preference: "Shape",
      };
      return labels[constraint.preference_type] || "Soft";
    }

    function compactCanvasLabel(value, maxLength = 18) {
      const text = String(value || "").trim();
      return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
    }

    function drawSoftConstraintTextLabel(text, centerX, centerY, markerRadius) {
      const label = compactCanvasLabel(text);
      if (!label) {
        return;
      }
      const fontSize = Math.max(8, Math.min(12, zoom * 0.85));
      ctx.font = `bold ${fontSize}px sans-serif`;
      const paddingX = 4;
      const paddingY = 2;
      const labelWidth = ctx.measureText(label).width + paddingX * 2;
      const labelHeight = fontSize + paddingY * 2;
      let labelX = centerX - labelWidth / 2;
      let labelY = centerY + markerRadius + 3;
      if (labelY + labelHeight > canvas.height) {
        labelY = centerY - markerRadius - labelHeight - 3;
      }
      labelX = Math.max(1, Math.min(canvas.width - labelWidth - 1, labelX));
      labelY = Math.max(1, Math.min(canvas.height - labelHeight - 1, labelY));

      ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
      ctx.strokeStyle = "rgba(36, 95, 150, 0.75)";
      ctx.lineWidth = 1;
      ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
      ctx.strokeRect(labelX + 0.5, labelY + 0.5, labelWidth - 1, labelHeight - 1);
      ctx.fillStyle = "#245f96";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, labelX + labelWidth / 2, labelY + labelHeight / 2);
    }

    function inspectedObjectIds(objectMap) {
      if (inspectedMapEntity?.type !== "object") {
        return new Set();
      }
      return new Set(
        [...objectMap.values()]
          .filter(object => (
            object.id === inspectedMapEntity.id ||
            object.category === inspectedMapEntity.category
          ))
          .map(object => object.id)
      );
    }

    function objectIdsFromDisplayCell(displayCell) {
      if (Array.isArray(displayCell)) {
        return displayCell;
      }
      return displayCell === 0 ? [] : [displayCell];
    }

    function objectDisplayBounds(displayObjectGrid, objectMap, objectIds) {
      const boundsById = new Map();
      for (const objectId of objectIds) {
        boundsById.set(objectId, {
          minRow: Number.POSITIVE_INFINITY,
          minCol: Number.POSITIVE_INFINITY,
          maxRow: Number.NEGATIVE_INFINITY,
          maxCol: Number.NEGATIVE_INFINITY,
          count: 0,
        });
      }
      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          for (const objectId of objectIdsFromDisplayCell(displayObjectGrid[row][col])) {
            const bounds = boundsById.get(objectId);
            if (!bounds) {
              continue;
            }
            bounds.minRow = Math.min(bounds.minRow, row);
            bounds.maxRow = Math.max(bounds.maxRow, row);
            bounds.minCol = Math.min(bounds.minCol, col);
            bounds.maxCol = Math.max(bounds.maxCol, col);
            bounds.count += 1;
          }
        }
      }
      for (const [objectId, bounds] of boundsById.entries()) {
        if (bounds.count > 0) {
          continue;
        }
        const object = objectMap.get(objectId);
        const center = Array.isArray(object?.center_grid) && object.center_grid.length === 2
          ? object.center_grid
          : Array.isArray(object?.metadata_grid_cell) && object.metadata_grid_cell.length === 2
            ? object.metadata_grid_cell
            : null;
        if (!center) {
          continue;
        }
        const centerRow = Number(center[0]);
        const centerCol = Number(center[1]);
        if (!Number.isFinite(centerRow) || !Number.isFinite(centerCol)) {
          continue;
        }
        const cellCount = Math.max(1, Number(object?.num_grid_cells || 1));
        const halfSide = Math.max(0.5, Math.sqrt(cellCount) / 2);
        bounds.minRow = centerRow - halfSide;
        bounds.maxRow = centerRow + halfSide;
        bounds.minCol = centerCol - halfSide;
        bounds.maxCol = centerCol + halfSide;
        bounds.count = cellCount;
        bounds.centerOnly = true;
      }
      return boundsById;
    }

    function drawObjectInspectionLabel(text, centerX, centerY, markerRadius, color, stackIndex = 0) {
      const label = compactCanvasLabel(text, 14);
      if (!label) {
        return;
      }
      const fontSize = Math.max(9, Math.min(14, zoom * 1.15));
      ctx.font = `bold ${fontSize}px sans-serif`;
      const paddingX = 5;
      const paddingY = 3;
      const width = ctx.measureText(label).width + paddingX * 2;
      const height = fontSize + paddingY * 2;
      let x = centerX - width / 2;
      let y = centerY - markerRadius - height - 4 - stackIndex * (height + 3);
      if (y < 1) {
        y = centerY + markerRadius + 4 + stackIndex * (height + 3);
      }
      x = Math.max(1, Math.min(canvas.width - width - 1, x));
      y = Math.max(1, Math.min(canvas.height - height - 1, y));

      ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.fillRect(x, y, width, height);
      ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);
      ctx.fillStyle = "#111111";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x + width / 2, y + height / 2);
    }

    function smallObjectMapLabel(object) {
      const rawName = String(object?.name || "").trim();
      const namePrefix = rawName.split("|")[0].trim();
      if (namePrefix) {
        return compactCanvasLabel(namePrefix, 22);
      }
      const category = String(object?.category || "").trim().replace(/_/g, " ");
      return compactCanvasLabel(category, 22);
    }

    function rectsOverlap(first, second, padding = 0) {
      return !(
        first.x + first.width + padding <= second.x ||
        second.x + second.width + padding <= first.x ||
        first.y + first.height + padding <= second.y ||
        second.y + second.height + padding <= first.y
      );
    }

    function clampLabelRect(rect) {
      return {
        ...rect,
        x: Math.max(1, Math.min(canvas.width - rect.width - 1, rect.x)),
        y: Math.max(1, Math.min(canvas.height - rect.height - 1, rect.y)),
      };
    }

    function objectBoundsRect(bounds) {
      return {
        x: bounds.minCol * zoom,
        y: bounds.minRow * zoom,
        width: (bounds.maxCol - bounds.minCol + 1) * zoom,
        height: (bounds.maxRow - bounds.minRow + 1) * zoom,
      };
    }

    function objectBoundsCenter(bounds) {
      return {
        x: ((bounds.minCol + bounds.maxCol + 1) / 2) * zoom,
        y: ((bounds.minRow + bounds.maxRow + 1) / 2) * zoom,
      };
    }

    function labelAnchorPoint(labelRect, targetPoint) {
      const centerX = labelRect.x + labelRect.width / 2;
      const centerY = labelRect.y + labelRect.height / 2;
      const deltaX = targetPoint.x - centerX;
      const deltaY = targetPoint.y - centerY;
      if (Math.abs(deltaX / labelRect.width) > Math.abs(deltaY / labelRect.height)) {
        return {
          x: deltaX > 0 ? labelRect.x + labelRect.width : labelRect.x,
          y: centerY,
        };
      }
      return {
        x: centerX,
        y: deltaY > 0 ? labelRect.y + labelRect.height : labelRect.y,
      };
    }

    function objectAnchorPoint(objectRect, labelPoint) {
      const centerX = objectRect.x + objectRect.width / 2;
      const centerY = objectRect.y + objectRect.height / 2;
      const deltaX = labelPoint.x - centerX;
      const deltaY = labelPoint.y - centerY;
      if (Math.abs(deltaX / Math.max(1, objectRect.width)) > Math.abs(deltaY / Math.max(1, objectRect.height))) {
        return {
          x: deltaX > 0 ? objectRect.x + objectRect.width : objectRect.x,
          y: centerY,
        };
      }
      return {
        x: centerX,
        y: deltaY > 0 ? objectRect.y + objectRect.height : objectRect.y,
      };
    }

    function candidateSmallObjectLabelRects(labelWidth, labelHeight, bounds) {
      const objectRect = objectBoundsRect(bounds);
      const center = objectBoundsCenter(bounds);
      const baseGap = Math.max(7, zoom * 1.5);
      const rings = [0, 1, 2, 3, 5, 8, 12];
      const candidates = [];
      for (const ring of rings) {
        const gap = baseGap + ring * Math.max(5, zoom * 1.4);
        candidates.push(
          {
            x: objectRect.x + objectRect.width + gap,
            y: center.y - labelHeight / 2,
            rank: 0 + ring,
          },
          {
            x: objectRect.x - labelWidth - gap,
            y: center.y - labelHeight / 2,
            rank: 1 + ring,
          },
          {
            x: center.x - labelWidth / 2,
            y: objectRect.y - labelHeight - gap,
            rank: 2 + ring,
          },
          {
            x: center.x - labelWidth / 2,
            y: objectRect.y + objectRect.height + gap,
            rank: 3 + ring,
          },
          {
            x: objectRect.x + objectRect.width + gap,
            y: objectRect.y - labelHeight - gap,
            rank: 4 + ring,
          },
          {
            x: objectRect.x + objectRect.width + gap,
            y: objectRect.y + objectRect.height + gap,
            rank: 5 + ring,
          },
          {
            x: objectRect.x - labelWidth - gap,
            y: objectRect.y - labelHeight - gap,
            rank: 6 + ring,
          },
          {
            x: objectRect.x - labelWidth - gap,
            y: objectRect.y + objectRect.height + gap,
            rank: 7 + ring,
          }
        );
      }
      return candidates.map(candidate => clampLabelRect({
        x: candidate.x,
        y: candidate.y,
        width: labelWidth,
        height: labelHeight,
        rank: candidate.rank,
      }));
    }

    function placeSmallObjectLabels(labels) {
      const placed = [];
      const occupied = [];
      const padding = Math.max(3, zoom * 0.8);
      const overlapArea = (first, second, extraPadding = 0) => {
        const overlapWidth = Math.max(
          0,
          Math.min(first.x + first.width, second.x + second.width + extraPadding) -
            Math.max(first.x, second.x - extraPadding)
        );
        const overlapHeight = Math.max(
          0,
          Math.min(first.y + first.height, second.y + second.height + extraPadding) -
            Math.max(first.y, second.y - extraPadding)
        );
        return overlapWidth * overlapHeight;
      };
      const sorted = labels.slice().sort((first, second) => {
        if (first.objectRect.y !== second.objectRect.y) {
          return first.objectRect.y - second.objectRect.y;
        }
        return first.objectRect.x - second.objectRect.x;
      });
      for (const label of sorted) {
        const candidates = candidateSmallObjectLabelRects(
          label.width,
          label.height,
          label.bounds
        );
        let best = null;
        let bestScore = Number.POSITIVE_INFINITY;
        let fallback = null;
        let fallbackScore = Number.POSITIVE_INFINITY;
        for (const candidate of candidates) {
          const overlapsLabel = occupied.some(rect => rectsOverlap(candidate, rect, padding));
          const overlapsObject = rectsOverlap(candidate, label.objectRect, padding);
          const distance = Math.hypot(
            candidate.x + candidate.width / 2 - label.target.x,
            candidate.y + candidate.height / 2 - label.target.y
          );
          const score = candidate.rank * 20 + distance + (overlapsObject ? 500 : 0);
          const collisionArea = occupied.reduce(
            (total, rect) => total + overlapArea(candidate, rect, padding),
            overlapsObject ? overlapArea(candidate, label.objectRect, padding) : 0
          );
          const fallbackCandidateScore = collisionArea * 12 + score;
          if (fallbackCandidateScore < fallbackScore) {
            fallback = candidate;
            fallbackScore = fallbackCandidateScore;
          }
          if (overlapsLabel) {
            continue;
          }
          if (score < bestScore) {
            best = candidate;
            bestScore = score;
          }
        }
        if (!best) {
          best = fallback;
        }
        if (!best) {
          continue;
        }
        occupied.push(best);
        placed.push({
          ...label,
          rect: best,
        });
      }
      return placed;
    }

    function drawSmallObjectNameArrow(labelRect, objectRect, color) {
      const target = objectAnchorPoint(
        objectRect,
        {
          x: labelRect.x + labelRect.width / 2,
          y: labelRect.y + labelRect.height / 2,
        }
      );
      const source = labelAnchorPoint(labelRect, target);
      const angle = Math.atan2(target.y - source.y, target.x - source.x);
      const headLength = Math.max(4, zoom * 1.1);
      ctx.save();
      ctx.strokeStyle = "rgba(17, 17, 17, 0.82)";
      ctx.lineWidth = Math.max(2, zoom * 0.2);
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.stroke();
      ctx.strokeStyle = hexToRgba(color, 0.95);
      ctx.lineWidth = Math.max(1, zoom * 0.12);
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.stroke();
      ctx.fillStyle = hexToRgba(color, 0.95);
      ctx.beginPath();
      ctx.moveTo(target.x, target.y);
      ctx.lineTo(
        target.x - headLength * Math.cos(angle - Math.PI / 6),
        target.y - headLength * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        target.x - headLength * Math.cos(angle + Math.PI / 6),
        target.y - headLength * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function drawSmallObjectNameTag(placement) {
      const { object, label, rect, fontSize, color, objectRect } = placement;
      drawSmallObjectNameArrow(rect, objectRect, color);
      ctx.save();
      ctx.font = `bold ${fontSize}px sans-serif`;
      ctx.shadowColor = "rgba(0, 0, 0, 0.28)";
      ctx.shadowBlur = Math.max(2, zoom * 0.35);
      ctx.shadowOffsetY = Math.max(1, zoom * 0.18);
      ctx.fillStyle = hexToRgba(color, 0.92);
      ctx.strokeStyle = "rgba(17, 17, 17, 0.78)";
      ctx.lineWidth = Math.max(1, zoom * 0.14);
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
      ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.width - 1, rect.height - 1);
      ctx.shadowBlur = 0;
      const stripeWidth = Math.max(3, Math.min(5, rect.height * 0.22));
      ctx.fillStyle = "rgba(255, 255, 255, 0.36)";
      ctx.fillRect(rect.x, rect.y, stripeWidth, rect.height);
      ctx.fillStyle = textColorForHex(color);
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(label, rect.x + stripeWidth + 6, rect.y + rect.height / 2);
      ctx.restore();
    }

    function drawSmallObjectNames(displayObjectGrid, objectMap) {
      if (!showSmallObjectNames) {
        return;
      }
      const objectIds = new Set(
        displayObjectGrid
          .flat()
          .flatMap(objectIdsFromDisplayCell)
          .filter(objectId => objectId !== 0)
      );
      const boundsById = objectDisplayBounds(displayObjectGrid, objectMap, objectIds);
      const labels = [];
      const fontSize = Math.max(9, Math.min(13, zoom * 0.95));
      ctx.save();
      ctx.font = `bold ${fontSize}px sans-serif`;
      for (const [objectId, bounds] of boundsById.entries()) {
        if (bounds.count === 0) {
          continue;
        }
        const object = objectMap.get(objectId);
        if (!object || shouldHideObjectCategory(object.category)) {
          continue;
        }
        const forceSmallObjectName = FORCE_SMALL_OBJECT_NAME_CATEGORIES.has(
          normalizedObjectCategory(object.category)
        );
        const width = bounds.maxCol - bounds.minCol + 1;
        const height = bounds.maxRow - bounds.minRow + 1;
        if (
          !forceSmallObjectName &&
          (
            width > MIN_OBJECT_VISUAL_SIDE ||
            height > MIN_OBJECT_VISUAL_SIDE ||
            bounds.count > SMALL_OBJECT_NAME_MAX_CELLS
          )
        ) {
          continue;
        }
        const label = smallObjectMapLabel(object);
        if (!label) {
          continue;
        }
        const color = currentLegends.object_categories[object.category]?.color || "#333333";
        const paddingX = 12;
        const paddingY = 3;
        const labelWidth = ctx.measureText(label).width + paddingX * 2;
        const labelHeight = fontSize + paddingY * 2;
        const objectRect = objectBoundsRect(bounds);
        labels.push({
          object,
          bounds,
          label,
          fontSize,
          color,
          objectRect,
          target: objectBoundsCenter(bounds),
          width: labelWidth,
          height: labelHeight,
        });
      }
      ctx.restore();

      for (const placement of placeSmallObjectLabels(labels)) {
        drawSmallObjectNameTag(placement);
      }
    }

    function drawObjectDisplayStackCell(row, col, objectIds, objectGrid, objectMap) {
      if (!Array.isArray(objectIds) || objectIds.length === 0) {
        return;
      }
      const visibleIds = objectIds.filter(objectId => {
        const object = objectMap.get(objectId);
        return object && currentLegends.object_categories[object.category]?.color;
      });
      if (visibleIds.length === 0) {
        return;
      }

      visibleIds.forEach((objectId, index) => {
        const object = objectMap.get(objectId);
        const color = currentLegends.object_categories[object.category]?.color;
        const isRawCell = objectGrid[row][col] === objectId;
        const isTop = index === visibleIds.length - 1;
        if (visibleIds.length === 1) {
          ctx.fillStyle = hexToRgba(color, isRawCell ? 0.85 : 0.48);
          ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
          return;
        }
        if (index === 0) {
          ctx.fillStyle = hexToRgba(color, isRawCell ? 0.58 : 0.28);
          ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
          return;
        }
        const inset = Math.max(1, zoom * (0.12 + index * 0.04));
        const side = Math.max(1, zoom - inset * 2);
        ctx.fillStyle = hexToRgba(color, isTop ? 0.88 : 0.62);
        ctx.fillRect(col * zoom + inset, row * zoom + inset, side, side);
        ctx.strokeStyle = "rgba(17, 17, 17, 0.58)";
        ctx.lineWidth = Math.max(1, zoom * 0.08);
        ctx.strokeRect(col * zoom + inset, row * zoom + inset, side, side);
      });
    }

    function drawObjectInspectionRings(displayObjectGrid, objectMap) {
      const objectIds = inspectedObjectIds(objectMap);
      if (objectIds.size === 0) {
        return;
      }
      const boundsById = objectDisplayBounds(displayObjectGrid, objectMap, objectIds);
      const centerStackCounts = new Map();
      for (const [objectId, bounds] of boundsById.entries()) {
        if (bounds.count === 0) {
          continue;
        }
        const centerRow = (bounds.minRow + bounds.maxRow + 1) / 2;
        const centerCol = (bounds.minCol + bounds.maxCol + 1) / 2;
        const centerX = centerCol * zoom;
        const centerY = centerRow * zoom;
        const objectWidth = (bounds.maxCol - bounds.minCol + 1) * zoom;
        const objectHeight = (bounds.maxRow - bounds.minRow + 1) * zoom;
        const centerKey = `${Math.round(centerRow)}:${Math.round(centerCol)}`;
        const stackIndex = centerStackCounts.get(centerKey) || 0;
        centerStackCounts.set(centerKey, stackIndex + 1);
        const markerRadius = (
          Math.hypot(objectWidth, objectHeight) / 2 +
          OBJECT_INSPECTION_RING_PADDING_GRIDS * zoom +
          stackIndex * Math.max(3, zoom * 0.7)
        );
        const isSelectedObject = objectId === inspectedMapEntity.id;
        const ringColor = bounds.centerOnly
          ? "#ff7a45"
          : isSelectedObject ? "#00a6c8" : "#f2c94c";

        ctx.save();
        ctx.shadowColor = "rgba(0, 0, 0, 0.35)";
        ctx.shadowBlur = Math.max(2, zoom * 0.35);
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = Math.max(3, zoom * 0.45);
        ctx.beginPath();
        ctx.arc(centerX, centerY, markerRadius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = ringColor;
        ctx.lineWidth = Math.max(2, zoom * 0.28);
        ctx.beginPath();
        ctx.arc(centerX, centerY, markerRadius, 0, Math.PI * 2);
        ctx.stroke();
        if (isSelectedObject) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = Math.max(1, zoom * 0.18);
          ctx.beginPath();
          ctx.arc(centerX, centerY, markerRadius + Math.max(3, zoom * 0.7), 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();

        const object = objectMap.get(objectId);
        const label = isSelectedObject
          ? `selected ${objectId}`
          : bounds.centerOnly
            ? `id ${objectId}*`
            : `id ${objectId}`;
        drawObjectInspectionLabel(label, centerX, centerY, markerRadius, ringColor, stackIndex);
      }
    }

    function drawGrid() {
      const occupancy = mapState.layers.occupancy;
      const room = mapState.layers.room;
      const objectGrid = mapState.layers.object_instance;
      const displayObjectStackGrid = expandedObjectStackGrid();
      const displayObjectGrid = expandedObjectGrid();
      const roomMap = roomInstanceMap();
      const objectMap = objectInstanceMap();

      for (let row = 0; row < GRID_SIZE; row += 1) {
        for (let col = 0; col < GRID_SIZE; col += 1) {
          const occupancyName = occupancyNameFromValue(occupancy[row][col]);
          ctx.fillStyle = currentLegends.occupancy[occupancyName].color;
          ctx.fillRect(col * zoom, row * zoom, zoom, zoom);

          const roomId = room[row][col];
          if (showRooms && roomId !== 0 && roomMap.has(roomId) && occupancyName === "free") {
            const roomItem = roomMap.get(roomId);
            const color = currentLegends.room_categories[roomItem.category]?.color;
            if (color) {
              ctx.fillStyle = hexToRgba(color, 0.45);
              ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
            }
          }

          if (showObjects) {
            drawObjectDisplayStackCell(
              row,
              col,
              displayObjectStackGrid[row][col],
              objectGrid,
              objectMap
            );
          }
        }
      }

      const sample = getActiveSample();
      if (showHardConstraints) {
        for (const constraint of sample.hard_constraints) {
          if (constraint.initialized === false) {
            continue;
          }
          const isMustPass = constraint.kind === "must_pass";
          drawConstraintRegion(
            constraint,
            isMustPass ? "rgba(0, 0, 0, 0.22)" : "rgba(209, 79, 69, 0.28)",
            isMustPass ? "#000000" : "#a9342c",
            constraint.constraint_id === selectedConstraintId
          );
        }
      }
      if (showSoftConstraints) {
        for (const constraint of sample.soft_constraints) {
          if (constraint.initialized === false) {
            continue;
          }
          if (isRegionlessTrajectoryMetricPreference(constraint.preference_type)) {
            continue;
          }
          drawConstraintRegion(
            constraint,
            "rgba(52, 120, 184, 0.24)",
            "#245f96",
            constraint.constraint_id === selectedSoftConstraintId
          );
          drawSoftReferenceRegion(
            constraint,
            constraint.constraint_id === selectedSoftConstraintId
          );
        }
      }

      if (pendingConstraintCorner) {
        const [row, col] = pendingConstraintCorner;
        ctx.strokeStyle = currentMode.startsWith("must_pass")
          ? "#000000"
          : currentMode.startsWith("must_avoid")
            ? "#a9342c"
            : "#245f96";
        ctx.lineWidth = 3;
        ctx.strokeRect(col * zoom + 1.5, row * zoom + 1.5, zoom - 3, zoom - 3);
      }

      sample.expert_route.forEach(([row, col], index) => {
        ctx.fillStyle = routeColorAt(index, sample.expert_route.length);
        ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
        if (zoom >= 4) {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
          ctx.lineWidth = 1;
          ctx.strokeRect(col * zoom + 0.5, row * zoom + 0.5, zoom - 1, zoom - 1);
        }
      });

      if (
        currentMode === "route_adjust" &&
        routeAdjustIndex !== null &&
        sample.expert_route[routeAdjustIndex]
      ) {
        const [sourceRow, sourceCol] = sample.expert_route[routeAdjustIndex];
        ctx.strokeStyle = "#f2c94c";
        ctx.lineWidth = Math.max(2, Math.floor(zoom / 2));
        ctx.strokeRect(
          sourceCol * zoom + 1,
          sourceRow * zoom + 1,
          Math.max(1, zoom - 2),
          Math.max(1, zoom - 2)
        );
        if (routeAdjustTargetCell) {
          const [targetRow, targetCol] = routeAdjustTargetCell;
          ctx.strokeStyle = "#00a6c8";
          ctx.strokeRect(
            targetCol * zoom + 1,
            targetRow * zoom + 1,
            Math.max(1, zoom - 2),
            Math.max(1, zoom - 2)
          );
        }
      }

      if (sample.start_pose.row !== null && sample.start_pose.col !== null) {
        const row = sample.start_pose.row;
        const col = sample.start_pose.col;
        ctx.fillStyle = INITIAL_STATE.start_color;
        ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
      }

      if (inspectedMapEntity?.type === "room") {
        ctx.fillStyle = "#111111";
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = Math.max(1, Math.floor(zoom / 4));
        for (let row = 0; row < GRID_SIZE; row += 1) {
          for (let col = 0; col < GRID_SIZE; col += 1) {
            if (room[row][col] !== inspectedMapEntity.id) {
              continue;
            }
            ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
            ctx.strokeRect(
              col * zoom + ctx.lineWidth / 2,
              row * zoom + ctx.lineWidth / 2,
              zoom - ctx.lineWidth,
              zoom - ctx.lineWidth
            );
          }
        }
      }

      if (selectedTemplateObjectId !== null) {
        ctx.fillStyle = "#000000";
        for (let row = 0; row < GRID_SIZE; row += 1) {
          for (let col = 0; col < GRID_SIZE; col += 1) {
            if (displayObjectStackGrid[row][col].includes(selectedTemplateObjectId)) {
              ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
            }
          }
        }
      }

      if (inspectedMapEntity?.type === "object") {
        ctx.fillStyle = "#111111";
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = Math.max(1, Math.floor(zoom / 4));
        for (let row = 0; row < GRID_SIZE; row += 1) {
          for (let col = 0; col < GRID_SIZE; col += 1) {
            const objectIds = displayObjectStackGrid[row][col];
            const matchesInspection = objectIds.some(objectId => {
              const objectItem = objectMap.get(objectId);
              return (
                objectId === inspectedMapEntity.id ||
                objectItem?.category === inspectedMapEntity.category
              );
            });
            if (!matchesInspection) {
              continue;
            }
            ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
            ctx.strokeRect(
              col * zoom + ctx.lineWidth / 2,
              row * zoom + ctx.lineWidth / 2,
              zoom - ctx.lineWidth,
              zoom - ctx.lineWidth
            );
          }
        }
        drawObjectInspectionRings(displayObjectStackGrid, objectMap);
      }

      if (showGrid) {
        for (let index = 0; index <= GRID_SIZE; index += 1) {
          const color = index % 16 === 0 ? "#B9AA8D" : "#D4CAB8";
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, index * zoom + 0.5);
          ctx.lineTo(canvas.width, index * zoom + 0.5);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(index * zoom + 0.5, 0);
          ctx.lineTo(index * zoom + 0.5, canvas.height);
          ctx.stroke();
        }
      }

      drawSmallObjectNames(displayObjectStackGrid, objectMap);

      if (showHardConstraints) {
        for (const constraint of sample.hard_constraints.filter(
          item => item.kind === "must_pass"
        )) {
          const [centerRow, centerCol] = constraintDisplayCenter(constraint);
          const centerX = (centerCol + 0.5) * zoom;
          const centerY = (centerRow + 0.5) * zoom;
          const markerRadius = Math.max(8, zoom * 1.6);
          ctx.fillStyle = "#000000";
          ctx.beginPath();
          ctx.arc(centerX, centerY, markerRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.fillStyle = "#ffffff";
          ctx.font = `bold ${Math.max(10, zoom * 1.5)}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(constraint.order), centerX, centerY);
        }
      }
      if (showSoftConstraints) {
        let softDisplayIndex = 1;
        for (const constraint of sample.soft_constraints) {
          if (
            constraint.initialized === false ||
            isRegionlessTrajectoryMetricPreference(constraint.preference_type)
          ) {
            continue;
          }
          const [centerRow, centerCol] = constraintDisplayCenter(constraint);
          const centerX = (centerCol + 0.5) * zoom;
          const centerY = (centerRow + 0.5) * zoom;
          const markerRadius = Math.max(8, zoom * 1.45);
          ctx.fillStyle = "#245f96";
          ctx.beginPath();
          ctx.arc(centerX, centerY, markerRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.fillStyle = "#ffffff";
          ctx.font = `bold ${Math.max(9, zoom * 1.05)}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(`S${softDisplayIndex}`, centerX, centerY);
          drawSoftConstraintTextLabel(
            softConstraintMapLabel(constraint),
            centerX,
            centerY,
            markerRadius
          );
          softDisplayIndex += 1;
        }
      }
    }

    function eventToCell(event) {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const col = Math.floor(x / zoom);
      const row = Math.floor(y / zoom);
      if (row < 0 || row >= GRID_SIZE || col < 0 || col >= GRID_SIZE) {
        return null;
      }
      return [row, col];
    }

    function describeCell(row, col) {
      const occupancy = occupancyNameFromValue(mapState.layers.occupancy[row][col]);
      const occupancyLabel = currentLegends.occupancy[occupancy].label;
      const roomId = mapState.layers.room[row][col];
      const objectId = mapState.layers.object_instance[row][col];
      const displayObjectIds = displayedObjectIdsAt(row, col);
      const roomItem = roomInstanceMap().get(roomId);
      const objectMap = objectInstanceMap();
      const objectItems = displayObjectIds
        .map(displayObjectId => objectMap.get(displayObjectId))
        .filter(Boolean);

      let text = `Cell (${row}, ${col}) = ${occupancyLabel}`;
      if (roomItem) {
        const roomLabel = currentLegends.room_categories[roomItem.category]?.label || roomItem.category;
        text += ` | room=${roomItem.name} (${roomLabel})`;
      }
      if (objectItems.length > 0) {
        const objectDescriptions = objectItems.map(objectItem => {
          const objectLabel = currentLegends.object_categories[objectItem.category]?.label || objectItem.category;
          const expandedPrefix = objectId !== objectItem.id ? "visualized " : "";
          return `${expandedPrefix}${objectItem.name} (${objectLabel})`;
        });
        text += ` | objects=${objectDescriptions.join("; ")}`;
      }
      return text;
    }

    function hideMapInspector() {
      inspectorSelection = null;
      inspectedMapEntity = null;
      mapInspector.classList.remove("visible");
      mapInspector.innerHTML = "";
    }

    function showMapInspector(type, item) {
      const selectionKey = `${type}:${item.id}`;
      if (inspectorSelection === selectionKey) {
        hideMapInspector();
        return;
      }
      inspectorSelection = selectionKey;
      inspectedMapEntity = {
        type,
        id: item.id,
        category: item.category,
      };
      mapInspector.innerHTML = "";
      const title = document.createElement("div");
      title.className = "sample-title";
      title.textContent = `${type === "object" ? "Object" : "Room"}: ${item.name}`;
      const category = document.createElement("div");
      category.className = "small";
      category.textContent = `id=${item.id} · category=${item.category}`;
      mapInspector.appendChild(title);
      mapInspector.appendChild(category);
      if (type === "object") {
        const count = document.createElement("div");
        count.className = "small";
        count.textContent = `same-category objects in scene: ${objectCategoryCount(item.category)}`;
        mapInspector.appendChild(count);
      }
      if (Array.isArray(item.attributes) && item.attributes.length > 0) {
        const attributesTitle = document.createElement("div");
        attributesTitle.className = "section-title";
        attributesTitle.style.marginTop = "10px";
        attributesTitle.textContent = "Attributes";
        mapInspector.appendChild(attributesTitle);
        for (const attribute of item.attributes) {
          const row = document.createElement("div");
          row.className = "small";
          row.textContent = `• ${attribute}`;
          mapInspector.appendChild(row);
        }
      }
      mapInspector.classList.add("visible");
    }

    function inspectCell(row, col) {
      const objectId = displayedObjectIdAt(row, col);
      const objectItem = objectInstanceMap().get(objectId);
      if (objectItem) {
        showMapInspector("object", objectItem);
        drawGrid();
        setStatus(
          `Inspecting object ${objectItem.name} (id=${objectItem.id}); ` +
          `drawing 5-grid padded circles around all visible ${objectItem.category} objects ` +
          `(${objectCategoryCount(objectItem.category)} in scene).`
        );
        return;
      }
      const roomId = mapState.layers.room[row][col];
      const roomItem = roomInstanceMap().get(roomId);
      if (roomItem) {
        showMapInspector("room", roomItem);
        drawGrid();
        setStatus(`Inspecting room ${roomItem.name} (id=${roomItem.id}).`);
        return;
      }
      hideMapInspector();
      drawGrid();
      setStatus(`Cleared inspect selection at (${row}, ${col}).`);
    }

    function pickReferenceObjectAt(row, col) {
      const objectId = displayedObjectIdAt(row, col);
      const objectItem = objectInstanceMap().get(objectId);
      if (!objectItem) {
        setStatus("Reference selection is active. Right-click a grid cell occupied by an object.");
        return;
      }
      const constraint = getSelectedSoftConstraint();
      if (!constraint) {
        referencePickTarget = null;
        setStatus("The selected soft constraint is no longer available.");
        return;
      }
      if (referencePickTarget === "single") {
        recordUndoSnapshot("select reference object");
        constraint.reference_region = {
          mode: "object",
          object_id: objectId,
          cells: cellsForObject(objectId),
        };
      } else {
        const index = referencePickTarget === "relative_a" ? 0 : 1;
        const otherIndex = index === 0 ? 1 : 0;
        const objectIds = constraint.object_ids.length === 2
          ? constraint.object_ids.slice()
          : defaultRelativeObjectIds();
        if (objectIds[otherIndex] === objectId) {
          setStatus("Relative preference requires two different reference objects.");
          return;
        }
        recordUndoSnapshot("select reference object");
        objectIds[index] = objectId;
        constraint.object_ids = objectIds;
        constraint.reference_regions = objectIds.map(id => ({
          object_id: id,
          cells: cellsForObject(id),
        }));
      }
      referencePickTarget = null;
      getActiveSample().updated_at = nowIso();
      renderSoftConstraintEditor();
      renderChecks();
      drawGrid();
      setStatus(`Selected ${objectItem.name} (id=${objectId}) as a reference object.`);
    }

    function setMode(modeName) {
      currentMode = modeName;
      astarStartCell = null;
      routeGestureStartCell = null;
      routeGestureDragged = false;
      cancelRouteSegment();
      routeAdjustIndex = null;
      routeAdjustTargetCell = null;
      pendingConstraintCorner = null;
      renderModeList();
      drawGrid();
      setStatus(`Mode set to ${MODES[modeName].label}.`);
    }

    function interpolateLine(startCell, endCell) {
      const points = [];
      let x0 = startCell[1];
      let y0 = startCell[0];
      const x1 = endCell[1];
      const y1 = endCell[0];
      const dx = Math.abs(x1 - x0);
      const dy = Math.abs(y1 - y0);
      const sx = x0 < x1 ? 1 : -1;
      const sy = y0 < y1 ? 1 : -1;
      let err = dx - dy;

      while (true) {
        points.push([y0, x0]);
        if (x0 === x1 && y0 === y1) {
          break;
        }
        const e2 = err * 2;
        if (e2 > -dy) {
          err -= dy;
          x0 += sx;
        }
        if (e2 < dx) {
          err += dx;
          y0 += sy;
        }
      }
      return points;
    }

    function nearestRoutePointIndex(cell, maxDistance = 2.5) {
      const route = getActiveSample().expert_route;
      let bestIndex = -1;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let index = 0; index < route.length; index += 1) {
        const point = route[index];
        const distance = Math.hypot(point[0] - cell[0], point[1] - cell[1]);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = index;
        }
      }
      return bestDistance <= maxDistance ? bestIndex : -1;
    }

    function replaceRouteWaypointWithAStar(index, targetCell) {
      const sample = getActiveSample();
      const route = sample.expert_route;
      if (!Number.isInteger(index) || index < 0 || index >= route.length) {
        setStatus("Select a route waypoint first.");
        return false;
      }
      if (!isTraversableCell(targetCell[0], targetCell[1])) {
        setStatus(`Route adjustment target must be traversable. ${describeCell(targetCell[0], targetCell[1])}`);
        return false;
      }
      if (sameCell(route[index], targetCell)) {
        setStatus("Route waypoint was not moved.");
        return false;
      }

      const previous = index > 0 ? route[index - 1] : null;
      const next = index < route.length - 1 ? route[index + 1] : null;
      let leftPath = null;
      let rightPath = null;

      if (previous) {
        leftPath = findAStarPath(previous, targetCell);
        if (!leftPath) {
          setStatus(`No A* path found from previous route cell to (${targetCell[0]}, ${targetCell[1]}).`);
          return false;
        }
      }
      if (next) {
        rightPath = findAStarPath(targetCell, next);
        if (!rightPath) {
          setStatus(`No A* path found from (${targetCell[0]}, ${targetCell[1]}) to next route cell.`);
          return false;
        }
      }

      recordUndoSnapshot("adjust expert trajectory");
      let adjustedRoute = [];
      if (previous) {
        adjustedRoute = route.slice(0, index);
        adjustedRoute.push(...leftPath.slice(1));
      } else {
        adjustedRoute = [[targetCell[0], targetCell[1]]];
      }
      if (rightPath) {
        adjustedRoute.push(...rightPath.slice(1));
        adjustedRoute.push(...route.slice(index + 2));
      }

      sample.expert_route = adjustedRoute;
      clearAStarHistory(sample);
      syncStartPoseFromRoute(sample);
      sample.updated_at = nowIso();
      renderStartPoseInfo();
      renderChecks();
      drawGrid();
      setStatus(`Adjusted route waypoint ${index + 1}/${route.length} to (${targetCell[0]}, ${targetCell[1]}).`);
      return true;
    }

    function ensureRouteSegmentHistory(sample = getActiveSample()) {
      if (!Array.isArray(sample.route_segments)) {
        sample.route_segments = [];
      }
      return sample.route_segments;
    }

    function cloneRouteCells(cells) {
      return cells.map(point => [point[0], point[1]]);
    }

    function beginRouteSegment(kind) {
      activeRouteSegment = {
        kind,
        cells: [],
      };
    }

    function recordRouteSegmentCells(kind, cells) {
      if (!Array.isArray(cells) || cells.length === 0) {
        return;
      }
      if (activeRouteSegment) {
        activeRouteSegment.cells.push(...cloneRouteCells(cells));
        if (kind === "astar") {
          activeRouteSegment.usedAStar = true;
        }
        return;
      }
      ensureRouteSegmentHistory().push({
        kind,
        cells: cloneRouteCells(cells),
      });
    }

    function finishRouteSegment() {
      if (!activeRouteSegment) {
        return;
      }
      if (activeRouteSegment.cells.length > 0) {
        ensureRouteSegmentHistory().push({
          kind: activeRouteSegment.usedAStar ? "freehand_with_astar" : activeRouteSegment.kind,
          cells: cloneRouteCells(activeRouteSegment.cells),
        });
      }
      activeRouteSegment = null;
    }

    function cancelRouteSegment() {
      activeRouteSegment = null;
    }

    function normalizeRouteSegment(segment) {
      if (Array.isArray(segment)) {
        return {
          kind: "astar",
          cells: cloneRouteCells(segment),
        };
      }
      if (!segment || !Array.isArray(segment.cells)) {
        return {
          kind: "manual",
          cells: [],
        };
      }
      return {
        kind: segment.kind || "manual",
        cells: cloneRouteCells(segment.cells),
      };
    }

	    function appendRouteCells(cells, segmentKind = "manual", options = {}) {
	      const sample = getActiveSample();
	      recordGestureUndo("edit expert trajectory");
	      const appended = [];
	      for (const [row, col] of cells) {
	        const last = sample.expert_route[sample.expert_route.length - 1];
	        if (last && last[0] === row && last[1] === col) {
	          continue;
	        }
	        const point = [row, col];
	        sample.expert_route.push(point);
	        appended.push(point);
	      }
	      if (appended.length === 0) {
	        return appended;
	      }
	      recordRouteSegmentCells(segmentKind, appended);
	      sample.updated_at = nowIso();
	      const startPoseChanged = syncStartPoseFromRoute(sample);
	      if (startPoseChanged || options.updateChecks !== false) {
	        renderStartPoseInfo();
	      }
	      if (options.updateChecks !== false) {
	        renderChecks();
	      }
	      if (options.deferDraw) {
	        scheduleDrawGrid();
	      } else {
	        drawGrid();
	      }
	      return appended;
	    }

	    function eraseRouteCells(cells, options = {}) {
	      recordGestureUndo("erase expert trajectory");
	      const toRemove = new Set(cells.map(([row, col]) => cellKey(row, col)));
	      const sample = getActiveSample();
	      const previousLength = sample.expert_route.length;
	      sample.expert_route = sample.expert_route.filter(
	        ([row, col]) => !toRemove.has(cellKey(row, col))
	      );
	      if (sample.expert_route.length === previousLength) {
	        return;
	      }
	      clearAStarHistory(sample);
	      const startPoseChanged = syncStartPoseFromRoute(sample);
	      sample.updated_at = nowIso();
	      if (startPoseChanged || options.updateChecks !== false) {
	        renderStartPoseInfo();
	      }
	      if (options.updateChecks !== false) {
	        renderChecks();
	      }
	      if (options.deferDraw) {
	        scheduleDrawGrid();
	      } else {
	        drawGrid();
	      }
	    }

    function undoLastRouteSegment() {
      const sample = getActiveSample();
      const routeSegments = ensureRouteSegmentHistory(sample);
      const legacyAStarSegments = Array.isArray(sample.astar_segments)
        ? sample.astar_segments
        : [];
      if (routeSegments.length === 0 && legacyAStarSegments.length === 0) {
        setStatus("No route segment is available to undo.");
        return;
      }
      recordUndoSnapshot("undo route segment");
      const rawSegment = routeSegments.length > 0
        ? routeSegments.pop()
        : legacyAStarSegments.pop();
      const lastSegment = normalizeRouteSegment(rawSegment);
      if (lastSegment.cells.length === 0) {
        setStatus("The latest route segment was empty.");
        return;
      }
      const remaining = sample.expert_route.slice(
        0,
        Math.max(0, sample.expert_route.length - lastSegment.cells.length)
      );
      const matchesTail = lastSegment.cells.every((point, index) => {
        const routePoint = sample.expert_route[
          sample.expert_route.length - lastSegment.cells.length + index
        ];
        return routePoint && routePoint[0] === point[0] && routePoint[1] === point[1];
      });
      sample.expert_route = matchesTail ? remaining : sample.expert_route.filter(([row, col]) => {
        return !lastSegment.cells.some(point => point[0] === row && point[1] === col);
      });
      syncStartPoseFromRoute(sample);
      sample.updated_at = nowIso();
      if (sample.expert_route.length > 0) {
        const tail = sample.expert_route[sample.expert_route.length - 1];
        astarStartCell = [tail[0], tail[1]];
      } else {
        astarStartCell = null;
      }
      renderStartPoseInfo();
      renderChecks();
      drawGrid();
      setStatus(
        `Removed the previous ${lastSegment.kind} route segment with ${lastSegment.cells.length} cell(s).`
      );
    }

    function isTraversableCell(row, col) {
      return mapState.layers.occupancy[row][col] === currentLegends.occupancy.free.value;
    }

    function canTraverseBetween(fromRow, fromCol, toRow, toCol) {
      if (!isTraversableCell(toRow, toCol)) {
        return false;
      }
      const dRow = toRow - fromRow;
      const dCol = toCol - fromCol;
      if (Math.abs(dRow) !== 1 || Math.abs(dCol) !== 1) {
        return true;
      }
      return (
        isTraversableCell(fromRow + dRow, fromCol) &&
        isTraversableCell(fromRow, fromCol + dCol)
      );
    }

    function heuristic(row, col, goalRow, goalCol) {
      const rowDelta = Math.abs(row - goalRow);
      const colDelta = Math.abs(col - goalCol);
      const diagonalSteps = Math.min(rowDelta, colDelta);
      const straightSteps = Math.max(rowDelta, colDelta) - diagonalSteps;
      return diagonalSteps * Math.SQRT2 + straightSteps;
    }

    function parseKey(key) {
      return key.split(":").map(Number);
    }

    function reconstructPath(cameFrom, endKey) {
      const path = [];
      let currentKey = endKey;
      while (currentKey) {
        path.push(parseKey(currentKey));
        currentKey = cameFrom.get(currentKey) || null;
      }
      path.reverse();
      return path;
    }

    function findAStarPath(startCell, goalCell) {
      const [startRow, startCol] = startCell;
      const [goalRow, goalCol] = goalCell;
      const startKey = cellKey(startRow, startCol);
      const goalKey = cellKey(goalRow, goalCol);
      if (startKey === goalKey) {
        return [startCell];
      }

      const openList = [startKey];
      const openKeys = new Set([startKey]);
      const cameFrom = new Map();
      const gScore = new Map([[startKey, 0]]);
      const fScore = new Map([[startKey, heuristic(startRow, startCol, goalRow, goalCol)]]);
      const directions = [
        [-1, 0, 1],
        [1, 0, 1],
        [0, -1, 1],
        [0, 1, 1],
        [-1, -1, Math.SQRT2],
        [-1, 1, Math.SQRT2],
        [1, -1, Math.SQRT2],
        [1, 1, Math.SQRT2],
      ];

      while (openList.length > 0) {
        let currentIndex = 0;
        let currentKey = openList[0];
        let currentScore = fScore.get(currentKey) ?? Number.POSITIVE_INFINITY;

        for (let index = 1; index < openList.length; index += 1) {
          const candidateKey = openList[index];
          const candidateScore = fScore.get(candidateKey) ?? Number.POSITIVE_INFINITY;
          if (candidateScore < currentScore) {
            currentIndex = index;
            currentKey = candidateKey;
            currentScore = candidateScore;
          }
        }

        openList.splice(currentIndex, 1);
        openKeys.delete(currentKey);

        if (currentKey === goalKey) {
          return reconstructPath(cameFrom, goalKey);
        }

        const [row, col] = parseKey(currentKey);
        const currentG = gScore.get(currentKey) ?? Number.POSITIVE_INFINITY;
        for (const [dRow, dCol, stepCost] of directions) {
          const nextRow = row + dRow;
          const nextCol = col + dCol;
          if (
            nextRow < 0 ||
            nextRow >= GRID_SIZE ||
            nextCol < 0 ||
            nextCol >= GRID_SIZE ||
            !canTraverseBetween(row, col, nextRow, nextCol)
          ) {
            continue;
          }

          const nextKey = cellKey(nextRow, nextCol);
          const tentativeG = currentG + stepCost;
          if (tentativeG >= (gScore.get(nextKey) ?? Number.POSITIVE_INFINITY)) {
            continue;
          }

          cameFrom.set(nextKey, currentKey);
          gScore.set(nextKey, tentativeG);
          fScore.set(
            nextKey,
            tentativeG + heuristic(nextRow, nextCol, goalRow, goalCol)
          );

          if (!openKeys.has(nextKey)) {
            openList.push(nextKey);
            openKeys.add(nextKey);
          }
        }
      }

      return null;
    }

	    function appendAStarRouteSegment(startCell, goalCell, options = {}) {
      const [startRow, startCol] = startCell;
      const [goalRow, goalCol] = goalCell;
      if (sameCell(startCell, goalCell)) {
        return true;
      }
      if (!isTraversableCell(startRow, startCol)) {
        setStatus(`Cannot A* connect from a non-traversable route tail. ${describeCell(startRow, startCol)}`);
        return false;
      }
      if (!isTraversableCell(goalRow, goalCol)) {
        setStatus(`A* only supports traversable target cells. ${describeCell(goalRow, goalCol)}`);
        return false;
      }

      const path = findAStarPath(startCell, goalCell);
      if (!path) {
        setStatus(`No A* path found from (${startRow}, ${startCol}) to (${goalRow}, ${goalCol}).`);
        return false;
      }
	      appendRouteCells(path, "astar", options);
	      setStatus(`A* connected (${startRow}, ${startCol}) to (${goalRow}, ${goalCol}) with ${path.length} cells.`);
	      return true;
	    }

    function routeTailCell(sample = getActiveSample()) {
      return sample.expert_route.length > 0
        ? sample.expert_route[sample.expert_route.length - 1]
        : null;
    }

    function connectRouteTailToCell(cell) {
      const sample = getActiveSample();
      const tail = routeTailCell(sample);
      if (!tail) {
        if (!isTraversableCell(cell[0], cell[1])) {
          setStatus(`Route start must be traversable. ${describeCell(cell[0], cell[1])}`);
          return false;
        }
        appendRouteCells([cell], "manual");
        setStatus(`Route start set at (${cell[0]}, ${cell[1]}).`);
        return true;
      }
      return appendAStarRouteSegment(tail, cell);
    }

    function handleRouteClick(cell) {
      connectRouteTailToCell(cell);
    }

	    function beginRouteFreehand(cell, options = {}) {
	      const sample = getActiveSample();
	      const tail = routeTailCell(sample);
	      if (tail && !sameCell(tail, cell)) {
	        if (!appendAStarRouteSegment(tail, cell, options)) {
	          return false;
	        }
	      } else {
        if (!tail && !isTraversableCell(cell[0], cell[1])) {
          setStatus(`Route start must be traversable. ${describeCell(cell[0], cell[1])}`);
          return false;
        }
	        appendRouteCells([cell], "manual", options);
	      }
	      return true;
	    }

    function nextMustPassOrder(sample) {
      return Math.max(
        0,
        ...sample.hard_constraints
          .filter(constraint => constraint.kind === "must_pass")
          .map(constraint => Number(constraint.order || 0))
      ) + 1;
    }

    function createHardConstraintTag(kind) {
      recordUndoSnapshot(`create ${kind.replace("_", " ")} constraint`);
      const sample = getActiveSample();
      const constraint = {
        constraint_id: `constraint_${Math.random().toString(36).slice(2, 10)}`,
        kind,
        shape: "freeform",
        center: [0, 0],
        radius: 6,
        width: 12,
        height: 12,
        order: kind === "must_pass" ? nextMustPassOrder(sample) : null,
        source: "custom",
        object_id: null,
        scope: { type: "global", from_order: null, to_order: null },
        label: kind === "must_pass" ? "Must Pass" : "Must Avoid",
        initialized: false,
        cells: [],
      };
      sample.hard_constraints.push(constraint);
      sample.updated_at = nowIso();
      selectedConstraintId = constraint.constraint_id;
      selectedSoftConstraintId = null;
      pendingConstraintCorner = null;
      referencePickTarget = null;
      renderConstraintEditor();
      renderSoftConstraintEditor();
      renderChecks();
      drawGrid();
      setStatus(
        `Created a ${kind.replace("_", " ")} tag. Choose Circle, Rectangle, Brush, or Add Box.`
      );
      return constraint;
    }

    function defaultRelativeObjectIds() {
      const objectIds = [...objectInstanceMap().keys()];
      return objectIds.slice(0, 2);
    }

    function createSoftConstraintTag() {
      const sample = getActiveSample();
      const firstObjectId = [...objectInstanceMap().keys()][0] ?? null;
      recordUndoSnapshot("create soft constraint");
      const constraint = {
        constraint_id: `soft_constraint_${Math.random().toString(36).slice(2, 10)}`,
        preference_type: "near_preference",
        shape: "freeform",
        center: [0, 0],
        radius: 6,
        width: 12,
        height: 12,
        object_ids: [],
        reference_regions: [],
        reference_region: firstObjectId === null ? null : {
          mode: "object",
          object_id: firstObjectId,
          cells: cellsForObject(firstObjectId),
        },
        D_ref: null,
        C_smooth_ref: null,
        C_clear_ref: null,
        reference_trajectory: [],
        reference_waypoint_count: 0,
        path_shape_annotation: null,
        scope: { type: "global", from_order: null, to_order: null },
        label: "Soft Constraint",
        initialized: false,
        cells: [],
      };
      sample.soft_constraints.push(constraint);
      sample.updated_at = nowIso();
      selectedSoftConstraintId = constraint.constraint_id;
      selectedConstraintId = null;
      pendingConstraintCorner = null;
      referencePickTarget = null;
      renderConstraintEditor();
      renderSoftConstraintEditor();
      renderChecks();
      drawGrid();
      setStatus(firstObjectId === null
        ? "Created a soft-constraint tag. Choose its type and drawing method."
        : "Created a soft-constraint tag. Choose its type, reference object, and drawing method."
      );
      return constraint;
    }

    function updateFreeformCenter(constraint) {
      if (!constraint.cells.length) {
        return;
      }
      constraint.center = [
        constraint.cells.reduce((sum, cell) => sum + cell[0], 0) / constraint.cells.length,
        constraint.cells.reduce((sum, cell) => sum + cell[1], 0) / constraint.cells.length,
      ];
    }

    function appendCellsToFreeform(constraint, cells) {
      recordGestureUndo("paint constraint region");
      const existing = new Set(
        constraint.cells.map(cell => cellKey(cell[0], cell[1]))
      );
      for (const [row, col] of cells) {
        const key = cellKey(row, col);
        if (!existing.has(key)) {
          constraint.cells.push([row, col]);
          existing.add(key);
        }
      }
      updateFreeformCenter(constraint);
      getActiveSample().updated_at = nowIso();
    }

    function removeCellsFromFreeform(constraint, cells) {
      recordGestureUndo("erase constraint region");
      const toRemove = new Set(cells.map(cell => cellKey(cell[0], cell[1])));
      constraint.cells = constraint.cells.filter(
        cell => !toRemove.has(cellKey(cell[0], cell[1]))
      );
      if (constraint.cells.length > 0) {
        updateFreeformCenter(constraint);
      }
      getActiveSample().updated_at = nowIso();
    }

    function ensureHardFreeform(kind) {
      let constraint = getSelectedConstraint();
      if (!constraint || constraint.kind !== kind) {
        setStatus(`Create and select a ${kind.replace("_", " ")} tag first.`);
        return null;
      }
      if (constraint && constraint.kind === kind && constraint.shape !== "freeform") {
        constraint.cells = constraint.initialized === false
          ? []
          : coveredCellsForConstraint(constraint);
        constraint.shape = "freeform";
        if (constraint.cells.length > 0) {
          updateFreeformCenter(constraint);
        }
      }
      return constraint;
    }

    function ensureSoftFreeform() {
      let constraint = getSelectedSoftConstraint();
      if (!constraint) {
        setStatus("Create and select a soft-constraint tag first.");
        return null;
      }
      if (constraint && constraint.shape !== "freeform") {
        constraint.cells = constraint.initialized === false
          ? []
          : coveredCellsForConstraint(constraint);
        constraint.shape = "freeform";
        if (constraint.cells.length > 0) {
          updateFreeformCenter(constraint);
        }
      }
      return constraint;
    }

    function deleteSelectedHardConstraintIfEmpty() {
      const constraint = getSelectedConstraint();
      if (!constraint || constraint.cells.length > 0) {
        return false;
      }
      const sample = getActiveSample();
      sample.hard_constraints = sample.hard_constraints.filter(
        item => item.constraint_id !== constraint.constraint_id
      );
      if (constraint.kind === "must_pass") {
        const ordered = sample.hard_constraints
          .filter(item => item.kind === "must_pass")
          .sort((first, second) => first.order - second.order);
        ordered.forEach((item, index) => {
          item.order = index + 1;
        });
        refreshScopedConstraintBounds(sample);
      }
      selectedConstraintId = null;
      return true;
    }

    function deleteSelectedSoftConstraintIfEmpty() {
      const constraint = getSelectedSoftConstraint();
      if (!constraint || constraint.cells.length > 0) {
        return false;
      }
      const sample = getActiveSample();
      sample.soft_constraints = sample.soft_constraints.filter(
        item => item.constraint_id !== constraint.constraint_id
      );
      selectedSoftConstraintId = null;
      return true;
    }

    function rectangleCells(first, second) {
      const cells = [];
      const minRow = Math.min(first[0], second[0]);
      const maxRow = Math.max(first[0], second[0]);
      const minCol = Math.min(first[1], second[1]);
      const maxCol = Math.max(first[1], second[1]);
      for (let row = minRow; row <= maxRow; row += 1) {
        for (let col = minCol; col <= maxCol; col += 1) {
          cells.push([row, col]);
        }
      }
      return cells;
    }

    function handleConstraintClick(row, col) {
      if (currentMode.startsWith("soft_")) {
        if (currentMode === "soft_free_erase") {
          const constraint = ensureSoftFreeform();
          if (!constraint) {
            return;
          }
          removeCellsFromFreeform(
            constraint,
            lastPaintCell ? interpolateLine(lastPaintCell, [row, col]) : [[row, col]]
          );
          const deleted = deleteSelectedSoftConstraintIfEmpty();
          renderAfterConstraintPaint({ soft: true });
          if (deleted) {
            setStatus("Removed the last cell and deleted the soft freeform region.");
          } else {
            setStatus(`Removed cell (${row}, ${col}) from the selected soft freeform region.`);
          }
          return;
        }
        if (currentMode === "soft_free_erase_rectangle") {
          if (!pendingConstraintCorner) {
            pendingConstraintCorner = [row, col];
            drawGrid();
            setStatus(`Soft erase box first corner set at (${row}, ${col}). Click the opposite corner.`);
            return;
          }
          const [startRow, startCol] = pendingConstraintCorner;
          const constraint = ensureSoftFreeform();
          if (!constraint) {
            pendingConstraintCorner = null;
            return;
          }
          removeCellsFromFreeform(
            constraint, rectangleCells([startRow, startCol], [row, col])
          );
          const deleted = deleteSelectedSoftConstraintIfEmpty();
          pendingConstraintCorner = null;
          renderSoftConstraintEditor();
          renderChecks();
          drawGrid();
          if (deleted) {
            setStatus("Removed the last cells and deleted the soft region.");
          } else {
            setStatus("Removed a rectangular patch from the selected soft region.");
          }
          return;
        }
        if (currentMode === "soft_brush") {
          const constraint = ensureSoftFreeform();
          if (!constraint) {
            return;
          }
          constraint.initialized = true;
          appendCellsToFreeform(
            constraint,
            lastPaintCell ? interpolateLine(lastPaintCell, [row, col]) : [[row, col]]
          );
          renderAfterConstraintPaint({ soft: true });
          return;
        }
        if (currentMode === "soft_point") {
          const constraint = getSelectedSoftConstraint();
          if (!constraint) {
            setStatus("Create and select a soft-constraint tag first.");
            return;
          }
          recordGestureUndo("place soft constraint");
          Object.assign(constraint, {
            shape: "circle",
            center: [row, col],
            cells: [],
            initialized: true,
          });
          getActiveSample().updated_at = nowIso();
          renderSoftConstraintEditor();
          renderChecks();
          drawGrid();
          setStatus(`Placed the selected soft constraint at (${row}, ${col}).`);
          return;
        }
        if (!pendingConstraintCorner) {
          pendingConstraintCorner = [row, col];
          drawGrid();
          setStatus(`Soft rectangle first corner set at (${row}, ${col}). Click the opposite corner.`);
          return;
        }
        const [startRow, startCol] = pendingConstraintCorner;
        if (currentMode === "soft_free_rectangle") {
          const constraint = ensureSoftFreeform();
          if (!constraint) {
            pendingConstraintCorner = null;
            return;
          }
          constraint.initialized = true;
          appendCellsToFreeform(
            constraint, rectangleCells([startRow, startCol], [row, col])
          );
          pendingConstraintCorner = null;
          renderSoftConstraintEditor();
          renderChecks();
          drawGrid();
          setStatus("Added a rectangular patch to the selected soft freeform region.");
          return;
        }
        const width = Math.abs(col - startCol) + 1;
        const height = Math.abs(row - startRow) + 1;
        const constraint = getSelectedSoftConstraint();
        if (!constraint) {
          pendingConstraintCorner = null;
          setStatus("Create and select a soft-constraint tag first.");
          return;
        }
        recordGestureUndo("place soft constraint");
        Object.assign(constraint, {
          shape: "rectangle",
          center: [(startRow + row) / 2, (startCol + col) / 2],
          width,
          height,
          cells: [],
          initialized: true,
        });
        getActiveSample().updated_at = nowIso();
        pendingConstraintCorner = null;
        renderSoftConstraintEditor();
        renderChecks();
        drawGrid();
        setStatus(`Placed the selected soft rectangle (${width} x ${height}).`);
        return;
      }

      const kind = currentMode.startsWith("must_pass") ? "must_pass" : "must_avoid";
      if (currentMode.endsWith("_free_erase_rectangle")) {
        if (!pendingConstraintCorner) {
          pendingConstraintCorner = [row, col];
          drawGrid();
          setStatus(`Erase box first corner set at (${row}, ${col}). Click the opposite corner.`);
          return;
        }
        const [startRow, startCol] = pendingConstraintCorner;
        const constraint = ensureHardFreeform(kind);
        if (!constraint) {
          pendingConstraintCorner = null;
          return;
        }
        removeCellsFromFreeform(
          constraint, rectangleCells([startRow, startCol], [row, col])
        );
        const deleted = deleteSelectedHardConstraintIfEmpty();
        pendingConstraintCorner = null;
        renderConstraintEditor();
        renderChecks();
        drawGrid();
        if (deleted) {
          setStatus(`Removed the last cells and deleted the ${kind.replace("_", " ")} region.`);
        } else {
          setStatus(`Removed a rectangular patch from the selected ${kind.replace("_", " ")} region.`);
        }
        return;
      }
      if (currentMode.endsWith("_free_erase")) {
        const constraint = ensureHardFreeform(kind);
        if (!constraint) {
          return;
        }
        removeCellsFromFreeform(
          constraint,
          lastPaintCell ? interpolateLine(lastPaintCell, [row, col]) : [[row, col]]
        );
        const deleted = deleteSelectedHardConstraintIfEmpty();
        renderAfterConstraintPaint({ hard: true });
        if (deleted) {
          setStatus(`Removed the last cell and deleted the ${kind.replace("_", " ")} freeform region.`);
        } else {
          setStatus(`Removed cell (${row}, ${col}) from the selected ${kind.replace("_", " ")} freeform region.`);
        }
        return;
      }
      if (currentMode.endsWith("_brush")) {
        const constraint = ensureHardFreeform(kind);
        if (!constraint) {
          return;
        }
        constraint.initialized = true;
        appendCellsToFreeform(
          constraint,
          lastPaintCell ? interpolateLine(lastPaintCell, [row, col]) : [[row, col]]
        );
        renderAfterConstraintPaint({ hard: true });
        return;
      }
      if (currentMode.endsWith("_point")) {
        const constraint = getSelectedConstraint();
        if (!constraint || constraint.kind !== kind) {
          setStatus(`Create and select a ${kind.replace("_", " ")} tag first.`);
          return;
        }
        recordGestureUndo(`place ${kind.replace("_", " ")} constraint`);
        Object.assign(constraint, {
          shape: "circle",
          center: [row, col],
          cells: [],
          initialized: true,
        });
        getActiveSample().updated_at = nowIso();
        renderConstraintEditor();
        renderChecks();
        drawGrid();
        setStatus(`Placed the selected ${kind.replace("_", " ")} circle at (${row}, ${col}).`);
        return;
      }

      if (!pendingConstraintCorner) {
        pendingConstraintCorner = [row, col];
        drawGrid();
        setStatus(`Rectangle first corner set at (${row}, ${col}). Click the opposite corner.`);
        return;
      }

      const [startRow, startCol] = pendingConstraintCorner;
      if (currentMode.endsWith("_free_rectangle")) {
        const constraint = ensureHardFreeform(kind);
        if (!constraint) {
          pendingConstraintCorner = null;
          return;
        }
        constraint.initialized = true;
        appendCellsToFreeform(
          constraint, rectangleCells([startRow, startCol], [row, col])
        );
        pendingConstraintCorner = null;
        renderConstraintEditor();
        renderChecks();
        drawGrid();
        setStatus(`Added a rectangular patch to the selected ${kind.replace("_", " ")} freeform region.`);
        return;
      }
      const width = Math.abs(col - startCol) + 1;
      const height = Math.abs(row - startRow) + 1;
      const center = [(startRow + row) / 2, (startCol + col) / 2];
      const constraint = getSelectedConstraint();
      if (!constraint || constraint.kind !== kind) {
        pendingConstraintCorner = null;
        setStatus(`Create and select a ${kind.replace("_", " ")} tag first.`);
        return;
      }
      recordGestureUndo(`place ${kind.replace("_", " ")} constraint`);
      Object.assign(constraint, {
        shape: "rectangle",
        center,
        width,
        height,
        cells: [],
        initialized: true,
      });
      getActiveSample().updated_at = nowIso();
      pendingConstraintCorner = null;
      renderConstraintEditor();
      renderChecks();
      drawGrid();
      setStatus(`Placed the selected ${kind.replace("_", " ")} rectangle (${width} x ${height}).`);
    }

    function handlePaintAtCell(cell) {
      const [row, col] = cell;
      if (
        currentMode.startsWith("must_pass") ||
        currentMode.startsWith("must_avoid") ||
        currentMode.startsWith("soft_")
      ) {
        handleConstraintClick(row, col);
        return;
      }
	      if (currentMode === "route_erase") {
	        eraseRouteCells([[row, col]], isPainting
	          ? { updateChecks: false, deferDraw: true }
	          : {}
	        );
	      }
	    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Request failed.");
      }
      return await response.json();
    }

    function buildSavePayload() {
      persistFormIntoActiveSample();
      const sample = getActiveSample();
      if (!normalizeDifficultyLevel(sample.difficulty_level)) {
        throw new Error("Select an instruction difficulty before saving.");
      }
      const collisionViolations = routeCollisionViolations(sample.expert_route);
      if (collisionViolations.length > 0) {
        const first = collisionViolations[0];
        throw new Error(
          `Route is not collision-free: ${collisionViolations.length} cell(s) are non-traversable, first at (${first[0]}, ${first[1]}).`
        );
      }
      if (sample.hard_constraints.some(constraint => constraint.initialized === false)) {
        throw new Error("Every hard-constraint tag must be drawn or deleted before saving.");
      }
      if (sample.soft_constraints.some(constraint => constraint.initialized === false)) {
        throw new Error("Every soft-constraint tag must be drawn or deleted before saving.");
      }
      const emptySoftFreeform = sample.soft_constraints.find(constraint => (
        constraint.shape === "freeform" &&
        !isRegionlessTrajectoryMetricPreference(constraint.preference_type) &&
        (!Array.isArray(constraint.cells) || constraint.cells.length === 0)
      ));
      if (emptySoftFreeform) {
        const index = sample.soft_constraints.indexOf(emptySoftFreeform) + 1;
        const label = emptySoftFreeform.label || emptySoftFreeform.preference_type;
        throw new Error(
          `Soft constraint #${index} (${label}, ${emptySoftFreeform.preference_type}) ` +
          "is freeform but has no painted cells. Select it and draw with Add Brush/Add Box, or delete it."
        );
      }
	      return {
	        map_id: currentMapId,
	        template_instruction_id: activeTemplateId,
	        active_sample_id: activeSampleId,
	        samples: [sample],
	      };
	    }

    function selectInitialTemplate(preferredTemplateId = null) {
      const visibleTemplates = orderedFilteredTemplates();
      let template = visibleTemplates.find(
        item => item.template_instruction_id === preferredTemplateId
      );
      if (!template) {
        template = visibleTemplates[0] || templates.find(
          item => item.template_instruction_id === preferredTemplateId
        ) || templates[0];
      }
      if (!template) {
        activeTemplateId = null;
        activeSampleId = samples[0]?.sample_id || null;
        return;
      }
      activeTemplateId = template.template_instruction_id;
      selectedTemplateObjectId = null;
      selectedConstraintId = null;
      selectedSoftConstraintId = null;
      pendingConstraintCorner = null;
      referencePickTarget = null;
      ensureSampleForTemplate(template);
    }

    function applyServerState(state, preferredTemplateId = null) {
      GRID_SIZE = state.grid_size;
      currentLegends = state.legends;
      mapState = cloneMapState(state.map_state);
      resetObjectVisualizationCaches();
      mapSummaries = state.map_summaries.slice();
      globalDifficultyTotals = normalizeGlobalDifficultyTotals(state.global_difficulty_totals);
      mapDifficultyTotals = normalizeDifficultyCounts(state.map_difficulty_totals);
      globalConstraintUsage = normalizeGlobalConstraintUsage(state.global_constraint_usage);
      currentMapId = state.current_map_id;
      paths = { ...state.paths };
      samples = cloneSamples(state.annotation_state.samples);
      activeSampleId = state.annotation_state.active_sample_id;
      templates = cloneTemplates(state.template_instruction_state.templates);
      templateDistanceRule = { ...state.template_instruction_state.distance_rule };
      selectedTemplateObjectId = null;
      selectedConstraintId = null;
      selectedSoftConstraintId = null;
      pendingConstraintCorner = null;
      referencePickTarget = null;
      cancelRouteSegment();
      undoHistory = [];
      hideMapInspector();
      selectInitialTemplate(preferredTemplateId);
      astarStartCell = null;
      renderMapSelect();
      renderGlobalDifficultySummary();
      renderGlobalConstraintUsage();
      renderMapInfo();
      renderPaths();
      renderSampleList();
      syncFormFromActiveSample();
      resizeCanvas();
      focusActiveAnnotationSoon();
    }

    async function saveInstruction() {
      const template = getActiveTemplate();
      if (!template) {
        throw new Error("Select a template instruction first.");
      }
      const savedTemplateId = template.template_instruction_id;
      const shouldStayOnSavedTemplate = template.status === "abandoned";
      setStatus("Saving instruction...");
      const payload = await postJson("/save_instruction", buildSavePayload());
      if (shouldStayOnSavedTemplate) {
        statusFilter.value = "labeled";
        difficultyFilter.value = "";
      }
      applyServerState(
        payload.state,
        shouldStayOnSavedTemplate
          ? savedTemplateId
          : payload.next_template_instruction_id
      );
      setStatus(payload.message);
    }

    async function abandonTemplate() {
      const template = getActiveTemplate();
      if (!template) {
        throw new Error("Select a template instruction first.");
      }
      setStatus("Abandoning template...");
      const payload = await postJson("/abandon_template", {
        map_id: currentMapId,
        template_instruction_id: template.template_instruction_id,
      });
      applyServerState(payload.state, payload.next_template_instruction_id);
      setStatus(payload.message);
    }

    async function openSelectedMap() {
      const targetId = mapSelect.value;
      if (!targetId) {
        return;
      }
      setStatus(`Opening map '${targetId}'...`);
      const payload = await postJson("/open", { map_id: targetId });
      applyServerState(payload.state);
      setStatus(payload.message);
    }

    canvas.addEventListener("mousedown", (event) => {
      if (event.button !== 0) {
        return;
      }
      gestureUndoRecorded = false;
      const cell = eventToCell(event);
      if (!cell) {
        return;
      }
      if (currentMode === "route_adjust") {
        const selectedIndex = nearestRoutePointIndex(cell);
        if (selectedIndex < 0) {
          setStatus("Click closer to an existing route waypoint to adjust it.");
          return;
        }
        const selectedPoint = getActiveSample().expert_route[selectedIndex];
        isPainting = true;
        routeAdjustIndex = selectedIndex;
        routeAdjustTargetCell = [selectedPoint[0], selectedPoint[1]];
        lastPaintCell = cell;
        drawGrid();
        setStatus(`Selected route waypoint ${selectedIndex + 1}. Drag it to a new traversable cell.`);
        return;
      }
      if (currentMode === "route_draw") {
        isPainting = true;
        routeGestureStartCell = cell;
        routeGestureDragged = false;
        cancelRouteSegment();
        lastPaintCell = cell;
        return;
      }
      isPainting = (
        currentMode === "route_erase" ||
        currentMode.endsWith("_brush")
      );
      lastPaintCell = cell;
      handlePaintAtCell(cell);
      if (!isPainting) {
        lastPaintCell = null;
      }
    });

    canvas.addEventListener("mousemove", (event) => {
      const cell = eventToCell(event);
      if (!cell) {
        return;
      }

      if (!isPainting) {
        setStatus(`${describeCell(cell[0], cell[1])} | Mode=${MODES[currentMode].label}`);
        return;
      }

      if (sameCell(lastPaintCell, cell)) {
        return;
      }
      if (currentMode === "route_adjust") {
        routeAdjustTargetCell = cell;
        lastPaintCell = cell;
	        scheduleDrawGrid();
	        setStatus(`Adjust target: (${cell[0]}, ${cell[1]}). Release to reconnect with A*.`);
	        return;
	      }
	      if (currentMode === "route_draw") {
	        if (!routeGestureDragged) {
	          routeGestureDragged = true;
	          beginRouteSegment("freehand");
	          if (!beginRouteFreehand(routeGestureStartCell, {
	            updateChecks: false,
	            deferDraw: true,
	          })) {
	            cancelRouteSegment();
	            isPainting = false;
	            routeGestureStartCell = null;
            lastPaintCell = null;
	            return;
	          }
	        }
	        appendRouteCells(interpolateLine(lastPaintCell, cell).slice(1), "manual", {
	          updateChecks: false,
	          deferDraw: true,
	        });
	        lastPaintCell = cell;
	        return;
	      }
      handlePaintAtCell(cell);
      lastPaintCell = cell;
    });

	    window.addEventListener("mouseup", () => {
	      const wasRouteAdjust = isPainting && currentMode === "route_adjust";
	      const wasRouteDraw = isPainting && currentMode === "route_draw";
	      const wasRouteErase = isPainting && currentMode === "route_erase";
	      const wasConstraintPaint = isPainting && isContinuousConstraintPaintMode();
	      if (
	        isPainting &&
        currentMode === "route_draw" &&
        routeGestureStartCell &&
        !routeGestureDragged
      ) {
        handleRouteClick(routeGestureStartCell);
      }
      if (
        isPainting &&
        currentMode === "route_adjust" &&
        routeAdjustIndex !== null &&
        routeAdjustTargetCell
      ) {
        replaceRouteWaypointWithAStar(routeAdjustIndex, routeAdjustTargetCell);
      }
	      if (
	        isPainting &&
	        currentMode === "route_draw" &&
        routeGestureDragged
      ) {
        finishRouteSegment();
      } else {
        cancelRouteSegment();
      }
      isPainting = false;
      lastPaintCell = null;
      routeGestureStartCell = null;
      routeGestureDragged = false;
	      routeAdjustIndex = null;
	      routeAdjustTargetCell = null;
	      if (wasRouteDraw || wasRouteErase) {
	        renderStartPoseInfo();
	        renderChecks();
	        flushDrawGrid();
	      } else if (wasRouteAdjust) {
	        flushDrawGrid();
	      } else if (wasConstraintPaint) {
	        finishContinuousConstraintPaint();
	      }
	      gestureUndoRecorded = false;
	    });

    canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      const cell = eventToCell(event);
      if (!cell) {
        return;
      }
      if (referencePickTarget) {
        pickReferenceObjectAt(cell[0], cell[1]);
        return;
      }
      inspectCell(cell[0], cell[1]);
    });

    document.getElementById("saveBtn").addEventListener("click", async () => {
      try {
        await saveInstruction();
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

    document.getElementById("abandonBtn").addEventListener("click", async () => {
      try {
        await abandonTemplate();
      } catch (error) {
        setStatus(error.message);
      }
    });

    objectCountFilter.addEventListener("change", () => {
      persistFormIntoActiveSample();
      selectInitialTemplate();
      renderSampleList();
      syncFormFromActiveSample();
      drawGrid();
      focusActiveAnnotationSoon();
    });

    statusFilter.addEventListener("change", () => {
      persistFormIntoActiveSample();
      selectInitialTemplate();
      renderSampleList();
      syncFormFromActiveSample();
      drawGrid();
      focusActiveAnnotationSoon();
    });

    difficultyFilter.addEventListener("change", () => {
      persistFormIntoActiveSample();
      selectInitialTemplate();
      renderSampleList();
      syncFormFromActiveSample();
      drawGrid();
      focusActiveAnnotationSoon();
    });

    labeledInstructionSelect.addEventListener("change", () => {
      const templateId = labeledInstructionSelect.value;
      if (!templateId) {
        return;
      }
      const template = orderedFilteredTemplates().find(
        item => item.template_instruction_id === templateId
      );
      if (!template) {
        setStatus("Selected instruction is not available under the current filters.");
        renderSampleList();
        return;
      }
      const instructionNumber = templateInstructionNumber(template);
      selectTemplate(
        template,
        Number.isFinite(instructionNumber)
          ? `Opened instruction_${String(instructionNumber).padStart(6, "0")}.`
          : "Opened labeled instruction."
      );
    });

    focusAnnotationBtn.addEventListener("click", () => {
      focusActiveAnnotation({ report: true, smooth: true });
    });

    document.getElementById("clearRouteBtn").addEventListener("click", () => {
      const sample = getActiveSample();
      recordUndoSnapshot("clear expert trajectory");
      sample.expert_route = [];
      clearAStarHistory(sample);
      syncStartPoseFromRoute(sample);
      sample.updated_at = nowIso();
      renderStartPoseInfo();
      renderChecks();
      drawGrid();
      setStatus("Cleared the current route. Save annotations to persist it.");
    });

    undoLastAStarBtn.addEventListener("click", () => {
      undoLastRouteSegment();
    });

    document.getElementById("resetAStarBtn").addEventListener("click", () => {
      astarStartCell = null;
      drawGrid();
      setStatus("Reset the pending A* start cell.");
    });

    zoomSlider.addEventListener("input", () => {
      zoom = Number(zoomSlider.value);
      resizeCanvas();
      focusActiveAnnotationSoon();
      postJson("/settings", { zoom }).catch(error => setStatus(error.message));
    });

    gridToggle.addEventListener("change", () => {
      showGrid = gridToggle.checked;
      drawGrid();
    });

    roomToggle.addEventListener("change", () => {
      showRooms = roomToggle.checked;
      drawGrid();
    });

    objectToggle.addEventListener("change", () => {
      showObjects = objectToggle.checked;
      drawGrid();
    });

    smallObjectNameToggle.addEventListener("change", () => {
      showSmallObjectNames = smallObjectNameToggle.checked;
      drawGrid();
    });

    hardConstraintToggle.addEventListener("change", () => {
      showHardConstraints = hardConstraintToggle.checked;
      drawGrid();
    });

    softConstraintToggle.addEventListener("change", () => {
      showSoftConstraints = softConstraintToggle.checked;
      drawGrid();
    });

    createMustPassBtn.addEventListener("click", () => {
      createHardConstraintTag("must_pass");
    });

    createMustAvoidBtn.addEventListener("click", () => {
      createHardConstraintTag("must_avoid");
    });

    createSoftConstraintBtn.addEventListener("click", () => {
      createSoftConstraintTag();
    });

    for (const [listEl, bucket] of [
      [sequenceConstraintListEl, "sequence"],
      [globalConstraintListEl, "global"],
    ]) {
      listEl.addEventListener("dragover", event => {
        event.preventDefault();
        listEl.classList.add("drop-target");
      });
      listEl.addEventListener("dragleave", () => {
        listEl.classList.remove("drop-target");
      });
      listEl.addEventListener("drop", event => {
        event.preventDefault();
        listEl.classList.remove("drop-target");
        if (!draggedConstraintRef) {
          return;
        }
        moveConstraintRefToBucket(draggedConstraintRef, bucket);
      });
    }

    function activateConstraintDrawing(mode) {
      if (mode.startsWith("soft_")) {
        if (!getSelectedSoftConstraint()) {
          setStatus("Create or select a soft-constraint tag first.");
          return;
        }
      } else {
        const expectedKind = mode.startsWith("must_pass")
          ? "must_pass"
          : "must_avoid";
        const constraint = getSelectedConstraint();
        if (!constraint || constraint.kind !== expectedKind) {
          setStatus(`Create or select a ${expectedKind.replace("_", " ")} tag first.`);
          return;
        }
      }
      setMode(mode);
    }

    function activateSelectedHardDrawing(action) {
      const constraint = getSelectedConstraint();
      if (!constraint) {
        setStatus("Create or select a hard-region tag first.");
        return;
      }
      const modeByAction = {
        circle: `${constraint.kind}_point`,
        rectangle: `${constraint.kind}_rectangle`,
        add_brush: `${constraint.kind}_brush`,
        add_box: `${constraint.kind}_free_rectangle`,
        erase_brush: `${constraint.kind}_free_erase`,
        erase_box: `${constraint.kind}_free_erase_rectangle`,
      };
      activateConstraintDrawing(modeByAction[action]);
    }

    for (const [button, action] of [
      [addPassCircleBtn, "circle"],
      [addPassRectangleBtn, "rectangle"],
      [paintPassBrushBtn, "add_brush"],
      [addPassBoxBtn, "add_box"],
    ]) {
      button.addEventListener("click", () => activateSelectedHardDrawing(action));
    }

    for (const [button, mode] of [
      [addSoftCircleBtn, "soft_point"],
      [addSoftRectangleBtn, "soft_rectangle"],
      [paintSoftBrushBtn, "soft_brush"],
      [addSoftBoxBtn, "soft_free_rectangle"],
    ]) {
      button.addEventListener("click", () => activateConstraintDrawing(mode));
    }

    eraseHardFreeformBtn.addEventListener("click", () => {
      activateSelectedHardDrawing("erase_brush");
    });

    eraseHardBoxBtn.addEventListener("click", () => {
      activateSelectedHardDrawing("erase_box");
    });

    eraseSoftFreeformBtn.addEventListener("click", () => {
      const constraint = getSelectedSoftConstraint();
      if (!constraint) {
        setStatus("Create or select a soft region first.");
        return;
      }
      setMode("soft_free_erase");
    });

    eraseSoftBoxBtn.addEventListener("click", () => {
      const constraint = getSelectedSoftConstraint();
      if (!constraint) {
        setStatus("Create or select a soft region first.");
        return;
      }
      setMode("soft_free_erase_rectangle");
    });

    pickSoftReferenceBtn.addEventListener("click", () => {
      const constraint = getSelectedSoftConstraint();
      if (
        !constraint ||
        constraint.preference_type === "relative_preference" ||
        !usesObjectReferenceRegion(constraint.preference_type)
      ) {
        setStatus("Select a near/far soft constraint first.");
        return;
      }
      referencePickTarget = "single";
      setStatus("Right-click an object on the map to use it as the reference.");
    });

    pickRelativeObjectABtn.addEventListener("click", () => {
      const constraint = getSelectedSoftConstraint();
      if (!constraint || constraint.preference_type !== "relative_preference") {
        setStatus("Select a relative-preference soft constraint first.");
        return;
      }
      referencePickTarget = "relative_a";
      setStatus("Right-click an object on the map to select relative reference A.");
    });

    pickRelativeObjectBBtn.addEventListener("click", () => {
      const constraint = getSelectedSoftConstraint();
      if (!constraint || constraint.preference_type !== "relative_preference") {
        setStatus("Select a relative-preference soft constraint first.");
        return;
      }
      referencePickTarget = "relative_b";
      setStatus("Right-click an object on the map to select relative reference B.");
    });

    sampleNameInput.addEventListener("input", () => {
      persistFormIntoActiveSample();
      renderSampleList();
    });

    instructionInput.addEventListener("input", () => {
      persistFormIntoActiveSample();
    });

    difficultyLevelInput.addEventListener("change", () => {
      recordUndoSnapshot("change instruction difficulty");
      persistFormIntoActiveSample();
      renderSampleList();
    });

    createdByInput.addEventListener("input", () => {
      sessionCreatedBy = createdByInput.value.trim();
      localStorage.setItem("sempathbench_created_by", sessionCreatedBy);
      for (const sample of samples) {
        sample.created_by = sessionCreatedBy;
      }
    });

    constraintShapeInput.addEventListener("change", () => {
      const constraint = getSelectedConstraint();
      if (!constraint) {
        return;
      }
      recordUndoSnapshot("change hard-constraint shape");
      if (constraintShapeInput.value === "freeform" && constraint.shape !== "freeform") {
        constraint.cells = coveredCellsForConstraint(constraint);
        updateFreeformCenter(constraint);
      }
      constraint.shape = constraintShapeInput.value;
      getActiveSample().updated_at = nowIso();
      renderConstraintEditor();
      renderChecks();
      drawGrid();
    });

    for (const [input, field] of [
      [constraintRadiusInput, "radius"],
      [constraintWidthInput, "width"],
      [constraintHeightInput, "height"],
    ]) {
      input.addEventListener("input", () => {
        const constraint = getSelectedConstraint();
        const value = Number(input.value);
        if (!constraint || !Number.isFinite(value) || value <= 0) {
          return;
        }
        recordUndoSnapshot(`change hard-constraint ${field}`);
        constraint[field] = value;
        getActiveSample().updated_at = nowIso();
        renderChecks();
        drawGrid();
      });
    }

    constraintLabelInput.addEventListener("input", () => {
      const constraint = getSelectedConstraint();
      if (!constraint) {
        return;
      }
      constraint.label = constraintLabelInput.value.trim() || (
        constraint.kind === "must_avoid" ? "Must Avoid" : "Must Pass"
      );
      getActiveSample().updated_at = nowIso();
      renderConstraintEditor();
      drawGrid();
    });

    constraintScopeInput.addEventListener("change", () => {
      const constraint = getSelectedConstraint();
      if (!constraint || constraint.kind !== "must_avoid") {
        return;
      }
      recordUndoSnapshot("change must-avoid scope");
      constraint.scope = inputValueToScope(constraintScopeInput.value);
      getActiveSample().updated_at = nowIso();
      renderConstraintEditor();
      renderChecks();
      drawGrid();
    });

    deleteConstraintBtn.addEventListener("click", () => {
      const sample = getActiveSample();
      if (!selectedConstraintId) {
        return;
      }
      recordUndoSnapshot("delete hard constraint");
      sample.hard_constraints = sample.hard_constraints.filter(
        constraint => constraint.constraint_id !== selectedConstraintId
      );
      const orderedMustPass = sample.hard_constraints
        .filter(constraint => constraint.kind === "must_pass")
        .sort((first, second) => first.order - second.order);
      for (let index = 0; index < orderedMustPass.length; index += 1) {
        orderedMustPass[index].order = index + 1;
      }
      refreshScopedConstraintBounds(sample);
      selectedConstraintId = null;
      sample.updated_at = nowIso();
      renderConstraintEditor();
      renderChecks();
      drawGrid();
      setStatus("Deleted the selected hard constraint.");
    });

    softPreferenceTypeInput.addEventListener("change", () => {
      const constraint = getSelectedSoftConstraint();
      if (!constraint) {
        return;
      }
      recordUndoSnapshot("change soft-preference type");
      const previousType = constraint.preference_type;
      constraint.preference_type = softPreferenceTypeInput.value;
      if (
        previousType === "path_shape_preference" &&
        constraint.preference_type !== "path_shape_preference"
      ) {
        constraint.path_shape_annotation = null;
      }
      if (constraint.preference_type === "relative_preference") {
        if (isRegionlessTrajectoryMetricPreference(previousType)) {
          constraint.initialized = false;
          constraint.shape = "freeform";
          constraint.cells = [];
        }
        constraint.object_ids = defaultRelativeObjectIds();
        constraint.reference_regions = constraint.object_ids.map(objectId => ({
          object_id: objectId,
          cells: cellsForObject(objectId),
        }));
        constraint.reference_region = null;
        constraint.C_smooth_ref = null;
        constraint.C_clear_ref = null;
        constraint.reference_trajectory = [];
      } else if (isRegionlessTrajectoryMetricPreference(constraint.preference_type)) {
        constraint.object_ids = [];
        constraint.reference_regions = [];
        constraint.reference_region = null;
        constraint.shape = "freeform";
        constraint.cells = [];
        constraint.initialized = true;
        constraint.D_ref = null;
        constraint.C_smooth_ref = null;
        constraint.C_clear_ref = null;
        constraint.reference_trajectory = [];
        constraint.reference_waypoint_count = 0;
        if (!constraint.label || constraint.label === "Soft Constraint") {
          constraint.label = constraint.preference_type === "clearance"
            ? "Clearance"
            : constraint.preference_type === "path_shape_preference"
              ? "Path Shape"
              : "Move Smoothness";
        }
      } else {
        if (isRegionlessTrajectoryMetricPreference(previousType)) {
          constraint.initialized = false;
          constraint.shape = "freeform";
          constraint.cells = [];
        }
        constraint.object_ids = [];
        constraint.reference_regions = [];
        constraint.reference_region = null;
        constraint.D_ref = null;
        constraint.C_smooth_ref = null;
        constraint.C_clear_ref = null;
        constraint.reference_trajectory = [];
        constraint.reference_waypoint_count = 0;
        if (
          !constraint.label ||
          ["Soft Constraint", "Clearance", "Move Smoothness"].includes(constraint.label)
        ) {
          constraint.label = constraint.preference_type === "path_shape_preference"
            ? "Path Shape"
            : "Soft Constraint";
        }
        if (
          usesObjectReferenceRegion(constraint.preference_type) &&
          (!constraint.reference_region || constraint.reference_region.mode !== "object")
        ) {
          const firstObjectId = [...objectInstanceMap().keys()][0] ?? null;
          constraint.reference_region = firstObjectId === null ? null : {
            mode: "object",
            object_id: firstObjectId,
            cells: firstObjectId === null ? [] : cellsForObject(firstObjectId),
          };
        }
      }
      referencePickTarget = null;
      getActiveSample().updated_at = nowIso();
      renderSoftConstraintEditor();
      renderChecks();
      drawGrid();
    });

    softReferenceObjectInput.addEventListener("change", () => {
      const constraint = getSelectedSoftConstraint();
      if (
        !constraint ||
        constraint.preference_type === "relative_preference" ||
        !usesObjectReferenceRegion(constraint.preference_type)
      ) {
        return;
      }
      recordUndoSnapshot("change soft reference object");
      const objectId = Number(softReferenceObjectInput.value);
      constraint.reference_region = {
        mode: "object",
        object_id: objectId,
        cells: cellsForObject(objectId),
      };
      getActiveSample().updated_at = nowIso();
      renderSoftConstraintEditor();
      renderChecks();
      drawGrid();
    });

    softConstraintShapeInput.addEventListener("change", () => {
      const constraint = getSelectedSoftConstraint();
      if (!constraint) {
        return;
      }
      recordUndoSnapshot("change soft-constraint shape");
      if (
        softConstraintShapeInput.value === "freeform" &&
        constraint.shape !== "freeform"
      ) {
        constraint.cells = coveredCellsForConstraint(constraint);
        updateFreeformCenter(constraint);
      }
      constraint.shape = softConstraintShapeInput.value;
      getActiveSample().updated_at = nowIso();
      renderSoftConstraintEditor();
      renderChecks();
      drawGrid();
    });

    for (const [input, field] of [
      [softConstraintRadiusInput, "radius"],
      [softConstraintWidthInput, "width"],
      [softConstraintHeightInput, "height"],
    ]) {
      input.addEventListener("input", () => {
        const constraint = getSelectedSoftConstraint();
        const value = Number(input.value);
        if (!constraint || !Number.isFinite(value) || value <= 0) {
          return;
        }
        recordUndoSnapshot(`change soft-constraint ${field}`);
        constraint[field] = value;
        getActiveSample().updated_at = nowIso();
        renderChecks();
        drawGrid();
      });
    }

    softConstraintLabelInput.addEventListener("input", () => {
      const constraint = getSelectedSoftConstraint();
      if (!constraint) {
        return;
      }
      constraint.label = softConstraintLabelInput.value.trim() || "Soft Constraint";
      getActiveSample().updated_at = nowIso();
      renderSoftConstraintEditor();
      drawGrid();
    });

    softConstraintScopeInput.addEventListener("change", () => {
      const constraint = getSelectedSoftConstraint();
      if (!constraint) {
        return;
      }
      recordUndoSnapshot("change soft-constraint scope");
      constraint.scope = inputValueToScope(softConstraintScopeInput.value);
      getActiveSample().updated_at = nowIso();
      renderSoftConstraintEditor();
      renderChecks();
      drawGrid();
    });

    function updateRelativeObjects() {
      const constraint = getSelectedSoftConstraint();
      if (!constraint || constraint.preference_type !== "relative_preference") {
        return;
      }
      const firstId = Number(relativeObjectAInput.value);
      const secondId = Number(relativeObjectBInput.value);
      if (firstId === secondId) {
        setStatus("Relative preference requires two different objects.");
        return;
      }
      recordUndoSnapshot("change ordered relative reference objects");
      constraint.object_ids = [firstId, secondId];
      constraint.reference_regions = [firstId, secondId].map(objectId => ({
        object_id: objectId,
        cells: cellsForObject(objectId),
      }));
      getActiveSample().updated_at = nowIso();
      renderChecks();
    }

    relativeObjectAInput.addEventListener("change", updateRelativeObjects);
    relativeObjectBInput.addEventListener("change", updateRelativeObjects);

    deleteSoftConstraintBtn.addEventListener("click", () => {
      const sample = getActiveSample();
      if (!selectedSoftConstraintId) {
        return;
      }
      recordUndoSnapshot("delete soft constraint");
      sample.soft_constraints = sample.soft_constraints.filter(
        constraint => constraint.constraint_id !== selectedSoftConstraintId
      );
      selectedSoftConstraintId = null;
      sample.updated_at = nowIso();
      renderSoftConstraintEditor();
      renderChecks();
      drawGrid();
      setStatus("Deleted the selected soft constraint.");
    });

    feasibleInput.addEventListener("change", () => {
      recordUndoSnapshot("change sample feasibility");
      persistFormIntoActiveSample();
      renderChecks();
      renderSampleList();
    });

    document.addEventListener("keydown", async (event) => {
      const targetTag = event.target.tagName;
      if (
        targetTag === "INPUT" ||
        targetTag === "TEXTAREA" ||
        targetTag === "SELECT" ||
        event.target.isContentEditable
      ) {
        return;
      }
      if (
        (event.ctrlKey || event.metaKey) &&
        !event.shiftKey &&
        event.key.toLowerCase() === "z"
      ) {
        event.preventDefault();
        undoLastOperation();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        try {
          await saveInstruction();
        } catch (error) {
          setStatus(error.message);
        }
        return;
      }
      if (!event.ctrlKey && !event.metaKey && !event.altKey) {
        const shortcutIndex = Number(event.key) - 1;
        const shortcutMode = MODE_ORDER[shortcutIndex];
        if (shortcutMode) {
          event.preventDefault();
          setMode(shortcutMode);
        }
      }
    });

    renderMapSelect();
    renderGlobalDifficultySummary();
    renderGlobalConstraintUsage();
    renderMapInfo();
    renderPaths();
    renderModeList();
    zoomSlider.value = String(zoom);
    createdByInput.value = sessionCreatedBy;
    selectInitialTemplate();
    renderSampleList();
    syncFormFromActiveSample();
    resizeCanvas();
    focusActiveAnnotationSoon();
    setStatus("Editor ready. Use mode 1/2/3 or the buttons on the left.");
  </script>
</body>
</html>
"""


class InstructionAnnotatorHandler(BaseHTTPRequestHandler):
    def _build_index_html(self) -> bytes:
        available = list_map_ids()
        map_id = DEFAULT_MAP_KEY if DEFAULT_MAP_KEY in available else available[0]
        html = HTML_PAGE.replace(
            "__INITIAL_STATE__",
            json.dumps(build_client_state(map_id), ensure_ascii=False),
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
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send_bytes(self._build_index_html(), "text/html; charset=utf-8")
            return
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
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
            if self.path == "/open":
                map_id = payload.get("map_id")
                if not isinstance(map_id, str) or not map_id.strip():
                    raise ValueError("map_id must be a non-empty string.")
                current_state = build_client_state(map_id)
                self._send_json(
                    {
                        "message": f"Opened map '{current_state['current_map_id']}'.",
                        "state": current_state,
                    }
                )
                return

            if self.path == "/settings":
                saved_arguments = save_arguments(payload)
                self._send_json(
                    {
                        "message": "Saved annotator settings.",
                        "arguments": saved_arguments,
                    }
                )
                return

            if self.path == "/save_instruction":
                map_id = payload.get("map_id")
                template_instruction_id = payload.get("template_instruction_id")
                if not isinstance(map_id, str) or not map_id.strip():
                    raise ValueError("map_id must be a non-empty string.")
                if (
                    not isinstance(template_instruction_id, str)
                    or not template_instruction_id.strip()
                ):
                    raise ValueError(
                        "template_instruction_id must be a non-empty string."
                    )

                current_map_state = load_map_state(map_id)
                grid_size = int(current_map_state["grid_size"])
                resolved_map_id = str(current_map_state["map_key"])
                template_bundle = load_template_instruction_bundle(resolved_map_id)
                template_by_id = {
                    template["template_instruction_id"]: template
                    for template in template_bundle["templates"]  # type: ignore[union-attr]
                }
                if template_instruction_id not in template_by_id:
                    raise ValueError(
                        f"Unknown template_instruction_id: {template_instruction_id!r}"
                    )

                active_sample = validate_active_sample_from_payload(
                    payload,
                    resolved_map_id,
                    grid_size,
                )
                if (
                    active_sample["template_instruction_id"]
                    != template_instruction_id
                ):
                    raise ValueError(
                        "The active sample does not match the selected template instruction."
                    )
                if not str(active_sample["instruction"]).strip():
                    raise ValueError("Instruction text cannot be empty.")
                if not active_sample["difficulty_level"]:
                    raise ValueError(
                        "Instruction difficulty must be easy, hard, or extreme."
                    )
                if not active_sample["expert_route"]:
                    raise ValueError("Human expert trajectory cannot be empty.")
                collision_violations = route_collision_violations_for_map(
                    active_sample,
                    current_map_state,
                )
                if collision_violations:
                    first_row, first_col = collision_violations[0]
                    raise ValueError(
                        "Human expert trajectory must be collision-free: "
                        f"{len(collision_violations)} non-traversable cell(s), "
                        f"first at ({first_row}, {first_col})."
                    )
                valid_object_ids = {
                    int(item["id"])
                    for item in current_map_state["object_instances"]  # type: ignore[union-attr]
                }
                for constraint in active_sample["soft_constraints"]:  # type: ignore[union-attr]
                    if constraint["preference_type"] == "relative_preference":
                        if not set(constraint["object_ids"]).issubset(
                            valid_object_ids
                        ):
                            raise ValueError(
                                "Relative preference references an object outside this map."
                            )
                        constraint["reference_regions"] = [
                            {
                                "object_id": object_id,
                                "cells": object_cells_for_map(
                                    current_map_state,
                                    int(object_id),
                                ),
                            }
                            for object_id in constraint["object_ids"]
                        ]
                        continue
                    if constraint["preference_type"] in {
                        "clearance",
                        "move_smoothness",
                        "path_shape_preference",
                    }:
                        constraint["object_ids"] = []
                        constraint["reference_regions"] = []
                        constraint["reference_region"] = None
                        constraint["D_ref"] = None
                        continue
                    reference_region = constraint["reference_region"]
                    if (
                        not isinstance(reference_region, dict)
                        or reference_region["mode"] != "object"
                    ):
                        raise ValueError(
                            "Near/far preference must reference an object."
                        )
                    if (
                        reference_region["object_id"] not in valid_object_ids
                    ):
                        raise ValueError(
                            "Near/far preference references an object outside this map."
                        )
                    object_id = reference_region["object_id"]
                    reference_region["cells"] = object_cells_for_map(
                        current_map_state,
                        int(object_id),
                    )

                annotate_soft_reference_distances(active_sample, current_map_state)
                template = template_by_id[template_instruction_id]
                instruction_record = save_instruction_file(
                    resolved_map_id,
                    template,
                    active_sample,
                    map_state=current_map_state,
                )
                template_bundle = update_template_instruction_status(
                    resolved_map_id,
                    template_instruction_id,
                    "labeled",
                    int(instruction_record["id"]),
                )
                annotation_state = response_annotation_state_from_payload(
                    payload,
                    resolved_map_id,
                    active_sample,
                    int(instruction_record["id"]),
                )
                current_state = build_client_state(
                    resolved_map_id,
                    map_state=current_map_state,
                    annotation_state=annotation_state,
                    template_instruction_state=template_bundle,
                )
                self._send_json(
                    {
                        "message": (
                            f"Saved instruction {instruction_record['id']} and labeled "
                            f"template '{template_instruction_id}'."
                        ),
                        "next_template_instruction_id": next_unlabeled_template_id(
                            template_bundle, template_instruction_id
                        ),
                        "state": current_state,
                    }
                )
                return

            if self.path == "/abandon_template":
                map_id = payload.get("map_id")
                template_instruction_id = payload.get("template_instruction_id")
                if not isinstance(map_id, str) or not map_id.strip():
                    raise ValueError("map_id must be a non-empty string.")
                if (
                    not isinstance(template_instruction_id, str)
                    or not template_instruction_id.strip()
                ):
                    raise ValueError(
                        "template_instruction_id must be a non-empty string."
                    )

                current_map_state = load_map_state(map_id)
                resolved_map_id = str(current_map_state["map_key"])
                existing_template_bundle = load_template_instruction_bundle(resolved_map_id)
                existing_template = next(
                    (
                        template
                        for template in existing_template_bundle["templates"]  # type: ignore[union-attr]
                        if template["template_instruction_id"]
                        == template_instruction_id
                    ),
                    None,
                )
                if existing_template is None:
                    raise ValueError(
                        f"Unknown template_instruction_id: {template_instruction_id!r}"
                    )

                labeled_instruction_id = existing_template.get(
                    "labeled_instruction_id"
                )
                if isinstance(labeled_instruction_id, int):
                    path = instruction_file_path(resolved_map_id, labeled_instruction_id)
                    if path.exists():
                        path.unlink()

                template_bundle = update_template_instruction_status(
                    resolved_map_id,
                    template_instruction_id,
                    "abandoned",
                )
                current_state = build_client_state(resolved_map_id)
                current_state["template_instruction_state"] = template_bundle
                self._send_json(
                    {
                        "message": f"Abandoned template '{template_instruction_id}'.",
                        "next_template_instruction_id": next_unlabeled_template_id(
                            template_bundle, template_instruction_id
                        ),
                        "state": current_state,
                    }
                )
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found.")
        except Exception as exc:  # pragma: no cover - defensive runtime validation.
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        return


def run_server(host: str, port: int, no_browser: bool) -> None:
    available = list_map_ids()
    if not available:
        raise RuntimeError(
            f"No layered maps available under {MAP_ROOT}. Run a map exporter first."
        )

    with ThreadingHTTPServer((host, port), InstructionAnnotatorHandler) as server:
        actual_host, actual_port = server.server_address[:2]
        display_host = "127.0.0.1" if actual_host == "0.0.0.0" else actual_host
        url = f"http://{display_host}:{actual_port}"

        print("Instruction + Expert Route annotator is ready.")
        print(f"Open this URL in your browser: {url}")
        print(f"Map root: {MAP_ROOT}")
        print(f"Instruction root: {INSTRUCTION_ROOT}")
        print("Press Ctrl+C to stop the server.")

        if not no_browser:
            webbrowser.open(url)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\\nServer stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SemPathBench instruction and expert-route annotation editor."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the local editor server to. Use 0.0.0.0 if needed.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the local editor server to.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open a browser tab.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_server(args.host, args.port, args.no_browser)


if __name__ == "__main__":
    main()
