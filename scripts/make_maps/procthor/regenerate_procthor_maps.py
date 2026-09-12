#!/usr/bin/env python3
"""Safely migrate or generate ProcTHOR maps with richer object localization."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.make_maps.procthor.transform_procthor_to_map import (  # noqa: E402
    DEFAULT_AI2THOR_BASE_DIR,
    normalize_sequence_mode,
    parse_bool,
    transform_procthor_batch,
    transform_procthor_to_map,
)
from scripts.evaluation.metric_cache import build_map_metric_cache

REPO_ROOT = Path(__file__).resolve().parents[3]
MAP_ROOT = REPO_ROOT / "resources" / "maps" / "procthor"
INSTRUCTION_ROOT = REPO_ROOT / "resources" / "instructions" / "procthor"


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def normalize_map_id(map_id: str) -> str:
    cleaned = map_id.strip().rstrip("/")
    if cleaned.startswith("procthor/"):
        cleaned = cleaned.split("/", 1)[1]
    if not cleaned:
        raise ValueError("map id cannot be empty.")
    return cleaned


def map_split_from_id(map_id: str) -> str:
    key = normalize_map_id(map_id)
    if key.endswith("_train"):
        return "train"
    if key.endswith("_valunseen"):
        return "valunseen"
    raise ValueError(f"ProcTHOR map id must end in _train or _valunseen: {key}")


def map_directory(map_id: str) -> Path:
    key = normalize_map_id(map_id)
    preferred = MAP_ROOT / map_split_from_id(key) / key
    # Read old exports during a staged migration. New writes always use the
    # split directory above.
    legacy = MAP_ROOT / key
    return legacy if legacy.exists() and not preferred.exists() else preferred


def map_json_path(map_id: str) -> Path:
    key = normalize_map_id(map_id)
    return map_directory(key) / f"{key}.json"


def metadata_json_path(map_id: str) -> Path:
    key = normalize_map_id(map_id)
    return map_directory(key) / f"{key}_metadata.json"


def thinggraph_json_path(map_id: str) -> Path:
    key = normalize_map_id(map_id)
    return map_directory(key) / f"{key}_thinggraph.json"


def template_json_path(map_id: str) -> Path:
    return map_directory(map_id) / "template_instruction.json"


def instruction_files_directory(map_id: str) -> Path:
    return INSTRUCTION_ROOT / normalize_map_id(map_id) / "instruction_files"


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        json.dump(_json_ready(dict(payload)), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.bak.{_iso_timestamp()}")
    shutil.copy2(path, backup_path)
    return backup_path


def _grid_size(map_payload: Mapping[str, object]) -> int:
    raw_size = map_payload.get("grid_size")
    if isinstance(raw_size, int) and not isinstance(raw_size, bool) and raw_size > 0:
        return raw_size
    layers = map_payload.get("layers")
    occupancy = layers.get("occupancy") if isinstance(layers, Mapping) else None
    return len(occupancy) if isinstance(occupancy, Sequence) else 0


def _valid_cell(value: object, grid_size: int) -> tuple[int, int] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        return None
    row, col = value
    if not isinstance(row, (int, float)) or not isinstance(col, (int, float)):
        return None
    cell = (int(round(float(row))), int(round(float(col))))
    if 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size:
        return cell
    return None


def _valid_cells(value: object, grid_size: int) -> list[tuple[int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in value:
        cell = _valid_cell(item, grid_size)
        if cell is None or cell in seen:
            continue
        cells.append(cell)
        seen.add(cell)
    return cells


def _attribute_value(attributes: object, name: str) -> object:
    if not isinstance(attributes, Sequence) or isinstance(attributes, (str, bytes)):
        return None
    prefix = f"{name}="
    for item in attributes:
        if not isinstance(item, str) or not item.startswith(prefix):
            continue
        return item[len(prefix) :]
    return None


def _object_instances(map_payload: Mapping[str, object]) -> list[dict[str, object]]:
    raw = map_payload.get("object_instances")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _legacy_object_cells(map_payload: Mapping[str, object]) -> dict[int, list[tuple[int, int]]]:
    layers = map_payload.get("layers")
    object_grid = layers.get("object_instance") if isinstance(layers, Mapping) else None
    if not isinstance(object_grid, Sequence) or isinstance(object_grid, (str, bytes)):
        return {}
    cells_by_id: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, row_values in enumerate(object_grid):
        if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
            continue
        for col, raw_value in enumerate(row_values):
            try:
                object_id = int(raw_value)
            except (TypeError, ValueError):
                continue
            if object_id > 0:
                cells_by_id[object_id].append((row, col))
    return dict(cells_by_id)


def _room_layer(map_payload: Mapping[str, object]) -> Sequence[Sequence[object]]:
    layers = map_payload.get("layers")
    room = layers.get("room") if isinstance(layers, Mapping) else None
    if not isinstance(room, Sequence) or isinstance(room, (str, bytes)):
        return []
    return room  # type: ignore[return-value]


def _room_at(room_layer: Sequence[Sequence[object]], row: int, col: int) -> int | None:
    if row < 0 or row >= len(room_layer):
        return None
    row_values = room_layer[row]
    if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
        return None
    if col < 0 or col >= len(row_values):
        return None
    try:
        room_id = int(row_values[col])
    except (TypeError, ValueError):
        return None
    return room_id if room_id > 0 else None


def _center(cells: Sequence[tuple[int, int]]) -> list[float] | None:
    if not cells:
        return None
    return [
        round(sum(row for row, _col in cells) / len(cells), 3),
        round(sum(col for _row, col in cells) / len(cells), 3),
    ]


def _metadata_by_instance_id(metadata_payload: Mapping[str, object] | None) -> dict[int, Mapping[str, object]]:
    raw = metadata_payload.get("object_metadata") if isinstance(metadata_payload, Mapping) else None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return {}
    result: dict[int, Mapping[str, object]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("instance_id")
        if isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id > 0:
            result[raw_id] = item
    return result


def _thinggraph_objects(thinggraph_payload: Mapping[str, object] | None) -> dict[int, Mapping[str, object]]:
    if not isinstance(thinggraph_payload, Mapping):
        return {}
    result: dict[int, Mapping[str, object]] = {}
    for room in thinggraph_payload.get("rooms", []):
        if not isinstance(room, Mapping):
            continue
        for item in room.get("objects", []):
            if isinstance(item, Mapping) and isinstance(item.get("id"), int):
                result[int(item["id"])] = item
    for item in thinggraph_payload.get("unassigned_objects", []):
        if isinstance(item, Mapping) and isinstance(item.get("id"), int):
            result[int(item["id"])] = item
    return result


def _metadata_grid_cell(record: Mapping[str, object], grid_size: int) -> tuple[int, int] | None:
    row = record.get("grid_row")
    col = record.get("grid_col")
    if isinstance(row, (int, float)) and isinstance(col, (int, float)):
        return _valid_cell([row, col], grid_size)
    return None


def _dominant_room(
    map_payload: Mapping[str, object],
    cells: Sequence[tuple[int, int]],
    fallback_cell: tuple[int, int] | None,
) -> int | None:
    room_layer = _room_layer(map_payload)
    votes: Counter[int] = Counter()
    for row, col in cells:
        room_id = _room_at(room_layer, row, col)
        if room_id is not None:
            votes[room_id] += 1
    if votes:
        return votes.most_common(1)[0][0]
    if fallback_cell is not None:
        return _room_at(room_layer, fallback_cell[0], fallback_cell[1])
    return None


def build_object_footprints(
    map_payload: Mapping[str, object],
    metadata_payload: Mapping[str, object] | None = None,
    thinggraph_payload: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Build rich object footprints while preserving legacy object ids."""

    grid_size = _grid_size(map_payload)
    metadata_by_id = _metadata_by_instance_id(metadata_payload)
    thinggraph_by_id = _thinggraph_objects(thinggraph_payload)
    legacy_cells = _legacy_object_cells(map_payload)
    footprints: dict[str, dict[str, object]] = {}

    for instance in _object_instances(map_payload):
        raw_id = instance.get("id")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool) or raw_id <= 0:
            continue
        object_id = raw_id
        metadata_record = metadata_by_id.get(object_id, {})
        thing_record = thinggraph_by_id.get(object_id, {})
        metadata_cell = _metadata_grid_cell(metadata_record, grid_size)

        cells = _valid_cells(metadata_record.get("grid_cells"), grid_size)
        if not cells:
            cells = legacy_cells.get(object_id, [])
        if not cells:
            for source in (thing_record, instance):
                for key in ("center_grid", "metadata_grid_cell"):
                    cell = _valid_cell(source.get(key), grid_size)
                    if cell is not None:
                        cells = [cell]
                        break
                if cells:
                    break
        if not cells and metadata_cell is not None:
            cells = [metadata_cell]

        center_grid = None
        for source in (thing_record, instance):
            center_cell = _valid_cell(source.get("center_grid"), grid_size)
            if center_cell is not None:
                center_grid = [center_cell[0], center_cell[1]]
                break
        if center_grid is None:
            center_grid = _center(cells)

        graph_metadata_cell = _valid_cell(thing_record.get("metadata_grid_cell"), grid_size)
        object_metadata_cell = _valid_cell(instance.get("metadata_grid_cell"), grid_size)
        best_metadata_cell = metadata_cell or graph_metadata_cell or object_metadata_cell
        room_id = thing_record.get("room_id") or instance.get("room_id")
        if not isinstance(room_id, int) or isinstance(room_id, bool) or room_id <= 0:
            room_id = _dominant_room(
                map_payload,
                cells,
                best_metadata_cell or (cells[0] if cells else None),
            )

        assignment_method = (
            thing_record.get("assignment_method")
            or instance.get("assignment_method")
            or ("footprint_cells" if cells else None)
        )
        footprint_source = (
            metadata_record.get("footprint_source")
            or thing_record.get("footprint_source")
            or _attribute_value(instance.get("attributes"), "footprint_source")
        )
        num_grid_cells = metadata_record.get("num_grid_cells")
        if not isinstance(num_grid_cells, (int, float)) or isinstance(num_grid_cells, bool):
            num_grid_cells = thing_record.get("num_grid_cells")
        if not isinstance(num_grid_cells, (int, float)) or isinstance(num_grid_cells, bool):
            num_grid_cells = len(cells)

        footprints[str(object_id)] = {
            "object_id": object_id,
            "category": instance.get("category"),
            "name": instance.get("name"),
            "cells": [[row, col] for row, col in cells],
            "center_grid": center_grid,
            "metadata_grid_cell": (
                [best_metadata_cell[0], best_metadata_cell[1]]
                if best_metadata_cell is not None
                else None
            ),
            "room_id": room_id,
            "assignment_method": assignment_method,
            "footprint_source": footprint_source,
            "num_grid_cells": int(num_grid_cells),
        }
    return footprints


def build_cell_object_ids(
    object_footprints: Mapping[str, Mapping[str, object]],
) -> dict[str, list[int]]:
    cell_index: dict[str, list[int]] = defaultdict(list)
    object_sizes: dict[int, int] = {}
    for raw_record in object_footprints.values():
        object_id = raw_record.get("object_id")
        if not isinstance(object_id, int) or isinstance(object_id, bool) or object_id <= 0:
            continue
        cells = raw_record.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            continue
        object_sizes[object_id] = len(cells)
        for cell in cells:
            if (
                isinstance(cell, Sequence)
                and not isinstance(cell, (str, bytes))
                and len(cell) == 2
            ):
                key = f"{int(cell[0])},{int(cell[1])}"
                if object_id not in cell_index[key]:
                    cell_index[key].append(object_id)

    return {
        key: sorted(ids, key=lambda object_id: (-object_sizes.get(object_id, 0), object_id))
        for key, ids in sorted(cell_index.items())
    }


def add_grid_coordinate_frame(map_payload: dict[str, object], metadata_payload: Mapping[str, object] | None) -> None:
    metadata = map_payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        map_payload["metadata"] = metadata
    map_info = metadata.get("map_info")
    if not isinstance(map_info, Mapping) and isinstance(metadata_payload, Mapping):
        map_info = metadata_payload.get("map_info")
    if not isinstance(map_info, Mapping):
        return
    frame: dict[str, object] = {
        "index_order": "[row, col]",
        "row_axis": "world_z",
        "col_axis": "world_x",
    }
    for key in ("resolution", "x_min", "z_min"):
        value = map_info.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            frame[key] = value
    metadata["grid_coordinate_frame"] = frame


def enrich_object_instances(
    map_payload: dict[str, object],
    object_footprints: Mapping[str, Mapping[str, object]],
) -> None:
    raw_instances = map_payload.get("object_instances")
    if not isinstance(raw_instances, list):
        return
    for instance in raw_instances:
        if not isinstance(instance, dict):
            continue
        object_id = instance.get("id")
        if not isinstance(object_id, int) or isinstance(object_id, bool):
            continue
        footprint = object_footprints.get(str(object_id))
        if not isinstance(footprint, Mapping):
            continue
        for key in (
            "center_grid",
            "metadata_grid_cell",
            "room_id",
            "assignment_method",
            "num_grid_cells",
            "footprint_source",
        ):
            value = footprint.get(key)
            if value is not None:
                instance[key] = value


def migrate_map_payload(
    map_payload: dict[str, object],
    metadata_payload: Mapping[str, object] | None = None,
    thinggraph_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    object_footprints = build_object_footprints(
        map_payload,
        metadata_payload=metadata_payload,
        thinggraph_payload=thinggraph_payload,
    )
    map_payload["object_footprints"] = object_footprints
    map_payload["cell_object_ids"] = build_cell_object_ids(object_footprints)
    enrich_object_instances(map_payload, object_footprints)
    add_grid_coordinate_frame(map_payload, metadata_payload)
    return map_payload


def instruction_map_ids() -> list[str]:
    if not INSTRUCTION_ROOT.exists():
        return []
    result: list[str] = []
    for directory in sorted(INSTRUCTION_ROOT.iterdir()):
        if directory.is_dir() and (directory / "instruction_files").is_dir():
            result.append(directory.name)
    return result


def existing_map_ids() -> list[str]:
    """Return every ProcTHOR map directory that contains its primary map JSON."""

    if not MAP_ROOT.exists():
        return []
    result: list[str] = []
    for split in ("train", "valunseen"):
        split_directory = MAP_ROOT / split
        if not split_directory.is_dir():
            continue
        for directory in sorted(split_directory.iterdir()):
            if directory.is_dir() and (directory / f"{directory.name}.json").exists():
                result.append(directory.name)
    # Retain read compatibility for an interrupted one-time layout migration.
    for directory in sorted(MAP_ROOT.iterdir()):
        if directory.name in {"train", "valunseen"} or not directory.is_dir():
            continue
        if (directory / f"{directory.name}.json").exists():
            result.append(directory.name)
    return sorted(dict.fromkeys(result))


def validate_map_shape(map_payload: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    grid_size = _grid_size(map_payload)
    if grid_size <= 0:
        return ["grid_size must be positive."]
    layers = map_payload.get("layers")
    if not isinstance(layers, Mapping):
        return ["layers must be an object."]
    for layer_name in ("occupancy", "room", "object_instance"):
        layer = layers.get(layer_name)
        if not isinstance(layer, Sequence) or isinstance(layer, (str, bytes)):
            errors.append(f"layers.{layer_name} must be a {grid_size}x{grid_size} grid.")
            continue
        if len(layer) != grid_size:
            errors.append(f"layers.{layer_name} has {len(layer)} rows, expected {grid_size}.")
        for row_index, row in enumerate(layer):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                errors.append(f"layers.{layer_name}[{row_index}] is not a row.")
                continue
            if len(row) != grid_size:
                errors.append(
                    f"layers.{layer_name}[{row_index}] has {len(row)} cols, expected {grid_size}."
                )
                break
    return errors


def validate_rich_object_schema(map_payload: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    grid_size = _grid_size(map_payload)
    object_ids = {
        int(instance["id"])
        for instance in _object_instances(map_payload)
        if isinstance(instance.get("id"), int)
    }
    footprints = map_payload.get("object_footprints")
    if isinstance(footprints, Mapping):
        for raw_key, raw_record in footprints.items():
            try:
                object_id = int(raw_key)
            except (TypeError, ValueError):
                errors.append(f"object_footprints key {raw_key!r} is not an object id.")
                continue
            if object_id not in object_ids:
                errors.append(f"object_footprints.{object_id} is not in object_instances.")
            if not isinstance(raw_record, Mapping):
                errors.append(f"object_footprints.{object_id} must be an object.")
                continue
            cells = _valid_cells(raw_record.get("cells"), grid_size)
            if not cells:
                errors.append(f"object_footprints.{object_id}.cells is empty.")
    cell_index = map_payload.get("cell_object_ids")
    if isinstance(cell_index, Mapping):
        for key, raw_ids in cell_index.items():
            try:
                row_text, col_text = str(key).split(",", 1)
                row, col = int(row_text), int(col_text)
            except ValueError:
                errors.append(f"cell_object_ids key {key!r} is not 'row,col'.")
                continue
            if not (0 <= row < grid_size and 0 <= col < grid_size):
                errors.append(f"cell_object_ids key {key!r} is outside the grid.")
            if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
                errors.append(f"cell_object_ids.{key} must be a list.")
                continue
            for raw_id in raw_ids:
                try:
                    object_id = int(raw_id)
                except (TypeError, ValueError):
                    errors.append(f"cell_object_ids.{key} contains non-integer id {raw_id!r}.")
                    continue
                if object_id not in object_ids:
                    errors.append(f"cell_object_ids.{key} contains unknown object id {object_id}.")
    return errors


def _object_ids_from_instruction_payload(payload: Mapping[str, object]) -> list[int]:
    ids: list[int] = []
    for item in payload.get("objects", []):
        if isinstance(item, Mapping) and isinstance(item.get("object_id"), int):
            ids.append(int(item["object_id"]))
    for item in payload.get("hard_constraints", []):
        if isinstance(item, Mapping) and isinstance(item.get("object_id"), int):
            ids.append(int(item["object_id"]))
    for item in payload.get("soft_constraints", []):
        if not isinstance(item, Mapping):
            continue
        reference_region = item.get("reference_region")
        if isinstance(reference_region, Mapping) and isinstance(reference_region.get("object_id"), int):
            ids.append(int(reference_region["object_id"]))
        reference_regions = item.get("reference_regions")
        if isinstance(reference_regions, Sequence) and not isinstance(reference_regions, (str, bytes)):
            for region in reference_regions:
                if isinstance(region, Mapping) and isinstance(region.get("object_id"), int):
                    ids.append(int(region["object_id"]))
    return ids


def _cells_from_instruction_payload(payload: Mapping[str, object]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for constraint_key in ("hard_constraints", "soft_constraints"):
        for item in payload.get(constraint_key, []):
            if not isinstance(item, Mapping):
                continue
            for raw_cell in item.get("cells", []):
                if isinstance(raw_cell, Sequence) and not isinstance(raw_cell, (str, bytes)) and len(raw_cell) == 2:
                    cells.append((int(raw_cell[0]), int(raw_cell[1])))
            reference_region = item.get("reference_region")
            if isinstance(reference_region, Mapping):
                for raw_cell in reference_region.get("cells", []):
                    if isinstance(raw_cell, Sequence) and not isinstance(raw_cell, (str, bytes)) and len(raw_cell) == 2:
                        cells.append((int(raw_cell[0]), int(raw_cell[1])))
            reference_regions = item.get("reference_regions")
            if isinstance(reference_regions, Sequence) and not isinstance(reference_regions, (str, bytes)):
                for region in reference_regions:
                    if not isinstance(region, Mapping):
                        continue
                    for raw_cell in region.get("cells", []):
                        if isinstance(raw_cell, Sequence) and not isinstance(raw_cell, (str, bytes)) and len(raw_cell) == 2:
                            cells.append((int(raw_cell[0]), int(raw_cell[1])))
    return cells


def validate_instruction_references(map_id: str, map_payload: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    grid_size = _grid_size(map_payload)
    object_ids = {
        int(instance["id"])
        for instance in _object_instances(map_payload)
        if isinstance(instance.get("id"), int)
    }
    instruction_dir = instruction_files_directory(map_id)
    if not instruction_dir.exists():
        return errors
    for path in sorted(instruction_dir.glob("instruction_*.json")):
        payload = load_json(path)
        if payload is None:
            errors.append(f"{path}: invalid instruction JSON.")
            continue
        for object_id in _object_ids_from_instruction_payload(payload):
            if object_id not in object_ids:
                errors.append(f"{path}: missing referenced object_id {object_id}.")
        for row, col in _cells_from_instruction_payload(payload):
            if not (0 <= row < grid_size and 0 <= col < grid_size):
                errors.append(f"{path}: reference cell [{row}, {col}] is outside the grid.")
    return errors


def validate_map(map_id: str, map_payload: Mapping[str, object]) -> list[str]:
    return [
        *validate_map_shape(map_payload),
        *validate_rich_object_schema(map_payload),
        *validate_instruction_references(map_id, map_payload),
    ]


def migrate_map(map_id: str, *, write: bool = True, backup: bool = True) -> dict[str, object]:
    key = normalize_map_id(map_id)
    path = map_json_path(key)
    payload = load_json(path)
    if payload is None:
        raise FileNotFoundError(f"Map JSON not found or invalid: {path}")
    migrated = migrate_map_payload(
        dict(payload),
        metadata_payload=load_json(metadata_json_path(key)),
        thinggraph_payload=load_json(thinggraph_json_path(key)),
    )
    errors = validate_map(key, migrated)
    if errors:
        raise ValueError("Map validation failed:\n" + "\n".join(errors[:50]))
    if write:
        if backup:
            backup_file(path)
        write_json_atomic(path, migrated)
        build_map_metric_cache(path, migrated)
    return migrated


def check_map(map_id: str) -> list[str]:
    key = normalize_map_id(map_id)
    payload = load_json(map_json_path(key))
    if payload is None:
        return [f"Map JSON not found or invalid: {map_json_path(key)}"]
    return validate_map(key, payload)


def regenerate_map(
    map_id: str,
    *,
    split: str | None,
    index: int | None,
    dataset_name: str,
    resolution: float,
    padding: float,
    save_overview: bool,
    ai2thor_base_dir: Path | None,
    backup: bool,
    preserve_annotations: bool,
) -> dict[str, object]:
    key = normalize_map_id(map_id)
    old_payload = load_json(map_json_path(key))
    old_metadata = old_payload.get("metadata") if isinstance(old_payload, Mapping) else {}
    if split is None and isinstance(old_metadata, Mapping):
        raw_split = old_metadata.get("procthor_split")
        if isinstance(raw_split, str) and raw_split.strip():
            split = raw_split
    if index is None and isinstance(old_metadata, Mapping):
        raw_index = old_metadata.get("procthor_index")
        if isinstance(raw_index, int) and not isinstance(raw_index, bool):
            index = raw_index
    if split is None or index is None:
        raise ValueError("generate mode needs --split and --index, or existing map metadata with procthor_split/procthor_index.")

    template_path = template_json_path(key)
    old_template = template_path.read_bytes() if preserve_annotations and template_path.exists() else None
    if backup:
        backup_file(map_json_path(key))
        backup_file(template_path)

    output_prefix = map_directory(key) / key
    map_position = None
    if isinstance(old_metadata, Mapping):
        raw_map_index = old_metadata.get("map_index")
        if isinstance(raw_map_index, int) and not isinstance(raw_map_index, bool):
            map_position = raw_map_index

    transform_procthor_to_map(
        split=split,
        index=index,
        dataset_name=dataset_name,
        resolution=resolution,
        padding=padding,
        output_prefix=output_prefix,
        save_overview=save_overview,
        save_simple_demo=True,
        use_simulator=True,
        ai2thor_base_dir=ai2thor_base_dir,
        map_position=map_position,
    )
    if old_template is not None:
        template_path.write_bytes(old_template)
    return migrate_map(key, write=True, backup=False)


def snapshot_templates(map_ids: Sequence[str]) -> dict[str, bytes]:
    snapshots: dict[str, bytes] = {}
    for map_id in map_ids:
        path = template_json_path(map_id)
        if path.exists():
            snapshots[normalize_map_id(map_id)] = path.read_bytes()
    return snapshots


def restore_templates(snapshots: Mapping[str, bytes]) -> None:
    for map_id, payload in snapshots.items():
        template_json_path(map_id).write_bytes(payload)


def regenerate_batch(args: argparse.Namespace) -> list[str]:
    """Run the original batch transformer, then migrate all touched maps."""

    protected_templates = (
        snapshot_templates(existing_map_ids()) if not args.no_preserve_annotations else {}
    )

    def print_progress(record: dict[str, object]) -> None:
        event = record.get("event")
        map_id = record.get("map_id")
        if event == "processing":
            print(
                f"Processing {map_id} "
                f"({record.get('position')}/{record.get('target')}, "
                f"procthor_scene_id={record.get('procthor_scene_id')})"
            )
        elif event == "resume_skip":
            print(f"Skipping existing {map_id} (resume).")
        elif event == "saved":
            print(f"Saved {map_id}.")

    results = transform_procthor_batch(
        split=args.split or "train",
        number=args.number,
        scene_size=args.scene_size,
        sequence=args.sequence,
        resume=args.resume,
        dataset_name=args.dataset_name,
        resolution=args.resolution,
        padding=args.padding,
        output_prefix=args.output_prefix,
        save_overview=not args.skip_overview,
        save_simple_demo=True,
        ai2thor_base_dir=args.ai2thor_base_dir,
        progress_callback=print_progress,
    )
    restore_templates(protected_templates)

    touched = []
    for section in ("processed", "skipped"):
        for record in results.get(section, []):
            if isinstance(record, Mapping) and isinstance(record.get("map_id"), str):
                touched.append(str(record["map_id"]))
    return sorted(dict.fromkeys(touched))


def resolve_target_map_ids(args: argparse.Namespace) -> list[str]:
    targets = [normalize_map_id(item) for item in args.map_id]
    if args.all_annotated:
        targets.extend(instruction_map_ids())
    if args.all_maps:
        targets.extend(existing_map_ids())
    if not targets:
        raise ValueError("Specify --map-id, --all-annotated, --all-maps, or use --mode generate --index -1.")
    return sorted(dict.fromkeys(targets))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely migrate/check/generate ProcTHOR maps without rewriting annotations."
    )
    parser.add_argument("--map-id", action="append", default=[], help="Map id, e.g. 033_valunseen or procthor/033_valunseen.")
    parser.add_argument("--all-annotated", action="store_true", help="Target every ProcTHOR map that has instruction_files.")
    parser.add_argument("--all-maps", action="store_true", help="Target every existing ProcTHOR map JSON under resources/maps/procthor.")
    parser.add_argument(
        "--mode",
        choices=("migrate", "check", "generate", "regenerate"),
        default="migrate",
        help="'generate' runs ProcTHOR export then migrates maps. 'regenerate' is kept as a backward-compatible alias.",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak timestamp files before writing map JSON.")
    parser.add_argument("--split", default=None, help="ProcTHOR split for generate mode.")
    parser.add_argument("--index", type=int, default=None, help="ProcTHOR source index for generate mode.")
    parser.add_argument("--number", type=int, default=1, help="Number of accepted scenes to export when --mode generate --index -1.")
    parser.add_argument("--scene-size", "--scene_size", dest="scene_size", type=float, default=None)
    parser.add_argument("--sequence", type=normalize_sequence_mode, default="sequence")
    parser.add_argument("--resume", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--dataset-name", default="procthor-10k")
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=1.0)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--skip-overview", action="store_true")
    parser.add_argument("--ai2thor-base-dir", type=Path, default=DEFAULT_AI2THOR_BASE_DIR)
    parser.add_argument("--no-preserve-annotations", action="store_true", help="Allow regenerated template_instruction.json to replace the old one.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generation_modes = {"generate", "regenerate"}
    if args.mode in generation_modes and args.index == -1:
        touched = regenerate_batch(args)
        for map_id in touched:
            migrated = migrate_map(map_id, write=True, backup=not args.no_backup)
            print(
                f"{map_id}: migrated after batch generate "
                f"objects={len(migrated.get('object_footprints', {}))} "
                f"cells={len(migrated.get('cell_object_ids', {}))}"
            )
        print(f"batch complete: touched={len(touched)}")
        return
    targets = resolve_target_map_ids(args)
    failures: dict[str, list[str]] = {}
    for map_id in targets:
        if args.mode == "check":
            errors = check_map(map_id)
            if errors:
                failures[map_id] = errors
                print(f"{map_id}: FAILED ({len(errors)} errors)")
            else:
                print(f"{map_id}: OK")
            continue
        if args.mode == "migrate":
            migrated = migrate_map(map_id, write=True, backup=not args.no_backup)
            print(
                f"{map_id}: migrated "
                f"objects={len(migrated.get('object_footprints', {}))} "
                f"cells={len(migrated.get('cell_object_ids', {}))}"
            )
            continue
        generated = regenerate_map(
            map_id,
            split=args.split,
            index=args.index,
            dataset_name=args.dataset_name,
            resolution=args.resolution,
            padding=args.padding,
            save_overview=not args.skip_overview,
            ai2thor_base_dir=args.ai2thor_base_dir,
            backup=not args.no_backup,
            preserve_annotations=not args.no_preserve_annotations,
        )
        print(
            f"{map_id}: generated "
            f"objects={len(generated.get('object_footprints', {}))} "
            f"cells={len(generated.get('cell_object_ids', {}))}"
        )
    if failures:
        summary = "\n".join(
            f"{map_id}: {errors[0]}" for map_id, errors in failures.items()
        )
        raise SystemExit(f"Validation failed:\n{summary}")


if __name__ == "__main__":
    main()
