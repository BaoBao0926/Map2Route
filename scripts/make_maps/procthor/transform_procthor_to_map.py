#!/usr/bin/env python3
"""Transform one ProcTHOR house into SemPathBench maps."""

from __future__ import annotations

import argparse
import colorsys
import copy
import hashlib
import json
import math
import os
import random
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.evaluation.metric_cache import build_map_metric_cache
from scripts.make_maps.procthor.convert_procthor_scene import (
    compute_house_largest_room_size,
    compute_house_room_sizes,
    compute_house_scene_size,
    convert_procthor_house_to_maps,
    rasterize_polygon_to_grid,
    save_scene_representation,
)
from scripts.make_maps.procthor.view_procthor_map import (
    _category_id_to_name,
    _default_png_path,
    draw_object_overlay,
    format_room_size_summary,
    invert_mapping,
    plot_scene_export,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AI2THOR_BASE_DIR = REPO_ROOT / "resources" / "procthor" / "ai2thor"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "resources" / "maps" / "procthor"

SIMPLE_DEMO_OCCUPANCY_TILES = {
    "free": {"value": 0, "label": "Free Space", "color": "#FFFFFF"},
    "obstacle": {"value": 1, "label": "Obstacle / Wall", "color": "#111111"},
    "unknown": {"value": 2, "label": "Unknown", "color": "#9E9E9E"},
}
SIMPLE_DEMO_ROOM_COLORS = {
    "hallway": "#F2E8CF",
    "kitchen": "#CDECCF",
    "bedroom": "#D6E8FF",
    "living_room": "#FFE0B5",
    "bathroom": "#BFE8E5",
}
SIMPLE_DEMO_OBJECT_COLORS = {
    "table": "#F4D03F",
    "chair": "#EFA8A0",
    "sofa": "#D96C5F",
    "desk": "#8EC5A4",
    "cabinet": "#7C9A6D",
    "wall": "#2F3437",
    "bed": "#9FB4D8",
    "door": "#B9855C",
}
SIMPLE_TEMPLATE_OBJECT_COUNTS = (2, 3, 4, 5, 10)
SIMPLE_TEMPLATE_MIN_DISTANCE = 40.0
SIMPLE_TEMPLATE_MAX_PER_OBJECT_COUNT = 200
SIMPLE_TEMPLATE_MAX_OBJECTS = 80
SIMPLE_TEMPLATE_MAX_TOTAL = SIMPLE_TEMPLATE_MAX_PER_OBJECT_COUNT * len(SIMPLE_TEMPLATE_OBJECT_COUNTS)
SIMPLE_TEMPLATE_GRID_METRIC = "object_centroid_euclidean_grid"
SIMPLE_CATEGORY_ALIASES = {
    "shelving_unit": "shelf",
}
MAP_SPLIT_TRAIN = "train"
MAP_SPLIT_VAL_UNSEEN = "valunseen"
MAP_SPLIT_RULE = {
    "name": "two_train_one_valunseen_mod_3",
    "period": 3,
    "train_remainders": [0, 1],
    "valunseen_remainders": [2],
}


def build_procthor_scene_id(split: str, index: int) -> str:
    """Build a stable source ProcTHOR scene id."""
    return f"{split}_{index:05d}"


def build_scene_id(split: str, index: int) -> str:
    """Build a stable source ProcTHOR scene id."""
    return build_procthor_scene_id(split, index)


def build_map_id(map_index: int, map_split: str) -> str:
    """Build the SemPathBench map id used for folders and file prefixes."""
    if map_index <= 0:
        raise ValueError("map_index must be one-based and positive.")
    return f"{map_index:03d}_{map_split}"


def _map_split_for_zero_based_offset(offset: int) -> str:
    if offset < 0:
        raise ValueError("map split offset must be non-negative.")
    return (
        MAP_SPLIT_VAL_UNSEEN
        if offset % int(MAP_SPLIT_RULE["period"]) == 2
        else MAP_SPLIT_TRAIN
    )


def map_split_for_source_index(index: int) -> str:
    """Assign source scene indices in a stable 2:1 train/valunseen pattern."""
    if index < 0:
        raise ValueError("index must be non-negative.")
    return _map_split_for_zero_based_offset(index)


def map_split_for_sequence_position(position: int) -> str:
    """Assign accepted batch positions in a 2:1 train/valunseen pattern."""
    if position <= 0:
        raise ValueError("position must be one-based and positive.")
    return _map_split_for_zero_based_offset(position - 1)


def benchmark_split_for_scene_index(index: int) -> str:
    """Backward-compatible alias for source-index map split assignment."""
    return map_split_for_source_index(index)


def benchmark_split_for_sequence_position(position: int) -> str:
    """Backward-compatible alias for accepted-position map split assignment."""
    return map_split_for_sequence_position(position)


def build_scene_map_metadata(
    split: str,
    index: int,
    *,
    sequence_position: int | None = None,
) -> dict[str, object]:
    """Build clear source ProcTHOR and exported map metadata for one scene."""
    if sequence_position is None:
        map_index = index + 1
        map_split = map_split_for_source_index(index)
        assignment_basis = "procthor_index"
    else:
        map_index = sequence_position
        map_split = map_split_for_sequence_position(sequence_position)
        assignment_basis = "accepted_sequence_position"

    return {
        "map_id": build_map_id(map_index, map_split),
        "map_split": map_split,
        "map_index": map_index,
        "map_assignment_basis": assignment_basis,
        "map_split_rule": {
            **copy.deepcopy(MAP_SPLIT_RULE),
            "assignment_basis": assignment_basis,
        },
        "procthor_split": split,
        "procthor_index": index,
        "procthor_scene_id": build_procthor_scene_id(split, index),
    }


def build_scene_benchmark_metadata(
    split: str,
    index: int,
    *,
    sequence_position: int | None = None,
) -> dict[str, object]:
    """Backward-compatible alias for map metadata construction."""
    return build_scene_map_metadata(
        split,
        index,
        sequence_position=sequence_position,
    )


def _load_prior_module():
    try:
        import prior
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install.
        raise ImportError(
            "prior is required to load ProcTHOR. Install it before running this script."
        ) from exc
    return prior


def load_procthor_dataset(dataset_name: str = "procthor-10k"):
    """Load the ProcTHOR dataset via prior."""
    prior = _load_prior_module()
    return prior.load_dataset(dataset_name)


def _load_controller_class():
    try:
        from ai2thor.controller import Controller
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install.
        raise ImportError(
            "ai2thor is required by the ProcTHOR map exporter. Install it before running."
        ) from exc
    return Controller


def load_procthor_house(
    split: str,
    index: int,
    dataset_name: str = "procthor-10k",
    dataset=None,
) -> dict[str, object]:
    """Load one house from a ProcTHOR dataset split."""
    if index < 0:
        raise ValueError("index must be non-negative.")

    if dataset is None:
        dataset = load_procthor_dataset(dataset_name)

    try:
        houses = dataset[split]
    except KeyError as exc:
        raise KeyError(f"Split {split!r} not found in dataset.") from exc

    if index >= len(houses):
        raise IndexError(
            f"House index {index} is out of range for split {split!r} with {len(houses)} houses."
        )

    house = houses[index]
    if not isinstance(house, dict):
        raise TypeError("Loaded ProcTHOR house must be a dictionary.")
    return house


def _get_split_houses(dataset: object, split: str) -> object:
    try:
        houses = dataset[split]  # type: ignore[index]
    except KeyError as exc:
        raise KeyError(f"Split {split!r} not found in dataset.") from exc

    if not hasattr(houses, "__len__"):
        raise TypeError(f"Dataset split {split!r} must support len().")
    return houses


def create_procthor_controller(
    house: dict[str, object],
    controller_class=None,
    ai2thor_base_dir: str | Path | None = None,
    **controller_kwargs,
):
    """Initialize an AI2-THOR controller from a loaded ProcTHOR house."""
    if controller_class is None:
        controller_class = _load_controller_class()

    if ai2thor_base_dir is None:
        return controller_class(scene=house, **controller_kwargs)

    base_dir = Path(ai2thor_base_dir).resolve()
    os.makedirs(base_dir, exist_ok=True)

    class RepoLocalController(controller_class):
        @property
        def base_dir(self):  # type: ignore[override]
            return str(base_dir)

    return RepoLocalController(scene=house, **controller_kwargs)


def build_default_output_prefix(
    split: str,
    index: int,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Build a default export prefix for one ProcTHOR scene."""
    output_dir = Path(output_dir)
    map_split = map_split_for_source_index(index)
    map_id = build_map_id(index + 1, map_split)
    return output_dir / map_split / map_id / map_id


def build_batch_output_prefix(
    split: str,
    index: int,
    output_prefix: str | Path | None,
    *,
    map_index: int | None = None,
    map_split: str | None = None,
) -> Path:
    """Build a per-scene output prefix for batch mode."""
    del split
    resolved_map_index = index + 1 if map_index is None else map_index
    resolved_map_split = (
        map_split_for_source_index(index) if map_split is None else map_split
    )
    map_id = build_map_id(resolved_map_index, resolved_map_split)
    # A caller-provided output prefix is an exact destination root, retained
    # for custom exports and tests. The standard resources layout is split by
    # train/valunseen automatically.
    if output_prefix is not None:
        return Path(output_prefix) / map_id / map_id
    return DEFAULT_OUTPUT_DIR / resolved_map_split / map_id / map_id


def normalize_sequence_mode(sequence: str) -> str:
    """Normalize batch traversal strategy names."""
    if sequence in {"bigroom", "bvigroom"}:
        return "bigscene"
    if sequence in {"sequence", "bigscene"}:
        return sequence
    raise argparse.ArgumentTypeError("--sequence must be either 'sequence' or 'bigscene'.")


def parse_bool(value: str | bool) -> bool:
    """Parse explicit boolean CLI values."""
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true or false.")


def expected_scene_output_paths(output_prefix: str | Path) -> tuple[Path, Path, Path]:
    """Return the expected maps, metadata, and overview paths for a scene prefix."""
    prefix = Path(output_prefix)
    maps_path = prefix.with_name(f"{prefix.name}_maps.npz")
    metadata_path = prefix.with_name(f"{prefix.name}_metadata.json")
    overview_path = _default_png_path(prefix, None)
    return maps_path, metadata_path, overview_path


def expected_simple_demo_output_paths(
    output_prefix: str | Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Return simple-demo-style JSON, PNG, PPM, template, and thinggraph paths."""
    prefix = Path(output_prefix)
    return (
        prefix.with_name(f"{prefix.name}.json"),
        prefix.with_name(f"{prefix.name}.png"),
        prefix.with_name(f"{prefix.name}.ppm"),
        prefix.parent / "template_instruction.json",
        prefix.with_name(f"{prefix.name}_thinggraph.json"),
    )


def is_scene_export_complete(
    output_prefix: str | Path,
    *,
    require_overview: bool,
    require_simple_demo: bool = False,
) -> bool:
    """Return True when the requested outputs already exist for a scene."""
    maps_path, metadata_path, overview_path = expected_scene_output_paths(output_prefix)
    required_paths = [maps_path, metadata_path]
    if require_overview:
        required_paths.append(overview_path)
    if require_simple_demo:
        simple_json_path, _simple_png_path, simple_ppm_path, template_path, thinggraph_path = (
            expected_simple_demo_output_paths(output_prefix)
        )
        required_paths.extend(
            [simple_json_path, simple_ppm_path, template_path, thinggraph_path]
        )
    return all(path.exists() for path in required_paths)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def annotate_existing_scene_map_metadata(
    output_prefix: str | Path,
    map_metadata: dict[str, object],
) -> None:
    """Patch map/source identity fields into existing JSON outputs during resume."""
    prefix = Path(output_prefix)
    _maps_path, metadata_path, _overview_path = expected_scene_output_paths(prefix)
    simple_json_path, _simple_png_path, _simple_ppm_path, _template_path, thinggraph_path = (
        expected_simple_demo_output_paths(prefix)
    )
    fields = {
        key: value
        for key, value in map_metadata.items()
        if value is not None
    }

    def load_json(path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    metadata_payload = load_json(metadata_path)
    if metadata_payload is not None:
        metadata_payload.update(fields)
        metadata_path.write_text(
            json.dumps(_json_ready(metadata_payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    simple_payload = load_json(simple_json_path)
    if simple_payload is not None:
        simple_metadata = simple_payload.get("metadata")
        if not isinstance(simple_metadata, dict):
            simple_metadata = {}
            simple_payload["metadata"] = simple_metadata
        simple_metadata.update(fields)
        simple_json_path.write_text(
            json.dumps(_json_ready(simple_payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    thinggraph_payload = load_json(thinggraph_path)
    if thinggraph_payload is not None:
        thinggraph_payload.update(fields)
        thinggraph_path.write_text(
            json.dumps(_json_ready(thinggraph_payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    simple_layers = simple_payload.get("layers") if isinstance(simple_payload, dict) else None
    if isinstance(simple_layers, dict) and isinstance(simple_layers.get("occupancy"), list):
        # Metadata changes update the source signature stored by the cache.
        build_map_metric_cache(simple_json_path, simple_payload)


def annotate_existing_scene_benchmark_metadata(
    output_prefix: str | Path,
    benchmark_metadata: dict[str, object],
) -> None:
    """Backward-compatible alias for existing export metadata annotation."""
    annotate_existing_scene_map_metadata(output_prefix, benchmark_metadata)


def _normalize_simple_category(value: object, fallback: str = "unknown") -> str:
    raw = str(value).strip() if value is not None else fallback
    if not raw:
        raw = fallback
    raw = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", raw)
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw)
    raw = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    normalized = raw or fallback
    return SIMPLE_CATEGORY_ALIASES.get(normalized, normalized)


def _simple_label(category: str) -> str:
    return " ".join(part.capitalize() for part in category.split("_")) or "Unknown"


def _generated_color(category: str, *, saturation: float, value: float) -> str:
    digest = hashlib.sha1(category.encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"


def _legend_for_categories(
    categories: set[str],
    known_colors: dict[str, str],
    *,
    saturation: float,
    value: float,
) -> dict[str, dict[str, str]]:
    return {
        category: {
            "label": _simple_label(category),
            "color": known_colors.get(
                category,
                _generated_color(category, saturation=saturation, value=value),
            ),
        }
        for category in sorted(categories)
    }


def _attribute(label: str, value: object) -> str | None:
    if value is None:
        return None
    return f"{label}={value}"


def _pad_to_square(array: np.ndarray, fill_value: int) -> np.ndarray:
    height, width = array.shape
    grid_size = max(height, width)
    padded = np.full((grid_size, grid_size), fill_value=fill_value, dtype=np.int32)
    padded[:height, :width] = array.astype(np.int32)
    return padded


def _room_instances_and_layer(
    scene_representation: dict[str, object],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    map_info = scene_representation["map_info"]
    height = int(map_info["H"])  # type: ignore[index]
    width = int(map_info["W"])  # type: ignore[index]
    room_layer = np.zeros((height, width), dtype=np.int32)
    room_instances: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}

    room_metadata = scene_representation.get("room_metadata", [])
    if not isinstance(room_metadata, list):
        return room_layer, room_instances

    for record in room_metadata:
        if not isinstance(record, dict):
            continue
        raw_room_type = record.get("room_type")
        category = _normalize_simple_category(raw_room_type, fallback="room")
        category_counts[category] = category_counts.get(category, 0) + 1
        instance_id = len(room_instances) + 1
        room_name = f"{category}_{category_counts[category]}"
        attributes = [
            item
            for item in (
                _attribute("procthor_room_id", record.get("room_id")),
                _attribute("procthor_room_type", raw_room_type),
                _attribute("area", record.get("area")),
                _attribute("grid_area", record.get("grid_area")),
            )
            if item is not None
        ]
        room_instances.append(
            {
                "id": instance_id,
                "category": category,
                "name": room_name,
                "attributes": attributes,
            }
        )

        polygon = record.get("floor_polygon_xz")
        if not isinstance(polygon, list):
            continue
        polygon_xz: list[tuple[float, float]] = []
        for point in polygon:
            if (
                isinstance(point, list)
                and len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
            ):
                polygon_xz.append((float(point[0]), float(point[1])))
        for row, col in rasterize_polygon_to_grid(polygon_xz, map_info):  # type: ignore[arg-type]
            if room_layer[row, col] == 0:
                room_layer[row, col] = instance_id

    return room_layer, room_instances


def _object_instances_from_metadata(
    scene_representation: dict[str, object],
) -> list[dict[str, object]]:
    object_metadata = scene_representation.get("object_metadata", [])
    if not isinstance(object_metadata, list):
        return []

    instances: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    seen_ids: set[int] = set()
    for fallback_id, record in enumerate(object_metadata, start=1):
        if not isinstance(record, dict):
            continue
        raw_instance_id = record.get("instance_id", fallback_id)
        if not isinstance(raw_instance_id, (int, np.integer)) or int(raw_instance_id) <= 0:
            continue
        instance_id = int(raw_instance_id)
        if instance_id in seen_ids:
            continue

        raw_category = (
            record.get("objectType")
            or record.get("category")
            or record.get("objectId")
            or record.get("id")
            or "object"
        )
        category = _normalize_simple_category(raw_category, fallback="object")
        category_counts[category] = category_counts.get(category, 0) + 1
        raw_name = record.get("name")
        name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else ""
        if not name:
            name = f"{category}_{category_counts[category]}"

        attributes = [
            item
            for item in (
                _attribute("procthor_object_id", record.get("objectId") or record.get("id")),
                _attribute("procthor_object_type", record.get("objectType") or record.get("category")),
                _attribute("asset_id", record.get("assetId")),
                _attribute("footprint_source", record.get("footprint_source")),
                _attribute("num_grid_cells", record.get("num_grid_cells")),
                _attribute("footprint_area", record.get("footprint_area")),
            )
            if item is not None
        ]
        instances.append(
            {
                "id": instance_id,
                "category": category,
                "name": name,
                "attributes": attributes,
            }
        )
        seen_ids.add(instance_id)

    return sorted(instances, key=lambda item: int(item["id"]))


def scene_representation_to_simple_demo_state(
    scene_representation: dict[str, object],
    map_id: str,
) -> dict[str, object]:
    """Convert ProcTHOR layers into the simple_demo layered-map JSON shape."""
    traversibility_map = np.asarray(scene_representation["traversibility_map"])
    object_instance_map = np.asarray(scene_representation["object_instance_map"])
    occupancy = np.where(
        traversibility_map > 0,
        SIMPLE_DEMO_OCCUPANCY_TILES["free"]["value"],
        SIMPLE_DEMO_OCCUPANCY_TILES["obstacle"]["value"],
    ).astype(np.int32)
    room_layer, room_instances = _room_instances_and_layer(scene_representation)
    object_instances = _object_instances_from_metadata(scene_representation)

    occupancy = _pad_to_square(occupancy, SIMPLE_DEMO_OCCUPANCY_TILES["obstacle"]["value"])
    room_layer = _pad_to_square(room_layer, 0)
    object_instance_map = _pad_to_square(object_instance_map, 0)
    grid_size = int(occupancy.shape[0])
    original_height = int(traversibility_map.shape[0])
    original_width = int(traversibility_map.shape[1])

    room_categories = {
        str(instance["category"])
        for instance in room_instances
        if isinstance(instance.get("category"), str)
    }
    object_categories = {
        str(instance["category"])
        for instance in object_instances
        if isinstance(instance.get("category"), str)
    }
    map_info = scene_representation.get("map_info", {})

    return {
        "metadata": {
            "map_id": map_id,
            "name": f"ProcTHOR {map_id}",
            "description": "ProcTHOR scene converted into simple_demo layered map format.",
            "source": "procthor",
            "map_split": scene_representation.get("map_split"),
            "map_index": scene_representation.get("map_index"),
            "map_assignment_basis": scene_representation.get("map_assignment_basis"),
            "map_split_rule": scene_representation.get("map_split_rule"),
            "procthor_split": scene_representation.get("procthor_split"),
            "procthor_index": scene_representation.get("procthor_index"),
            "procthor_scene_id": scene_representation.get("procthor_scene_id"),
            "scene_size": scene_representation.get("scene_size"),
            "original_grid_shape": [original_height, original_width],
            "padded_grid_size": grid_size,
            "map_info": _json_ready(map_info),
        },
        "layers": {
            "occupancy": occupancy.astype(int).tolist(),
            "room": room_layer.astype(int).tolist(),
            "object_instance": object_instance_map.astype(int).tolist(),
        },
        "room_instances": room_instances,
        "object_instances": object_instances,
        "layer_legends": {
            "occupancy": SIMPLE_DEMO_OCCUPANCY_TILES,
            "room_categories": _legend_for_categories(
                room_categories,
                SIMPLE_DEMO_ROOM_COLORS,
                saturation=0.32,
                value=0.95,
            ),
            "object_categories": _legend_for_categories(
                object_categories,
                SIMPLE_DEMO_OBJECT_COLORS,
                saturation=0.52,
                value=0.86,
            ),
        },
        "grid_size": grid_size,
    }


def _simple_demo_payload(state: dict[str, object]) -> dict[str, object]:
    return {
        "grid_size": state["grid_size"],
        "layer_legends": state["layer_legends"],
        "metadata": state["metadata"],
        "layers": state["layers"],
        "room_instances": state["room_instances"],
        "object_instances": state["object_instances"],
    }


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    cleaned = color.lstrip("#")
    return tuple(int(cleaned[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _render_simple_demo_preview_rgb(
    state: dict[str, object],
) -> list[tuple[int, int, int]]:
    layers = state["layers"]  # type: ignore[assignment]
    occupancy = layers["occupancy"]  # type: ignore[index]
    room = layers["room"]  # type: ignore[index]
    object_grid = layers["object_instance"]  # type: ignore[index]
    legends = state["layer_legends"]  # type: ignore[assignment]
    occupancy_legends = legends["occupancy"]  # type: ignore[index]
    room_categories = legends["room_categories"]  # type: ignore[index]
    object_categories = legends["object_categories"]  # type: ignore[index]
    occupancy_value_to_name = {
        int(tile["value"]): name for name, tile in occupancy_legends.items()
    }
    room_instance_map = {
        int(instance["id"]): instance for instance in state["room_instances"]  # type: ignore[union-attr]
    }
    object_instance_map = {
        int(instance["id"]): instance for instance in state["object_instances"]  # type: ignore[union-attr]
    }

    pixels: list[tuple[int, int, int]] = []
    grid_size = int(state["grid_size"])
    for row in range(grid_size):
        for col in range(grid_size):
            occupancy_name = occupancy_value_to_name.get(
                int(occupancy[row][col]),
                "unknown",
            )
            color = occupancy_legends.get(occupancy_name, occupancy_legends["unknown"])[
                "color"
            ]

            room_id = int(room[row][col])
            if room_id != 0 and occupancy_name == "free" and room_id in room_instance_map:
                category = room_instance_map[room_id]["category"]
                if isinstance(category, str) and category in room_categories:
                    color = room_categories[category]["color"]

            object_id = int(object_grid[row][col])
            if object_id != 0 and object_id in object_instance_map:
                category = object_instance_map[object_id]["category"]
                if isinstance(category, str) and category in object_categories:
                    color = object_categories[category]["color"]

            pixels.append(_hex_to_rgb(str(color)))
    return pixels


def _export_simple_demo_ppm(state: dict[str, object], output_path: Path) -> None:
    pixels = _render_simple_demo_preview_rgb(state)
    grid_size = int(state["grid_size"])
    with output_path.open("wb") as handle:
        handle.write(f"P6\n{grid_size} {grid_size}\n255\n".encode("ascii"))
        for pixel in pixels:
            handle.write(bytes(pixel))


def _export_simple_demo_png(state: dict[str, object], output_path: Path) -> None:
    pixels = _render_simple_demo_preview_rgb(state)
    grid_size = int(state["grid_size"])
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is optional at runtime.
        Image = None

    if Image is not None:
        image = Image.new("RGB", (grid_size, grid_size))
        image.putdata(pixels)
        image.save(output_path)
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_array = np.asarray(pixels, dtype=np.uint8).reshape((grid_size, grid_size, 3))
    plt.imsave(output_path, image_array)


def _collect_simple_demo_object_centers(
    state: dict[str, object],
) -> list[dict[str, object]]:
    object_grid = state["layers"]["object_instance"]  # type: ignore[index]
    object_cells: dict[int, list[tuple[int, int]]] = {}
    grid_size = int(state["grid_size"])
    for row in range(grid_size):
        for col in range(grid_size):
            object_id = int(object_grid[row][col])
            if object_id != 0:
                object_cells.setdefault(object_id, []).append((row, col))

    objects: list[dict[str, object]] = []
    for instance in state["object_instances"]:  # type: ignore[union-attr]
        object_id = int(instance["id"])
        cells = object_cells.get(object_id, [])
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


def _object_center_distance(
    first: dict[str, object],
    second: dict[str, object],
) -> float:
    first_center = first["center"]
    second_center = second["center"]
    return math.hypot(
        float(first_center[0]) - float(second_center[0]),  # type: ignore[index]
        float(first_center[1]) - float(second_center[1]),  # type: ignore[index]
    )


def _template_pair_distances(
    objects: list[dict[str, object]],
) -> dict[tuple[int, int], float]:
    distances: dict[tuple[int, int], float] = {}
    for first in objects:
        for second in objects:
            first_id = int(first["object_id"])
            second_id = int(second["object_id"])
            if first_id != second_id:
                distances[(first_id, second_id)] = _object_center_distance(first, second)
    return distances


def _template_diversity_signature(
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


def _sample_diverse_template_paths(
    objects: list[dict[str, object]],
    distances: dict[tuple[int, int], float],
    target_count: int,
    grid_size: int,
) -> list[list[dict[str, object]]]:
    rng = random.SystemRandom()
    paths: list[list[dict[str, object]]] = []
    seen_object_sets: set[tuple[int, ...]] = set()
    seen_signatures: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
    attempts = max(2500, SIMPLE_TEMPLATE_MAX_PER_OBJECT_COUNT * target_count * 80)

    def candidate_is_valid(
        path: list[dict[str, object]], candidate: dict[str, object]
    ) -> bool:
        candidate_id = int(candidate["object_id"])
        return all(
            distances[(int(item["object_id"]), candidate_id)]
            > SIMPLE_TEMPLATE_MIN_DISTANCE
            for item in path
        )

    for _attempt in range(attempts):
        if len(paths) >= SIMPLE_TEMPLATE_MAX_PER_OBJECT_COUNT:
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
        signature = _template_diversity_signature(path, grid_size)
        if signature in seen_signatures:
            continue
        seen_object_sets.add(object_set)
        seen_signatures.add(signature)
        paths.append(path)

    return paths


def _generate_simple_demo_template_instructions(
    map_id: str,
    state: dict[str, object],
) -> dict[str, object]:
    objects = _collect_simple_demo_object_centers(state)[:SIMPLE_TEMPLATE_MAX_OBJECTS]
    grid_size = int(state["grid_size"])
    distances = _template_pair_distances(objects)
    created_at = _iso_now()
    templates: list[dict[str, object]] = []

    for target_count in SIMPLE_TEMPLATE_OBJECT_COUNTS:
        sampled_paths = _sample_diverse_template_paths(
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
                    "map_id": map_id,
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
    distance_rule: dict[str, object] = {
        "metric": SIMPLE_TEMPLATE_GRID_METRIC,
        "minimum_exclusive": SIMPLE_TEMPLATE_MIN_DISTANCE,
    }

    return {
        "version": 1,
        "map_id": map_id,
        "distance_rule": distance_rule,
        "object_counts": list(SIMPLE_TEMPLATE_OBJECT_COUNTS),
        "randomized_order": True,
        "template_limits": {
            "max_objects_considered": SIMPLE_TEMPLATE_MAX_OBJECTS,
            "max_per_object_count": SIMPLE_TEMPLATE_MAX_PER_OBJECT_COUNT,
            "max_total_templates": SIMPLE_TEMPLATE_MAX_TOTAL,
            "spacing_scope": "all_object_pairs",
            "diversity_strategy": "random_spatial_category_sampling",
        },
        "templates": templates,
        "created_at": created_at,
        "updated_at": created_at,
    }


def _attribute_value(attributes: object, key: str) -> str | None:
    """Extract a key=value attribute generated for simple-demo instances."""
    if not isinstance(attributes, list):
        return None
    prefix = f"{key}="
    for attribute in attributes:
        if isinstance(attribute, str) and attribute.startswith(prefix):
            return attribute[len(prefix) :]
    return None


def _simple_grid_cells_by_instance(
    grid: object,
) -> dict[int, list[tuple[int, int]]]:
    cells_by_id: dict[int, list[tuple[int, int]]] = {}
    if not isinstance(grid, list):
        return cells_by_id
    for row_index, row in enumerate(grid):
        if not isinstance(row, list):
            continue
        for col_index, value in enumerate(row):
            try:
                instance_id = int(value)
            except (TypeError, ValueError):
                continue
            if instance_id != 0:
                cells_by_id.setdefault(instance_id, []).append((row_index, col_index))
    return cells_by_id


def _dominant_room_for_object_cells(
    cells: list[tuple[int, int]],
    room_layer: object,
) -> tuple[int | None, int, dict[int, int]]:
    room_counts: dict[int, int] = {}
    if not isinstance(room_layer, list):
        return None, 0, room_counts
    for row, col in cells:
        if row < 0 or row >= len(room_layer):
            continue
        room_row = room_layer[row]
        if not isinstance(room_row, list) or col < 0 or col >= len(room_row):
            continue
        try:
            room_id = int(room_row[col])
        except (TypeError, ValueError):
            continue
        if room_id != 0:
            room_counts[room_id] = room_counts.get(room_id, 0) + 1
    if not room_counts:
        return None, 0, room_counts
    room_id, overlap = max(room_counts.items(), key=lambda item: (item[1], -item[0]))
    return room_id, overlap, room_counts


def _room_at_grid_cell(room_layer: object, row: object, col: object) -> int | None:
    if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
        return None
    row_index = int(row)
    col_index = int(col)
    if not isinstance(room_layer, list) or row_index < 0 or row_index >= len(room_layer):
        return None
    room_row = room_layer[row_index]
    if not isinstance(room_row, list) or col_index < 0 or col_index >= len(room_row):
        return None
    try:
        room_id = int(room_row[col_index])
    except (TypeError, ValueError):
        return None
    return room_id if room_id != 0 else None


def _room_adjacency_edges(room_layer: object) -> list[dict[str, object]]:
    if not isinstance(room_layer, list):
        return []
    edge_counts: dict[tuple[int, int], int] = {}
    for row_index, row in enumerate(room_layer):
        if not isinstance(row, list):
            continue
        for col_index, value in enumerate(row):
            try:
                room_id = int(value)
            except (TypeError, ValueError):
                continue
            if room_id == 0:
                continue
            for d_row, d_col in ((1, 0), (0, 1)):
                next_row_index = row_index + d_row
                next_col_index = col_index + d_col
                if next_row_index < 0 or next_row_index >= len(room_layer):
                    continue
                next_row = room_layer[next_row_index]
                if (
                    not isinstance(next_row, list)
                    or next_col_index < 0
                    or next_col_index >= len(next_row)
                ):
                    continue
                try:
                    next_room_id = int(next_row[next_col_index])
                except (TypeError, ValueError):
                    continue
                if next_room_id == 0 or next_room_id == room_id:
                    continue
                edge_key = tuple(sorted((room_id, next_room_id)))
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
    return [
        {
            "source_room_id": first,
            "target_room_id": second,
            "touching_grid_edges": count,
        }
        for (first, second), count in sorted(edge_counts.items())
    ]


def build_thinggraph_payload(
    scene_representation: dict[str, object],
    state: dict[str, object],
    map_id: str,
    *,
    files: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build a room-object graph aligned with simple-demo ids and ProcTHOR metadata."""
    layers = state["layers"]  # type: ignore[assignment]
    room_layer = layers["room"]  # type: ignore[index]
    object_grid = layers["object_instance"]  # type: ignore[index]
    object_cells = _simple_grid_cells_by_instance(object_grid)

    raw_room_metadata = scene_representation.get("room_metadata", [])
    raw_object_metadata = scene_representation.get("object_metadata", [])
    raw_rooms_by_room_id = {
        str(record.get("room_id")): (metadata_index, record)
        for metadata_index, record in enumerate(raw_room_metadata)
        if isinstance(record, dict) and record.get("room_id") is not None
    } if isinstance(raw_room_metadata, list) else {}
    raw_objects_by_instance_id = {
        int(record["instance_id"]): (metadata_index, record)
        for metadata_index, record in enumerate(raw_object_metadata)
        if (
            isinstance(record, dict)
            and isinstance(record.get("instance_id"), (int, np.integer))
        )
    } if isinstance(raw_object_metadata, list) else {}

    rooms_by_id: dict[int, dict[str, object]] = {}
    for room in state["room_instances"]:  # type: ignore[union-attr]
        if not isinstance(room, dict):
            continue
        room_id = int(room["id"])
        procthor_room_id = _attribute_value(room.get("attributes"), "procthor_room_id")
        metadata_index = None
        raw_room: dict[str, object] = {}
        if procthor_room_id in raw_rooms_by_room_id:
            metadata_index, raw_room = raw_rooms_by_room_id[procthor_room_id]
        rooms_by_id[room_id] = {
            "id": room_id,
            "node_id": f"room:{room_id}",
            "category": room.get("category"),
            "name": room.get("name"),
            "procthor_room_id": procthor_room_id,
            "procthor_room_type": raw_room.get("room_type")
            or _attribute_value(room.get("attributes"), "procthor_room_type"),
            "metadata_index": metadata_index,
            "area": raw_room.get("area"),
            "grid_area": raw_room.get("grid_area"),
            "bbox": _json_ready(raw_room.get("bbox")),
            "centroid_xz": _json_ready(raw_room.get("centroid_xz")),
            "objects": [],
        }

    unassigned_objects: list[dict[str, object]] = []
    for instance in state["object_instances"]:  # type: ignore[union-attr]
        if not isinstance(instance, dict):
            continue
        object_id = int(instance["id"])
        metadata_index = None
        raw_object: dict[str, object] = {}
        if object_id in raw_objects_by_instance_id:
            metadata_index, raw_object = raw_objects_by_instance_id[object_id]

        cells = object_cells.get(object_id, [])
        room_id, overlap_count, room_overlap_counts = _dominant_room_for_object_cells(
            cells,
            room_layer,
        )
        assignment_method = "footprint_overlap" if room_id is not None else None
        if room_id is None:
            room_id = _room_at_grid_cell(
                room_layer,
                raw_object.get("grid_row"),
                raw_object.get("grid_col"),
            )
            assignment_method = "metadata_grid_cell" if room_id is not None else None

        center_grid = None
        if cells:
            center_grid = [
                round(sum(row for row, _col in cells) / len(cells), 3),
                round(sum(col for _row, col in cells) / len(cells), 3),
            ]
        elif isinstance(raw_object.get("grid_row"), (int, np.integer)) and isinstance(
            raw_object.get("grid_col"), (int, np.integer)
        ):
            center_grid = [int(raw_object["grid_row"]), int(raw_object["grid_col"])]

        graph_object = {
            "id": object_id,
            "node_id": f"object:{object_id}",
            "category": instance.get("category"),
            "name": instance.get("name"),
            "procthor_object_id": raw_object.get("objectId")
            or _attribute_value(instance.get("attributes"), "procthor_object_id"),
            "procthor_object_type": raw_object.get("objectType")
            or _attribute_value(instance.get("attributes"), "procthor_object_type"),
            "metadata_index": metadata_index,
            "room_id": room_id,
            "assignment_method": assignment_method,
            "room_overlap_grid_cells": overlap_count,
            "room_overlap_counts": {
                str(key): value for key, value in sorted(room_overlap_counts.items())
            },
            "num_grid_cells": len(cells) if cells else raw_object.get("num_grid_cells"),
            "center_grid": center_grid,
            "metadata_grid_cell": (
                [int(raw_object["grid_row"]), int(raw_object["grid_col"])]
                if isinstance(raw_object.get("grid_row"), (int, np.integer))
                and isinstance(raw_object.get("grid_col"), (int, np.integer))
                else None
            ),
            "position": _json_ready(raw_object.get("position")),
            "footprint_source": raw_object.get("footprint_source")
            or _attribute_value(instance.get("attributes"), "footprint_source"),
        }

        if room_id is not None and room_id in rooms_by_id:
            rooms_by_id[room_id]["objects"].append(graph_object)  # type: ignore[union-attr]
        else:
            unassigned_objects.append(graph_object)

    rooms = list(rooms_by_id.values())
    for room in rooms:
        objects = room["objects"]
        if isinstance(objects, list):
            objects.sort(key=lambda item: int(item["id"]))
            room["object_count"] = len(objects)

    return {
        "version": 1,
        "graph_type": "room_object_thinggraph",
        "map_id": map_id,
        "map_split": scene_representation.get("map_split"),
        "map_index": scene_representation.get("map_index"),
        "map_assignment_basis": scene_representation.get("map_assignment_basis"),
        "map_split_rule": _json_ready(scene_representation.get("map_split_rule")),
        "procthor_split": scene_representation.get("procthor_split"),
        "procthor_index": scene_representation.get("procthor_index"),
        "procthor_scene_id": scene_representation.get("procthor_scene_id"),
        "created_at": _iso_now(),
        "files": files or {},
        "id_semantics": {
            "room.id": "simple-demo room instance id used in layers.room",
            "object.id": "simple-demo object instance id used in layers.object_instance",
            "procthor_room_id": "room_id from ProcTHOR room_metadata",
            "procthor_object_id": "objectId from ProcTHOR object_metadata",
            "metadata_index": "index into the corresponding *_metadata list in <map_id>_metadata.json",
        },
        "summary": {
            "room_count": len(rooms),
            "assigned_object_count": sum(
                int(room.get("object_count", 0)) for room in rooms
            ),
            "unassigned_object_count": len(unassigned_objects),
        },
        "rooms": sorted(rooms, key=lambda item: int(item["id"])),
        "room_edges": _room_adjacency_edges(room_layer),
        "unassigned_objects": sorted(
            unassigned_objects,
            key=lambda item: int(item["id"]),
        ),
    }


def save_simple_demo_scene_representation(
    scene_representation: dict[str, object],
    output_prefix: str | Path,
    *,
    map_id: str | None = None,
) -> dict[str, Path]:
    """Save ProcTHOR output in the simple_demo layered-map file layout."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    simple_json_path, simple_png_path, simple_ppm_path, template_path, thinggraph_path = (
        expected_simple_demo_output_paths(prefix)
    )
    safe_map_id = _normalize_simple_category(map_id or prefix.name, fallback=prefix.name)
    state = scene_representation_to_simple_demo_state(scene_representation, safe_map_id)
    simple_json_path.write_text(
        json.dumps(_json_ready(_simple_demo_payload(state)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _export_simple_demo_ppm(state, simple_ppm_path)
    _export_simple_demo_png(state, simple_png_path)
    template_path.write_text(
        json.dumps(
            _json_ready(_generate_simple_demo_template_instructions(safe_map_id, state)),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    thinggraph_path.write_text(
        json.dumps(
            _json_ready(
                build_thinggraph_payload(
                    scene_representation,
                    state,
                    safe_map_id,
                    files={
                        "simple_map_json": simple_json_path.name,
                        "metadata_json": f"{prefix.name}_metadata.json",
                        "maps_npz": f"{prefix.name}_maps.npz",
                        "template_instruction_json": template_path.name,
                    },
                )
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Map-level metric artifacts are derived solely from this layered map and
    # are therefore safe to build at export time for every later instruction.
    build_map_metric_cache(simple_json_path, state)
    return {
        "json_path": simple_json_path,
        "png_path": simple_png_path,
        "ppm_path": simple_ppm_path,
        "template_instruction_path": template_path,
        "thinggraph_path": thinggraph_path,
    }


def capture_topdown_frame(controller: object) -> np.ndarray:
    """Capture a top-down RGB view from AI2-THOR."""
    event = controller.step(action="GetMapViewCameraProperties")
    camera_pose = copy.deepcopy(event.metadata["actionReturn"])
    camera_pose["orthographic"] = True
    camera_pose["skyboxColor"] = "white"

    event = controller.step(action="AddThirdPartyCamera", **camera_pose)
    return event.third_party_camera_frames[-1]


def save_map_overview(
    scene_representation: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Save one PNG containing the exported traversibility, room, and object maps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    map_data = {
        "traversibility_map": scene_representation["traversibility_map"],
        "room_type_map": scene_representation["room_type_map"],
        "object_category_map": scene_representation["object_category_map"],
        "object_instance_map": scene_representation["object_instance_map"],
    }
    metadata = {
        "map_info": scene_representation["map_info"],
        "object_metadata": scene_representation["object_metadata"],
        "room_metadata": scene_representation.get("room_metadata", []),
        "room_type_to_id": scene_representation["room_type_to_id"],
        "category_to_id": scene_representation["category_to_id"],
    }
    for key in (
        "map_id",
        "map_split",
        "map_index",
        "map_assignment_basis",
        "map_split_rule",
        "procthor_split",
        "procthor_index",
        "procthor_scene_id",
    ):
        if scene_representation.get(key) is not None:
            metadata[key] = scene_representation[key]
    figure, _axes = plot_scene_export(map_data, metadata)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_four_panel_scene_view(
    controller: object,
    scene_representation: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Save one PNG containing top-down scene, traversibility, room, and object maps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topdown_frame = capture_topdown_frame(controller)
    traversibility_map = scene_representation["traversibility_map"]
    room_type_map = scene_representation["room_type_map"]
    object_category_map = scene_representation["object_category_map"]
    room_type_to_id = scene_representation["room_type_to_id"]

    room_id_to_type = invert_mapping(room_type_to_id)
    category_id_to_name = _category_id_to_name(
        {"category_to_id": scene_representation["category_to_id"]}
    )

    figure, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    axes[0].imshow(topdown_frame)
    axes[0].set_title("Top-Down Scene View")
    axes[0].axis("off")

    axes[1].imshow(traversibility_map, origin="lower", cmap="gray")
    axes[1].set_title("Traversibility Map")

    room_display = np.where(room_type_map >= 0, room_type_map, np.nan)
    axes[2].imshow(room_display, origin="lower", cmap="tab20")
    axes[2].set_title("Room Type Map")
    for room_id, room_name in room_id_to_type.items():
        rows, cols = np.where(room_type_map == room_id)
        if len(rows) == 0:
            continue
        axes[2].text(
            int(np.round(cols.mean())),
            int(np.round(rows.mean())),
            room_name,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.65, "pad": 1},
        )

    draw_object_overlay(
        axes[3],
        traversibility_map,
        object_category_map,
        category_id_to_name,
    )

    for axis in axes[1:]:
        axis.set_xlabel("Grid Col (x)")
        axis.set_ylabel("Grid Row (z)")

    map_info = scene_representation["map_info"]
    title_lines = [
        "ProcTHOR Scene Overview",
        f"resolution={map_info['resolution']}  size=({map_info['H']}, {map_info['W']})",
    ]
    map_split = scene_representation.get("map_split")
    if map_split:
        title_lines.append(f"map_split={map_split}")
    room_size_summary = format_room_size_summary(
        scene_representation.get("room_metadata", [])
    )
    if room_size_summary:
        title_lines.append(room_size_summary)
    figure.suptitle(
        "\n".join(title_lines),
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def transform_procthor_to_map(
    split: str,
    index: int,
    *,
    dataset_name: str = "procthor-10k",
    resolution: float = 0.25,
    padding: float = 1.0,
    output_prefix: str | Path | None = None,
    save_overview: bool = True,
    save_simple_demo: bool = True,
    use_simulator: bool = True,
    ai2thor_base_dir: str | Path | None = DEFAULT_AI2THOR_BASE_DIR,
    dataset=None,
    controller_class=None,
    controller_kwargs: dict[str, object] | None = None,
    map_position: int | None = None,
    benchmark_position: int | None = None,
) -> tuple[dict[str, object], Path, Path, Path | None]:
    """Load one ProcTHOR house, export its maps, and optionally save an overview PNG."""
    if map_position is None:
        map_position = benchmark_position
    controller_kwargs = dict(controller_kwargs or {})
    house = load_procthor_house(
        split=split,
        index=index,
        dataset_name=dataset_name,
        dataset=dataset,
    )
    prefix = (
        Path(output_prefix)
        if output_prefix is not None
        else build_default_output_prefix(split, index)
    )
    map_metadata = build_scene_map_metadata(
        split,
        index,
        sequence_position=map_position,
    )

    controller = None
    if use_simulator:
        controller_kwargs["gridSize"] = resolution
        controller = create_procthor_controller(
            house,
            controller_class=controller_class,
            ai2thor_base_dir=ai2thor_base_dir,
            **controller_kwargs,
        )

    try:
        scene_representation = convert_procthor_house_to_maps(
            house,
            controller=controller,
            resolution=resolution,
            padding=padding,
        )
        scene_representation.update(map_metadata)
        maps_path, metadata_path = save_scene_representation(scene_representation, prefix)
        if save_simple_demo:
            save_simple_demo_scene_representation(
                scene_representation,
                prefix,
                map_id=str(scene_representation.get("map_id") or prefix.name),
            )
        overview_path = None
        if save_overview:
            if controller is None:
                overview_path = save_map_overview(
                    scene_representation,
                    _default_png_path(prefix, None),
                )
            else:
                overview_path = save_four_panel_scene_view(
                    controller,
                    scene_representation,
                    _default_png_path(prefix, None),
                )
        return scene_representation, maps_path, metadata_path, overview_path
    finally:
        stop = getattr(controller, "stop", None)
        if callable(stop):
            stop()


def transform_procthor_batch(
    split: str,
    *,
    number: int,
    scene_size: float | None = None,
    sequence: str = "sequence",
    resume: bool = False,
    dataset_name: str = "procthor-10k",
    resolution: float = 0.25,
    padding: float = 1.0,
    output_prefix: str | Path | None = None,
    save_overview: bool = True,
    save_simple_demo: bool = True,
    ai2thor_base_dir: str | Path | None = DEFAULT_AI2THOR_BASE_DIR,
    dataset=None,
    controller_class=None,
    controller_kwargs: dict[str, object] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Scan a split from index 0 and export `number` scenes that pass filters."""
    sequence = normalize_sequence_mode(sequence)
    if number <= 0:
        raise ValueError("number must be positive when --index is -1.")
    if scene_size is not None and scene_size < 0:
        raise ValueError("scene_size must be non-negative.")

    if dataset is None:
        dataset = load_procthor_dataset(dataset_name)
    houses = _get_split_houses(dataset, split)

    if sequence == "bigscene":
        candidate_records: list[dict[str, float | int]] = []
        for candidate_index in range(len(houses)):  # type: ignore[arg-type]
            house = load_procthor_house(
                split=split,
                index=candidate_index,
                dataset_name=dataset_name,
                dataset=dataset,
            )
            room_sizes = compute_house_room_sizes(house)
            candidate_records.append(
                {
                    "index": candidate_index,
                    "scene_size": sum(room_sizes),
                    "largest_room_size": max(room_sizes, default=0.0),
                }
            )
        candidate_records.sort(
            key=lambda record: (
                -float(record["scene_size"]),
                -float(record["largest_room_size"]),
                int(record["index"]),
            )
        )
    else:
        candidate_records = [
            {"index": candidate_index, "scene_size": -1.0, "largest_room_size": -1.0}
            for candidate_index in range(len(houses))  # type: ignore[arg-type]
        ]

    results: dict[str, list[dict[str, object]]] = {"processed": [], "skipped": []}
    eligible_position = 0
    for candidate_record in candidate_records:
        candidate_index = int(candidate_record["index"])
        house = load_procthor_house(
            split=split,
            index=candidate_index,
            dataset_name=dataset_name,
            dataset=dataset,
        )
        current_scene_size = float(candidate_record["scene_size"])
        if current_scene_size < 0:
            current_scene_size = compute_house_scene_size(house)
        largest_room_size = float(candidate_record["largest_room_size"])
        if largest_room_size < 0:
            largest_room_size = compute_house_largest_room_size(house)
        if scene_size is not None and current_scene_size < scene_size:
            results["skipped"].append(
                {
                    "index": candidate_index,
                    "scene_size": current_scene_size,
                    "largest_room_size": largest_room_size,
                    "reason": "scene_size",
                }
            )
            continue

        eligible_position += 1
        map_split = map_split_for_sequence_position(eligible_position)
        map_id = build_map_id(eligible_position, map_split)
        procthor_scene_id = build_procthor_scene_id(split, candidate_index)
        per_scene_prefix = build_batch_output_prefix(
            split,
            candidate_index,
            output_prefix,
            map_index=eligible_position,
            map_split=map_split,
        )
        if resume and is_scene_export_complete(
            per_scene_prefix,
            require_overview=save_overview,
            require_simple_demo=save_simple_demo,
        ):
            map_metadata = build_scene_map_metadata(
                split,
                candidate_index,
                sequence_position=eligible_position,
            )
            annotate_existing_scene_map_metadata(
                per_scene_prefix,
                map_metadata,
            )
            skipped_record = {
                "index": candidate_index,
                "map_id": map_id,
                "map_split": map_split,
                "procthor_scene_id": procthor_scene_id,
                "scene_size": current_scene_size,
                "largest_room_size": largest_room_size,
                "reason": "resume",
            }
            results["skipped"].append(skipped_record)
            if progress_callback is not None:
                progress_callback(
                    {
                        **skipped_record,
                        "event": "resume_skip",
                        "position": eligible_position,
                        "target": number,
                    }
                )
            if eligible_position >= number:
                break
            continue

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "processing",
                    "index": candidate_index,
                    "map_id": map_id,
                    "map_split": map_split,
                    "procthor_scene_id": procthor_scene_id,
                    "position": eligible_position,
                    "target": number,
                    "scene_size": current_scene_size,
                    "largest_room_size": largest_room_size,
                }
            )

        _scene_representation, maps_path, metadata_path, overview_path = (
            transform_procthor_to_map(
                split=split,
                index=candidate_index,
                dataset_name=dataset_name,
                resolution=resolution,
                padding=padding,
                output_prefix=per_scene_prefix,
                save_overview=save_overview,
                save_simple_demo=save_simple_demo,
                use_simulator=True,
                ai2thor_base_dir=ai2thor_base_dir,
                dataset=dataset,
                controller_class=controller_class,
                controller_kwargs=controller_kwargs,
                map_position=eligible_position,
            )
        )
        processed_record = {
            "index": candidate_index,
            "map_id": map_id,
            "map_split": map_split,
            "procthor_scene_id": procthor_scene_id,
            "scene_size": current_scene_size,
            "largest_room_size": largest_room_size,
            "maps_path": maps_path,
            "metadata_path": metadata_path,
            "overview_path": overview_path,
        }
        results["processed"].append(processed_record)
        if progress_callback is not None:
            progress_callback(
                {
                    **processed_record,
                    "event": "saved",
                    "position": eligible_position,
                    "target": number,
                }
            )
        if eligible_position >= number:
            break

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform one ProcTHOR house into SemPathBench maps."
    )
    parser.add_argument("--split", default="train", help="Dataset split, e.g. train or val.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="House index inside the split. Use -1 to scan from the beginning.",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=1,
        help="Number of accepted scenes to export when --index is -1.",
    )
    parser.add_argument(
        "--scene-size",
        "--scene_size",
        dest="scene_size",
        type=float,
        default=None,
        help=(
            "Minimum true scene size, measured as total room floorPolygon area. "
            "Scenes below this threshold are skipped and do not count toward --number."
        ),
    )
    parser.add_argument(
        "--sequence",
        type=normalize_sequence_mode,
        default="sequence",
        help=(
            "Batch traversal strategy when --index is -1. Use 'sequence' for "
            "dataset order or 'bigscene' to visit scenes by total scene area first."
        ),
    )
    parser.add_argument(
        "--resume",
        "--resumet",
        dest="resume",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Skip scenes whose requested outputs already exist. Accepts true/false; "
            "passing --resume without a value means true."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default="procthor-10k",
        help="Dataset name passed to prior.load_dataset.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.25,
        help="Map grid resolution in ProcTHOR world units.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="Extra world-space padding added around the map bounds.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output path prefix for single-scene mode. In batch mode, this is treated "
            "as an output directory. Defaults to resources/maps/procthor/<split>/<map_id>/<map_id>."
        ),
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Export maps only, without generating the overview PNG.",
    )
    parser.add_argument(
        "--skip-simple-demo",
        action="store_true",
        help="Do not export the additional simple_demo-style JSON/PPM/PNG files.",
    )
    parser.add_argument(
        "--ai2thor-base-dir",
        type=Path,
        default=DEFAULT_AI2THOR_BASE_DIR,
        help=(
            "Directory used by AI2-THOR for tmp/releases/cache."
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    dataset=None,
    controller_class=None,
    controller_kwargs: dict[str, object] | None = None,
) -> None:
    args = parse_args(argv)
    if args.index < -1:
        raise ValueError("--index must be -1 or a non-negative integer.")

    if args.index == -1:
        print(
            "Starting batch: "
            f"split={args.split}, sequence={args.sequence}, target={args.number}, "
            f"resume={args.resume}"
        )

        def print_batch_progress(record: dict[str, object]) -> None:
            map_id = str(record["map_id"])
            if record["event"] == "processing":
                print(
                    "Processing "
                    f"{map_id} "
                    f"({int(record['position'])}/{int(record['target'])}, "
                    f"procthor_scene_id={record['procthor_scene_id']}, "
                    f"scene_size={float(record['scene_size']):.2f}, "
                    f"largest_room={float(record['largest_room_size']):.2f}, "
                    f"map_split={record['map_split']})"
                )
                return
            if record["event"] == "saved":
                print(
                    f"Saved {map_id} -> {record['metadata_path']} "
                    f"(procthor_scene_id={record['procthor_scene_id']}, "
                    f"map_split={record['map_split']})"
                )
                return
            if record["event"] == "resume_skip":
                print(
                    "Skip "
                    f"{map_id} "
                    f"({int(record['position'])}/{int(record['target'])}, already exists) "
                    f"procthor_scene_id={record['procthor_scene_id']} "
                    f"map_split={record['map_split']}"
                )

        batch_results = transform_procthor_batch(
            split=args.split,
            number=args.number,
            scene_size=args.scene_size,
            sequence=args.sequence,
            resume=args.resume,
            dataset_name=args.dataset_name,
            resolution=args.resolution,
            padding=args.padding,
            output_prefix=args.output_prefix,
            save_overview=not args.skip_overview,
            save_simple_demo=not args.skip_simple_demo,
            ai2thor_base_dir=args.ai2thor_base_dir,
            dataset=dataset,
            controller_class=controller_class,
            controller_kwargs=controller_kwargs,
            progress_callback=print_batch_progress,
        )
        print(
            f"Processed {len(batch_results['processed'])} accepted scenes; "
            f"skipped {len(batch_results['skipped'])} scenes."
        )
        return

    single_prefix = (
        Path(args.output_prefix)
        if args.output_prefix is not None
        else build_default_output_prefix(args.split, args.index)
    )
    if args.resume and is_scene_export_complete(
        single_prefix,
        require_overview=not args.skip_overview,
        require_simple_demo=not args.skip_simple_demo,
    ):
        return

    loaded_dataset = dataset
    if args.scene_size is not None:
        if loaded_dataset is None:
            loaded_dataset = load_procthor_dataset(args.dataset_name)
        house = load_procthor_house(
            split=args.split,
            index=args.index,
            dataset_name=args.dataset_name,
            dataset=loaded_dataset,
        )
        current_scene_size = compute_house_scene_size(house)
        if current_scene_size < args.scene_size:
            return

    _scene_representation, maps_path, metadata_path, overview_path = transform_procthor_to_map(
        split=args.split,
        index=args.index,
        dataset_name=args.dataset_name,
        resolution=args.resolution,
        padding=args.padding,
        output_prefix=single_prefix,
        save_overview=not args.skip_overview,
        save_simple_demo=not args.skip_simple_demo,
        ai2thor_base_dir=args.ai2thor_base_dir,
        dataset=loaded_dataset,
        controller_class=controller_class,
        controller_kwargs=controller_kwargs,
    )
    print(f"Saved maps to {maps_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Map split: {map_split_for_source_index(args.index)}")
    if overview_path is not None:
        print(f"Saved overview to {overview_path}")


if __name__ == "__main__":
    main()
