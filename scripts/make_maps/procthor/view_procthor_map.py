#!/usr/bin/env python3
"""Visualize exported ProcTHOR SemPathBench maps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

FREE_SPACE_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float32)
OCCLUSION_COLOR = np.array([0.0, 0.0, 0.0], dtype=np.float32)
IGNORED_OBJECT_CATEGORIES = {"Floor"}


def load_scene_export(
    maps_path: str | Path,
    metadata_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load a saved ProcTHOR scene export from `.npz` and `.json`."""
    maps_path = Path(maps_path)
    metadata_path = Path(metadata_path)

    arrays = np.load(maps_path)
    map_data = {key: arrays[key] for key in arrays.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return map_data, metadata


def invert_mapping(mapping: dict[str, int]) -> dict[int, str]:
    """Invert a string-to-id mapping loaded from JSON."""
    return {int(value): key for key, value in mapping.items()}


def _object_label(record: dict[str, object]) -> str:
    for key in ("objectType", "category", "name", "objectId", "id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.split("|")[0]
    return "?"


def _instance_id_to_label(object_metadata: object) -> dict[int, str]:
    if not isinstance(object_metadata, list):
        return {}

    labels: dict[int, str] = {}
    for record in object_metadata:
        if not isinstance(record, dict):
            continue
        instance_id = record.get("instance_id")
        if isinstance(instance_id, int):
            labels[instance_id] = _object_label(record)
    return labels


def _category_id_to_name(metadata: dict[str, object]) -> dict[int, str]:
    category_to_id = metadata.get("category_to_id", {})
    return invert_mapping(category_to_id) if isinstance(category_to_id, dict) else {}


def _format_number(value: object) -> str | None:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return None


def format_room_size_summary(
    room_metadata: object,
    max_rooms: int = 4,
) -> str:
    """Format per-room true sizes for the overview title."""
    if not isinstance(room_metadata, list):
        return ""

    valid_records = [record for record in room_metadata if isinstance(record, dict)]
    if not valid_records:
        return ""

    labels: list[str] = []
    total_area = 0.0
    has_total_area = False
    for record in valid_records:
        area = record.get("area")
        if isinstance(area, (int, float)):
            total_area += float(area)
            has_total_area = True

    for index, record in enumerate(valid_records[:max_rooms]):
        room_type = record.get("room_type")
        room_label = room_type if isinstance(room_type, str) and room_type else f"room{index}"

        area = _format_number(record.get("area"))
        if area is None:
            continue

        bbox = record.get("bbox")
        width = depth = None
        if isinstance(bbox, dict):
            width = _format_number(bbox.get("width_x"))
            depth = _format_number(bbox.get("depth_z"))

        if width is not None and depth is not None:
            labels.append(f"{room_label}={area} ({width}x{depth})")
        else:
            labels.append(f"{room_label}={area}")

    if not labels:
        return ""

    remaining = len(valid_records) - len(labels)
    if remaining > 0 and has_total_area:
        labels.append(f"+{remaining} more; total={total_area:.2f}")
    elif remaining > 0:
        labels.append(f"+{remaining} more")
    elif len(valid_records) > 1 and has_total_area:
        labels.append(f"total={total_area:.2f}")

    return "room sizes: " + ", ".join(labels)


def _build_category_color_map(category_ids: list[int]) -> dict[int, tuple[float, float, float, float]]:
    """Assign a deterministic RGBA color to each object category id."""
    from matplotlib import colormaps

    palette_names = ("tab20", "tab20b", "tab20c")
    palette_colors: list[tuple[float, float, float, float]] = []
    for palette_name in palette_names:
        palette_colors.extend(list(colormaps[palette_name].colors))

    color_map: dict[int, tuple[float, float, float, float]] = {}
    for index, category_id in enumerate(sorted(category_ids)):
        color_map[category_id] = palette_colors[index % len(palette_colors)]
    return color_map


def render_object_map_image(
    traversibility_map: np.ndarray,
    object_category_map: np.ndarray,
    category_colors: dict[int, tuple[float, float, float, float]],
    ignored_category_ids: set[int],
) -> np.ndarray:
    """Render an RGB object map with white free space and black occlusion."""
    height, width = traversibility_map.shape
    image = np.empty((height, width, 3), dtype=np.float32)
    image[traversibility_map > 0] = FREE_SPACE_COLOR
    image[traversibility_map <= 0] = OCCLUSION_COLOR

    occupied_rows, occupied_cols = np.where(object_category_map > 0)
    for row, col in zip(occupied_rows.tolist(), occupied_cols.tolist()):
        category_id = int(object_category_map[row, col])
        if category_id in ignored_category_ids:
            continue
        color = category_colors.get(category_id)
        if color is None:
            continue
        image[row, col] = np.array(color[:3], dtype=np.float32)

    return image


def draw_object_overlay(
    axis,
    traversibility_map: np.ndarray,
    object_category_map: np.ndarray,
    category_id_to_name: dict[int, str],
) -> None:
    """Draw object semantics with explicit object/free/occlusion colors."""
    import matplotlib.patches as mpatches

    ignored_category_ids = {
        category_id
        for category_id, category_name in category_id_to_name.items()
        if category_name in IGNORED_OBJECT_CATEGORIES
    }
    present_ids = sorted(
        int(value)
        for value in np.unique(object_category_map)
        if value > 0 and int(value) not in ignored_category_ids
    )
    category_colors = _build_category_color_map(present_ids)
    object_image = render_object_map_image(
        traversibility_map,
        object_category_map,
        category_colors,
        ignored_category_ids,
    )

    axis.imshow(object_image, origin="lower")
    axis.set_title("Object Map")

    legend_handles = [
        mpatches.Patch(color=FREE_SPACE_COLOR, label="Free Space"),
        mpatches.Patch(color=OCCLUSION_COLOR, label="Occlusion"),
    ]
    legend_handles.extend(
        mpatches.Patch(
            color=category_colors[category_id],
            label=category_id_to_name.get(category_id, str(category_id)),
        )
        for category_id in present_ids
    )
    if legend_handles:
        axis.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=7,
            framealpha=0.9,
        )


def draw_room_type_map(
    axis,
    room_type_map: np.ndarray,
    room_id_to_type: dict[int, str],
) -> None:
    """Draw room types and place one label near each room center."""
    room_display = np.where(room_type_map >= 0, room_type_map, np.nan)
    axis.imshow(room_display, origin="lower", cmap="tab20")
    axis.set_title("Room Type Map")

    for room_id, room_name in room_id_to_type.items():
        rows, cols = np.where(room_type_map == room_id)
        if len(rows) == 0:
            continue
        center_row = int(np.round(rows.mean()))
        center_col = int(np.round(cols.mean()))
        axis.text(
            center_col,
            center_row,
            room_name,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.65, "pad": 1},
        )


def draw_semantic_overview(
    axis,
    traversibility_map: np.ndarray,
    room_type_map: np.ndarray,
    object_category_map: np.ndarray,
    category_id_to_name: dict[int, str],
) -> None:
    """Draw a compact map-only overview for the 2x2 scene layout."""
    from matplotlib import colormaps

    height, width = traversibility_map.shape
    image = np.zeros((height, width, 3), dtype=np.float32)

    room_ids = sorted(int(value) for value in np.unique(room_type_map) if value >= 0)
    room_palette = colormaps["tab20"]
    for index, room_id in enumerate(room_ids):
        color = np.array(room_palette(index % room_palette.N)[:3], dtype=np.float32)
        image[room_type_map == room_id] = 0.72 * color + 0.28 * FREE_SPACE_COLOR

    image[(traversibility_map > 0) & (room_type_map < 0)] = FREE_SPACE_COLOR

    ignored_category_ids = {
        category_id
        for category_id, category_name in category_id_to_name.items()
        if category_name in IGNORED_OBJECT_CATEGORIES
    }
    present_ids = sorted(
        int(value)
        for value in np.unique(object_category_map)
        if value > 0 and int(value) not in ignored_category_ids
    )
    category_colors = _build_category_color_map(present_ids)
    for category_id in present_ids:
        image[object_category_map == category_id] = np.array(
            category_colors[category_id][:3],
            dtype=np.float32,
        )

    axis.imshow(image, origin="lower")
    axis.set_title("Semantic Overview")
    axis.axis("off")


def plot_scene_export(
    map_data: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    annotate_objects: bool = True,
    max_annotations: int = 25,
):
    """Create a figure showing the saved scene layers."""
    import matplotlib.pyplot as plt

    traversibility_map = map_data["traversibility_map"]
    room_type_map = map_data["room_type_map"]
    object_category_map = map_data["object_category_map"]
    object_instance_map = map_data["object_instance_map"]

    room_type_to_id = metadata.get("room_type_to_id", {})
    object_metadata = metadata.get("object_metadata", [])

    room_id_to_type = invert_mapping(room_type_to_id) if isinstance(room_type_to_id, dict) else {}
    category_id_to_name = _category_id_to_name(metadata)

    figure, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    draw_semantic_overview(
        axes[0],
        traversibility_map,
        room_type_map,
        object_category_map,
        category_id_to_name,
    )

    axes[1].imshow(traversibility_map, origin="lower", cmap="gray")
    axes[1].set_title("Traversibility Map")

    draw_room_type_map(axes[2], room_type_map, room_id_to_type)

    del annotate_objects, max_annotations, object_instance_map, object_metadata
    draw_object_overlay(
        axes[3],
        traversibility_map,
        object_category_map,
        category_id_to_name,
    )

    for axis in axes[1:]:
        axis.set_xlabel("Grid Col (x)")
        axis.set_ylabel("Grid Row (z)")

    map_info = metadata.get("map_info", {})
    if isinstance(map_info, dict):
        title_lines = [
            "ProcTHOR Scene Overview",
            (
                f"resolution={map_info.get('resolution')}  "
                f"size=({map_info.get('H')}, {map_info.get('W')})"
            ),
        ]
        map_split = metadata.get("map_split")
        if map_split:
            title_lines.append(f"map_split={map_split}")
        room_size_summary = format_room_size_summary(metadata.get("room_metadata", []))
        if room_size_summary:
            title_lines.append(room_size_summary)
        figure.suptitle(
            "\n".join(title_lines),
            fontsize=14,
        )

    figure.tight_layout()
    return figure, axes


def _default_paths_from_prefix(prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(prefix)
    maps_path = prefix.with_name(f"{prefix.name}_maps.npz")
    metadata_path = prefix.with_name(f"{prefix.name}_metadata.json")
    return maps_path, metadata_path


def _default_png_path(
    prefix: str | Path | None,
    maps_path: str | Path | None,
) -> Path:
    """Infer a default PNG path for SSH-friendly visualization output."""
    if prefix is not None:
        prefix = Path(prefix)
        return prefix.with_name(f"{prefix.name}_view.png")

    if maps_path is None:
        raise ValueError("maps_path is required when prefix is not provided.")

    maps_path = Path(maps_path)
    stem = maps_path.stem
    if stem.endswith("_maps"):
        stem = stem[:-5]
    return maps_path.with_name(f"{stem}_view.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a saved ProcTHOR SemPathBench scene export."
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=None,
        help=(
            "Shared prefix used by the exporter, e.g. "
            "resources/maps/procthor/train/001_train/001_train."
        ),
    )
    parser.add_argument(
        "--maps",
        type=Path,
        default=None,
        help="Path to the *_maps.npz file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to the *_metadata.json file.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the rendered figure as an image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window instead of only saving a PNG.",
    )
    parser.add_argument(
        "--no-annotate-objects",
        action="store_true",
        help="Disable object-name text annotations on the object map.",
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=25,
        help="Maximum number of object labels to draw.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prefix is not None:
        maps_path, metadata_path = _default_paths_from_prefix(args.prefix)
    else:
        maps_path = args.maps
        metadata_path = args.metadata

    if maps_path is None or metadata_path is None:
        raise ValueError("Provide either --prefix, or both --maps and --metadata.")

    map_data, metadata = load_scene_export(maps_path, metadata_path)
    figure, _axes = plot_scene_export(
        map_data,
        metadata,
        annotate_objects=not args.no_annotate_objects,
        max_annotations=args.max_annotations,
    )

    save_path = args.save or _default_png_path(args.prefix, maps_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to {save_path}")

    if not args.show:
        return

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
