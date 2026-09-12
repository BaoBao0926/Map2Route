"""Shared method-output helpers."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Mapping

from scripts.annotation.path_shape_schema import path_shape_reference_trajectory
from scripts.make_instruction.make_instruction import REPO_ROOT


Trajectory = Sequence[Sequence[int]]
ConstraintOverlays = Mapping[str, Sequence[Mapping[str, object]]]
_TRAJECTORY_IMAGE_LOCK = Lock()


def display_path(path: Path) -> str:
    if path.is_relative_to(REPO_ROOT):
        return path.relative_to(REPO_ROOT).as_posix()
    return str(path)


def trajectory_image_path_for_output(output_path: Path) -> Path:
    return output_path.with_suffix(".png")


def trajectory_with_gt_image_path_for_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_with_gt.png")


def constraint_overlays_from_instruction(
    instruction: Mapping[str, object] | None,
) -> ConstraintOverlays | None:
    """Return benchmark annotation overlays for visualization only."""
    if instruction is None:
        return None
    hard_constraints = instruction.get("hard_constraints", ())
    soft_constraints = instruction.get("soft_constraints", ())
    hard_items = (
        [item for item in hard_constraints if isinstance(item, Mapping)]
        if isinstance(hard_constraints, Sequence) and not isinstance(hard_constraints, (str, bytes))
        else []
    )
    soft_items = (
        [item for item in soft_constraints if isinstance(item, Mapping)]
        if isinstance(soft_constraints, Sequence) and not isinstance(soft_constraints, (str, bytes))
        else []
    )
    if not hard_items and not soft_items:
        return None
    return {
        "hard_constraints": hard_items,
        "soft_constraints": soft_items,
    }


def _hex_to_rgb(value: object, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(value, str):
        return fallback
    color = value.strip().lstrip("#")
    if len(color) == 3:
        color = "".join(char * 2 for char in color)
    if len(color) != 6:
        return fallback
    try:
        return (
            int(color[0:2], 16),
            int(color[2:4], 16),
            int(color[4:6], 16),
        )
    except ValueError:
        return fallback


def _blend(
    base: tuple[int, int, int],
    overlay: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    return (
        max(0, min(255, round(base[0] * (1.0 - alpha) + overlay[0] * alpha))),
        max(0, min(255, round(base[1] * (1.0 - alpha) + overlay[1] * alpha))),
        max(0, min(255, round(base[2] * (1.0 - alpha) + overlay[2] * alpha))),
    )


def _legend_by_value(legend: object) -> dict[int, Mapping[str, object]]:
    if not isinstance(legend, Mapping):
        return {}
    result: dict[int, Mapping[str, object]] = {}
    for item in legend.values():
        if not isinstance(item, Mapping):
            continue
        value = item.get("value")
        if isinstance(value, int) and not isinstance(value, bool):
            result[value] = item
    return result


def _instance_category_by_id(
    instances: object,
) -> dict[int, str]:
    if not isinstance(instances, Sequence) or isinstance(instances, (str, bytes)):
        return {}
    result: dict[int, str] = {}
    for item in instances:
        if not isinstance(item, Mapping):
            continue
        item_id = item.get("id")
        category = item.get("category")
        if (
            isinstance(item_id, int)
            and not isinstance(item_id, bool)
            and isinstance(category, str)
            and category
        ):
            result[item_id] = category
    return result


def _layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    raw = layers.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return raw  # type: ignore[return-value]


def _layer_value(
    layer: Sequence[Sequence[object]],
    row: int,
    col: int,
    default: int = 0,
) -> int:
    if row >= len(layer):
        return default
    row_values = layer[row]
    if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
        return default
    if col >= len(row_values):
        return default
    try:
        return int(row_values[col])
    except (TypeError, ValueError):
        return default


def _constraint_center(constraint: Mapping[str, object]) -> tuple[float, float] | None:
    center = constraint.get("center")
    if (
        isinstance(center, Sequence)
        and not isinstance(center, (str, bytes))
        and len(center) == 2
    ):
        try:
            return float(center[0]), float(center[1])
        except (TypeError, ValueError):
            return None
    cells = constraint.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)) or not cells:
        return None
    parsed: list[tuple[float, float]] = []
    for cell in cells:
        if (
            isinstance(cell, Sequence)
            and not isinstance(cell, (str, bytes))
            and len(cell) == 2
        ):
            try:
                parsed.append((float(cell[0]), float(cell[1])))
            except (TypeError, ValueError):
                continue
    if not parsed:
        return None
    return (
        sum(row for row, _col in parsed) / len(parsed),
        sum(col for _row, col in parsed) / len(parsed),
    )


def _constraint_cells(constraint: Mapping[str, object]) -> list[tuple[int, int]]:
    raw = constraint.get("cells")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    cells: list[tuple[int, int]] = []
    for cell in raw:
        if (
            isinstance(cell, Sequence)
            and not isinstance(cell, (str, bytes))
            and len(cell) == 2
        ):
            try:
                cells.append((int(cell[0]), int(cell[1])))
            except (TypeError, ValueError):
                continue
    return cells


def _draw_freeform_boundary(
    draw: object,
    cells: Sequence[tuple[int, int]],
    *,
    scale: int,
    stroke: tuple[int, int, int, int],
    width: int,
) -> None:
    cell_set = set(cells)
    for row, col in cells:
        x0 = col * scale
        y0 = row * scale
        x1 = (col + 1) * scale
        y1 = (row + 1) * scale
        if (row - 1, col) not in cell_set:
            draw.line((x0, y0, x1, y0), fill=stroke, width=width)
        if (row + 1, col) not in cell_set:
            draw.line((x0, y1, x1, y1), fill=stroke, width=width)
        if (row, col - 1) not in cell_set:
            draw.line((x0, y0, x0, y1), fill=stroke, width=width)
        if (row, col + 1) not in cell_set:
            draw.line((x1, y0, x1, y1), fill=stroke, width=width)


def _draw_region_label(
    image: object,
    center: tuple[float, float] | None,
    label: str,
    *,
    scale: int,
    fill: tuple[int, int, int, int],
    large: bool = False,
) -> None:
    if center is None:
        return
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    center_row, center_col = center
    x = int(round((center_col + 0.5) * scale))
    y = int(round((center_row + 0.5) * scale))
    radius = max(24, round(scale * (8.5 if large else 1.8)))
    font = _label_font(max(42, round(scale * (14.0 if large else 3.2))))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(255, 255, 255, 240), width=max(1, scale))
    draw.text((x, y), label, fill=(255, 255, 255, 255), anchor="mm", font=font)


def _label_font(size: int) -> object | None:
    from PIL import ImageFont

    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _constraint_bbox_pixels(
    constraint: Mapping[str, object],
    *,
    scale: int,
) -> tuple[int, int, int, int] | None:
    shape = constraint.get("shape")
    if shape == "freeform":
        cells = _constraint_cells(constraint)
        if not cells:
            return None
        rows = [row for row, _col in cells]
        cols = [col for _row, col in cells]
        return (
            min(cols) * scale,
            min(rows) * scale,
            (max(cols) + 1) * scale,
            (max(rows) + 1) * scale,
        )
    center = _constraint_center(constraint)
    if center is None:
        return None
    center_row, center_col = center
    x = (center_col + 0.5) * scale
    y = (center_row + 0.5) * scale
    if shape == "circle":
        radius = float(constraint.get("radius", 1.0) or 1.0) * scale
        return int(x - radius), int(y - radius), int(x + radius), int(y + radius)
    width = float(constraint.get("width", constraint.get("radius", 1.0) or 1.0) or 1.0) * scale
    height = float(constraint.get("height", constraint.get("radius", 1.0) or 1.0) or 1.0) * scale
    return int(x - width / 2), int(y - height / 2), int(x + width / 2), int(y + height / 2)


def _draw_region_corner_label(
    image: object,
    constraint: Mapping[str, object],
    label: str,
    *,
    scale: int,
    fill: tuple[int, int, int, int],
) -> None:
    bbox = _constraint_bbox_pixels(constraint, scale=scale)
    if bbox is None:
        return
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    font = _label_font(max(48, round(scale * 14.0)))
    x0, y0, _x1, _y1 = bbox
    padding = max(10, scale * 5)
    text_x = max(0, x0 + padding)
    text_y = max(0, y0 + padding)
    text_bbox = draw.textbbox((text_x, text_y), label, font=font)
    draw.rounded_rectangle(
        (
            text_bbox[0] - padding,
            text_bbox[1] - padding,
            text_bbox[2] + padding,
            text_bbox[3] + padding,
        ),
        radius=max(2, scale),
        fill=fill,
        outline=(255, 255, 255, 230),
        width=max(1, scale // 2),
    )
    draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)


def _draw_constraint_region(
    image: object,
    constraint: Mapping[str, object],
    *,
    scale: int,
    fill: tuple[int, int, int, int],
    stroke: tuple[int, int, int, int],
    label: str | None = None,
    label_large: bool = False,
    corner_label: str | None = None,
) -> None:
    from PIL import Image, ImageDraw

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    line_width = max(2, scale)
    shape = constraint.get("shape")
    if shape == "freeform":
        cells = _constraint_cells(constraint)
        for row, col in cells:
            draw.rectangle(
                (col * scale, row * scale, (col + 1) * scale - 1, (row + 1) * scale - 1),
                fill=fill,
            )
        _draw_freeform_boundary(draw, cells, scale=scale, stroke=stroke, width=line_width)
        image.alpha_composite(overlay)
        if label:
            _draw_region_label(image, _constraint_center(constraint), label, scale=scale, fill=stroke, large=label_large)
        if corner_label:
            _draw_region_corner_label(image, constraint, corner_label, scale=scale, fill=stroke)
        return

    center = _constraint_center(constraint)
    if center is None:
        return
    center_row, center_col = center
    x = (center_col + 0.5) * scale
    y = (center_row + 0.5) * scale
    if shape == "circle":
        radius = float(constraint.get("radius", 1.0) or 1.0) * scale
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(box, fill=fill, outline=stroke, width=line_width)
    else:
        width = float(constraint.get("width", constraint.get("radius", 1.0) or 1.0) or 1.0) * scale
        height = float(constraint.get("height", constraint.get("radius", 1.0) or 1.0) or 1.0) * scale
        box = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
        draw.rectangle(box, fill=fill, outline=stroke, width=line_width)
    image.alpha_composite(overlay)
    if label:
        _draw_region_label(image, center, label, scale=scale, fill=stroke, large=label_large)
    if corner_label:
        _draw_region_corner_label(image, constraint, corner_label, scale=scale, fill=stroke)


def _draw_soft_reference_region(
    image: object,
    constraint: Mapping[str, object],
    *,
    scale: int,
) -> None:
    preference_type = constraint.get("preference_type")
    if preference_type == "relative_preference":
        raw_regions = constraint.get("reference_regions")
        if not isinstance(raw_regions, Sequence) or isinstance(raw_regions, (str, bytes)):
            return
        styles = [
            ((34, 139, 92, 48), (31, 122, 82, 230), "A"),
            ((141, 83, 194, 48), (109, 62, 160, 230), "B"),
        ]
        for region, (fill, stroke, label) in zip(raw_regions[:2], styles):
            if isinstance(region, Mapping):
                _draw_constraint_region(
                    image,
                    {"shape": "freeform", "cells": region.get("cells", [])},
                    scale=scale,
                    fill=fill,
                    stroke=stroke,
                    label=label,
                )
        return

    reference_region = constraint.get("reference_region")
    if isinstance(reference_region, Mapping):
        _draw_constraint_region(
            image,
            {"shape": "freeform", "cells": reference_region.get("cells", [])},
            scale=scale,
            fill=(213, 142, 31, 52),
            stroke=(213, 142, 31, 230),
            label="R",
        )


def _draw_path_shape_reference_trajectory(
    image: object,
    constraint: Mapping[str, object],
    *,
    scale: int,
) -> None:
    raw_trajectory = path_shape_reference_trajectory(constraint)
    if not isinstance(raw_trajectory, Sequence) or isinstance(raw_trajectory, (str, bytes)):
        return
    points: list[tuple[int, int]] = []
    for point in raw_trajectory:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            continue
        row, col = int(point[0]), int(point[1])
        points.append((col * scale + scale // 2, row * scale + scale // 2))
    if not points:
        return

    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    line_width = max(4, scale * 3)
    if len(points) >= 2:
        draw.line(points, fill=(172, 38, 168, 245), width=line_width, joint="curve")
        highlight_width = max(2, scale)
        draw.line(points, fill=(255, 232, 255, 210), width=highlight_width, joint="curve")
    else:
        x, y = points[0]
        radius = max(4, scale * 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(172, 38, 168, 245))


def _soft_preference_label(constraint: Mapping[str, object]) -> str:
    preference_type = str(constraint.get("preference_type") or "preference")
    labels = {
        "near_preference": "near",
        "far_preference": "far",
        "relative_preference": "relative",
        "clearance": "clearance",
        "move_smoothness": "smooth",
        "path_shape_preference": "shape",
    }
    return labels.get(preference_type, preference_type.replace("_preference", "").replace("_", " "))


def _draw_constraint_overlays(
    image: object,
    overlays: ConstraintOverlays | None,
    *,
    scale: int,
) -> None:
    if not overlays:
        return
    hard_constraints = overlays.get("hard_constraints", ())
    for index, constraint in enumerate(hard_constraints, start=1):
        if not isinstance(constraint, Mapping) or constraint.get("initialized") is False:
            continue
        is_must_pass = constraint.get("kind") == "must_pass"
        _draw_constraint_region(
            image,
            constraint,
            scale=scale,
            fill=(0, 0, 0, 56) if is_must_pass else (209, 79, 69, 72),
            stroke=(0, 0, 0, 230) if is_must_pass else (169, 52, 44, 235),
            label=str(index),
            label_large=True,
        )

    soft_constraints = overlays.get("soft_constraints", ())
    for index, constraint in enumerate(soft_constraints, start=1):
        if not isinstance(constraint, Mapping) or constraint.get("initialized") is False:
            continue
        preference_type = constraint.get("preference_type")
        if preference_type in {"clearance", "move_smoothness"}:
            continue
        if preference_type == "path_shape_preference":
            _draw_constraint_region(
                image,
                constraint,
                scale=scale,
                fill=(172, 38, 168, 44),
                stroke=(139, 31, 136, 235),
                corner_label=_soft_preference_label(constraint),
            )
            _draw_path_shape_reference_trajectory(image, constraint, scale=scale)
            continue
        _draw_constraint_region(
            image,
            constraint,
            scale=scale,
            fill=(52, 120, 184, 62),
            stroke=(36, 95, 150, 235),
            corner_label=_soft_preference_label(constraint),
        )
        _draw_soft_reference_region(image, constraint, scale=scale)


def write_trajectory_image(
    map_state: Mapping[str, object],
    trajectory: Trajectory,
    output_path: Path,
    *,
    scale: int = 3,
    constraint_overlays: ConstraintOverlays | None = None,
    gt_trajectory: Trajectory | None = None,
) -> None:
    """Write a semantic-map PNG with a human-solver-style trajectory overlay."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to write SemPathBench trajectory images."
        ) from exc

    grid_size = int(map_state["grid_size"])
    legends = map_state.get("layer_legends")
    legends = legends if isinstance(legends, Mapping) else {}
    occupancy_legend = legends.get("occupancy")
    occupancy_by_value = _legend_by_value(occupancy_legend)
    room_category_legend = legends.get("room_categories")
    object_category_legend = legends.get("object_categories")
    room_category_legend = room_category_legend if isinstance(room_category_legend, Mapping) else {}
    object_category_legend = (
        object_category_legend if isinstance(object_category_legend, Mapping) else {}
    )
    free_value = 0
    if isinstance(occupancy_legend, Mapping):
        free = occupancy_legend.get("free")
        if isinstance(free, Mapping) and isinstance(free.get("value"), int):
            free_value = int(free["value"])

    occupancy_layer = _layer(map_state, "occupancy")
    room_layer = _layer(map_state, "room")
    object_layer = _layer(map_state, "object_instance")
    room_category_by_id = _instance_category_by_id(map_state.get("room_instances"))
    object_category_by_id = _instance_category_by_id(map_state.get("object_instances"))

    image = Image.new("RGBA", (grid_size * scale, grid_size * scale), (40, 40, 40, 255))
    draw = ImageDraw.Draw(image)
    for row in range(grid_size):
        for col in range(grid_size):
            occupancy_value = _layer_value(occupancy_layer, row, col)
            occupancy_item = occupancy_by_value.get(occupancy_value)
            occ_name = (
                str(occupancy_item.get("name"))
                if isinstance(occupancy_item, Mapping) and occupancy_item.get("name")
                else "free"
                if occupancy_value == free_value
                else "blocked"
            )
            color = _hex_to_rgb(
                occupancy_item.get("color") if isinstance(occupancy_item, Mapping) else None,
                (255, 255, 255) if occ_name == "free" else (17, 17, 17),
            )

            room_id = _layer_value(room_layer, row, col)
            room_category = room_category_by_id.get(room_id)
            room_legend = room_category_legend.get(room_category) if room_category else None
            if occ_name == "free" and isinstance(room_legend, Mapping):
                room_color = _hex_to_rgb(room_legend.get("color"), color)
                color = _blend(color, room_color, 0.45)

            object_id = _layer_value(object_layer, row, col)
            object_category = object_category_by_id.get(object_id)
            object_legend = (
                object_category_legend.get(object_category) if object_category else None
            )
            if isinstance(object_legend, Mapping):
                object_color = _hex_to_rgb(object_legend.get("color"), color)
                color = _blend(color, object_color, 0.82)

            draw.rectangle(
                (
                    col * scale,
                    row * scale,
                    (col + 1) * scale - 1,
                    (row + 1) * scale - 1,
                ),
                fill=(*color, 255),
            )

    _draw_constraint_overlays(image, constraint_overlays, scale=scale)

    points: list[tuple[int, int]] = []
    valid_cells: list[tuple[int, int]] = []
    for point in trajectory:
        if len(point) != 2:
            continue
        row, col = int(point[0]), int(point[1])
        if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
            continue
        valid_cells.append((row, col))
        points.append((col * scale + scale // 2, row * scale + scale // 2))

    for index, (row, col) in enumerate(valid_cells):
        ratio = 0.0 if len(valid_cells) <= 1 else index / (len(valid_cells) - 1)
        color = (
            round(18 + 170 * ratio),
            round(107 - 35 * ratio),
            round(93 - 35 * ratio),
        )
        draw.rectangle(
            (
                col * scale,
                row * scale,
                (col + 1) * scale - 1,
                (row + 1) * scale - 1,
            ),
            fill=(*color, 255),
        )

    if gt_trajectory is not None:
        gt_points: list[tuple[int, int]] = []
        for point in gt_trajectory:
            if len(point) != 2:
                continue
            row, col = int(point[0]), int(point[1])
            if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
                continue
            x = col * scale + scale // 2
            y = row * scale + scale // 2
            gt_points.append((x, y))
        if len(gt_points) >= 2:
            draw.line(gt_points, fill=(0, 0, 0, 255), width=max(2, scale * 2), joint="curve")
        elif gt_points:
            radius = max(2, round(scale * 1.2))
            x, y = gt_points[0]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 0, 255))

    if valid_cells:
        start_row, start_col = valid_cells[0]
        draw.rectangle(
            (
                start_col * scale,
                start_row * scale,
                (start_col + 1) * scale - 1,
                (start_row + 1) * scale - 1,
            ),
            fill=(*_hex_to_rgb("#2d6cdf", (45, 108, 223)), 255),
        )
        start_x, start_y = points[0]
        radius = max(4, scale * 5)
        draw.ellipse(
            (start_x - radius, start_y - radius, start_x + radius, start_y + radius),
            outline=(*_hex_to_rgb("#2d6cdf", (45, 108, 223)), 255),
            width=max(1, scale),
        )
        end_row, end_col = valid_cells[-1]
        draw.rectangle(
            (
                end_col * scale,
                end_row * scale,
                (end_col + 1) * scale - 1,
                (end_row + 1) * scale - 1,
            ),
            fill=(*_hex_to_rgb("#d92d20", (217, 45, 32)), 255),
        )
        end_x, end_y = points[-1]
        end_radius = max(4, scale * 4)
        draw.ellipse(
            (end_x - end_radius, end_y - end_radius, end_x + end_radius, end_y + end_radius),
            outline=(*_hex_to_rgb("#d92d20", (217, 45, 32)), 255),
            width=max(1, scale),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def save_trajectory_image_for_record(
    record: MutableMapping[str, object],
    map_state: Mapping[str, object],
    trajectory: Trajectory,
    output_path: Path,
    *,
    nested_record_keys: Sequence[str] = (),
    instruction: Mapping[str, object] | None = None,
    constraint_overlays: ConstraintOverlays | None = None,
) -> Path:
    """Write the trajectory PNG next to a JSON output and attach its path."""
    with _TRAJECTORY_IMAGE_LOCK:
        return _save_trajectory_image_for_record(
            record,
            map_state,
            trajectory,
            output_path,
            nested_record_keys=nested_record_keys,
            instruction=instruction,
            constraint_overlays=constraint_overlays,
        )


def _save_trajectory_image_for_record(
    record: MutableMapping[str, object],
    map_state: Mapping[str, object],
    trajectory: Trajectory,
    output_path: Path,
    *,
    nested_record_keys: Sequence[str] = (),
    instruction: Mapping[str, object] | None = None,
    constraint_overlays: ConstraintOverlays | None = None,
) -> Path:
    """Render one trajectory image while the process-wide image lock is held."""
    if constraint_overlays is None:
        constraint_overlays = constraint_overlays_from_instruction(instruction)
    image_path = trajectory_image_path_for_output(output_path)
    write_trajectory_image(
        map_state,
        trajectory,
        image_path,
        constraint_overlays=constraint_overlays,
    )
    image_path_display = display_path(image_path)
    record["trajectory_image"] = image_path_display
    for key in nested_record_keys:
        nested = record.get(key)
        if isinstance(nested, MutableMapping):
            nested["trajectory_image"] = image_path_display

    gt_trajectory = None
    if instruction is not None:
        raw_gt_trajectory = instruction.get("human_expert_trajectory")
        if isinstance(raw_gt_trajectory, Sequence) and not isinstance(raw_gt_trajectory, (str, bytes)):
            gt_trajectory = raw_gt_trajectory  # type: ignore[assignment]
    if gt_trajectory:
        gt_image_path = trajectory_with_gt_image_path_for_output(output_path)
        write_trajectory_image(
            map_state,
            trajectory,
            gt_image_path,
            constraint_overlays=constraint_overlays,
            gt_trajectory=gt_trajectory,
        )
        gt_image_path_display = display_path(gt_image_path)
        record["trajectory_with_gt_image"] = gt_image_path_display
        for key in nested_record_keys:
            nested = record.get(key)
            if isinstance(nested, MutableMapping):
                nested["trajectory_with_gt_image"] = gt_image_path_display
    return image_path
