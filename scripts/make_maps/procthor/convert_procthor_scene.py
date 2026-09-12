#!/usr/bin/env python3
"""Convert ProcTHOR scenes into SemPathBench map layers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NAV_OBJECT_PRIORITY = {
    "Wall": 5,
    "Sofa": 100,
    "Table": 100,
    "Desk": 100,
    "Bed": 100,
    "Chair": 90,
    "CounterTop": 90,
    "Fridge": 90,
    "Cabinet": 80,
    "Sink": 80,
    "Toilet": 80,
    "Bathtub": 80,
    "Door": 80,
    "GarbageCan": 60,
    "HousePlant": 60,
    "Apple": 10,
    "Egg": 10,
    "Fork": 10,
    "Plate": 10,
}

IGNORED_OBJECT_CATEGORIES = {"Floor"}

OBJECT_CATEGORY_ALIASES = {
    "ShelvingUnit": "Shelf",
}

NAVIGATION_OBJECT_CATEGORIES = {
    "Wall",
    "Sofa",
    "Chair",
    "Table",
    "Desk",
    "Bed",
    "Fridge",
    "CounterTop",
    "Cabinet",
    "Sink",
    "Toilet",
    "Bathtub",
    "Door",
    "GarbageCan",
    "HousePlant",
}


def canonicalize_object_category(category: str) -> str:
    """Map equivalent ProcTHOR object categories to one canonical label."""
    return OBJECT_CATEGORY_ALIASES.get(category, category)


def world_to_grid(
    x: float,
    z: float,
    x_min: float,
    z_min: float,
    resolution: float,
) -> tuple[int, int]:
    """Convert a ProcTHOR world coordinate into a grid row/column."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    col = int(round((x - x_min) / resolution))
    row = int(round((z - z_min) / resolution))
    return row, col


def grid_to_world(
    row: int,
    col: int,
    x_min: float,
    z_min: float,
    resolution: float,
) -> tuple[float, float]:
    """Convert a grid row/column back into a ProcTHOR world coordinate."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    x = x_min + col * resolution
    z = z_min + row * resolution
    return x, z


def parse_object_category(object_id: str) -> str:
    """Extract the semantic category from a ProcTHOR object id."""
    return canonicalize_object_category(object_id.split("|")[0])


def _normalize_point3d(point: object) -> dict[str, float] | None:
    if isinstance(point, dict):
        x = point.get("x")
        y = point.get("y")
        z = point.get("z")
        if all(isinstance(value, (int, float)) for value in (x, y, z)):
            return {"x": float(x), "y": float(y), "z": float(z)}

    if isinstance(point, (list, tuple)) and len(point) >= 3:
        x, y, z = point[:3]
        if all(isinstance(value, (int, float)) for value in (x, y, z)):
            return {"x": float(x), "y": float(y), "z": float(z)}

    return None


def _point_on_segment(
    x: float,
    z: float,
    x1: float,
    z1: float,
    x2: float,
    z2: float,
    eps: float = 1e-9,
) -> bool:
    cross = (z - z1) * (x2 - x1) - (x - x1) * (z2 - z1)
    if abs(cross) > eps:
        return False

    dot = (x - x1) * (x2 - x1) + (z - z1) * (z2 - z1)
    if dot < -eps:
        return False

    squared_length = (x2 - x1) ** 2 + (z2 - z1) ** 2
    if dot - squared_length > eps:
        return False

    return True


def _point_in_polygon(x: float, z: float, polygon: list[tuple[float, float]]) -> bool:
    """Return True when the point lies inside or on the boundary."""
    inside = False
    num_vertices = len(polygon)
    if num_vertices < 3:
        return False

    for index in range(num_vertices):
        x1, z1 = polygon[index]
        x2, z2 = polygon[(index + 1) % num_vertices]

        if _point_on_segment(x, z, x1, z1, x2, z2):
            return True

        intersects = ((z1 > z) != (z2 > z)) and (
            x <= (x2 - x1) * (z - z1) / ((z2 - z1) + 1e-12) + x1
        )
        if intersects:
            inside = not inside

    return inside


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute the 2D convex hull with the monotonic chain algorithm."""
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(
        origin: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _rectangle_from_bounds(
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
) -> list[tuple[float, float]]:
    return [
        (x_min, z_min),
        (x_min, z_max),
        (x_max, z_max),
        (x_max, z_min),
    ]


def polygon_area_xz(polygon_xz: list[tuple[float, float]]) -> float:
    """Return the continuous x-z area of a floor polygon."""
    if len(polygon_xz) < 3:
        return 0.0

    signed_area = 0.0
    for index, (x1, z1) in enumerate(polygon_xz):
        x2, z2 = polygon_xz[(index + 1) % len(polygon_xz)]
        signed_area += x1 * z2 - x2 * z1
    return abs(signed_area) * 0.5


def polygon_centroid_xz(polygon_xz: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Return the area-weighted centroid of an x-z polygon."""
    if not polygon_xz:
        return None

    signed_cross_sum = 0.0
    centroid_x = 0.0
    centroid_z = 0.0
    for index, (x1, z1) in enumerate(polygon_xz):
        x2, z2 = polygon_xz[(index + 1) % len(polygon_xz)]
        cross = x1 * z2 - x2 * z1
        signed_cross_sum += cross
        centroid_x += (x1 + x2) * cross
        centroid_z += (z1 + z2) * cross

    if abs(signed_cross_sum) <= 1e-12:
        xs = [point[0] for point in polygon_xz]
        zs = [point[1] for point in polygon_xz]
        return float(np.mean(xs)), float(np.mean(zs))

    scale = 1.0 / (3.0 * signed_cross_sum)
    return centroid_x * scale, centroid_z * scale


def get_bbox_corner_points(obj: dict[str, object]) -> list[dict[str, float]] | None:
    """Return bbox corner points, preferring OBB over AABB."""
    for bbox_key in ("objectOrientedBoundingBox", "axisAlignedBoundingBox"):
        bbox = obj.get(bbox_key)
        if not isinstance(bbox, dict):
            continue
        corner_points = bbox.get("cornerPoints")
        if not isinstance(corner_points, list):
            continue

        normalized_points = [
            normalized_point
            for point in corner_points
            if (normalized_point := _normalize_point3d(point)) is not None
        ]
        if normalized_points:
            return normalized_points

    return None


def get_object_footprint_polygon_xz(
    obj: dict[str, object],
) -> list[tuple[float, float]] | None:
    """Project a 3D bounding box onto the x-z plane as a 2D footprint polygon."""
    corner_points = get_bbox_corner_points(obj)
    if not corner_points:
        return None

    xz_points = sorted({(point["x"], point["z"]) for point in corner_points})
    if len(xz_points) == 1:
        return None
    if len(xz_points) == 2:
        xs = [point[0] for point in xz_points]
        zs = [point[1] for point in xz_points]
        return _rectangle_from_bounds(min(xs), max(xs), min(zs), max(zs))

    if isinstance(obj.get("objectOrientedBoundingBox"), dict):
        hull = _convex_hull(xz_points)
        if len(hull) >= 3:
            return hull

    xs = [point[0] for point in xz_points]
    zs = [point[1] for point in xz_points]
    return _rectangle_from_bounds(min(xs), max(xs), min(zs), max(zs))


def rasterize_polygon_to_grid(
    polygon_xz: list[tuple[float, float]],
    map_info: dict[str, float | int],
) -> list[tuple[int, int]]:
    """Return grid cells whose centers fall inside the polygon."""
    if len(polygon_xz) < 3:
        return []

    x_min = float(map_info["x_min"])
    z_min = float(map_info["z_min"])
    resolution = float(map_info["resolution"])
    height = int(map_info["H"])
    width = int(map_info["W"])

    poly_x_min = min(point[0] for point in polygon_xz)
    poly_x_max = max(point[0] for point in polygon_xz)
    poly_z_min = min(point[1] for point in polygon_xz)
    poly_z_max = max(point[1] for point in polygon_xz)

    row_min, col_min = world_to_grid(poly_x_min, poly_z_min, x_min, z_min, resolution)
    row_max, col_max = world_to_grid(poly_x_max, poly_z_max, x_min, z_min, resolution)

    row_start = max(0, min(row_min, row_max))
    row_end = min(height - 1, max(row_min, row_max))
    col_start = max(0, min(col_min, col_max))
    col_end = min(width - 1, max(col_min, col_max))

    cells: list[tuple[int, int]] = []
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            x, z = grid_to_world(row, col, x_min, z_min, resolution)
            if _point_in_polygon(x, z, polygon_xz):
                cells.append((row, col))

    return cells


def parse_wall_segment_from_object_id(
    object_id: str,
) -> tuple[float, float, float, float] | None:
    """Parse ProcTHOR wall ids shaped like `wall|room|x1|z1|x2|z2`."""
    parts = object_id.split("|")
    if len(parts) != 6 or parts[0].lower() != "wall":
        return None

    try:
        return (
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5]),
        )
    except ValueError:
        return None


def _point_to_segment_distance(
    x: float,
    z: float,
    x1: float,
    z1: float,
    x2: float,
    z2: float,
) -> float:
    dx = x2 - x1
    dz = z2 - z1
    length_squared = dx * dx + dz * dz
    if length_squared <= 1e-12:
        return float(np.hypot(x - x1, z - z1))

    t = ((x - x1) * dx + (z - z1) * dz) / length_squared
    t = min(1.0, max(0.0, t))
    projection_x = x1 + t * dx
    projection_z = z1 + t * dz
    return float(np.hypot(x - projection_x, z - projection_z))


def rasterize_wall_segment_to_grid(
    segment_xz: tuple[float, float, float, float],
    map_info: dict[str, float | int],
    radius: float | None = None,
) -> list[tuple[int, int]]:
    """Rasterize a thin ProcTHOR wall segment without using its broad mesh bbox."""
    x1, z1, x2, z2 = segment_xz
    x_min = float(map_info["x_min"])
    z_min = float(map_info["z_min"])
    resolution = float(map_info["resolution"])
    height = int(map_info["H"])
    width = int(map_info["W"])
    if radius is None:
        radius = max(resolution * 0.5, 0.05)

    col_start = max(0, int(np.floor((min(x1, x2) - radius - x_min) / resolution)))
    col_end = min(width - 1, int(np.ceil((max(x1, x2) + radius - x_min) / resolution)))
    row_start = max(0, int(np.floor((min(z1, z2) - radius - z_min) / resolution)))
    row_end = min(height - 1, int(np.ceil((max(z1, z2) + radius - z_min) / resolution)))

    cells: list[tuple[int, int]] = []
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            x, z = grid_to_world(row, col, x_min, z_min, resolution)
            if _point_to_segment_distance(x, z, x1, z1, x2, z2) <= radius + 1e-9:
                cells.append((row, col))

    return cells


def _center_fallback_cells(
    position: dict[str, object] | None,
    map_info: dict[str, float | int],
    radius_cells: int = 1,
) -> list[tuple[int, int]]:
    if not isinstance(position, dict):
        return []

    x = position.get("x")
    z = position.get("z")
    if not isinstance(x, (int, float)) or not isinstance(z, (int, float)):
        return []

    row, col = world_to_grid(
        float(x),
        float(z),
        float(map_info["x_min"]),
        float(map_info["z_min"]),
        float(map_info["resolution"]),
    )
    height = int(map_info["H"])
    width = int(map_info["W"])

    cells: list[tuple[int, int]] = []
    for row_offset in range(-radius_cells, radius_cells + 1):
        for col_offset in range(-radius_cells, radius_cells + 1):
            if row_offset * row_offset + col_offset * col_offset > radius_cells * radius_cells:
                continue
            candidate_row = row + row_offset
            candidate_col = col + col_offset
            if 0 <= candidate_row < height and 0 <= candidate_col < width:
                cells.append((candidate_row, candidate_col))
    return cells


def _object_priority(object_type: str) -> int:
    return NAV_OBJECT_PRIORITY.get(object_type, 50)


def _candidate_runtime_objects(
    runtime_objects: list[dict[str, object]],
    keep_only_navigation_objects: bool,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for obj in runtime_objects:
        object_type = obj.get("objectType")
        object_id = obj.get("objectId")
        position = obj.get("position")
        if not isinstance(object_type, str) or not isinstance(object_id, str):
            continue
        if object_type in IGNORED_OBJECT_CATEGORIES:
            continue
        if keep_only_navigation_objects and object_type not in NAVIGATION_OBJECT_CATEGORIES:
            continue
        if not isinstance(position, dict):
            continue
        candidates.append(obj)
    return candidates


def collect_house_objects(objects: list[dict[str, object]]) -> list[dict[str, object]]:
    """Recursively collect top-level objects and child objects from a house JSON."""
    all_objects: list[dict[str, object]] = []

    def visit(obj: dict[str, object], parent_id: str | None = None) -> None:
        object_id = obj.get("id")
        position = obj.get("position")
        if isinstance(object_id, str) and isinstance(position, dict):
            x = position.get("x")
            y = position.get("y")
            z = position.get("z")
            if all(isinstance(value, (int, float)) for value in (x, y, z)):
                all_objects.append(
                    {
                        "id": object_id,
                        "assetId": obj.get("assetId"),
                        "category": parse_object_category(object_id),
                        "parent_id": parent_id,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "rotation": obj.get("rotation"),
                    }
                )

        children = obj.get("children", [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    visit(child, parent_id=object_id if isinstance(object_id, str) else None)

    for obj in objects:
        if isinstance(obj, dict):
            visit(obj)

    return all_objects


def _parse_room_floor_polygons(
    house: dict[str, object],
) -> list[tuple[list[tuple[float, float]], str, str | None]]:
    rooms = house.get("rooms", [])
    if not isinstance(rooms, list):
        return []

    parsed_rooms: list[tuple[list[tuple[float, float]], str, str | None]] = []
    for room in rooms:
        if not isinstance(room, dict):
            continue
        room_type = room.get("roomType")
        polygon = room.get("floorPolygon")
        if not isinstance(room_type, str) or not isinstance(polygon, list):
            continue

        polygon_xz: list[tuple[float, float]] = []
        for point in polygon:
            if not isinstance(point, dict):
                continue
            x = point.get("x")
            z = point.get("z")
            if isinstance(x, (int, float)) and isinstance(z, (int, float)):
                polygon_xz.append((float(x), float(z)))

        room_id = room.get("id")
        parsed_rooms.append(
            (
                polygon_xz,
                room_type,
                room_id if isinstance(room_id, str) else None,
            )
        )

    return parsed_rooms


def compute_house_scene_size(house: dict[str, object]) -> float:
    """Return total true floor area across all ProcTHOR room polygons."""
    return sum(compute_house_room_sizes(house))


def compute_house_room_sizes(house: dict[str, object]) -> list[float]:
    """Return true floor area for each room polygon in a ProcTHOR house."""
    return [
        polygon_area_xz(polygon_xz)
        for polygon_xz, _room_type, _room_id in _parse_room_floor_polygons(house)
    ]


def compute_house_largest_room_size(house: dict[str, object]) -> float:
    """Return the largest single-room true floor area in a ProcTHOR house."""
    return max(compute_house_room_sizes(house), default=0.0)


def build_map_info_from_house(
    house: dict[str, object],
    resolution: float = 0.25,
    padding: float = 1.0,
) -> dict[str, float | int]:
    """Build map bounds directly from ProcTHOR house geometry."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    points: list[tuple[float, float]] = []
    for polygon_xz, _room_type, _room_id in _parse_room_floor_polygons(house):
        points.extend(polygon_xz)

    if not points:
        objects = house.get("objects", [])
        for obj in collect_house_objects(objects if isinstance(objects, list) else []):
            points.append((float(obj["x"]), float(obj["z"])))

    if not points:
        raise ValueError("ProcTHOR house has no room floor polygons or positioned objects.")

    xs = [point[0] for point in points]
    zs = [point[1] for point in points]

    x_min = min(xs) - padding
    x_max = max(xs) + padding
    z_min = min(zs) - padding
    z_max = max(zs) + padding

    width = int(np.ceil((x_max - x_min) / resolution)) + 1
    height = int(np.ceil((z_max - z_min) / resolution)) + 1

    return {
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
        "resolution": resolution,
        "H": height,
        "W": width,
    }


def build_floor_traversibility_map(
    house: dict[str, object],
    map_info: dict[str, float | int],
) -> np.ndarray:
    """Build a simulator-free floor mask from room floor polygons."""
    height = int(map_info["H"])
    width = int(map_info["W"])
    traversibility_map = np.zeros((height, width), dtype=np.uint8)

    for polygon_xz, _room_type, _room_id in _parse_room_floor_polygons(house):
        if len(polygon_xz) < 3:
            continue
        for row, col in rasterize_polygon_to_grid(polygon_xz, map_info):
            traversibility_map[row, col] = 1

    return traversibility_map


def build_traversibility_map(
    controller: object,
    resolution: float = 0.25,
    padding: float = 1.0,
) -> tuple[np.ndarray, dict[str, float | int], list[dict[str, float]]]:
    """Build a binary traversibility map from AI2-THOR reachable positions."""
    event = controller.step(action="GetReachablePositions", gridSize=resolution)
    reachable_positions = event.metadata.get("actionReturn") or []
    if not reachable_positions:
        raise ValueError("GetReachablePositions returned no reachable positions.")

    xs = [float(position["x"]) for position in reachable_positions]
    zs = [float(position["z"]) for position in reachable_positions]

    x_min = min(xs) - padding
    x_max = max(xs) + padding
    z_min = min(zs) - padding
    z_max = max(zs) + padding

    width = int(np.ceil((x_max - x_min) / resolution)) + 1
    height = int(np.ceil((z_max - z_min) / resolution)) + 1

    traversibility_map = np.zeros((height, width), dtype=np.uint8)
    for position in reachable_positions:
        row, col = world_to_grid(
            float(position["x"]),
            float(position["z"]),
            x_min,
            z_min,
            resolution,
        )
        if 0 <= row < height and 0 <= col < width:
            traversibility_map[row, col] = 1

    map_info: dict[str, float | int] = {
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
        "resolution": resolution,
        "H": height,
        "W": width,
    }
    return traversibility_map, map_info, reachable_positions


def build_room_type_map(
    house: dict[str, object],
    map_info: dict[str, float | int],
    traversibility_map: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int], dict[str, str]]:
    """Assign room type ids to every grid cell covered by room floor polygons."""
    del traversibility_map  # Room semantics should cover occupied cells too.

    height = int(map_info["H"])
    width = int(map_info["W"])

    room_type_map = np.full((height, width), fill_value=-1, dtype=np.int32)
    rooms = house.get("rooms", [])
    if not isinstance(rooms, list) or not rooms:
        return room_type_map, {}, {}

    room_types = sorted(
        {
            str(room["roomType"])
            for room in rooms
            if isinstance(room, dict) and isinstance(room.get("roomType"), str)
        }
    )
    room_type_to_id = {room_type: index for index, room_type in enumerate(room_types)}
    room_id_to_type = {
        str(room["id"]): str(room["roomType"])
        for room in rooms
        if isinstance(room, dict)
        and isinstance(room.get("id"), str)
        and isinstance(room.get("roomType"), str)
    }

    for polygon_xz, room_type, _room_id in _parse_room_floor_polygons(house):
        if len(polygon_xz) < 3:
            continue
        room_type_id = room_type_to_id[room_type]
        for row, col in rasterize_polygon_to_grid(polygon_xz, map_info):
            if room_type_map[row, col] == -1:
                room_type_map[row, col] = room_type_id

    return room_type_map, room_type_to_id, room_id_to_type


def build_room_metadata(
    house: dict[str, object],
    map_info: dict[str, float | int],
    room_type_to_id: dict[str, int],
) -> list[dict[str, object]]:
    """Build per-room size metadata from ProcTHOR floor polygons."""
    resolution = float(map_info["resolution"])
    room_metadata: list[dict[str, object]] = []

    for room_index, (polygon_xz, room_type, room_id) in enumerate(
        _parse_room_floor_polygons(house)
    ):
        if len(polygon_xz) < 3:
            continue

        cells = rasterize_polygon_to_grid(polygon_xz, map_info)
        xs = [point[0] for point in polygon_xz]
        zs = [point[1] for point in polygon_xz]
        centroid = polygon_centroid_xz(polygon_xz)

        record: dict[str, object] = {
            "room_index": room_index,
            "room_id": room_id,
            "room_type": room_type,
            "room_type_id": room_type_to_id.get(room_type),
            "floor_polygon_xz": [list(point) for point in polygon_xz],
            "area": polygon_area_xz(polygon_xz),
            "num_grid_cells": len(cells),
            "grid_area": len(cells) * resolution * resolution,
            "bbox": {
                "x_min": min(xs),
                "x_max": max(xs),
                "z_min": min(zs),
                "z_max": max(zs),
                "width_x": max(xs) - min(xs),
                "depth_z": max(zs) - min(zs),
            },
            "centroid_xz": list(centroid) if centroid is not None else None,
        }
        if len(cells) <= 256:
            record["grid_cells"] = [list(cell) for cell in cells]
        room_metadata.append(record)

    return room_metadata


def build_object_center_map(
    house: dict[str, object],
    map_info: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, int], dict[int, str]]:
    """Build point-based object semantic and instance maps."""
    height = int(map_info["H"])
    width = int(map_info["W"])
    x_min = float(map_info["x_min"])
    z_min = float(map_info["z_min"])
    resolution = float(map_info["resolution"])

    objects = house.get("objects", [])
    all_objects = collect_house_objects(objects if isinstance(objects, list) else [])

    categories = sorted(
        {
            canonicalize_object_category(str(obj["category"]))
            for obj in all_objects
            if canonicalize_object_category(str(obj["category"]))
            not in IGNORED_OBJECT_CATEGORIES
        }
    )
    category_to_id = {category: index + 1 for index, category in enumerate(categories)}
    id_to_category = {value: key for key, value in category_to_id.items()}

    object_category_map = np.zeros((height, width), dtype=np.int32)
    object_instance_map = np.zeros((height, width), dtype=np.int32)
    object_metadata: list[dict[str, object]] = []

    for instance_id, obj in enumerate(all_objects, start=1):
        row, col = world_to_grid(
            float(obj["x"]),
            float(obj["z"]),
            x_min,
            z_min,
            resolution,
        )
        category = canonicalize_object_category(str(obj["category"]))
        if category in IGNORED_OBJECT_CATEGORIES:
            continue

        metadata = {
            "id": str(obj["id"]),
            "assetId": obj.get("assetId"),
            "category": category,
            "x": float(obj["x"]),
            "y": float(obj["y"]),
            "z": float(obj["z"]),
            "grid_row": row,
            "grid_col": col,
            "instance_id": instance_id,
            "category_id": category_to_id[category],
            "parent_id": obj.get("parent_id"),
        }
        object_metadata.append(metadata)

        if 0 <= row < height and 0 <= col < width:
            object_category_map[row, col] = category_to_id[category]
            object_instance_map[row, col] = instance_id

    return (
        object_category_map,
        object_instance_map,
        object_metadata,
        category_to_id,
        id_to_category,
    )


def build_object_footprint_maps(
    controller: object,
    map_info: dict[str, float | int],
    traversibility_map: np.ndarray | None = None,
    keep_only_navigation_objects: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, int], dict[int, str]]:
    """Build object maps from AI2-THOR runtime geometry."""
    del traversibility_map  # Footprints can occupy non-traversable cells.

    height = int(map_info["H"])
    width = int(map_info["W"])
    object_category_map = np.zeros((height, width), dtype=np.int32)
    object_instance_map = np.zeros((height, width), dtype=np.int32)

    last_event = getattr(controller, "last_event", None)
    metadata = getattr(last_event, "metadata", {}) if last_event is not None else {}
    runtime_objects = metadata.get("objects", []) if isinstance(metadata, dict) else []
    if not isinstance(runtime_objects, list):
        runtime_objects = []

    candidates = _candidate_runtime_objects(runtime_objects, keep_only_navigation_objects)
    categories = sorted(
        {
            canonicalize_object_category(str(obj["objectType"]))
            for obj in candidates
            if isinstance(obj.get("objectType"), str)
        }
    )
    category_to_id = {category: index + 1 for index, category in enumerate(categories)}
    id_to_category = {value: key for key, value in category_to_id.items()}

    priority_grid = np.full((height, width), fill_value=-1, dtype=np.int32)
    area_grid = np.full((height, width), fill_value=-1.0, dtype=np.float32)
    height_grid = np.full((height, width), fill_value=np.inf, dtype=np.float32)

    object_metadata: list[dict[str, object]] = []
    for instance_id, obj in enumerate(candidates, start=1):
        object_id = str(obj["objectId"])
        raw_object_type = str(obj["objectType"])
        object_type = canonicalize_object_category(raw_object_type)
        position = obj.get("position")
        category_id = category_to_id[object_type]
        polygon_xz = None
        wall_segment_xz = None
        if object_type == "Wall":
            wall_segment_xz = parse_wall_segment_from_object_id(object_id)
            cells = (
                rasterize_wall_segment_to_grid(wall_segment_xz, map_info)
                if wall_segment_xz
                else []
            )
            source = "wall_segment" if cells else "center_fallback"
        else:
            polygon_xz = get_object_footprint_polygon_xz(obj)
            cells = rasterize_polygon_to_grid(polygon_xz, map_info) if polygon_xz else []
            source = "bbox"

        if not cells:
            cells = _center_fallback_cells(
                position if isinstance(position, dict) else None,
                map_info,
                radius_cells=0 if object_type == "Wall" else 1,
            )
            source = "center_fallback"

        object_height = float(position.get("y", 0.0)) if isinstance(position, dict) else 0.0
        priority = _object_priority(object_type)
        footprint_area = (
            len(cells) * float(map_info["resolution"]) * float(map_info["resolution"])
        )

        center_row, center_col = (-1, -1)
        if isinstance(position, dict):
            x = position.get("x")
            z = position.get("z")
            if isinstance(x, (int, float)) and isinstance(z, (int, float)):
                center_row, center_col = world_to_grid(
                    float(x),
                    float(z),
                    float(map_info["x_min"]),
                    float(map_info["z_min"]),
                    float(map_info["resolution"]),
                )

        record: dict[str, object] = {
            "instance_id": instance_id,
            "objectId": object_id,
            "objectType": object_type,
            "name": obj.get("name"),
            "position": position,
            "category_id": category_id,
            "has_obb": isinstance(obj.get("objectOrientedBoundingBox"), dict),
            "has_aabb": isinstance(obj.get("axisAlignedBoundingBox"), dict),
            "bbox_polygon_xz": [list(point) for point in polygon_xz] if polygon_xz else None,
            "wall_segment_xz": (
                [
                    [wall_segment_xz[0], wall_segment_xz[1]],
                    [wall_segment_xz[2], wall_segment_xz[3]],
                ]
                if wall_segment_xz
                else None
            ),
            "grid_row": center_row,
            "grid_col": center_col,
            "num_grid_cells": len(cells),
            "priority": priority,
            "footprint_area": footprint_area,
            "footprint_source": source,
        }
        if raw_object_type != object_type:
            record["raw_objectType"] = raw_object_type
        if len(cells) <= 128:
            record["grid_cells"] = [list(cell) for cell in cells]
        object_metadata.append(record)

        candidate_score = (priority, footprint_area, -object_height)
        for row, col in cells:
            existing_score = (
                int(priority_grid[row, col]),
                float(area_grid[row, col]),
                -float(height_grid[row, col]),
            )
            if candidate_score > existing_score:
                object_category_map[row, col] = category_id
                object_instance_map[row, col] = instance_id
                priority_grid[row, col] = priority
                area_grid[row, col] = footprint_area
                height_grid[row, col] = object_height

    return (
        object_category_map,
        object_instance_map,
        object_metadata,
        category_to_id,
        id_to_category,
    )


def convert_procthor_house_to_maps(
    house: dict[str, object],
    controller: object | None = None,
    resolution: float = 0.25,
    padding: float = 1.0,
    keep_only_navigation_objects: bool = False,
) -> dict[str, object]:
    """Convert a ProcTHOR house into map layers."""
    if not isinstance(house, dict):
        raise TypeError("house must be a ProcTHOR house dictionary.")

    if controller is None:
        map_info = build_map_info_from_house(
            house,
            resolution=resolution,
            padding=padding,
        )
        traversibility_map = build_floor_traversibility_map(house, map_info)
    else:
        traversibility_map, map_info, _reachable_positions = build_traversibility_map(
            controller,
            resolution=resolution,
            padding=padding,
        )

    room_type_map, room_type_to_id, _room_id_to_type = build_room_type_map(
        house,
        map_info,
    )
    room_metadata = build_room_metadata(house, map_info, room_type_to_id)
    scene_size = compute_house_scene_size(house)
    if controller is None:
        (
            object_category_map,
            object_instance_map,
            object_metadata,
            category_to_id,
            _id_to_category,
        ) = build_object_center_map(house, map_info)
    else:
        (
            object_category_map,
            object_instance_map,
            object_metadata,
            category_to_id,
            _id_to_category,
        ) = build_object_footprint_maps(
            controller,
            map_info,
            traversibility_map=traversibility_map,
            keep_only_navigation_objects=keep_only_navigation_objects,
        )
        if not object_metadata:
            (
                object_category_map,
                object_instance_map,
                object_metadata,
                category_to_id,
                _id_to_category,
            ) = build_object_center_map(house, map_info)

    return {
        "traversibility_map": traversibility_map,
        "room_type_map": room_type_map,
        "object_category_map": object_category_map,
        "object_instance_map": object_instance_map,
        "object_metadata": object_metadata,
        "room_metadata": room_metadata,
        "scene_size": scene_size,
        "map_info": map_info,
        "room_type_to_id": room_type_to_id,
        "category_to_id": category_to_id,
    }


def visualize_maps(
    traversibility_map: np.ndarray,
    room_type_map: np.ndarray,
    object_category_map: np.ndarray,
):
    """Create a quick side-by-side visualization of the generated maps."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(traversibility_map, origin="lower")
    axes[0].set_title("Traversibility Map")
    axes[0].axis("off")

    axes[1].imshow(room_type_map, origin="lower")
    axes[1].set_title("Room Type Map")
    axes[1].axis("off")

    axes[2].imshow(object_category_map, origin="lower")
    axes[2].set_title("Object Semantic Map")
    axes[2].axis("off")

    return figure, axes


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def save_scene_representation(
    scene_representation: dict[str, object],
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Save map arrays to `.npz` and metadata to `.json`."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    maps_path = output_prefix.with_name(f"{output_prefix.name}_maps.npz")
    metadata_path = output_prefix.with_name(f"{output_prefix.name}_metadata.json")

    np.savez_compressed(
        maps_path,
        traversibility_map=scene_representation["traversibility_map"],
        room_type_map=scene_representation["room_type_map"],
        object_category_map=scene_representation["object_category_map"],
        object_instance_map=scene_representation["object_instance_map"],
    )

    metadata = {
        "map_info": scene_representation["map_info"],
        "object_metadata": scene_representation["object_metadata"],
        "room_metadata": scene_representation.get("room_metadata", []),
        "scene_size": scene_representation.get("scene_size"),
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
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")

    return maps_path, metadata_path
