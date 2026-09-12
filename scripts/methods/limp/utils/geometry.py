"""2D grid geometry helpers for the LIMP adapter."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence


Cell = tuple[int, int]


def normalize_name(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("|", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def display_name(value: object) -> str:
    return normalize_name(value).replace("_", " ")


def centroid(cells: Sequence[Cell]) -> tuple[float, float]:
    if not cells:
        return 0.0, 0.0
    return (
        sum(row for row, _col in cells) / len(cells),
        sum(col for _row, col in cells) / len(cells),
    )


def bbox(cells: Sequence[Cell]) -> tuple[int, int, int, int]:
    if not cells:
        return 0, 0, 0, 0
    rows = [row for row, _col in cells]
    cols = [col for _row, col in cells]
    return min(rows), min(cols), max(rows), max(cols)


def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def min_cell_distance(first: Sequence[Cell], second: Sequence[Cell]) -> float:
    if not first or not second:
        return math.inf
    if len(first) * len(second) > 50000:
        first_point = centroid(first)
        second_point = centroid(second)
        return distance(first_point, second_point)
    best = math.inf
    for row_a, col_a in first:
        for row_b, col_b in second:
            best = min(best, math.hypot(row_a - row_b, col_a - col_b))
    return best


def cells_within_radius(
    seeds: Iterable[Cell],
    radius: int,
    grid_size: int,
) -> set[Cell]:
    result: set[Cell] = set()
    radius = max(0, int(radius))
    radius_sq = radius * radius
    for seed_row, seed_col in seeds:
        row_min = max(0, seed_row - radius)
        row_max = min(grid_size - 1, seed_row + radius)
        col_min = max(0, seed_col - radius)
        col_max = min(grid_size - 1, seed_col + radius)
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                if (row - seed_row) ** 2 + (col - seed_col) ** 2 <= radius_sq:
                    result.add((row, col))
    return result


def point_segment_distance(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    point_row, point_col = point
    first_row, first_col = first
    second_row, second_col = second
    delta_row = second_row - first_row
    delta_col = second_col - first_col
    denom = delta_row * delta_row + delta_col * delta_col
    if denom == 0:
        return distance(point, first)
    ratio = ((point_row - first_row) * delta_row + (point_col - first_col) * delta_col) / denom
    ratio = max(0.0, min(1.0, ratio))
    projected = (first_row + ratio * delta_row, first_col + ratio * delta_col)
    return distance(point, projected)

