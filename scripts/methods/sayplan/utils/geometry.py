"""Small geometry helpers for grid masks."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


Cell = tuple[int, int]


def centroid(cells: Sequence[Cell]) -> Cell | None:
    if not cells:
        return None
    row = round(sum(cell[0] for cell in cells) / len(cells))
    col = round(sum(cell[1] for cell in cells) / len(cells))
    return int(row), int(col)


def nearest_cell(seed: Cell, candidates: Iterable[Cell]) -> Cell | None:
    best: Cell | None = None
    best_distance = math.inf
    for cell in candidates:
        distance = math.hypot(cell[0] - seed[0], cell[1] - seed[1])
        if distance < best_distance:
            best = cell
            best_distance = distance
    return best


def path_length(path: Sequence[Sequence[int]]) -> float:
    if len(path) <= 1:
        return 0.0
    total = 0.0
    for first, second in zip(path, path[1:]):
        total += math.hypot(int(second[0]) - int(first[0]), int(second[1]) - int(first[1]))
    return total

