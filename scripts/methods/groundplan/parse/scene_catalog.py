"""Compact, annotation-free scene catalog for GroundPlan LLM ablations.

The raw benchmark map JSON contains dense grid layers and object footprints and
is far too large to place in an LLM prompt.  This module exposes only the
groundable entity inventory and lightweight geometry needed by the Direct-ID
and LTL representation experiments.  It deliberately does not read benchmark
hard/soft constraints or human trajectories.
"""

from __future__ import annotations

from typing import Mapping

from scripts.methods.groundplan.grounding.scene import SceneMap
from scripts.methods.groundplan.ir import GPRef


def _bbox(ref: GPRef) -> list[int] | None:
    if not ref.cells:
        return None
    rows = [row for row, _col in ref.cells]
    cols = [col for _row, col in ref.cells]
    return [min(rows), min(cols), max(rows), max(cols)]


def _center(ref: GPRef) -> list[float] | None:
    if ref.center is None:
        return None
    return [round(float(ref.center[0]), 3), round(float(ref.center[1]), 3)]


def compact_scene_catalog(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    *,
    scene: SceneMap | None = None,
) -> dict[str, object]:
    """Return a compact semantic catalog containing no dense map arrays."""

    scene = scene or SceneMap(map_state, instruction)
    rooms = [
        {
            "id": ref.id,
            "category": ref.category,
            "center_grid": _center(ref),
            "bbox_grid": _bbox(ref),
            "area_cells": len(ref.cells),
        }
        for ref in sorted(scene.rooms.values(), key=lambda item: item.id)
    ]
    objects = [
        {
            "id": ref.id,
            "category": ref.category,
            "room_id": ref.room_id,
            "center_grid": _center(ref),
            "bbox_grid": _bbox(ref),
            "footprint_cell_count": len(ref.cells),
        }
        for ref in sorted(scene.entities.values(), key=lambda item: item.id)
    ]
    adjacency = [
        {
            "room_ids": list(pair),
            "passage_count": len(passages),
        }
        for pair, passages in sorted(scene.passages.items())
    ]
    return {
        "schema_version": 1,
        "coordinate_frame": {
            "type": "grid_row_col",
            "resolution_m": float(scene.resolution),
            "grid_size": int(scene.grid_size),
        },
        "start": {
            "id": scene.start.id,
            "room_id": scene.start.room_id,
            "center_grid": _center(scene.start),
        },
        "rooms": rooms,
        "objects": objects,
        "room_adjacency": adjacency,
    }


def catalog_reference(scene: SceneMap, reference_id: object) -> GPRef:
    """Resolve one canonical catalog ID without consulting annotations."""

    if reference_id in {"task_start", "start_position"}:
        return scene.start
    if not isinstance(reference_id, str) or not reference_id.strip():
        raise ValueError("Reference ID must be a non-empty string.")
    normalized = reference_id.strip()
    if normalized in scene.rooms:
        return scene.rooms[normalized]
    if normalized in scene.entities:
        return scene.entities[normalized]
    raise ValueError(f"Unknown compact-catalog reference ID: {normalized}")

