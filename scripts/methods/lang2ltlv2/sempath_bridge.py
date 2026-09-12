"""Bridge SemPathBench maps into Lang2LTL-2-style semantic inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from scripts.methods.lang2ltl.map_features import MapSymbols, build_map_symbols


Point = tuple[int, int]


@dataclass(frozen=True)
class BridgeArtifact:
    graph_dpath: Path
    osm_fpath: Path
    obj_locs_fpath: Path
    metadata_fpath: Path
    symbols: MapSymbols
    ap_metadata: dict[str, dict[str, object]]
    bridge_metadata: dict[str, object]


def _entity_center_xy(center: tuple[float, float]) -> dict[str, float]:
    row, col = center
    return {"x": float(col), "y": float(row)}


def _semantic_description(kind: str, category: str) -> str:
    article = "an" if category[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    label = category.replace("_", " ")
    if kind == "room":
        return f"{article} {label} room"
    return f"{article} {label}"


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_lang2ltl2_map_cache(
    *,
    map_id: str,
    map_state: Mapping[str, object],
    start_cell: Point,
    cache_root: Path,
    overwrite: bool = False,
) -> BridgeArtifact:
    """Write pseudo graph/OSM files consumed by the original Lang2LTL-2 modules."""
    symbols = build_map_symbols(map_state)
    map_cache = cache_root / map_id
    graph_dpath = map_cache / "graph"
    images_dpath = graph_dpath / "images"
    osm_fpath = map_cache / "osm_semantics.json"
    obj_locs_fpath = graph_dpath / "obj_locs.json"
    metadata_fpath = map_cache / "bridge_metadata.json"

    images_dpath.mkdir(parents=True, exist_ok=True)

    obj_locs: dict[str, dict[str, float]] = {
        "waypoint_0": {"x": float(start_cell[1]), "y": float(start_cell[0])}
    }
    osm_semantics: dict[str, dict[str, object]] = {}
    ap_metadata: dict[str, dict[str, object]] = {}

    for entity in (*symbols.room_entities, *symbols.object_entities):
        waypoint_id = f"wp_{entity.ap}"
        obj_locs[waypoint_id] = _entity_center_xy(entity.center)
        description = _semantic_description(entity.kind, entity.category)
        osm_semantics[entity.ap] = {
            "name": entity.ap,
            "kind": entity.kind,
            "category": entity.category,
            "description": description,
            "wid": waypoint_id,
            "lat": 0.0,
            "long": 0.0,
        }
        ap_metadata[entity.ap] = {
            "ap": entity.ap,
            "waypoint_id": waypoint_id,
            "kind": entity.kind,
            "id": entity.id,
            "category": entity.category,
            "label": entity.label,
            "center": [float(entity.center[0]), float(entity.center[1])],
            "cell_count": entity.cell_count,
            "room_id": entity.room_id,
            "room_category": entity.room_category,
            "attributes": list(entity.attributes),
        }

    bridge_metadata: dict[str, object] = {
        "map_id": map_id,
        "grid_size": int(map_state["grid_size"]),
        "text_only": True,
        "visual_branch_enabled": False,
        "coordinate_convention": {
            "x": "grid_col",
            "y": "grid_row",
            "origin": "top_left",
        },
        "room_count": len(symbols.room_entities),
        "object_count": len(symbols.object_entities),
        "graph_dpath": str(graph_dpath),
        "osm_fpath": str(osm_fpath),
        "obj_locs_fpath": str(obj_locs_fpath),
    }

    if overwrite or not obj_locs_fpath.exists():
        _json_write(obj_locs_fpath, obj_locs)
    if overwrite or not osm_fpath.exists():
        _json_write(osm_fpath, osm_semantics)
    if overwrite or not metadata_fpath.exists():
        _json_write(metadata_fpath, bridge_metadata)

    return BridgeArtifact(
        graph_dpath=graph_dpath,
        osm_fpath=osm_fpath,
        obj_locs_fpath=obj_locs_fpath,
        metadata_fpath=metadata_fpath,
        symbols=symbols,
        ap_metadata=ap_metadata,
        bridge_metadata=bridge_metadata,
    )
