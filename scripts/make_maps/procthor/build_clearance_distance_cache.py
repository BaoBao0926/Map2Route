#!/usr/bin/env python3
"""Precompute clearance distance fields for ProcTHOR map JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluation_metrics import (
    CLEARANCE_DISTANCE_CACHE_SUFFIX,
    CLEARANCE_DISTANCE_CACHE_VERSION,
    EPSILON,
    REPO_ROOT,
    clearance_obstacle_cells,
)
from scripts.evaluation.metric_cache import build_map_metric_cache


DEFAULT_MAP_ROOT = REPO_ROOT / "resources" / "maps" / "procthor"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return str(path)


def is_layered_map_payload(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        return False
    return isinstance(layers.get("occupancy"), list)


def iter_map_json_files(map_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(map_root.rglob("*.json")):
        if any(part.endswith("_metric_cache") for part in path.relative_to(map_root).parts):
            continue
        if path.name == "template_instruction.json":
            continue
        if path.name.endswith("_metadata.json") or path.name.endswith("_thinggraph.json"):
            continue
        candidates.append(path)
    return candidates


def cache_path_for_map_json(map_json_path: Path) -> Path:
    return map_json_path.with_name(
        f"{map_json_path.stem}{CLEARANCE_DISTANCE_CACHE_SUFFIX}"
    )


def occupancy_shape(map_state: Mapping[str, object]) -> tuple[int, int]:
    layers = map_state.get("layers")
    occupancy = layers.get("occupancy") if isinstance(layers, Mapping) else None
    if not isinstance(occupancy, list) or not occupancy:
        raise ValueError("Map payload must contain a non-empty layers.occupancy grid.")
    row_count = len(occupancy)
    col_count = max(len(row) for row in occupancy if isinstance(row, list))
    return row_count, col_count


def build_obstacle_mask(map_state: Mapping[str, object]) -> np.ndarray:
    row_count, col_count = occupancy_shape(map_state)
    obstacle_mask = np.zeros((row_count, col_count), dtype=bool)
    for row, col, _object_id in clearance_obstacle_cells(map_state):
        if 0 <= row < row_count and 0 <= col < col_count:
            obstacle_mask[row, col] = True
    return obstacle_mask


def build_clearance_arrays(
    map_state: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, int]:
    obstacle_mask = build_obstacle_mask(map_state)
    obstacle_count = int(obstacle_mask.sum())
    if obstacle_count == 0:
        distance = np.full(obstacle_mask.shape, np.inf, dtype=np.float32)
        cost = np.zeros(obstacle_mask.shape, dtype=np.float32)
        return distance, cost, obstacle_count

    distance = distance_transform_edt(~obstacle_mask).astype(np.float32)
    cost = (1.0 / np.square(distance.astype(np.float64) + EPSILON)).astype(np.float32)
    return distance, cost, obstacle_count


def build_cache_for_map(
    map_json_path: Path,
    *,
    overwrite: bool,
) -> dict[str, object]:
    output_path = cache_path_for_map_json(map_json_path)
    if (
        output_path.exists()
        and not overwrite
        and output_path.stat().st_mtime >= map_json_path.stat().st_mtime
    ):
        payload = json.loads(map_json_path.read_text(encoding="utf-8"))
        metric_cache = (
            build_map_metric_cache(map_json_path, payload)
            if is_layered_map_payload(payload)
            else None
        )
        return {
            "status": "skipped_exists",
            "map": display_path(map_json_path),
            "path": display_path(output_path),
            "metric_cache": metric_cache.get("status") if metric_cache else None,
        }

    payload = json.loads(map_json_path.read_text(encoding="utf-8"))
    if not is_layered_map_payload(payload):
        return {
            "status": "skipped_not_layered_map",
            "map": display_path(map_json_path),
        }

    distance, cost, obstacle_count = build_clearance_arrays(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        version=np.array(CLEARANCE_DISTANCE_CACHE_VERSION, dtype=np.int32),
        map_json=np.array(display_path(map_json_path)),
        grid_shape=np.array(distance.shape, dtype=np.int32),
        obstacle_count=np.array(obstacle_count, dtype=np.int64),
        epsilon=np.array(EPSILON, dtype=np.float64),
        distance=distance,
        cost=cost,
    )
    metric_cache = build_map_metric_cache(map_json_path, payload, overwrite=overwrite)
    return {
        "status": "written",
        "map": display_path(map_json_path),
        "path": display_path(output_path),
        "grid_shape": list(distance.shape),
        "obstacle_count": obstacle_count,
        "metric_cache": metric_cache.get("status"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ProcTHOR clearance distance-field caches."
    )
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_files = iter_map_json_files(args.map_root.resolve())
    if args.limit is not None:
        map_files = map_files[: args.limit]

    results: list[dict[str, object]] = []
    total = len(map_files)
    for index, map_json_path in enumerate(map_files, start=1):
        print(
            f"[{index}/{total}] {display_path(map_json_path)} started",
            flush=True,
        )
        result = build_cache_for_map(map_json_path, overwrite=args.overwrite)
        results.append(result)
        print(
            f"[{index}/{total}] {result['status']} {result.get('path', result['map'])}",
            flush=True,
        )

    written = sum(1 for result in results if result["status"] == "written")
    skipped = sum(1 for result in results if str(result["status"]).startswith("skipped"))
    print(f"total={len(results)} written={written} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
