"""Versioned, sidecar caches used by the SemPathBench evaluator.

The cache deliberately lives next to, rather than inside, source map and
instruction JSON files.  This keeps annotations portable and makes derived
evaluation data safe to delete and rebuild.  The layout is:

``<map>_metric_cache/{manifest.json,map_arrays.npz}``
``<instruction>_metric_cache/{manifest.json,segment_XXX_distance.npz}``

Map artifacts contain values shared by every instruction on that map.
Instruction artifacts contain values that depend on its ordered constraints.
New components can be added to the manifests without changing the source
schemas.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scripts.evaluation.metric_geometry import (
    clearance_distance_field_from_obstacles,
    clearance_obstacle_cells,
)
from scripts.methods.util.grid_astar import build_traversable_grid


METRIC_CACHE_SCHEMA_VERSION = 1
MAP_CACHE_DIRECTORY_SUFFIX = "_metric_cache"
INSTRUCTION_CACHE_DIRECTORY_SUFFIX = "_metric_cache"
MAP_CACHE_MANIFEST_NAME = "manifest.json"
MAP_CACHE_ARRAYS_NAME = "map_arrays.npz"
INSTRUCTION_CACHE_MANIFEST_NAME = "manifest.json"
INSTRUCTION_PATH_KEY = "_metric_instruction_path"


class MetricCacheWarning(RuntimeWarning):
    """A valid metric cache was unavailable and evaluation used a fallback."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolved_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repo_root() / candidate


def _source_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _source_is_current(signature: object, path: Path) -> bool:
    if not isinstance(signature, Mapping) or not path.exists():
        return False
    stat = path.stat()
    return (
        signature.get("size") == stat.st_size
        and signature.get("mtime_ns") == stat.st_mtime_ns
    )


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz_write(path: Path, **arrays: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def map_path_from_state(map_state: Mapping[str, object]) -> Path | None:
    raw_path = map_state.get("map_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = _resolved_path(raw_path)
    return path if path.exists() else None


def map_cache_directory(map_json_path: Path | str) -> Path:
    path = _resolved_path(map_json_path)
    return path.with_name(f"{path.stem}{MAP_CACHE_DIRECTORY_SUFFIX}")


def instruction_cache_directory(instruction_path: Path | str) -> Path:
    path = _resolved_path(instruction_path)
    return path.with_name(f"{path.stem}{INSTRUCTION_CACHE_DIRECTORY_SUFFIX}")


def _load_manifest(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_map_metric_cache(
    map_json_path: Path | str,
    map_state: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build the reusable map-level evaluator cache.

    ``map_state`` may be the validated evaluator state or a compatible layered
    map payload.  The cache uses only data that is independent of an
    instruction and of a predicted trajectory.
    """

    source_path = _resolved_path(map_json_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Map JSON does not exist: {source_path}")
    directory = map_cache_directory(source_path)
    manifest_path = directory / MAP_CACHE_MANIFEST_NAME
    arrays_path = directory / MAP_CACHE_ARRAYS_NAME
    existing = _load_manifest(manifest_path) if manifest_path.exists() else None
    if (
        not overwrite
        and arrays_path.exists()
        and isinstance(existing, Mapping)
        and existing.get("schema_version") == METRIC_CACHE_SCHEMA_VERSION
        and _source_is_current(existing.get("source"), source_path)
    ):
        return {"status": "up_to_date", "path": str(directory)}

    cache_state: Mapping[str, object] = map_state
    if not isinstance(map_state.get("grid_size"), int):
        layers = map_state.get("layers")
        occupancy = layers.get("occupancy") if isinstance(layers, Mapping) else None
        if isinstance(occupancy, Sequence) and not isinstance(occupancy, (str, bytes)):
            cache_state = {**map_state, "grid_size": len(occupancy)}
    traversable = np.asarray(build_traversable_grid(cache_state), dtype=np.bool_)
    obstacles = clearance_obstacle_cells(cache_state)
    clearance_distance = clearance_distance_field_from_obstacles(cache_state, obstacles)
    if clearance_distance is None:
        raise ValueError("Map has no valid occupancy grid for metric cache generation.")
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_npz_write(
        arrays_path,
        schema_version=np.asarray(METRIC_CACHE_SCHEMA_VERSION, dtype=np.int32),
        traversable=traversable,
        clearance_distance=np.asarray(clearance_distance, dtype=np.float32),
    )
    manifest: dict[str, object] = {
        "schema_version": METRIC_CACHE_SCHEMA_VERSION,
        "kind": "map_metric_cache",
        "source": _source_signature(source_path),
        "arrays_file": MAP_CACHE_ARRAYS_NAME,
        "components": {
            "traversable_grid": {"array": "traversable", "dtype": "bool"},
            "clearance_distance": {
                "array": "clearance_distance",
                "dtype": "float32",
                "unit": "grid_cells",
            },
        },
        "grid_shape": list(traversable.shape),
        "obstacle_cell_count": len(obstacles),
    }
    _atomic_json_write(manifest_path, manifest)
    return {"status": "written", "path": str(directory), "manifest": manifest}


def load_map_metric_cache(map_state: Mapping[str, object]) -> dict[str, np.ndarray] | None:
    """Load a current map cache, returning ``None`` when it is unavailable."""

    source_path = map_path_from_state(map_state)
    if source_path is None:
        return None
    directory = map_cache_directory(source_path)
    manifest = _load_manifest(directory / MAP_CACHE_MANIFEST_NAME)
    arrays_path = directory / MAP_CACHE_ARRAYS_NAME
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != METRIC_CACHE_SCHEMA_VERSION
        or not _source_is_current(manifest.get("source"), source_path)
        or not arrays_path.exists()
    ):
        return None
    try:
        with np.load(arrays_path) as payload:
            traversable = np.asarray(payload["traversable"], dtype=np.bool_)
            clearance_distance = np.asarray(payload["clearance_distance"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return None
    expected_size = map_state.get("grid_size")
    if isinstance(expected_size, int) and traversable.shape != (expected_size, expected_size):
        return None
    return {
        "traversable": traversable,
        "clearance_distance": clearance_distance,
    }


def warn_missing_map_cache(map_state: Mapping[str, object]) -> None:
    source_path = map_path_from_state(map_state)
    if source_path is None:
        return
    expected = map_cache_directory(source_path)
    warnings.warn(
        "Metric map cache is unavailable or stale for "
        f"{source_path}. Falling back to on-demand computation; expected {expected}. "
        "Rebuild it with scripts/evaluation/build_metric_cache.py.",
        MetricCacheWarning,
        stacklevel=2,
    )


def _instruction_path(instruction: Mapping[str, object]) -> Path | None:
    raw_path = instruction.get(INSTRUCTION_PATH_KEY)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = _resolved_path(raw_path)
    return path if path.exists() else None


def write_instruction_metric_cache(
    instruction_path: Path | str,
    map_state: Mapping[str, object],
    segments: Sequence[tuple[Mapping[str, object], np.ndarray]],
) -> dict[str, object]:
    """Write source-independent segment distance fields for one instruction."""

    source_path = _resolved_path(instruction_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Instruction JSON does not exist: {source_path}")
    map_path = map_path_from_state(map_state)
    if map_path is None:
        raise ValueError("Instruction metric cache requires map_state['map_path'].")
    directory = instruction_cache_directory(source_path)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_segments: list[dict[str, object]] = []
    for index, (descriptor, distance) in enumerate(segments, start=1):
        filename = f"segment_{index:03d}_distance.npz"
        _atomic_npz_write(
            directory / filename,
            schema_version=np.asarray(METRIC_CACHE_SCHEMA_VERSION, dtype=np.int32),
            # Preserve SPL values exactly relative to the on-demand Dijkstra.
            distance=np.asarray(distance, dtype=np.float64),
        )
        manifest_segments.append({**dict(descriptor), "file": filename})
    manifest: dict[str, object] = {
        "schema_version": METRIC_CACHE_SCHEMA_VERSION,
        "kind": "instruction_metric_cache",
        "source": _source_signature(source_path),
        "map_source": _source_signature(map_path),
        "segments": manifest_segments,
    }
    _atomic_json_write(directory / INSTRUCTION_CACHE_MANIFEST_NAME, manifest)
    return {"status": "written", "path": str(directory), "segment_count": len(segments)}


def _descriptor_matches(actual: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    keys = (
        "index",
        "from_order",
        "to_order",
        "goal_region_constraint_id",
        "active_must_avoid_ids",
    )
    return all(actual.get(key) == expected.get(key) for key in keys)


def load_instruction_segment_distance_field(
    instruction: Mapping[str, object],
    map_state: Mapping[str, object],
    descriptor: Mapping[str, object],
) -> np.ndarray | None:
    """Load one valid precomputed segment field, if available."""

    source_path = _instruction_path(instruction)
    map_path = map_path_from_state(map_state)
    if source_path is None or map_path is None:
        return None
    directory = instruction_cache_directory(source_path)
    manifest = _load_manifest(directory / INSTRUCTION_CACHE_MANIFEST_NAME)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != METRIC_CACHE_SCHEMA_VERSION
        or not _source_is_current(manifest.get("source"), source_path)
        or not _source_is_current(manifest.get("map_source"), map_path)
    ):
        return None
    raw_segments = manifest.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        return None
    matching = next(
        (
            item
            for item in raw_segments
            if isinstance(item, Mapping) and _descriptor_matches(item, descriptor)
        ),
        None,
    )
    if not isinstance(matching, Mapping) or not isinstance(matching.get("file"), str):
        return None
    try:
        with np.load(directory / str(matching["file"])) as payload:
            distance = np.asarray(payload["distance"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return None
    grid_size = map_state.get("grid_size")
    if isinstance(grid_size, int) and distance.shape != (grid_size, grid_size):
        return None
    return distance


def warn_missing_instruction_cache(instruction: Mapping[str, object]) -> None:
    source_path = _instruction_path(instruction)
    if source_path is None:
        return
    expected = instruction_cache_directory(source_path)
    warnings.warn(
        "Metric instruction cache is unavailable or stale for "
        f"{source_path}. Segment-wise SPL will run on-demand shortest-path search; "
        f"expected {expected}. Rebuild it with scripts/evaluation/build_metric_cache.py.",
        MetricCacheWarning,
        stacklevel=2,
    )
