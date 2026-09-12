#!/usr/bin/env python3
"""High-level SemPathBench prediction evaluation helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.annotation.path_shape_schema import (
    PATH_SHAPE_ANNOTATION_FIELD,
    PATH_SHAPE_PREFERENCE_TYPES,
    normalize_path_shape_annotation,
)
from scripts.evaluation.evaluation_metrics import Point, evaluate_trajectory
from scripts.make_instruction.make_instruction import (
    REPO_ROOT,
    instruction_files_directory,
    load_map_state,
    normalize_map_key,
    validate_route,
)
from scripts.methods.util.instructions import instruction_id_from_payload, load_instruction


def apply_embedded_path_shape_annotations(
    instruction: dict[str, object],
    *,
    grid_size: int,
    instruction_path: Path | None = None,
) -> list[dict[str, object]]:
    """Load embedded canonical references into the evaluator compatibility field."""
    raw_constraints = instruction.get("soft_constraints")
    if not isinstance(raw_constraints, list):
        return []

    loaded: list[dict[str, object]] = []
    for index, constraint in enumerate(raw_constraints):
        if not isinstance(constraint, dict) or constraint.get(
            "preference_type"
        ) not in PATH_SHAPE_PREFERENCE_TYPES:
            continue
        constraint_id = constraint.get("constraint_id")
        if not isinstance(constraint_id, str) or not constraint_id.strip():
            continue

        raw_annotation = constraint.get(PATH_SHAPE_ANNOTATION_FIELD)
        if raw_annotation is None:
            continue
        annotation = normalize_path_shape_annotation(
            raw_annotation,
            grid_size=grid_size,
            label=f"soft_constraints[{index}].{PATH_SHAPE_ANNOTATION_FIELD}",
        )
        reference = annotation["shape_reference_trajectory"]
        source = annotation["shape_reference_source"]
        constraint[PATH_SHAPE_ANNOTATION_FIELD] = annotation
        constraint["reference_trajectory"] = reference
        constraint["reference_waypoint_count"] = len(reference)  # type: ignore[arg-type]
        constraint["_path_shape_reference_source"] = source

        instruction_display = None
        if instruction_path is not None:
            instruction_display = (
                instruction_path.relative_to(REPO_ROOT).as_posix()
                if instruction_path.is_relative_to(REPO_ROOT)
                else str(instruction_path)
            )
            constraint["_path_shape_reference_file"] = instruction_display

        loaded.append(
            {
                "constraint_id": constraint_id,
                "source": source,
                "file": instruction_display,
                "field": f"soft_constraints[{index}].{PATH_SHAPE_ANNOTATION_FIELD}",
                "waypoint_count": len(reference),  # type: ignore[arg-type]
            }
        )
    return loaded


def find_instruction_file(map_id: str, instruction_id: str) -> Path:
    """Resolve one instruction id in the map's instruction directory."""
    map_key = normalize_map_key(map_id)
    directory = instruction_files_directory(map_key)
    requested = instruction_id.strip()
    candidates = [directory / requested]
    if not requested.endswith(".json"):
        candidates.append(directory / f"{requested}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path in sorted(directory.glob("instruction_*.json")):
        try:
            payload = load_instruction(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if instruction_id_from_payload(path, payload) == requested:
            return path
    raise ValueError(f"Unknown instruction_id for {map_key}: {instruction_id!r}")


def load_instruction_by_id(
    map_id: str,
    instruction_id: str,
) -> tuple[dict[str, object], Path]:
    path = find_instruction_file(map_id, instruction_id)
    return load_instruction(path), path


def evaluate_prediction(
    trajectory: Sequence[Point],
    *,
    map_id: str | None = None,
    instruction_id: str | None = None,
    instruction_file: Path | str | None = None,
) -> dict[str, object]:
    """Evaluate one predicted trajectory against one instruction."""
    if instruction_file is None:
        if not map_id or not instruction_id:
            raise ValueError(
                "evaluate_prediction requires instruction_file or map_id + instruction_id."
            )
        instruction, _path = load_instruction_by_id(map_id, instruction_id)
    else:
        instruction = load_instruction(instruction_file)
        _path = Path(instruction_file)

    # Private runtime metadata: never written back to the instruction JSON.
    # It lets the evaluator locate the instruction-level metric sidecar cache.
    instruction["_metric_instruction_path"] = str(_path)

    effective_map_id = map_id
    if not effective_map_id:
        raw_map_id = instruction.get("map_id")
        if not isinstance(raw_map_id, str) or not raw_map_id.strip():
            raise ValueError("map_id is required when the instruction has no map_id.")
        effective_map_id = raw_map_id

    map_state = load_map_state(effective_map_id)
    apply_embedded_path_shape_annotations(
        instruction,
        grid_size=int(map_state["grid_size"]),
        instruction_path=_path,
    )
    route = validate_route(
        [list(point) for point in trajectory],
        int(map_state["grid_size"]),
        label="trajectory",
    )
    return evaluate_trajectory(route, instruction, map_state)


def _trajectory_from_payload(payload: object) -> Sequence[Point]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("trajectory", "predicted_trajectory", "human_expert_trajectory"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(
        "Trajectory JSON must be a point list or contain trajectory, "
        "predicted_trajectory, or human_expert_trajectory."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trajectory using SemPathBench metrics."
    )
    parser.add_argument("--map-id", help="Map id, e.g. procthor/002_train.")
    parser.add_argument("--instruction-id", help="Instruction id, e.g. instruction_000001.")
    parser.add_argument("--instruction-file", type=Path)
    parser.add_argument("--trajectory-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_payload = json.loads(args.trajectory_file.read_text(encoding="utf-8"))
    metrics = evaluate_prediction(
        _trajectory_from_payload(trajectory_payload),
        map_id=args.map_id,
        instruction_id=args.instruction_id,
        instruction_file=args.instruction_file,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
