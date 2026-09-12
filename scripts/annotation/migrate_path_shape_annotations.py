#!/usr/bin/env python3
"""Move standalone path-shape sidecars into their owning instruction JSON files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.path_shape_schema import (  # noqa: E402
    PATH_SHAPE_ANNOTATION_FIELD,
    PATH_SHAPE_PREFERENCE_TYPES,
    normalize_path_shape_annotation,
)
from scripts.make_instruction.make_instruction import (  # noqa: E402
    instruction_files_directory,
    load_map_state,
    normalize_map_key,
)


DEFAULT_SOURCE_ROOT = REPO_ROOT / "resources" / "path_shape_annotations"


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_write(path: Path, value: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def current_sidecars(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*.json")
        if "_history" not in path.parts
    )


def embedded_record(
    sidecar_path: Path,
    payload: Mapping[str, object],
    *,
    grid_size: int,
) -> dict[str, object]:
    history_directory = (
        sidecar_path.parent / "_history" / sidecar_path.stem
    )
    history = [
        normalize_path_shape_annotation(
            load_object(history_path),
            grid_size=grid_size,
            include_history=False,
            label=str(history_path),
        )
        for history_path in sorted(history_directory.glob("*.json"))
    ]
    annotation = {
        key: payload.get(key)
        for key in (
            "version",
            "annotation_type",
            "shape_spec",
            "shape_reference_trajectory",
            "shape_reference_source",
            "annotator_id",
            "notes",
            "created_at",
            "updated_at",
        )
    }
    annotation["history"] = history
    return normalize_path_shape_annotation(
        annotation,
        grid_size=grid_size,
        label=str(sidecar_path),
    )


def find_constraint(
    instruction: Mapping[str, object],
    constraint_id: str,
) -> dict[str, object]:
    matches = [
        constraint
        for constraint in instruction.get("soft_constraints", [])
        if isinstance(constraint, dict)
        and constraint.get("preference_type") in PATH_SHAPE_PREFERENCE_TYPES
        and constraint.get("constraint_id") == constraint_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one path-shape constraint {constraint_id!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def instruction_path_for(payload: Mapping[str, object]) -> Path:
    map_id = payload.get("map_id")
    instruction_id = payload.get("instruction_id")
    if not isinstance(map_id, str) or not isinstance(instruction_id, str):
        raise ValueError("Sidecar map_id and instruction_id must be strings.")
    path = instruction_files_directory(normalize_map_key(map_id)) / f"{instruction_id}.json"
    if not path.is_file():
        raise ValueError(f"Instruction file does not exist: {path}")
    return path


def migrate(
    source_root: Path,
    *,
    write: bool,
    delete_source: bool,
) -> tuple[int, int, int]:
    if delete_source and not write:
        raise ValueError("--delete-source requires --write.")
    if not source_root.is_dir():
        raise ValueError(f"Source directory does not exist: {source_root}")

    sidecars = current_sidecars(source_root)
    grouped: dict[Path, list[tuple[Path, dict[str, object]]]] = defaultdict(list)
    for path in sidecars:
        payload = load_object(path)
        grouped[instruction_path_for(payload)].append((path, payload))

    annotations = 0
    history_records = 0
    rewritten: dict[Path, dict[str, object]] = {}
    for instruction_path, records in sorted(grouped.items()):
        instruction = load_object(instruction_path)
        map_id = instruction.get("map_id")
        if not isinstance(map_id, str):
            raise ValueError(f"Instruction has no map_id: {instruction_path}")
        grid_size = int(load_map_state(map_id)["grid_size"])
        for sidecar_path, payload in records:
            constraint_id = payload.get("constraint_id")
            if not isinstance(constraint_id, str) or not constraint_id:
                raise ValueError(f"Sidecar has no constraint_id: {sidecar_path}")
            constraint = find_constraint(instruction, constraint_id)
            annotation = embedded_record(
                sidecar_path,
                payload,
                grid_size=grid_size,
            )
            constraint[PATH_SHAPE_ANNOTATION_FIELD] = annotation
            trajectory = annotation["shape_reference_trajectory"]
            constraint["reference_trajectory"] = trajectory
            constraint["reference_waypoint_count"] = len(trajectory)  # type: ignore[arg-type]
            constraint["reference_metric_status"] = "computed"
            annotations += 1
            history_records += len(annotation["history"])  # type: ignore[arg-type]
        rewritten[instruction_path] = instruction

    all_instruction_constraints = 0
    for path in (REPO_ROOT / "resources" / "instructions").rglob("instruction_*.json"):
        instruction = rewritten.get(path) or load_object(path)
        all_instruction_constraints += sum(
            isinstance(constraint, Mapping)
            and constraint.get("preference_type") in PATH_SHAPE_PREFERENCE_TYPES
            for constraint in instruction.get("soft_constraints", [])
        )
    if annotations != all_instruction_constraints:
        raise ValueError(
            "Migration coverage mismatch: "
            f"{annotations} sidecars for {all_instruction_constraints} constraints."
        )

    if write:
        for path, instruction in sorted(rewritten.items()):
            atomic_write(path, instruction)
        if delete_source:
            shutil.rmtree(source_root)

    return annotations, history_records, len(rewritten)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite instruction files; otherwise only validate and report.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the standalone directory after a successful write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations, history_records, instruction_files = migrate(
        args.source_root.resolve(),
        write=args.write,
        delete_source=args.delete_source,
    )
    action = "Migrated" if args.write else "Validated"
    print(
        f"{action} {annotations} annotations and {history_records} history records "
        f"across {instruction_files} instruction files."
    )
    if args.delete_source:
        print(f"Deleted standalone source directory: {args.source_root.resolve()}")


if __name__ == "__main__":
    main()
