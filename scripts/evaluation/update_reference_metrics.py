#!/usr/bin/env python3
"""Populate saved human-reference metrics in instruction JSON files."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Mapping

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluation_metrics import build_instruction_metric_cache
from scripts.evaluation.metric_cache import build_map_metric_cache
from scripts.make_instruction.make_instruction import (
    REPO_ROOT,
    annotate_soft_reference_distances,
    default_global_clearance_constraint,
    load_map_state,
)
from scripts.methods.util.instructions import (
    iter_instruction_files,
    map_id_from_instruction_path,
)


DEFAULT_INPUT_ROOT = REPO_ROOT / "resources" / "instructions"


REFERENCE_FIELDS_BY_TYPE = {
    "near_preference": ("D_ref", "reference_waypoint_count", "reference_metric_status"),
    "far_preference": ("D_ref", "reference_waypoint_count", "reference_metric_status"),
    "clearance": ("C_clear_ref", "reference_waypoint_count", "reference_metric_status"),
    "move_smoothness": (
        "C_smooth_ref",
        "reference_waypoint_count",
        "reference_metric_status",
    ),
    "path_shape_preference": (
        "reference_trajectory",
        "reference_waypoint_count",
        "reference_metric_status",
    ),
    "relative_preference": (
        "Q_relative_ref",
        "reference_waypoint_count",
        "reference_valid_path_length",
        "reference_metric_status",
        "relative_reference_status",
    ),
}
PRIMARY_REFERENCE_FIELD_BY_TYPE = {
    "near_preference": "D_ref",
    "far_preference": "D_ref",
    "clearance": "C_clear_ref",
    "move_smoothness": "C_smooth_ref",
    "path_shape_preference": "reference_trajectory",
    "relative_preference": "Q_relative_ref",
}


def load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json_object(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def constraint_reference_fields(constraint: Mapping[str, object]) -> tuple[str, ...]:
    preference_type = constraint.get("preference_type")
    if not isinstance(preference_type, str):
        return ()
    return REFERENCE_FIELDS_BY_TYPE.get(preference_type, ())


def needs_reference_update(
    constraint: Mapping[str, object],
    *,
    overwrite: bool,
    preference_types: set[str] | None = None,
) -> bool:
    preference_type = constraint.get("preference_type")
    if preference_types is not None and preference_type not in preference_types:
        return False
    if overwrite:
        return bool(constraint_reference_fields(constraint))
    if not constraint_reference_fields(constraint):
        return False
    if constraint.get("reference_metric_status") == "invalid_human_reference":
        return PRIMARY_REFERENCE_FIELD_BY_TYPE.get(str(preference_type)) not in constraint
    if preference_type == "relative_preference" and (
        constraint.get("relative_reference_status") == "invalid_human_reference"
    ):
        return "Q_relative_ref" not in constraint
    primary_field = PRIMARY_REFERENCE_FIELD_BY_TYPE.get(str(preference_type))
    if primary_field is None:
        return False
    return primary_field not in constraint or constraint.get(primary_field) is None


def copy_reference_fields(
    target: dict[str, object],
    source: Mapping[str, object],
) -> bool:
    changed = False
    for field in constraint_reference_fields(target):
        if target.get(field) != source.get(field):
            target[field] = source.get(field)
            changed = True
    return changed


def instruction_needs_map(
    soft_constraints: list[object],
    *,
    overwrite: bool,
    preference_types: set[str] | None = None,
) -> bool:
    for constraint in soft_constraints:
        if not isinstance(constraint, Mapping):
            continue
        if constraint.get("preference_type") != "clearance":
            continue
        if needs_reference_update(
            constraint,
            overwrite=overwrite,
            preference_types=preference_types,
        ):
            return True
    return False


def should_ensure_clearance(preference_types: set[str] | None) -> bool:
    return preference_types is None or "clearance" in preference_types


def ensure_clearance_constraint(
    soft_constraints: list[object],
    *,
    preference_types: set[str] | None = None,
) -> bool:
    if not should_ensure_clearance(preference_types):
        return False
    if any(
        isinstance(constraint, Mapping)
        and constraint.get("preference_type") == "clearance"
        for constraint in soft_constraints
    ):
        return False
    soft_constraints.append(default_global_clearance_constraint())
    return True


def update_instruction_payload(
    payload: dict[str, object],
    *,
    instruction_path: Path,
    input_root: Path,
    overwrite: bool = False,
    preference_types: set[str] | None = None,
) -> dict[str, int]:
    trajectory = payload.get("human_expert_trajectory")
    soft_constraints = payload.get("soft_constraints")
    hard_constraints = payload.get("hard_constraints")
    if not isinstance(trajectory, list):
        raise ValueError("Instruction has no human_expert_trajectory list.")
    if not isinstance(soft_constraints, list):
        return {
            "constraints": 0,
            "updated_constraints": 0,
            "updated_fields": 0,
            "inserted_clearance": 0,
        }
    if not isinstance(hard_constraints, list):
        hard_constraints = []

    inserted_clearance = ensure_clearance_constraint(
        soft_constraints,
        preference_types=preference_types,
    )
    pending_indexes = [
        index
        for index, constraint in enumerate(soft_constraints)
        if isinstance(constraint, Mapping)
        and needs_reference_update(
            constraint,
            overwrite=overwrite,
            preference_types=preference_types,
        )
    ]
    if not pending_indexes:
        return {
            "constraints": 0,
            "updated_constraints": 0,
            "updated_fields": 0,
            "inserted_clearance": int(inserted_clearance),
        }

    needs_full_soft_context = any(
        isinstance(soft_constraints[index], Mapping)
        and soft_constraints[index].get("preference_type") == "clearance"
        for index in pending_indexes
    )
    sample_constraints = (
        copy.deepcopy(soft_constraints)
        if needs_full_soft_context
        else [copy.deepcopy(soft_constraints[index]) for index in pending_indexes]
    )
    sample = {
        "expert_route": trajectory,
        "hard_constraints": hard_constraints,
        "soft_constraints": sample_constraints,
    }
    map_state = None
    if instruction_needs_map(
        soft_constraints,
        overwrite=overwrite,
        preference_types=preference_types,
    ):
        map_id = map_id_from_instruction_path(instruction_path, input_root)
        map_state = load_map_state(map_id)

    annotate_soft_reference_distances(sample, map_state)
    computed_constraints = sample["soft_constraints"]
    if not isinstance(computed_constraints, list):
        raise ValueError("Reference metric computation produced invalid constraints.")

    updated_constraints = 0
    updated_fields = 0
    for pending_offset, index in enumerate(pending_indexes):
        original = soft_constraints[index]
        computed_index = index if needs_full_soft_context else pending_offset
        computed = computed_constraints[computed_index]
        if not isinstance(original, dict) or not isinstance(computed, Mapping):
            continue
        before = dict(original)
        if copy_reference_fields(original, computed):
            updated_constraints += 1
            updated_fields += sum(
                1
                for field in constraint_reference_fields(original)
                if before.get(field) != original.get(field)
            )

    return {
        "constraints": len(pending_indexes),
        "updated_constraints": updated_constraints,
        "updated_fields": updated_fields,
        "inserted_clearance": int(inserted_clearance),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute saved GT reference metrics such as D_ref, C_clear_ref, "
            "C_smooth_ref, reference_trajectory, and Q_relative_ref."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--set",
        choices=("valunseen", "train", "all"),
        default="all",
        dest="instruction_set",
        help="Instruction split to update.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing reference fields as well.",
    )
    parser.add_argument(
        "--preference-type",
        action="append",
        choices=tuple(REFERENCE_FIELDS_BY_TYPE),
        dest="preference_types",
        help="Restrict updates to one soft preference type. Repeat as needed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first malformed instruction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preference_types = (
        set(args.preference_types)
        if isinstance(args.preference_types, list) and args.preference_types
        else None
    )
    totals = {
        "files": 0,
        "changed_files": 0,
        "constraints": 0,
        "updated_constraints": 0,
        "updated_fields": 0,
        "inserted_clearance": 0,
        "failed_files": 0,
    }
    for path in iter_instruction_files(
        args.input_root,
        instruction_set=args.instruction_set,
    ):
        totals["files"] += 1
        try:
            payload = load_json_object(path)
            stats = update_instruction_payload(
                payload,
                instruction_path=path,
                input_root=args.input_root,
                overwrite=args.overwrite,
                preference_types=preference_types,
            )
        except Exception as exc:
            totals["failed_files"] += 1
            print(f"[failed] {path}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise
            continue

        totals["constraints"] += stats["constraints"]
        totals["updated_constraints"] += stats["updated_constraints"]
        totals["updated_fields"] += stats["updated_fields"]
        totals["inserted_clearance"] += stats["inserted_clearance"]
        if stats["updated_fields"] == 0 and stats["inserted_clearance"] == 0:
            continue
        totals["changed_files"] += 1
        action = "would update" if args.dry_run else "updated"
        print(
            f"[{action}] {path} constraints={stats['constraints']} "
            f"updated_constraints={stats['updated_constraints']} "
            f"updated_fields={stats['updated_fields']}"
        )
        if not args.dry_run:
            write_json_object(path, payload)
            map_id = map_id_from_instruction_path(path, args.input_root)
            map_state = load_map_state(map_id)
            build_map_metric_cache(Path(str(map_state["map_path"])), map_state)
            build_instruction_metric_cache(path, payload, map_state)

    print(json.dumps(totals, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
