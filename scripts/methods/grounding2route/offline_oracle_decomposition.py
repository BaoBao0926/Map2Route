#!/usr/bin/env python3
"""Offline Direct-CaP route-grounding decomposition.

This script never instantiates an LLM client.  It deterministically executes the
saved, verifier-accepted Direct-CaP source to recover the exact full-geometry
GroundedProgram, then sends controlled variants to the unchanged fixed planner.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluate_prediction import evaluate_prediction
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state
from scripts.methods.grounding2route.constraint_evaluation import (
    _match_cost,
    canonical_ground_truth_constraints,
    canonicalize_constraint,
)
from scripts.methods.grounding2route.grounding.code import execute_grounding_code
from scripts.methods.grounding2route.grounding.scene import SceneMap
from scripts.methods.grounding2route.ir import GPRef, GroundedConstraint, GroundedProgram, GroundedSegment
from scripts.methods.grounding2route.planner import plan_grounded_program
from scripts.methods.grounding2route.planner.debug_annotation_planner import _grounded_program_from_annotations
from scripts.methods.tutorial.utils import metric_block
from scripts.methods.util.instructions import load_instruction

SETTINGS = ("pred_all", "oracle_entities", "oracle_relations", "oracle_scope", "oracle_all_grounding")


def _prediction_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("instruction_*.json") if not path.name.endswith(".steps.json"))


def _source_for_prediction(path: Path) -> str:
    payload = json.loads(path.with_suffix(".steps.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    if not isinstance(steps, Mapping):
        raise ValueError(f"missing steps in {path}")
    grounding = steps.get("grounding_code")
    if isinstance(grounding, Mapping) and isinstance(grounding.get("source"), str):
        return str(grounding["source"])
    for attempt in reversed(steps.get("grounding_code_attempts", [])):
        if isinstance(attempt, Mapping) and attempt.get("status") == "success" and isinstance(attempt.get("code"), str):
            return str(attempt["code"])
    raise ValueError(f"no accepted grounding code in {path}")


def _ref(scene: SceneMap, raw: Mapping[str, object]) -> GPRef:
    identifier = str(raw.get("id") or "")
    if identifier == scene.start.id:
        return scene.start
    if identifier in scene.entities:
        return scene.entities[identifier]
    if identifier in scene.rooms:
        return scene.rooms[identifier]
    # Annotation route constraints currently reference semantic objects.  This
    # fallback keeps the experiment explicit if a future annotation adds a region.
    center = raw.get("center")
    parsed_center = None
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)) and len(center) == 2:
        parsed_center = (float(center[0]), float(center[1]))
    return GPRef(kind=str(raw.get("kind") or "region"), id=identifier, category=raw.get("category") if isinstance(raw.get("category"), str) else None, center=parsed_center)  # type: ignore[arg-type]


def _oracle_constraint(scene: SceneMap, raw: Mapping[str, object]) -> GroundedConstraint:
    refs = tuple(_ref(scene, item) for item in raw.get("refs", []) if isinstance(item, Mapping))
    scope = raw.get("segment_scope", ())
    segment_scope = tuple(str(item) for item in scope) if isinstance(scope, Sequence) and not isinstance(scope, (str, bytes)) else ()
    return GroundedConstraint(
        kind=str(raw.get("kind")),  # type: ignore[arg-type]
        refs=refs,
        segment_scope=segment_scope,
        relation=raw.get("relation") if isinstance(raw.get("relation"), str) else None,
        constraint_id=str(raw.get("constraint_id") or ""),
        argument_roles=tuple(str(item) for item in raw.get("argument_roles", ()) if isinstance(item, str)),
        hardness=raw.get("hardness") if raw.get("hardness") in {"hard", "soft"} else None,  # type: ignore[arg-type]
        source_text=raw.get("source_text") if isinstance(raw.get("source_text"), str) else None,
    )


def _matches(program: GroundedProgram, gt_raw: Sequence[Mapping[str, object]]) -> tuple[dict[tuple[int, int], Mapping[str, object]], set[int]]:
    observed = []
    for segment_index, segment in enumerate(program.segments):
        for constraint_index, constraint in enumerate(segment.constraints):
            observed.append((segment_index, constraint_index, canonicalize_constraint(constraint.to_json())))
    gt = [canonicalize_constraint(item) for item in gt_raw]
    unmatched = set(range(len(observed)))
    matched: dict[tuple[int, int], Mapping[str, object]] = {}
    matched_gt: set[int] = set()
    for gt_index, target in enumerate(gt):
        candidates = [index for index in unmatched if observed[index][2]["hardness"] == target["hardness"]]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: _match_cost(target, observed[index][2]))
        if _match_cost(target, observed[best][2]) >= 12:
            continue
        unmatched.remove(best)
        matched[(observed[best][0], observed[best][1])] = gt_raw[gt_index]
        matched_gt.add(gt_index)
    return matched, matched_gt


def _transform(program: GroundedProgram, scene: SceneMap, instruction: Mapping[str, object], setting: str) -> GroundedProgram:
    if setting == "pred_all":
        return program
    oracle_program, _annotation_dump = _grounded_program_from_annotations(scene, instruction)
    if setting == "oracle_all_grounding":
        # Full grounding upper bound: annotation-derived ordered goals and route
        # constraints, sent to exactly the same fixed planner.
        return oracle_program
    gt_raw = canonical_ground_truth_constraints(instruction, scene)
    matches, matched_gt = _matches(program, gt_raw)
    segments: list[GroundedSegment] = []
    for segment_index, segment in enumerate(program.segments):
        constraints: list[GroundedConstraint] = []
        for constraint_index, constraint in enumerate(segment.constraints):
            target = matches.get((segment_index, constraint_index))
            if target is None:
                if setting == "oracle_all_grounding":
                    continue  # Remove an unmatched predicted route constraint.
                constraints.append(constraint)
                continue
            oracle = _oracle_constraint(scene, target)
            if setting == "oracle_entities":
                constraint = replace(constraint, refs=oracle.refs, argument_roles=oracle.argument_roles)
            elif setting == "oracle_relations":
                constraint = replace(constraint, relation=oracle.relation)
            elif setting == "oracle_scope":
                constraint = replace(constraint, segment_scope=oracle.segment_scope, spatial_scope=oracle.spatial_scope)
            elif setting == "oracle_all_grounding":
                constraint = oracle
            constraints.append(constraint)
        oracle_target = oracle_program.segments[segment_index].target if segment_index < len(oracle_program.segments) else segment.target
        segments.append(replace(segment, target=oracle_target if setting == "oracle_entities" else segment.target, constraints=tuple(constraints)))
    return replace(program, segments=tuple(segments))


def _difficulty(record: Mapping[str, object]) -> str:
    value = record.get("difficulty_level")
    return str(value) if value in {"easy", "hard"} else "other"


def _run_one(path: Path, setting: str, max_expansions: int) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    map_id = str(record["map_id"])
    instruction_id = str(record["instruction_id"])
    instruction_path = REPO_ROOT / str(record["instruction_file"])
    instruction = load_instruction(instruction_path)
    scene = SceneMap(load_map_state(map_id), instruction)
    source = _source_for_prediction(path)
    try:
        program = execute_grounding_code(source, scene, allow_helpers=True).grounded
    except Exception:
        if setting != "oracle_all_grounding":
            raise
        program, _annotation_dump = _grounded_program_from_annotations(scene, instruction)
    transformed = _transform(program, scene, instruction, setting)
    planned = plan_grounded_program(scene, transformed, max_expansions=max_expansions)
    trajectory = planned.trajectory or [[row, col] for row, col in scene.start.cells]
    metrics = evaluate_prediction(trajectory, map_id=map_id, instruction_id=instruction_id)
    return {
        "map_id": map_id,
        "instruction_id": instruction_id,
        "difficulty_level": _difficulty(record),
        "metrics": metrics,
        "planner_status": planned.status,
        "planner_failure_reason": planned.failure_reason,
        "grounded_program": transformed.to_json(),
    }


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    return metric_block(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Grounding2Route Direct-CaP oracle decomposition; no LLM calls.")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-expansions", type=int, default=250000)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=None)
    args = parser.parse_args()
    paths = _prediction_paths(args.prediction_root)
    if args.output.exists():
        output = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        output = {"settings": {}}
    output["protocol"] = {"llm_calls": 0, "planner": "fixed_plan_grounded_program", "oracle_entities": "annotation-aligned goals plus matched route-constraint entities", "oracle_all_grounding": "annotation-derived complete GroundedTask (goals plus route constraints)"}
    output["episode_count"] = len(paths)
    for setting in (args.settings or SETTINGS):
        rows: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(_run_one, path, setting, args.max_expansions): path for path in paths}
            for completed, future in enumerate(as_completed(futures), start=1):
                path = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
                rows.append(row)
                print(f"[{setting} {completed}/{len(paths)}] {path.name} {'error' if 'error' in row else 'done'}", flush=True)
        valid = [row for row in rows if isinstance(row.get("metrics"), Mapping)]
        output["settings"][setting] = {
            "overall": _aggregate(valid),
            "easy": _aggregate([row for row in valid if row.get("difficulty_level") == "easy"]),
            "hard": _aggregate([row for row in valid if row.get("difficulty_level") == "hard"]),
            "records": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
