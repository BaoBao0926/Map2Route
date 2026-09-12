"""Canonical Direct-CaP route-constraint utilities.

This module is deliberately planner-free: it adapts annotations, compares a
serialized GroundedProgram to those annotations, and supports controlled
component replacement for later oracle experiments.
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from scripts.methods.groundplan.grounding.scene import SceneMap
from scripts.methods.groundplan.ir import GPRef, GroundedConstraint, GroundedProgram, GroundedSegment

SOFT_KINDS = {"prefer_near", "prefer_far", "prefer_relative", "prefer_path_shape"}
PATH_SHAPE_WAYPOINT_TYPES = {
    "circle_waypoint",
    "wall_waypoint",
    "relative_waypoint",
}
PREFERENCE_KIND = {
    "near_preference": "prefer_near",
    "far_preference": "prefer_far",
    "relative_preference": "prefer_relative",
    "geometric_path_preference": "prefer_path_shape",
    "path_shape_preference": "prefer_path_shape",
}


def apply_scope_ablation(program: GroundedProgram, mode: str = "full") -> GroundedProgram:
    """Return a planner-input ablation with all other grounding held fixed."""
    if mode not in {
        "full",
        "no_spatial_scope",
        "no_segment_scope",
        "no_scope",
        "no_soft_constraints",
    }:
        raise ValueError(f"Unsupported scope ablation: {mode!r}")
    if mode == "full":
        return program
    segments: list[GroundedSegment] = []
    for segment in program.segments:
        if mode == "no_soft_constraints":
            constraints: list[GroundedConstraint] = []
            for constraint in segment.constraints:
                if constraint.hardness == "soft" or constraint.kind in SOFT_KINDS:
                    continue
                if _is_generated_path_shape_sequence(constraint):
                    collapsed = _collapse_path_shape_sequence(constraint)
                    if collapsed is not None:
                        constraints.append(collapsed)
                    continue
                constraints.append(constraint)
            segments.append(replace(segment, constraints=tuple(constraints)))
            continue
        constraints = tuple(
            replace(
                constraint,
                spatial_scope=None if mode in {"no_spatial_scope", "no_scope"} else constraint.spatial_scope,
                segment_scope=() if mode in {"no_segment_scope", "no_scope"} else constraint.segment_scope,
            )
            for constraint in segment.constraints
        )
        segments.append(replace(segment, constraints=constraints))
    return replace(program, segments=tuple(segments))


def _is_generated_path_shape_sequence(constraint: GroundedConstraint) -> bool:
    """Identify hard waypoint chains that encode a soft geometric path shape."""

    return (
        constraint.kind == "require_visit_in_order"
        and bool(constraint.refs)
        and all(
            ref.construction.get("type") in PATH_SHAPE_WAYPOINT_TYPES
            for ref in constraint.refs
        )
    )


def _collapse_path_shape_sequence(
    constraint: GroundedConstraint,
) -> GroundedConstraint | None:
    """Keep one hard visit anchor without retaining the ordered soft shape."""

    cells = frozenset(cell for ref in constraint.refs for cell in ref.cells)
    if not cells:
        return None
    centers = [ref.center for ref in constraint.refs if ref.center is not None]
    center = (
        (
            sum(point[0] for point in centers) / len(centers),
            sum(point[1] for point in centers) / len(centers),
        )
        if centers
        else None
    )
    room_ids = {ref.room_id for ref in constraint.refs if ref.room_id is not None}
    waypoint_types = sorted(
        {str(ref.construction.get("type")) for ref in constraint.refs}
    )
    anchor = GPRef(
        kind="region",
        id=f"no_soft_anchor_{constraint.constraint_id or 'path_shape'}",
        category="path_anchor",
        room_id=next(iter(room_ids)) if len(room_ids) == 1 else None,
        center=center,
        cells=cells,
        construction={
            "type": "no_soft_path_anchor",
            "source_waypoint_types": waypoint_types,
            "source_waypoint_count": len(constraint.refs),
        },
    )
    return replace(
        constraint,
        kind="require_visit",
        refs=(anchor,),
        argument_roles=("region",),
    )


def apply_oracle_grounding(
    predicted: GroundedProgram,
    oracle: GroundedProgram,
    mode: str = "pred_all",
) -> GroundedProgram:
    """Replace selected semantic components by same-ID oracle constraints.

    The fixed planner receives the returned GroundedProgram; no planner signal
    is used for matching or replacement.
    """
    allowed = {"pred_all", "oracle_entities", "oracle_relations", "oracle_scope", "oracle_all_grounding", "oracle_segment_scope", "oracle_spatial_scope", "oracle_kind"}
    if mode not in allowed:
        raise ValueError(f"Unsupported oracle grounding mode: {mode!r}")
    if mode == "pred_all":
        return predicted
    oracle_by_id = {constraint.constraint_id: constraint for segment in oracle.segments for constraint in segment.constraints}
    segments: list[GroundedSegment] = []
    for segment in predicted.segments:
        transformed: list[GroundedConstraint] = []
        for constraint in segment.constraints:
            reference = oracle_by_id.get(constraint.constraint_id)
            if reference is None:
                transformed.append(constraint)
                continue
            kwargs: dict[str, object] = {}
            if mode in {"oracle_entities", "oracle_all_grounding"}:
                kwargs["refs"] = reference.refs
                kwargs["argument_roles"] = reference.argument_roles
            if mode in {"oracle_relations", "oracle_all_grounding"}:
                kwargs["relation"] = reference.relation
            if mode in {"oracle_scope", "oracle_all_grounding", "oracle_segment_scope"}:
                kwargs["segment_scope"] = reference.segment_scope
            if mode in {"oracle_scope", "oracle_all_grounding", "oracle_spatial_scope"}:
                kwargs["spatial_scope"] = reference.spatial_scope
            if mode in {"oracle_kind", "oracle_all_grounding"}:
                kwargs["kind"] = reference.kind
                kwargs["hardness"] = reference.hardness
            transformed.append(replace(constraint, **kwargs))
        segments.append(replace(segment, constraints=tuple(transformed)))
    return replace(predicted, segments=tuple(segments))


def _ref_json(ref: GPRef) -> dict[str, object]:
    return ref.to_json()


def _construction(ref: Mapping[str, object]) -> dict[str, object] | None:
    raw = ref.get("construction")
    return dict(raw) if isinstance(raw, Mapping) else None


def canonicalize_constraint(constraint: Mapping[str, object]) -> dict[str, object]:
    refs = [dict(item) for item in constraint.get("refs", []) if isinstance(item, Mapping)]
    constructions = [_construction(ref) for ref in refs]
    constructions = [item for item in constructions if item is not None]
    scope = constraint.get("scope")
    scope_map = scope if isinstance(scope, Mapping) else {}
    spatial = constraint.get("spatial_scope", scope_map.get("spatial_scope"))
    segment_scope = constraint.get("segment_scope", scope_map.get("segment_scope", []))
    return {
        "constraint_id": str(constraint.get("constraint_id") or ""),
        "kind": constraint.get("kind"),
        "hardness": constraint.get("hardness"),
        "relation": constraint.get("relation"),
        "refs": refs,
        "ref_ids": tuple(str(ref.get("id")) for ref in refs),
        "argument_roles": tuple(str(item) for item in constraint.get("argument_roles", []) if isinstance(item, str)),
        "segment_scope": tuple(str(item) for item in segment_scope if isinstance(item, str)) if isinstance(segment_scope, Sequence) and not isinstance(segment_scope, str) else (),
        "spatial_scope": _scope_signature(spatial),
        "construction": tuple(_construction_signature(item) for item in constructions),
    }


def _scope_signature(scope: object) -> object:
    if isinstance(scope, Mapping):
        return (scope.get("kind"), scope.get("id"), _construction_signature(_construction(scope)))
    if isinstance(scope, Sequence) and not isinstance(scope, (str, bytes)):
        return tuple(_scope_signature(item) for item in scope)
    return None


def _construction_signature(construction: Mapping[str, object] | None) -> object:
    if not construction:
        return None
    references = construction.get("references")
    if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
        references = tuple(sorted(str(item) for item in references))
    return (construction.get("type"), references, construction.get("reference"), construction.get("entity_id"), construction.get("room_id"))


def canonical_ground_truth_constraints(instruction: Mapping[str, object], scene: SceneMap) -> list[dict[str, object]]:
    """Adapt existing annotations without changing their source files.

    Goal must-pass regions are actions/targets, not route constraints here.  We
    adapt must-avoid and soft route constraints, retaining available object IDs
    and route-order scope.
    """
    output: list[dict[str, object]] = []
    for index, raw in enumerate(instruction.get("hard_constraints", [])):
        if not isinstance(raw, Mapping) or raw.get("kind") != "must_avoid":
            continue
        output.append(_annotation_constraint(raw, scene, index, kind="forbid", hardness="hard"))
    for index, raw in enumerate(instruction.get("soft_constraints", [])):
        if not isinstance(raw, Mapping):
            continue
        preference = raw.get("preference_type")
        kind = PREFERENCE_KIND.get(str(preference))
        if kind is None or kind == "clearance":
            continue
        output.append(_annotation_constraint(raw, scene, index, kind=kind, hardness="soft"))
    return output


def _annotation_constraint(raw: Mapping[str, object], scene: SceneMap, index: int, *, kind: str, hardness: str) -> dict[str, object]:
    object_ids: list[object] = []
    reference = raw.get("reference_region")
    if isinstance(reference, Mapping) and reference.get("object_id") is not None:
        object_ids.append(reference["object_id"])
    values = raw.get("object_ids")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        object_ids.extend(values)
    if raw.get("object_id") is not None:
        object_ids.append(raw["object_id"])
    refs: list[dict[str, object]] = []
    for value in object_ids:
        candidate = scene.entities.get(f"object_{value}")
        if candidate is not None:
            refs.append(_ref_json(candidate))
    scope = raw.get("scope")
    segment_scope: tuple[str, ...] = ()
    if isinstance(scope, Mapping) and scope.get("type") == "between_hard_constraints":
        to_order = scope.get("to_order")
        if isinstance(to_order, int) and to_order > 1:
            segment_scope = (f"s{to_order - 1}",)
    relation = None
    if kind == "prefer_relative":
        relation = "closer_to"
    return {
        "constraint_id": str(raw.get("constraint_id") or f"gt_c{index}"),
        "kind": kind,
        "hardness": hardness,
        "relation": relation,
        "refs": refs,
        "argument_roles": ("closer_ref", "farther_ref") if kind == "prefer_relative" else tuple("reference" for _ in refs),
        "segment_scope": segment_scope,
        "spatial_scope": None,
        "source_text": raw.get("source_text"),
    }


def evaluate_constraint_grounding(predicted: GroundedProgram | Mapping[str, object] | None, instruction: Mapping[str, object], scene: SceneMap) -> dict[str, object]:
    gt_raw = canonical_ground_truth_constraints(instruction, scene)
    if isinstance(predicted, GroundedProgram):
        pred_raw = [constraint.to_json() for segment in predicted.segments for constraint in segment.constraints]
    elif isinstance(predicted, Mapping):
        pred_raw = [constraint for segment in predicted.get("segments", []) if isinstance(segment, Mapping) for constraint in segment.get("constraints", []) if isinstance(constraint, Mapping)]
    else:
        pred_raw = []
    gt = [canonicalize_constraint(item) for item in gt_raw]
    pred = [canonicalize_constraint(item) for item in pred_raw]
    unmatched = set(range(len(pred)))
    matched: list[tuple[dict[str, object], dict[str, object]]] = []
    missing: list[dict[str, object]] = []
    for target in gt:
        candidates = [idx for idx in unmatched if pred[idx]["hardness"] == target["hardness"]]
        if not candidates:
            missing.append(target)
            continue
        best = min(candidates, key=lambda idx: _match_cost(target, pred[idx]))
        # A different family is still a useful wrong-kind match only when scope
        # or arguments offer evidence; otherwise retain it as an extra/missing.
        if _match_cost(target, pred[best]) >= 12:
            missing.append(target)
            continue
        unmatched.remove(best)
        matched.append((target, pred[best]))
    extras = [pred[index] for index in sorted(unmatched)]
    component_names = ("kind", "relation", "ref_ids", "argument_roles", "segment_scope", "spatial_scope", "hardness")
    correct = {name: 0 for name in component_names}
    errors: list[dict[str, object]] = [{"error": "missing_constraint", "gt": item} for item in missing]
    errors.extend({"error": "extra_constraint", "pred": item} for item in extras)
    full = 0
    for target, observed in matched:
        differences = []
        labels = {"kind": "wrong_kind", "relation": "wrong_relation", "ref_ids": "wrong_argument_entity", "argument_roles": "wrong_argument_role", "segment_scope": "wrong_segment_scope", "spatial_scope": "wrong_spatial_scope", "hardness": "wrong_hard_soft_type"}
        for name in component_names:
            if target[name] == observed[name]:
                correct[name] += 1
            else:
                differences.append(labels[name])
        if not differences:
            full += 1
        else:
            errors.append({"error": differences, "gt": target, "pred": observed})
    precision = len(matched) / len(pred) if pred else (1.0 if not gt else 0.0)
    recall = len(matched) / len(gt) if gt else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = len(matched)
    accuracy = {name: (correct[name] / denominator if denominator else None) for name in component_names}
    return {
        "gt_constraint_count": len(gt), "pred_constraint_count": len(pred), "matched_constraint_count": len(matched),
        "constraint_precision": precision, "constraint_recall": recall, "constraint_f1": f1,
        "type_accuracy": accuracy["kind"], "relation_accuracy": accuracy["relation"], "argument_accuracy": accuracy["ref_ids"],
        "argument_role_accuracy": accuracy["argument_roles"], "segment_scope_accuracy": accuracy["segment_scope"],
        "spatial_scope_accuracy": accuracy["spatial_scope"],
        "full_scope_accuracy": ((sum(1 for a,b in matched if a["segment_scope"] == b["segment_scope"] and a["spatial_scope"] == b["spatial_scope"]) / denominator) if denominator else None),
        "hard_soft_accuracy": accuracy["hardness"], "full_constraint_accuracy": (full / denominator if denominator else None),
        "program_exact_grounding": bool(not missing and not extras and len(matched) == len(gt) and full == len(gt)),
        "errors": errors,
    }


def _match_cost(target: Mapping[str, object], observed: Mapping[str, object]) -> int:
    return (10 if target["hardness"] != observed["hardness"] else 0) + (5 if target["kind"] != observed["kind"] else 0) + (2 if target["segment_scope"] != observed["segment_scope"] else 0) + (1 if target["ref_ids"] != observed["ref_ids"] else 0)


def write_grounding_report(reports: Sequence[Mapping[str, object]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "grounding_metrics.json"
    errors_path = output_dir / "grounding_errors.jsonl"
    summary_path = output_dir / "grounding_summary.csv"
    aggregate_keys = ("constraint_precision", "constraint_recall", "constraint_f1", "type_accuracy", "argument_accuracy", "relation_accuracy", "segment_scope_accuracy", "spatial_scope_accuracy", "full_scope_accuracy", "hard_soft_accuracy", "full_constraint_accuracy")
    averages = {key: _mean([item.get(key) for item in reports]) for key in aggregate_keys}
    metrics_path.write_text(json.dumps({"episode_count": len(reports), "metrics": averages}, indent=2), encoding="utf-8")
    with errors_path.open("w", encoding="utf-8") as handle:
        for episode_index, report in enumerate(reports):
            for error in report.get("errors", []):
                handle.write(json.dumps({"episode_index": episode_index, **error}, ensure_ascii=False) + "\\n")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_index", *aggregate_keys, "program_exact_grounding"])
        writer.writeheader()
        for episode_index, report in enumerate(reports):
            writer.writerow({"episode_index": episode_index, **{key: report.get(key) for key in aggregate_keys}, "program_exact_grounding": report.get("program_exact_grounding")})
    return {"grounding_metrics": metrics_path, "grounding_errors": errors_path, "grounding_summary": summary_path}


def _mean(values: Sequence[object]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(numbers) / len(numbers) if numbers else None
