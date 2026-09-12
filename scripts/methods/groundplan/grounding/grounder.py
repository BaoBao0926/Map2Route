"""Deterministic DSL grounding for GroundPlan."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from scripts.methods.groundplan.alias_utils import merged_entity_aliases, merged_room_aliases
from scripts.methods.groundplan.config import (
    CIRCLE_OBJECT_WAYPOINT_CLEARANCE_CELLS,
    CIRCLE_OBJECT_WAYPOINT_CLEARANCE_WEIGHT,
    CIRCLE_ROOM_WAYPOINT_CLEARANCE_CELLS,
    CIRCLE_ROOM_WAYPOINT_CLEARANCE_WEIGHT,
    CIRCLE_WAYPOINT_SEARCH_RADIUS_CELLS,
    ENTITY_ALIASES,
    NEAR_RADIUS_METERS,
    OBJECT_GOAL_RADIUS_METERS,
    ROOM_ALIASES,
)
from scripts.methods.groundplan.grounding.scene import SceneMap, canonical_category
from scripts.methods.util.grid_astar import is_traversable
from scripts.methods.groundplan.ir import (
    GPConstraintSpec,
    GPExpr,
    GPProgramSpec,
    GPRef,
    GroundedConstraint,
    GroundedProgram,
    GroundedSegment,
)


class GroundingFailure(ValueError):
    pass


def ground_program(program: GPProgramSpec, scene: SceneMap) -> GroundedProgram:
    bindings: dict[str, object] = {"task_start": scene.start, "start_position": scene.start}
    diagnostics: list[str] = []
    try:
        for binding in program.bindings:
            try:
                bindings[binding.name] = _eval_expr(binding.expr, scene, bindings)
            except GroundingFailure as exc:
                raise GroundingFailure(f"BINDING_FAILED: {binding.name}: {exc}") from exc

        grounded_segments: list[GroundedSegment] = []
        segment_targets: dict[str, GPRef] = {}
        for segment in program.segments:
            try:
                bindings.setdefault("segment_start", _eval_expr(segment.start, scene, bindings))
                start_ref = _require_ref(_eval_expr(segment.start, scene, bindings), "segment start")
                target_ref = _require_ref(_eval_expr(segment.target, scene, bindings), "segment target")
            except GroundingFailure as exc:
                raise GroundingFailure(f"SEGMENT_FAILED: {segment.id}: {exc}") from exc
            bindings[segment.id] = target_ref
            segment_targets[segment.id] = target_ref
            constraints: list[GroundedConstraint] = []
            for constraint in segment.constraints:
                try:
                    # Preserve the pre-existing IR/DSL default where an omitted
                    # scope is local to its declaration segment. Direct-CaP has
                    # already resolved its explicit global () before calling the
                    # shared grounding primitive below.
                    legacy_constraint = replace(
                        constraint,
                        segment_scope=constraint.segment_scope or (segment.id,),
                    )
                    grounded_constraints = _ground_constraint(legacy_constraint, scene, bindings, segment.id)
                    constraints.extend(
                        _finalize_constraint(
                            item,
                            segment_id=segment.id,
                            index=len(constraints) + offset,
                        )
                        for offset, item in enumerate(grounded_constraints)
                    )
                except GroundingFailure as exc:
                    raise GroundingFailure(
                        f"CONSTRAINT_FAILED: {segment.id}: {constraint.source_text or constraint.kind}: {exc}"
                    ) from exc
            grounded_segments.append(
                GroundedSegment(
                    id=segment.id,
                    start_reference=start_ref,
                    target=target_ref,
                    constraints=tuple(constraints),
                )
            )
            bindings.pop("segment_start", None)

        if not grounded_segments:
            raise GroundingFailure("INVALID_SEGMENT_REFERENCE: no segments.")
        return GroundedProgram(
            segments=tuple(grounded_segments),
            bindings=bindings,
            diagnostics=tuple(diagnostics),
        )
    except Exception as exc:
        reason = str(exc)
        return GroundedProgram(
            segments=(),
            bindings=bindings,
            status="failed",
            failure_reason=reason,
            diagnostics=tuple(diagnostics),
        )


def _eval_expr(expr: object, scene: SceneMap, env: Mapping[str, object]) -> object:
    if isinstance(expr, GPRef):
        return expr
    if isinstance(expr, str):
        if expr in env:
            return env[expr]
        # String literals are also used for categories and relation names.
        return expr
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, (list, tuple)):
        return [_eval_expr(item, scene, env) for item in expr]
    if not isinstance(expr, GPExpr):
        return expr

    op = expr.op
    if op == "where":
        if len(expr.args) != 2:
            raise GroundingFailure("INVALID_ARGUMENT: where requires candidates and predicate.")
        refs = _require_ref_list(_eval_expr(expr.args[0], scene, env), "where")
        return [ref for ref in refs if _eval_predicate(expr.args[1], ref, scene, env)]
    if op == "compare":
        if len(expr.args) != 3:
            raise GroundingFailure("INVALID_ARGUMENT: compare requires left, operator, right.")
        left = _eval_expr(expr.args[0], scene, env)
        operator = str(expr.args[1])
        right = _eval_expr(expr.args[2], scene, env)
        return _compare_values(left, operator, right)
    if op == "and":
        return all(bool(_eval_expr(arg, scene, env)) for arg in expr.args)
    if op == "or":
        return any(bool(_eval_expr(arg, scene, env)) for arg in expr.args)
    if op == "not":
        if len(expr.args) != 1:
            raise GroundingFailure("INVALID_ARGUMENT: not requires one predicate.")
        return not bool(_eval_expr(expr.args[0], scene, env))

    args = [_eval_expr(arg, scene, env) for arg in expr.args]
    kwargs = {key: _eval_expr(value, scene, env) for key, value in expr.kwargs.items()}

    if op == "target_of":
        if not args:
            raise GroundingFailure("INVALID_ARGUMENT: target_of requires a segment reference.")
        return _require_ref(args[0], "target_of")
    if op == "entities":
        category = str(args[0]) if args else None
        if category:
            normalized = canonical_category(category, aliases=merged_entity_aliases(ENTITY_ALIASES))
            known_categories = {ref.category for ref in scene.entities.values()}
            if normalized not in known_categories:
                raise GroundingFailure(
                    f"UNKNOWN_ENTITY_CATEGORY: {category} is not in the current object category inventory."
                )
        return scene.refs_by_category("entity", category)
    if op == "rooms":
        category = str(args[0]) if args else None
        if category:
            normalized = canonical_category(category, aliases=merged_room_aliases(ROOM_ALIASES))
            known_categories = {ref.category for ref in scene.rooms.values()}
            if normalized not in known_categories:
                raise GroundingFailure(
                    f"UNKNOWN_ROOM_CATEGORY: {category} is not in the current room category inventory."
                )
        return scene.refs_by_category("room", category)
    if op == "in":
        return _entities_in(args[0], args[1], scene)
    if op == "room_of":
        return scene.room_of(_require_ref(args[0], "room_of"))
    if op == "contains":
        room = _require_ref(args[0], "contains")
        if room.kind != "room":
            raise GroundingFailure("INVALID_TYPE: contains requires a room.")
        category = canonical_category(args[1], aliases=merged_entity_aliases(ENTITY_ALIASES)) if len(args) > 1 else ""
        return any(entity.room_id == room.id and entity.category == category for entity in scene.entities.values())
    if op == "count_next_to":
        category = canonical_category(args[0], aliases=merged_entity_aliases(ENTITY_ALIASES)) if args else ""
        reference = _require_ref(args[1] if len(args) > 1 else kwargs.get("reference"), "count_next_to reference")
        threshold = float(kwargs.get("threshold", _next_to_predicate_radius(scene)))
        return sum(
            1
            for entity in scene.entities.values()
            if entity.id != reference.id
            and entity.category == category
            and scene.distance(entity, reference, "euclidean") <= threshold
        )
    if op == "count_near":
        category = canonical_category(args[0], aliases=merged_entity_aliases(ENTITY_ALIASES)) if args else ""
        reference = _require_ref(args[1] if len(args) > 1 else kwargs.get("reference"), "count_near reference")
        radius = float(kwargs.get("radius", _near_predicate_radius(scene)))
        return sum(
            1
            for entity in scene.entities.values()
            if entity.id != reference.id
            and entity.category == category
            and scene.distance(entity, reference, "euclidean") <= radius
        )
    if op == "adjacent_rooms":
        room = _require_ref(args[0], "adjacent_rooms")
        if room.kind != "room":
            raise GroundingFailure("INVALID_TYPE: adjacent_rooms requires a room.")
        room_ids = {
            other
            for pair in scene.passages
            if room.id in pair
            for other in pair
            if other != room.id
        }
        return [scene.rooms[room_id] for room_id in sorted(room_ids) if room_id in scene.rooms]
    if op == "unique":
        refs = _require_ref_list(args[0], "unique")
        if not refs:
            raise GroundingFailure("EMPTY_CANDIDATE: unique received no candidates.")
        if len(refs) != 1:
            raise GroundingFailure(f"AMBIGUOUS_CANDIDATE: unique expected 1 candidate, got {len(refs)}.")
        return refs[0]
    if op == "choose_any":
        refs = _require_ref_list(args[0], "choose_any")
        reference = _require_ref(kwargs.get("reference", env.get("segment_start", scene.start)), "choose_any reference")
        return _select_by_distance(refs, reference, scene, reverse=False, op_name="choose_any")
    if op == "kth_nearest":
        return _select_kth(args, kwargs, scene, reverse=False, op_name="kth_nearest")
    if op == "kth_farthest":
        return _select_kth(args, kwargs, scene, reverse=True, op_name="kth_farthest")
    if op == "kth_largest":
        return _select_by_size(_require_ref_list(args[0], op), int(kwargs.get("k", args[2] if len(args) > 2 else 1)), reverse=True)
    if op == "kth_smallest":
        return _select_by_size(_require_ref_list(args[0], op), int(kwargs.get("k", args[2] if len(args) > 2 else 1)), reverse=False)
    if op == "order_by_distance":
        refs = _require_ref_list(args[0], op)
        reference = _require_ref(args[1], op)
        reverse = str(args[2] if len(args) > 2 else kwargs.get("order", "ascending")) == "descending"
        return sorted(refs, key=lambda ref: (scene.distance(ref, reference), ref.id), reverse=reverse)
    if op == "union":
        values: list[GPRef] = []
        for arg in args:
            values.extend(_require_ref_list(arg, "union"))
        return _dedupe(values)
    if op == "intersection":
        sets = [set(ref.id for ref in _require_ref_list(arg, "intersection")) for arg in args]
        if not sets:
            return []
        keep = set.intersection(*sets)
        all_refs = [ref for arg in args for ref in _require_ref_list(arg, "intersection")]
        return [ref for ref in _dedupe(all_refs) if ref.id in keep]
    if op == "exclude":
        base = _require_ref_list(args[0], "exclude")
        excluded = {ref.id for ref in _require_ref_list(args[1], "exclude")}
        return [ref for ref in base if ref.id not in excluded]
    if op == "count":
        return len(_require_ref_list(args[0], "count"))
    if op == "region_of":
        return _region_of_entity(_require_ref(args[0], "region_of"), scene)
    if op == "room_region":
        return _region_of_room(_require_ref(args[0], "room_region"), scene)
    if op == "midpoint_region":
        return _midpoint_region(_require_ref(args[0], op), _require_ref(args[1], op), scene)
    if op == "between_region":
        return _between_region(_require_ref(args[0], op), _require_ref(args[1], op), scene)
    if op == "near_region":
        radius = float(args[1] if len(args) > 1 else kwargs.get("radius", OBJECT_GOAL_RADIUS_METERS))
        return _near_region(_require_ref(args[0], op), radius, scene)
    if op == "boundary_region":
        return _boundary_region(_require_ref(args[0], op), scene)
    if op == "side_region":
        return _side_region(_require_ref(args[0], op), _require_ref(args[1], op), scene)
    if op == "half_room":
        return _half_room(_require_ref(args[0], op), _require_ref(args[1], op), scene)
    if op == "relative_waypoint":
        if len(args) < 2:
            raise GroundingFailure("INVALID_ARGUMENT: relative_waypoint requires two anchor references.")
        first = _require_ref(args[0], op)
        second = _require_ref(args[1], op)
        along = float(kwargs.get("along", args[2] if len(args) > 2 else 0.5))
        lateral = float(kwargs.get("lateral", args[3] if len(args) > 3 else 0.0))
        radius = float(kwargs.get("radius", 0.20))
        room_value = kwargs.get("room")
        room = _require_ref(room_value, op) if room_value is not None else None
        return _relative_waypoint(first, second, along, lateral, radius, scene, room=room)
    if op == "circle":
        reference = _require_ref(args[0], op)
        fraction = float(kwargs.get("fraction", args[1] if len(args) > 1 else 1.0))
        direction = str(kwargs.get("direction", args[2] if len(args) > 2 else "counterclockwise"))
        start_toward = kwargs.get("start_toward")
        start_reference = env.get("segment_start")
        return _circle_regions(
            reference,
            fraction,
            direction,
            scene,
            start_toward=start_toward if isinstance(start_toward, GPRef) else None,
            start_reference=start_reference if isinstance(start_reference, GPRef) else None,
        )
    if op == "follow_wall":
        reference = _require_ref(args[0], op)
        fraction = float(kwargs.get("fraction", args[1] if len(args) > 1 else 1.0))
        direction = str(kwargs.get("direction", args[2] if len(args) > 2 else "auto"))
        first_toward = kwargs.get(
            "first_toward",
            kwargs.get("pass_by", kwargs.get("start_toward", args[3] if len(args) > 3 else None)),
        )
        start_reference = env.get("segment_start")
        return _follow_wall_regions(
            reference,
            fraction,
            direction,
            scene,
            first_toward=first_toward if isinstance(first_toward, GPRef) else None,
            start_reference=start_reference if isinstance(start_reference, GPRef) else None,
        )
    if op == "passage_regions":
        return scene.passage_regions(_require_ref(args[0], op), _require_ref(args[1], op))
    if op == "closest_pair_member":
        candidates = _require_ref_list(kwargs.get("candidates", args[0] if args else []), op)
        references = _require_ref_list(kwargs.get("references", args[1] if len(args) > 1 else []), op)
        if not candidates or not references:
            raise GroundingFailure("EMPTY_CANDIDATE: closest_pair_member received empty set.")
        return min(candidates, key=lambda cand: (min(scene.distance(cand, ref, "euclidean") for ref in references), cand.id))
    raise GroundingFailure(f"UNSUPPORTED_OPERATOR: {op}")


def _eval_predicate(
    predicate: object,
    candidate: GPRef,
    scene: SceneMap,
    env: Mapping[str, object],
) -> bool:
    local_env = {**env, "self": candidate}
    if isinstance(predicate, GPExpr):
        op = predicate.op
        if op in {"compare", "and", "or", "not"}:
            return bool(_eval_expr(predicate, scene, local_env))
        args = [_eval_expr(arg, scene, local_env) for arg in predicate.args]
        if op == "near_to":
            reference = _require_ref(args[0], "near_to")
            return scene.distance(candidate, reference, "euclidean") <= _near_predicate_radius(scene)
        if op == "far_from":
            reference = _require_ref(args[0], "far_from")
            return scene.distance(candidate, reference, "euclidean") > _near_predicate_radius(scene)
        if op == "next_to":
            reference = _require_ref(args[0], "next_to")
            return scene.distance(candidate, reference, "euclidean") <= _next_to_predicate_radius(scene)
        if op == "on_top_of":
            reference = _require_ref(args[0], "on_top_of")
            return _on_top_of(candidate, reference)
        if op == "in_corner":
            room = _require_ref(args[0], "in_corner") if args else scene.room_of(candidate)
            return _in_corner(candidate, room)
        if op == "between":
            first = _require_ref(args[0], "between")
            second = _require_ref(args[1], "between")
            return _between_refs(candidate, first, second, scene)
    return bool(_eval_expr(predicate, scene, local_env))


def _near_predicate_radius(scene: SceneMap) -> float:
    return max(2.0, NEAR_RADIUS_METERS / scene.resolution)


def _next_to_predicate_radius(scene: SceneMap) -> float:
    return max(2.0, 0.75 / scene.resolution)


def _compare_values(left: object, operator: str, right: object) -> bool:
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator in {">", ">=", "<", "<="}:
        try:
            left_value = float(left)  # type: ignore[arg-type]
            right_value = float(right)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise GroundingFailure("INVALID_COMPARISON: ordered comparison requires numeric operands.") from exc
        if operator == ">":
            return left_value > right_value
        if operator == ">=":
            return left_value >= right_value
        if operator == "<":
            return left_value < right_value
        return left_value <= right_value
    raise GroundingFailure(f"INVALID_COMPARISON: unsupported operator {operator!r}.")


def _on_top_of(candidate: GPRef, reference: GPRef) -> bool:
    if candidate.center is None or reference.center is None:
        return False
    candidate_row, candidate_col = candidate.center
    reference_row, reference_col = reference.center
    return abs(candidate_col - reference_col) <= 8 and candidate_row <= reference_row


def _in_corner(candidate: GPRef, room: GPRef) -> bool:
    if room.kind != "room" or candidate.center is None or not room.cells:
        return False
    rows = [row for row, _col in room.cells]
    cols = [col for _row, col in room.cells]
    corners = (
        (min(rows), min(cols)),
        (min(rows), max(cols)),
        (max(rows), min(cols)),
        (max(rows), max(cols)),
    )
    return min(math.hypot(candidate.center[0] - row, candidate.center[1] - col) for row, col in corners) <= 20


def _between_refs(candidate: GPRef, first: GPRef, second: GPRef, scene: SceneMap) -> bool:
    if candidate.center is None or first.center is None or second.center is None:
        return False
    row, col = candidate.center
    row1, col1 = first.center
    row2, col2 = second.center
    line_length = math.hypot(row2 - row1, col2 - col1)
    if line_length == 0:
        return False
    projection = ((row - row1) * (row2 - row1) + (col - col1) * (col2 - col1)) / (line_length**2)
    if projection < 0.0 or projection > 1.0:
        return False
    nearest = (row1 + projection * (row2 - row1), col1 + projection * (col2 - col1))
    return math.hypot(row - nearest[0], col - nearest[1]) <= max(3.0, 0.5 / scene.resolution)


def _ground_constraint(
    constraint: GPConstraintSpec,
    scene: SceneMap,
    env: Mapping[str, object],
    segment_id: str,
) -> tuple[GroundedConstraint, ...]:
    # ``constraint()`` resolves None to its current host segment before this
    # point. Preserve () here: it is the explicit global-scope representation.
    scope = constraint.segment_scope
    if constraint.kind in {"prefer_near", "prefer_far"}:
        pref_refs = _require_ref_list(
            _eval_expr(constraint.exprs[0], scene, env),
            constraint.kind,
        )
        if not pref_refs:
            raise GroundingFailure(f"EMPTY_CANDIDATE: {constraint.kind} received no references.")
        explicit_spatial_scope = _ground_spatial_scope(constraint.spatial_scope, scene, env)
        grounded_preferences: list[GroundedConstraint] = []
        for pref_ref in pref_refs:
            spatial_scope = explicit_spatial_scope
            if spatial_scope is None and pref_ref.kind == "entity":
                spatial_scope = scene.room_of(pref_ref)
            grounded_preferences.append(
                GroundedConstraint(
                    kind=constraint.kind,
                    refs=(pref_ref,),
                    segment_scope=scope,
                    spatial_scope=spatial_scope,
                    argument_roles=("reference",),
                    hardness="soft",
                    source_text=constraint.source_text,
                )
            )
        return tuple(grounded_preferences)
    if constraint.kind == "prefer_relative":
        first = _require_ref(_eval_expr(constraint.exprs[0], scene, env), constraint.kind)
        relation = _canonical_relative_relation(
            str(_eval_expr(constraint.exprs[1], scene, env)) if len(constraint.exprs) >= 3 else "closer_to"
        )
        second = _require_ref(_eval_expr(constraint.exprs[2], scene, env), constraint.kind) if len(constraint.exprs) >= 3 else _require_ref(_eval_expr(constraint.exprs[1], scene, env), constraint.kind)
        first_room = scene.room_of(first)
        second_room = scene.room_of(second)
        if first_room.id != second_room.id:
            raise GroundingFailure("INVALID_SCOPE: relative preference references are in different rooms.")
        spatial_scope = _ground_spatial_scope(constraint.spatial_scope, scene, env) or first_room
        return (
            GroundedConstraint(
                kind="prefer_relative",
                refs=(first, second),
                relation=relation,
                segment_scope=scope,
                spatial_scope=spatial_scope,
                argument_roles=(
                    ("farther_ref", "closer_ref")
                    if relation == "farther_from"
                    else ("closer_ref", "farther_ref")
                ),
                hardness="soft",
                source_text=constraint.source_text,
            ),
        )
    if constraint.kind == "prefer_path_shape":
        refs = _ground_hard_constraint_refs(constraint.exprs, scene, env)
        if not refs:
            raise GroundingFailure("EMPTY_CANDIDATE: prefer_path_shape received no references.")
        return (
            GroundedConstraint(
                kind="prefer_path_shape",
                refs=refs,
                segment_scope=scope,
                spatial_scope=_ground_spatial_scope(constraint.spatial_scope, scene, env),
                argument_roles=tuple("waypoint" for _ in refs),
                hardness="soft",
                source_text=constraint.source_text,
            ),
        )
    refs = _ground_hard_constraint_refs(constraint.exprs, scene, env)
    return (
        GroundedConstraint(
            kind=constraint.kind,
            refs=refs,
            segment_scope=scope,
            argument_roles=tuple("region" for _ in refs),
            hardness="hard",
            source_text=constraint.source_text,
        ),
    )


def _finalize_constraint(
    constraint: GroundedConstraint,
    *,
    segment_id: str,
    index: int,
) -> GroundedConstraint:
    """Assign Direct-CaP/DSL-stable identity after any one-to-many expansion."""
    hardness = constraint.hardness or (
        "soft" if constraint.kind.startswith("prefer_") else "hard"
    )
    roles = constraint.argument_roles or tuple("reference" for _ in constraint.refs)
    return replace(
        constraint,
        constraint_id=constraint.constraint_id or f"{segment_id}_c{index}",
        # Empty scope is global; do not rewrite it to the declaration host.
        segment_scope=constraint.segment_scope,
        argument_roles=roles,
        hardness=hardness,  # type: ignore[arg-type]
    )


def _canonical_relative_relation(relation: str) -> str:
    normalized = relation.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "farther",
        "further",
        "farther_than",
        "further_than",
        "farther_from",
        "further_from",
        "away_from",
        "more_far",
        "more_distant",
        ">",
    }:
        return "farther_from"
    if normalized in {
        "closer",
        "nearer",
        "closer_than",
        "nearer_than",
        "closer_to",
        "nearer_to",
        "near",
        "<",
    }:
        return "closer_to"
    return normalized


def _ground_hard_constraint_refs(
    exprs: Sequence[object],
    scene: SceneMap,
    env: Mapping[str, object],
) -> tuple[GPRef, ...]:
    refs: list[GPRef] = []
    for expr in exprs:
        value = _eval_expr(expr, scene, env)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, GPRef)):
            refs.extend(_ensure_region_if_hard(ref, scene) for ref in _require_ref_list(value, "hard constraint"))
        else:
            refs.append(_ensure_region_if_hard(value, scene))
    return tuple(refs)


def _ground_spatial_scope(scope_expr: object | None, scene: SceneMap, env: Mapping[str, object]) -> GPRef | tuple[GPRef, ...] | None:
    if scope_expr is None:
        return None
    value = _eval_expr(scope_expr, scene, env)
    if isinstance(value, GPRef):
        return _ensure_region_if_hard(value, scene)
    refs = tuple(_ensure_region_if_hard(ref, scene) for ref in _require_ref_list(value, "spatial scope"))
    return refs or None


def _ensure_region_if_hard(value: object, scene: SceneMap) -> GPRef:
    ref = _require_ref(value, "hard constraint")
    if ref.kind == "region":
        return ref
    if ref.kind == "room":
        return _region_of_room(ref, scene)
    if ref.kind == "entity":
        return _region_of_entity(ref, scene)
    return ref


def _entities_in(entity_set: object, room_or_rooms: object, scene: SceneMap) -> list[GPRef]:
    entities = _require_ref_list(entity_set, "in")
    rooms = _require_ref_list(room_or_rooms, "in")
    room_ids = {room.id for room in rooms}
    return [entity for entity in entities if entity.room_id in room_ids]


def _select_kth(args: Sequence[object], kwargs: Mapping[str, object], scene: SceneMap, *, reverse: bool, op_name: str) -> GPRef:
    refs = _require_ref_list(args[0], op_name)
    reference = _require_ref(args[1] if len(args) > 1 else kwargs.get("reference"), f"{op_name} reference")
    k = int(kwargs.get("k", args[2] if len(args) > 2 else 1))
    ordered = sorted(refs, key=lambda ref: (scene.distance(ref, reference, str(kwargs.get("metric", "geodesic"))), ref.id), reverse=reverse)
    if k < 1 or k > len(ordered):
        raise GroundingFailure(f"ORDINAL_OUT_OF_RANGE: {op_name} requested k={k}, candidate_count={len(ordered)}.")
    return ordered[k - 1]


def _select_by_distance(refs: Sequence[GPRef], reference: GPRef, scene: SceneMap, *, reverse: bool, op_name: str) -> GPRef:
    if not refs:
        raise GroundingFailure(f"EMPTY_CANDIDATE: {op_name} received no candidates.")
    return sorted(refs, key=lambda ref: (scene.distance(ref, reference), ref.id), reverse=reverse)[0]


def _select_by_size(refs: Sequence[GPRef], k: int, *, reverse: bool) -> GPRef:
    ordered = sorted(refs, key=lambda ref: (len(ref.cells), ref.id), reverse=reverse)
    if k < 1 or k > len(ordered):
        raise GroundingFailure(f"ORDINAL_OUT_OF_RANGE: size ranking requested k={k}, candidate_count={len(ordered)}.")
    return ordered[k - 1]


def _region_of_entity(ref: GPRef, scene: SceneMap) -> GPRef:
    if ref.kind != "entity":
        raise GroundingFailure("INVALID_TYPE: region_of requires entity.")
    return scene.region_from_cells(
        f"region_of_{ref.id}",
        set(ref.cells),
        {"type": "entity_region", "entity_id": ref.id},
    )


def _region_of_room(ref: GPRef, scene: SceneMap) -> GPRef:
    if ref.kind != "room":
        raise GroundingFailure("INVALID_TYPE: room_region requires room.")
    return scene.region_from_cells(
        f"region_of_{ref.id}",
        set(ref.cells),
        {"type": "room_region", "room_id": ref.id},
    )


def _midpoint_region(first: GPRef, second: GPRef, scene: SceneMap) -> GPRef:
    if first.center is None or second.center is None:
        raise GroundingFailure("INVALID_REGION: midpoint references need centers.")
    center = ((first.center[0] + second.center[0]) / 2, (first.center[1] + second.center[1]) / 2)
    cells = _disk_cells(center, max(2, round(0.3 / scene.resolution)), scene.grid_size)
    return scene.region_from_cells(
        f"midpoint_{first.id}_{second.id}",
        cells,
        {"type": "midpoint", "references": [first.id, second.id]},
    )


def _between_region(first: GPRef, second: GPRef, scene: SceneMap) -> GPRef:
    if first.center is None or second.center is None:
        raise GroundingFailure("INVALID_REGION: between references need centers.")
    cells: set[tuple[int, int]] = set()
    steps = max(2, int(math.hypot(first.center[0] - second.center[0], first.center[1] - second.center[1])))
    radius = max(2, round(0.25 / scene.resolution))
    for index in range(steps + 1):
        ratio = index / steps
        center = (
            first.center[0] * (1 - ratio) + second.center[0] * ratio,
            first.center[1] * (1 - ratio) + second.center[1] * ratio,
        )
        cells.update(_disk_cells(center, radius, scene.grid_size))
    return scene.region_from_cells(
        f"between_{first.id}_{second.id}",
        cells,
        {"type": "between", "references": [first.id, second.id]},
    )


def _near_region(ref: GPRef, radius: float, scene: SceneMap) -> GPRef:
    if ref.center is None:
        raise GroundingFailure("INVALID_REGION: near_region reference needs center.")
    radius_cells = max(1, round(radius / scene.resolution if radius < 10 else radius))
    cells = _disk_cells(ref.center, radius_cells, scene.grid_size)
    if ref.room_id and ref.room_id in scene.rooms:
        room_cells = scene.rooms[ref.room_id].cells
        cells = {cell for cell in cells if cell in room_cells}
    return scene.region_from_cells(
        f"near_{ref.id}_{radius_cells}",
        cells,
        {
            "type": "near",
            "reference": ref.id,
            "radius_cells": radius_cells,
            "radius_meters": radius if radius < 10 else radius * scene.resolution,
            "room_id": ref.room_id,
        },
    )


def _boundary_region(ref: GPRef, scene: SceneMap) -> GPRef:
    if ref.kind != "room":
        raise GroundingFailure("INVALID_TYPE: boundary_region requires room.")
    cells = set(ref.cells)
    boundary = {
        (row, col)
        for row, col in cells
        if any((row + dr, col + dc) not in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    }
    return scene.region_from_cells(
        f"boundary_{ref.id}",
        boundary,
        {"type": "boundary", "room_id": ref.id},
    )


def _side_region(center_ref: GPRef, side_reference: GPRef, scene: SceneMap) -> GPRef:
    if center_ref.center is None or side_reference.center is None:
        raise GroundingFailure("INVALID_REGION: side_region references need centers.")
    base_room = scene.room_of(center_ref)
    center_row, center_col = center_ref.center
    vector_row = side_reference.center[0] - center_row
    vector_col = side_reference.center[1] - center_col
    vector_norm = math.hypot(vector_row, vector_col)
    if vector_norm == 0:
        raise GroundingFailure("INVALID_REGION: side_region references have the same center.")
    cells = {
        (row, col)
        for row, col in base_room.cells
        if is_traversable(scene.traversable, row, col)
        and (row - center_row) * vector_row + (col - center_col) * vector_col >= 0
    }
    return scene.region_from_cells(
        f"side_{center_ref.id}_toward_{side_reference.id}",
        cells,
        {
            "type": "side",
            "center_reference": center_ref.id,
            "side_reference": side_reference.id,
            "room_id": base_room.id,
        },
    )


def _half_room(room: GPRef, reference: GPRef, scene: SceneMap) -> GPRef:
    if room.kind != "room":
        raise GroundingFailure("INVALID_TYPE: half_room requires a room.")
    if room.center is None or reference.center is None:
        raise GroundingFailure("INVALID_REGION: half_room references need centers.")
    room_row, room_col = room.center
    vector_row = reference.center[0] - room_row
    vector_col = reference.center[1] - room_col
    vector_norm = math.hypot(vector_row, vector_col)
    if vector_norm == 0:
        raise GroundingFailure("INVALID_REGION: half_room reference is at the room center.")
    cells = {
        (row, col)
        for row, col in room.cells
        if is_traversable(scene.traversable, row, col)
        and (row - room_row) * vector_row + (col - room_col) * vector_col >= 0
    }
    return scene.region_from_cells(
        f"half_{room.id}_toward_{reference.id}",
        cells,
        {"type": "half_room", "room_id": room.id, "reference": reference.id},
    )


def _relative_waypoint(
    first: GPRef,
    second: GPRef,
    along: float,
    lateral_meters: float,
    radius_meters: float,
    scene: SceneMap,
    *,
    room: GPRef | None = None,
) -> GPRef:
    """Create a traversable region in the anchor-relative frame from ``first`` to ``second``.

    ``along`` is measured along the directed first-to-second axis (0 at
    ``first`` and 1 at ``second``); ``lateral_meters`` offsets along its
    left-hand normal.  This lets grounding code express a shape relative to
    semantic anchors without constructing map coordinates or cells directly.
    """
    if first.center is None or second.center is None:
        raise GroundingFailure("INVALID_REGION: relative_waypoint anchors need centers.")
    if not -1.0 <= along <= 2.0:
        raise GroundingFailure("INVALID_ARGUMENT: relative_waypoint along must lie in [-1, 2].")
    if abs(lateral_meters) > 5.0:
        raise GroundingFailure("INVALID_ARGUMENT: relative_waypoint lateral offset must be at most 5 meters.")
    if not 0.05 <= radius_meters <= 2.0:
        raise GroundingFailure("INVALID_ARGUMENT: relative_waypoint radius must lie in [0.05, 2] meters.")
    if room is None:
        first_room = scene.room_of(first)
        second_room = scene.room_of(second)
        if first_room.id != second_room.id:
            raise GroundingFailure("INVALID_SCOPE: anchors in different rooms require an explicit room.")
        room = first_room
    if room.kind != "room":
        raise GroundingFailure("INVALID_TYPE: relative_waypoint room must be a room reference.")
    axis_row = second.center[0] - first.center[0]
    axis_col = second.center[1] - first.center[1]
    axis_norm = math.hypot(axis_row, axis_col)
    if axis_norm <= 1e-6:
        raise GroundingFailure("INVALID_REGION: relative_waypoint anchors have the same center.")
    normal_row = -axis_col / axis_norm
    normal_col = axis_row / axis_norm
    lateral_cells = lateral_meters / scene.resolution
    desired_center = (
        first.center[0] + along * axis_row + lateral_cells * normal_row,
        first.center[1] + along * axis_col + lateral_cells * normal_col,
    )
    allowed = {
        cell
        for cell in room.cells
        if is_traversable(scene.traversable, cell[0], cell[1])
    }
    if not allowed:
        raise GroundingFailure("INVALID_REGION: relative_waypoint room has no traversable cells.")
    clearance = _clearance_within_room(scene, room)
    clearance_target = max(2.0, 0.25 / scene.resolution)
    cell = _clearance_aware_nearest_cell(
        desired_center,
        allowed,
        clearance,
        clearance_target=clearance_target,
        clearance_weight=1.0,
    )
    if cell is None:
        raise GroundingFailure("INVALID_REGION: relative_waypoint has no reachable cell.")
    radius_cells = max(1, round(radius_meters / scene.resolution))
    cells = {
        candidate
        for candidate in _disk_cells((float(cell[0]), float(cell[1])), radius_cells, scene.grid_size)
        if candidate in room.cells and is_traversable(scene.traversable, candidate[0], candidate[1])
    }
    if not cells:
        cells = {cell}
    identifier = (
        f"relative_waypoint_{first.id}_{second.id}_"
        f"a{along:+.3f}_l{lateral_meters:+.3f}_r{radius_meters:.3f}"
    ).replace("+", "p").replace("-", "m").replace(".", "_")
    return scene.region_from_cells(
        identifier,
        cells,
        {
            "type": "relative_waypoint",
            "first_anchor": first.id,
            "second_anchor": second.id,
            "room_id": room.id,
            "along": along,
            "lateral_meters": lateral_meters,
            "radius_meters": radius_meters,
            "projected_cell": [cell[0], cell[1]],
        },
    )


def _circle_regions(
    ref: GPRef,
    fraction: float,
    direction: str,
    scene: SceneMap,
    *,
    start_toward: GPRef | None = None,
    start_reference: GPRef | None = None,
) -> list[GPRef]:
    if ref.center is None:
        raise GroundingFailure("INVALID_REGION: circle reference needs center.")
    fraction = max(0.1, min(1.0, fraction))
    waypoint_count = max(3, int(math.ceil(8 * fraction)))
    if ref.kind == "room":
        centers = _room_loop_centers(ref, waypoint_count, direction, interior=fraction >= 0.95)
        room = ref
    else:
        room = scene.room_of(ref)
        centers = _object_loop_centers(ref, waypoint_count, direction, scene)
    waypoints: list[GPRef] = []
    seen: set[Cell] = set()
    allowed = {cell for cell in room.cells if is_traversable(scene.traversable, cell[0], cell[1])}
    clearance = _clearance_within_room(scene, room)
    clearance_target, clearance_weight = _circle_clearance_params(ref)
    for index, center in enumerate(centers, start=1):
        cell = _clearance_aware_nearest_cell(
            center,
            allowed - seen,
            clearance,
            clearance_target=clearance_target,
            clearance_weight=clearance_weight,
        ) or _clearance_aware_nearest_cell(
            center,
            allowed,
            clearance,
            clearance_target=clearance_target,
            clearance_weight=clearance_weight,
        )
        if cell is None:
            continue
        seen.add(cell)
        cells = {
            candidate
            for candidate in _disk_cells((float(cell[0]), float(cell[1])), max(2, round(0.2 / scene.resolution)), scene.grid_size)
            if candidate in room.cells and is_traversable(scene.traversable, candidate[0], candidate[1])
        }
        if not cells:
            cells = {cell}
        waypoints.append(
            scene.region_from_cells(
                f"circle_{ref.id}_{index}",
                cells,
                {
                    "type": "circle_waypoint",
                    "reference": ref.id,
                    "room_id": room.id,
                    "index": index,
                    "count": waypoint_count,
                    "fraction": fraction,
                    "direction": direction,
                    "clearance_cells": float(clearance[cell[0], cell[1]]),
                    "clearance_target_cells": clearance_target,
                    "clearance_weight": clearance_weight,
                },
            )
        )
    if not waypoints:
        raise GroundingFailure("INVALID_REGION: circle produced no reachable waypoints.")
    if start_toward is not None:
        waypoints = _rotate_waypoints_to_start(waypoints, start_toward, scene)
    elif start_reference is not None and (ref.kind != "room" or fraction >= 0.95):
        waypoints = _rotate_waypoints_to_start(waypoints, start_reference, scene)
    if fraction >= 0.95 and len(waypoints) > 1:
        waypoints.append(waypoints[0])
    return waypoints


def _follow_wall_regions(
    ref: GPRef,
    fraction: float,
    direction: str,
    scene: SceneMap,
    *,
    first_toward: GPRef | None = None,
    start_reference: GPRef | None = None,
) -> list[GPRef]:
    room = ref if ref.kind == "room" else scene.room_of(ref)
    fraction = max(0.1, min(1.0, fraction))
    waypoint_count = max(3, int(math.ceil(8 * fraction)))
    normalized_direction = _normalize_loop_direction(direction)
    candidates: list[tuple[str, list[GPRef]]] = []
    directions = ("clockwise", "counterclockwise") if normalized_direction == "auto" else (normalized_direction,)
    for candidate_direction in directions:
        waypoints = _wall_waypoints(room, waypoint_count, fraction, candidate_direction, scene)
        if start_reference is not None:
            waypoints = _rotate_waypoints_to_start(waypoints, start_reference, scene)
        if fraction >= 0.95 and len(waypoints) > 1:
            waypoints.append(waypoints[0])
        candidates.append((candidate_direction, waypoints))
    if not candidates:
        raise GroundingFailure("INVALID_REGION: follow_wall produced no reachable waypoints.")
    selected_direction, selected_waypoints = min(
        candidates,
        key=lambda item: _first_toward_rank(item[1], first_toward, scene),
    )
    return [
        _with_construction(
            waypoint,
            {
                **dict(waypoint.construction),
                "direction": selected_direction,
                "first_toward": first_toward.id if first_toward is not None else None,
            },
        )
        for waypoint in selected_waypoints
    ]


def _wall_waypoints(
    room: GPRef,
    waypoint_count: int,
    fraction: float,
    direction: str,
    scene: SceneMap,
) -> list[GPRef]:
    centers = _room_loop_centers(room, waypoint_count, direction, interior=False)
    allowed = {
        cell
        for cell in room.cells
        if is_traversable(scene.traversable, cell[0], cell[1])
    }
    if not allowed:
        raise GroundingFailure("INVALID_REGION: follow_wall room has no traversable cells.")
    boundary = _boundary_region(room, scene)
    clearance = _clearance_within_room(scene, room)
    waypoints: list[GPRef] = []
    seen: set[Cell] = set()
    for index, center in enumerate(centers, start=1):
        cell = _clearance_aware_nearest_cell(
            center,
            allowed - seen,
            clearance,
            clearance_target=1.0,
            clearance_weight=0.25,
        ) or _clearance_aware_nearest_cell(
            center,
            allowed,
            clearance,
            clearance_target=1.0,
            clearance_weight=0.25,
        )
        if cell is None:
            continue
        seen.add(cell)
        cells = {
            candidate
            for candidate in _disk_cells((float(cell[0]), float(cell[1])), max(2, round(0.25 / scene.resolution)), scene.grid_size)
            if candidate in room.cells and is_traversable(scene.traversable, candidate[0], candidate[1])
        }
        if not cells:
            cells = {cell}
        waypoints.append(
            scene.region_from_cells(
                f"follow_wall_{room.id}_{direction}_{index}",
                cells,
                {
                    "type": "wall_waypoint",
                    "room_id": room.id,
                    "boundary_region": boundary.id,
                    "index": index,
                    "count": waypoint_count,
                    "fraction": fraction,
                    "direction": direction,
                },
            )
        )
    if not waypoints:
        raise GroundingFailure("INVALID_REGION: follow_wall produced no reachable waypoints.")
    return waypoints


def _normalize_loop_direction(direction: str) -> str:
    normalized = direction.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"clockwise", "cw"}:
        return "clockwise"
    if normalized in {"counterclockwise", "counter_clockwise", "anticlockwise", "ccw"}:
        return "counterclockwise"
    return "auto"


def _first_toward_rank(waypoints: Sequence[GPRef], first_toward: GPRef | None, scene: SceneMap) -> tuple[int, float, str]:
    if first_toward is None or not waypoints:
        return (0, 0.0, waypoints[0].id if waypoints else "")
    anchor = _reference_anchor_cell(first_toward, scene)
    if anchor is None:
        return (0, 0.0, waypoints[0].id)
    best_index = min(
        range(len(waypoints)),
        key=lambda index: (
            _min_cell_distance(anchor, waypoints[index].cells),
            waypoints[index].id,
        ),
    )
    return (
        best_index,
        _min_cell_distance(anchor, waypoints[best_index].cells),
        waypoints[0].id,
    )


def _with_construction(ref: GPRef, construction: Mapping[str, object]) -> GPRef:
    return GPRef(
        kind=ref.kind,
        id=ref.id,
        category=ref.category,
        room_id=ref.room_id,
        center=ref.center,
        cells=ref.cells,
        construction={key: value for key, value in construction.items() if value is not None},
    )


def _object_loop_centers(ref: GPRef, waypoint_count: int, direction: str, scene: SceneMap) -> list[tuple[float, float]]:
    if ref.center is None:
        return []
    rows = [row for row, _col in ref.cells] or [ref.center[0]]
    cols = [col for _row, col in ref.cells] or [ref.center[1]]
    object_radius = max(max(rows) - min(rows), max(cols) - min(cols), 1) / 2
    radius = object_radius + max(4.0, 0.45 / scene.resolution)
    angles = [2 * math.pi * index / waypoint_count for index in range(waypoint_count)]
    if direction.lower() in {"counterclockwise", "ccw"}:
        angles = list(reversed(angles))
    return [
        (ref.center[0] + math.sin(angle) * radius, ref.center[1] + math.cos(angle) * radius)
        for angle in angles
    ]


def _room_loop_centers(
    room: GPRef,
    waypoint_count: int,
    direction: str,
    *,
    interior: bool = False,
) -> list[tuple[float, float]]:
    rows = [row for row, _col in room.cells]
    cols = [col for _row, col in room.cells]
    if not rows or not cols:
        return []
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    if interior and room.center is not None:
        row_radius = max(6.0, (bottom - top) * 0.36)
        col_radius = max(6.0, (right - left) * 0.36)
        center_row, center_col = room.center
        angles = [2 * math.pi * index / waypoint_count for index in range(waypoint_count)]
        loop = [
            (center_row + math.sin(angle) * row_radius, center_col + math.cos(angle) * col_radius)
            for angle in angles
        ]
    else:
        loop = _room_boundary_loop_centers(top, bottom, left, right, waypoint_count)
    if direction.lower() in {"counterclockwise", "ccw"}:
        loop = list(reversed(loop))
    return loop


def _room_boundary_loop_centers(
    top: int,
    bottom: int,
    left: int,
    right: int,
    waypoint_count: int,
) -> list[tuple[float, float]]:
    inset = 4
    top = min(top + inset, bottom)
    bottom = max(bottom - inset, top)
    left = min(left + inset, right)
    right = max(right - inset, left)
    base_loop = [
        (top, left),
        (top, (left + right) / 2),
        (top, right),
        ((top + bottom) / 2, right),
        (bottom, right),
        (bottom, (left + right) / 2),
        (bottom, left),
        ((top + bottom) / 2, left),
    ]
    if waypoint_count <= len(base_loop):
        return base_loop[:waypoint_count]
    return [
        base_loop[round(index * len(base_loop) / waypoint_count) % len(base_loop)]
        for index in range(waypoint_count)
    ]


def _rotate_waypoints_to_start(waypoints: list[GPRef], start_reference: GPRef, scene: SceneMap) -> list[GPRef]:
    if len(waypoints) <= 1:
        return waypoints
    start_cell = _reference_anchor_cell(start_reference, scene)
    if start_cell is None:
        return waypoints
    nearest_index = min(
        range(len(waypoints)),
        key=lambda index: (
            _min_cell_distance(start_cell, waypoints[index].cells),
            waypoints[index].id,
        ),
    )
    return waypoints[nearest_index:] + waypoints[:nearest_index]


def _reference_anchor_cell(ref: GPRef, scene: SceneMap) -> Cell | None:
    if ref.center is not None:
        allowed = {cell for cell in ref.cells if is_traversable(scene.traversable, cell[0], cell[1])}
        return _nearest_cell(ref.center, allowed) or (round(ref.center[0]), round(ref.center[1]))
    if ref.cells:
        row = sum(cell[0] for cell in ref.cells) / len(ref.cells)
        col = sum(cell[1] for cell in ref.cells) / len(ref.cells)
        return _nearest_cell((row, col), set(ref.cells))
    return None


def _min_cell_distance(cell: Cell, cells: frozenset[Cell]) -> float:
    if not cells:
        return float("inf")
    row, col = cell
    return min(math.hypot(row - candidate[0], col - candidate[1]) for candidate in cells)


def _clearance_within_room(scene: SceneMap, room: GPRef) -> np.ndarray:
    room_mask = np.zeros((scene.grid_size, scene.grid_size), dtype=bool)
    for row, col in room.cells:
        if 0 <= row < scene.grid_size and 0 <= col < scene.grid_size:
            room_mask[row, col] = True
    traversable = np.array(scene.traversable, dtype=bool)
    return distance_transform_edt(room_mask & traversable)


def _circle_clearance_params(ref: GPRef) -> tuple[float, float]:
    if ref.kind == "room":
        return float(CIRCLE_ROOM_WAYPOINT_CLEARANCE_CELLS), float(CIRCLE_ROOM_WAYPOINT_CLEARANCE_WEIGHT)
    return float(CIRCLE_OBJECT_WAYPOINT_CLEARANCE_CELLS), float(CIRCLE_OBJECT_WAYPOINT_CLEARANCE_WEIGHT)


def _clearance_aware_nearest_cell(
    center: tuple[float, float],
    cells: set[Cell],
    clearance: np.ndarray,
    *,
    clearance_target: float,
    clearance_weight: float,
) -> Cell | None:
    if not cells:
        return None
    row0, col0 = center
    search_radius = max(1.0, float(CIRCLE_WAYPOINT_SEARCH_RADIUS_CELLS))
    search_radius_sq = search_radius * search_radius
    local_cells = [
        cell
        for cell in cells
        if (cell[0] - row0) ** 2 + (cell[1] - col0) ** 2 <= search_radius_sq
    ]
    candidates = local_cells or list(cells)

    def score(cell: Cell) -> tuple[float, float, int, int]:
        distance = math.hypot(cell[0] - row0, cell[1] - col0)
        clearance_deficit = max(0.0, clearance_target - float(clearance[cell[0], cell[1]]))
        clearance_penalty = clearance_weight * clearance_deficit * clearance_deficit
        return distance + clearance_penalty, distance, cell[0], cell[1]

    return min(candidates, key=score)


def _nearest_cell(center: tuple[float, float], cells: set[Cell]) -> Cell | None:
    if not cells:
        return None
    return min(cells, key=lambda cell: ((cell[0] - center[0]) ** 2 + (cell[1] - center[1]) ** 2, cell[0], cell[1]))


def _disk_cells(center: tuple[float, float], radius: int, grid_size: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    row0, col0 = center
    for row in range(max(0, math.floor(row0 - radius)), min(grid_size - 1, math.ceil(row0 + radius)) + 1):
        for col in range(max(0, math.floor(col0 - radius)), min(grid_size - 1, math.ceil(col0 + radius)) + 1):
            if math.hypot(row - row0, col - col0) <= radius:
                cells.add((row, col))
    return cells


def _require_ref(value: object, label: str) -> GPRef:
    if isinstance(value, GPRef):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        refs = [item for item in value if isinstance(item, GPRef)]
        if len(refs) != len(value):
            raise GroundingFailure(f"INVALID_TYPE: {label} expected single reference.")
        if len(refs) == 1:
            return refs[0]
        if not refs:
            raise GroundingFailure(f"EMPTY_CANDIDATE: {label} expected single reference.")
        raise GroundingFailure(f"AMBIGUOUS_CANDIDATE: {label} expected single reference, got {len(refs)} candidates.")
    raise GroundingFailure(f"INVALID_TYPE: {label} expected single reference, got {type(value).__name__}.")


def _require_ref_list(value: object, label: str) -> list[GPRef]:
    if isinstance(value, GPRef):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        refs = [item for item in value if isinstance(item, GPRef)]
        if len(refs) != len(value):
            raise GroundingFailure(f"INVALID_TYPE: {label} expected reference list.")
        return refs
    raise GroundingFailure(f"INVALID_TYPE: {label} expected reference set.")


def _dedupe(refs: Sequence[GPRef]) -> list[GPRef]:
    seen: set[str] = set()
    result: list[GPRef] = []
    for ref in refs:
        if ref.id not in seen:
            result.append(ref)
            seen.add(ref.id)
    return result
