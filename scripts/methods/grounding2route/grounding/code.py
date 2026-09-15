"""Executable, planner-free grounding code for Grounding2Route.

This module deliberately sits on the grounding side of the method boundary:
it compiles :class:`GPProgramSpec` into a small Python scaffold, optionally lets
an LLM refine that scaffold, executes it against a read-only ``SceneMap`` API,
and validates the returned ``GroundedProgram``.  The planner is neither exposed
to the runtime nor imported by the generated program.
"""

from __future__ import annotations

import ast
import builtins
from collections import Counter
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from scripts.methods.grounding2route.config import PROMPT_DIR
from scripts.methods.grounding2route.grounding.grounder import (
    GroundingFailure,
    _eval_expr,
    _finalize_constraint,
    _ground_constraint,
)
from scripts.methods.grounding2route.grounding.scene import SceneMap
from scripts.methods.grounding2route.grounding.semantic_api import (
    CONSTRAINT_KINDS as _CONSTRAINT_KINDS,
    OPERATOR_NAMES as _OPERATOR_NAMES,
    PREDICATE_NAMES as _PREDICATE_NAMES,
    SEMANTIC_API_NAMES,
)
from scripts.methods.grounding2route.ir import (
    GPConstraintSpec,
    GPExpr,
    GPProgramSpec,
    GPRef,
    GroundedConstraint,
    GroundedProgram,
    GroundedSegment,
)
from scripts.methods.grounding2route.llm_client import GeminiClient


class GroundingCodeError(RuntimeError):
    """A structured, local failure from code validation or execution."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, object] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def feedback(self) -> dict[str, object]:
        return {"kind": self.code, "message": self.message, "evidence": self.details}


@dataclass(frozen=True)
class GroundingCodeExecution:
    grounded: GroundedProgram
    source: str
    diagnostics: tuple[str, ...] = ()
    verification: Mapping[str, object] = field(default_factory=dict)


_PURE_BUILTINS = {"len", "min", "max", "sum", "sorted", "range", "enumerate", "zip", "abs", "round", "all", "any"}


def compile_program_to_grounding_code(program: GPProgramSpec) -> str:
    """Deterministically compile the JSON/IR scaffold to executable code.

    The output is intentionally ordinary, readable Python.  Every operation is
    an invocation of the restricted semantic-map runtime.  It is therefore a
    stable starting point for the optional code-refinement LLM rather than an
    opaque serialization of the original JSON.
    """

    # Match the original deterministic interpreter's sequential binding scope:
    # a string can name only an *already resolved* binding.  In particular, a
    # binding called ``garbage_can`` must not turn entities("garbage_can") into
    # a self-reference in the generated Python scaffold.
    known = {"task_start", "start_position"}
    lines = ["def ground():"]
    for binding in program.bindings:
        lines.append(f"    {binding.name} = {_expr_to_code(binding.expr, known)}")
        known.add(binding.name)
    segment_vars: list[str] = []
    for segment in program.segments:
        start_var = f"_{segment.id}_start"
        target_var = f"_{segment.id}_target"
        constraints_var = f"_{segment.id}_constraints"
        lines.append(f"    {start_var} = {_expr_to_code(segment.start, known)}")
        lines.append(f"    set_segment_context({segment.id!r}, {start_var})")
        lines.append(f"    {target_var} = {_expr_to_code(segment.target, known)}")
        constraint_sources: list[str] = []
        for constraint in segment.constraints:
            args = ", ".join(_expr_to_code(value, known) for value in constraint.exprs)
            kwargs: list[str] = []
            if constraint.spatial_scope is not None:
                kwargs.append(f"spatial_scope={_expr_to_code(constraint.spatial_scope, known)}")
            if constraint.source_text:
                kwargs.append(f"source_text={constraint.source_text!r}")
            extra = ", " + ", ".join(kwargs) if kwargs else ""
            constraint_sources.append(f"constraint({constraint.kind!r}, {args}{extra})")
        lines.append(f"    {constraints_var} = [{', '.join(constraint_sources)}]")
        segment_var = f"_{segment.id}"
        segment_vars.append(segment_var)
        lines.append(f"    {segment_var} = segment({segment.id!r}, {start_var}, {target_var}, {constraints_var})")
        known.add(segment.id)
    bindings_source = "{" + ", ".join(f"{binding.name!r}: {binding.name}" for binding in program.bindings) + "}"
    lines.append(f"    return task([{', '.join(segment_vars)}], {bindings_source})")
    return "\n".join(lines) + "\n"


def execute_grounding_code(
    source: str,
    scene: SceneMap,
    *,
    expected_program: GPProgramSpec | None = None,
    allow_helpers: bool = True,
) -> GroundingCodeExecution:
    """Validate and run planner-free grounding code on one ``SceneMap``."""

    tree = _validate_code(source, allow_helpers=allow_helpers)
    runtime = GroundingRuntime(scene)
    namespace = runtime.namespace()
    try:
        compiled = compile(tree, "<grounding2route-grounding>", "exec")
        namespace["__builtins__"] = _safe_builtins()
        exec(compiled, namespace, namespace)
        ground = namespace.get("ground")
        if not callable(ground):
            raise GroundingCodeError("MISSING_GROUND", "Code must define ground() with no arguments.")
        result = ground()
    except GroundingCodeError:
        raise
    except GroundingFailure as exc:
        raise _execution_error(str(exc)) from exc
    except Exception as exc:
        raise GroundingCodeError(
            "EXECUTION_ERROR",
            f"{type(exc).__name__}: {exc}",
            details={"exception_type": type(exc).__name__},
        ) from exc
    if not isinstance(result, GroundedProgram):
        raise GroundingCodeError(
            "INVALID_RETURN_TYPE",
            "ground() must return task(...), which produces a GroundedProgram.",
            details={"actual_type": type(result).__name__},
        )
    verification = verify_grounded_program(result, scene, expected_program=expected_program, code_tree=tree)
    errors = verification.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, Mapping):
            raise GroundingCodeError(
                str(first.get("kind", "VERIFICATION_FAILED")),
                str(first.get("message", "Grounding verification failed.")),
                details=first,
            )
        raise GroundingCodeError("VERIFICATION_FAILED", str(first))
    diagnostics = tuple(str(item) for item in verification.get("warnings", []) if isinstance(item, str))
    return GroundingCodeExecution(result, source, diagnostics, verification)


def verify_grounded_program(
    grounded: GroundedProgram,
    scene: SceneMap,
    *,
    expected_program: GPProgramSpec | None,
    code_tree: ast.Module | None = None,
) -> dict[str, object]:
    """Grounding-only structural and semantic-map verification.

    This intentionally never queries the planner or path feasibility.
    """
    errors: list[dict[str, object]] = []
    warnings: list[str] = []
    known_refs = {scene.start.id, *scene.entities.keys(), *scene.rooms.keys()}
    if grounded.status != "success":
        errors.append({"kind": "GROUNDING_STATUS", "message": grounded.failure_reason or grounded.status})
    if not grounded.segments:
        errors.append({"kind": "EMPTY_SEGMENTS", "message": "Grounded task contains no ordered segments."})

    segment_ids = [segment.id for segment in grounded.segments]
    if any(not segment_id.strip() for segment_id in segment_ids):
        errors.append({"kind": "INVALID_SEGMENT_ID", "message": "Every segment must have a non-empty id."})
    duplicate_segments = sorted({item for item in segment_ids if segment_ids.count(item) > 1})
    if duplicate_segments:
        errors.append({"kind": "DUPLICATE_SEGMENT_ID", "message": f"Duplicate segment ids: {', '.join(duplicate_segments)}."})
    known_segments = set(segment_ids)
    constraint_ids: set[str] = set()

    for segment in grounded.segments:
        for label, ref in (("start", segment.start_reference), ("target", segment.target)):
            _verify_ref(ref, known_refs, f"segment {segment.id} {label}", errors)
        for position, constraint in enumerate(segment.constraints):
            cid = constraint.constraint_id or f"{segment.id}_c{position}"
            if not constraint.constraint_id:
                errors.append({"kind": "MISSING_CONSTRAINT_ID", "message": f"Constraint {segment.id}[{position}] has no constraint_id."})
            elif cid in constraint_ids:
                errors.append({"kind": "DUPLICATE_CONSTRAINT_ID", "message": f"Constraint id {cid!r} is not unique."})
            constraint_ids.add(cid)
            if constraint.kind not in _CONSTRAINT_KINDS:
                errors.append({"kind": "UNKNOWN_CONSTRAINT", "message": f"Constraint {cid} has unsupported kind {constraint.kind!r}."})
                continue
            _verify_constraint_shape(constraint, cid, segment.id, known_segments, scene, known_refs, errors)

    if expected_program is not None:
        missing_bindings = [binding.name for binding in expected_program.bindings if binding.name not in grounded.bindings]
        if missing_bindings:
            errors.append({"kind": "BINDING_STRUCTURE_MISMATCH", "message": "Grounding code omitted bindings required by the JSON IR scaffold.", "missing_bindings": missing_bindings})
        expected_ids = [segment.id for segment in expected_program.segments]
        if expected_ids != segment_ids:
            errors.append({"kind": "SEGMENT_ORDER_MISMATCH", "message": "Grounding code changed the ordered segment scaffold.", "expected_segments": expected_ids, "observed_segments": segment_ids})
        expected_kinds = [[constraint.kind for constraint in segment.constraints] for segment in expected_program.segments]
        observed_kinds = [[constraint.kind for constraint in segment.constraints] for segment in grounded.segments]
        missing_constraint_kinds = [
            {"segment": segment.id, "missing": dict(Counter(expected) - Counter(observed))}
            for segment, expected, observed in zip(expected_program.segments, expected_kinds, observed_kinds)
            if Counter(expected) - Counter(observed)
        ]
        if missing_constraint_kinds:
            errors.append({"kind": "CONSTRAINT_STRUCTURE_MISMATCH", "message": "Grounding code omitted or changed an IR constraint kind.", "missing_constraint_kinds": missing_constraint_kinds})
        if code_tree is not None:
            warnings.extend(_dependency_warnings(expected_program, code_tree))
    return {"status": "success" if not errors else "failed", "errors": errors, "warnings": warnings}


def _verify_constraint_shape(
    constraint: GroundedConstraint,
    constraint_id: str,
    host_segment: str,
    known_segments: set[str],
    scene: SceneMap,
    known_refs: set[str],
    errors: list[dict[str, object]],
) -> None:
    exact_arity = {"prefer_near": 1, "prefer_far": 1, "prefer_relative": 2, "forbid": 1}
    minimum_arity = {"require_visit": 1, "require_visit_in_order": 1, "prefer_path_shape": 1}
    received = len(constraint.refs)
    if constraint.kind in exact_arity and received != exact_arity[constraint.kind]:
        errors.append({"kind": "CONSTRAINT_ARITY", "message": f"Constraint {constraint_id} has invalid arity: {constraint.kind} requires {exact_arity[constraint.kind]} refs but received {received}."})
    if constraint.kind in minimum_arity and received < minimum_arity[constraint.kind]:
        errors.append({"kind": "CONSTRAINT_ARITY", "message": f"Constraint {constraint_id} has invalid arity: {constraint.kind} requires at least {minimum_arity[constraint.kind]} ref."})
    for ref in constraint.refs:
        _verify_ref(ref, known_refs, f"constraint {constraint_id}", errors)
    expected_hardness = "soft" if constraint.kind.startswith("prefer_") else "hard"
    if constraint.hardness not in {"hard", "soft"}:
        errors.append({"kind": "INVALID_HARDNESS", "message": f"Constraint {constraint_id} must declare hard or soft hardness."})
    elif constraint.hardness != expected_hardness:
        errors.append({"kind": "HARDNESS_MISMATCH", "message": f"Constraint {constraint_id} declares {constraint.hardness} but {constraint.kind} is {expected_hardness}."})
    if len(constraint.argument_roles) != received:
        errors.append({"kind": "ARGUMENT_ROLE_ARITY", "message": f"Constraint {constraint_id} has {received} refs but {len(constraint.argument_roles)} argument roles."})
    if constraint.kind == "prefer_relative":
        if constraint.relation not in {"closer_to", "farther_from"}:
            errors.append({"kind": "INVALID_RELATION", "message": f"Constraint {constraint_id} must use closer_to or farther_from."})
        expected_roles = (("farther_ref", "closer_ref") if constraint.relation == "farther_from" else ("closer_ref", "farther_ref"))
        if constraint.argument_roles and constraint.argument_roles != expected_roles:
            errors.append({"kind": "INVALID_ARGUMENT_ROLES", "message": f"Constraint {constraint_id} has invalid relative argument roles."})
        if received == 2:
            try:
                if scene.room_of(constraint.refs[0]).id != scene.room_of(constraint.refs[1]).id:
                    errors.append({"kind": "RELATION_MAP_INCONSISTENCY", "message": f"Constraint {constraint_id} relative refs are not in the same room."})
            except Exception as exc:
                errors.append({"kind": "RELATION_MAP_INCONSISTENCY", "message": f"Constraint {constraint_id} cannot resolve relative-room fact: {exc}"})
    invalid_scope_ids = sorted(set(constraint.segment_scope) - known_segments)
    if invalid_scope_ids:
        errors.append({"kind": "INVALID_SEGMENT_SCOPE", "message": f"Constraint {constraint_id} references unknown segment(s): {', '.join(invalid_scope_ids)}."})
    scope = constraint.spatial_scope
    scope_refs = scope if isinstance(scope, tuple) else (scope,) if scope is not None else ()
    for ref in scope_refs:
        _verify_ref(ref, known_refs, f"constraint {constraint_id} spatial scope", errors)
        if ref.kind not in {"room", "region"}:
            errors.append({"kind": "INVALID_SPATIAL_SCOPE", "message": f"Constraint {constraint_id} spatial scope must be a room or region, got {ref.kind}."})


def llm_refine_grounding_code(
    instruction_text: str,
    *,
    program: GPProgramSpec,
    scaffold: str,
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    scene_summary: Mapping[str, object],
    llm_client: GeminiClient,
    allow_helpers: bool,
    attempt: int = 0,
) -> tuple[str, dict[str, object]]:
    prompt = _grounding_prompt(
        instruction_text,
        ir=program.to_json(),
        code=scaffold,
        entity_categories=entity_categories,
        room_categories=room_categories,
        scene_summary=scene_summary,
        allow_helpers=allow_helpers,
        feedback=None,
        direct_cap=False,
    )
    text, metadata = llm_client.response_text(
        system_prompt="You refine planner-free Grounding2Route grounding code. Return Python code only.",
        user_prompt=prompt,
        cache_namespace=f"grounding2route_grounding_code_refine_v{attempt}",
    )
    return _extract_python(text), {"mode": "code_refinement", "attempt": attempt, "llm": metadata, "raw_code": text}


def llm_generate_direct_grounding_code(
    instruction_text: str,
    *,
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    scene_summary: Mapping[str, object],
    llm_client: GeminiClient,
    allow_helpers: bool,
) -> tuple[str, dict[str, object]]:
    prompt = _grounding_prompt(
        instruction_text,
        ir=None,
        code=None,
        entity_categories=entity_categories,
        room_categories=room_categories,
        scene_summary=scene_summary,
        allow_helpers=allow_helpers,
        feedback=None,
        direct_cap=True,
    )
    text, metadata = llm_client.response_text(
        system_prompt="You generate planner-free Grounding2Route grounding code. Return Python code only.",
        user_prompt=prompt,
        cache_namespace="grounding2route_direct_cap_grounding_code",
    )
    return _extract_python(text), {"mode": "direct_cap", "llm": metadata, "raw_code": text}


def llm_repair_grounding_code(
    instruction_text: str,
    *,
    program: GPProgramSpec | None,
    previous_code: str,
    feedback: Mapping[str, object],
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    scene_summary: Mapping[str, object],
    llm_client: GeminiClient,
    allow_helpers: bool,
    attempt: int,
) -> tuple[str, dict[str, object]]:
    prompt = _grounding_prompt(
        instruction_text,
        ir=program.to_json() if program else None,
        code=previous_code,
        entity_categories=entity_categories,
        room_categories=room_categories,
        scene_summary=scene_summary,
        allow_helpers=allow_helpers,
        feedback=feedback,
        direct_cap=program is None,
    )
    text, metadata = llm_client.response_text(
        system_prompt="You repair planner-free Grounding2Route grounding code from execution feedback. Return Python code only.",
        user_prompt=prompt,
        cache_namespace=f"grounding2route_grounding_code_repair_v{attempt}",
    )
    return _extract_python(text), {"mode": "code_repair", "attempt": attempt, "llm": metadata, "raw_code": text}


class GroundingRuntime:
    """Read-only semantic-map API and GroundedProgram builders for sandbox code."""

    def __init__(self, scene: SceneMap) -> None:
        self.scene = scene
        self.env: dict[str, object] = {"task_start": scene.start, "start_position": scene.start}
        self.current_segment_id: str | None = None

    def namespace(self) -> dict[str, object]:
        namespace: dict[str, object] = {
            "task_start": self.scene.start,
            "start_position": self.scene.start,
            "set_segment_context": self.set_segment_context,
            "segment": self.segment,
            "constraint": self.constraint,
            "task": self.task,
            "position": self.position,
            "distance": self.distance,
            "geodesic_distance": self.geodesic_distance,
            "inside": self.inside,
            "intersects": self.intersects,
            "object_bbox": self.object_bbox,
            "room_polygon": self.room_polygon,
            "where": self.where,
            "compare": self.compare,
            "all_of": lambda *values: all(bool(value) for value in values),
            "any_of": lambda *values: any(bool(value) for value in values),
            "not_": lambda value: not bool(value),
        }
        for op, name in _OPERATOR_NAMES.items():
            namespace[name] = self._operator(op)
        for name in _PREDICATE_NAMES:
            namespace[name] = self._predicate(name)
        return namespace

    def _operator(self, op: str) -> Callable[..., object]:
        def invoke(*args: object, **kwargs: object) -> object:
            try:
                return _eval_expr(GPExpr(op=op, args=tuple(args), kwargs=kwargs), self.scene, self.env)
            except GroundingFailure:
                raise
            except Exception as exc:
                raise GroundingFailure(f"{op}: {exc}") from exc

        return invoke

    def _predicate(self, op: str) -> Callable[..., bool]:
        def predicate(candidate: object, *args: object) -> bool:
            if not isinstance(candidate, GPRef):
                raise GroundingFailure(f"INVALID_TYPE: {op} candidate must be a map reference.")
            expression = GPExpr(op=op, args=tuple(args))
            # Reuse the DSL's predicate machinery via a one-element where query.
            selected = _eval_expr(GPExpr(op="where", args=([candidate], expression)), self.scene, self.env)
            return bool(selected)

        return predicate

    def where(self, candidates: object, predicate: Callable[[GPRef], object]) -> list[GPRef]:
        refs = _ref_list(candidates, "where")
        return [ref for ref in refs if bool(predicate(ref))]

    def compare(self, left: object, operator: str, right: object) -> bool:
        return bool(_eval_expr(GPExpr("compare", (left, operator, right)), self.scene, self.env))

    def set_segment_context(self, segment_id: str, start: object) -> None:
        self.current_segment_id = str(segment_id)
        self.env["segment_start"] = _ref(start, "segment start")

    def constraint(
        self,
        kind: str,
        *exprs: object,
        spatial_scope: object | None = None,
        segment_scope: str | Sequence[str] | None = None,
        source_text: str | None = None,
    ) -> tuple[GroundedConstraint, ...]:
        if kind not in _CONSTRAINT_KINDS:
            raise GroundingFailure(f"UNSUPPORTED_CONSTRAINT: {kind}")
        if not self.current_segment_id:
            raise GroundingFailure("MISSING_SEGMENT_CONTEXT: call set_segment_context before constraint().")
        if segment_scope is None:
            resolved_segment_scope = (self.current_segment_id,)
        elif isinstance(segment_scope, str):
            resolved_segment_scope = () if segment_scope == "global" else (segment_scope,)
        elif isinstance(segment_scope, Sequence) and not isinstance(segment_scope, (str, bytes)):
            resolved_segment_scope = tuple(str(item) for item in segment_scope)
        else:
            raise GroundingFailure("INVALID_SCOPE: segment_scope must be a segment id, sequence, or 'global'.")
        spec = GPConstraintSpec(
            kind=kind,  # type: ignore[arg-type]
            exprs=tuple(exprs),
            segment_scope=resolved_segment_scope,
            spatial_scope=spatial_scope,
            source_text=source_text,
        )
        return _ground_constraint(spec, self.scene, self.env, self.current_segment_id)

    def segment(
        self,
        segment_id: str,
        start: object,
        target: object,
        constraints: Iterable[object] = (),
    ) -> GroundedSegment:
        flattened: list[GroundedConstraint] = []
        for value in constraints:
            if isinstance(value, GroundedConstraint):
                flattened.append(value)
            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, GPRef)):
                flattened.extend(item for item in value if isinstance(item, GroundedConstraint))
            else:
                raise GroundingFailure("INVALID_CONSTRAINT: segment constraints must be constraint(...) results.")
        resolved_segment_id = str(segment_id)
        canonical_constraints = tuple(
            _finalize_constraint(item, segment_id=resolved_segment_id, index=index)
            for index, item in enumerate(flattened)
        )
        return GroundedSegment(
            id=resolved_segment_id,
            start_reference=_ref(start, "segment start"),
            target=_ref(target, "segment target"),
            constraints=canonical_constraints,
        )

    def task(self, segments: object, bindings: object | None = None) -> GroundedProgram:
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            raise GroundingFailure("INVALID_SEGMENTS: task requires a sequence of segment(...).")
        grounded_segments = tuple(segment for segment in segments if isinstance(segment, GroundedSegment))
        if len(grounded_segments) != len(segments) or not grounded_segments:
            raise GroundingFailure("INVALID_SEGMENTS: task requires at least one GroundedSegment.")
        resolved_bindings = {"task_start": self.scene.start, "start_position": self.scene.start}
        if isinstance(bindings, Mapping):
            resolved_bindings.update({str(key): value for key, value in bindings.items()})
        return GroundedProgram(segments=grounded_segments, bindings=resolved_bindings)

    def position(self, ref: object) -> tuple[float, float]:
        value = _ref(ref, "position")
        if value.center is None:
            raise GroundingFailure("INVALID_REFERENCE: position requires a reference with a center.")
        return value.center

    def distance(self, first: object, second: object) -> float:
        return self.scene.distance(_ref(first, "distance"), _ref(second, "distance"), "euclidean")

    def geodesic_distance(self, first: object, second: object) -> float:
        return self.scene.distance(_ref(first, "geodesic_distance"), _ref(second, "geodesic_distance"), "geodesic")

    def inside(self, child: object, container: object) -> bool:
        return bool(_ref(child, "inside").cells) and _ref(child, "inside").cells.issubset(_ref(container, "inside").cells)

    def intersects(self, first: object, second: object) -> bool:
        return bool(_ref(first, "intersects").cells & _ref(second, "intersects").cells)

    def object_bbox(self, ref: object) -> tuple[int, int, int, int]:
        return _bbox(_ref(ref, "object_bbox"))

    def room_polygon(self, ref: object) -> tuple[tuple[int, int], ...]:
        value = _ref(ref, "room_polygon")
        if value.kind != "room":
            raise GroundingFailure("INVALID_TYPE: room_polygon requires a room.")
        row0, col0, row1, col1 = _bbox(value)
        return ((row0, col0), (row0, col1), (row1, col1), (row1, col0))


def _expr_to_code(value: object, known_names: set[str], *, predicate: bool = False) -> str:
    if isinstance(value, GPExpr):
        op = value.op
        if op == "where" and len(value.args) == 2:
            return f"where({_expr_to_code(value.args[0], known_names)}, lambda self: {_expr_to_code(value.args[1], known_names, predicate=True)})"
        if op == "compare" and len(value.args) == 3:
            return f"compare({_expr_to_code(value.args[0], known_names, predicate=predicate)}, {_expr_to_code(value.args[1], known_names, predicate=predicate)}, {_expr_to_code(value.args[2], known_names, predicate=predicate)})"
        if op == "and":
            return "all_of(" + ", ".join(_expr_to_code(item, known_names, predicate=predicate) for item in value.args) + ")"
        if op == "or":
            return "any_of(" + ", ".join(_expr_to_code(item, known_names, predicate=predicate) for item in value.args) + ")"
        if op == "not" and len(value.args) == 1:
            return f"not_({_expr_to_code(value.args[0], known_names, predicate=predicate)})"
        if predicate and op in _PREDICATE_NAMES:
            args = ", ".join(_expr_to_code(item, known_names, predicate=True) for item in value.args)
            return f"{op}(self{', ' if args else ''}{args})"
        function = _OPERATOR_NAMES.get(op, op)
        args = [_expr_to_code(item, known_names, predicate=predicate) for item in value.args]
        args.extend(f"{key}={_expr_to_code(item, known_names, predicate=predicate)}" for key, item in value.kwargs.items())
        return f"{function}({', '.join(args)})"
    if isinstance(value, str):
        return value if value in known_names or (predicate and value == "self") else repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_expr_to_code(item, known_names, predicate=predicate) for item in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_expr_to_code(item, known_names, predicate=predicate) for item in value)
        return f"({inner}{',' if len(value) == 1 else ''})"
    return repr(value)


def _validate_code(source: str, *, allow_helpers: bool) -> ast.Module:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise GroundingCodeError("SYNTAX_ERROR", str(exc), details={"line": exc.lineno, "offset": exc.offset}) from exc
    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) != len(tree.body) or not any(node.name == "ground" for node in function_defs):
        raise GroundingCodeError("INVALID_TOP_LEVEL", "Only function definitions are allowed and ground() is required.")
    if not allow_helpers and any(node.name != "ground" for node in function_defs):
        raise GroundingCodeError("HELPERS_DISABLED", "This ablation permits only ground(), not LLM-defined helpers.")
    function_names = {node.name for node in function_defs}
    api_names = {"task_start", "start_position", *SEMANTIC_API_NAMES, *_PURE_BUILTINS}
    helper_names = [node.name for node in function_defs if node.name != "ground"]
    if len(set(node.name for node in function_defs)) != len(function_defs):
        raise GroundingCodeError("DUPLICATE_FUNCTION", "Grounding code may not redefine a function.")
    collisions = sorted(set(helper_names) & api_names)
    if collisions:
        raise GroundingCodeError(
            "FIXED_API_REDEFINITION",
            f"Grounding helpers may not redefine fixed APIs: {', '.join(collisions)}.",
        )
    for function in function_defs:
        has_type_parameters = bool(getattr(function, "type_params", ()))
        has_annotations = function.returns is not None or any(arg.annotation is not None for arg in (*function.args.args, *function.args.kwonlyargs))
        if function.decorator_list or has_annotations or has_type_parameters:
            raise GroundingCodeError("UNSUPPORTED_FUNCTION_SIGNATURE", "Decorators, annotations, and type parameters are not allowed.")
        if function.args.vararg is not None or function.args.kwarg is not None or function.args.kwonlyargs:
            raise GroundingCodeError("UNSUPPORTED_FUNCTION_SIGNATURE", "Variadic and keyword-only parameters are not allowed.")
        if function.name == "ground" and (function.args.args or function.args.posonlyargs):
            raise GroundingCodeError("INVALID_GROUND_SIGNATURE", "ground() must not take arguments.")
        local_names = {arg.arg for arg in (*function.args.posonlyargs, *function.args.args)}
        _validate_statements(function.body, local_names, api_names | function_names, function_names)
    return tree


def _validate_statements(
    statements: Sequence[ast.stmt],
    local_names: set[str],
    available_names: set[str],
    function_names: set[str],
) -> None:
    for statement in statements:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise GroundingCodeError("UNSAFE_ASSIGNMENT", "Only simple name assignments are allowed.")
            _validate_expression(statement.value, local_names | available_names, function_names)
            local_names.add(statement.targets[0].id)
        elif isinstance(statement, ast.Return):
            if statement.value is None:
                raise GroundingCodeError("INVALID_RETURN", "grounding functions must return a value.")
            _validate_expression(statement.value, local_names | available_names, function_names)
        elif isinstance(statement, ast.Expr):
            _validate_expression(statement.value, local_names | available_names, function_names)
        elif isinstance(statement, ast.If):
            _validate_expression(statement.test, local_names | available_names, function_names)
            _validate_statements(statement.body, set(local_names), available_names, function_names)
            _validate_statements(statement.orelse, set(local_names), available_names, function_names)
        else:
            raise GroundingCodeError("UNSAFE_STATEMENT", f"{type(statement).__name__} is not allowed in grounding code.")


def _validate_expression(node: ast.AST, available_names: set[str], function_names: set[str]) -> None:
    if isinstance(node, ast.Name):
        if node.id not in available_names:
            raise GroundingCodeError("UNDEFINED_BINDING", f"Name {node.id!r} is not defined by the scaffold or API.")
        return
    if isinstance(node, ast.Constant):
        return
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for item in node.elts:
            _validate_expression(item, available_names, function_names)
        return
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is not None:
                _validate_expression(key, available_names, function_names)
            _validate_expression(value, available_names, function_names)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise GroundingCodeError("FORBIDDEN_CALL", "Only named semantic-map API or helper calls are allowed.")
        if node.func.id not in available_names:
            raise GroundingCodeError("FORBIDDEN_CALL", f"Call to {node.func.id!r} is not part of the grounding API.")
        for item in node.args:
            _validate_expression(item, available_names, function_names)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise GroundingCodeError("FORBIDDEN_CALL", "Starred keyword arguments are not allowed.")
            _validate_expression(keyword.value, available_names, function_names)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
        _validate_expression(node.left, available_names, function_names)
        _validate_expression(node.right, available_names, function_names)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        _validate_expression(node.operand, available_names, function_names)
        return
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _validate_expression(value, available_names, function_names)
        return
    if isinstance(node, ast.Compare):
        _validate_expression(node.left, available_names, function_names)
        for comparator in node.comparators:
            _validate_expression(comparator, available_names, function_names)
        return
    if isinstance(node, ast.IfExp):
        _validate_expression(node.test, available_names, function_names)
        _validate_expression(node.body, available_names, function_names)
        _validate_expression(node.orelse, available_names, function_names)
        return
    if isinstance(node, ast.Subscript):
        _validate_expression(node.value, available_names, function_names)
        _validate_expression(node.slice, available_names, function_names)
        return
    if isinstance(node, ast.Slice):
        for value in (node.lower, node.upper, node.step):
            if value is not None:
                _validate_expression(value, available_names, function_names)
        return
    if isinstance(node, ast.Lambda):
        names = available_names | {arg.arg for arg in node.args.args}
        _validate_expression(node.body, names, function_names)
        return
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        names = set(available_names)
        for generator in node.generators:
            _validate_expression(generator.iter, names, function_names)
            if not isinstance(generator.target, ast.Name) or generator.is_async:
                raise GroundingCodeError("UNSAFE_COMPREHENSION", "Only simple synchronous comprehension targets are allowed.")
            names.add(generator.target.id)
            for condition in generator.ifs:
                _validate_expression(condition, names, function_names)
        _validate_expression(node.elt, names, function_names)
        return
    raise GroundingCodeError("UNSAFE_EXPRESSION", f"{type(node).__name__} is not allowed in grounding code.")


def _grounding_prompt(
    instruction_text: str,
    *,
    ir: Mapping[str, object] | None,
    code: str | None,
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    scene_summary: Mapping[str, object],
    allow_helpers: bool,
    feedback: Mapping[str, object] | None,
    direct_cap: bool,
) -> str:
    template_path = PROMPT_DIR / "grounding" / "gemini_grounding_code_prompt.md"
    template = template_path.read_text(encoding="utf-8") if template_path.exists() else _DEFAULT_PROMPT
    blocks = [
        template,
        f"## Instruction\n{instruction_text.strip()}",
        f"## Object categories\n{', '.join(entity_categories)}",
        f"## Room categories\n{', '.join(room_categories)}",
        f"## Scene category summary\n```json\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}\n```",
        f"## Helper functions\n{'You may define pure helper functions.' if allow_helpers else 'Do not define helper functions.'}",
    ]
    if ir is not None:
        blocks.append(f"## JSON IR semantic scaffold\n```json\n{json.dumps(ir, ensure_ascii=False, indent=2)}\n```")
    if code is not None:
        blocks.append(f"## Current grounding code\n```python\n{code}\n```")
    if feedback is not None:
        blocks.append(f"## Structured execution feedback\n```json\n{json.dumps(feedback, ensure_ascii=False, indent=2)}\n```")
    blocks.append("Return one complete Python program only. " + ("This is Direct CaP: create ground() directly." if direct_cap else "Preserve the IR segment order and constraint kinds unless the feedback proves the IR itself is wrong."))
    return "\n\n".join(blocks)


_DEFAULT_PROMPT = """You write Grounding2Route grounding code only. The code executes against a read-only semantic map and must define `ground()` with no arguments. It must return `task(segments, bindings)`.

Fixed map APIs: entities(category), rooms(category), position(ref), distance(a,b), geodesic_distance(a,b), room_of(ref), inside(a,b), intersects(a,b), object_bbox(ref), room_polygon(room).

Grounding selection/region APIs: kth_nearest, kth_farthest, choose_any, unique, where, in_room, union, intersection, exclude, count, region_of, room_region, midpoint_region, between_region, near_region, side_region, boundary_region, half_room, relative_waypoint, circle, follow_wall, passage_regions, closest_pair_member.

Task builders: set_segment_context(id,start), constraint(kind,...), segment(id,start,target,constraints), task(segments,bindings).

When no single fixed operator expresses a grounding relation, top-level pure helpers may compose only the exposed map APIs. They may not invent scene facts, inspect IDs, access raw cells, create absolute coordinates, or guess candidates; if a relation cannot be derived from exposed primitives, grounding must fail.

The code must only ground references, regions, constraints, scopes, and ordered segments. Never import anything; never use attributes; never access files/network; never output a path; never call or mention a planner, cost, weight, search, A*, or policy. Planning is outside this module."""


def _extract_python(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:python|py)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # ``GeminiClient._extract_text`` strips a fence but historically only
    # recognizes text/json labels, so a Python fence can arrive as
    # ``python\ndef ground(): ...``.  Treat that transport artefact as code,
    # not as a top-level identifier in the sandbox.
    lines = stripped.splitlines()
    if lines and lines[0].strip().lower() in {"python", "py"}:
        return "\n".join(lines[1:]).strip()
    return stripped


def _execution_error(reason: str) -> GroundingCodeError:
    match = re.search(r"(?:BINDING_FAILED|SEGMENT_FAILED|CONSTRAINT_FAILED):\s*([^:]+):\s*(.*)", reason)
    details: dict[str, object] = {"grounding_error": reason}
    if match:
        details["location"] = match.group(1)
        details["execution_evidence"] = match.group(2)
    if "ORDINAL_OUT_OF_RANGE" in reason:
        return GroundingCodeError("CARDINALITY_ERROR", reason, details=details)
    if "EMPTY_CANDIDATE" in reason or "AMBIGUOUS_CANDIDATE" in reason:
        return GroundingCodeError("CARDINALITY_ERROR", reason, details=details)
    if "UNKNOWN_" in reason or "UNSUPPORTED_" in reason:
        return GroundingCodeError("TYPE_OR_API_ERROR", reason, details=details)
    return GroundingCodeError("GROUNDING_EXECUTION_ERROR", reason, details=details)


def _verify_ref(ref: GPRef, known_refs: set[str], location: str, errors: list[dict[str, object]]) -> None:
    if ref.kind != "region" and ref.id not in known_refs:
        errors.append({"kind": "SCENE_REFERENCE_ERROR", "message": f"{location} refers to unknown {ref.kind} {ref.id!r}."})
    if not ref.cells:
        errors.append({"kind": "EMPTY_REFERENCE", "message": f"{location} resolved to no map cells.", "reference": ref.id})


def _dependency_warnings(program: GPProgramSpec, tree: ast.Module) -> list[str]:
    binding_names = {binding.name for binding in program.bindings} | {"task_start", "start_position"}
    expected = {
        binding.name: _expr_dependencies(binding.expr) & binding_names
        for binding in program.bindings
    }
    assignments: dict[str, set[str]] = {}
    ground = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ground"), None)
    if ground is None:
        return []
    for statement in ground.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            assignments[statement.targets[0].id] = {name.id for name in ast.walk(statement.value) if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)}
    warnings: list[str] = []
    for name, dependencies in expected.items():
        observed = assignments.get(name)
        missing = sorted(dependency for dependency in dependencies if observed is not None and dependency not in observed)
        if missing:
            warnings.append(f"DEPENDENCY_WARNING: binding {name!r} no longer visibly references {', '.join(missing)} from the JSON IR.")
    return warnings


def _expr_dependencies(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, GPExpr):
        deps = set()
        for item in value.args:
            deps.update(_expr_dependencies(item))
        for item in value.kwargs.values():
            deps.update(_expr_dependencies(item))
        return deps
    if isinstance(value, (list, tuple)):
        return set().union(*(_expr_dependencies(item) for item in value)) if value else set()
    return set()


def _safe_builtins() -> dict[str, object]:
    return {name: getattr(builtins, name) for name in _PURE_BUILTINS}


def _ref(value: object, label: str) -> GPRef:
    if isinstance(value, GPRef):
        return value
    raise GroundingFailure(f"INVALID_TYPE: {label} must be a map reference.")


def _ref_list(value: object, label: str) -> list[GPRef]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, GPRef)):
        refs = list(value)
        if all(isinstance(item, GPRef) for item in refs):
            return refs  # type: ignore[return-value]
    raise GroundingFailure(f"INVALID_TYPE: {label} must be a sequence of map references.")


def _bbox(ref: GPRef) -> tuple[int, int, int, int]:
    if not ref.cells:
        raise GroundingFailure("EMPTY_REFERENCE: bounding box requires map cells.")
    rows = [cell[0] for cell in ref.cells]
    cols = [cell[1] for cell in ref.cells]
    return min(rows), min(cols), max(rows), max(cols)
