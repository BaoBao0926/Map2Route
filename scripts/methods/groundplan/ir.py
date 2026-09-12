"""Typed intermediate records for the GroundPlan method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


Cell = tuple[int, int]
JsonDict = dict[str, object]


@dataclass(frozen=True)
class GPRef:
    kind: Literal["position", "entity", "room", "region"]
    id: str
    category: str | None = None
    room_id: str | None = None
    center: tuple[float, float] | None = None
    cells: frozenset[Cell] = frozenset()
    construction: Mapping[str, object] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        payload: JsonDict = {"kind": self.kind, "id": self.id}
        if self.category:
            payload["category"] = self.category
        if self.room_id:
            payload["room_id"] = self.room_id
        if self.center is not None:
            payload["center"] = [self.center[0], self.center[1]]
        if self.cells:
            payload["cell_count"] = len(self.cells)
            rows = [row for row, _col in self.cells]
            cols = [col for _row, col in self.cells]
            payload["bbox"] = [min(rows), min(cols), max(rows), max(cols)]
        if self.construction:
            payload["construction"] = dict(self.construction)
        return payload


@dataclass(frozen=True)
class GPExpr:
    op: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> object:
        return {
            "op": self.op,
            "args": [_json_ready(arg) for arg in self.args],
            "kwargs": {key: _json_ready(value) for key, value in self.kwargs.items()},
        }


@dataclass(frozen=True)
class GPBinding:
    name: str
    expr: Any

    def to_json(self) -> JsonDict:
        return {"name": self.name, "expr": _json_ready(self.expr)}


@dataclass(frozen=True)
class GPConstraintSpec:
    kind: Literal[
        "require_visit",
        "require_visit_in_order",
        "forbid",
        "prefer_near",
        "prefer_far",
        "prefer_relative",
        "prefer_path_shape",
    ]
    exprs: tuple[Any, ...]
    segment_scope: tuple[str, ...] = ()
    spatial_scope: Any | None = None
    source_text: str | None = None

    def to_json(self) -> JsonDict:
        payload: JsonDict = {
            "kind": self.kind,
            "exprs": [_json_ready(expr) for expr in self.exprs],
            "segment_scope": list(self.segment_scope),
        }
        if self.spatial_scope is not None:
            payload["spatial_scope"] = _json_ready(self.spatial_scope)
        if self.source_text:
            payload["source_text"] = self.source_text
        return payload


@dataclass(frozen=True)
class GPSegmentSpec:
    id: str
    start: Any
    target: Any
    constraints: tuple[GPConstraintSpec, ...] = ()

    def to_json(self) -> JsonDict:
        return {
            "id": self.id,
            "start": _json_ready(self.start),
            "target": _json_ready(self.target),
            "constraints": [constraint.to_json() for constraint in self.constraints],
        }


@dataclass(frozen=True)
class GPProgramSpec:
    bindings: tuple[GPBinding, ...]
    segments: tuple[GPSegmentSpec, ...]
    source: str
    parser_mode: str
    diagnostics: tuple[str, ...] = ()

    def to_json(self) -> JsonDict:
        return {
            "parser_mode": self.parser_mode,
            "source": self.source,
            "bindings": [binding.to_json() for binding in self.bindings],
            "segments": [segment.to_json() for segment in self.segments],
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class GroundedConstraint:
    """Canonical, planner-independent grounded route constraint.

    ``kind`` and ``refs`` retain the existing planner contract.  The remaining
    fields make the Direct-CaP output independently inspectable without
    changing how the fixed planner consumes it.
    """
    kind: Literal[
        "require_visit",
        "require_visit_in_order",
        "forbid",
        "prefer_near",
        "prefer_far",
        "prefer_relative",
        "prefer_path_shape",
    ]
    refs: tuple[GPRef, ...]
    segment_scope: tuple[str, ...] = ()
    spatial_scope: GPRef | tuple[GPRef, ...] | None = None
    relation: str | None = None
    constraint_id: str = ""
    argument_roles: tuple[str, ...] = ()
    hardness: Literal["hard", "soft"] | None = None
    source_text: str | None = None

    def to_json(self) -> JsonDict:
        spatial_scope: object = None
        if self.spatial_scope is not None:
            if isinstance(self.spatial_scope, tuple):
                spatial_scope = [ref.to_json() for ref in self.spatial_scope]
            else:
                spatial_scope = self.spatial_scope.to_json()
        payload: JsonDict = {
            "constraint_id": self.constraint_id,
            "kind": self.kind,
            "refs": [ref.to_json() for ref in self.refs],
            "argument_roles": list(self.argument_roles),
            "hardness": self.hardness,
            "segment_scope": list(self.segment_scope),
            "spatial_scope": spatial_scope,
            "scope": {
                "segment_scope": list(self.segment_scope),
                "spatial_scope": spatial_scope,
                "is_global": not self.segment_scope and self.spatial_scope is None,
            },
        }
        if self.relation:
            payload["relation"] = self.relation
        if self.source_text:
            payload["source_text"] = self.source_text
        return payload


@dataclass(frozen=True)
class GroundedSegment:
    id: str
    start_reference: GPRef
    target: GPRef
    constraints: tuple[GroundedConstraint, ...]

    def to_json(self) -> JsonDict:
        return {
            "id": self.id,
            "segment_id": self.id,
            "start_reference": self.start_reference.to_json(),
            "action": {"type": "go_to", "target": self.target.to_json()},
            "constraints": [constraint.to_json() for constraint in self.constraints],
        }


@dataclass(frozen=True)
class GroundedProgram:
    segments: tuple[GroundedSegment, ...]
    bindings: Mapping[str, object]
    status: str = "success"
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def to_json(self) -> JsonDict:
        return {
            "schema_version": 2,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "segments": [segment.to_json() for segment in self.segments],
            "bindings": {key: _json_ready(value) for key, value in self.bindings.items()},
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class PlannedSegment:
    id: str
    path: list[list[int]]
    endpoint: Cell | None
    status: str
    path_length: float
    expanded_states: int
    details: Mapping[str, object] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        return {
            "id": self.id,
            "status": self.status,
            "endpoint": [self.endpoint[0], self.endpoint[1]] if self.endpoint else None,
            "waypoint_count": len(self.path),
            "path_length": self.path_length,
            "expanded_states": self.expanded_states,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PlannedProgram:
    trajectory: list[list[int]]
    segments: tuple[PlannedSegment, ...]
    status: str
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        return {
            "status": self.status,
            "failure_reason": self.failure_reason,
            "trajectory_length": len(self.trajectory),
            "segments": [segment.to_json() for segment in self.segments],
            "diagnostics": list(self.diagnostics),
            "details": dict(self.details),
        }


def _json_ready(value: object) -> object:
    if isinstance(value, GPExpr):
        return value.to_json()
    if isinstance(value, GPRef):
        return value.to_json()
    if isinstance(value, GPBinding):
        return value.to_json()
    if isinstance(value, GPConstraintSpec):
        return value.to_json()
    if isinstance(value, GPSegmentSpec):
        return value.to_json()
    if isinstance(value, GPProgramSpec):
        return value.to_json()
    if isinstance(value, GroundedConstraint):
        return value.to_json()
    if isinstance(value, GroundedSegment):
        return value.to_json()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, frozenset):
        return [list(cell) for cell in sorted(value)]
    return value


def path_length(path: Sequence[Sequence[int]]) -> float:
    import math

    total = 0.0
    for previous, current in zip(path, path[1:]):
        total += math.hypot(int(current[0]) - int(previous[0]), int(current[1]) - int(previous[1]))
    return total
