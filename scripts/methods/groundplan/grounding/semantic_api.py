"""Single registry for GroundPlan's planner-free semantic-map API.

The executable-code sandbox and the native Gemini function-calling adapter use
the names in this registry.  Implementations remain the existing callables in
``GroundingRuntime``; tool calling only translates JSON handles to their
Python values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

OPERATOR_NAMES = {
    "target_of": "target_of", "entities": "entities", "rooms": "rooms", "in": "in_room",
    "room_of": "room_of", "contains": "contains", "count_next_to": "count_next_to",
    "count_near": "count_near", "adjacent_rooms": "adjacent_rooms", "passage_regions": "passage_regions",
    "union": "union", "intersection": "intersection", "exclude": "exclude", "count": "count",
    "unique": "unique", "choose_any": "choose_any", "kth_nearest": "kth_nearest",
    "kth_farthest": "kth_farthest", "kth_largest": "kth_largest", "kth_smallest": "kth_smallest",
    "order_by_distance": "order_by_distance", "closest_pair_member": "closest_pair_member",
    "region_of": "region_of", "room_region": "room_region", "midpoint_region": "midpoint_region",
    "between_region": "between_region", "near_region": "near_region", "side_region": "side_region",
    "boundary_region": "boundary_region", "half_room": "half_room", "relative_waypoint": "relative_waypoint",
    "circle": "circle", "follow_wall": "follow_wall",
}
PREDICATE_NAMES = frozenset({"near_to", "far_from", "next_to", "on_top_of", "in_corner", "between"})
CONSTRAINT_KINDS = frozenset({
    "require_visit", "require_visit_in_order", "forbid", "prefer_near",
    "prefer_far", "prefer_relative", "prefer_path_shape",
})


@dataclass(frozen=True)
class SemanticAPISpec:
    name: str
    description: str
    parameters: Mapping[str, object]

    def function_declaration(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


def _object(properties: Mapping[str, object], required: tuple[str, ...] = ()) -> dict[str, object]:
    schema: dict[str, object] = {"type": "OBJECT", "properties": dict(properties)}
    if required:
        schema["required"] = list(required)
    return schema


STRING = {"type": "STRING"}
NUMBER = {"type": "NUMBER"}
INTEGER = {"type": "INTEGER"}
BOOLEAN = {"type": "BOOLEAN"}
HANDLE = {"type": "STRING", "description": "Opaque handle returned by a previous semantic API call."}
HANDLES = {"type": "ARRAY", "items": HANDLE}
STRINGS = {"type": "ARRAY", "items": STRING}


def _spec(name: str, description: str, properties: Mapping[str, object], *required: str) -> SemanticAPISpec:
    return SemanticAPISpec(name, description, _object(properties, tuple(required)))


SEMANTIC_API_REGISTRY: tuple[SemanticAPISpec, ...] = (
    _spec("entities", "Return all entity references in a category.", {"category": STRING}, "category"),
    _spec("rooms", "Return all room references in a category.", {"category": STRING}, "category"),
    _spec("target_of", "Return a previously resolved segment target.", {"segment": HANDLE}, "segment"),
    _spec("in_room", "Filter entity references to one or more rooms.", {"entities": HANDLE, "rooms": HANDLE}, "entities", "rooms"),
    _spec("room_of", "Return the room containing a reference.", {"ref": HANDLE}, "ref"),
    _spec("contains", "Test whether a room contains an entity category.", {"room": HANDLE, "category": STRING}, "room", "category"),
    _spec("count_next_to", "Count category instances next to a reference.", {"category": STRING, "reference": HANDLE, "threshold": NUMBER}, "category", "reference"),
    _spec("count_near", "Count category instances near a reference.", {"category": STRING, "reference": HANDLE, "radius": NUMBER}, "category", "reference"),
    _spec("adjacent_rooms", "Return rooms adjacent to a room.", {"room": HANDLE}, "room"),
    _spec("passage_regions", "Return passage regions between two rooms.", {"first_room": HANDLE, "second_room": HANDLE}, "first_room", "second_room"),
    _spec("union", "Union reference sets.", {"sets": HANDLES}, "sets"),
    _spec("intersection", "Intersect reference sets.", {"sets": HANDLES}, "sets"),
    _spec("exclude", "Remove excluded references from candidates.", {"candidates": HANDLE, "excluded": HANDLE}, "candidates", "excluded"),
    _spec("count", "Count references in a set.", {"candidates": HANDLE}, "candidates"),
    _spec("unique", "Require and return the sole candidate.", {"candidates": HANDLE}, "candidates"),
    _spec("choose_any", "Deterministically choose the nearest candidate.", {"candidates": HANDLE, "reference": HANDLE}, "candidates"),
    _spec("kth_nearest", "Return the k-th nearest candidate.", {"candidates": HANDLE, "reference": HANDLE, "k": INTEGER, "metric": STRING}, "candidates", "reference"),
    _spec("kth_farthest", "Return the k-th farthest candidate.", {"candidates": HANDLE, "reference": HANDLE, "k": INTEGER, "metric": STRING}, "candidates", "reference"),
    _spec("kth_largest", "Return the k-th largest candidate by footprint.", {"candidates": HANDLE, "k": INTEGER}, "candidates"),
    _spec("kth_smallest", "Return the k-th smallest candidate by footprint.", {"candidates": HANDLE, "k": INTEGER}, "candidates"),
    _spec("order_by_distance", "Order candidates by distance.", {"candidates": HANDLE, "reference": HANDLE, "order": STRING}, "candidates", "reference"),
    _spec("closest_pair_member", "Return the candidate closest to any reference.", {"candidates": HANDLE, "references": HANDLE}, "candidates", "references"),
    _spec("region_of", "Construct the region occupied by an entity.", {"entity": HANDLE}, "entity"),
    _spec("room_region", "Construct the full region of a room.", {"room": HANDLE}, "room"),
    _spec("midpoint_region", "Construct a midpoint region between references.", {"first": HANDLE, "second": HANDLE}, "first", "second"),
    _spec("between_region", "Construct the region between references.", {"first": HANDLE, "second": HANDLE}, "first", "second"),
    _spec("near_region", "Construct a region near a reference.", {"reference": HANDLE, "radius": NUMBER}, "reference"),
    _spec("side_region", "Construct the side of one reference relative to another.", {"reference": HANDLE, "relative_to": HANDLE}, "reference", "relative_to"),
    _spec("boundary_region", "Construct a room boundary region.", {"room": HANDLE}, "room"),
    _spec("half_room", "Construct the half of a room relative to a reference.", {"room": HANDLE, "relative_to": HANDLE}, "room", "relative_to"),
    _spec("relative_waypoint", "Construct an anchor-relative waypoint region.", {"first_anchor": HANDLE, "second_anchor": HANDLE, "along": NUMBER, "lateral": NUMBER, "radius": NUMBER, "room": HANDLE}, "first_anchor", "second_anchor"),
    _spec("circle", "Construct ordered regions for a circular path shape.", {"reference": HANDLE, "fraction": NUMBER, "direction": STRING, "start_toward": HANDLE}, "reference"),
    _spec("follow_wall", "Construct ordered regions for a wall-following path shape.", {"reference": HANDLE, "fraction": NUMBER, "direction": STRING, "first_toward": HANDLE}, "reference"),
    _spec("position", "Return a reference center for comparison only.", {"ref": HANDLE}, "ref"),
    _spec("distance", "Return Euclidean map distance between references.", {"first": HANDLE, "second": HANDLE}, "first", "second"),
    _spec("geodesic_distance", "Return geodesic map distance between references.", {"first": HANDLE, "second": HANDLE}, "first", "second"),
    _spec("inside", "Test whether a reference lies inside another.", {"child": HANDLE, "container": HANDLE}, "child", "container"),
    _spec("intersects", "Test whether two references intersect.", {"first": HANDLE, "second": HANDLE}, "first", "second"),
    _spec("object_bbox", "Return an entity bounding box for comparison only.", {"ref": HANDLE}, "ref"),
    _spec("room_polygon", "Return a room polygon for comparison only.", {"room": HANDLE}, "room"),
    _spec("where", "Filter candidates with one fixed semantic predicate.", {"candidates": HANDLE, "predicate": {"type": "STRING", "enum": sorted(PREDICATE_NAMES)}, "arguments": HANDLES}, "candidates", "predicate"),
    _spec("compare", "Compare two prior scalar results or literals.", {"left": STRING, "operator": STRING, "right": STRING}, "left", "operator", "right"),
    _spec("all_of", "Logical conjunction of prior boolean results.", {"values": HANDLES}, "values"),
    _spec("any_of", "Logical disjunction of prior boolean results.", {"values": HANDLES}, "values"),
    _spec("not_", "Logical negation of a prior boolean result.", {"value": HANDLE}, "value"),
    _spec("near_to", "Test whether a candidate is near a reference.", {"candidate": HANDLE, "reference": HANDLE}, "candidate", "reference"),
    _spec("far_from", "Test whether a candidate is far from a reference.", {"candidate": HANDLE, "reference": HANDLE}, "candidate", "reference"),
    _spec("next_to", "Test whether a candidate is next to a reference.", {"candidate": HANDLE, "reference": HANDLE}, "candidate", "reference"),
    _spec("on_top_of", "Test whether a candidate is on top of a reference.", {"candidate": HANDLE, "reference": HANDLE}, "candidate", "reference"),
    _spec("in_corner", "Test whether a candidate is in a room corner.", {"candidate": HANDLE, "room": HANDLE}, "candidate"),
    _spec("between", "Test whether a candidate is between two references.", {"candidate": HANDLE, "first": HANDLE, "second": HANDLE}, "candidate", "first", "second"),
    _spec("set_segment_context", "Set the segment start before constructing its constraints.", {"segment_id": STRING, "start": HANDLE}, "segment_id", "start"),
    _spec("constraint", "Construct one RouteIR constraint using resolved references.", {"kind": {"type": "STRING", "enum": sorted(CONSTRAINT_KINDS)}, "references": HANDLES, "relation": STRING, "spatial_scope": HANDLE, "segment_scope": {"type": "ARRAY", "items": STRING}, "global_scope": BOOLEAN, "source_text": STRING}, "kind", "references"),
    _spec("segment", "Construct one ordered RouteIR segment.", {"segment_id": STRING, "start": HANDLE, "target": HANDLE, "constraints": HANDLES}, "segment_id", "start", "target"),
    _spec("task", "Submit the final RouteIR program. Call exactly once grounding is complete.", {"segments": HANDLES, "bindings": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"name": STRING, "value": HANDLE}, "required": ["name", "value"]}}}, "segments"),
)

SEMANTIC_API_NAMES = frozenset(spec.name for spec in SEMANTIC_API_REGISTRY)


def gemini_function_declarations() -> list[dict[str, object]]:
    return [spec.function_declaration() for spec in SEMANTIC_API_REGISTRY]
