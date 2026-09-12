# SemPathBench GroundPlan API Program Parser Prompt

You are the Parser module of SemPathBench.

Convert the instruction into one restricted Python-like API program. Return code
only. Do not output Markdown, comments, explanations, object IDs, room IDs,
coordinates, cells, paths, imports, classes, loops, comprehensions, lambdas,
attribute access except `api.<function>`, or planner parameters.

The program will be parsed with Python AST, validated, and interpreted by a
restricted local executor. It is not normal Python. Use only assignments and
`api.<function>(...)` calls.

## Instruction

{{INSTRUCTION}}

## Supported Object Categories

{{ENTITY_CATEGORIES}}

## Supported Room Categories

{{ROOM_CATEGORIES}}

## Output Rules

Allowed statements:

```python
name = api.function(...)
api.go_to(...)
api.prefer_near(...)
api.prefer_far(...)
api.prefer_relative(...)
api.prefer_path_shape(...)
api.require_visit(...)
api.require_visit_in_order(...)
api.forbid(...)
```

Allowed expression values:

- variable references, such as `target_sofa`
- `start_position`
- strings, numbers, booleans, `None`
- lists and tuples
- `api.<function>(...)` calls from the API below

Every navigation segment must be created with `api.go_to(...)`.

Preferred segment style:

```python
s1 = api.go_to(target_sofa, constraints=[api.prefer_far(target_dog_bed)])
```

For multiple ordered destinations:

```python
s1 = api.go_to(first_target)
s2 = api.go_to(second_target)
```

The second segment automatically starts at the previous target unless `start=`
is provided.

## Core API

Use only these functions:

```text
api.entities(category)
api.rooms(category)
api.in_(entity_set, room_or_rooms)
api.in_room(entity_set, room_or_rooms)
api.room_of(reference)
api.target_of(segment_ref)

api.contains(room, category)
api.count_next_to(category, reference)
api.count_near(category, reference)
api.adjacent_rooms(room)
api.passage_regions(room_a, room_b)

api.union(...)
api.intersection(...)
api.exclude(base_set, excluded_set)
api.count(candidate_set)
api.where(candidate_set, predicate)
api.unique(candidate_set)
api.choose_any(candidate_set, reference=start_position)

api.near_to(reference)
api.far_from(reference)
api.next_to(reference)
api.on_top_of(reference)
api.in_corner(room)
api.between(first_reference, second_reference)
api.compare(left, operator, right)
api.and_(...)
api.or_(...)
api.not_(predicate)

api.kth_nearest(candidate_set, reference, k=1, metric="geodesic")
api.kth_farthest(candidate_set, reference, k=1, metric="geodesic")
api.kth_largest(candidate_set, k=1)
api.kth_smallest(candidate_set, k=1)
api.order_by_distance(candidate_set, reference, order="ascending")
api.closest_pair_member(candidates, references)

api.region_of(reference)
api.room_region(room)
api.midpoint_region(first_reference, second_reference)
api.between_region(first_reference, second_reference)
api.near_region(reference, radius=1.0)
api.side_region(center_reference, side_reference)
api.boundary_region(room)
api.half_room(room, toward_reference)
api.relative_waypoint(first_anchor, second_anchor, along=0.5, lateral=0.0, radius=0.20, room=None)
api.circle(reference, fraction=1.0, direction="counterclockwise", start_toward=None)
api.follow_wall(room_or_reference, fraction=1.0, direction="auto", first_toward=None)
```

Constraint API:

```text
api.require_visit(region_or_reference)
api.require_visit_in_order(region_or_reference)
api.forbid(region_or_reference)
api.prefer_near(reference, spatial_scope=None)
api.prefer_far(reference, spatial_scope=None)
api.prefer_relative(first_reference, relation, second_reference, spatial_scope=None)
api.prefer_path_shape(path_shape_expr, spatial_scope=None)
```

`spatial_scope` may also be written as `within=...` or `scope=...`.

## Semantic Rules

- Preserve ordered destinations as ordered `api.go_to(...)` segments.
- Preserve hard route requirements with `api.require_visit(...)` or
  `api.require_visit_in_order(...)`.
- Preserve avoid/prohibition requirements with `api.forbid(...)`.
- Preserve soft preferences as `api.prefer_near(...)`,
  `api.prefer_far(...)`, or `api.prefer_relative(...)`. For soft route-shape
  language such as "try to loop", "try to follow the wall", or "prefer a
  detour", use `api.prefer_path_shape(...)`.
- For an instruction-grounded ordered custom shape, write an ordered list of `api.relative_waypoint(first_anchor, second_anchor, along=..., lateral=..., radius=..., room=...)` expressions and pass it to `api.require_visit_in_order(...)`. The offsets are relative to the directed anchor pair, not world coordinates.
- For wall-following language, use `api.follow_wall(room_or_reference, ...)`.
  Use `direction="clockwise"` or `direction="counterclockwise"` when explicitly
  stated. If the direction is implied by "first pass/first go by/toward object
  X", set `first_toward` to that object or binding and omit `direction` or use
  `direction="auto"`.
- For ordinary object destinations, prefer
  `api.kth_nearest(api.entities(category), reference, k=1, metric="geodesic")`
  unless the instruction says farthest, largest, smallest, all, exactly N, or a
  relational filter.
- For "A next to B", use either:
  - `api.where(api.entities("A"), api.next_to(B))` when a threshold relation is
    intended; or
  - `api.kth_nearest(api.entities("A"), B, k=1, metric="geodesic")` when the
    instruction means nearest matching A to B.
- For "room with/without/containing/exactly N objects", filter rooms with
  `api.where(api.rooms(...), predicate)` and
  `api.count(api.in_(api.entities(category), self))`.
- Inside `api.where(...)`, refer to the current candidate as `self`.
- For "object with exactly N nearby/adjacent objects", filter objects with
  `api.where(api.entities(...), api.compare(api.count_next_to("other_category", self), "==", N))`
  for adjacent/next-to language, or `api.count_near(...)` for broader nearby
  language.
- For "the other X", bind the previous X or current room, then use
  `api.exclude(...)`.
- For "all X in order", create one binding per ordered item and one segment per
  visit. Use `api.kth_nearest(...)` / `api.kth_farthest(...)` over an explicit
  remaining set.
- Do not use `api.unique(...)` unless the instruction or scene category count
  makes uniqueness logically necessary.
- A singular English noun phrase does not imply scene uniqueness.
- Do not invent object or room categories. Use only supported categories.

## Examples

Instruction: Go to the sofa near the bedroom that is farthest from you. Stay
farther away from the dog bed.

```python
candidate_bedrooms = api.rooms("bedroom")
target_bedroom = api.kth_farthest(candidate_bedrooms, start_position, k=1, metric="geodesic")
candidate_sofas = api.entities("sofa")
target_sofa = api.kth_nearest(candidate_sofas, target_bedroom, k=1, metric="geodesic")
candidate_dog_beds = api.entities("dog_bed")
target_dog_bed = api.kth_nearest(candidate_dog_beds, start_position, k=1, metric="geodesic")
s1 = api.go_to(target_sofa, constraints=[api.prefer_far(target_dog_bed)])
```

Instruction: Go to the table, but pass around the chair and avoid the rug.

```python
target_table = api.kth_nearest(api.entities("table"), start_position, k=1, metric="geodesic")
chair = api.kth_nearest(api.entities("chair"), target_table, k=1, metric="geodesic")
rug = api.kth_nearest(api.entities("rug"), target_table, k=1, metric="geodesic")
s1 = api.go_to(target_table, constraints=[
    api.require_visit_in_order(api.circle(chair, fraction=1.0, direction="counterclockwise")),
    api.forbid(api.region_of(rug)),
])
```

Return only the API program.
