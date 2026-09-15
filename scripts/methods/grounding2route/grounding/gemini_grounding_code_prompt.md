# Grounding2Route executable grounding-code contract

Write **only one complete Python program**. Do not use Markdown fences. The
program is executed in a restricted, read-only semantic-map sandbox. Its only
job is to resolve map references, regions, constraints, and ordered segments;
it cannot plan a path.

## Required program shape

The program must contain exactly one entry point:

```python
def ground():
    ...
    return task(segments, bindings)
```

`ground()` has **no parameters**. Two map references already exist inside it:

```python
start_position  # a PositionRef for the agent's actual start cell
task_start      # alias of start_position
```

Never redefine either name. In particular, do **not** write
`position("agent")`, `position("robot")`, `entities("agent")`, or synthesize
an agent object: no such reference exists. Use `start_position` directly as a
segment start or distance/ranking reference.

The final return must look like:

```python
return task([s1, s2], {"target_a": target_a, "target_b": target_b})
```

When JSON IR and a compiled scaffold are supplied, preserve every original
binding name, segment ID/order, target, and constraint kind unless the supplied
execution feedback specifically proves that local expression is invalid.

## Value types

`Ref` means an opaque semantic-map object, room, region, or position reference.
Treat a `Ref` as opaque: never inspect attributes, IDs, or coordinates.

```text
EntitySet / RoomSet = list[Ref]
Position            = numeric (row, col) tuple, NOT a Ref
```

Most grounding APIs take and return `Ref` or `list[Ref]`. `position(ref)`
returns a numeric `Position`, so its output must **never** be passed to
`kth_nearest`, `kth_farthest`, `segment`, `set_segment_context`, `room_of`, or
`constraint`. Those functions require a `Ref`.

## Fixed semantic-map APIs

These APIs expose environment facts. Their meanings cannot be changed or
redefined.

```python
entities(category: str) -> list[Ref]
rooms(category: str) -> list[Ref]
position(ref: Ref) -> tuple[float, float]
distance(first: Ref, second: Ref) -> float
geodesic_distance(first: Ref, second: Ref) -> float
room_of(ref: Ref) -> Ref                 # returns the containing room
inside(child: Ref, container: Ref) -> bool
intersects(first: Ref, second: Ref) -> bool
object_bbox(entity: Ref) -> tuple[int, int, int, int]
room_polygon(room: Ref) -> tuple[tuple[int, int], ...]
```

Only use a category from the supplied object/room category inventories. Do not
invent object IDs, room IDs, coordinates, categories, or references.

## Selection and relation APIs

```python
in_room(entity_set: list[Ref], room_or_rooms: Ref | list[Ref]) -> list[Ref]
where(candidates: list[Ref], predicate: lambda self: bool) -> list[Ref]
unique(candidates: list[Ref]) -> Ref       # valid only if exactly one candidate exists
choose_any(candidates: list[Ref], reference: Ref = start_position) -> Ref

kth_nearest(candidates: list[Ref], reference: Ref, *, k: int = 1,
            metric: str = "geodesic") -> Ref
kth_farthest(candidates: list[Ref], reference: Ref, *, k: int = 1,
             metric: str = "geodesic") -> Ref
order_by_distance(candidates: list[Ref], reference: Ref,
                  order: str = "ascending") -> list[Ref]
kth_largest(candidates: list[Ref], *, k: int = 1) -> Ref
kth_smallest(candidates: list[Ref], *, k: int = 1) -> Ref

union(*sets: list[Ref]) -> list[Ref]
intersection(*sets: list[Ref]) -> list[Ref]
exclude(candidates: list[Ref], excluded: list[Ref]) -> list[Ref]
count(candidates: list[Ref]) -> int
contains(room: Ref, category: str) -> bool
count_next_to(category: str, reference: Ref, threshold: float = ...) -> int
count_near(category: str, reference: Ref, radius: float = ...) -> int
adjacent_rooms(room: Ref) -> list[Ref]
passage_regions(first_room: Ref, second_room: Ref) -> list[Ref]
closest_pair_member(candidates: list[Ref], references: list[Ref]) -> Ref
```

Correct ranking examples:

```python
countertop = kth_nearest(entities("counter_top"), start_position, k=1)
window = kth_nearest(entities("window"), countertop, k=1)
chairs_here = in_room(entities("chair"), room_of(countertop))
```

Incorrect ranking examples:

```python
kth_nearest(entities("window"), position(countertop), k=1)  # Position is not Ref
kth_nearest(entities("window"), "countertop", k=1)          # string is not Ref
unique(entities("chair"))                                    # invalid unless known singleton
```

For plural sets, use `kth_nearest` or a filter; do not select an arbitrary
`items[0]`. A singular English noun phrase does not imply `unique(...)`.

## Region construction APIs

Every function below returns a `Ref` region (except `circle`/`follow_wall`,
which return one or more ordered regions suitable for a hard path-shape
constraint):

```python
region_of(entity: Ref) -> Ref
room_region(room: Ref) -> Ref
midpoint_region(first: Ref, second: Ref) -> Ref
between_region(first: Ref, second: Ref) -> Ref
near_region(reference: Ref, radius: float = ...) -> Ref
side_region(reference: Ref, relative_to: Ref) -> Ref
boundary_region(room: Ref) -> Ref
half_room(room: Ref, relative_to: Ref) -> Ref
relative_waypoint(first_anchor: Ref, second_anchor: Ref,
                  along: float = 0.5, lateral: float = 0.0,
                  radius: float = 0.20, room: Ref = None) -> Ref
circle(reference: Ref, fraction: float = 1.0,
       direction: str = "counterclockwise", start_toward: Ref = ...) -> list[Ref]
follow_wall(reference: Ref, fraction: float = 1.0,
            direction: str = "auto", first_toward: Ref = ...) -> list[Ref]
```

For an unregistered but instruction-grounded shape, a pure helper may return an ordered list of `relative_waypoint(...)` regions. `along` is measured on the directed first-to-second anchor axis (0 at the first anchor, 1 at the second); `lateral` is a signed meter offset on its left-hand normal. Use `require_visit_in_order(helper(...))` when the instruction requires the shape. Do not construct coordinates, cells, trajectories, or arbitrary regions directly.

## Constraint and task-builder APIs

Always call `set_segment_context` immediately before constructing constraints
for that segment. Preserve the exact relevant instruction clause in
`source_text`. Omit `segment_scope` to use the current segment; use
`segment_scope="global"` only for a globally active route constraint.

```python
set_segment_context(segment_id: str, start: Ref) -> None
constraint(kind: str, *references: Ref | list[Ref],
           spatial_scope: Ref | list[Ref] | None = None,
           segment_scope: str | list[str] | None = None,
           source_text: str | None = None) -> tuple[GroundedConstraint, ...]
segment(segment_id: str, start: Ref, target: Ref,
        constraints: list[tuple[GroundedConstraint, ...]] = []) -> GroundedSegment
task(segments: list[GroundedSegment], bindings: dict[str, object]) -> GroundedProgram
```

Allowed constraint forms:

```python
constraint("require_visit", region)
constraint("require_visit_in_order", follow_wall(target_room, fraction=1.0))
constraint("forbid", between_region(sofa, dresser))
constraint("prefer_near", window, spatial_scope=room_of(window))
constraint("prefer_far", entities("painting"))
constraint("prefer_relative", toilet, "farther_from", sink, spatial_scope=room_of(toilet))
constraint("prefer_path_shape", follow_wall(target_room, direction="clockwise"))
```

Complete minimal example:

```python
def ground():
    countertop = kth_nearest(entities("counter_top"), start_position, k=1)
    window = kth_nearest(entities("window"), countertop, k=1)
    set_segment_context("s1", start_position)
    s1 = segment(
        "s1", start_position, countertop,
        [constraint("prefer_near", window, spatial_scope=room_of(window))],
    )
    return task([s1], {"countertop": countertop, "window": window})
```

## Allowed Python subset and helper functions

If helpers are allowed, define them at the **top level**, alongside `ground`:

```python
def nearest_to(items, reference):
    return kth_nearest(items, reference, k=1)

def ground():
    target = nearest_to(entities("chair"), start_position)
    ...
```

Helpers must be pure grounding helpers that combine only the APIs above. Do
not define a helper inside `ground()`.

### Missing-relation protocol

Use the fixed API directly whenever it expresses the required grounding
relation. If no single fixed operator expresses a relation, you may define a
top-level helper that composes the exposed map queries, predicates, set
operations, distance tests, and region constructors. For example, a helper may
filter a candidate category by room membership and `near_to(...)`, or build an
ordered list of anchor-relative waypoint regions. A helper is not permission to
invent a new scene fact: it may not inspect IDs, access raw map cells, create
coordinates or arbitrary regions, or guess a candidate. If the required
relation cannot be derived from the exposed primitives, preserve the
instruction structure and let grounding fail with the reported unsupported or
cardinality error.

The accepted Python subset is:

- top-level `def` only; `ground()` plus optional top-level helpers;
- simple assignments, `return`, simple `if`, lambda, and list/set/generator
  comprehensions;
- numeric/boolean expressions and the listed pure builtins (`len`, `min`,
  `max`, `sum`, `sorted`, `range`, `enumerate`, `zip`, `abs`, `round`, `all`,
  `any`).

Never use `import`, `from ... import`, decorators, type annotations, classes,
attribute access, nested `def`, ordinary `for`/`while` statements, `try`, file
or network access, or APIs not listed here.

## Grounding/planning boundary

Never generate a trajectory and never call or mention planner, search, A*,
policy, cost, weight, or a planning method. The deterministic planner is not
available in this sandbox.

## Repair protocol

When **Structured execution feedback** is provided, it is a real error from the
semantic-map execution. Rewrite the complete program to fix that specific
error, then return code only.

1. Treat the error's required type/cardinality as authoritative.
2. Make the smallest change that fixes the local failure.
3. Preserve instruction semantics, JSON-IR bindings, segment order, and all
   constraint kinds/scopes unless feedback proves that structure wrong.
4. Do not replace a failed reference with `start_position`, `position(...)`, a
   fabricated agent, or an arbitrary first list element.
5. For `CARDINALITY_ERROR`, remove only the impossible `unique`/ordinal/filter
   assumption and use a deterministic symbolic ranking consistent with the
   instruction.
6. For `UNSAFE_*`, `INVALID_TOP_LEVEL`, or `FORBIDDEN_CALL`, use only the
   allowed Python/API subset above; do not import or bypass the sandbox.
