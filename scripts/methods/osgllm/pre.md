# Adapting Optimal Scene Graph Planning with LLM Guidance to SemPathBench

## 1. Goal

This document describes how to adapt **Optimal Scene Graph Planning with Large Language Model Guidance** to the SemPathBench interface while preserving the original method as faithfully as possible.

The adapted method is referred to as:

```text
Optimal Scene Graph Planning
```

or, for internal code naming only:

```text
osg_llm
```

`OSG-LLM` is not an official acronym from the original paper, so the paper and result tables should preferably use the full method name.

Original resources:

- Paper: https://arxiv.org/abs/2309.09182
- Official repository: https://github.com/ExistentialRobotics/LLM-Scene-Graph-LTL-Planning

The target pipeline is:

```text
SemPathBench map and instruction
        |
        v
Hierarchical scene graph construction
        |
        v
Scene-graph-based entity grounding
        |
        v
Grounded natural-language mission
        |
        v
LTL translation and validation
        |
        v
LTL automaton
        |
        v
Hierarchical scene-graph × automaton planning
        |
        v
Occupancy-grid trajectory
        |
        v
Shared SemPathBench evaluator
```

The objective is **not** to redesign the original method into a new SemPathBench-specific planner. Changes should be restricted to:

1. converting SemPathBench maps into the scene-graph representation expected by the method;
2. adapting the input/output interface;
3. resolving SemPathBench-specific referring expressions that cannot be grounded from the original attribute hierarchy alone;
4. handling unsupported instruction components explicitly and transparently.

---

## 2. Original Method Summary

The original method solves the following problem:

```text
known hierarchical scene graph
+
initial robot location
+
natural-language mission
+
edge/path cost
        |
        v
minimum-cost path satisfying the mission
```

The environment is represented hierarchically using levels such as:

```text
floor
  └── room
        └── object
              └── occupancy/free-space nodes
```

The method contains four important components.

### 2.1 Scene-graph attribute grounding

The scene graph contains uniquely identified entities such as:

```text
floor_0
room_bedroom_3
object_sofa_12
object_bed_17
```

The LLM receives a compact attribute hierarchy and rewrites ambiguous entity references into references to unique scene-graph entities.

Example:

```text
Original:
Go to the sofa in the bedroom containing a desk.

Grounded:
Go to object_sofa_12 in room_bedroom_3.
```

### 2.2 Natural language to LTL

The grounded mission is translated into a co-safe LTL formula.

Example:

```text
Visit bedroom_3, then reach sofa_12, while avoiding bathroom_2.
```

may become conceptually:

```text
F(bedroom_3 AND F(sofa_12)) AND G(NOT bathroom_2)
```

The implementation validates the generated formula and retries or repairs invalid translations.

### 2.3 Product planning over scene graph and automaton

The LTL formula is converted into an automaton. Planning states combine:

```text
scene-graph state
×
LTL automaton state
```

A path is successful only when its automaton state reaches an accepting state.

### 2.4 Hierarchical multi-heuristic planning

The method constructs multiple planning resolutions, such as occupancy, object, room, and floor levels.

It combines:

- an LTL-aware anchor heuristic;
- an LLM-generated semantic heuristic;
- hierarchical multi-resolution search;
- AMRA*-style anytime planning.

The LLM heuristic accelerates search, while the formal anchor heuristic is responsible for the method's search guarantees.

---

## 3. SemPathBench Interface

A SemPathBench episode provides approximately:

```text
instruction
start pose
occupancy/traversibility map
semantic object-instance map
room-instance map
object and room metadata
```

The required output is:

```text
trajectory = [(x0, y0), (x1, y1), ..., (xT, yT)]
```

The instruction may contain:

- target object or room constraints;
- ordered subgoals;
- required visits;
- forbidden regions;
- nearest or farthest references;
- containment queries;
- relative-distance relations;
- near/far preferences;
- between relations;
- path-shape constraints;
- hard constraints;
- soft constraints.

The shared evaluator computes metrics such as:

```text
PLR
HCS
SCS
OSS
```

---

## 4. Compatibility Assessment

### 4.1 Directly compatible components

The following SemPathBench components are strongly compatible with the original method:

| SemPathBench component | Original method counterpart |
|---|---|
| Complete known environment | Complete known scene graph |
| Occupancy grid | Occupancy/free-space graph |
| Room-instance map | Room nodes |
| Object-instance map | Object nodes |
| Start pose | Initial occupancy node |
| Natural-language instruction | Natural-language mission |
| Ordered hard constraints | LTL temporal ordering |
| Required room/object visits | Atomic propositions |
| Hard avoidance | LTL negation/safety condition |
| Grid trajectory | Occupancy-level path |

### 4.2 Partially compatible components

The following require an adapter:

| SemPathBench component | Main issue |
|---|---|
| Nearest/farthest entity | Requires metric computation |
| Room without an object | Requires set filtering or explicit negative attributes |
| Relative-distance relation | Requires deterministic geometric evaluation |
| Midpoint/between region | Requires construction of a virtual proposition region |
| Pass between two objects | Requires a geometric crossing region |
| Circle around an object/room | Requires trajectory-level event logic |
| Follow a wall | Requires a path-shape representation |
| Soft near/far constraints | Cannot be represented naturally as Boolean LTL |
| Human-like path preference | Not part of the original method |

The adapter must not silently pretend these capabilities are native to the original method.

---

## 5. Proposed Repository Structure

A clean implementation could follow:

```text
scripts/methods/osg_llm/
├── run.py
├── config.yaml
├── README.md
├── adapter.py
├── scene_graph/
│   ├── build_scene_graph.py
│   ├── graph_types.py
│   ├── occupancy_graph.py
│   ├── room_graph.py
│   ├── object_graph.py
│   └── serialize_hierarchy.py
├── grounding/
│   ├── entity_grounder.py
│   ├── geometric_resolver.py
│   ├── relation_resolver.py
│   └── grounded_mission.py
├── ltl/
│   ├── prompt_builder.py
│   ├── translator.py
│   ├── validator.py
│   └── automaton.py
├── planning/
│   ├── domain_builder.py
│   ├── ltl_heuristic.py
│   ├── llm_heuristic.py
│   ├── hierarchical_planner.py
│   └── path_projection.py
└── outputs/
```

Whenever practical, original repository code should be imported or minimally wrapped rather than rewritten.

---

## 6. Step 1: Build a Hierarchical Scene Graph

### 6.1 Proposed hierarchy

SemPathBench is usually a single-floor 2D environment. Represent it as:

```text
building_0
  └── floor_0
        ├── room_0
        │     ├── object_0
        │     ├── object_1
        │     └── occupancy nodes
        ├── room_1
        │     ├── object_2
        │     └── occupancy nodes
        └── shared/free-space occupancy nodes
```

The floor layer should remain even when there is only one floor. Removing it would unnecessarily change the hierarchy expected by the original method.

### 6.2 Occupancy nodes

Convert every traversable grid cell into an occupancy node:

```python
OccupancyNode(
    node_id="cell_120_85",
    row=120,
    col=85,
    traversable=True,
    room_id="room_bedroom_2",
)
```

Connect nodes using the same motion model used by SemPathBench:

- 4-connected movement, or
- 8-connected movement.

The edge cost must match the benchmark's path-length calculation:

```text
horizontal/vertical edge cost = 1
diagonal edge cost = sqrt(2)
```

Do not use a different connectivity model from the shared benchmark unless the method requires it and the difference is documented.

### 6.3 Room nodes

Create one node per room instance:

```python
RoomNode(
    node_id="room_bedroom_2",
    category="bedroom",
    instance_id=2,
    cells=[...],
    centroid=(x, y),
    entrances=[...],
    neighboring_rooms=[...],
)
```

Room adjacency should be derived from actual traversable connections, preferably through door or boundary-transition cells.

A room-level edge should exist only if a valid occupancy-level path crosses between the rooms.

### 6.4 Object nodes

Create one node per object instance:

```python
ObjectNode(
    node_id="object_sofa_7",
    category="sofa",
    instance_id=7,
    room_id="room_bedroom_2",
    mask_cells=[...],
    interaction_cells=[...],
    centroid=(x, y),
)
```

Because object cells may be obstacles, the proposition for “reach sofa” should normally be evaluated on a set of nearby traversable interaction cells rather than on the object's occupied pixels.

Define:

```text
reach_region(object_i)
=
traversable cells within an object-specific radius
```

The radius should be taken from the shared SemPathBench semantics rather than invented separately for this method.

### 6.5 Containment edges

Add explicit edges or attributes:

```text
object_sofa_7 --contained_in--> room_bedroom_2
room_bedroom_2 --on_floor--> floor_0
cell_120_85 --inside--> room_bedroom_2
```

These relations are critical for entity grounding.

### 6.6 Attribute hierarchy serialization

Serialize a compact hierarchy for the LLM:

```yaml
floor_0:
  rooms:
    room_bedroom_1:
      category: bedroom
      connected_rooms:
        - room_hallway_0
      objects:
        - object_bed_1
        - object_dresser_3

    room_bedroom_2:
      category: bedroom
      connected_rooms:
        - room_hallway_0
        - room_kitchen_0
      objects:
        - object_bed_4
        - object_sofa_7
        - object_desk_8
```

Avoid including every occupancy cell in the LLM prompt. Occupancy geometry belongs in the planner, not in the language grounding prompt.

---

## 7. Step 2: Define Atomic Propositions

Atomic propositions should correspond to grounded scene entities or explicitly constructed regions.

### 7.1 Room propositions

```text
in_room_bedroom_2
```

True when the current occupancy node belongs to `room_bedroom_2`.

### 7.2 Object propositions

```text
near_object_sofa_7
```

True when the current occupancy node is in the reach region of `object_sofa_7`.

### 7.3 Avoidance propositions

```text
inside_avoid_region_object_dog_bed_4
```

True inside the benchmark-defined forbidden or near-object region.

### 7.4 Virtual propositions

For SemPathBench-only spatial relations, a geometric adapter may create propositions such as:

```text
midpoint_table_3_dresser_5
between_table_3_fridge_2
doorway_room_1_room_2
wall_band_room_bedroom_2
```

Each virtual proposition must correspond to a deterministic set of occupancy cells.

The language model must never invent proposition names. It should select only from a supplied proposition inventory.

---

## 8. Step 3: Ground the Natural-Language Instruction

Grounding should use two stages.

```text
Stage A: original scene-graph entity grounding
Stage B: deterministic SemPathBench geometric resolution
```

### 8.1 Stage A: Scene-graph entity grounding

Use the original method's attribute hierarchy to resolve references based on:

- entity category;
- room/object containment;
- floor membership;
- room connectivity;
- unique IDs;
- explicit ordinal or instance information;
- contextual descriptions.

Examples:

```text
the bedroom containing the sofa
→ room_bedroom_2
```

```text
the sofa in the bedroom with a desk
→ object_sofa_7
```

```text
the kitchen connected to the bedroom
→ room_kitchen_0
```

The output should be structured rather than free text:

```json
{
  "entities": {
    "target_room": "room_bedroom_2",
    "target_object": "object_sofa_7"
  },
  "grounded_instruction": "Enter room_bedroom_2 and reach object_sofa_7."
}
```

### 8.2 Stage B: Deterministic geometric resolution

Use map computations for descriptions that cannot be reliably resolved from symbolic hierarchy alone.

#### Nearest entity

```text
the nearest sofa
```

Resolve with:

```text
argmin over candidate sofas of shortest-path distance
from the start position to the sofa reach region
```

#### Farthest entity

```text
the bedroom farthest from the start
```

Resolve with:

```text
argmax over candidate bedrooms of shortest-path distance
from the start position to the nearest valid entrance/cell of the room
```

Use shortest traversable path distance rather than Euclidean distance unless the annotation explicitly defines straight-line distance.

#### Entity without another entity

```text
the bedroom without a garbage can
```

Resolve by filtering room nodes:

```text
candidate rooms
=
all bedroom instances
-
bedrooms containing garbage_can instances
```

#### Relative distance

```text
the sofa farther from the bed than from the table
```

Evaluate each candidate using benchmark-defined distance functions and retain candidates satisfying the relation.

#### Between or midpoint region

```text
the midpoint between the dining table and dresser
```

Construct a virtual target region from the two grounded object instances.

The exact construction must match the benchmark annotation/evaluation semantics.

### 8.3 Ambiguous grounding

If multiple candidates remain after all applicable constraints:

1. do not silently choose the first instance;
2. use a deterministic benchmark-wide tie-breaking rule;
3. log the ambiguity and candidate set;
4. apply the same rule to every method that needs such grounding.

Possible tie-breaking order:

```text
shortest path distance from start
then smallest instance ID
```

However, if the benchmark's ground-truth annotation already identifies the intended entity, that hidden annotation must not be exposed to the method at inference time.

### 8.4 Grounding output schema

Recommended representation:

```json
{
  "original_instruction": "...",
  "grounded_instruction": "...",
  "entity_bindings": [
    {
      "text_span": "the bedroom containing the sofa",
      "entity_id": "room_bedroom_2",
      "resolver": "scene_graph_containment"
    }
  ],
  "virtual_regions": [],
  "unresolved_constraints": [],
  "unsupported_constraints": []
}
```

This log is necessary for debugging and qualitative analysis.

---

## 9. Step 4: Separate Supported and Unsupported Constraints

Before LTL generation, classify each instruction component.

### 9.1 Naturally supported hard constraints

Examples:

```text
visit room A
reach object B
visit A before B
visit A and then B
avoid room C
avoid object D
eventually reach A
visit one of A or B
```

These should be translated into LTL.

### 9.2 Supported after geometric grounding

Examples:

```text
visit the farthest bedroom
reach the sofa nearest the table
go to the bedroom without a garbage can
reach the midpoint between A and B
```

After grounding, these become ordinary entity/region propositions.

### 9.3 Not naturally supported by the original method

Examples:

```text
try to remain farther from the dog bed
prefer the wider passage
stay slightly closer to the sofa than the bed
follow the wall
move smoothly
produce a human-like path
```

These are graded path preferences rather than Boolean mission constraints.

The baseline should use one of the policies below.

#### Recommended faithful policy

```text
Hard constraints:
translated into LTL.

Soft constraints:
parsed and logged, but not used by the original planner.
```

This is the most faithful baseline and should be the default result reported as:

```text
Optimal Scene Graph Planning
```

#### Optional extended policy

```text
Hard constraints:
translated into LTL.

Soft constraints:
converted into additional occupancy-edge costs.
```

This should be reported separately as:

```text
Optimal Scene Graph Planning + Soft-Cost Adapter
```

Do not merge this extension into the main baseline without an ablation.

---

## 10. Step 5: Translate the Grounded Mission to LTL

### 10.1 Prompt input

The LTL translator should receive:

- grounded instruction;
- exact proposition inventory;
- proposition descriptions;
- allowed LTL operators;
- translation demonstrations;
- required output format.

Example proposition inventory:

```yaml
propositions:
  - id: in_room_bedroom_2
    meaning: robot is inside room_bedroom_2
  - id: near_object_sofa_7
    meaning: robot is in the reach region of object_sofa_7
  - id: in_room_bathroom_1
    meaning: robot is inside room_bathroom_1
```

### 10.2 Example translation

Grounded instruction:

```text
Enter room_bedroom_2, then reach object_sofa_7.
Always avoid room_bathroom_1.
```

Conceptual LTL:

```text
F(in_room_bedroom_2 AND F(near_object_sofa_7))
AND
G(NOT in_room_bathroom_1)
```

Use the exact syntax expected by the original repository and Spot.

### 10.3 Formula validation

Preserve the original validation loop:

1. check syntax;
2. verify all proposition names are valid;
3. check co-safety or supported formula class;
4. reject hallucinated entities;
5. retry translation with error feedback;
6. stop after a fixed retry limit.

Do not automatically weaken a formula to make it solvable.

### 10.4 Semantic validation

In addition to syntax checking, add deterministic checks:

- every required grounded subgoal appears in the formula;
- every hard avoidance appears in the formula;
- temporal order is preserved;
- no unsupported soft constraint is converted into a hard constraint;
- no proposition outside the inventory is used.

This validation is an interface safeguard, not a replacement for the original method.

---

## 11. Step 6: Construct the LTL Automaton

Use Spot or the automaton construction already used by the official repository.

The automaton should expose:

```text
initial state
accepting states
transition conditions
sink/rejecting states
```

At each occupancy node, evaluate the set of true propositions:

```python
labels = proposition_labeler(cell)
next_q = automaton.transition(current_q, labels)
```

The product state is:

```text
(cell_id, automaton_state)
```

For higher hierarchy levels it may be:

```text
(room_id, automaton_state)
(object_id, automaton_state)
(floor_id, automaton_state)
```

---

## 12. Step 7: Build the Hierarchical Planning Domain

### 12.1 Occupancy level

The occupancy level is the ground-truth executable planning domain.

State:

```text
(grid cell, automaton state)
```

Transitions:

```text
move to neighboring traversable cell
update propositions
advance automaton
```

Cost:

```text
geometric path length
```

Optional soft-cost extension:

```text
geometric path length + semantic preference cost
```

### 12.2 Object level

Object-level nodes represent object reach regions or object-associated free-space regions.

Use object-level transitions only when a valid occupancy-level connection exists.

Do not treat every object pair in the same room as directly connected without checking traversability.

### 12.3 Room level

Room-level nodes represent room regions.

Room-level adjacency should be based on valid cross-room transitions.

Room transition costs may be estimated using:

- shortest occupancy path between room entrances;
- precomputed entrance-to-entrance distances;
- lower-bound geometric distances.

### 12.4 Floor level

SemPathBench currently has one floor in most scenes.

Keep:

```text
floor_0
```

as a trivial hierarchy level for compatibility, but do not claim speed gains from multi-floor reasoning.

### 12.5 Product domain

Each hierarchical level must include the automaton state. Otherwise high-level search may ignore mission progress and violate temporal ordering.

Incorrect:

```text
room node only
```

Correct:

```text
(room node, automaton state)
```

---

## 13. Step 8: Preserve the Original Heuristics

### 13.1 LTL heuristic

Retain the original LTL-aware heuristic as the anchor heuristic.

Its role is to estimate progress toward an accepting automaton state while respecting the formal planning structure.

Do not replace it with a generic shortest-path heuristic and still claim the full original method.

### 13.2 LLM heuristic

The LLM heuristic should receive:

- compact scene-graph hierarchy;
- current hierarchy node;
- remaining mission inferred from the automaton;
- available high-level actions/functions.

Its output should guide search rather than directly determine the final path.

Example allowed guidance:

```text
move(room_bedroom_1, room_hallway_0)
move(room_hallway_0, room_kitchen_0)
reach(room_kitchen_0, object_countertop_3)
```

The response must be validated against the scene graph.

Invalid edges or entities must be discarded rather than executed.

### 13.3 LLM provider

The original paper used GPT-4. For benchmark consistency, the implementation may use the same LLM provider used by other LLM baselines, such as Gemini, provided that:

- the model name is recorded;
- prompt content is preserved as closely as possible;
- temperature and decoding settings are fixed;
- the replacement is disclosed;
- all LLM-based baselines use a consistent policy when possible.

Changing GPT-4 to Gemini is an implementation substitution, not a conceptual modification of the method.

---

## 14. Step 9: Run Hierarchical AMRA*-Style Planning

The preferred implementation is to reuse the original repository's planner.

Expected planner inputs:

```text
hierarchical scene graph
initial product state
goal/accepting product states
transition model
edge costs
LTL anchor heuristic
LLM heuristic
```

Expected output:

```text
sequence of product states
```

The final occupancy projection should be:

```text
[(row_0, col_0), ..., (row_T, col_T)]
```

### 14.1 Do not replace the planner prematurely

Replacing AMRA* with sequential A* would remove a central contribution of the original method:

- hierarchical multi-resolution planning;
- multi-heuristic guidance;
- anytime refinement;
- formal anchor heuristic.

A sequential A* fallback may be implemented only for debugging and must be labeled separately.

### 14.2 If the original planner cannot scale

Use the following order:

1. profile the original implementation;
2. reduce graph redundancy without changing search semantics;
3. precompute room and object connectivity;
4. cache occupancy distances;
5. use sparse graph representations;
6. restrict the graph to connected traversable components;
7. preserve AMRA* state and heuristic logic.

Only after these attempts should a planner substitution be considered.

If substituted, report:

```text
Optimal Scene Graph Planning – A* Adapter
```

rather than the full original method.

---

## 15. Step 10: Convert the Plan to a SemPathBench Trajectory

The final returned path must satisfy the benchmark interface:

```python
trajectory: list[tuple[int, int]]
```

Post-processing should be minimal.

Allowed:

- remove consecutive duplicate cells;
- convert coordinate convention;
- ensure the path starts at the benchmark start cell;
- expand high-level edges using stored occupancy subpaths.

Avoid:

- smoothing that changes semantic satisfaction;
- shortcutting across obstacles;
- changing target reach regions;
- repairing failed constraints using ground-truth annotations.

Validate:

```text
all cells are in bounds
all cells are traversable
each consecutive pair is connected
trajectory begins at the start pose
```

---

## 16. SemPathBench Runner

A possible runner interface:

```python
def solve_episode(episode, config):
    graph = build_scene_graph(
        occupancy_map=episode.occupancy_map,
        semantic_map=episode.semantic_map,
        room_map=episode.room_map,
        metadata=episode.metadata,
    )

    grounding = ground_instruction(
        instruction=episode.instruction,
        scene_graph=graph,
        start_pose=episode.start_pose,
    )

    supported_mission = extract_supported_mission(grounding)

    ltl_formula = translate_to_ltl(
        grounded_instruction=supported_mission.text,
        proposition_inventory=supported_mission.propositions,
    )

    automaton = build_automaton(ltl_formula)

    planning_domain = build_hierarchical_domain(
        scene_graph=graph,
        automaton=automaton,
        start_pose=episode.start_pose,
    )

    result = run_amra_star(
        domain=planning_domain,
        use_ltl_heuristic=True,
        use_llm_heuristic=True,
    )

    trajectory = project_to_occupancy_path(result.path)

    return {
        "trajectory": trajectory,
        "grounding": grounding.to_dict(),
        "ltl_formula": ltl_formula,
        "planner_status": result.status,
        "runtime": result.runtime,
    }
```

---

## 17. Failure Handling

The method should return explicit failure states.

Recommended statuses:

```text
SUCCESS
GROUNDING_FAILED
AMBIGUOUS_GROUNDING
LTL_TRANSLATION_FAILED
LTL_VALIDATION_FAILED
UNSUPPORTED_INSTRUCTION
NO_FEASIBLE_PATH
PLANNER_TIMEOUT
INTERNAL_ERROR
```

### 17.1 Unsupported soft constraints

If hard constraints are solvable but soft constraints are unsupported:

```text
planner_status = SUCCESS
unsupported_constraints = [...]
```

The trajectory should still be evaluated normally. Low SCS is a meaningful result.

### 17.2 Partial grounding

Do not discard unresolved text and continue as though grounding succeeded.

A required hard constraint that remains unresolved should cause:

```text
GROUNDING_FAILED
```

### 17.3 Infeasible mission

If no accepting product state is reachable:

```text
NO_FEASIBLE_PATH
```

Do not return the unconstrained shortest path.

---

## 18. Logging and Reproducibility

Save one record per episode:

```json
{
  "episode_id": "...",
  "instruction": "...",
  "start_pose": [0, 0],
  "scene_graph_summary": {...},
  "grounded_instruction": "...",
  "entity_bindings": [...],
  "unsupported_constraints": [...],
  "atomic_propositions": [...],
  "ltl_formula": "...",
  "automaton_state_count": 0,
  "planner_configuration": {
    "hierarchy_levels": ["occupancy", "object", "room", "floor"],
    "use_ltl_heuristic": true,
    "use_llm_heuristic": true
  },
  "runtime": {
    "grounding_seconds": 0.0,
    "ltl_seconds": 0.0,
    "planning_seconds": 0.0,
    "total_seconds": 0.0
  },
  "planner_status": "SUCCESS",
  "trajectory": []
}
```

Also log:

- LLM model;
- prompt version;
- temperature;
- retry count;
- token usage if available;
- random seed;
- planner timeout;
- number of expanded nodes at each hierarchy level;
- first feasible path cost;
- final path cost;
- whether optimality was reached before timeout.

---

## 19. Recommended Ablations

To demonstrate which parts of the original method matter, run:

### 19.1 Full adapted method

```text
Optimal Scene Graph Planning
```

Components:

```text
scene graph
entity grounding
LTL automaton
hierarchical AMRA*
LTL heuristic
LLM heuristic
```

### 19.2 No LLM heuristic

```text
Optimal Scene Graph Planning w/o LLM Heuristic
```

This tests whether LLM guidance improves planning efficiency.

### 19.3 Occupancy-only planner

```text
Optimal Scene Graph Planning – Occupancy Only
```

This tests the contribution of hierarchy.

### 19.4 No geometric resolver

```text
Optimal Scene Graph Planning w/o Metric Grounding
```

This measures how much SemPathBench-specific entity descriptions depend on deterministic geometric grounding.

### 19.5 Optional soft-cost extension

```text
Optimal Scene Graph Planning + Soft-Cost Adapter
```

This must be separated from the faithful baseline.

---

## 20. Fairness Rules

### 20.1 Allowed inference-time information

The method may use only information available through the standard SemPathBench input:

- instruction;
- start pose;
- occupancy map;
- semantic object-instance map;
- room-instance map;
- public instance metadata.

### 20.2 Disallowed information

Do not expose:

- ground-truth trajectory;
- evaluator-specific target IDs unavailable to other methods;
- human annotations that directly resolve ambiguous language;
- future path labels;
- hidden constraint structures, unless every method is explicitly given them.

### 20.3 Shared geometry utilities

Basic deterministic utilities may be shared across baselines:

```text
distance transform
shortest-path distance
room containment
object reach region
nearest/farthest candidate calculation
```

However, the paper must disclose which utilities are benchmark-provided and which belong to a method.

---

## 21. Major Implementation Mistakes to Avoid

### Mistake 1: Calling a flat A* implementation the full original method

If hierarchy, AMRA*, and multi-heuristic planning are removed, the implementation no longer reproduces the paper's main planning contribution.

### Mistake 2: Giving the LLM the answer through hidden metadata

Grounding must use only inference-time map information.

### Mistake 3: Letting the LLM invent object or room IDs

Every generated entity and proposition must be validated against the scene graph.

### Mistake 4: Converting all soft constraints into hard LTL constraints

This changes instruction semantics and may make feasible tasks infeasible.

### Mistake 5: Ignoring automaton state at high hierarchy levels

A room-only high-level search cannot correctly represent temporal task progress.

### Mistake 6: Using object obstacle pixels as object goals

Object goals should map to benchmark-consistent reachable regions.

### Mistake 7: Claiming native support for nearest/farthest grounding

Metric grounding is a SemPathBench adapter and must be documented as such.

### Mistake 8: Using sequential waypoint planning for arbitrary LTL

Planning independently to each extracted subgoal loses disjunction, avoidance, until, and other temporal logic semantics.

---

## 22. Minimum Acceptable Implementation

An implementation can be considered a credible adaptation only if it includes:

- SemPathBench-to-scene-graph conversion;
- unique room and object instances;
- explicit hierarchy and connectivity;
- scene-graph-based entity grounding;
- deterministic metric grounding for required referring expressions;
- grounded NL-to-LTL translation;
- LTL syntax and proposition validation;
- LTL automaton construction;
- product-state planning;
- occupancy-level executable trajectory output;
- original hierarchical planner or a clearly disclosed planner substitution;
- explicit unsupported-soft-constraint handling;
- complete per-episode logs.

A version that only performs:

```text
instruction
→ LLM waypoint list
→ sequential A*
```

should not be called Optimal Scene Graph Planning.

---

## 23. Recommended Development Order

### Phase 1: Scene graph and path interface

Implement:

```text
occupancy graph
room graph
object graph
containment
room adjacency
object reach regions
path projection
```

Test without language using manually specified propositions.

### Phase 2: LTL planning

Implement:

```text
proposition labeling
LTL formula input
automaton
product planning
```

Test manually specified formulas.

### Phase 3: Original grounding and translation

Add:

```text
attribute hierarchy prompt
entity-ID grounding
LTL translation
validation and retries
```

Test simple visit/order/avoid tasks.

### Phase 4: SemPathBench geometric grounding

Add:

```text
nearest
farthest
without-object
relative-distance
between/midpoint regions
```

Each resolver should have unit tests.

### Phase 5: Hierarchical AMRA* and heuristics

Integrate:

```text
occupancy/object/room/floor levels
LTL anchor heuristic
LLM heuristic
anytime planning
```

### Phase 6: Full evaluation

Run:

```text
small debug split
25-episode subset
full validation split
```

Inspect both aggregate metrics and grounding failures.

---

## 24. Expected Strengths and Weaknesses

### Expected strengths

The method should perform well on:

- ordered room/object visits;
- long hard-constraint sequences;
- must-visit constraints;
- hard room/object avoidance;
- instructions grounded by containment;
- scenes where hierarchy reduces search complexity.

### Expected weaknesses

The method may perform poorly on:

- soft near/far preferences;
- path smoothness preferences;
- clearance preferences;
- wall-following;
- circling;
- human-like trajectory similarity;
- highly geometric language not covered by the resolver;
- ambiguous descriptions without a unique scene-graph answer.

This is acceptable. A baseline should expose the inductive bias and limitations of the original method rather than be modified until it solves every benchmark feature.

---

## 25. Recommended Method Description for the Paper

A concise description could be:

> **Optimal Scene Graph Planning** grounds a natural-language mission to uniquely identified room and object entities in a hierarchical scene graph, translates the grounded mission into an LTL automaton, and performs LLM-guided multi-resolution product-state planning to obtain a minimum-cost occupancy-level path.

For the adaptation section:

> We construct a floor-room-object-occupancy hierarchy from the SemPathBench maps and preserve the original scene-graph grounding, LTL translation, automaton construction, and hierarchical multi-heuristic planning pipeline. We add only deterministic metric resolution for benchmark-specific referring expressions such as nearest, farthest, and relative-distance descriptions. Soft path preferences unsupported by the original Boolean LTL formulation are ignored in the faithful baseline and evaluated through the shared SemPathBench metrics.

---

## 26. Final Recommended Pipeline

```text
instruction
+
start pose
+
occupancy map
+
semantic object-instance map
+
room-instance map
        |
        v
build floor-room-object-occupancy scene graph
        |
        v
LLM scene-graph entity grounding
        |
        +---- deterministic metric resolver
        |     nearest / farthest / without / relative relation
        v
grounded mission with unique entity IDs
        |
        v
LLM grounded mission → co-safe LTL
        |
        v
syntax, proposition, and semantic validation
        |
        v
Spot LTL automaton
        |
        v
hierarchical scene graph × automaton domain
        |
        +---- consistent LTL anchor heuristic
        |
        +---- validated LLM semantic heuristic
        v
AMRA*-style anytime hierarchical planning
        |
        v
occupancy-level path projection
        |
        v
SemPathBench trajectory and shared evaluation
```

The central implementation principle is:

> Adapt the representation and grounding boundary required by SemPathBench, but preserve the original formal and hierarchical planning method.
