# LIMP to SemPathBench Adaptation Plan

## 1. Goal

This document describes how to adapt **LIMP: Language Instruction Grounding for Motion Planning** to **SemPathBench** while preserving the original LIMP method as much as possible.

The purpose of this adaptation is **not** to redesign LIMP into a stronger SemPathBench-specific planner. The purpose is to evaluate whether the original LIMP methodology can be transferred to a known-map, 2D semantic path-planning benchmark.

A low score is acceptable. Unsupported constraints, grounding failures, infeasible plans, and poor soft-constraint performance should be treated as meaningful benchmark outcomes rather than reasons to replace LIMP's core method.

The intended baseline name is:

> **LIMP-Nav for SemPathBench**

or, more explicitly:

> **LIMP with Oracle Semantic-Instance Perception for SemPathBench**

---

## 2. Original LIMP Task Setting

LIMP is a general framework for following complex natural-language robot instructions involving:

- navigation;
- mobile manipulation;
- open-vocabulary object references;
- object-instance disambiguation;
- spatial constraints;
- temporal constraints;
- robot skills such as `near`, `pick`, and `release`.

Its original input is approximately:

```text
natural-language instruction
+
posed RGB-D observations
+
3D metric geometry map
+
robot skill library
```

Its original output is:

```text
a task-and-motion plan
=
a sequence of navigation and manipulation behaviors
that satisfies the generated temporal-logic specification
```

The original high-level pipeline is:

```text
Natural-language instruction
        ↓
Two-stage LLM translation
        ↓
LTL specification
+ parameterized robot predicates
+ Composable Referent Descriptors
        ↓
Open-vocabulary visual grounding
        ↓
Referent Semantic Map
        ↓
LTL finite-state machine
        ↓
Task Progression Semantic Maps
        ↓
Progressive task-and-motion planning
        ↓
Navigation / pick / place execution
```

LIMP's central method components are:

1. **Two-stage language-to-LTL translation**
2. **Composable Referent Descriptors (CRDs)**
3. **Object-instance grounding through spatial relations**
4. **LTL progression / finite-state task representation**
5. **Instruction-conditioned semantic maps**
6. **Progressive task and motion planning**

These components should be preserved in the SemPathBench adaptation.

---

## 3. SemPathBench Task Setting

SemPathBench provides:

```text
Input:
- natural-language instruction
- complete 2D occupancy / traversability map
- complete 2D semantic object-instance map
- room-instance map
- object and room metadata
- start pose

Output:
- one complete 2D grid trajectory
```

The instruction may specify:

```text
- target object or room
- target-instance selection
- ordered visitation
- enter / exit conditions
- near / far relations
- between relations
- avoidance constraints
- return-to-start constraints
- soft path preferences
- path-shape constraints
```

SemPathBench does not require:

```text
- online RGB-D perception
- active exploration
- pick / place execution
- manipulation configuration planning
- velocity control
```

Therefore, the adaptation should use only LIMP's navigation branch while preserving its original language grounding and task-progression structure.

---

## 4. Core Adaptation Principle

The adaptation should follow this rule:

> Replace only the environment-perception interface and robot embodiment interface. Do not replace LIMP's language representation, referent grounding logic, temporal progression, or progressive planning philosophy.

The allowed substitution is:

```text
Original LIMP perception:
posed RGB-D images
→ open-vocabulary detection and segmentation
→ 3D object instances
→ Referent Semantic Map

SemPathBench adaptation:
oracle semantic instance map
→ candidate object instances
→ LIMP spatial-relation filtering
→ Referent Semantic Map
```

This replacement changes where object candidates come from, but it does not remove LIMP's referent-resolution problem.

The LIMP adaptation must still decide:

```text
- which referent phrases appear in the instruction;
- which candidate instances match each referent;
- which spatial relations disambiguate the correct instance;
- which regions are goals;
- which regions are forbidden;
- which constraints are active at each task stage.
```

---

## 5. Proposed SemPathBench Pipeline

```text
SemPathBench instruction
+ occupancy map
+ semantic instance map
+ room map
+ object metadata
+ start pose
        ↓
LIMP Language Instruction Module
        ↓
Stage 1:
natural language → conventional LTL
        ↓
Stage 2:
conventional LTL
→ LIMP skill predicates
+ Composable Referent Descriptors
        ↓
SemPathBench object-instance adapter
        ↓
candidate object / room instances
        ↓
LIMP-style spatial grounding
        ↓
Referent Semantic Map
        ↓
compile LTL to finite-state machine
        ↓
for each current automaton state:
generate Task Progression Semantic Map
        ↓
derive:
- current goal region
- current infeasible semantic regions
- currently enabled transition
        ↓
2D motion planner
        ↓
execute the planned segment symbolically
        ↓
update LTL / FSM state
        ↓
repeat until accepting state or failure
        ↓
concatenate all path segments
        ↓
SemPathBench trajectory
```

---

## 6. What Must Be Preserved

### 6.1 Two-Stage Instruction Translation

The original LIMP translation structure should remain:

```text
Stage 1:
instruction → conventional LTL

Stage 2:
instruction + conventional LTL
→ skill-aware LTL
+ CRDs
```

Do not replace this with:

```text
instruction → JSON constraints
```

or:

```text
instruction → direct waypoint list
```

because doing so would remove one of LIMP's central contributions.

The prompts may be minimally adapted so that the available navigation-only predicates are clear.

---

### 6.2 LTL as the Task Specification

The final task specification should remain an LTL formula compiled into an automaton or finite-state task representation.

The planner should use task progression rather than simply splitting the instruction into an ordered list and calling A* independently.

A path state should conceptually include:

```text
(grid position, LTL/FSM progress state)
```

Even if implementation uses progressive segment planning, the task state must still come from the LTL automaton.

---

### 6.3 Composable Referent Descriptors

CRDs should remain the representation for object-instance references.

Examples:

```text
chair :: isnextto(table)

chair :: isbetween(sofa, bed)

whiteboard :: isinfrontof(green_plush_toy)
```

The SemPathBench adapter may add a small number of map-specific comparators, but it should not replace CRDs with a completely different grounding language.

---

### 6.4 Spatial-Relation Grounding

The method must still perform geometric filtering over candidate instances.

For example:

```text
Instruction:
Go to the chair between the sofa and the table.

Candidates:
chair_1, chair_2, chair_3

Grounding:
evaluate isbetween(chair_i, sofa_j, table_k)
→ select the best valid chair instance
```

It is not acceptable to use the benchmark's hidden target annotation directly.

The semantic map provides candidate instances, not the final grounded answer.

---

### 6.5 Task Progression Semantic Maps

At every task stage, the method should construct an instruction-conditioned map containing:

```text
- current goal region;
- geometric obstacles;
- active semantic forbidden regions;
- skill-initiation or proposition-satisfaction regions.
```

The map must be updated when the LTL state changes.

Do not build one static global costmap for the entire instruction and call that LIMP.

---

### 6.6 Progressive Planning

LIMP should progressively plan toward the next automaton transition:

```text
current task state
→ current TPSM
→ motion plan
→ satisfy proposition
→ update task state
```

This should continue until:

```text
- an accepting state is reached;
- no valid transition can be grounded;
- no feasible path exists;
- the maximum planning budget is exceeded.
```

---

## 7. Allowed Adaptations

The following changes are acceptable because they adapt the embodiment or data interface rather than the core LIMP method.

### 7.1 Replace Visual Perception with Oracle Candidate Extraction

Original:

```text
RGB-D
→ VLM detector / segmentation
→ 3D candidate instances
```

Adapted:

```text
semantic instance map
→ connected instance masks
→ 2D candidate instances
```

The adapted system should extract for every instance:

```text
- instance ID
- semantic class
- occupied cells
- centroid
- bounding box
- room membership
- distance from start
- adjacency relationships
```

This version should be described as:

> oracle semantic-instance perception

not:

> oracle grounding

because grounding is still performed by LIMP.

---

### 7.2 Use 2D Geometry Instead of 3D Geometry

LIMP's spatial comparators operate on grounded object geometry. In SemPathBench, all comparators should operate in the benchmark's 2D map coordinates.

For example:

```text
3D centroid → 2D centroid
3D object volume → 2D instance mask
3D free space → 2D traversable cells
3D navigation pose set → 2D reachable goal region
```

The comparator semantics should remain as close as possible to the original.

---

### 7.3 Restrict the Skill Library to Navigation

The SemPathBench skill set should initially be:

```text
near(referent)
```

Optionally, use explicit aliases internally:

```text
visit(referent) := near(referent)
navigate_to(referent) := near(referent)
```

Do not introduce a large SemPathBench-specific action language.

The original manipulation predicates:

```text
pick(...)
release(...)
```

should be disabled because the benchmark does not include manipulation.

If the LLM generates `pick` or `release`, the episode should fail translation or be marked unsupported rather than silently rewriting the task.

---

### 7.4 Replace Continuous Motion Planning with Grid Motion Planning

Original LIMP uses continuous navigation planning over feasible and infeasible regions.

SemPathBench requires a grid trajectory, so the low-level motion planner may be:

```text
A*
Dijkstra
or another deterministic grid planner
```

The planner must consume the current Task Progression Semantic Map.

The planner should not receive the benchmark's ground-truth trajectory.

---

### 7.5 Use the Benchmark Occupancy Map

All geometric obstacles should come from the SemPathBench traversability map.

This is preferable to reconstructing occupancy from RGB-D because the benchmark is designed to evaluate semantic planning rather than perception accuracy.

---

## 8. Minimal Navigation Predicate Set

To remain faithful, the first implementation should keep a small predicate set.

Recommended predicates:

```text
near(referent)
```

Recommended logical use:

```text
F near(A)
F (near(A) & F near(B))
G ~near(C)
near(A) U near(B)
```

The exact LTL encoding should follow LIMP's original prompting and predicate conventions whenever possible.

Do not immediately add specialized predicates such as:

```text
circle_room(...)
follow_wall(...)
stay_far_from(...)
pass_through_gap(...)
```

Those would significantly modify the original method.

---

## 9. Referent Comparator Library

### 9.1 Preserve Original Comparator Types

Implement the 2D equivalents of the original spatial comparators:

```text
isbetween
isabove
isbelow
isleftof
isrightof
isnextto
isinfrontof
isbehind
```

For a top-down map, some comparators require a coordinate convention.

Recommended convention:

```text
- map x-axis defines left/right;
- map y-axis defines above/below;
- front/behind is only used when an object or room orientation is available.
```

If orientation is unavailable, `isinfrontof` and `isbehind` should return unresolved instead of inventing a direction.

---

### 9.2 Minimal SemPathBench Extensions

Only add comparators that are required to make basic benchmark referents representable:

```text
isinside(object, room)
contains(room, object)
```

These are justified because SemPathBench explicitly provides room maps, while the original LIMP environments did not use the same room-level representation.

Optional second-stage extensions:

```text
nearest_to(candidate_set, reference)
farthest_from(candidate_set, reference)
not_contains(room, object)
```

These should be clearly reported as adaptation extensions rather than original LIMP operators.

The first experiment should preferably report results both:

```text
LIMP-Core:
only original comparators + room containment

LIMP-Extended:
adds nearest / farthest / negative containment
```

This separates original-method capability from benchmark-specific extensions.

---

## 10. Goal Region Construction

For a grounded object or room, `near(referent)` must be converted into a reachable grid region.

For object instance \(o\):

```text
goal_region(o)
=
traversable cells within distance r_near
from the object mask
```

For room instance \(r\):

```text
goal_region(r)
=
traversable cells inside the room
or near the room's designated interior target region
```

The same `r_near` rule should be shared across all methods where possible.

The goal region must not be the object's occupied cells if those cells are non-traversable.

---

## 11. Forbidden Region Construction

For a negative proposition such as:

```text
G ~near(red_chair)
```

construct:

```text
forbidden_region(red_chair)
=
cells within r_avoid of the grounded instance
```

The region should be added to the current TPSM as infeasible.

Hard avoidance should be binary, following LIMP's feasibility interpretation.

Do not convert it into a soft distance cost unless the original specification itself is soft, because LIMP is fundamentally a hard-constraint method.

---

## 12. Handling SemPathBench Constraints

### 12.1 Naturally Supported

The following constraints are appropriate for the faithful first version:

```text
- go to an object;
- go to a room;
- visit A before B;
- visit A and eventually B;
- avoid object or room C;
- after visiting A, do not return to C;
- return to the start after visiting A;
- disambiguate an instance using between / left / right / next-to relations;
- select an object inside a specified room.
```

Examples:

```text
Go to the chair next to the table.

Visit the sofa, then go to the bed.

Go to the table while avoiding the dog bed.

Go to the chair between the sofa and the dresser.

Visit the kitchen and then return to the starting point.
```

---

### 12.2 Partially Supported

The following may work only after minimal comparator extensions:

```text
- nearest object;
- farthest object;
- object inside the nearest / farthest room;
- room without a specified object;
- relative instance selection using the start pose.
```

These constraints affect referent grounding, not the core planner.

They should be implemented as deterministic candidate-ranking functions invoked by CRDs.

---

### 12.3 Unsupported in the Faithful Version

The following should not be forcibly added to the first LIMP baseline:

```text
- soft near / far preferences;
- path smoothness preferences;
- clearance preferences;
- circle around an object;
- circle around a room;
- follow an entire wall boundary;
- pass through a narrow gap as a path-shape event;
- exact relative-distance inequalities over an extended path;
- velocity constraints.
```

For these cases, the method should:

```text
1. preserve the original instruction;
2. attempt normal LIMP translation;
3. reject unsupported generated predicates;
4. return a documented failure reason.
```

It should not silently drop a hard constraint.

For soft constraints, two reporting modes are acceptable:

```text
Strict:
episode fails translation if the instruction cannot be represented.

Baseline:
unsupported soft clauses are ignored, but the episode is marked
"soft constraint unsupported" and receives normal benchmark evaluation.
```

The preferred main result is the second mode, because it allows the method to output a path while naturally obtaining poor SCS.

---

## 13. Failure Semantics

Every episode should produce one of the following statuses:

```text
SUCCESS_ACCEPTING_STATE
TRANSLATION_FAILED
UNSUPPORTED_PREDICATE
GROUNDING_FAILED
AMBIGUOUS_GROUNDING
AUTOMATON_COMPILATION_FAILED
NO_FEASIBLE_TRANSITION
NO_PATH
MAX_PROGRESS_STEPS
INVALID_OUTPUT_PATH
```

Do not repair failed LIMP outputs with hand-written fallback planning in the main evaluation.

A fallback A* result may be logged for debugging, but it must not be counted as LIMP performance.

---

## 14. Suggested Software Architecture

```text
methods/limp/
├── run.py
├── config.py
├── adapter/
│   ├── sempath_scene.py
│   ├── instance_extractor.py
│   ├── room_adapter.py
│   └── coordinate_adapter.py
├── language/
│   ├── stage1_ltl.py
│   ├── stage2_skill_ltl.py
│   ├── prompts/
│   └── parser.py
├── grounding/
│   ├── crd_parser.py
│   ├── candidate_registry.py
│   ├── comparators_2d.py
│   ├── room_comparators.py
│   └── referent_map.py
├── logic/
│   ├── ltl_compile.py
│   ├── automaton.py
│   └── progression.py
├── planning/
│   ├── tpsm.py
│   ├── goal_regions.py
│   ├── forbidden_regions.py
│   ├── grid_motion_planner.py
│   └── progressive_planner.py
├── outputs/
│   ├── trace_logger.py
│   └── visualization.py
└── tests/
```

Whenever possible, import and reuse code from the official LIMP repository rather than rewriting it.

---

## 15. Required Per-Episode Artifacts

Save the following for every episode:

```text
1. original instruction
2. Stage-1 LTL output
3. Stage-2 skill-aware LTL output
4. parsed CRDs
5. candidate instances for each referent
6. selected grounded instance
7. grounding scores / comparator results
8. compiled automaton
9. sequence of automaton states
10. TPSM at each planning stage
11. path segment at each stage
12. final concatenated path
13. failure status, if any
14. runtime by module
```

This is essential because LIMP's value includes interpretability and verification.

---

## 16. Implementation Stages

### Stage 0: Reproduce Official Demo

Before adapting to SemPathBench:

```text
- install the official repository;
- initialize the open-spatial-grounding submodule;
- run the demo notebook;
- verify LTL translation;
- verify CRD parsing;
- verify automaton generation;
- verify task progression output.
```

Do not begin by rewriting LIMP from the paper alone if official code can be reused.

---

### Stage 1: Navigation-Only Official LIMP

Restrict the original system to instructions that use only:

```text
near(...)
```

Verify that:

```text
instruction
→ LTL
→ CRDs
→ grounding
→ navigation plan
```

works before replacing perception.

---

### Stage 2: SemPathBench Scene Adapter

Implement:

```text
semantic map
→ object instances

room map
→ room instances

occupancy map
→ free / blocked cells
```

Produce a registry such as:

```json
{
  "sofa_1": {
    "class": "sofa",
    "cells": [],
    "centroid": [x, y],
    "room": "living_room_1"
  }
}
```

---

### Stage 3: 2D CRD Grounding

Port the original comparator interface to 2D.

Test manually with synthetic maps:

```text
- left / right
- above / below
- next to
- between
- room containment
```

Do not use LLM reasoning to compute these relations.

---

### Stage 4: Referent Semantic Map

Convert each grounded referent into a map-aligned mask.

Visualize:

```text
- all candidates;
- selected instance;
- goal ring;
- forbidden ring;
- room region.
```

---

### Stage 5: LTL Compilation and Progression

Use the official Spot-based or repository-provided logic path whenever possible.

Verify:

```text
- simple eventual goal;
- ordered goals;
- persistent avoidance;
- stage-dependent avoidance;
- return-to-start.
```

---

### Stage 6: Task Progression Semantic Maps

For every automaton state, build:

```text
TPSM = {
  geometric_obstacles,
  active_forbidden_regions,
  enabled_goal_regions
}
```

The next planning segment must come from this map.

---

### Stage 7: Grid Motion Planner

Use a deterministic planner with no SemPathBench-specific semantic heuristics.

Recommended first version:

```text
8-connected A*
cost = geometric path length
```

Semantic constraints should enter only through TPSM goal and infeasible regions.

---

### Stage 8: Benchmark Integration

Convert the progressive segment output to the shared method interface:

```text
run_episode(record) -> trajectory, metadata
```

Evaluate using the existing SemPathBench evaluator without method-specific metric changes.

---

## 17. Main Experimental Variants

### Variant A: LIMP-Core

```text
- official two-stage translation;
- original LIMP comparator family in 2D;
- room containment only as necessary adaptation;
- hard constraints only;
- no nearest / farthest extension;
- no soft-cost planning.
```

This should be the primary fidelity result.

---

### Variant B: LIMP-Extended-Grounding

```text
LIMP-Core
+
nearest
+
farthest
+
contains / not_contains
```

This tests whether SemPathBench-specific target selection is the main bottleneck.

---

### Variant C: Oracle-Translation Diagnostic

Use ground-truth symbolic/LTL structure while retaining:

```text
- LIMP grounding;
- automaton;
- TPSM;
- progressive planning.
```

This is an ablation only, not the main baseline.

It separates language translation errors from grounding and planning errors.

---

### Variant D: Oracle-Grounding Diagnostic

Use ground-truth referent IDs while retaining:

```text
- LIMP language translation;
- automaton;
- TPSM;
- progressive planning.
```

Again, this is diagnostic only.

---

## 18. Evaluation Expectations

Expected strengths:

```text
- ordered hard constraints;
- persistent hard avoidance;
- stage-dependent constraints;
- compositional object-instance grounding;
- interpretable intermediate representations;
- explicit failure detection.
```

Expected weaknesses:

```text
- soft constraints;
- nearest / farthest without extensions;
- room-level negative descriptions;
- path-shape constraints;
- very long instructions;
- LLM LTL translation reliability;
- 2D ambiguity for front / behind;
- planner runtime for complex automata.
```

A poor SCS score is expected and should not motivate adding a soft-cost optimizer to the main LIMP baseline.

A low completion rate on unsupported instruction families is also acceptable if failures are logged faithfully.

---

## 19. What Must Not Be Done

The following changes would make the method no longer a faithful LIMP adaptation:

```text
- replace LTL with a custom JSON task program;
- let the LLM directly output the final trajectory;
- use the benchmark's annotated target instance as grounding input;
- skip CRDs and directly query object IDs;
- replace task progression with a hand-written waypoint sequence;
- use one static costmap instead of TPSMs;
- add a large library of SemPathBench-specific path primitives;
- convert all soft constraints into a new learned or hand-tuned cost function;
- silently delete unsupported hard constraints;
- use the human reference trajectory during inference;
- use the benchmark evaluator to optimize the path during inference.
```

---

## 20. Fidelity Checklist

An implementation should only be called **LIMP-Nav** if the answer to all of the following is yes:

```text
[ ] Does it use two-stage language translation?
[ ] Does it produce an LTL specification?
[ ] Does it use LIMP-style robot predicates?
[ ] Does it represent referents with CRDs?
[ ] Does it enumerate and filter candidate instances?
[ ] Does it use spatial comparators for disambiguation?
[ ] Does it compile or progress an LTL task state?
[ ] Does it generate task-stage-dependent semantic maps?
[ ] Does it progressively plan toward enabled transitions?
[ ] Does it stop only at an accepting state or explicit failure?
[ ] Does it avoid benchmark-specific fallback planning?
```

If several of these are missing, the method should be renamed:

```text
LIMP-inspired baseline
```

rather than:

```text
LIMP adaptation
```

---

## 21. Final Recommended Scope

The first faithful implementation should target only:

```text
- navigation-only instructions;
- object and room goals;
- ordered visitation;
- hard semantic avoidance;
- object-instance disambiguation;
- return-to-start;
- stage-dependent prohibitions.
```

It should not attempt to solve:

```text
- soft preference optimization;
- circle / loop constraints;
- follow-wall behavior;
- velocity instructions;
- complex continuous relative-distance objectives.
```

The correct research question is:

> How much of SemPathBench can the original LIMP methodology solve when its open-vocabulary perception layer is replaced by oracle semantic-instance perception, while its language-to-LTL translation, compositional referent grounding, task progression, and progressive motion planning remain unchanged?

The implementation should prioritize methodological fidelity over benchmark performance.

---

## 22. References

- Benedict Quartey, Eric Rosen, Stefanie Tellex, and George Konidaris.  
  **Verifiably Following Complex Robot Instructions with Foundation Models.**  
  ICRA 2025.  
  Paper: https://arxiv.org/abs/2402.11498

- Project website:  
  https://robotlimp.github.io/

- Official code:  
  https://github.com/benedictquartey/robotlimp