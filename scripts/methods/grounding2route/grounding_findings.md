# Grounding2Route Grounding Findings

This note summarizes the current findings from comparing two Grounding2Route
grounding designs:

1. **Intent + deterministic grounding**: the LLM extracts structured intent,
   and local code deterministically grounds objects, regions, hard constraints,
   soft constraints, cost maps, and trajectories.
2. **LLM API program grounding**: the LLM writes a restricted Python-like API
   program, which is parsed, interpreted, grounded, and planned.

The main conclusion is that the API-program design is more interpretable and
expressive, but currently less reliable. The stronger baseline remains
intent-based parsing followed by deterministic grounding.

## Experimental Snapshot

The API-program run below used `valunseen`, first 50 instructions:

```bash
python scripts/methods/grounding2route/run.py \
  --set valunseen \
  --limit 50 \
  --output-root resources/methods/grounding2route/previous/Grounding2Route_api_program \
  --llm-cache-root resources/methods/grounding2route/previous/Grounding2Route_api_program/_llm_cache \
  --overwrite \
  --save-debug-artifacts \
  --verbose
```

Overall comparison:

| Method | N | PLR | HCS | SCS | OSS | Avg Seconds |
|---|---:|---:|---:|---:|---:|---:|
| API program | 50 | 0.881 | 0.707 | 0.458 | 0.319 | 20.2 |
| Intent pure | 50 | 0.863 | 0.723 | 0.783 | 0.519 | 71.0 |
| Previous intent | 50 | 0.862 | 0.683 | 0.538 | 0.346 | 33.9 |
| Previous DSL | 50 | 0.910 | 0.567 | 0.471 | 0.294 | 44.3 |

The API-program method has competitive path length ratio and reasonable hard
constraint completion, but its soft-constraint score is much lower. This is the
main reason its OSS is substantially worse than intent pure.

SCS detail:

| Method | Near | Far | Relative | Clearance | Path Shape |
|---|---:|---:|---:|---:|---:|
| API program | 0.387 | 0.603 | 0.356 | 0.445 | 0.332 |
| Intent pure | 0.460 | 0.931 | 0.908 | 0.855 | 0.443 |

## Prompt-Tuning Attempts

We also tried several API-program prompt variants, each evaluated on the first
25 instructions only.

| Run | PLR | HCS | SCS | OSS |
|---|---:|---:|---:|---:|
| Baseline API first 25 | 0.859 | 0.787 | 0.652 | 0.483 |
| Relation fix only | 0.859 | 0.787 | 0.632 | 0.483 |
| Strong prompt v2 | 0.904 | 0.727 | 0.572 | 0.428 |
| Conservative prompt v3 | 0.860 | 0.780 | 0.651 | 0.489 |

The strong prompt tried to force route decomposition, `near_region` targets,
ordered `between_region` visits, and stronger path-shape rules. It hurt
performance because the LLM started acting like a low-level route planner and
over-composed programs that did not match the evaluator.

The conservative prompt produced only a very small improvement, too thin to be
treated as a reliable gain. The current recommendation is not to rely on prompt
tuning alone.

## Main Failure Modes

The API-program failures are mostly not syntax failures. In the first 25
relation-fix run:

```text
parse_failed: 1
grounding_failed: 1
planner_failed: 2
HCS=0: 4
HCS partial: 3
SCS < 0.5: 8
OSS=0: 6
```

Most programs are valid and executable. The dominant problem is semantic
mismatch: the program is locally plausible, but it does not preserve the exact
route-level or scoped preference semantics used by the benchmark evaluator.

### Route Semantics Become Too Weak

Example instruction:

> Go through the bathroom to the painting in the nearest bedroom. While you are
> in the initial room, you should stay closer to the garbage can. When entering
> the bathroom, you should stay closer to the plunger.

The LLM wrote a single final target with a hard visit:

```python
s1 = api.go_to(target_painting, constraints=[
    api.require_visit(bathroom),
    api.prefer_near(garbage_can, within=initial_room),
    api.prefer_near(plunger, within=bathroom),
])
```

This is syntactically valid and semantically plausible, but it does not
reliably encode the ordered route "go through the bathroom to the painting".
The resulting HCS was 0.

### Soft Constraint Scope Is Fragile

Preferences such as "while leaving the room", "when entering the bathroom", or
"in the next living room" require a correct segment and spatial scope. The LLM
often attaches the preference to a plausible segment, but the scope does not
match the evaluator's expected region. This mainly hurts near, far, and
relative preference scores.

### Relative Preferences Need Canonicalization

The planner expects canonical relation strings such as `closer_to` and
`farther_from`, but LLM programs often use variants like `"farther"`,
`"closer_than"`, `<`, or `>`. We added deterministic canonicalization in the
grounder. This is a correctness fix, but by itself it did not improve OSS on
the first 25 instructions.

### Path Shape Is Not Captured by Simple Region Visits

For loop or walk-around instructions, the LLM can write:

```python
loop_region = api.circle(current_room, fraction=1.0, start_toward=garbage_can)
s1 = api.go_to(api.near_region(table), constraints=[
    api.require_visit_in_order(loop_region),
])
```

This can satisfy some required waypoints, but the benchmark evaluates the
trajectory shape. Visiting generated regions is not always equivalent to
matching the ground-truth loop or wall-following behavior.

### API Programs Push Too Much Planning Responsibility to the LLM

The current API is relatively low level: `kth_nearest`, `between_region`,
`circle`, `require_visit`, `prefer_near`, `prefer_relative`, and so on. The LLM
must decide whether each instruction clause is a target, a must-pass region, a
soft preference, a forbidden region, a segment boundary, or a scope. This creates
many valid-but-wrong programs.

## Interpretation

The weakness is not simply that the LLM cannot write code. The LLM usually
writes syntactically valid restricted programs. The core issue is that code
generation introduces a larger semantic search space:

- A valid API call sequence may not preserve the instruction.
- A plausible route decomposition may not match the benchmark annotation.
- A soft preference may be attached to the wrong segment or scope.
- A path-shape instruction may be reduced to waypoint visits.
- The validator checks syntax and API legality, not semantic faithfulness.

Therefore, the API-program method is more interpretable but less faithful under
the current interface.

## Recommendation

For the main method, use intent-based parsing plus deterministic grounding.
This constrains the LLM to semantic extraction and leaves object selection,
region construction, hard-constraint grounding, soft-cost construction, and
planner interface details to deterministic code.

The API-program direction can still be useful as an ablation or future variant,
but it should be improved with:

1. **A semantic verifier** that checks whether every hard and soft instruction
   clause is represented in the generated program.
2. **Higher-level API operators**, such as `pass_through(room)`,
   `leave_current_room()`, `loop_around(reference)`, and
   `stay_near(reference, while_in=room)`.
3. **Repair based on verifier failures**, not only syntax or grounding errors.

## Suggested Paper Framing

A concise way to describe the design choice:

> We found that delegating grounding composition to the LLM through an
> executable API program increased expressivity but reduced reliability. The
> generated programs were often syntactically valid and executable, yet failed
> to preserve route-level semantics such as ordered pass-through constraints,
> scoped preferences, and path-shape requirements. Therefore, we adopt an
> intent-based parser followed by deterministic grounding, which better aligns
> with the benchmark's semantic constraints and evaluation protocol.

The key message is not "hard-coded grounding is better" in a generic sense.
Rather:

> We constrain the role of the LLM to semantic parsing and use deterministic
> grounding to ensure faithfulness and evaluation alignment.
