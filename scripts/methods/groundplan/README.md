# GroundPlan SemPathBench

GroundPlan is a benchmark-native SemPathBench method. It follows the design in
`design.md`:

```text
instruction -> Direct-CaP grounding code -> sandbox execution + verification
-> bounded code repair (R <= 3) -> frozen grounded task IR -> scope-aware deterministic Planner -> trajectory
```

The generated code is grounding-only: its sandbox exposes read-only semantic-map
queries and task builders, but no planner, route, cost, weight, search, file, or
network API. The planner is called only after the code returns a verified
`GroundedProgram`.

The main inference path does not use `hard_constraints`, `soft_constraints`,
instruction `objects`, `human_expert_trajectory`, evaluator feedback, or metric
details while planning.

## Canonical Cap-style run

Canonical Direct-CaP method (code representation, full temporal/spatial scope, repair budget R=3):

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --overwrite --verbose
```

## Ablations

### Planner backend

The default `--planner-mode sequential_greedy_astar` plans the grounded route
segments in order. Each segment receives the endpoint selected by the previous
segment and runs deterministic scope-aware A* over its required regions and
destination.

The exact global planner ablation uses layered multi-source dynamic programming.
It retains every legal arrival state at one ordered stage, transfers their
cumulative costs to the next stage, and backtracks the globally best endpoint
sequence. Only one dense map layer is retained at a time. A sequential plan is
kept as an executable incumbent, so a bounded diagnostic run never degrades to
a start-only trajectory. The automatic ablation runs this backend without an
expansion limit.

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --output-root resources/methods/groundplan/ablation/05_global_vs_sequential_planning/global_planner \
  --replay-grounding-code-root resources/methods/groundplan/main_result \
  --planner-mode global_layered_dp \
  --max-expansions 0 \
  --overwrite --verbose
```

`--max-expansions 0` means unlimited. Successful layered results record
`search_complete=true` and `optimality_proven=true`. The older
`global_progress_astar` product-state backend remains available for a direct
implementation comparison. All planner modes receive the same frozen
`GroundedProgram`, target masks, hard scopes, and soft cost fields.

A guarded global experiment can retain the sequential incumbent unless the
exact global candidate produces a meaningful geometric path reduction. The
guard uses no benchmark hard/soft annotations:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --output-root resources/methods/groundplan/main_result \
  --replay-grounding-code-root resources/methods/groundplan/main_result \
  --planner-mode global_layered_dp \
  --max-expansions 0 \
  --global-min-path-improvement-m 1.0 \
  --workers 1
```

`--soft-weight-scale` separately scales grounded near/far/relative/path-shape
fields while leaving the clearance regularizer unchanged.

### Grounding representation

The default `--grounding-representation code` resolves to `direct_cap_repair`: the LLM writes restricted grounding code directly, verification returns structured errors, and at most three code repairs are allowed. The JSON representation experiment is run separately with:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --grounding-representation json \
  --max-parse-repairs 0 \
  --max-grounding-repairs 0 \
  --execution-repair off \
  --output-root resources/methods/groundplan/previous/experiments/incomplete_schema_no_repair \
  --overwrite --verbose
```

`--grounding-representation json` resolves to one-shot JSON IR → deterministic
code with no refinement, parse repair, execution repair, alias retry, or IR
fallback. Invalid one-shot outputs count as failures. The repair-enabled
pipeline remains available explicitly through `--grounding-variant full`.
`--grounding-variant` remains an advanced override. The available variants are:

A native ToolCall representation uses Gemini function calling over the exact callable registry used by the CaG sandbox. It receives only the free-form instruction, category inventories, category counts, and the opaque `start_position` handle; it receives no benchmark constraint decomposition. API results are stored as session handles, and `task(...)` must construct the existing `GroundedProgram` before the unchanged verifier and planner run. No generated text is executed as code.

Primary no-repair comparison:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --grounding-representation tool_call \
  --max-parse-repairs 0 \
  --max-grounding-repairs 0 \
  --execution-repair off \
  --max-tool-calls 32 \
  --max-tool-query-retries 3 \
  --llm-cache-root resources/methods/groundplan/main_result/_llm_cache \
  --output-root resources/methods/groundplan/ablation/01_grounding_strategy/tool_call \
  --workers 1 \
  --overwrite --verbose
```

For the repair-enabled comparison, use `--max-grounding-repairs 3 --execution-repair tool`. An identical tool query that errors or returns an empty list is executed at most three times; later identical calls are blocked without accessing the map, while other queries and graceful grounding termination remain available. ToolCall retains its tool context, receives the existing verifier errors, and may make more calls before resubmitting `task(...)`. Per-episode traces and token/call counts are saved in `.steps.json`; Easy/Hard mean and median statistics are written to `tool_call_efficiency.json`. `function_call` is accepted as an alias of `tool_call`.

Two additional representation ablations use a compact semantic scene catalog instead of the multi-megabyte raw map JSON. Dense map layers, footprints, benchmark constraints, and human trajectories are never placed in the prompt.

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --grounding-representation direct_id \
  --max-parse-repairs 0 \
  --max-grounding-repairs 0 \
  --execution-repair off \
  --output-root resources/methods/groundplan/ablation/01_grounding_strategy/direct_id \
  --overwrite --verbose

python scripts/methods/groundplan/run.py \
  --set valunseen \
  --grounding-representation ltl \
  --max-parse-repairs 0 \
  --max-grounding-repairs 0 \
  --execution-repair off \
  --output-root resources/methods/groundplan/ablation/02_intermediate_representation/ltl \
  --overwrite --verbose
```

`direct_id` asks the LLM to emit final room/object IDs plus typed constraints. `ltl` emits a restricted LTL hard-task formula over grounded propositions; quantitative soft preferences remain typed annotations because standard LTL does not define Near/Far/Path-Shape optimization costs. Both compile to the same `GPProgramSpec`, grounder, planner, and evaluator as the other variants. All three representation experiments are one-shot and default to `--max-parse-repairs 0`; malformed or ungroundable outputs are scored as failures rather than repaired.

| Variant | JSON IR | Code refinement | Helpers | Execution repair |
| --- | --- | --- | --- | --- |
| `direct_dsl` | yes | no | no | no |
| `direct_cap` | no | yes | yes | no |
| `direct_cap_repair` | no | yes | yes | code repair |
| `ir_to_code` | yes | no | no | no |
| `ir_code_refine` | yes | yes | yes | no |
| `ir_code_refine_code_repair` | yes | yes | yes | code repair |
| `full` | yes | yes | yes | code first, then IR |
| `direct_id` | no | no | no | no |
| `ltl` | no | no | no | no |
| `tool_call` | no | no | no | configurable tool repair |

For example, run the deterministic IR-to-code ablation without refinement:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --grounding-variant ir_to_code \
  --overwrite --verbose
```

The granular overrides are useful for controlled ablations:

```bash
--code-refinement / --no-code-refinement
--allow-grounding-helpers / --no-allow-grounding-helpers
--execution-repair {off,code,code_then_ir,tool}
--max-tool-calls N / --max-tool-llm-turns N
--max-tool-query-retries N  # hard-capped at 3
```

### Automatic non-representation ablations

First run the canonical method so every episode has a complete `.steps.json` attempt history. Then run:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --ablation \
  --overwrite --verbose
```

The suite writes PDF-aligned results under `resources/methods/groundplan/ablation/`: repair cases go to `03_verification_and_repair/`, scope and Hard Only cases to `04_soft_constraint_modeling/`, and the global planner to `05_global_vs_sequential_planning/`. It also writes the consolidated `table_iii_summary.json`. The canonical `R=3` result is reused from `main_result/`; the replay makes zero LLM calls. Grounding-representation experiments are run separately with `--grounding-representation`.

### Restricted parser representation

Restricted API-program parser ablation:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --parse-mode api \
  --overwrite --verbose
```

## Debugging

Debug subset with artifacts:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --limit 25 \
  --output-root resources/methods/groundplan/main_result \
  --overwrite \
  --save-debug-artifacts \
  --overwrite-llm-cache \
  --verbose

```




Planner-only debug mode from benchmark annotations:

```bash
python scripts/methods/groundplan/planner/debug_annotation_planner.py \
  --set valunseen \
  --limit 25 \
  --overwrite
```

Run one specific instruction through the planner-only oracle path:

```bash
python scripts/methods/groundplan/planner/debug_annotation_planner.py \
  --instruction-file resources/instructions/procthor/003_valunseen/instruction_files/instruction_000006.json \
  --overwrite
```

This bypasses the Parser and Grounder. It reads annotated `hard_constraints`,
`must_avoid`, and supported `soft_constraints`, converts them into Grounded IR,
then calls the current GroundPlan planner. Outputs are written to:

```text
resources/methods/groundplan/ablation/06_planner_over_oracle_grounding/ours
```

Without `--overwrite`, existing planner-debug prediction JSON files are reused
and reported as skipped. Add `--overwrite` after changing planner weights or
planner logic.

The planner-only PNGs overlay benchmark annotations for inspection: hard
constraints are drawn in black/red, soft constraint scopes in blue, and soft
reference regions in orange/green/purple.

Planner clearance currently uses a strengthened wall/object clearance cost. The
field is computed from evaluator-style obstacle cells, including non-traversable
occupancy and non-door object footprints, with defaults in `config.py`:

```text
CLEARANCE_RADIUS_METERS = 0.75
WEIGHT_NEAR = 8.00
WEIGHT_FAR = 96.00
DEFAULT_RELATIVE_COST_MODE = "ratio"
WEIGHT_RELATIVE = 64.00
WEIGHT_CLEARANCE = 36.00
MAX_EXPANSIONS = 1000000
```

The canonical relative field minimizes
`d_close / (d_close + d_far)`, which is the bounded monotonic transform
`1 / (1 + D_far / D_close)` of the evaluator's relative metric. The previous
hinge-difference field remains available as `relative_cost_mode="difference"`
for controlled experiments.

The full pipeline passes the ratio mode and weight explicitly through
`GroundPlanRunConfig` and records them under `planner_config` in the
GroundPlan metadata. CLI overrides are `--relative-cost-mode` and

`--relative-weight`.
Reproduce the zero-LLM oracle comparison on the first 20 val-unseen
instructions containing a relative preference:

```bash
python scripts/methods/groundplan/relative_cost_experiment.py \
  --limit-relative 20 \
  --difference-weights 64 160 320 640 1280 \
  --ratio-weights 1 4 16 64 256 \
  --workers 4
```

The runner uses annotation-derived Grounded IR, makes no LLM calls, supports
resume, and updates its output JSON after every completed planner run.

In planner-only oracle mode, annotated `path_shape_preference` constraints are
recorded as skipped and are not passed to the planner. Annotated
`near_preference` constraints add a reachable near-object anchor scoped to the
reference object's room. These are debug-oracle controls for isolating planner
behavior with GT information; the normal GroundPlan parser/grounder path does
not receive these annotation fields.

Run the first 25 val-unseen instructions as a normal GroundPlan experiment:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --limit 25 \
  --overwrite \
  --save-debug-artifacts \
  --overwrite-llm-cache \
  --verbose
```

If you are iterating on Parser or Grounder code and want to resume instead of
regenerating completed predictions, omit `--overwrite`:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --limit 25 \
  --verbose
```

By default, the full variant uses bounded execution repair. It first asks for a
local grounding-code repair; if that still fails, the next repair falls back to
the JSON IR (or API-program) scaffold and recompiles code. To disable all
execution repair and run a strict single-pass pipeline:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --limit 25 \
  --overwrite \
  --execution-repair off \
  --max-grounding-repairs 0 \
  --verbose
```

If you changed the parser prompt or model and want to ignore old Gemini cache:

```bash
python scripts/methods/groundplan/run.py \
  --set valunseen \
  --limit 25 \
  --overwrite \
  --overwrite-llm-cache \
  --verbose
```

The default output directory is:

```text
resources/methods/groundplan/main_result
```

After the run, inspect:

```text
resources/methods/groundplan/main_result/summary.json
resources/methods/groundplan/main_result/<map_id>/<instruction_id>.json
resources/methods/groundplan/main_result/<map_id>/<instruction_id>.steps.json
resources/methods/groundplan/main_result/<map_id>/<instruction_id>.png
resources/methods/groundplan/main_result/<map_id>/<instruction_id>.artifacts/
```

For Parser/Grounder debugging, start with the `.steps.json` file:

```text
steps.parse.metadata.raw_intent or steps.parse.metadata.raw_api_program
steps.parse.program
steps.grounding.status
steps.grounding.failure_reason
steps.grounding.bindings
steps.grounding.segments
steps.grounding_attempts
steps.repair_attempts
steps.llm_trace  # every system/user prompt, response, raw provider payload, and cache provenance
```

With `--save-debug-artifacts`, each artifact directory contains:

```text
parser.json
parser_intent.json or parser_api.py
parser_program.txt
grounding.json
grounded_program.py
grounding_trace.py
planner.json
cost_map.png
semantic_map.png
manifest.json
```

The Grounder now executes the core benchmark-native symbolic subset used by the
prompt: set filtering with `where(...)`, count/comparison predicates,
`contains(...)`, `adjacent_rooms(...)`, relation predicates such as
`near_to(...)`, and scoped regions including `side_region(...)` and
`half_room(...)`.

Configure Gemini with either environment variables or `scripts/methods/api_key.py`:

```python
MODEL = "gemini-3.5-flash"
API_KEY = "your_gemini_api_key_here"
```

## Arguments

| Argument | Purpose |
| --- | --- |
| `--input-root PATH` | Instruction root, default `resources/instructions`. |
| `--output-root PATH` | Output root, default `resources/methods/groundplan/main_result`. |
| `--set {valunseen,train,all}` | Instruction split. |
| `--parse-mode {intent,api}` | Parser mode. `intent` is the default main path; `api` is the restricted API-program ablation. |
| `--model MODEL` | Gemini model override. |
| `--llm-cache-root PATH` | Parser response cache. |
| `--overwrite-llm-cache` | Refresh Gemini cache. |
| `--max-parse-repairs N` | LLM rewrites after malformed representation output; default `0` (disabled). |
| `--max-expansions N` | Planner expansion budget; `0` means unlimited. |
| `--planner-mode MODE` | `sequential_greedy_astar`, exact `global_layered_dp`, or legacy product-state `global_progress_astar`. |
| `--planner-heuristic-weight W` | Weight for product-state global A*; exact layered DP requires `1.0`. |
| `--soft-weight-scale X` | Multiply grounded near/far/relative/path-shape fields; default `1.0`, clearance unchanged. |
| `--global-min-path-improvement-m M` | With exact layered DP, retain the sequential incumbent unless the global path is at least `M` metres shorter. |
| `--max-grounding-repairs N` | LLM rewrites after grounding failures; default 2. Repairs stay in the selected parser format. |
| `--grounding-variant` | Grounding ablation: `direct_dsl`, `direct_cap`, `direct_cap_repair`, `ir_to_code`, `ir_code_refine`, `ir_code_refine_code_repair`, or default `full`. The downstream planner is identical for every variant. |
| `--code-refinement` | Enable/disable the optional LLM refinement of deterministic IR-to-code scaffolds. |
| `--allow-grounding-helpers` | Enable/disable pure LLM-defined helper functions in the grounding-code sandbox. |
| `--execution-repair` | `off`, `code`, or `code_then_ir`; overrides the selected ablation's repair policy. |
| `--workers N` | Number of episodes to run concurrently; default 1. Useful when API latency dominates. |
| `--overwrite` | Regenerate predictions. |
| `--limit N` | Debug subset. |
| `--verbose` | Print progress. |

## Files

| File | Role |
| --- | --- |
| `run.py` | Tutorial-style SemPathBench runner. |
| `pipeline.py` | One-episode Parser/Grounder/Planner pipeline. |
| `ir.py` | Typed AST, grounded IR, and planner records. |
| `parse/` | Gemini intent parser and restricted API-program parser wrappers. |
| `grounding/` | Restricted `SceneMap`, deterministic DSL interpreter, and `code.py` executable grounding-code sandbox. |
| `planner/` | Sequential semantic A*, exact layered global DP, and product-state global A*. |
| `planner/debug_annotation_planner.py` | Planner-only oracle debug runner using benchmark annotations. |
| `llm_client.py` | Gemini REST helper with file cache. |
| `gap.md` | Current adaptation and implementation gaps. |

## Output

Each prediction is written under:

```text
resources/methods/groundplan/main_result/<map_id>/<instruction_id>.json
```

Each episode also writes a sibling intermediate file:

```text
resources/methods/groundplan/main_result/<map_id>/<instruction_id>.steps.json
```

The prediction JSON contains the final trajectory and standard SemPathBench
metrics. The `.steps.json` file records parser metadata and AST, grounded IR,
planner segment details, and final status for inspection. When grounding repair
runs, inspect `parse_attempts`, `grounding_attempts`, and `repair_attempts`.
