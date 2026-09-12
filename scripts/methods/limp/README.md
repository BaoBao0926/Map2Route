# LIMP-Nav-Prototype-SemPathBench

This directory contains the LIMP adapter for SemPathBench. The implementation must be identified as **LIMP-Nav-Prototype**. It includes the engineering structure for LIMP-style two-stage Gemini translation, candidate extraction, CRD grounding, LTLf residual-formula progression, and TPSM-like grid planning. It must not be presented as a complete official LIMP baseline until the official Spot-DFA formal core is vendored.

[Paper](https://arxiv.org/abs/2402.11498) | [Official Repository](https://github.com/benedictquartey/robotlimp)

## 1. Running the Method

### Environment Setup

The existing SemPathBench environment is sufficient. Pillow is recommended for saving trajectory images:

```bash
pip install pillow
```

The default automaton backend is the local LTLf residual-formula progression implementation:

```text
--automaton-backend ltlf
```

It reuses the Formula/progression semantics from the SemPathBench `lang2ltl` adapter and does not require `spot`. A closer reproduction of official LIMP will require vendoring the official Spot/DFA path. The `residual_fallback` backend is allowed only for explicit debugging or smoke tests.

### Model and API Key

Main experiments must use:

```text
--translation-mode llm
```

If the API key is unavailable or an LLM request fails, the episode returns `TRANSLATION_FAILED` instead of silently falling back to a heuristic.

Gemini is called through the same REST interface as `scripts/methods/lang2ltl/translator.py`; the `google-generativeai` package is not required. Configure:

```python
# scripts/methods/api_key.py
MODEL = "your_gemini_model_here"
API_KEY = "your_gemini_api_key_here"
```

Alternatively, use environment variables:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export GEMINI_MODEL="your_gemini_model_here"
```

### Minimal Run

```bash
python scripts/methods/limp/run.py --limit 1 --overwrite --verbose
```

Full run:

```bash
python scripts/methods/limp/run.py \
  --set valunseen \
  --translation-mode llm \
  --overwrite --verbose
```

### Arguments

| Argument | Description |
| --- | --- |
| `--input-root PATH` | Instruction input directory. Defaults to `resources/instructions`. |
| `--output-root PATH` | Directory for predictions and `summary.json`. Defaults to `resources/methods/baselines/LIMP`. |
| `--set {valunseen,train,all}` | Instruction split to run. Defaults to `valunseen`. |
| `--model MODEL` | Gemini model. Defaults to the local `api_key.py` configuration or `GEMINI_MODEL`. |
| `--llm-cache-root PATH` | Translation cache. Defaults to `resources/methods/baselines/LIMP/_llm_cache`. |
| `--method-variant {faithful,extended}` | `faithful` retains the complete LTL/TPSM fixes but disables adapter-specific context aliases, selector stripping/tie-breaking, and closest-instance refinement. Defaults to `extended`. |
| `--translation-mode {llm,heuristic}` | `llm` is the main experiment mode and reports `TRANSLATION_FAILED` on failure; `heuristic` is for smoke tests only. |
| `--grounding-mode {core,extended,oracle}` | `core` reports `AMBIGUOUS_GROUNDING` for unresolved multiple instances. `extended` applies a deterministic shortest-reachable-path tie-break from the start and records the decision in metadata. Defaults to `extended`. |
| `--translation-context {none,categories}` | `none` hides the map inventory from the LLM. `categories` is reserved for diagnostics. Defaults to `none`. |
| `--near-radius N` | Radius of a `near[...]` goal region. |
| `--avoid-radius N` | Radius of a negative-near forbidden region. |
| `--max-progress-steps N` | Maximum number of progressive-planner stages. |
| `--automaton-backend {ltlf,spot,residual-debug}` | `ltlf` is the default. `spot` is reserved for a future official-aligned backend; `residual-debug` is for smoke tests only. |
| `--allow-residual-fallback` | Debug-only option that permits residual fallback without Spot. |
| `--max-goal-candidates N` | Debug-only goal-region limit. The default `0` means no truncation; main experiments must not truncate goal regions. |
| `--overwrite` | Replace existing predictions. |
| `--overwrite-llm-cache` | Ignore cached translations and translate again. |
| `--limit N` | Run only the first N instructions. |
| `--verbose` | Print detailed execution logs. |
| `--evaluate` | Deprecated compatibility option; metrics are always computed by the shared evaluator. |

## 2. Key Files

| File | Purpose |
| --- | --- |
| `run.py` | SemPathBench method runner, structured similarly to the tutorial runner. |
| `pipeline.py` | End-to-end LIMP pipeline for one instruction. |
| `config.py` | Method name, output directory, radii, and search-budget defaults. |
| `language/translator.py` | Two-stage LLM translation; the heuristic mode is only for explicit smoke tests. |
| `language/predicate_parser.py` | LIMP lifted-predicate encoding. |
| `grounding/scene_adapter.py` | Builds the object/room candidate registry from SemPathBench `map_state`. |
| `grounding/crd_parser.py` | CRD parser. |
| `grounding/comparators_2d.py` | Two-dimensional spatial comparators. |
| `grounding/referent_map.py` | CRD grounding and ambiguity/failure reporting. |
| `logic/automaton.py` | Task-progression facade. |
| `logic/ltlf_progression.py` | Default LTLf residual-formula progression backend. |
| `logic/residual_fallback.py` | Debug-only sequential progression; not valid as the main LIMP backend. |
| `planning/tpsm.py` | TPSM summary representation. |
| `planning/progressive_planner.py` | Progressive TPSM and grid-A* planner. |
| `vendor/README.md` | Policy for vendoring official method-level LIMP code. |

### 2.1 Official Component Alignment

| LIMP component | Official code reused | Adapted | Reimplemented | Reason |
| --- | ---: | ---: | ---: | --- |
| Stage-1 prompt |  | yes |  | Follows the official two-stage structure but is stored locally. |
| Stage-2 prompt |  | yes |  | Adds a navigation-only skill set and soft-clause warning. |
| Predicate encoding |  |  | yes | Mirrors lifted-predicate replacement locally; the official function is not imported at runtime. |
| CRD parser |  |  | yes | Supports nested CRDs locally; official code requires broader repository imports. |
| LTL backend |  | yes | partial | Uses local LTLf Formula/progression semantics; the official Spot/DFA path remains future work. |
| Progression |  | yes | partial | `ltlf` progresses residual formulas; `residual_fallback` remains debug-only. |
| Perception |  | yes |  | Replaces RGB-D/VLM perception with semantic-instance candidate enumeration. |
| Motion planner |  | yes |  | Replaces FMT* with grid A* for SemPathBench. |

Reimplemented or partially reimplemented components are method-level deviations:

- **Predicate encoding:** Implemented locally from the paper and official lifted-predicate format to avoid importing the robot runtime package. The expected impact is low when the LTL string and encoding map match the official syntax.
- **CRD parser:** Implemented locally from the official CRD syntax because the original helpers depend on broader repository imports. The expected impact is medium, especially for rare nested descriptors.
- **LTL backend and progression:** The local `ltlf` backend progresses and records residual-formula states but is not the official Spot/DFA wrapper. The expected impact is medium to high until official DFA progression is vendored.
- **Motion planner:** Continuous FMT* is replaced with grid A* because SemPathBench is a two-dimensional grid benchmark. This is suitable for 2D navigation but is not a reproduction of manipulation or continuous control.

## 3. Adaptation from Original LIMP

LIMP is adapted as a navigation-only variant with semantic-map perception. The original pipeline translates language into LTL formulas with lifted robot predicates and composable referent descriptors (CRDs), then grounds object referents through perception-driven 3D scene representations. This adapter preserves the language-to-LTL and CRD-grounding structure while replacing RGB-D/VLM perception with the benchmark semantic-instance map.

The method enumerates room and object candidates from the semantic layers, translates instructions into predicates such as `near[referent]`, grounds CRDs with deterministic two-dimensional spatial relations and room membership, and constructs proposition regions for task progression. Local LTLf residual-formula progression and full-goal grid planning adapt the task-progression semantic-map idea to a discrete grid.

The extended grounding mode adds documented non-oracular handling for common navigation language: synonym normalization, virtual midpoint referents for between-region goals, nearest-reachable tie-breaking, and deterministic distance, ordinal, or size ranking for explicit selectors. Inference uses only instruction text, start pose, traversability, and map-derived semantic instances. It does not use evaluator feedback, human demonstrations, hidden target identifiers, or benchmark constraint annotations.

### 3.1 Perception

Original LIMP uses RGB-D observations, OWL-ViT/SAM, and 3D back-projection to build an instruction-conditioned referent semantic map.

The SemPathBench adapter uses:

```text
map_state.layers.object_instance
map_state.object_instances
map_state.layers.room
map_state.room_instances
```

These fields provide semantic-instance perception, not oracle grounding. Final CRD instance selection is still performed by LIMP-style spatial grounding.

### 3.2 Skill Library

Because SemPathBench is a navigation benchmark, the main implementation supports only:

```text
near[referent]
```

If the LLM emits `pick[...]` or `release[...]`, the episode returns `UNSUPPORTED_PREDICATE` without silently rewriting the action.

`--method-variant faithful` forces the core grounding policy and disables adapter-specific start-context candidates, selector-token stripping, and `isnextto` closest-instance refinement. The complete residual-LTL/TPSM planner fixes remain enabled.

### 3.3 Grounding

Core grounding never silently selects one instance when grounding is ambiguous. If multiple candidates satisfy a CRD and the instruction has no explicit ranking selector, the episode returns:

```text
AMBIGUOUS_GROUNDING
```

The default extended mode applies explicit, recorded, deterministic fallbacks without hidden target annotations, human trajectories, or evaluator scores:

- For ordinary ambiguity, it performs full-goal reachability planning from the start cell to each candidate proposition region and selects the shortest reachable candidate.
- For explicit selectors, `nearest/closest` uses shortest reachable path, `farthest/furthest` uses straight-line distance from the start, `second/third/other/next` uses reachable order, and `largest/smallest` uses instance footprint. Exactly one instance is selected and the policy is recorded.

Extended grounding also includes two basic navigation adaptations:

- Common aliases such as `countertop -> counter_top`, `plant -> house_plant`, `door -> doorway`, and `armchair -> arm_chair`.
- Virtual between targets such as `midpoint::isbetween(sofa,tv)`, represented by a navigation candidate at the midpoint of the grounded references.

These SemPathBench-specific adaptations are recorded in grounding metadata and must not be described as direct official `robotlimp` grounding code.

### 3.4 Planning

The continuous FMT* planner from original LIMP is replaced with 8-connected grid A*. At each stage, the adapter computes residual-LTL transitions for the complete proposition valuation over every traversable cell. Cells that change the formula without rejecting it form the goal-transition region; cells that progress the formula to `false` form the forbidden region. A* therefore avoids hard-constraint violations along the path instead of selecting a destination only from one `current_symbol`.

### 3.5 LTL Progression

The default `ltlf` backend records:

```json
"automaton_summary": {
  "backend": "local_ltlf_residual_formula"
}
```

Only the explicit options below enable `residual_fallback`:

```bash
--automaton-backend residual-debug --allow-residual-fallback
```

The output then records:

```json
"automaton_summary": {
  "backend": "residual_fallback"
}
```

`residual_fallback` is a method-level deviation. A future vendored official Spot/DFA path should replace or align the `ltlf` backend.

## 4. Information Excluded from Inference

Main LIMP inference does not use:

- `hard_constraints`
- `soft_constraints`
- instruction `objects`
- `human_expert_trajectory`
- evaluator feedback

These fields are used only by the shared evaluator after a trajectory is produced.

## 5. Output

The default output directory is:

```text
resources/methods/baselines/LIMP
```

Each prediction JSON contains:

- `trajectory`
- shared-evaluator `metrics`
- `limp.stage1_ltl`
- `limp.stage2_ltl`
- `limp.encoding_map`
- `limp.groundings`
- `limp.automaton_summary`
- `limp.logic_trace`
- `limp.tpsm_summaries`
- `limp.planner_segments`
- `limp.method_level_deviations`
