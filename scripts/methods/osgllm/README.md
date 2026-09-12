# Optimal Scene Graph Planning-SemPathBench

This directory adapts *Optimal Scene Graph Planning with Large Language Model Guidance* to the SemPathBench method interface.

## 1. Running the Method

### Environment Setup

The official OSG-LLM repository, Spot, and the C++ planner are not required. The existing SemPathBench environment is sufficient.

Configure Gemini when using `--grounding-mode llm`, `--translation-mode llm`, or `--use-llm-heuristic`:

```python
# scripts/methods/api_key.py
MODEL = "gemini-3.5-flash"
API_KEY = "your_gemini_api_key_here"
```

Environment variables are also supported:

```bash
export GEMINI_MODEL=gemini-3.5-flash
export GEMINI_API_KEY=...
```

Minimal run:

```bash
python scripts/methods/osgllm/run.py --limit 1 --overwrite
```

Full run:

```bash
python scripts/methods/osgllm/run.py \
  --set valunseen \
  --grounding-mode llm \
  --translation-mode llm \
  --planner amra \
  --use-llm-heuristic \
  --overwrite
```

### Arguments

| Argument | Description |
| --- | --- |
| `--input-root PATH` | Instruction input directory. Defaults to `resources/instructions`. |
| `--output-root PATH` | Directory for predictions and `summary.json`. Defaults to `resources/methods/baselines/OSGLLM`. |
| `--set {valunseen,train,all}` | Instruction split to run. Defaults to `valunseen`. |
| `--translation-mode {heuristic,llm,auto}` | LTL translation mode. `auto` tries the LLM and falls back to the heuristic. |
| `--grounding-mode {heuristic,llm,auto}` | Entity-grounding mode. `auto` tries the LLM and falls back to the heuristic. |
| `--planner {astar,product,hierarchical,amra}` | `astar` connects grounded ordered AP goals with sequential grid A* and no product state; `product` runs the product anchor plus fallback; `hierarchical` and `amra` enable hierarchy-guided refinement after product-anchor failure. |
| `--object-reach-radius N` | Radius of each object approach region. Defaults to `20`. |
| `--max-expansions N` | Maximum product-search expansions. Defaults to `250000`. |
| `--max-planning-seconds N` | Planner timeout per episode. Defaults to `30.0` seconds. |
| `--model MODEL` | Gemini model used for translation and heuristic guidance. |
| `--llm-cache-root PATH` | Cache for LLM grounding, translation, and heuristic calls. |
| `--overwrite-llm-cache` | Force regeneration of LLM cache entries. |
| `--use-llm-heuristic` | Enable experimental high-level scene-graph guidance from the LLM. |
| `--disable-llm-heuristic` | Disable the LLM heuristic. |
| `--overwrite` | Replace existing predictions; otherwise, existing results are skipped. |
| `--limit N` | Run only the first N instructions for debugging. |
| `--verbose` | Print planner-search logs. |
| `--evaluate` | Deprecated compatibility option; metrics are always computed by the shared evaluator. |

## 2. Key Files

| File | Purpose |
| --- | --- |
| `run.py` | SemPathBench method runner, structured similarly to `scripts/methods/tutorial/run.py`. |
| `pipeline.py` | End-to-end episode pipeline: graph construction, translation, labeling, planning, and metadata. |
| `config.py` | Adapter configuration dataclass. |
| `scene_graph/graph_types.py` | `SceneGraph` and `AttributeRegion` data structures. |
| `scene_graph/builder.py` | Builds floor, room, and object regions from `map_state.layers.occupancy/room/object_instance`. |
| `scene_graph/propositions.py` | Builds the AP inventory and grid-cell label map. |
| `language/translator.py` | Local deterministic grounding and sequential-eventual LTLf translation. |
| `language/llm_adapter.py` | Experimental Gemini grounding, LTL translation, caching, and validation. |
| `language/llm_heuristic.py` | Experimental Gemini high-level scene-graph heuristic guidance. |
| `language/validator.py` | Formula validation against the AP inventory. |
| `planning/product_planner.py` | `(row, col, residual_formula)` product-state A*. |
| `planning/hierarchical_planner.py` | Experimental object/room/floor hierarchy-guided refinement. |
| `docs/unsupported_cases.md` | Soft and Path-Shape cases that are not currently supported faithfully. |
| `test_scene_graph.py` | Unit tests for scene-graph construction and AP labeling. |
| `test_product_planner.py` | Product-planner unit tests. |
| `__main__.py` | Supports `python -m scripts.methods.osgllm`. |
| `plan.md` | Detailed design plan and future AMRA* implementation roadmap. |

## 3. Adaptation to SemPathBench

This directory is a local SemPathBench adaptation of OSG-LLM, not a wrapper around the official repository.

### 3.1 Input Format

The original OSG-LLM method targets Gibson and 3D scene graphs. This runner reads SemPathBench instruction JSON, `map_state`, `start_pose`, semantic layers, and map metadata.

The adapter does not use inference-time oracle information. It never uses `hard_constraints`, `soft_constraints`, annotation `objects`, or `human_expert_trajectory` to select a target.

### 3.2 Scene Graph and Attribute Regions

SemPathBench provides a two-dimensional grid map. The adapter builds:

```text
floor_0
├── room_<id>
└── object_<id>
```

The base graph consists of traversable occupancy cells. Floors, rooms, and objects are represented as attribute regions:

- `floor_0`: all traversable cells.
- `room_<id>`: traversable cells belonging to the corresponding room-layer instance.
- `object_<id>`: traversable approach cells surrounding the object mask.

Room adjacency is derived from neighboring traversable cells and never read from hidden annotations.

### 3.3 Atomic Propositions

AP names follow the action style used by OSG-LLM:

```text
enter(room_<id>)
reach(object_<id>)
enter(floor_0)
```

The internal label map associates these APs with grid cells.

### 3.4 Grounding and Translation

The adapter provides two grounding/translation paths.

The default heuristic path uses a deterministic translator that:

- Matches map-derived room/object categories, names, and attributes in the instruction text.
- Supports basic nearest/farthest ranking.
- Filters objects using nearby room mentions when possible.
- Converts phrases such as `avoid`, `not allowed`, and `must not enter` into `G ! ap`.
- Records soft and Path-Shape preferences as unsupported instead of converting them into hard LTL.

The LLM path in `language/llm_adapter.py`:

- Receives a compact scene graph.
- Produces JSON grounding.
- Produces `ltl_prefix` or `ltl_ast`.
- Validates entity IDs, the AP inventory, and the formula.
- Writes cache entries to `resources/methods/baselines/OSGLLM/_llm_cache` by default.

The `auto` mode tries the LLM first and falls back to the heuristic. Explicit `llm` mode records `LTL_TRANSLATION_FAILED` on failure and does not silently report success.

Generated LTL uses the local `Formula` and progression implementation from `scripts/methods/lang2ltl/ltl.py`; Spot is not required by default.

### 3.5 Planner

The primary planner is product-state A*:

```text
state = (row, col, residual_formula)
```

After moving to a new cell, the planner:

1. Reads the AP labels active at that cell.
2. Progresses the residual formula.
3. Returns the trajectory if the residual formula is accepting.
4. Prunes the successor if progression yields `false`.

The heuristic uses positive eventual APs in the residual formula and estimates octile distance to their AP regions.

With `--planner hierarchical` or `--planner amra`, a timeout or expansion-budget failure in product-state A* triggers experimental hierarchy-guided refinement:

```text
room adjacency BFS skeleton -> augmented AP goals -> grid A* refinement
```

If hierarchy refinement also fails and the formula contains only sequential hard goals without avoidance, `sequential_ap_astar_fallback` connects the grounded AP goals. This fallback does not read hidden constraints; it uses only the AP sequence from the translator and map-derived AP regions. Metadata retains details from the product anchor, hierarchy refinement, and fallback.

When `--use-llm-heuristic` is enabled, the LLM returns only preferred high-level scene-graph nodes used to order hierarchy BFS. It does not output a dense trajectory or modify LTL goals.

### 3.6 Output

The runner writes prediction JSON, PNG trajectories, and `summary.json`. Each prediction includes:

- `trajectory`
- shared-evaluator `metrics`
- `osgllm.scene_graph_summary`
- `osgllm.grounding`
- `osgllm.ltl`
- `osgllm.planner`
- `osgllm.method_level_deviations`

Grounding, translation, and planning failures are never presented as success. A trajectory may contain only the start position, but `osgllm.status` records the actual failure, such as `GROUNDING_FAILED`, `PLANNER_TIMEOUT`, or `NO_FEASIBLE_PATH`.

## 4. Current Limitations

This is an executable experimental adapter, not a complete faithful OSG-LLM reproduction:

- LLM grounding, LTL translation, and heuristic interfaces exist, but prompts, retry/repair logic, and benchmark-scale validation remain incomplete.
- The LLM heuristic currently performs high-level node ordering rather than implementing the AMRA* auxiliary heuristic queue from the paper.
- The complete AMRA* multi-resolution, multi-queue planner from the paper is not implemented.
- The optional Spot automaton backend is not implemented.
- Soft path preferences, clearance, smoothness, wall-following, and circling are recorded as unsupported.
- The deterministic translator has limited recall on complex natural-language instructions.

These limitations are written to prediction metadata for failure analysis.

## 5. Tests

```bash
python -m py_compile \
  scripts/methods/osgllm/*.py \
  scripts/methods/osgllm/scene_graph/*.py \
  scripts/methods/osgllm/language/*.py \
  scripts/methods/osgllm/planning/*.py

python -m unittest \
  scripts.methods.osgllm.test_scene_graph \
  scripts.methods.osgllm.test_product_planner
```

Smoke test:

```bash
python scripts/methods/osgllm/run.py \
  --set valunseen \
  --limit 1 \
  --overwrite \
  --output-root /tmp/sempathbench_osgllm_smoke \
  --planner hierarchical \
  --max-planning-seconds 2 \
  --max-expansions 5000
```

## 6. Implementation Roadmap

Suggested priorities:

1. Implement a true AMRA*-style multi-resolution product planner.
2. Improve LLM grounding and LTL retry, repair, semantic validation, and ablations.
3. Connect the LLM heuristic to a true AMRA* auxiliary queue instead of using it only as a hierarchy-BFS tie-break.
4. Add doorway, between, and virtual-region resolvers.
5. Systematically analyze `GROUNDING_FAILED`, `PLANNER_TIMEOUT`, and `NO_FEASIBLE_PATH` cases.
