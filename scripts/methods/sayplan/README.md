# SayPlan-SemPathBench

This directory adapts the SayPlan approach to SemPathBench. The implementation preserves the core ideas of a collapsed scene graph, `expand_node` / `contract_node` semantic search, high-level planning, classical path completion, and feedback-driven replanning. Its goal is method fidelity rather than maximizing benchmark scores.

The adapter does not import or execute code from `SayPlan_Reconstruct/`; that third-party directory is retained only as a reference.

## 1. Running the Method

### Environment Setup

The existing SemPathBench environment is sufficient. The SayPlan runner uses the grid A*, evaluator, instruction/map loaders, and Gemini REST helpers already included in the repository.

Install Pillow to save trajectory images:

```bash
pip install pillow
```

### Gemini Model and API Key

The LLM backend is Gemini. Configure:

```python
# scripts/methods/api_key.py
MODEL = "your_gemini_model_here"
API_KEY = "your_gemini_api_key_here"
```

The default model is read directly from `MODEL` in `scripts/methods/api_key.py`. SayPlan does not hard-code `gemini-2.5-flash` or fall back to `GEMINI_MODEL`.

`API_KEY` may be stored in `scripts/methods/api_key.py` or provided through the environment:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### Inference Data Isolation

SayPlan inference receives only the free-form `instruction`, `start_pose`, and the scene graph derived from `map_state`. The adapter does not expose instruction annotations such as `objects`, `hard_constraints`, `soft_constraints`, or `human_expert_trajectory` to semantic search, high-level planning, or start-pose resolution. Predictions without the current inference-contract version are considered stale and recomputed.

### Minimal Run

```bash
python scripts/methods/sayplan/run.py --limit 1 --overwrite --verbose
```

Full run:

```bash
python scripts/methods/sayplan/run.py \
  --set valunseen \
  --overwrite --verbose
```

### Common Arguments

| Argument | Description |
| --- | --- |
| `--input-root PATH` | Instruction input directory. Defaults to the SemPathBench instruction root. |
| `--output-root PATH` | Directory for predictions and `summary.json`. Defaults to `resources/methods/baselines/SayPlan`. |
| `--set {valunseen,train,all}` | Instruction split to run. Defaults to `valunseen`. |
| `--model MODEL` | Gemini model override. If omitted, the runner reads `MODEL` from `scripts/methods/api_key.py`. |
| `--max-search-steps N` | Maximum number of SayPlan semantic-search expand/contract steps. Defaults to `20`. |
| `--max-replans N` | Maximum number of high-level replans after verifier feedback. Defaults to `5`. |
| `--max-json-retries N` | Number of retries for malformed LLM JSON. Defaults to `2`. |
| `--overwrite` | Replace existing predictions; otherwise, existing results are skipped. |
| `--limit N` | Run only the first N instructions for debugging. |
| `--verbose` | Print detailed runner, semantic-search, LLM-planning, A* completion, and verifier-feedback logs. |
| `--evaluate` | Deprecated compatibility option; metrics are always computed by the shared evaluator. |

## 2. Key Files

| File | Purpose |
| --- | --- |
| `run.py` | SemPathBench method runner, structured similarly to `scripts/methods/tutorial/run.py`. |
| `config.py` | Default method name, search budget, replan count, and Gemini timeout. |
| `adapters/sempathbench_loader.py` | Infers the map ID from the instruction path, loads map/instruction data, and constructs the SayPlan episode. |
| `adapters/map_to_scene_graph.py` | Converts SemPathBench room, object, and traversability maps into a SayPlan scene graph. |
| `graph/scene_graph.py` | Nodes, edges, and base query structures for the full scene graph. |
| `graph/scene_graph_view.py` | Collapsed/expanded visible graph view and semantic-search memory. |
| `llm/client.py` | Gemini REST helper; reads the default model from `scripts/methods/api_key.py`. |
| `llm/prompts.py` | SayPlan-style semantic-search and planning prompts. |
| `llm/schemas.py` | Validates LLM JSON responses and parses `goto(node_id)` / `done()` plans. |
| `planning/semantic_search.py` | Lets the LLM iteratively execute `expand_node` / `contract_node`. |
| `planning/high_level_planner.py` | Generates a grounded high-level navigation plan over the task graph. |
| `planning/classical_planner.py` | Uses A* to complete high-level `goto(...)` actions into a dense grid trajectory. |
| `planning/verifier.py` | Checks node visibility, validity, and reachability and produces textual feedback. |
| `planning/pipeline.py` | End-to-end SayPlan pipeline for one instruction. |

## 3. Adaptation to SemPathBench

### 3.0 Paper-Style Summary

To adapt SayPlan to SemPathBench, we replace the original 3D scene-graph representation with a task-specific 2D semantic scene graph constructed from the benchmark map layers. Each environment is represented as a hierarchy of scene, agent, room, and object nodes. Room and object instances are grounded by their SemPathBench instance identifiers and associated with grid-level navigation targets or traversable approach cells.

The core SayPlan procedure is preserved. The LLM first performs semantic search over a collapsed graph using `expand_node` and `contract_node`, then generates a grounded high-level plan over the discovered task-relevant subgraph. Because SemPathBench is a navigation benchmark rather than a mobile-manipulation setting, the action space is restricted to `goto(node_id)` followed by `done()`; manipulation actions such as `pickup`, `open`, and `release` are removed.

A classical A* planner completes the node-level plan on the full traversable grid and produces the dense trajectory required by the SemPathBench evaluator. The feedback-based replanning loop is also retained: invalid or unreachable node selections become textual feedback for a revised high-level plan. Final outputs follow the standard SemPathBench prediction format and include the trajectory, metrics, semantic-search trace, and replanning attempts.

This adaptation retains the SayPlan graph abstraction, semantic search, and classical-planner decomposition while replacing the original 3D manipulation simulator with a lightweight SemPathBench navigation verifier.

### 3.1 Input Format

The original SayPlan method uses a 3D scene graph. This runner reads SemPathBench instruction JSON and `map_state`, and infers the loadable `map_id` from the instruction-file path.

### 3.2 Scene Graph

The SemPathBench adapter builds a 2D scene graph:

```text
scene_0
├── agent_0
├── room_<id>
└── object_<id>
```

Room/object locations, categories, attributes, and parent relations come from SemPathBench map layers and instance metadata.

Grounding maps SemPathBench instance IDs to graph node IDs. For example, an object instance becomes `object_<id>`, and the LLM may select only a node visible in the current graph. The classical planner then maps that node to a room target or an object approach cell.

### 3.3 Collapsed-Graph Search

The initial prompt shows only `scene_0`, `agent_0`, and room nodes. The LLM manages the visible graph through `expand_node(node_id)` and `contract_node(node_id)` and cannot directly output a dense grid trajectory.

### 3.4 High-Level Plan

The LLM output format is fixed:

```text
goto(node_id)
done()
```

Each `node_id` must exist in the current task graph. The classical planner converts selected nodes into grid targets and connects them with A*.

This planner is the SemPathBench replacement for the low-level path planner in the original SayPlan method. The original system completes navigation over a 3D scene/pose graph; this adapter runs A* directly on the full SemPathBench traversability grid and returns the dense trajectory expected by the evaluator.

### 3.5 Feedback Replanning

If the verifier detects an invisible or invalid node, an unreachable target, or an A* failure, it returns textual feedback to the LLM. The process repeats up to `--max-replans` times.

### 3.6 Output

The adapter writes prediction JSON, trajectories, shared-evaluator metrics, trajectory images, and SayPlan metadata, including the semantic-search trace, task graph, high-level plan, and replanning attempts.
