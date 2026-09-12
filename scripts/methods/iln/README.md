# ILN for SemPathBench

This directory adapts **Intelligent LiDAR Navigation: Leveraging External Information and Semantic Maps with LLM as Copilot** to SemPathBench.

This is the single paper-facing ILN baseline. It selects one destination area,
evaluates passage costs, searches the area-passage graph, and realizes that route
on the grid. It does not use object-instance endpoints, ordered task
decomposition, instruction-derived avoid constraints, object-room output
correction, or replacement objects. It is not a full ROS/Gazebo reproduction of
the official repository.


Configure Gemini with either environment variables or `scripts/methods/api_key.py`:

```python
MODEL = "gemini-2.3-flash"
API_KEY = "your_gemini_api_key_here"
```


## Run

Paper-facing ILN baseline:

```bash
python scripts/methods/iln/run.py \
  --set valunseen \
  --grounding-mode llm
```

Smoke test without an API call:

```bash
python scripts/methods/iln/run.py --limit 1 --grounding-mode heuristic --overwrite
```

Outputs and caches default to `resources/methods/baselines/ILN`.

## Main Arguments

| Argument | Purpose |
| --- | --- |
| `--input-root PATH` | Instruction root, default `resources/instructions`. |
| `--output-root PATH` | Output root; default `resources/methods/baselines/ILN`. |
| `--set {valunseen,train,all}` | Instruction split. |
| `--grounding-mode {llm,heuristic,auto}` | Destination grounding mode; default `llm`. |
| `--model MODEL` | Gemini model override. |
| `--llm-cache-root PATH` | Gemini response cache. |
| `--graph-cache-root PATH` | Reserved graph cache root recorded in metadata. |
| `--event-mode {empty,llm}` | Benchmark default is `empty`. |
| `--history-mode {empty,file}` | Benchmark default is `empty`. |
| `--unknown-passage-cost FLOAT` | Base cost for passages with no experience, default `10.0`. |
| `--max-prompt-objects N` | Object inventory prompt cap. |
| `--passage-extraction {boundary,doorway,auto}` | Passage extraction mode. |
| `--allow-grid-fallback` | Allows explicit debug fallback for object endpoints/global grid search. |
| `--overwrite` | Regenerate predictions. |
| `--overwrite-llm-cache` | Refresh Gemini cache. |
| `--limit N` | Debug subset. |
| `--verbose` | Print detailed progress. |

## Files

| File | Role |
| --- | --- |
| `run.py` | Tutorial-style SemPathBench runner. |
| `pipeline.py` | One episode ILN adapter pipeline. |
| `graph_builder.py` | Builds room/passage graph from map layers. |
| `graph_serializer.py` | Serializes graph and inventory for prompts. |
| `grounding_adapter.py` | Single-destination-area grounding adapter. |
| `goal_resolver.py` | Resolves the selected area target to a traversable endpoint. |
| `passage_cost_evaluator.py` | Official-style ILN PassageCostEvaluator. |
| `event_monitor.py` | No-op benchmark NavigationEventMonitor interface. |
| `graph_planner.py` | Cost-aware area-passage route search. |
| `grid_realizer.py` | Dense grid A* realization through selected passage anchors. |
| `llm_client.py` | Gemini REST helper with cache. |
| `gap.md` | Explicit reproduction/adaptation gaps. |

## Paper-Style Adaptation Description

ILN converts the benchmark room and occupancy layers into an
area-passage graph. A narrow LLM interface selects exactly one destination area
from the instruction and area-level semantic landmarks. The PassageCostEvaluator
assigns passage costs, graph A* selects the topometric route, and local A*
realizes the selected passage sequence as a dense trajectory ending at the
destination-area medoid. It never predicts an object instance or intermediate
stage and does not consume instruction-derived route constraints.

Native osmAG files, ROS execution, live notification streams, and passage
history are unavailable in SemPathBench. We therefore derive the graph from
benchmark map layers, use offline grid realization, and leave notification and
history inputs empty by default. These benchmark-interface changes are recorded
in every prediction under `iln.benchmark_adaptations`.

## Input Contract

The adapter may use instruction text, `start_pose`, occupancy/room/object layers, `room_instances`, `object_instances`, and geometry derived from those fields.

It must not use `hard_constraints`, `soft_constraints`, instruction `objects`, `human_expert_trajectory`, evaluator feedback, or metric details during planning.

## Output

Predictions are written under
`resources/methods/baselines/ILN/<map_id>/<instruction_id>.json`. Records include:

- `trajectory`
- shared SemPathBench `metrics`
- `metrics_summary`
- `iln.area_passage_graph`
- `iln.grounding_adapter`
- `iln.passage_cost_evaluator`
- `iln.navigation_event_monitor`
- `iln.graph_planner`
- `iln.grid_realizer`
- `iln.method_level_deviations`

Failed episodes still write valid JSON and expose `iln.status` plus diagnostics.

