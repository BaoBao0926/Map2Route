# SayPlan Implementation Plan for SemPathBench

## 0. Goal

在 `scripts/methods/sayplan/` 下实现一个面向 SemPathBench 的 SayPlan baseline。

这个 baseline 的目标不是追求最高分，而是尽量忠实复现原始 SayPlan 的方法骨架，并只做 SemPathBench 输入/输出所必需的适配。

核心原则：

1. **尽量使用原本 SayPlan 的实现思想**。即使性能不好，也优先保留原论文中的 collapsed graph、expand/contract semantic search、high-level planning、classical path completion 和 simulator-feedback replanning。
2. **所有新代码放在 `scripts/methods/sayplan/` 下**。
3. **任何新代码都不能依赖 `SayPlan_Reconstruct`**。该目录只作为阅读参考，不作为 import 来源，不复用其 runtime module。
4. **入口和运行方式参考 `scripts/methods/tutorial/run.py`**，保持 SemPathBench method runner 的统一风格。

推荐最终命名：

```text
SayPlan
```

对外描述：

```text
SayPlan (2D scene-graph adaptation)
```

输出目录固定为：

```text
resources/methods/baselines/SayPlan
```

不要把新输出写到旧的 SayPlan 根目录或小写方法目录，也不要另起 `SayPlan-Faithful`、`sayplan` 等输出目录名。

---

## 1. What To Preserve From Original SayPlan

原始 SayPlan 的 Algorithm 1 是本实现的主骨架：

```text
Input:
  instruction I
  full scene graph G
  prompt P
  scene graph simulator psi
  classical path planner phi
  LLM

1. G' <- collapse(G)

Stage 1: Semantic Search
2. while command != terminate:
3.     command, node <- LLM(P, G', I)
4.     if command == expand:
5.         G' <- expand(node)
6.     if command == contract:
7.         G' <- contract(node)

Stage 2: Planning and Replanning
8. feedback = ""
9. while feedback != success:
10.    plan <- LLM(P, G', I, feedback)
11.    full_plan <- classical_path_planner(plan, G')
12.    feedback <- verify_plan(full_plan)

13. return full_plan
```

SemPathBench 版本只替换必要部分：

| Original SayPlan | SemPathBench adaptation |
| --- | --- |
| prebuilt 3D scene graph | graph built from 2D room/object/traversibility maps |
| floor-room-asset-object hierarchy | scene-room-object-pose-agent hierarchy |
| Dijkstra over pose graph | A* over SemPathBench traversibility grid |
| manipulation actions | removed by default; pure navigation actions only |
| scene graph simulator with affordance/state checks | navigation scene-graph verifier |
| executable robot plan | dense 2D trajectory for SemPathBench evaluator |

必须保留：

- `collapse()`
- `expand(node_id)`
- `contract(node_id)`
- LLM iterative semantic search
- high-level grounded node plan
- classical path completion
- textual feedback
- iterative replanning

不应该保留或新增：

- 不让 LLM 直接输出 dense grid trajectory
- 不把完整 graph 一次性塞给 LLM
- 不用 embedding retrieval 替代 expand/contract
- 不把 SemPathBench evaluator 的 `HCS/SCS/OSS` 反馈给 LLM
- 不给 LLM 暴露 benchmark-specific solver API，例如 `pass_between()`、`circle_room()`、`satisfy_soft_constraints()`

---

## 2. Relationship To `SayPlan_Reconstruct`

`scripts/methods/sayplan/SayPlan_Reconstruct/` 只作为参考材料。

它的实际 pipeline 更接近：

```text
LLM instance pruning
-> recursive part pruning
-> LLM plan
-> add kinematic relations
-> LLM replan
```

这和原 SayPlan 的核心流程不同：

```text
collapse
-> expand/contract semantic search
-> high-level planning
-> classical path completion
-> scene graph simulation
-> textual-feedback replanning
```

因此本实现不 import、不继承、不调用 `SayPlan_Reconstruct` 中的任何代码。

允许借鉴的只是设计经验：

- graph database 的概念
- LLM client abstraction 的概念
- prompt helper 的组织方式

但会在 `scripts/methods/sayplan/` 下重新实现。

---

## 3. Directory Structure

推荐目录：

```text
scripts/methods/sayplan/
├── README.md
├── plan.md
├── run.py
├── config.py
├── __init__.py
│
├── adapters/
│   ├── __init__.py
│   ├── sempathbench_loader.py
│   └── map_to_scene_graph.py
│
├── graph/
│   ├── __init__.py
│   ├── scene_graph.py
│   ├── scene_graph_view.py
│   └── topology_builder.py
│
├── llm/
│   ├── __init__.py
│   ├── client.py
│   ├── prompts.py
│   └── schemas.py
│
├── planning/
│   ├── __init__.py
│   ├── semantic_search.py
│   ├── high_level_planner.py
│   ├── classical_planner.py
│   ├── verifier.py
│   └── pipeline.py
│
└── utils/
    ├── __init__.py
    ├── geometry.py
    ├── grid.py
    ├── logging.py
    └── output.py
```

说明：

- `run.py` 负责 SemPathBench method runner：加载 instructions、处理 overwrite/limit/set、调用 pipeline、评估、写结果。
- `adapters/` 负责从 SemPathBench map/instruction 变成 SayPlan 输入。
- `graph/` 负责 full graph 和 visible graph view。
- `llm/` 负责 LLM 调用、prompt 和 JSON schema。
- `planning/` 负责 SayPlan 主算法。
- `utils/` 放几何、日志、输出辅助。

---

## 4. `run.py` Design

`run.py` 应该仿照 `scripts/methods/tutorial/run.py`。

保留类似结构：

```python
METHOD_NAME = "SayPlan"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "SayPlan"

def build_prediction_record(...):
    ...

def build_all(
    input_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
    instruction_set: str,
    limit: int | None,
    ...
) -> list[dict[str, object]]:
    ...

def parse_args() -> argparse.Namespace:
    ...

def main() -> None:
    ...
```

每条 instruction 的流程应和 tutorial 保持一致：

```text
load instruction
-> infer map_id / scene_id / instruction_id
-> build output path
-> if exists and not overwrite: skip
-> load map_state
-> run SayPlan pipeline
-> evaluate_prediction()
-> save trajectory image
-> write prediction JSON
-> update summary.json
```

实现细节必须和现有 method runner 对齐：

- `instruction_files = iter_instruction_files(input_root, instruction_set=...)`
- `map_id = map_id_from_instruction_path(instruction_file)`，不要优先使用 instruction payload 里的 `map_id` 来加载地图。
- `scene_id` 可以优先使用 instruction payload 里的 `map_id` 作为记录字段；如果没有，则 fallback 到 path-derived `map_id`。
- `instruction_id = instruction_id_from_payload(instruction_file, instruction)`
- `map_state = load_map_state(map_id)`
- 加载后必须检查 `map_state.get("map_key") == normalize_map_key(map_id)`；如果不一致，记录失败 `failure_reason = "map_load_mismatch"`，不要继续在 fallback 地图上规划。
- 输出路径使用 `output_root / normalize_map_key(map_id) / f"{instruction_id}.json"`。

原因：当前 ProcTHOR instruction payload 里的 `map_id` 可能是源场景名，例如 `procthor/train_00008`，但实际可加载的 map key 来自 instruction 文件路径，例如 `procthor/002_train`。如果直接用 payload `map_id` 调 `load_map_state()`，可能会错误 fallback 到默认地图。

推荐 CLI 参数：

| Parameter | Meaning |
| --- | --- |
| `--input-root` | instruction root，默认 `resources/instructions` |
| `--output-root` | method output root，默认 `resources/methods/baselines/SayPlan` |
| `--set` | `train` / `valunseen` / `all` |
| `--overwrite` | 是否重跑已有结果 |
| `--limit` | 调试时限制样本数 |
| `--model` | Gemini model name；不传时直接使用 `scripts/methods/api_key.py` 中定义的 `MODEL` |
| `--max-search-steps` | semantic search 最大步数 |
| `--max-replans` | replanning 最大轮数 |

默认配置固定为：

```python
DEFAULT_MAX_SEARCH_STEPS = 20
DEFAULT_MAX_REPLANS = 5
DEFAULT_MAX_JSON_RETRIES = 2
DEFAULT_GEMINI_TEMPERATURE = 0
DEFAULT_GEMINI_TIMEOUT_SECONDS = 90.0
```

这些值放在 `config.py`，`run.py` 的 CLI 参数只覆盖需要暴露的部分。`max_json_retries` 第一版可以不做 CLI 参数，但内部必须使用这个默认值。

注意：模型名不在 SayPlan 里硬编码默认值。默认模型必须来自：

```python
scripts/methods/api_key.py
MODEL = "..."
```

输出仍然遵循 method 统一格式：

```text
resources/methods/baselines/SayPlan/<map_id>/<instruction_id>.json
resources/methods/baselines/SayPlan/<map_id>/<instruction_id>.png
resources/methods/baselines/SayPlan/summary.json
```

---

## 5. Input Adapter

输入来自 SemPathBench：

```text
instruction
start_pose
map_state
  occupancy / traversibility
  room layer
  object instance layer
  metadata
```

Adapter 负责构造：

```python
SayPlanEpisode(
    map_id,
    scene_id,
    instruction_id,
    instruction_text,
    start_pose,
    map_state,
    full_graph,
)
```

如果当前 map_state 的字段格式和 ProcTHOR map `.npz/.json` 不完全一致，adapter 应该负责统一到内部格式。

内部格式建议：

```text
traversable_grid: H x W bool
room_instances: list[RoomInstance]
object_instances: list[ObjectInstance]
start_cell: (row, col)
metadata: dict
```

### 5.1 Required Map Fields

实现只依赖 `load_map_state()` 返回的 layered map JSON，不直接依赖 `*_metadata.json`、`*_maps.npz` 或 `*_thinggraph.json`。

必须存在：

```text
map_state["grid_size"]: int
map_state["layers"]["occupancy"]: grid_size x grid_size int grid
map_state["layers"]["room"]: grid_size x grid_size int grid
map_state["layers"]["object_instance"]: grid_size x grid_size int grid
map_state["room_instances"]: list[dict]
map_state["object_instances"]: list[dict]
map_state["layer_legends"]: dict
map_state["metadata"]: dict
```

`grid_size` 可能大于真实 ProcTHOR map width，因为 ProcTHOR layered map 会 pad 成 square。所有算法必须尊重 square grid bounds；不要假设 `metadata["map_info"]["W"] == grid_size`。

### 5.2 Coordinate Convention

所有内部轨迹和 grid 坐标统一使用：

```text
(row, col)
```

即：

```text
trajectory item = [row, col]
node.position = [row, col]
```

不要混用 world `(x, z)` 或 image `(x, y)`。如果 metadata 中有 world pose，只作为属性保留，不参与 SemPathBench trajectory 输出。

### 5.3 Start Cell Rule

起点解析顺序固定为：

1. 如果 `instruction["start_pose"]` 有整数 `row` 和 `col`，使用它；若不可通行，则用最近可通行格。
2. 否则返回失败：`failure_reason = "no_start_cell"`。

禁止读取 `human_expert_trajectory` 或 annotated `objects` 推断起点。

### 5.4 Inference Data Isolation

SayPlan 的 semantic search 和 high-level planning 只能接收 free-form
`instruction`。以下 annotation fields 不允许进入 prompt 或其他 inference
逻辑：

```text
objects
hard_constraints
soft_constraints
human_expert_trajectory
```

---

## 6. Map To Scene Graph

SemPathBench map 转 SayPlan graph：

```text
scene_0
  -> room nodes
      -> object instance nodes
  -> pose/topology nodes, optional but recommended
  -> agent_0
```

最小 node schema：

```json
{
  "id": "object_2",
  "type": "object",
  "category": "sofa",
  "name": "sofa_2",
  "room_id": "room_1",
  "position": [120, 85],
  "attributes": []
}
```

推荐 node types：

| Type | Meaning |
| --- | --- |
| `scene` | root node |
| `room` | room or region instance |
| `object` | object instance |
| `pose` | optional topology / doorway / waypoint node |
| `agent` | robot start state |

推荐 edges：

| Edge | Meaning |
| --- | --- |
| `contains` | scene contains room, room contains object |
| `adjacent_to` | room-room or pose-pose topology |
| `located_at` | agent located at pose/room |
| `approach` | optional object-to-approach-pose edge |

第一版可以先构造：

```text
scene -> room -> object
agent_0
```

随后再补：

```text
room adjacency
doorway / pose nodes
object approach cells
```

注意：graph 只能表达输入事实，不能写入 benchmark answer。

允许：

```text
object_2 has name sofa_2 and is in room_1
object_5 has name garbage_can_1 and is in room_2
```

不允许：

```text
object_2 is the correct target
room_1 is the only valid bedroom
```

### 6.1 Stable Node IDs

LLM-facing node IDs 必须稳定、短、可 JSON 安全表示：

```text
scene_0
room_<instance_id>
object_<instance_id>
agent_0
pose_<index>
```

不要把原始 object name 当 node id，例如 `DiningTable|7|3|0`。原始名字放在 `name` 字段。

### 6.2 Room Extraction

对于每个 `room_instances` item：

```text
room_id = f"room_{id}"
category = item["category"]
name = item["name"]
mask = cells where layers.room == id
position = mask centroid rounded to [row, col]
attributes = item["attributes"]
```

如果 room mask 为空，仍保留 room node，但：

```text
position = null
navigation_target = None
```

### 6.3 Object Extraction

对于每个 `object_instances` item：

```text
object_id = f"object_{id}"
category = item["category"]
name = item["name"]
mask = cells where layers.object_instance == id
position = mask centroid rounded to [row, col]
attributes = item["attributes"]
```

如果 object mask 为空：

```text
position = null
approach_cells = []
```

对象仍保留在 graph 中，因为 instruction 可能引用它，但 verifier/path planner 会报告 no approach cell。

### 6.4 Object-Room Assignment

对象归属房间的规则固定为：

1. 统计 object mask 中每个 cell 对应的 room layer id。
2. 选择 overlap cell 数最多且非 0 的 room id。
3. 如果没有 overlap，则用 object centroid 所在 cell 的 room id。
4. 如果仍没有，则选择最近 room centroid。
5. 如果没有任何 room，则挂到 `scene_0`。

记录：

```json
{
  "room_id": "room_3",
  "room_assignment_method": "footprint_overlap|centroid_cell|nearest_room|unassigned"
}
```

### 6.5 Prompt Graph vs Internal Graph

内部 graph 可以保存 masks、approach cells、room cells 等大对象。

传给 LLM 的 `visible_graph_json` 不能包含大规模 cell list。LLM 只看 compact node summary：

```json
{
  "nodes": [
    {
      "id": "room_1",
      "type": "room",
      "category": "bedroom",
      "name": "bedroom_1",
      "position": [120, 85],
      "child_count": 12,
      "visible_children": ["object_4", "object_9"]
    }
  ],
  "edges": [
    ["scene_0", "room_1", "contains"]
  ]
}
```

Attributes 需要截断：

```text
最多保留前 8 条 attribute，每条最多 120 chars。
```

---

## 7. Scene Graph View

需要两个层次：

1. `SceneGraph`: 完整 graph，不随 search 删除节点。
2. `SceneGraphView`: 当前 LLM 可见 graph，支持 collapse/expand/contract。

API：

```python
class SceneGraph:
    def to_json(self) -> dict: ...
    def get_node(node_id: str) -> GraphNode: ...
    def children(node_id: str) -> list[str]: ...
    def ancestors(node_id: str) -> list[str]: ...

class SceneGraphView:
    def collapse(full_graph: SceneGraph) -> "SceneGraphView": ...
    def expand(node_id: str) -> None: ...
    def contract(node_id: str) -> None: ...
    def visible_json(self) -> dict: ...
    def task_subgraph_json(self) -> dict: ...
```

初始 collapse：

```text
scene_0
  -> room_1
  -> room_2
  -> room_3
  -> room_4
  -> agent_0
```

`expand("room_1")` 后：

```text
room_1
  -> object_3
  -> object_8
  -> object_21
```

`contract("room_1")` 后隐藏其 children。

需要记录 memory：

```json
{
  "expanded_nodes": ["room_1", "room_2"],
  "contracted_nodes": ["room_1"],
  "commands": [
    {"command_name": "expand_node", "node_name": "room_1"},
    {"command_name": "contract_node", "node_name": "room_1"}
  ]
}
```

memory 用于 prompt，不需要无限累计完整 chat history。

Memory 中 command 名称也使用论文 API 名称：

```json
{
  "expanded_nodes": ["room_1", "room_2"],
  "contracted_nodes": ["room_1"],
  "commands": [
    {"command_name": "expand_node", "node_name": "room_1"},
    {"command_name": "contract_node", "node_name": "room_1"}
  ]
}
```

---

## 8. LLM Client

LLM 固定使用 Gemini，参考 `scripts/methods/lang2ltl/translator.py` 的实现方式。

不要使用 `SayPlan_Reconstruct` 里的 `GeminiVLMClient`，也不要引入 `google-genai` 风格的新依赖。推荐直接实现和 Lang2LTL 类似的轻量 REST helper：

```python
DEFAULT_GEMINI_TEMPERATURE = 0
DEFAULT_GEMINI_TIMEOUT_SECONDS = 90.0
GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

def configured_model() -> str:
    model = _api_key_module_value("MODEL")
    if not model:
        raise RuntimeError(
            "MODEL in scripts/methods/api_key.py is required for SayPlan."
        )
    return model

def configured_api_key() -> str | None:
    return _api_key_module_value("API_KEY") or os.environ.get("GEMINI_API_KEY")

def gemini_response_text(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> str:
    ...

def gemini_response_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> dict[str, object]:
    ...
```

Gemini REST 调用风格参考 `lang2ltl`；模型来源按 SayPlan 规则固定为 `api_key.py`：

```text
scripts/methods/api_key.py:
  MODEL = "..."
  API_KEY = "..."

fallback:
  GEMINI_API_KEY
```

这里的 fallback 只适用于 API key。SayPlan 的默认 model 不使用 `GEMINI_MODEL`，也不在本方法里硬编码 `gemini-2.5-flash` 等默认模型名。

所有 LLM 输出都应走：

```text
Gemini text response
-> strip markdown fence
-> extract JSON object
-> json.loads
-> schema validation
-> repair / retry if malformed
```

如果没有 API key，应直接报错并说明需要配置 `scripts/methods/api_key.py` 或 `GEMINI_API_KEY`。不要在正式 runner 中静默 fallback 到规则 planner。

---

## 9. Semantic Search

这是 SayPlan 的核心，必须按原论文保留。

输入：

```text
instruction
visible graph
memory
available APIs
```

输出 JSON：

```json
{
  "reasoning": "I need to inspect bedrooms because the instruction asks for a sofa in a bedroom without a garbage can.",
  "mode": "exploring",
  "command": {
    "command_name": "expand_node",
    "node_name": "room_1",
    "plan": null
  }
}
```

允许 `command.command_name`：

```text
expand_node
contract_node
none
```

Semantic search 终止条件是：

```text
mode == "planning"
```

或者：

```text
mode == "exploring" and command.command_name == "none"
```

第二种情况只作为兼容 malformed-but-usable 输出；正式 prompt 应鼓励进入 `mode: "planning"`。

循环：

```text
visible_graph = collapse(full_graph)

for step in range(max_search_steps):
    response = llm(...)
    if response.mode == "planning":
        break
    command = response.command.command_name
    node_id = response.command.node_name
    if command == "expand_node":
        visible_graph.expand(node_id)
    elif command == "contract_node":
        visible_graph.contract(node_id)
    elif command == "none":
        break
```

如果 LLM 输出非法 node：

```text
record invalid_command
append textual feedback for the next semantic-search attempt
retry up to max_json_retries
```

如果超过最大步数：

```text
continue to planning with current visible graph
planning_failed_reason = "semantic_search_max_steps"
```

不要用 retrieval top-k 替代这个过程。

---

## 10. High-Level Planner

SemPathBench 是 pure navigation，因此 action space 默认只保留：

```text
goto(node_id)
```

可接受输出：

```json
{
  "reasoning": "The target sofa is object_2 because it is in room_2 and that room has no garbage can.",
  "mode": "planning",
  "command": {
    "command_name": "none",
    "node_name": null,
    "plan": ["goto(object_2)", "done()"]
  }
}
```

多事件输出：

```json
{
  "mode": "planning",
  "command": {
    "command_name": "none",
    "node_name": null,
    "plan": ["goto(room_1)", "goto(object_2)", "done()"]
  }
}
```

禁止：

```text
LLM output grid coordinates
LLM output dense path
LLM invent node IDs
```

如果 task-relevant graph 不足以完成指令，也允许输出：

```json
{
  "reasoning": "No visible node satisfies the instruction.",
  "mode": "planning",
  "command": {
    "command_name": "none",
    "node_name": null,
    "plan": ["done()"]
  }
}
```

内部 parser 将 plan string 解析成：

```python
HighLevelAction(action="goto", target="object_2")
HighLevelAction(action="done", target=None)
```

只接受：

```text
goto(<node_id>)
done()
```

其他动作交给 verifier 产生 feedback。

---

## 11. Classical Path Completion

原 SayPlan 用 classical path planner 连接 high-level nodes。

SemPathBench 版本使用：

```text
A* over traversibility grid
```

这是合理适配，因为 benchmark 本身是 2D grid map。

流程：

```text
start cell
-> target cell for high-level node 1
-> target cell for high-level node 2
-> ...
```

Room target：

```text
room mask
-> representative traversable cell near centroid
```

Object target：

```text
object mask
-> dilated boundary
-> candidate traversable approach cells
-> nearest reachable candidate from current position
```

不要直接把 object center 当 robot goal，因为 object footprint 可能不可通行。

如果某段 A* 失败：

```text
return partial trajectory
record unreachable segment
send textual feedback to replanner
```

Path completion 返回结构固定为：

```python
PathCompletionResult(
    trajectory: list[list[int]],
    segments: list[PathSegmentResult],
    success: bool,
    failure_reason: str | None,
)
```

每个 segment：

```python
PathSegmentResult(
    from_node: str,
    to_node: str,
    start: tuple[int, int],
    goal: tuple[int, int] | None,
    reachable: bool,
    path_length: float,
    failure_reason: str | None,
)
```

如果第一个目标不可达，trajectory 至少包含 start cell，除非 start cell 本身无法确定。

---

## 12. Navigation Verifier

原 SayPlan 的 `verify_plan()` 是 scene graph simulator。

SemPathBench 版本实现 navigation-specific verifier。

第一版检查：

| Check | Feedback example |
| --- | --- |
| node exists | `Target node object_99 does not exist.` |
| target visible in task graph | `Target object_2 is not visible in the task-relevant graph.` |
| action valid | `Action pickup is invalid for this navigation-only adaptation.` |
| reachable | `Target object_2 is unreachable from the current position.` |
| object approach point exists | `No traversable approach cell exists near object_2.` |
| basic semantic consistency | `The selected object_2 is in room_1, which contains a garbage can.` |

重要限制：

Verifier 不能直接调用最终 evaluator 指标作为反馈。

禁止反馈：

```text
Your HCS is 0.5
Your SCS is too low
Move farther from dog_bed_1 to improve SCS
```

允许反馈环境事实：

```text
The planned target room contains a garbage can.
The target object is unreachable.
The target node does not exist.
```

Verifier 输入固定为：

```python
verify(
    instruction: dict,
    task_graph_view: SceneGraphView,
    high_level_plan: list[HighLevelAction],
    path_result: PathCompletionResult,
) -> VerificationResult
```

输出：

```python
VerificationResult(
    success: bool,
    feedback: str,
    failures: list[dict[str, object]],
)
```

如果 success：

```text
feedback = "success"
```

Verifier 不负责计算最终 SemPathBench metrics；metrics 只在 `run.py` 写 prediction 前由 `evaluate_prediction()` 计算。

---

## 13. Iterative Replanning

循环：

```python
feedback = ""

for iteration in range(max_replans):
    high_level_plan = llm_plan(instruction, task_graph, feedback)
    dense_path = classical_path_completion(high_level_plan)
    feedback = verifier.verify(high_level_plan, dense_path)

    if feedback == "success":
        return dense_path

return best_or_last_path
```

`best_or_last_path` 规则固定为：

1. 优先返回第一个 verifier success 的 path。
2. 如果没有 success，返回最长的非空 trajectory。
3. 如果所有 attempt 都没有 trajectory，但 start cell 存在，返回 `[start_cell]`。
4. 否则返回 `[]`。

默认：

```text
max_replans = 5
```

失败时仍应写 prediction record：

```json
{
  "planning_failed": true,
  "failure_reason": "...",
  "trajectory": [...]
}
```

如果没有任何有效 trajectory，可以输出空 trajectory，但必须记录失败原因。

---

## 14. Soft Constraints

原 SayPlan 不是 soft path-cost planner。

因此 `SayPlan` 默认不专门优化 soft constraints。

对于软约束：

```text
LLM may mention relevant nodes in high-level plan
A* still uses shortest path
SCS may naturally be poor
```

这是可接受的，因为本 baseline 的目标是 faithfully adapt SayPlan，而不是为 SemPathBench 写一个强 constraint planner。

如果后续想增强，需要作为另一个实验清楚区分；本计划不实现增强版，也不把增强版输出混到 `resources/methods/baselines/SayPlan`。

增强版可能允许：

```text
avoid(node_id)
weighted A*
```

但不能和本计划的 SayPlan 主版本混在一起。

---

## 15. Hard Constraints And Geometry Constraints

对 SemPathBench 中不同约束的处理：

| Constraint type | Faithful SayPlan treatment |
| --- | --- |
| go to object | native `goto(object)` |
| go to room | native `goto(room)` |
| object in room | graph hierarchy helps |
| room without object | semantic search can inspect rooms |
| ordered targets | high-level ordered `goto` plan |
| nearest/farthest | expose coordinates, let LLM reason |
| avoid region | not native; verifier may catch obvious forbidden target, but A* does not optimize it |
| soft near/far | not native |
| pass between | not native |
| circle around | not native |
| visibility | not native in first version |

这种能力边界需要在 README 和 experiment note 里写清楚。

---

## 16. Prediction Record

输出 JSON 应保留 SemPathBench method 统一字段，并增加 SayPlan trace。

建议：

```json
{
  "version": 1,
  "method": "SayPlan",
  "prediction_id": "SayPlan_xxxxxxxx",
  "map_id": "...",
  "scene_id": "...",
  "instruction_id": "...",
  "instruction_file": "...",
  "created_at": "...",
  "trajectory": [[1, 2], [1, 3]],
  "metrics": {...},
  "metrics_summary": {...},
  "runtime_seconds": 0.0,
  "SayPlan": {
    "variant": "2d_scene_graph_adaptation",
    "model": "...",
    "planning_failed": false,
    "failure_reason": null,
    "semantic_search_trace": [
      {
        "step": 1,
        "mode": "exploring",
        "command": {
          "command_name": "expand_node",
          "node_name": "room_1",
          "plan": null
        },
        "reasoning": "...",
        "visible_node_count": 12
      }
    ],
    "task_subgraph": {...},
    "planning_attempts": [
      {
        "iteration": 1,
        "raw_llm_response": {...},
        "high_level_plan": ["goto(object_2)", "done()"],
        "path_result": {
          "reachable": true,
          "length": 42
        },
        "verifier_feedback": "success",
        "failures": []
      }
    ],
    "final_high_level_plan": ["goto(object_2)", "done()"]
  },
  "instruction": {
    "text": "...",
    "difficulty_level": "...",
    "template_instruction_id": "..."
  }
}
```

Trajectory PNG 保存逻辑可以仿照 tutorial：

```python
save_trajectory_image_for_record(
    record,
    map_state,
    trajectory,
    output_path,
    nested_record_keys=("SayPlan",),
)
```

---

## 17. Prompt Design

我已经核对了两个 prompt 来源：

1. 原始 SayPlan 论文在 Appendix J 给出了 `Input Prompt Structure`，Appendix K/L 给出了 semantic search 和 iterative replanning 的示例交互。论文提供的是 prompt 结构、环境函数、API、输出格式和示例，不是一个完整可直接复制的代码文件。
2. `SayPlan_Reconstruct/utils/llm_utils/gemini_message.py` 提供了第三方重构 prompt，但它是 `instance selection / part selection / task planning / kinematic replanning` 风格，和原文 Appendix J 的 `expand_node / contract_node / verify_plan` 结构不一致。

因此本实现的 prompt 应以论文 Appendix J/K/L 为准；第三方库 prompt 只能作为参考，不能照搬。

原文 prompt 的固定结构包括：

```text
Agent Role
Environment Functions
Environment State
Environment API
Output Response Format
Example
Instruction
3D Scene Graph
Memory
Feedback
```

SemPathBench 版本应尽量保持这些块，只对 environment functions 做 pure-navigation 适配。

原文 Environment API 是：

```text
expand_node(<node>)
contract_node(<node>)
verify_plan()
```

所以本实现中 prompt 也应使用 `expand_node` / `contract_node` 这两个名字，而不是随意改成 `expand` / `contract`。

### 17.1 Semantic Search Prompt

```text
Agent Role:
You are an excellent graph planning agent. Given a graph representation of an
environment, you can explore the graph by expanding nodes to find the items of
interest. You can then use this graph to generate a step-by-step navigation plan
that the agent can follow to solve a given instruction.

Environment Functions:
goto(<node>): Move the agent to any room, object approach node, or pose node.
done(): Call when the navigation task is completed.

Environment State:
located_at(<node>): The agent is currently located at a node.
reachable(<node>): The node can be reached by the classical path planner.
contains(<room>, <object>): The object is located in the room.
adjacent_to(<node>, <node>): Two rooms or poses are topologically connected.

Environment API:
expand_node(<node>): Reveal objects or lower-level nodes connected to a room/scene node.
contract_node(<node>): Hide lower-level nodes, reducing graph size for memory constraints.
verify_plan(): Verify generated plan in the scene graph navigation environment.

Rules:
- Use only node IDs in the visible graph.
- Expand a node if its hidden children may contain task-relevant entities.
- Contract a node if its visible children are irrelevant.
- Do not invent nodes.
- Switch to planning mode only when enough grounded entities are visible.

Output Response Format:
Return JSON parseable by Python json.loads:
{
  "reasoning": "brief reason for the next command",
  "mode": "exploring" OR "planning",
  "command": {
    "command_name": "expand_node" OR "contract_node" OR "none",
    "node_name": "node_id or null",
    "plan": null
  }
}

Instruction:
{instruction}

3D Scene Graph:
{visible_graph_json}

Memory:
{memory_json}

Feedback:
{feedback}
```

### 17.2 Planning Prompt

```text
Agent Role:
You are an excellent graph planning agent. Given a graph representation of an
environment, generate a high-level grounded navigation plan that the agent can
follow to solve the instruction.

Environment Functions:
goto(<node>): Move the agent to any room, object approach node, or pose node.
done(): Call when the navigation task is completed.

Environment API:
verify_plan(): Verify generated plan in the scene graph navigation environment.

Rules:
- Generate a short plan using only node IDs present in the task-relevant graph.
- Do not output grid coordinates.
- Do not output a dense path.
- Do not invent node IDs.
- A classical path planner will connect the selected targets.

Output Response Format:
Return JSON parseable by Python json.loads:
{
  "reasoning": "brief reason for the selected high-level plan",
  "mode": "planning",
  "command": {
    "command_name": "none",
    "node_name": null,
    "plan": [
      "goto(node_id)",
      "done()"
    ]
  }
}

Instruction:
{instruction}

3D Scene Graph:
{task_graph_json}

Memory:
{memory_json}

Feedback:
{feedback}
```

### 17.3 Replanning Prompt

```text
Use the same planning prompt structure as above.

The only dynamic change is the Feedback field, which contains external textual
feedback from the scene graph navigation simulator, matching the original
SayPlan iterative replanning structure.

Instruction:
{instruction}

3D Scene Graph:
{task_graph_json}

Previous Plan:
{previous_plan_json}

Feedback:
{feedback}

Generate a corrected high-level plan.

Use only existing node IDs.
Do not invent paths or coordinates.
The classical path planner will handle low-level routing.
```

### 17.4 Prompt Fidelity Notes

- The paper uses `mode: "exploring" OR "planning"` and a nested `command` object. This should be preserved.
- The paper uses `expand_node(<node>)` and `contract_node(<node>)`. This should be preserved.
- The paper includes manipulation functions such as `pickup`, `release`, `open`, `close`, and `turn_on/off`. These should be removed from the SemPathBench prompt because this adaptation is pure navigation.
- The paper's `goto(<pose>)` becomes `goto(<node>)`, where node can be a room, pose, or object approach node.
- The paper's examples in Appendix K/L should be converted into SemPathBench-style examples, not copied literally, because coffee-making manipulation is outside this benchmark.
- The third-party reconstruction prompt should not be used as the main prompt because it implements relevance pruning rather than original SayPlan graph exploration.

---

## 18. Testing Plan

Unit tests should cover:

1. map-to-graph conversion
2. object-room assignment
3. `collapse()`
4. `expand(node)`
5. `contract(node)`
6. semantic search loop with mocked Gemini responses
7. high-level plan validation
8. object approach cell selection
9. A* segment connection
10. verifier feedback
11. replanning loop with mocked Gemini responses
12. prediction record writing
13. path-derived map id vs payload map id behavior
14. prompt graph serialization excludes large cell lists

Smoke test:

```bash
python scripts/methods/sayplan/run.py --limit 1 --overwrite
```

Real LLM test, using `MODEL` from `scripts/methods/api_key.py`:

```bash
python scripts/methods/sayplan/run.py --limit 1 --overwrite
```

Default benchmark run:

```bash
python scripts/methods/sayplan/run.py --set valunseen --overwrite
```

---

## 19. Implementation Contracts

本节是实现时的固定接口。除非发现现有数据格式与这里冲突，否则按这里写代码。

### 19.1 Core Data Objects

```python
@dataclass
class SayPlanEpisode:
    map_id: str
    scene_id: str
    instruction_id: str
    instruction_file: Path
    instruction: dict[str, object]
    instruction_text: str
    map_state: dict[str, object]
    traversable: list[list[bool]]
    start_cell: tuple[int, int] | None
    full_graph: SceneGraph

@dataclass
class GraphNode:
    id: str
    type: str
    category: str | None
    name: str
    position: tuple[int, int] | None
    room_id: str | None
    attributes: list[str]
    metadata: dict[str, object]

@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    metadata: dict[str, object]
```

`metadata` 可以保存 masks、approach cells、assignment method 等内部信息；`visible_json()` 必须过滤这些大字段。

### 19.2 LLM Response Schema

所有 Gemini 调用统一解析成：

```python
@dataclass
class SayPlanCommand:
    command_name: str
    node_name: str | None
    plan: list[str] | None

@dataclass
class SayPlanLLMResponse:
    reasoning: str
    mode: str
    command: SayPlanCommand
    raw_text: str
    raw_json: dict[str, object]
```

合法值：

```text
mode: exploring | planning
command_name: expand_node | contract_node | none
plan item: goto(<node_id>) | done()
```

Validation failures use these reason strings:

```text
malformed_json
missing_command
invalid_mode
invalid_command_name
invalid_node_id
invalid_plan_item
```

### 19.3 Pipeline Result

```python
@dataclass
class SayPlanRunResult:
    trajectory: list[list[int]]
    final_high_level_plan: list[str]
    semantic_search_trace: list[dict[str, object]]
    planning_attempts: list[dict[str, object]]
    planning_failed: bool
    failure_reason: str | None
    runtime: dict[str, object]
```

Known `failure_reason` values:

```text
no_start_cell
map_load_mismatch
semantic_search_max_steps
llm_json_failed
llm_invalid_command
llm_invalid_plan
no_visible_task_graph
no_path_to_target
max_replans_exceeded
empty_trajectory
```

### 19.4 File And Cache Layout

Output files:

```text
resources/methods/baselines/SayPlan/<normalized_map_id>/<instruction_id>.json
resources/methods/baselines/SayPlan/<normalized_map_id>/<instruction_id>.png
resources/methods/baselines/SayPlan/summary.json
```

LLM cache, if implemented:

```text
resources/methods/baselines/SayPlan/_llm_cache/<hash>.json
```

Cache key must include:

```text
model
prompt_type
system_prompt
user_prompt
```

Do not put cache files under `scripts/methods/sayplan/`.

### 19.5 Reused Existing Helpers

Use existing SemPathBench helpers where possible:

```python
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state, normalize_map_key
from scripts.methods.util.grid_astar import build_traversable_grid, astar_path, nearest_traversable_cell
from scripts.methods.util.instructions import (
    DEFAULT_INSTRUCTION_SET,
    INSTRUCTION_SET_CHOICES,
    INSTRUCTION_ROOT,
    instruction_id_from_payload,
    iter_instruction_files,
    load_instruction,
    map_id_from_instruction_path,
)
from scripts.methods.util.methods import save_trajectory_image_for_record
from scripts.evaluation.evaluate_prediction import evaluate_prediction
```

Do not duplicate these unless SayPlan needs a small wrapper.

### 19.6 Things To Avoid During Implementation

- Do not import from `scripts/methods/sayplan/SayPlan_Reconstruct`.
- Do not read `resources/maps/.../*_metadata.json` directly in the first version.
- Do not read `*_thinggraph.json` directly in the first version.
- Do not put object footprint cells into Gemini prompts.
- Do not use `human_expert_trajectory`, annotated `objects`, `hard_constraints`, or `soft_constraints` during inference.
- Do not use final evaluator metrics as replanning feedback.
- Do not silently fallback to default map if `load_map_state(map_id)` fails; record the failure.

---

## 20. Implementation Phases

### Phase 1: Tutorial-style runner

Implement:

```text
run.py
config.py
utils/output.py
```

Goal:

```text
same input/output behavior as tutorial method
```

### Phase 2: Graph adapter

Implement:

```text
adapters/sempathbench_loader.py
adapters/map_to_scene_graph.py
graph/scene_graph.py
```

Goal:

```text
load one instruction and build scene-room-object-agent graph
```

### Phase 3: Collapse / expand / contract

Implement:

```text
graph/scene_graph_view.py
```

Goal:

```text
faithful visible graph manipulation
```

### Phase 4: Mocked Gemini semantic search

Implement:

```text
llm/client.py
llm/prompts.py
planning/semantic_search.py
```

Goal:

```text
complete semantic search loop with mocked Gemini responses
```

### Phase 5: Real LLM integration

Implement:

```text
JSON schema validation
retry / repair
provider config
```

Goal:

```text
LLM can issue expand_node/contract_node commands and switch to planning mode
```

### Phase 6: High-level planner

Implement:

```text
planning/high_level_planner.py
```

Goal:

```text
LLM outputs goto(node_id) plan
```

### Phase 7: Classical planner

Implement:

```text
planning/classical_planner.py
utils/grid.py
```

Goal:

```text
connect high-level targets with grid A*
```

### Phase 8: Verifier and replanning

Implement:

```text
planning/verifier.py
planning/pipeline.py
```

Goal:

```text
plan -> path -> verify -> feedback -> replan loop
```

### Phase 9: Logging and evaluation integration

Goal:

```text
write full trace
save PNG
evaluate_prediction()
write summary.json
```

---

## 21. Expected Weaknesses

Because this implementation intentionally stays close to original SayPlan, it may perform poorly on some SemPathBench constraints.

Expected weak cases:

- soft near/far constraints
- path-shape constraints such as `between` or `circle`
- strict global avoidance if not expressible as high-level target selection
- distance/count/negation reasoning by LLM
- long semantic search over many similar rooms
- object instances with ambiguous names or sparse metadata

These weaknesses should be treated as honest baseline behavior, not necessarily bugs.

If stronger performance is needed later, it should be planned as a separate experiment and must not change the outputs under `resources/methods/baselines/SayPlan`.

---

## 22. Final Pipeline

The final intended pipeline is:

```text
SemPathBench instruction + map
        |
        v
load map_state and instruction
        |
        v
build hierarchical scene graph
        |
        v
collapse graph
        |
        v
LLM semantic search
  expand_node / contract_node / switch to planning mode
        |
        v
task-relevant visible subgraph
        |
        v
LLM high-level goto(node) plan
        |
        v
grid A* path completion
        |
        v
navigation scene-graph verifier
        |
        +---- failure feedback ----> LLM replan
        |
      success or max replans
        |
        v
dense 2D trajectory
        |
        v
shared SemPathBench evaluator
        |
        v
prediction JSON + PNG + summary
```

This is the version I would implement first.
