# Optimal Scene Graph Planning Adaptation Plan

## 0. Goal

在 SemPathBench 中实现一个面向 `Optimal Scene Graph Planning with Large Language Model Guidance` 的 method adapter。

这个适配版本的目标是尽量 faithful 地保留原方法的核心结构：

```text
SemPathBench map + instruction
        |
        v
floor-room-object-occupancy scene graph
        |
        v
scene-graph entity grounding
        |
        v
grounded mission -> LTL formula / automaton
        |
        v
AMRA*-style multi-resolution product planning
        |
        v
occupancy-level executable trajectory
        |
        v
shared SemPathBench evaluator
```

这份计划的关键修订点是：

> 不要把方法重写成 `occupancy product A* + scene-graph heuristic`。主线 planner 应该是一个 integrated AMRA*-style hierarchical product planner。根据论文，AMRA* 的 anchor space 是 `X0 = V x Q`，其中 `V` 是底层 scene-graph / free-space node set，`Q` 是 automaton states；object、room、floor 层是 attribute-associated subsets `Xk = Vk x Q`，并通过同一个 AMRA* process 参与搜索。

同时必须遵守当前项目要求：

1. **不 import、不调用、不依赖官方 repo 的运行时代码**。`scripts/methods/osgllm/LLM-Scene-Graph-LTL-Planning` 只作为阅读参考。
2. **代码结构尽量对齐 `scripts/methods/tutorial/run.py`**，尤其是 runner 输入、输出、summary、overwrite、limit、set、evaluation 的组织方式。
3. **尽量保留原方法，不把 unsupported case 偷偷改成另一个 solver**。处理不了的 SemPathBench instruction component 要显式记录，并写入 unsupported 文档。

对外方法名建议使用：

```text
Optimal Scene Graph Planning
```

代码目录：

```text
scripts/methods/osgllm
```

默认输出目录：

```text
resources/methods/baselines/OSGLLM
```

---

## 1. What To Preserve From The Original Method

官方方法的核心不是“用 scene graph 给 A* 一个更聪明的 heuristic”，而是：

- 构造 floor / room / object / occupancy 层级 scene graph；
- 将自然语言中的 room/object 引用 grounding 到唯一 scene graph entity；
- 将 grounded mission 转成 LTL；
- 将 LTL 转成 automaton；
- 在 **scene graph node / attribute region × automaton state** 的 product space 中规划；
- 使用 LTL-aware anchor heuristic；
- 使用 LLM scene-graph heuristic；
- 使用 AMRA*-style multi-resolution multi-heuristic search；
- 最终输出 occupancy-level path。

SemPathBench 版本不能运行官方 C++/Python 代码，因此需要本地实现这些接口。但概念边界必须保留：

| Original Component | SemPathBench Local Implementation |
| --- | --- |
| Gibson scene graph | 从 SemPathBench `occupancy/room/object_instance` layers 构造 floor-room-object-occupancy graph |
| `enter(room_x)` / `reach(object_y)` | 使用 room cells / object approach cells 定义 AP label regions |
| GPT UUID conversion | 使用 compact scene graph prompt + deterministic validation |
| GPT-to-Spot LTL | 本地 prefix LTL parser / validator / Formula backend |
| Spot automaton | 默认使用本地 LTL progression；Spot 可作为 future optional backend |
| AMRA* C++ planner | 本地 AMRA*-style multi-resolution product planner，anchor 为 `V x Q` |
| LLM heuristic | 本地 Gemini helper，离线/缓存生成 function-call guidance，在线转成 heuristic cost |

如果某一版实现没有 integrated AMRA*-style search，只能记录为 planner substitution，不能在 metadata 中 claim full AMRA*。

---

## 2. Directory Structure

推荐代码结构：

```text
scripts/methods/osgllm/
├── __init__.py
├── __main__.py
├── README.md
├── plan.md
├── config.py
├── pipeline.py
├── run.py
│
├── adapters/
│   ├── __init__.py
│   └── sempathbench_loader.py
│
├── scene_graph/
│   ├── __init__.py
│   ├── graph_types.py
│   ├── builder.py
│   ├── hierarchy.py
│   └── propositions.py
│
├── grounding/
│   ├── __init__.py
│   ├── entity_grounder.py
│   ├── metric_resolver.py
│   └── unsupported.py
│
├── language/
│   ├── __init__.py
│   ├── client.py
│   ├── prompts.py
│   ├── translator.py
│   └── validator.py
│
├── planning/
│   ├── __init__.py
│   ├── product_state.py
│   ├── multi_resolution_domain.py
│   ├── amra_planner.py
│   ├── ltl_heuristic.py
│   ├── scene_graph_heuristic.py
│   ├── llm_heuristic.py
│   └── reconstruction.py
│
└── docs/
    └── unsupported_cases.md
```

说明：

- `run.py` 只负责 method runner 外壳，结构对齐 tutorial。
- `pipeline.py` 负责单条 episode 的 end-to-end logic。
- `scene_graph/` 构建 base graph `V,E` 和 object/room/floor attribute regions，不新增 SemPathBench-specific planning resolution。
- `grounding/` 负责 scene-graph entity grounding 和 deterministic geometric resolver。
- `language/` 负责 LLM prompt、LTL translation、validation。
- `planning/` 负责 product-state、四层 multi-resolution domain、AMRA*-style planner、heuristics、path reconstruction。
- `docs/unsupported_cases.md` 记录当前不能 faithfully support 的 instruction cases。

---

## 3. Runner Contract

`run.py` 应尽量照着 `scripts/methods/tutorial/run.py` 写，保留同一类函数形状：

```python
METHOD_NAME = "Optimal Scene Graph Planning"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "OSGLLM"

def scene_id_from_instruction(instruction, map_id):
    ...

def build_osgllm_trajectory(map_state, instruction, *, config):
    ...

def build_prediction_record(...):
    ...

def build_all(input_root, output_root, *, overwrite, instruction_set, limit, ...):
    ...

def parse_args():
    ...

def main():
    ...
```

`build_all` 行为和 tutorial 保持一致：

- 用 `iter_instruction_files(...)` 选择 `--set valunseen/train/all`。
- 支持 `--limit`。
- 已有输出且没有 `--overwrite` 时跳过。
- 每条 episode 调用 `load_map_state(map_id)`。
- 生成 trajectory 后调用 shared `evaluate_prediction(...)`。
- 调用 `save_trajectory_image_for_record(...)` 保存 PNG。
- 写单条 prediction JSON。
- 每条 episode 后刷新 `summary.json`。
- `summary.json` 格式沿用 tutorial utils 的 `write_summary(...)`。

推荐 CLI：

```bash
python scripts/methods/osgllm/run.py --limit 1 --overwrite --verbose
```

全量示例：

```bash
python scripts/methods/osgllm/run.py \
  --set valunseen \
  --planner amra \
  --translation-mode llm \
  --overwrite --verbose
```

推荐参数：

| 参数 | 作用 |
| --- | --- |
| `--input-root PATH` | instruction root，默认 `resources/instructions` |
| `--output-root PATH` | 输出目录，默认 `resources/methods/baselines/OSGLLM` |
| `--set {valunseen,train,all}` | instruction split |
| `--model MODEL` | Gemini model override |
| `--llm-cache-root PATH` | LLM grounding / translation / heuristic cache |
| `--translation-mode {llm,heuristic}` | 默认 `llm`，`heuristic` 仅 smoke test |
| `--planner {amra,anchor-only,debug-fast}` | 默认 `amra` |
| `--disable-llm-heuristic` | AMRA* ablation |
| `--max-planning-seconds N` | 单 episode planning timeout |
| `--w1 FLOAT` | AMRA* inflation weight |
| `--w2 FLOAT` | AMRA* suboptimality / queue selection weight |
| `--overwrite` | 覆盖已有 prediction |
| `--overwrite-llm-cache` | 重新调用 LLM |
| `--limit N` | 调试用 |
| `--verbose` | 打印 grounding / LTL / planning 细节 |
| `--evaluate` | 兼容旧参数，metrics 总是计算 |

`anchor-only` 是 ablation：只跑 `X0=VxQ` anchor search，不启用 object/room/floor auxiliary queues。它可以帮助 debug，但不能作为 full method 主结果。

`debug-fast` 是 smoke test：从 formula 中提取 sequential eventual AP 后串接 A*。它必须明确记录为非 faithful planner。

---

## 4. Prediction JSON Schema

每条输出 JSON 应保留 SemPathBench method 通用字段：

```json
{
  "version": 1,
  "method": "Optimal Scene Graph Planning",
  "prediction_id": "...",
  "map_id": "procthor/003_valunseen",
  "scene_id": "procthor/003_valunseen",
  "instruction_id": "instruction_000002",
  "difficulty_level": "hard",
  "instruction_file": "...",
  "created_at": "...",
  "trajectory": [[471, 314], [470, 314]],
  "metrics": {},
  "metrics_summary": {},
  "runtime_seconds": 0.0,
  "runtime": {
    "seconds": 0.0,
    "status": "written"
  },
  "instruction": {
    "text": "...",
    "difficulty_level": "hard",
    "template_instruction_id": "..."
  },
  "osgllm": {}
}
```

`osgllm` metadata 建议包含：

```json
{
  "status": "SUCCESS",
  "scene_graph_summary": {
    "floor_count": 1,
    "room_count": 8,
    "object_count": 276,
    "traversable_cell_count": 12345,
    "room_adjacency_count": 12,
    "object_edge_count": 0
  },
  "grounding": {
    "grounded_instruction": "...",
    "entity_bindings": [],
    "virtual_regions": [],
    "ambiguous_bindings": [],
    "unsupported_constraints": []
  },
  "ltl": {
    "translation_mode": "llm",
    "raw_ltl": "...",
    "normalized_ltl": "...",
    "atomic_propositions": [],
    "validation": {}
  },
  "planner": {
    "planner_type": "amra_hierarchical_product",
    "planning_sets": ["X0=VxQ", "X1=V_objectxQ", "X2=V_roomxQ", "X3=V_floorxQ"],
    "semantic_attribute_levels": ["object", "room", "floor"],
    "anchor_space": "X0=VxQ",
    "use_ltl_heuristic": true,
    "use_llm_heuristic": true,
    "expanded_states": {
      "anchor": 0,
      "object": 0,
      "room": 0,
      "floor": 0
    },
    "first_solution_time": null,
    "first_solution_cost": null,
    "final_solution_time": null,
    "final_solution_cost": null,
    "solution_cost_history": [],
    "inflation_history": [],
    "optimality_reached": false,
    "timeout_seconds": null
  },
  "method_level_deviations": []
}
```

如果失败，仍然写 prediction JSON。trajectory 可以退回 start-only trajectory，但 `osgllm.status` 必须说明真实失败原因。不要把 fallback path 伪装成成功。

---

## 5. Scene Graph And Attribute Regions

论文中的 scene graph 表示为：

```text
G = (V, E, {A_k})
```

其中 `V` 是底层可规划 node set，`E` 是底层 connectivity，`A_k` 是不同层级的 attribute sets。每个 attribute `a in A_k` 对应一个底层 node subset：

```text
V_a subset V
```

SemPathBench 适配时，推荐令：

```text
V = traversable occupancy cells
E = 8-connected traversable grid edges
A_1 = object attributes
A_2 = room attributes
A_3 = floor attributes
```

因此 hierarchy 不是把 state 简化成单个 `object_i` 或 `room_i`，而是为每个 attribute 维护它覆盖的底层 free-space region：

```text
object_i -> V_object_i  # approach / reachable cells around object_i
room_i   -> V_room_i    # traversable cells inside room_i
floor_0  -> V_floor_0   # all traversable cells on floor_0
```

不要新增额外 planning resolution，例如：

```text
room-portal resolution
object-component resolution
virtual connector resolution
```

这些可以作为 edge evidence / refinement seed / metadata，但不能成为 AMRA* 的新 hierarchy level。

### 5.1 Data Sources

输入来自 `map_state`：

- `layers.occupancy`
- `layers.room`
- `layers.object_instance`
- `room_instances`
- `object_instances`
- `layer_legends`
- `metadata`

不得使用 instruction 中的 hidden annotations 来构图，例如 `hard_constraints` 的 target object、human trajectory 等。

### 5.2 Local Types

本地 dataclass 建议：

```python
@dataclass(frozen=True)
class BaseNode:
    node_id: str          # e.g. cell_120_85
    row: int
    col: int
    labels: frozenset[str]

@dataclass(frozen=True)
class AttributeRegion:
    attribute_id: str     # object_7, room_3, floor_0
    level: str            # object | room | floor
    category: str
    cells: frozenset[Cell]
    boundary_cells: frozenset[Cell]
    center: Cell | None
    parent_attribute_id: str | None
```

允许的 AMRA* semantic levels 只有：

```text
object
room
floor
```

底层 anchor 使用 `V x Q`，也就是 occupancy cells 与 automaton states 的 product space。`building_0` 可以作为 serialization root，但不作为 AMRA* planning level。

### 5.3 Occupancy / Base Graph

底层 `V` 是唯一可执行空间。

State:

```text
(cell_<row>_<col>, automaton_state)
```

Transition:

1. 移动到 4/8-connected traversable neighbor。
2. 在 neighbor cell 上 evaluate AP labels。
3. 推进 automaton / residual formula。
4. 拒绝 invalid / rejecting automaton state。
5. 插入 product successor。

Edge cost 必须匹配 evaluator：

- horizontal / vertical: `1`
- diagonal: `sqrt(2)`

faithful baseline 只优化 geometric path cost。soft constraints 不进入 edge cost，除非作为单独 extension 报告。

### 5.4 Object Attribute Regions

每个 object instance 对应一个 object attribute：

```text
a = object_<id> in A_1
V_a = traversable approach cells around object_<id>
```

`reach(object_i)` 在所有 planning levels 中必须使用同一个 `V_object_i`：

```text
reach(object_i) is true at cell s iff s in V_object_i
```

Object-level actions should follow the paper's attribute-region rule: transitions go from one object attribute region toward the boundary of another object attribute region, with cost equal to shortest feasible path distance in the base graph. Do not connect every object pair merely because they are in the same room unless the lower-level shortest path and boundary target are valid.

### 5.5 Room Attribute Regions

每个 room instance 对应一个 room attribute：

```text
a = room_<id> in A_2
V_a = traversable cells inside room_<id>
```

Room connectivity / room-level actions must originate from actual map connectivity:

1. 遍历 traversable cells。
2. 检查 neighbor 是否属于不同 room id。
3. 若存在可通行 cross-room transition，则记录 room boundary evidence。
4. room-level transition cost 使用 underlying shortest feasible path distance。

不要仅通过 bounding-box contact、centroid distance、semantic similarity 推断 room transition。

`enter(room_i)` 在所有 planning levels 中必须使用同一个 `V_room_i`：

```text
enter(room_i) is true at cell s iff s in V_room_i
```

### 5.6 Floor Attribute Region

SemPathBench 当前大多是单层环境：

```text
a = floor_0 in A_3
V_a = all traversable cells on floor_0
```

Floor attribute 仍保留以匹配原方法结构，但在单层场景中通常是 trivial。不要人为构造 fake floor subdivisions。如果后续支持多楼层，floor transitions 必须对应真实 stairs、elevators 或 inter-floor connectors。

### 5.7 Cross-Attribute Hierarchy

用于 prompt 和 validation 的 hierarchy 是：

```text
floor_0
  └── room attributes
        └── object attributes
```

用于 AMRA* 的 planning sets 是：

```text
X0 = V x Q
X1 = V_object x Q, where V_object = union_a_in_A1 V_a
X2 = V_room   x Q, where V_room   = union_a_in_A2 V_a
X3 = V_floor  x Q, where V_floor  = union_a_in_A3 V_a
```

高层 solution / guidance 必须最终 refine 成 valid occupancy path 才能返回。

---

## 6. Atomic Propositions

只允许 translator 使用 proposition inventory 中的 AP。

基础 AP：

```text
enter(room_<id>)
reach(object_<id>)
reach(region_<id>)
```

在本地 planner 里映射为：

| AP | True 条件 |
| --- | --- |
| `enter(room_i)` | 当前 occupancy cell 的 room layer value 是 `i`，或高层 room state 经 refinement 进入同一 room |
| `reach(object_i)` | 当前 occupancy cell 在 object_i approach region 中，或高层 object state 经 refinement 进入同一 approach region |
| `reach(region_i)` | 当前 occupancy cell 在 deterministic virtual region 中 |

avoidance 不需要单独 AP 类型，可以由 LTL 中的 `ALWAYS NEGATION enter(...)` 或 `ALWAYS NEGATION reach(...)` 表示。

---

## 7. Product-State Definition

每个 planning state 都必须组合：

```text
base scene-graph node
+
LTL automaton state
```

概念类型：

```python
@dataclass(frozen=True)
class ProductState:
    resolution: Literal["anchor", "object", "room", "floor"]
    base_node_id: str          # cell_<row>_<col>, an element of V
    automaton_state: str
    attribute_id: str | None   # object_i / room_i / floor_0 for auxiliary levels
```

例子：

```text
(anchor, cell_100_80, q_2)
(object, cell_104_81, q_2, object_7)  # cell is in V_object_7
(room, cell_120_90, q_2, room_3)      # cell is in V_room_3
(floor, cell_120_90, q_2, floor_0)
```

automaton state 必须参与：

- state equality；
- hashing；
- `g` values；
- predecessor records；
- CLOSED / OPEN bookkeeping；
- heuristic computation；
- path reconstruction。

任何不带 automaton state 的 room/object/floor state 都不是合法 product-planning state。任何 auxiliary state 也必须绑定到底层 `V` 中的一个 node，不能只是抽象 attribute ID。

---

## 8. Grounding

Grounding 分两层：

```text
scene-graph entity grounding
deterministic metric/geometric resolver
```

### 8.1 Scene-Graph Entity Grounding

LLM grounding prompt 输入 compact hierarchy，不暴露 occupancy cells：

```yaml
floor_0:
  rooms:
    room_3:
      category: bedroom
      objects:
        - object_25: bed
        - object_31: dresser
      connected_rooms:
        - room_1
```

LLM 输出必须是 JSON：

```json
{
  "grounded_instruction": "Enter room_1, then reach object_25.",
  "entity_bindings": [
    {
      "text_span": "the bedroom",
      "entity_id": "room_3",
      "reason": "category match"
    }
  ],
  "unsupported_phrases": []
}
```

所有 `room_<id>` 和 `object_<id>` 必须被 validator 检查是否存在。

### 8.2 Deterministic Geometric Resolver

SemPathBench 中常见但原方法不是 native 的 referring expressions，需要在 adapter 层 deterministic 处理：

| Expression | Resolver |
| --- | --- |
| nearest object/room | 从 start 到 candidate AP region 的 shortest-path distance |
| farthest object/room | shortest-path distance 最大者 |
| room containing object | scene graph containment |
| room without object | room candidate filtering |
| object in room | containment filtering |
| between A and B | 构造 virtual region，但不新增 planning level |
| midpoint of A and B | 构造 virtual AP region，但不新增 planning level |
| closer to A than B | candidate distance comparison |

resolver 不能读取 evaluator target annotation。若多候选仍无法唯一确定，默认 tie-break：

```text
shortest path distance from start
then smallest instance id
```

但必须记录：

```json
{
  "status": "tie_broken",
  "candidates": ["object_4", "object_9"],
  "rule": "shortest_path_then_instance_id"
}
```

### 8.3 Unsupported Grounding

如果 required hard target 无法 grounding：

```text
GROUNDING_FAILED
```

如果 text 描述清楚指向多个 candidate，而 tie-break 会明显改变语义：

```text
AMBIGUOUS_GROUNDING
```

不要静默选第一个。

---

## 9. Unsupported Cases Policy

原方法是 Boolean LTL planning，不自然支持所有 SemPathBench soft/path-shape preferences。

默认 faithful baseline：

- hard visit / order / avoid：进入 grounding 和 LTL。
- nearest/farthest/containment/without/between：作为 adapter grounding，进入 LTL。
- soft constraints：记录为 unsupported，不用于 planning。
- path smoothness / clearance / human-like shape：记录为 unsupported，不用于 planning。

需要在 `scripts/methods/osgllm/docs/unsupported_cases.md` 中维护表格：

```text
case
example
status
reason
possible future extension
```

初始 unsupported 列表：

- `near_preference`
- `far_preference`
- `relative_preference` as soft preference
- `move_smoothness`
- `clearance`
- `path_shape_preference`
- wall-following
- circling a room/object
- human-like route shape
- preference-only "try to" language

这些仍会由 shared evaluator 评分；低 SCS 是 faithful baseline 的真实结果。

---

## 10. LTL Translation And Validation

官方 prompt 的动作空间主要是：

```text
enter(room_x)
reach(object_y)
```

SemPathBench adapter 应保留这个动作风格。

### 10.1 Two-Stage Language Pipeline

论文中的 language pipeline 是两阶段：

1. 使用 attribute hierarchy `G_bar` 让 LLM 把自然语言中的实体引用替换成 unique IDs，得到 `mu_unique`。
2. 从 `mu_unique` 中用 regex 提取涉及的 `room_i` / `object_i` environment elements，得到 `mu_regex`，再让 LLM 生成 prefix LTL。

SemPathBench 版本应保留这个分工：

```text
instruction + compact hierarchy
        -> unique-id grounded instruction
        -> env element extraction
        -> prefix LTL
        -> syntax / co-safety validation
        -> automaton / local Formula backend
```

可以用 JSON 包装 LLM 输出以方便 validation，但语义上仍应区分 unique-ID grounding 和 LTL translation。

### 10.2 Translation Input

LLM translator 输入：

- grounded instruction；
- allowed AP inventory；
- allowed operators；
- examples；
- strict output format。

输出格式建议沿用 prefix list：

```text
['AND', 'EVENTUALLY', 'enter(room_1)', 'EVENTUALLY', 'reach(object_25)']
```

### 10.3 Parser / Automaton Backend

为了避免依赖 official repo 和 Spot，第一版推荐复用已有本地 LTL 工具：

- `scripts/methods/lang2ltl/ltl.py`
- `scripts/methods/lang2ltl/ltl_parser.py`
- 或新增 osgllm 自己的 prefix parser，并转换成同一套 `Formula`

后续如果要加 Spot backend，应作为 optional backend：

```text
--automaton-backend {local,spot}
```

默认不能要求用户安装官方 C++/Spot pipeline 才能跑 baseline。

### 10.4 Validation

必须验证：

1. prefix syntax arity 正确；
2. 只使用 `AND/OR/NEGATION/IMPLY/EQUAL/UNTIL/ALWAYS/EVENTUALLY`；
3. 每个 `enter(room_i)` / `reach(object_i)` 在 inventory 中；
4. grounding 产生的 required hard target 出现在 formula 中；
5. hard avoid 被编码为 safety/negation；
6. soft constraints 没有被错误转成 hard constraints；
7. formula 可被 planner backend 处理；
8. formula 通过 co-safe / finite-path acceptance 检查，或者明确记录为 backend-supported safety-plus-co-safe fragment。

失败时允许有限 retry；超过 retry：

```text
LTL_TRANSLATION_FAILED
```

或：

```text
LTL_VALIDATION_FAILED
```

---

## 11. AMRA*-Style Planning

### 11.1 Main Requirement

主 planner 应实现一个 integrated AMRA*-style search over four original hierarchy levels：

```text
anchor queue over X0 = V x Q
object-level auxiliary queues over X1 = V_object x Q
room-level auxiliary queues over X2 = V_room x Q
floor-level auxiliary queues over X3 = V_floor x Q
```

这些队列属于同一个 planning process，共享 product-state / g-value / predecessor / incumbent solution bookkeeping。

不要实现：

```text
first plan rooms
then independently run A*
```

不要实现：

```text
run four separate planners
then select one path
```

不要让 room/object/floor 只生成 waypoint list。

四个 hierarchy levels 必须参与同一个 multi-resolution product search。

### 11.2 Planning Domain Sets

按论文定义：

```text
Xk = Vk x Q
Vk = union_{a in Ak} Va
```

SemPathBench 映射：

```text
X0 = V x Q
X1 = V_object x Q
X2 = V_room x Q
X3 = V_floor x Q
```

Goal region:

```text
R = V x F
```

其中 `F` 是 accepting automaton states。

### 11.3 Anchor Search

它必须在以下情况仍能找到 valid solution：

- object-level guidance 很差；
- room-level guidance 很差；
- LLM heuristic 错误；
- high-level abstraction 无法表示狭窄通道。

Anchor space is `X0 = V x Q`, not merely a separate "occupancy-only baseline". Its actions `U0` should be derived from:

```text
base scene-graph edges E
automaton transitions
union of higher-level actions Uk
```

This follows the paper's definition that the anchor is built on `V x Q` while still incorporating the hierarchical action structure. A debug occupancy-only anchor may exist as an ablation, but the faithful `amra` planner should distinguish it from the full anchor.

Anchor search 使用 deterministic LTL-aware heuristic，不依赖 LLM。

保守初版 heuristic：

```text
distance from current cell
to nearest proposition region
that can advance current automaton state
```

anchor search 不可被 auxiliary queues 禁用或替代。

### 11.4 Auxiliary Resolution Searches

Object、room、floor spaces 提供 auxiliary searches。

每个 auxiliary state 仍然是 product state：

```text
(base_node in V_object_i, q_j, object_i)
(base_node in V_room_i, q_j, room_i)
(base_node in V_floor_0, q_j, floor_0)
```

它们可以：

- 更快探索高层 semantic progress；
- 改变 expansion priority；
- 改善 first-solution runtime；
- 提供 LLM-guided preferred high-level expansion。

它们不能：

- 删除 occupancy-level valid actions；
- 忽略 automaton state；
- 独立 declare success；
- 直接返回 room/object/floor sequence 作为 trajectory。

### 11.5 Auxiliary Action Definition

For level `k`, an action from `xi = (si, qi)` to `xj = (sj, qj)` follows the paper's conditions:

1. `si in Va` for some source attribute `a in Ak`；
2. `sj in boundary(Vb)` for a different target attribute `b in Ak`；
3. the source and target attribute interiors do not overlap；
4. automaton progress is respected: `qj = T(qi, label(si))` or the local backend's equivalent transition convention；
5. `sj` lies on a shortest feasible path from `si` toward the target attribute boundary；
6. cost is the shortest feasible base-graph path distance `d(si, sj)`。

This is more faithful than treating `object_i` or `room_i` as a single abstract node.

### 11.6 AMRA* Bookkeeping

Planner 至少维护：

```python
g_value: dict[ProductState, float]
predecessor: dict[ProductState, ProductState | EdgeRef]
open_queues: dict[level_or_heuristic_id, PriorityQueue]
closed: dict[level_or_heuristic_id, set[ProductState]]
incumbent_solution: ProductState | None
incumbent_cost: float | None
```

`ProductState.automaton_state` 必须参与所有 key。

Queue configuration:

```text
anchor: X0 = V x Q + consistent LTL heuristic
aux_occ_llm: optional LLM heuristic at occupancy/base resolution
aux_object: object + scene-graph/LTL heuristic
aux_room: room + scene-graph/LTL heuristic
aux_floor: floor + trivial/floor heuristic
aux_llm: optional LLM heuristic assigned to occupancy/object/room/floor auxiliary queues
```

如果实现只有 occupancy product A* 加 room/object heuristic，应记录：

```text
planner_type = hierarchy_informed_product_astar
```

不能 claim：

```text
planner_type = amra_hierarchical_product
```

### 11.7 LTL Heuristic

保留原方法的 LTL-aware heuristic 思想。

heuristic 估计从 current automaton state 到 accepting state 的 progress，应考虑：

- 当前 enabled automaton transitions；
- 能推进 task 的 propositions；
- remaining required room/object goals；
- scene-graph distances to proposition regions；
- ordered visits；
- intermediate goals；
- avoidance；
- disjunction；
- temporal dependencies。

不要只用：

```text
distance to final target
```

更 faithful 的实现应接近论文定义：

```text
c_l(l1, l2) = min distance between nodes with labels l1 and l2
g(l, q) = min_{l'} c_l(l, l') + g(l', T(q, l'))
h_LTL(s, q) = min_t c(s, t) + g(label(t), T(q, label(t)))
h_LTL(s, q) = 0 for q in F
```

其中 `g` 可通过 automaton graph 上的 Dijkstra 预计算。若第一版近似这个 heuristic，必须记录 approximation；如果 claim optimality，anchor heuristic 必须满足 consistency requirement。

### 11.8 Scene-Graph Hierarchical Heuristics

Object-level heuristic 使用：

- target object identity；
- object containment；
- object-to-object graph distance；
- current automaton progress。

Room-level heuristic 使用：

- target room identity；
- rooms containing remaining target objects；
- room connectivity；
- current automaton progress。

Floor-level heuristic 使用：

- remaining target floors；
- inter-floor connectivity；
- current automaton progress。

当前 single-floor SemPathBench 中 floor heuristic 可以是 constant / trivial。

这些 heuristics 只指导 search，不删除 valid occupancy actions。

### 11.9 LLM Heuristic

LLM heuristic 角色：

```text
scene graph + task state
-> semantic guidance for search
```

论文里的 LLM heuristic `h_LLM: V x Q -> R` 不是直接输出 dense path，也不是只输出 preferred node list。它根据 current attributes、automaton state、attribute hierarchy 和 remaining mission 生成 high-level function calls；这些 function calls 的 user-defined costs 加总成 online heuristic value。

LLM 可接收：

- compact floor-room-object hierarchy；
- current attributes at base node `s`；
- grounded mission；
- current automaton state `q`；
- remaining task inferred from shortest automaton path to accepting state；
- allowed functions。

输出示例：

```xml
<command>move(room_1, room_2)</command>
<command>move(room_2, room_3)</command>
<command>reach(room_3, object_11)</command>
```

输出必须 validated against scene graph。每个 function call 返回 user-defined cost，例如两个 attribute centers 或 regions 之间的 shortest-path lower estimate。总和作为 `h_LLM(s, q)`。

因为 LLM query 延迟较大，计划中应缓存/离线生成：

```text
(map_id, grounded_mission, automaton_state, current_attribute_signature)
-> function-call sequence
```

LLM heuristic 可以改变：

- expansion priority；
- search order；
- first-solution runtime。

它不能：

- 直接生成 final dense trajectory；
- 移除 non-preferred valid states；
- 改变 graph connectivity；
- override LTL transitions；
- declare success；
- replace anchor search。

### 11.10 High-Level Edge Validation

object / room / floor level 的任何 transition 都必须对应 underlying scene graph 中的 valid connection。

接受 high-level path 前必须：

1. refine to lower hierarchy levels；
2. eventually recover occupancy-level path；
3. replay occupancy path through automaton；
4. verify final automaton state is accepting。

不能因为两个 semantic nodes 在 YAML/hierarchy 中相邻就接受 high-level transition。

### 11.11 Path Reconstruction

planner output 最终必须转成：

```text
[(row_0, col_0), ..., (row_T, col_T)]
```

Path reconstruction 步骤：

1. trace selected product-state predecessors；
2. refine room/floor/object transitions to lower levels；
3. recover occupancy-level connections；
4. concatenate occupancy subpaths；
5. remove duplicate consecutive cells；
6. validate all grid transitions；
7. replay propositions and automaton transitions；
8. verify accepting state。

不要把 room sequence 或 object sequence 当作 SemPathBench trajectory。

### 11.12 Anytime Behavior

AMRA* 应先找 feasible solution，再在剩余时间内改进。

记录：

```json
{
  "first_solution_time": null,
  "first_solution_cost": null,
  "final_solution_time": null,
  "final_solution_cost": null,
  "solution_cost_history": [],
  "inflation_history": []
}
```

期望：

```text
final_solution_cost <= first_solution_cost
```

如果 timeout 前没有 feasible solution：

```text
PLANNER_TIMEOUT
```

如果已有 feasible path 但 optimal refinement 未完成，返回 best feasible occupancy path，并记录 `optimality_reached = false`。

---

## 12. Debug / Ablation Planners

允许实现两个非主线 planner，但必须清楚命名。

### 12.1 `anchor-only`

只运行 `X0=VxQ` anchor search，不启用 object/room/floor auxiliary queues。

用途：

- correctness reference；
- 单元测试；
- ablation。

metadata：

```json
"planner_type": "occupancy_product_anchor_only"
```

### 12.2 `debug-fast`

从 formula 中提取 sequential eventual AP，用 A* 串接。

用途：

- smoke test；
- runner / output / evaluation 链路调试。

metadata：

```json
"planner_type": "debug_fast_sequential_astar",
"method_level_deviations": [
  "debug-fast planner is not the full Optimal Scene Graph Planning method"
]
```

默认主实验不能使用 `debug-fast` 冒充完整方法。

---

## 13. Failure Statuses

推荐统一状态：

```text
SUCCESS
GROUNDING_FAILED
AMBIGUOUS_GROUNDING
UNSUPPORTED_INSTRUCTION
LTL_TRANSLATION_FAILED
LTL_VALIDATION_FAILED
NO_FEASIBLE_PATH
PLANNER_TIMEOUT
INTERNAL_ERROR
```

处理原则：

- required hard component 失败：episode 失败，返回 start-only trajectory，并记录 status。
- soft component unsupported：episode 可以成功，记录 unsupported list。
- planner 找不到 accepting state：`NO_FEASIBLE_PATH`，不要返回 unconstrained shortest path。
- timeout 且已有 feasible path：返回 best feasible occupancy path，并记录未达到 optimality。
- timeout 且无 feasible path：start-only trajectory + `PLANNER_TIMEOUT`。

---

## 14. Tests

建议新增测试：

```text
scripts/methods/osgllm/test_scene_graph.py
scripts/methods/osgllm/test_grounding.py
scripts/methods/osgllm/test_ltl_translation.py
scripts/methods/osgllm/test_product_state.py
scripts/methods/osgllm/test_amra_planner.py
```

最低测试覆盖：

- map_state -> base-node count and object/room/floor attribute-region counts；
- semantic attribute levels 只有 `object/room/floor`，anchor space 是 `X0=VxQ`；
- room adjacency 来自 traversable neighbor；
- object approach cells 是 traversable；
- `enter(room_i)` label；
- `reach(object_i)` label；
- product state hashing 包含 automaton state；
- nearest / farthest resolver；
- prefix LTL parser arity validation；
- formula hallucinated AP rejection；
- `X0=VxQ` anchor 能完成 simple `EVENTUALLY enter(room_i)`；
- `X0=VxQ` anchor 能处理 ordered `A then B`；
- `X0=VxQ` anchor 避开 hard avoid region；
- AMRA* queues 都处理 product states；
- auxiliary search 不能绕过 anchor validation；
- reconstructed path replay 后 accepting；
- `run.py --limit 1 --overwrite` 能写 prediction、PNG、summary。

---

## 15. Development Order

### Phase 1: Runner Skeleton

- 新建 / 完善 `scripts/methods/osgllm/`。
- 按 tutorial runner 风格实现 `run.py`。
- 用 start-only trajectory 验证 output / summary / PNG / evaluator 链路。

### Phase 2: Scene Graph

- 从 `map_state` 构建 base graph `V,E` 和 object/room/floor attribute regions。
- 严格限制 semantic attribute levels 为 `object/room/floor`，不新增 planning resolution。
- 构造 room adjacency 和 object approach regions。
- 写 scene graph summary 到 output metadata。

### Phase 3: Product State And Anchor Planner

- 实现 `ProductState(resolution, base_node_id, automaton_state, attribute_id)`。
- 实现 AP labeler。
- 实现 `X0=VxQ` anchor search。
- 用手写 formula 测试 simple room/object visit、order、avoid。

### Phase 4: Multi-Resolution Domain

- 构建 object / room / floor product transition model。
- 所有高层 edges 必须有 lower-level connectivity evidence。
- 建立 cross-resolution refinement / reconstruction API。

### Phase 5: AMRA*-Style Planner

- 实现 anchor queue + auxiliary queues。
- 共享 g-value / predecessor / incumbent solution bookkeeping。
- 实现 LTL heuristic 和 scene-graph hierarchical heuristic。
- 实现 anytime improvement logging。

### Phase 6: Grounding

- compact hierarchy prompt。
- JSON entity grounding。
- deterministic resolver。
- unsupported logging。

### Phase 7: LTL Translation

- grounded instruction -> prefix LTL。
- parser / validator / retry / cache。
- 接入 local Formula progression / automaton backend。

### Phase 8: LLM Heuristic

- 按论文方式生成 `move(...)` / `reach(...)` function-call guidance。
- validate function calls against attribute hierarchy and connectivity。
- 将 function-call sequence 转成 cached heuristic cost。
- 只作为 auxiliary / non-anchor heuristic，不替代 consistent LTL anchor heuristic。

### Phase 9: Evaluation

- `--limit 1` smoke test。
- `--set valunseen --limit 25`。
- 分析 failure status、unsupported cases、metrics、planner metadata。

---

## 16. Paper/README Wording

推荐描述：

> We adapt Optimal Scene Graph Planning to SemPathBench by constructing a base traversable graph and object, room, and floor attribute regions from the benchmark's 2D semantic map. The method grounds natural-language missions to unique scene-graph attributes, translates the grounded mission into LTL over `enter(room)` and `reach(object)` propositions, and performs an AMRA*-style multi-resolution product search over `X0=VxQ` and attribute-region spaces `Xk=VkxQ`. Since the official implementation is not used as a runtime dependency, we implement a local SemPathBench-compatible planner while documenting deviations from the original C++ AMRA* implementation.

对 soft constraints：

> Soft path preferences such as clearance, smoothness, and human-like path shape are not native to the original Boolean LTL formulation. The faithful baseline logs them as unsupported during planning and lets the shared evaluator score the resulting trajectories.

如果实际实现只完成 anchor-only / hierarchy-informed A*，必须写：

> This implementation uses a planner substitution and should not be interpreted as the full AMRA* planner from the original method.

---

## 17. Non-Goals

不要做：

- 不 import official repo 里的 Python/C++ code。
- 不要求用户 build 官方 C++ planner 才能跑 SemPathBench baseline。
- 不新增非原方法 planning resolutions，例如 room portal / object component / virtual connector level。
- 不把完整方法降级成 `occupancy product A* + scene graph heuristic` 却仍 claim AMRA*。
- 不先 plan rooms 再独立跑 A*，然后称为 integrated multi-resolution search。
- 不让 LLM 直接输出 dense trajectory。
- 不把 `hard_constraints` / `soft_constraints` 当作 inference-time oracle 直接喂给 planner。
- 不用 human trajectory。
- 不把 soft preferences 默认转成 hard LTL。
- 不把 sequential A* baseline 叫作完整 Optimal Scene Graph Planning。
- 不在失败时静默返回 tutorial-style hard-constraint path。

---

## 18. Acceptance Criteria

第一版可信适配至少需要：

- `scripts/methods/osgllm/run.py` 和 tutorial runner 风格一致；
- 输出路径为 `resources/methods/baselines/OSGLLM`；
- prediction JSON / PNG / summary 正常生成；
- SemPathBench map 能转换成本地 scene graph；
- semantic attribute levels 只有 `object/room/floor`，anchor space 为 `X0=VxQ`；
- room/object AP inventory 正确；
- grounded instruction 中所有 entity ID 都经过 validation；
- LTL translation 有 syntax/proposition validation；
- planning state 至少使用 `(resolution, base_node_id, automaton_state, attribute_id)`；
- automaton state 参与 hashing、g-value、predecessor、OPEN/CLOSED；
- AMRA* 主线包含 `X0=VxQ` anchor queue 和 object/room/floor auxiliary queues；
- auxiliary high-level edge 都能 refine / validate 到 occupancy path；
- final trajectory 是 occupancy cells，且 replay 后 accepting；
- unsupported soft/path-shape cases 有 explicit log；
- method-level deviations 写入每条 output；
- 有单元测试覆盖 scene graph、grounding、LTL、product state、AMRA* planner 的核心行为。
