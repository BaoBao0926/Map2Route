# LIMP Implementation Plan for SemPathBench

## 0. Goal

在 `scripts/methods/limp/` 下实现一个面向 SemPathBench 的 LIMP baseline。

这个 baseline 的目标不是追求最高 benchmark 分数，而是尽量忠实地把 LIMP 的核心方法迁移到 SemPathBench：

```text
language instruction
-> two-stage LTL translation
-> LIMP-style predicates and CRDs
-> spatial-relation grounding
-> LTL / DFA task progression
-> task-stage-dependent semantic maps
-> progressive grid planning
-> SemPathBench trajectory
```

推荐方法名：

```text
LIMP
```

对外描述：

```text
LIMP (oracle semantic-instance 2D adaptation)
```

核心研究问题：

> 当原始 LIMP 的 RGB-D / VLM perception layer 被 SemPathBench 的 oracle semantic-instance map 替换后，LIMP 的 language-to-LTL、CRD grounding、task progression 和 progressive planning 能在 SemPathBench 上解决多少任务？

---

## 1. Hard Requirements

### 1.1 Reuse Official LIMP Method Components Where Possible

`scripts/methods/limp/robotlimp/` 是 official implementation 的本地参考来源。SemPathBench 版本不应该依赖原始 robot runtime、Open3D perception stack、manipulation environment、notebook execution state 或 OSG / VLM pipeline，但应该优先复用 official LIMP 中能够独立出来的 method-level components。

复用优先级：

```text
Option A: Vendored exact reuse
copy isolated official method-level code into scripts/methods/limp/vendor/
-> preserve original behavior, license, attribution, and modification notes

Option B: Vendored adapted reuse
copy official code into scripts/methods/limp/vendor/
-> remove robot/perception imports only where necessary
-> document every modification

Option C: Local reimplementation
only when official code cannot be isolated or reused
```

优先复用或 vendor 的内容：

- original two-stage prompts
- predicate naming and formatting
- predicate encoding convention
- CRD syntax and parsing behavior
- LTL formula conventions
- LTL compilation / progression path
- task-state update logic
- transition-selection logic
- progressive planning control flow

应该替换的内容：

- RGB-D perception
- open-vocabulary visual detection / segmentation
- Open3D point-cloud processing
- 3D semantic-map construction
- manipulation skills
- robot execution stack
- continuous FMT* motion planning

如果某个 method-level component 必须本地重写，必须在代码和 metadata / README 中记录：

- 对应的 original LIMP component
- 为什么不能复用
- semantic equivalence assumption
- known deviations

所有 SemPathBench runtime code 和 vendored code 都放在 `scripts/methods/limp/` 下；不从 `robotlimp/` 做 runtime import，不把输出写进 `robotlimp/`，也不要求用户进入 `robotlimp` 的原始 notebook / robot environment。

初始 reuse audit targets：

| Official source | Reuse target | Expected policy |
| --- | --- | --- |
| `robotlimp/limp/language/prompts/*.txt` | two-stage prompt wording, predicate/comparator format | vendor or copy with attribution |
| `robotlimp/limp/language/llm4tl.py` | predicate encoding, Stage-1/Stage-2 translation structure, `parse_spatial_lifted_ltl` behavior | vendor reusable functions; replace API wrapper / model config if needed |
| `robotlimp/limp/utils/gen_utils.py` | CRD extraction, task-structure English trace, Spot conversion, `ltl2dfa` wrapper | vendor isolated pieces only; the file imports planner modules at top level |
| `robotlimp/limp/language/temporal_logic/dfa.py` | DFA wrapper and transition evaluation | vendor if dependencies remain lightweight |
| `robotlimp/limp/language/temporal_logic/ltl_progression.py` | co-safe LTL progression semantics | vendor if exact semantics are needed and licensing is preserved |
| `robotlimp/limp/planner/multi_level_planner.py` | progressive planning control flow and TPSM concept | do not import directly; it imports `osg`, Open3D, and FMT modules |
| `robotlimp/limp/planner/fmt.py` | continuous FMT* planner | replace with grid A* |

如果某个 official file 的顶层 import 会把 `osg`、Open3D point clouds、notebook state 或 robot runtime 拉进来，就不要直接 import 该文件；只 vendor 可隔离函数，或本地实现 SemPathBench adapter，并记录原因。

### 1.2 Follow Tutorial Runner Structure

`run.py` 应尽量参考 `scripts/methods/tutorial/run.py` 的结构，不做过度复杂化。

保留类似主流程：

```text
parse_args()
-> build_all(...)
   -> iter_instruction_files(...)
   -> load_instruction(...)
   -> map_id_from_instruction_path(...)
   -> load_map_state(...)
   -> build_limp_trajectory(...)
   -> evaluate_prediction(...)
   -> build_prediction_record(...)
   -> save_trajectory_image_for_record(...)
   -> write_prediction_record(...)
   -> write_summary(...)
```

也就是说，runner 外壳尽量和 tutorial 一致；LIMP 的复杂性放在 pipeline / translator / grounding / planning modules 里。

### 1.3 Add README After Plan

实现时需要生成：

```text
scripts/methods/limp/README.md
```

README 风格参考 `scripts/methods/lang2ltl/README.md`，内容包括：

- 方法简介
- 怎么跑起来
- API key / model 配置
- 最小运行命令
- 全量运行命令
- CLI 参数表
- 主要文件说明
- 和原始 LIMP 相比改了什么
- 不使用哪些 benchmark annotation
- 输出路径和 metadata 说明

README 中的默认输出路径必须写成：

```text
resources/methods/baselines/LIMP
```

本文件只先规划 README，不在当前阶段写 README。

### 1.4 Dependency Policy

默认依赖应尽量沿用 SemPathBench 当前 method 环境。只有 official LIMP formal core 需要时才新增轻量依赖。

特别是 LTL 部分：

- official LIMP 使用 Spot 做 formula parsing / simplification，并用 co-safe LTL progression 构造 DFA。
- 当前默认实现使用 local `ltlf` residual-formula progression，复用 SemPathBench Lang2LTL 的 Formula/progress semantics。
- 如果环境中可安装 `spot` 或能 vendor official wrapper，future official-aligned backend 应优先使用 Spot-compatible path。
- 如果选择 `--automaton-backend spot` 但 `spot` 不可用，run 必须失败为 `AUTOMATON_BACKEND_UNAVAILABLE`，不能静默改用 sequential fallback。
- 本地 residual fallback 只能作为显式 debug / smoke-test backend，并必须在 metadata 中标记，不得作为 main result。

README 需要单独写清楚 `spot` 是 faithful LTL path 的推荐依赖，而不是 perception / robot runtime 依赖。

---

## 2. What To Preserve From LIMP

LIMP 必须保留以下方法骨架：

### 2.1 Two-Stage Translation

第一阶段：

```text
instruction -> conventional LTL
```

第二阶段：

```text
instruction + conventional LTL
-> skill-aware LTL
-> predicates such as near[referent]
-> CRDs such as chair::isbetween(sofa,table)
```

第一版不应该替换为：

```text
instruction -> JSON constraints
```

也不应该让 LLM 直接输出 waypoint 或 trajectory。

### 2.2 LIMP Predicate Style

SemPathBench 是 navigation benchmark，因此第一版只启用：

```text
near[referent]
```

可在内部支持 alias：

```text
visit[referent] -> near[referent]
navigate_to[referent] -> near[referent]
```

但最终 LTL predicate 仍建议统一落到 `near[...]`。

原始 LIMP 的 manipulation predicates：

```text
pick[...]
release[...]
```

第一版应标记为 unsupported，而不是静默改写成 navigation。

Prompt adaptation rule:

- Official prompt wording and CRD comparator definitions should be preserved as much as possible.
- The robot skill set should be explicitly reduced from `(near, pick, release)` to navigation-only `(near)` for the main SemPathBench run.
- This skill-library reduction must be documented as an embodiment adaptation.
- If a model still emits `pick[...]` or `release[...]`, the episode should return `UNSUPPORTED_PREDICATE`; it should not be silently converted to `near[...]`.

### 2.3 CRD Grounding

保留 Composable Referent Descriptor 表达：

```text
chair::isbetween(sofa,table)
cabinet::isleftof(fridge)
basket_ball::isinside(bedroom)
```

SemPathBench map 可以提供 candidate instances，但不能直接提供最终 grounding answer。

grounding 必须经历：

```text
referent phrase
-> candidate instances by category / room category
-> spatial comparator filtering / ranking
-> selected grounded instance or ambiguous / failed status
```

### 2.4 LTL / DFA Progression

任务状态必须来自 LTL progression 或 DFA / automaton state。

即使实现上使用 progressive segment planning，也应该概念上维护：

```text
(grid position, task progress state)
```

不能简单把 instruction 拆成 ordered target list 后直接串 A*，然后称为 LIMP。

### 2.5 Task Progression Semantic Maps

每个 task stage 都应该构造 TPSM-like map：

```text
TPSM = {
  geometric_obstacles,
  active_forbidden_regions,
  enabled_goal_regions,
  proposition_satisfaction_regions
}
```

当 LTL / DFA state 改变时，TPSM 必须随之更新。

---

## 3. Allowed Adaptations

### 3.1 Replace VLM Perception With Oracle Candidate Extraction

原始 LIMP：

```text
RGB-D observations
-> VLM detection / segmentation
-> 3D object candidates
```

SemPathBench adaptation：

```text
map_state.layers.object_instance
map_state.object_instances
map_state.layers.room
map_state.room_instances
-> 2D object / room candidates
```

这应该叫：

```text
oracle semantic-instance perception
```

而不是：

```text
oracle grounding
```

因为 final referent selection 仍由 LIMP-style CRD grounding 完成。

### 3.2 Use 2D Geometry

所有 spatial comparators 改成 2D grid geometry：

```text
3D centroid -> 2D centroid
3D mask -> 2D instance cells
3D point cloud obstacles -> SemPathBench traversability grid
3D near sphere -> 2D goal ring / disk
```

LIMP 论文中 spatial comparators 是相对于 origin coordinate frame 解析的。SemPathBench 版本必须明确 2D frame convention，并把它写进 metadata：

```text
grid point = [row, col]
row increases downward in image/grid coordinates
col increases rightward
above means smaller row, below means larger row
left means smaller col, right means larger col
origin frame is the SemPathBench map frame unless a comparator explicitly uses start-relative wording
```

如果实现采用 start pose 作为 relative frame，或把 row/col 转成 x/y world frame，必须在 `limp.spatial_frame` 中记录，并保证 comparator tests 覆盖该约定。

第一版 comparator：

```text
isbetween
isabove
isbelow
isleftof
isrightof
isnextto
isinfrontof
isbehind
isinside
contains
```

其中 `isinside` / `contains` 是 SemPathBench room map 需要的最小扩展。

如果 object / room 没有 orientation，则：

```text
isinfrontof
isbehind
```

应返回 unresolved，或者只在有明确 orientation metadata 时启用。

### 3.3 Replace Continuous Planner With Grid Planner

原始 LIMP 用 FMT*。

SemPathBench 第一版使用 deterministic grid planner：

```text
8-connected A*
```

规划器输入：

```text
current grid cell
TPSM free / blocked cells
current goal region
```

规划器不能使用 human expert trajectory，也不能调用 evaluator 来优化路径。

---

## 4. Unsupported Or Diagnostic-Only Capabilities

第一版 LIMP-Core 不主动支持：

- soft near / far preference optimization
- relative preference optimization over an extended path
- move smoothness optimization
- clearance optimization as a soft score
- circle / loop / walk-around path shape
- follow-wall behavior
- exact velocity constraints
- manipulation
- dynamic obstacles

处理策略：

- unsupported hard predicate：episode 失败并记录原因。
- unsupported soft clause：主实验可允许忽略，但 metadata 必须标记 `soft_constraint_unsupported`，让 shared evaluator 自然给出较低 SCS。
- 不允许静默删除 hard constraint。

---

## 5. Proposed Directory Structure

```text
scripts/methods/limp/
├── README.md
├── plan.md
├── pre.md
├── run.py
├── config.py
├── __init__.py
├── __main__.py
├── vendor/
│   ├── README.md
│   └── ...
│
├── language/
│   ├── __init__.py
│   ├── client.py
│   ├── prompts.py
│   ├── translator.py
│   ├── predicate_parser.py
│   └── cache.py
│
├── logic/
│   ├── __init__.py
│   ├── spot_adapter.py
│   ├── official_progression.py
│   ├── residual_fallback.py
│   └── automaton.py
│
├── grounding/
│   ├── __init__.py
│   ├── scene_adapter.py
│   ├── candidates.py
│   ├── crd_parser.py
│   ├── comparators_2d.py
│   └── referent_map.py
│
├── planning/
│   ├── __init__.py
│   ├── goal_regions.py
│   ├── forbidden_regions.py
│   ├── tpsm.py
│   ├── grid_planner.py
│   └── progressive_planner.py
│
├── utils/
│   ├── __init__.py
│   ├── geometry.py
│   ├── output.py
│   └── logging.py
│
└── tests/
```

所有 runtime implementation 都在这些文件下完成。若复用 official LIMP method-level code，应优先放入 `vendor/` 并保留 source attribution 和 modification notes。

`robotlimp/` 保留在目录中作为 upstream reference；runtime 不依赖它的 robot / perception environment。

---

## 6. Runner Design

### 6.1 `run.py`

`run.py` 参考 tutorial，建议保留这些函数：

```python
METHOD_NAME = "LIMP-Nav-Prototype"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "resources" / "LIMP"

def scene_id_from_instruction(...): ...

def build_limp_trajectory(...): ...

def build_prediction_record(...): ...

def build_all(...): ...

def parse_args(...): ...

def main(...): ...
```

`build_all` 主体和 tutorial 保持接近：

```text
instruction_files = iter_instruction_files(...)
for instruction_file in instruction_files:
    instruction = load_instruction(instruction_file)
    map_id = map_id_from_instruction_path(instruction_file)
    scene_id = scene_id_from_instruction(instruction, map_id)
    instruction_id = instruction_id_from_payload(...)
    output_path = output_path_for_prediction(...)

    if exists and not overwrite:
        load_existing_result(...)
    else:
        map_state = load_map_state(map_id)
        trajectory, limp_metadata = build_limp_trajectory(...)
        metrics = evaluate_prediction(...)
        record = build_prediction_record(...)
        save_trajectory_image_for_record(...)
        write_prediction_record(...)

    write_summary(...)
```

### 6.2 CLI Arguments

第一版 CLI 建议：

| 参数 | 作用 |
| --- | --- |
| `--input-root PATH` | instruction 输入目录，默认 `resources/instructions` |
| `--output-root PATH` | prediction 和 `summary.json` 输出目录，默认 `resources/methods/baselines/LIMP` |
| `--set {valunseen,train,all}` | instruction set |
| `--model MODEL` | LLM model，默认读 `scripts/methods/api_key.py` 的 `MODEL` |
| `--llm-cache-root PATH` | LIMP translation cache |
| `--overwrite` | 覆盖已有 prediction |
| `--overwrite-llm-cache` | 强制重新调用 LLM |
| `--limit N` | 调试时只跑前 N 条 |
| `--verbose` | 打印 translation / grounding / planning trace |
| `--near-radius N` | `near[...]` 转 goal region 的 grid 半径 |
| `--avoid-radius N` | negative `near[...]` 转 forbidden region 的 grid 半径 |
| `--max-progress-steps N` | progressive planner 最大 stage 数 |
| `--translation-mode {llm,heuristic}` | 主模式为 `llm`；`heuristic` 只做 smoke test，不允许 silent fallback |
| `--grounding-mode {core,extended,oracle}` | 默认 `extended`；`core` 保守返回 unresolved ambiguity；oracle 只做 diagnostic |
| `--automaton-backend {ltlf,spot,residual-debug}` | 默认 `ltlf`；`spot` 预留 official-aligned backend；`residual-debug` 只做 smoke test |
| `--allow-residual-fallback` | Debug only；没有该参数时 residual backend 不允许运行 |
| `--max-goal-candidates N` | Debug only；默认 `0` 表示不截断 goal region |
| `--strict-soft` | unsupported soft clause 是否导致 episode failed |
| `--evaluate` | 兼容旧参数，metrics 总是计算 |

---

## 7. Pipeline Design

`build_limp_trajectory(...)` 调用一个独立 pipeline：

```text
run_episode(instruction, map_state, config)
-> LimpEpisodeResult
```

### 7.1 Stage A: Scene Adapter

从 `map_state` 构造 candidate registry：

```json
{
  "object_23": {
    "kind": "object",
    "instance_id": 23,
    "category": "cart",
    "cells": [[...]],
    "centroid": [row, col],
    "bbox": [r0, c0, r1, c1],
    "room_id": 5,
    "room_category": "living_room"
  },
  "room_5": {
    "kind": "room",
    "instance_id": 5,
    "category": "living_room",
    "cells": [[...]],
    "centroid": [row, col]
  }
}
```

只使用 inference-time 可见信息：

- instruction text
- start_pose
- traversability / occupancy
- room layer / room metadata
- object instance layer / object metadata

不使用：

- hard_constraints
- soft_constraints
- instruction `objects`
- human_expert_trajectory
- evaluator result

### 7.2 Stage B: Two-Stage Translation

输入：

```text
instruction text
available robot predicate set
available comparator set
```

默认 translation prompt 应尽量保持 official LIMP：instruction + in-context LTL examples + predicate/comparator definitions。不要默认把完整 SemPathBench object / room inventory 塞进 prompt，因为这会把方法从 open-vocabulary instruction interpretation 推向 map-conditioned target selection。

允许的例外：

```text
--translation-context none       # default, faithful
--translation-context categories # optional diagnostic: only category vocabulary, no instance IDs
```

即使使用 category context，也不能暴露 hidden annotation、target object IDs、hard constraints、soft constraints 或 human trajectory。

输出：

```json
{
  "stage1_ltl": "...",
  "stage2_ltl": "F ( near[cart] & F near[basket_ball::isinside(bedroom)] )",
  "encoded_ltl": "...",
  "encoding_map": {
    "A": "near[cart]",
    "B": "near[basket_ball::isinside(bedroom)]"
  },
  "unsupported_predicates": []
}
```

Gemini / LLM 调用方式应和其他 methods 一致，优先读：

```python
scripts/methods/api_key.py
```

并支持 cache，避免重复调用。

### 7.3 Stage C: Predicate And CRD Parsing

解析：

```text
near[chair::isbetween(sofa,table)]
G ! near[kitchen]
```

得到：

```json
{
  "predicate": "near",
  "referent": {
    "base": "chair",
    "comparators": [
      {
        "name": "isbetween",
        "args": ["sofa", "table"]
      }
    ]
  }
}
```

CRD parser 必须支持 nested comparator：

```text
chair::isbetween(sofa,table::isleftof(fridge))
```

### 7.4 Stage D: Spatial Grounding

对每个 CRD：

```text
base referent -> category / synonym candidate lookup
comparator args -> recursively grounded candidate sets
comparator evaluation -> filter or rank
```

第一版策略：

- category exact / normalized match
- simple synonyms from local table if needed
- room category match for room referents
- if one candidate remains：grounding success
- if no candidate remains：`GROUNDING_FAILED`
- if multiple candidates remain and no explicit ranking comparator exists：`AMBIGUOUS_GROUNDING`
- if multiple candidates remain and an explicit ranking comparator exists：apply that comparator; success only if it yields a unique candidate

Extended variant 可添加：

- nearest
- farthest
- contains / not_contains
- start-relative comparators

但 main result 应先报告 Core。Core 配置不能把普通 relation 自动变成 ranking heuristic。例如 `chair::isnextto(table)` 不能自动选择最近的一把 chair；只有 instruction 明确包含 nearest / farthest / leftmost / rightmost 等 selector 时，才能使用对应 ranking。

### 7.5 Stage E: LTL Automaton / Progression

默认不应从零实现自定义 lightweight LTL engine。优先顺序：

```text
Option A:
reuse official LIMP LTL compilation / task-progression path

Option B:
use the same underlying LTL library and preserve LIMP's formula encoding

Option C:
local residual-formula progression only as explicit debug / smoke-test fallback
```

如果使用 Option C，不能把结果报告为 main LIMP run；必须把它记录为 method-level reimplementation，并和 standard LTL implementation 或 official LIMP traces 做交叉验证。至少验证：

```text
F A
F (A & F B)
F (A & F (B & F C))
F A & G !C
(!C) U A
F A & G (A -> G !C)
F (A & X F B)
F A & F B
F (A | B)
G !C & F A
```

每条 episode 的 metadata 必须记录：

```text
original Stage-2 LTL formula
encoded formula
automaton state ID or residual formula at every stage
true propositions after every planned segment
selected transition
accepting-state decision
```

重要的是 planning state 来自 LTL progression，而不是手写 ordered list。

此外，planner 必须在起点先 evaluate true propositions 并 progression 一次：

```text
start_pose
-> evaluate propositions true at initial cell
-> progress LTL / DFA before first planned segment
```

这样可以处理 “Start from the cart beside you” 或起点已经满足某个 `near[...]` proposition 的情况，避免无意义地重新规划到当前已满足的 referent。

### 7.6 Stage F: Goal And Forbidden Region Construction

`near[object]`：

```text
goal_region = traversable cells within near_radius from object mask
```

`near[room]`：

```text
goal_region = traversable cells inside room cells
```

`! near[object]` or `G ! near[object]`：

```text
forbidden_region = traversable cells within avoid_radius from object mask
```

`! near[room]`：

```text
forbidden_region = traversable cells inside room
```

Object occupied cells themselves通常不可通行，不能直接作为 goal。

### 7.7 Stage G: TPSM

每个 task stage 生成：

```json
{
  "stage_index": 1,
  "current_formula": "...",
  "goal_regions": [...],
  "forbidden_regions": [...],
  "blocked_cells": "occupancy + active forbidden",
  "enabled_transition": "A & !B"
}
```

TPSM 只表达当前 stage 的 goal / forbidden，不做 soft cost optimization。

### 7.8 Stage H: Progressive Grid Planning

循环：

```text
current cell
current residual formula / DFA state
-> evaluate true propositions at current cell if this is the initial state
-> build current TPSM
-> choose enabled transition goal region
-> A* to goal region while avoiding blocked cells
-> append segment
-> evaluate true propositions at reached cell
-> progress formula / DFA state
-> repeat until accepting or failure
```

失败条件：

```text
TRANSLATION_FAILED
UNSUPPORTED_PREDICATE
UNSUPPORTED_COMPARATOR
GROUNDING_FAILED
AMBIGUOUS_GROUNDING
LTL_PARSE_FAILED
AUTOMATON_COMPILATION_FAILED
NO_ENABLED_TRANSITION
NO_GOAL_REGION
NO_PATH
MAX_PROGRESS_STEPS
INVALID_OUTPUT_PATH
```

失败 episode 仍应输出合法 SemPathBench prediction record。推荐 trajectory 使用：

```text
[[start_row, start_col]]
```

并在 `limp.status` / `limp.failure_reason` 中记录失败原因。这样 shared evaluator 和 summary writer 不会因为缺少 trajectory 中断，同时 failure 仍会在 metrics 中自然体现。

---

## 8. Output Record

Prediction JSON 顶层和其他 methods 对齐：

```json
{
  "version": 1,
  "method": "LIMP",
  "prediction_id": "LIMP_xxxxxxxx",
  "map_id": "...",
  "scene_id": "...",
  "instruction_id": "...",
  "difficulty_level": "...",
  "instruction_file": "...",
  "created_at": "...",
  "trajectory": [[row, col], ...],
  "metrics": {...},
  "metrics_summary": {...},
  "limp": {...},
  "instruction": {...}
}
```

`limp` metadata 至少保存：

```json
{
  "description": "...",
  "input_contract": "...",
  "status": "SUCCESS_ACCEPTING_STATE",
  "failure_reason": null,
  "model": "...",
  "translation_mode": "llm",
  "grounding_mode": "core",
  "stage1_ltl": "...",
  "stage2_ltl": "...",
  "encoded_ltl": "...",
  "encoding_map": {...},
  "parsed_predicates": [...],
  "parsed_crds": [...],
  "candidate_counts": {...},
  "groundings": [...],
  "unsupported_predicates": [...],
  "unsupported_comparators": [...],
  "automaton_summary": {...},
  "official_component_reuse": {...},
  "method_level_deviations": [...],
  "logic_trace": [...],
  "task_state_sequence": [...],
  "tpsm_summaries": [...],
  "planner_segments": [...],
  "runtime_by_module": {...}
}
```

这部分很重要，因为 LIMP 的价值之一是可解释和可验证。

---

## 9. Experimental Variants

### 9.1 Future Main: LIMP-Core

```text
- two-stage LLM translation
- near[...] only
- original comparator family in 2D
- room containment extension
- hard constraints through formal LTL / automaton-derived TPSM
- no soft-cost planning
- no nearest / farthest unless explicitly represented by core comparator
```

这是 formal backend 接入后的主结果。当前 residual-debug backend 只能报告为 `LIMP-Nav-Prototype` smoke-test / diagnostic。

### 9.2 LIMP-Extended-Grounding

```text
Future LIMP-Core
+ nearest
+ farthest
+ contains / not_contains
+ start-relative ranking
```

用于分析 grounding extension 对 SemPathBench 的影响。

### 9.3 Oracle-Translation Diagnostic

使用 benchmark symbolic annotation 或 heuristic structure 构造 LTL，只用于诊断：

```text
translation error vs grounding / planning error
```

不能作为 main LIMP result。

### 9.4 Oracle-Grounding Diagnostic

使用 benchmark object IDs 做 grounding，只用于诊断：

```text
grounding error vs planning error
```

不能作为 main LIMP result。

Diagnostic variants 必须满足：

- 默认 CLI 不启用 oracle mode。
- 输出 metadata 必须标记 `is_diagnostic: true`。
- `summary.json` 应能区分 main run 和 diagnostic run，避免混入主表。
- README 中必须说明 oracle variants 使用了 benchmark annotation，不能和 main LIMP 数字直接比较。
- main `LIMP` result 的 inference path 仍然不能读取 `hard_constraints`、`soft_constraints`、instruction `objects` 或 human trajectory。

---

## 10. Testing Plan

建议添加 focused unit tests：

```text
tests/test_crd_parser.py
tests/test_comparators_2d.py
tests/test_scene_adapter.py
tests/test_goal_regions.py
tests/test_progression.py
tests/test_progressive_planner.py
tests/test_official_equivalence.py
```

最小测试覆盖：

- parse `near[chair]`
- parse nested CRD
- left / right / above / below comparator
- between comparator
- object inside room
- object goal region excludes occupied object cells
- room goal region is traversable cells inside room
- hard forbidden region blocks A*
- simple `F A`
- ordered `F (A & F B)`
- ordered `F (A & F (B & F C))`
- persistent avoidance `F (A & F B) & G !C`
- until formula `(!C) U A`
- after-visit prohibition `F A & G (A -> G !C)`
- next operator `F (A & X F B)`
- disjunction `F (A | B)`
- multiple eventuals `F A & F B`
- ambiguous grounding returns `AMBIGUOUS_GROUNDING`
- explicit nearest / farthest selector may rank candidates
- no-path returns failure status

---

## 11. Implementation Phases

### Phase 1: Skeleton Runner

- Add `run.py`, `__init__.py`, `__main__.py`, `config.py`.
- Copy tutorial runner shape.
- Stub `build_limp_trajectory`.
- Produce valid prediction JSON with failure status if pipeline not ready.

### Phase 2: Scene Adapter

- Convert `map_state` to object / room candidate registry.
- Extract cells, centroid, bbox, category, room membership.
- Add debug inventory metadata.

### Phase 3: Predicate Parser And Official Logic Reuse Audit

- Implement LIMP predicate parser.
- Implement CRD parser.
- Identify reusable official prompt / encoding / CRD / LTL / progression components.
- Vendor reusable official method-level code when direct reuse would pull in robot runtime dependencies.
- Use local LTL progression only as a documented fallback.

### Phase 4: LLM Translation

- Add LIMP two-stage prompts.
- Add Gemini client / cache consistent with other methods.
- Add unsupported predicate detection.

### Phase 5: 2D Grounding

- Implement comparator library.
- Ground object and room CRDs.
- Record candidate lists, comparator results, selected instance if unique, and ambiguity failures.
- Return `AMBIGUOUS_GROUNDING` when multiple candidates remain without an explicit ranking selector.

### Phase 6: TPSM And Progressive Planner

- Build goal / forbidden regions.
- Use shared grid A* or local wrapper around `scripts.methods.util.grid_astar`.
- Progress LTL state after each segment.
- Return concatenated trajectory.

### Phase 7: README And Documentation

- Add `README.md` in Lang2LTL style.
- Document faithful adaptation choices and unsupported constraints.
- Add command examples.

### Phase 8: Evaluation Sanity Run

- Run `--limit 1 --overwrite --verbose`.
- Run a small `valunseen` subset.
- Inspect prediction JSON, trajectory image, summary metrics.

---

## 12. Fidelity Checklist

Only call the method `LIMP` as a main baseline if these are true:

```text
[ ] official method-level components are reused or vendored where feasible
[ ] every method-level reimplementation documents its original LIMP counterpart and deviation
[ ] runner structure follows tutorial style
[ ] uses two-stage translation
[ ] produces LIMP-style LTL
[ ] uses near[...] predicates
[ ] uses CRDs for referents
[ ] enumerates candidate object / room instances from map_state
[ ] grounds CRDs with deterministic 2D comparators but does not use unstated tie-breaking
[ ] returns AMBIGUOUS_GROUNDING for unresolved multiple candidates
[ ] uses official LIMP LTL / task progression path or the same underlying library where feasible
[ ] never silently falls back from missing Spot / formal backend to residual progression
[ ] treats local residual progression as explicit debug / smoke-test only
[ ] does not truncate goal regions in main configuration
[ ] builds stage-dependent TPSMs
[ ] progressively plans toward enabled transitions
[ ] records explicit failure status
[ ] does not use hard_constraints / soft_constraints / objects / human trajectory during inference
[ ] does not use evaluator feedback during inference
```

If the formal backend and official-aligned progression are still missing, label the method:

```text
LIMP-Nav-Prototype
```

rather than reporting it as:

```text
LIMP
```
