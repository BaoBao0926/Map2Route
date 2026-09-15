# Grounding2Route 当前实现说明

本文档记录 `scripts/methods/grounding2route` 当前版本的完整实现，并补充 SemPathBench benchmark setup 与当前实验结果。结构按照六个部分展开：

1. Overview
2. SemPathBench Setup
3. Benchmark Results
4. Parser
5. Grounding
6. Planner

Grounding2Route 的核心思想是：让 LLM 只负责把自然语言翻译成可检查的符号意图，不让 LLM 直接选择 object id、room id、坐标、路径或 planner 参数；随后由确定性 grounding 和 planner 在当前 map 上完成对象绑定、区域构造和路径搜索。

## 1. Overview

### 1.1 端到端流程

当前主入口是 `scripts/methods/grounding2route/run.py`。一次 episode 的执行流程是：

```text
instruction.json + map_state
        |
        v
parse_instruction(...)
        |
        v
GPProgramSpec
        |
        v
SceneMap(map_state, instruction)
        |
        v
ground_program(program, scene)
        |
        v
GroundedProgram
        |
        v
plan_grounded_program(scene, grounded)
        |
        v
trajectory
        |
        v
evaluate_prediction(...) + visualization + output json
```

其中 `pipeline.py` 是单条 instruction 的总控：

- parser 成功后，把 `GPProgramSpec` 写入 `steps["parse"]`。
- grounding 成功后，把 `GroundedProgram` 写入 `steps["grounding"]`。
- planning 成功后，把 `PlannedProgram` 写入 `steps["planner"]`。
- 如果 parse、grounding 或 planning 失败，pipeline 返回 start-only trajectory，保证每条 episode 都有一个合法输出。

### 1.2 输入契约

Grounding2Route 在 inference 时允许使用：

- instruction text
- `start_pose`
- occupancy / traversable map
- room layer
- object instance layer
- `room_instances`
- `object_instances`
- map-derived geometry，例如 room cells、object cells、passage regions

Grounding2Route 不使用：

- annotation 里的 hard constraints
- annotation 里的 soft constraints
- `instruction.objects`
- human expert trajectory
- evaluator feedback

这点在 `pipeline.py` 的 metadata 里也会记录为 `input_contract`，用于保证方法没有偷看 evaluation label。

### 1.3 中间表示

核心 IR 在 `ir.py`：

- `GPRef`：已经或将要被 grounded 的引用，可以是 `position`、`entity`、`room`、`region`。
- `GPExpr`：符号表达式，例如 `entities("chair")`、`kth_nearest(...)`、`circle(...)`。
- `GPBinding`：全局 `let` 变量。
- `GPConstraintSpec`：parser 输出的 hard/soft constraint。
- `GPSegmentSpec`：一个 navigation segment，包含 start、target 和 constraints。
- `GPProgramSpec`：parser 的完整输出。
- `GroundedConstraint`：已经绑定到具体 `GPRef` 的约束。
- `GroundedSegment`：已经绑定 start 和 target 的 segment。
- `GroundedProgram`：grounder 输出。
- `PlannedSegment` / `PlannedProgram`：planner 输出。

设计上 parser 和 planner 不直接耦合。当前 parser 支持 `intent` 和 `api` 两种模式：JSON intent 和受限 Python-like API program 都会编译成统一的 `GPProgramSpec`，下游 grounding 和 planner 不需要知道上游格式。

### 1.4 运行模式

`config.py` 当前支持两种 parser mode：

```text
intent    JSON intent parser，默认主路径
api       restricted API-program parser，用于 ablation/debug
```

常用命令：

```bash
python scripts/methods/grounding2route/run.py \
  --set valunseen \
  --parse-mode intent \
  --workers 4 \
  --overwrite \
  --verbose
```

`--workers` 使用 episode-level 并行。每个 worker 会创建自己的 `GeminiClient`，因此主要加速的是等待 API 回复的时间。主线程负责收集 future、持续写 summary，避免多个线程同时写同一个 summary 文件。

### 1.5 输出文件

每条 episode 会保存：

- `instruction_xxxxxx.json`：最终 prediction record。
- `instruction_xxxxxx.steps.json`：parse、grounding、repair、planner 的中间步骤。
- trajectory visualization。
- trajectory + GT overlay visualization，其中 GT 使用黑色点，算法轨迹使用原本的颜色。

prediction record 里包含：

- `trajectory`
- `metrics`
- `metrics_summary`
- `grounding2route`
- `instruction`
- `runtime_seconds`

当前实现把 `trajectory` 保留在结果 JSON 中，并在 evaluator/recompute 逻辑里支持重新生成图片。

### 1.6 主要超参数

Grounding2Route 自身的 planner/grounding 超参数在 `config.py`：

```text
OBJECT_GOAL_RADIUS_METERS = 0.30
NEAR_RADIUS_METERS = 1.50
FAR_SIGMA_METERS = 1.50
RELATIVE_RADIUS_METERS = 1.50
CLEARANCE_RADIUS_METERS = 0.75

WEIGHT_NEAR = 8.00
WEIGHT_FAR = 96.00
WEIGHT_RELATIVE = 640.00
WEIGHT_CLEARANCE = 36.00
WEIGHT_SMOOTHNESS = 0.02

CIRCLE_OBJECT_WAYPOINT_CLEARANCE_CELLS = 16.0
CIRCLE_OBJECT_WAYPOINT_CLEARANCE_WEIGHT = 8.0
CIRCLE_ROOM_WAYPOINT_CLEARANCE_CELLS = 10.0
CIRCLE_ROOM_WAYPOINT_CLEARANCE_WEIGHT = 16.0
CIRCLE_WAYPOINT_SEARCH_RADIUS_CELLS = 32

MAX_EXPANSIONS = 250000
MAX_GOAL_HEURISTIC_CELLS = 512
MAX_GROUNDING_REPAIRS = 2
```

evaluation 相关超参数在 `scripts/evaluation/hyparameter.py`：

```text
PLR_MIN_EFFECTIVE_PATH_LENGTH_GRID = 1.0
SCS_BAD_RATIO_THRESHOLD = 1.5
SCS_RATIO_EPSILON = 1e-8
PATH_SHAPE_TOLERANCE_GRID = 10.0
PATH_SHAPE_NDTW_SUCCESS_DISTANCE = 4.0
EVALUATE_MOVE_SMOOTHNESS = False
```

这里要区分 planner 和 evaluator：

- planner 的 path-shape 逻辑负责生成更像 circle/path-shape 的路线。
- evaluator 的 path-shape SCS 使用 tolerant nDTW。当前设置是 GT 周围 10 grid 内不扣分，超过 tolerance 后按 `success_distance=4 grid` 的尺度指数衰减。
- PLR 对非空 trajectory 使用最小有效长度 1 grid，也就是 start-only trajectory 不再让 PLR 变成 0。

## 2. SemPathBench Setup

### 2.1 Benchmark 任务形式

SemPathBench 的每条任务是一个 grid-world navigation episode。输入包含一个来自 ProcTHOR 的语义地图和一条自然语言 instruction；方法需要输出一条二维 grid trajectory：

```json
[
  [row_0, col_0],
  [row_1, col_1],
  ...
]
```

trajectory 必须在 map 的 traversable space 中移动。Grounding2Route 的 planner 使用 8-connected movement，也就是上下左右和四个对角方向都可以走，但每一步都需要通过 `can_traverse_between(...)` 检查，避免穿墙、穿过障碍或斜穿不可通行 corner。

当前 Grounding2Route 实验主要使用 ProcTHOR val-unseen maps。当前本地 instruction root 中有：

```text
resources/instructions/procthor/003_valunseen/instruction_files  25 instructions
resources/instructions/procthor/006_valunseen/instruction_files  25 instructions
resources/instructions/procthor/009_valunseen/instruction_files  25 instructions
resources/instructions/procthor/012_valunseen/instruction_files  25 instructions
resources/instructions/procthor/015_valunseen/instruction_files  25 instructions
resources/instructions/procthor/018_valunseen/instruction_files   1 instruction
```

因此当前本地 val-unseen instruction 文件总数是 126。当前 Grounding2Route summary 中的 50 条结果来自 output root 里已经存在的两个 map：

```text
resources/methods/grounding2route/previous/grounding2route_previous/Grounding2Route/procthor/003_valunseen  25 predictions
resources/methods/grounding2route/previous/grounding2route_previous/Grounding2Route/procthor/006_valunseen  25 predictions
```

这也是为什么你只看 `003_valunseen` 时会看到 25 条，但 `resources/methods/grounding2route/previous/grounding2route_previous/Grounding2Route/summary.json` 会聚合成 50 条。

### 2.2 Map 文件

ProcTHOR map 保存在：

```text
resources/maps/procthor/<split>/<map_id>/<map_leaf>.json
resources/maps/procthor/<split>/<map_id>/<map_leaf>_maps.npz
resources/maps/procthor/<split>/<map_id>/<map_leaf>_metadata.json
resources/maps/procthor/<split>/<map_id>/<map_leaf>.png
resources/maps/procthor/<split>/<map_id>/<map_leaf>.ppm
resources/maps/procthor/<split>/<map_id>/<map_leaf>_thinggraph.json
resources/maps/procthor/<split>/<map_id>/<map_leaf>_view.png
```

以 `003_valunseen` 为例：

```text
resources/maps/procthor/valunseen/003_valunseen/003_valunseen.json
resources/maps/procthor/valunseen/003_valunseen/003_valunseen_maps.npz
resources/maps/procthor/valunseen/003_valunseen/003_valunseen_metadata.json
resources/maps/procthor/valunseen/003_valunseen/003_valunseen.png
resources/maps/procthor/valunseen/003_valunseen/003_valunseen.ppm
resources/maps/procthor/valunseen/003_valunseen/003_valunseen_thinggraph.json
resources/maps/procthor/valunseen/003_valunseen/003_valunseen_view.png
```

`003_valunseen.json` 当前包含：

- `grid_size = 652`
- `layers`
- `layer_legends`
- `metadata`
- `room_instances`
- `object_instances`

其中 `layers` 包含：

```text
occupancy
room
object_instance
```

`room_instances` 是房间实例列表，包含 room id、category 和相关几何信息。`object_instances` 是物体实例列表，包含 object id、object type/category、world position、grid row/col、footprint cells 等信息。

当前几个 val-unseen map 的基本规模如下：

| Map | Grid Size | Rooms | Objects | Resolution |
| --- | ---: | ---: | ---: | ---: |
| `003_valunseen` | 652 | 8 | 276 | 0.05 |
| `006_valunseen` | 680 | 8 | 299 | 0.05 |
| `009_valunseen` | 633 | 8 | 245 | 0.05 |
| `012_valunseen` | 678 | 8 | 293 | 0.05 |
| `015_valunseen` | 654 | 8 | 287 | 0.05 |
| `018_valunseen` | 633 | 8 | 282 | 0.05 |

Grounding2Route 在 inference 时读取 `<map_leaf>.json`，通过 `load_map_state(map_id)` 得到 map_state，然后 `SceneMap` 会从中构造可通行 grid、room refs、object refs、start ref 和 passage refs。

### 2.3 Map 生成

README 中提供的 ProcTHOR map export 命令是：

```bash
python -m scripts.make_maps.procthor.transform_procthor_to_map \
  --split train \
  --resolution 0.05 \
  --index -1 \
  --number 120 \
  --scene_size 400 \
  --sequence bigscene \
  --resume true
```

主要参数含义：

| 参数 | 说明 |
| --- | --- |
| `--split` | ProcTHOR split，例如 `train` 或 `val`。 |
| `--index` | 单个 house index；设为 `-1` 时进入 batch export。 |
| `--number` | batch export 时需要接受的 scene 数量。 |
| `--resolution` | grid cell size，同时传给 AI2-THOR `gridSize`。当前常用 `0.05`。 |
| `--scene_size` | 只保留真实总房间面积不小于该值的 scene。 |
| `--sequence` | `sequence` 按原始顺序，`bigscene` 优先导出大场景。 |
| `--resume` | 跳过已经存在的目标 map。 |
| `--padding` | world bounds 外额外 padding。 |
| `--output-prefix` | 输出目录或文件前缀。 |
| `--dataset-name` | 传给 `prior.load_dataset` 的 ProcTHOR dataset 名称。 |
| `--skip-overview` | 只导出地图文件，不生成 overview PNG。 |
| `--skip-simple-demo` | 不导出 simple-demo 风格 JSON/PPM/PNG。 |
| `--ai2thor-base-dir` | AI2-THOR 本地 cache 目录。 |

导出的 map id 形如：

```text
001_train
002_train
003_valunseen
006_valunseen
```

数字部分是 accepted map 的 one-based position。metadata 中同时保留 ProcTHOR 源信息和 SemPathBench map 信息，例如 `procthor_split`、`procthor_index`、`procthor_scene_id`、`map_id`、`scene_size`、`room_metadata`。

### 2.4 Instruction 文件

instruction 文件保存在：

```text
resources/instructions/procthor/<map_id>/instruction_files/instruction_xxxxxx.json
```

可以通过交互式脚本生成：

```bash
python scripts/make_instruction/make_instruction.py --port 8002
```

一条 instruction JSON 当前包含的信息包括：

- `id`
- `map_id`
- `template_instruction_id`
- `object_count`
- `objects`
- `human_expert_trajectory`
- `start_pose`
- `feasible`
- `name`
- `notes`
- `created_by`
- `hard_constraints`
- `soft_constraints`
- `ordering`
- `difficulty_level`
- `created_at`
- `updated_at`
- `instruction`

其中 `objects` 是 benchmark annotation 中的目标 object 序列，包含 order、object_id、name、category 和 center。Grounding2Route inference 不使用这个字段，只用于 benchmark/evaluation/debug。

`human_expert_trajectory` 是专家轨迹。Grounding2Route inference 不使用它，但 evaluator 会用它计算 PLR 和 path-shape preference。

`hard_constraints` 是必须满足的硬约束，常见形式包括：

- must-pass region
- ordered must-pass region
- must-avoid region

`soft_constraints` 是软偏好，当前 evaluator 支持的主要类型包括：

- `near_preference`
- `far_preference`
- `relative_preference`
- `clearance`
- `path_shape_preference`
- `move_smoothness`，当前默认不计入 SCS

`difficulty_level` 当前主要是：

```text
easy
hard
```

summary 会按 difficulty 聚合指标。

### 2.5 方法输出结构

Grounding2Route 默认输出到：

```text
resources/methods/grounding2route/main_result
```

每条 prediction record 的路径是：

```text
resources/methods/grounding2route/main_result/<map_id>/instruction_xxxxxx.json
```

对应中间文件是：

```text
resources/methods/grounding2route/main_result/<map_id>/instruction_xxxxxx.steps.json
```

prediction JSON 主要字段：

- `version`
- `method`
- `prediction_id`
- `map_id`
- `scene_id`
- `instruction_id`
- `difficulty_level`
- `instruction_file`
- `created_at`
- `metrics`
- `metrics_summary`
- `grounding2route`
- `instruction`
- `runtime_seconds`
- `runtime`
- `trajectory`

`grounding2route` 内会记录：

- parser status/mode/diagnostics
- grounding status/failure reason/attempt
- planner status/failure reason
- repair attempt count
- alias resolution attempts
- intermediate steps path
- method-level deviations

`.steps.json` 主要用于 debug，包含：

- `steps.parse`
- `steps.parse_attempts`
- `steps.grounding`
- `steps.grounding_attempts`
- `steps.repair_attempts`
- `steps.alias_resolution_attempts`
- `steps.planner`

### 2.6 评估指标

SemPathBench 当前主要报告四个指标：

```text
PLR
HCS
SCS
OSS
```

#### PLR

PLR 是 path length ratio：

```text
PLR = min(expert_path_length / prediction_path_length, 1.0)
```

如果没有专家轨迹，则 PLR 为 `None`。

当前实现对非空 trajectory 使用最小有效长度：

```text
PLR_MIN_EFFECTIVE_PATH_LENGTH_GRID = 1.0
```

也就是说 start-only trajectory 至少按 1 grid 计算，避免解析失败导致空移动时 PLR 直接变成 0。

#### HCS

HCS 是 hard constraint score。Evaluator 会检查 trajectory 是否按有效顺序完成 must-pass constraints，并且是否违反 must-avoid constraints。

HCS 细节会记录：

- matched must-pass 数量
- valid completed count
- total must-pass count
- ignored record must-pass
- first failed transition
- failure type
- must-avoid violations
- per-hard-constraint diagnostics

#### SCS

SCS 是 soft constraint score，等于所有 active soft constraint score 的平均值。

当前 SCS 会记录 per-type average：

```text
near_preference
far_preference
relative_preference
clearance
path_shape_preference
```

SCS normalization 超参数：

```text
SCS_BAD_RATIO_THRESHOLD = 1.5
SCS_RATIO_EPSILON = 1e-8
```

对于 path-shape preference，当前 evaluator 使用 tolerant nDTW：

```text
PATH_SHAPE_TOLERANCE_GRID = 10.0
PATH_SHAPE_NDTW_SUCCESS_DISTANCE = 4.0
```

含义是：

- prediction 点与 GT 对齐时，距离 GT 10 grid 以内不扣分。
- 超过 10 grid 后才累计 excess DTW cost。
- 平均 excess deviation 达到约 4 grid 时，nDTW 分数约为 `exp(-1) = 0.37`。
- path-shape 是顺序敏感的，也就是同样绕了一圈但方向或局部顺序与 GT 不一致，仍然会扣分。

#### OSS

OSS 是 episode-level 总分：

```text
OSS = PLR * HCS * SCS
```

如果某条 episode 没有 active soft constraints，则 SCS 在 OSS 中按 1.0 处理。

### 2.7 Recompute 和可视化

修改 evaluator 或 planner 后，可以重新计算 method metrics：

```bash
python scripts/evaluation/recompute_method_metrics.py \
  --method Grounding2Route
```

当前 recompute 支持：

- 更新每个 `instruction_xxxxxx.json` 中的 `metrics`
- 写入 `scs_config`
- 写入 `scs_detail_averages`
- 重新生成 trajectory 图片
- 重新生成 trajectory + GT overlay 图片
- 重写 method `summary.json`

输出图片用于快速判断：

- 轨迹是否到达目标
- 是否经过 hard constraint region
- 是否符合 path-shape preference
- 是否与 GT 大体重合

## 3. Benchmark Results

### 3.1 当前比较的 runs

目前有五个主要 Grounding2Route 版本用于比较：

| Run | 结果目录 | 主要变化 | 结论 |
| --- | --- | --- | --- |
| Grounding2Route-1st | `resources/methods/grounding2route/previous/grounding2route_previous/1st_run` | 主要验证 planner 链路 | Planner 基本可以工作；主要瓶颈暴露在 parser/grounding。 |
| Grounding2Route-2nd | `resources/methods/grounding2route/previous/grounding2route_previous/2nd_run` | 修复 parser 和 grounding 基础问题 | Parser/grounding 基本连通，成功率提升，但复杂语义仍不稳定。 |
| Grounding2Route-3rd | `resources/methods/grounding2route/previous/grounding2route_previous/Grounding2Route` | 修复 grounding、cost field、path-shape、PLR/evaluation 记录 | PLR 明显提高；剩余主要问题是复杂目标选择和 path-shape。 |
| Grounding2Route-Intent-Mixed | `resources/methods/grounding2route/previous/grounding2route_previous/Grounding2Route_intent` | 初始 parser 输出 JSON intent，但 grounding repair 曾混入 DSL repair | HCS/OSS 提升，但不是纯 JSON 消融。 |
| Grounding2Route-Intent-PureJSON | `resources/methods/grounding2route/previous/Grounding2Route_intent_pure` | 初始 parsing 和 grounding repair 都保持 JSON intent | 当前最好结果，更干净地说明 JSON intent 比直接 DSL 更稳。 |

### 3.2 Overall/Easy/Hard 指标

| Method | Overall N | PLR | HCS | SCS | OSS | Easy N | Easy PLR | Easy HCS | Easy SCS | Easy OSS | Hard N | Hard PLR | Hard HCS | Hard SCS | Hard OSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Grounding2Route-1st | 25 | 0.581 | 0.300 | 0.607 | 0.201 | 17 | 0.623 | 0.265 | 0.680 | 0.179 | 8 | 0.491 | 0.375 | 0.452 | 0.247 |
| Grounding2Route-2nd | 50 | 0.635 | 0.490 | 0.646 | 0.338 | 34 | 0.707 | 0.515 | 0.717 | 0.367 | 16 | 0.482 | 0.438 | 0.495 | 0.276 |
| Grounding2Route-3rd | 50 | 0.910 | 0.567 | 0.723 | 0.417 | 35 | 0.910 | 0.629 | 0.720 | 0.481 | 15 | 0.909 | 0.422 | 0.731 | 0.266 |
| Grounding2Route-Intent-Mixed | 50 | 0.862 | 0.683 | 0.780 | 0.492 | 35 | 0.868 | 0.714 | 0.798 | 0.533 | 15 | 0.850 | 0.611 | 0.739 | 0.396 |
| Grounding2Route-Intent-PureJSON | 50 | 0.863 | 0.723 | 0.783 | 0.519 | 35 | 0.864 | 0.743 | 0.798 | 0.557 | 15 | 0.862 | 0.678 | 0.748 | 0.432 |

### 3.3 SCS 细分指标

当前 `Grounding2Route-3rd` 的 SCS detail average：

| Split | Near | Far | Relative | Clearance | Path Shape |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 0.502 | 0.868 | 0.901 | 0.803 | 0.324 |
| Easy | 0.467 | 0.974 | 0.861 | 0.754 | 0.423 |
| Hard | 0.533 | 0.745 | 1.000 | 0.917 | 0.226 |

当前 `Grounding2Route-Intent-PureJSON` 的 SCS detail average：

| Split | Near | Far | Relative | Clearance | Path Shape |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 0.460 | 0.931 | 0.908 | 0.855 | 0.443 |
| Easy | 0.401 | 0.937 | 0.860 | 0.829 | 0.652 |
| Hard | 0.513 | 0.925 | 0.969 | 0.914 | 0.275 |

### 3.4 结果解读

从 1st 到 3rd，最明显的变化是 PLR：

```text
0.581 -> 0.635 -> 0.910
```

这说明 planner 链路、fallback、cost field 和 evaluation 记录修复以后，Grounding2Route 已经能稳定输出长度合理的路线。

从 3rd 到 Intent-PureJSON，最明显的变化是 HCS 和 OSS：

```text
HCS: 0.567 -> 0.723
OSS: 0.417 -> 0.519
```

这说明当前最大收益不是继续微调 A* 权重，而是 parser 生成的 symbolic task 更稳定。JSON intent mode 让 LLM 输出更结构化，减少了 DSL 格式错误、漏 segment、漏 scope、错误 `unique(...)` 和复杂 count relation 写错的问题。

SCS 的变化也说明 path-shape 有改善但仍是短板：

```text
Grounding2Route-3rd path_shape overall: 0.324
Intent-PureJSON path_shape overall: 0.443
```

当前 tolerant nDTW 已经给了较大容忍度，所以 path-shape 分数低时，更多说明生成轨迹在顺序和整体形状上仍不像 expert，而不是 evaluator 过于严格。

### 3.5 当前最好版本

当前最好结果是 `Grounding2Route-Intent-PureJSON`：

```text
Overall: PLR 0.863, HCS 0.723, SCS 0.783, OSS 0.519
Easy:    PLR 0.864, HCS 0.743, SCS 0.798, OSS 0.557
Hard:    PLR 0.862, HCS 0.678, SCS 0.748, OSS 0.432
```

它的 pipeline 是：

```text
instruction
  -> LLM outputs typed JSON intent graph
  -> code validates JSON
  -> code compiles JSON to GPProgramSpec
  -> deterministic grounding
  -> if grounding fails, LLM repairs JSON intent graph
  -> deterministic grounding
  -> semantic A*
  -> trajectory
```

Grounding 和 planner 的核心逻辑与 Grounding2Route-3rd 保持一致，所以这是一个比较干净的 parser representation ablation。

## 4. Parser

### 4.1 Parser 的职责

Parser 只做语义解析，不做 grounding。它需要把自然语言 instruction 转成 scene-independent 的 symbolic program。

Parser 允许表达：

- ordered navigation segments
- 每个 segment 的 start 和 destination
- object category / room category
- room containment
- object filtering
- count condition
- set operation
- nearest / farthest / ordinal / largest / smallest selection
- previous target / current segment start / task start reference
- hard required region
- hard forbidden region
- ordered must-visit region
- soft near / far / relative preference
- segment scope
- spatial scope
- path-shape-like preference，例如 circle around object/room

Parser 明确不能输出：

- object id
- room id
- coordinate
- grid cell
- path
- planner cost/weight
- evaluator score
- free-form unresolved placeholder

### 4.2 API-program parser 模式

`--parse-mode api` 使用 `parse/api_program.py` 中的 `llm_parse_api_program_instruction()`。

Prompt 文件是：

```text
scripts/methods/grounding2route/parse/gemini_api_program_prompt.md
```

LLM 输出受限 Python-like API program，例如：

```python
candidate_paintings = api.entities("painting")
target_object = api.kth_nearest(candidate_paintings, start_position, k=1, metric="geodesic")
s1 = api.go_to(target_object, constraints=[api.prefer_near(target_object)])
```

API program 的语义：

- assignment 会编译成 `GPBinding`。
- `api.go_to(...)` 会编译成有序 `GPSegmentSpec`。
- `api.require_visit(...)`、`api.forbid(...)`、`api.prefer_near(...)` 等会编译成 `GPConstraintSpec`。
- parser 只允许赋值、字面量、变量引用、list/tuple 和白名单 `api.<function>` 调用。
- 不允许 import、循环、任意 attribute access、文件/网络/eval/exec。

API-program mode 的好处是：

- LLM 用接近代码的形式组合 API，复杂表达比嵌套 JSON 更容易读。
- AST validator 仍能严格限制语法和函数集合。
- parser 可以把 API program 编译成统一的 `GPProgramSpec`，所以 grounding 和 planner 完全复用。
- `program.source` 就是 LLM 输出的 API program，artifact 也会保存为 `parser_api.py`。

### 4.4 Expression/operator registry

Parser 只能使用 grounder 支持的 operator。当前主要 operator 包括：

```text
target_of
entities(category)
rooms(category)
in(entity_set, room_or_rooms)
room_of(reference)
contains(room, category)
count_next_to(category, reference)
count_near(category, reference)
adjacent_rooms(room)
passage_regions(room_a, room_b)
union
intersection
exclude
count
where
unique
choose_any
kth_nearest
kth_farthest
kth_largest
kth_smallest
order_by_distance
closest_pair_member
region_of
room_region
midpoint_region
between_region
near_region
boundary_region
side_region
half_room
circle
compare
and
or
not
near_to
far_from
next_to
on_top_of
in_corner
between
```

其中 `count_next_to(category, reference)` 和 `count_near(category, reference)` 是为了解决 object count relation 加的 deterministic operator。

典型例子是：

```text
the dining table with exactly two chairs
```

更稳定的表达是：

```text
where(
    entities("dining_table"),
    compare(count_next_to("chair", self), "==", 2)
)
```

这样避免了嵌套 `where(..., self)` 时内层 `self` 覆盖外层 `self` 的问题。

### 4.5 Parser repair

Parser 有两类 repair：

1. Parse repair
2. Grounding repair

Parse repair 用于初始 LLM 输出无法被 parser 解析时：

- api mode 调用 API-program parse repair。
- repair prompt 会包含原始 instruction、上一次输出和 parse error。

Grounding repair 用于 parser 输出 syntactically valid，但 deterministic grounding 失败时：

- api mode 调用 `llm_repair_api_after_grounding_failure()`，要求 LLM 返回新的 API program。
- repair prompt 会包含 grounding error、scene category summary 和 previous API program。
- repair 仍然不能输出 object id、room id、coordinate、cell 或 path。

默认 `MAX_GROUNDING_REPAIRS = 2`。每次 attempt 都会写入 `steps["grounding_attempts"]` 和 `steps["repair_attempts"]`。

### 4.6 Alias resolution

Grounding2Route 还有一个专门的 unknown object category alias resolver。

触发条件非常严格：

- 只有 grounding failure reason 是 `UNKNOWN_ENTITY_CATEGORY` 时才触发。
- 普通 grounding failure 不触发 alias resolver。
- parser mode 必须是 LLM-like，也就是 `llm` 或 `intent`。

如果出现未知 object category，代码会问 LLM：在当前 supported object categories 里，哪个 category 与未知词最相近。然后把结果写入 `alias.py`，后续运行可以复用这个增长出来的 alias table。

已有静态 alias 在 `config.py`，例如：

```text
armchair -> arm_chair
countertop -> counter_top
plant -> house_plant
tv -> tv_stand
door -> doorway
```

实际 canonicalization 会合并静态 alias 和动态 alias。

## 5. Grounding

### 5.1 Grounding 的职责

Grounding 把 parser 产生的 symbolic program 绑定到当前 map 中的具体 semantic references。

输入：

- `GPProgramSpec`
- `SceneMap`

输出：

- `GroundedProgram`

Grounding 仍然不生成路径，也不计算 evaluator score。它只负责：

- 解析 object/room category。
- 选择满足条件的 object/room。
- 构造 region。
- 展开 hard/soft constraints。
- 给 soft preference 设置 spatial scope。
- 为 planner 准备 `GroundedSegment`。

### 5.2 SceneMap

`SceneMap` 在 `grounding/scene.py`，它是 grounder 能访问的受限 map view。

初始化时会构造：

- `grid_size`
- `traversable`
- `room_layer`
- `object_layer`
- `resolution`
- `rooms`
- `entities`
- `object_to_room`
- `start`
- `passages`

`rooms` 来自 room layer 和 `room_instances`，每个 room 被表示为：

```text
GPRef(kind="room", id="room_...", category=..., center=..., cells=...)
```

`entities` 来自 object instance layer 和 `object_instances`，每个 object 被表示为：

```text
GPRef(kind="entity", id="object_...", category=..., room_id=..., center=..., cells=...)
```

`start` 来自 instruction 的 `start_pose`。如果 start_pose 不可通行，会找附近最近的 traversable cell。

`passages` 通过 room layer 中相邻 room 的边界 transition 构造。每个 passage 是一个 region ref：

```text
GPRef(kind="region", category="passage", construction={"type": "passage", "rooms": [...]})
```

### 5.3 Category canonicalization

Grounding 使用 `canonical_category()` 对 room/object category 做规范化：

- lower-case
- `-` 和空格替换成 `_`
- 应用 room aliases 或 entity aliases

例如：

```text
"living room" -> "living_room"
"counter top" -> "counter_top"
"television" -> "tv_stand"
```

如果 `entities(category)` 中的 category 不在当前 scene object inventory，grounder 会返回：

```text
UNKNOWN_ENTITY_CATEGORY: ...
```

如果 `rooms(category)` 中的 category 不在当前 scene room inventory，grounder 会返回：

```text
UNKNOWN_ROOM_CATEGORY: ...
```

只有前者会触发 alias resolver。

### 5.4 Expression evaluation

Grounder 用 `_eval_expr()` 递归执行 `GPExpr`。

环境变量初始包含：

```text
task_start -> scene.start
start_position -> scene.start
```

执行 bindings 时，每个 `let` 会被加入 env。执行 segment 时，会临时加入：

```text
segment_start
```

并在 segment 完成后移除。每个 segment 的 target 还会被加入 env：

```text
s1 -> target_ref
s2 -> target_ref
...
```

因此后续 segment 可以通过 segment id 或 binding name 引用之前的目标。

`where(candidate_set, predicate)` 会在 predicate 的 local env 中设置：

```text
self -> 当前 candidate
```

这使得表达式可以写成：

```text
where(entities("chair"), next_to(target_table))
where(rooms("bedroom"), compare(count(in(entities("bed"), self)), "==", 1))
```

### 5.5 Selection grounding

当前 selection operator 的行为：

- `entities(category)`：返回当前 scene 中该 category 的所有 object refs。
- `rooms(category)`：返回当前 scene 中该 category 的所有 room refs。
- `in(entity_set, room_or_rooms)`：筛选位于目标 room 集合中的 entity。
- `room_of(reference)`：返回 reference 所在 room。
- `contains(room, category)`：判断 room 中是否有某类 object。
- `unique(set)`：要求候选集长度严格等于 1。
- `choose_any(set, reference=...)`：当前实现按距离 reference 最近来选一个 deterministic candidate。
- `kth_nearest(set, reference, k, metric)`：按距离从近到远选第 k 个。
- `kth_farthest(set, reference, k, metric)`：按距离从远到近选第 k 个。
- `kth_largest(set, k)` / `kth_smallest(set, k)`：按 cells 数量排序。
- `order_by_distance(set, reference, order)`：返回排序后的 ref list。
- `closest_pair_member(candidates, references)`：选出与 reference set 最近的一项。

注意：`SceneMap.distance()` 当前 geodesic 和 euclidean 都实际使用 ref center 的欧氏距离。这是一个重要实现细节。也就是说 parser 可以保留 `metric="geodesic"`，但当前 grounding ranking 还没有真正跑 geodesic distance transform。

### 5.6 Predicate grounding

`where` 中支持的 predicate 包括：

- `near_to(reference)`
- `far_from(reference)`
- `next_to(reference)`
- `on_top_of(reference)`
- `in_corner(room)`
- `between(first, second)`
- `compare(left, operator, right)`
- `and(...)`
- `or(...)`
- `not(...)`

当前距离阈值：

- `near_to` 使用 `max(2 grid, NEAR_RADIUS_METERS / resolution)`。
- `next_to` 使用 `max(2 grid, 0.75 / resolution)`。
- `far_from` 判断距离大于 near threshold。
- `between` 使用 object center 到两 reference center 连线的距离。
- `in_corner` 用 object center 到 room bbox 四个角的距离近似。
- `on_top_of` 用 center 的 row/col 相对位置近似。

这些 predicate 是 deterministic 的，失败或空集会在后续 selection 阶段暴露为 `EMPTY_CANDIDATE` 或 `AMBIGUOUS_CANDIDATE`。

### 5.7 Region construction

Grounder 支持把 object/room/relation 转成 region：

- `region_of(entity)`：object cells。
- `room_region(room)`：room cells。
- `near_region(reference, radius)`：reference 附近 disk，默认限制在 reference 所在 room 内。
- `midpoint_region(first, second)`：两者中心点附近 disk。
- `between_region(first, second)`：两者中心连线附近的 corridor-like region。
- `boundary_region(room)`：room 的边界 cells。
- `side_region(center_ref, side_reference)`：center_ref 所在 room 中朝向 side_reference 的半边。
- `half_room(room, reference)`：room 中朝向 reference 的半边。
- `passage_regions(room_a, room_b)`：两个 room 之间的 passage cells。

Hard constraints 中如果出现 entity 或 room，会通过 `_ensure_region_if_hard()` 自动转成 region：

- entity -> `region_of(entity)`
- room -> `room_region(room)`
- region -> 保持 region

### 5.8 Circle/path-shape grounding

`circle(reference, fraction, direction, start_toward)` 用于 path-shape-like preference 或 around/circle 指令。

Grounding 阶段不会直接生成完整曲线，而是生成一组有序 waypoint regions。Planner 后续必须按顺序经过这些 regions。

当前逻辑：

- `fraction` 被限制在 `[0.1, 1.0]`。
- waypoint 数量约为 `ceil(8 * fraction)`，最少 3 个。
- 如果 reference 是 room，使用 room-scale loop centers。
- 如果 reference 是 object，先找 object 所在 room，再围绕 object 生成 loop centers。
- 默认 direction 是 counterclockwise。
- 支持 `start_toward`，用于把 loop 的起点旋转到更符合语义的位置。
- full loop 会追加第一个 waypoint，让 planner 形成闭环效果。

为了避免 waypoint 太贴墙或太贴 object，当前有 clearance-aware waypoint snapping：

- 先计算 room 内部 clearance field。
- 对每个理想 waypoint center，在 room 内 traversable cells 中搜索最近且 clearance 更好的 cell。
- object circle 使用更大的 clearance target，让路线更愿意离 object 和边界远一些。
- room circle 使用 room-scale clearance target，让路线不要太靠墙但仍保持 room-level loop。

相关超参数：

```text
CIRCLE_OBJECT_WAYPOINT_CLEARANCE_CELLS = 16.0
CIRCLE_OBJECT_WAYPOINT_CLEARANCE_WEIGHT = 8.0
CIRCLE_ROOM_WAYPOINT_CLEARANCE_CELLS = 10.0
CIRCLE_ROOM_WAYPOINT_CLEARANCE_WEIGHT = 16.0
CIRCLE_WAYPOINT_SEARCH_RADIUS_CELLS = 32
```

### 5.9 Constraint grounding

Grounder 支持以下 constraint kind：

```text
require_visit
require_visit_in_order
forbid
prefer_near
prefer_far
prefer_relative
```

Hard constraints：

- `require_visit(expr)`：要求 planner 经过 expr 对应 region。
- `require_visit_in_order(exprs...)`：要求 planner 按顺序经过多个 region。
- `forbid(expr)`：把 expr 对应 region 设为 forbidden mask。

Soft constraints：

- `prefer_near(ref)`：路径倾向靠近 ref。
- `prefer_far(ref)`：路径倾向远离 ref。
- `prefer_relative(first, relation, second)`：路径倾向满足 first/second 的相对距离关系。

现在有一个重要修复：如果 `prefer_near` 或 `prefer_far` 的 expr 返回多个 refs，会被 grounder 拆成 N 个 independent `GroundedConstraint`。例如：

```text
prefer_far(entities("painting"))
```

如果当前 room/scene 有 5 个 painting，会变成 5 个独立的 `prefer_far`，每个 preference 只有一个 object。这样代码表面上支持“一条 preference 对多个 object”，实际 planner 收到的是 N 条单 object preference。

### 5.10 Soft spatial scope

Soft preference 有 spatial scope 概念，用于控制 cost field 只在哪些区域生效。

当前规则：

- 如果 DSL/JSON 显式写了 `within(...)` 或 `spatial_scope`，grounder 会把它 grounded 成 room/region refs。
- 如果没有显式 spatial scope，且 soft preference 的 reference 是 entity，则默认 scope 是该 entity 所在 room。
- 因此普通 object-level near/far preference 的 cost field 不会扩散到其他房间。
- `prefer_relative` 要求两个 reference 在同一个 room，否则会报 `INVALID_SCOPE`；默认 spatial scope 是二者所在 room。

这是为了避免一个 object 的 soft field 在全局传播，导致 unrelated room 里的路径也被它影响。

## 6. Planner

### 6.1 Planner 的职责

Planner 在 `planner/semantic_astar.py`，输入是：

- `SceneMap`
- `GroundedProgram`

输出是：

- `PlannedProgram`
- 最终 `trajectory`

Planner 不再调用 LLM。它只用 grounded refs、hard constraints、soft constraints 和 map traversability 做 deterministic search。

### 6.2 Segment-by-segment planning

`plan_grounded_program()` 会从 `scene.start` 开始，按 `grounded.segments` 顺序规划：

```text
current = scene.start
for segment in grounded.segments:
    planned = plan_segment(scene, segment, current)
    append planned path to full trajectory
    current = planned.endpoint
```

如果某个 segment 失败，planner 会停止，返回当前已经规划出的 partial trajectory，并把失败原因写进 `PlannedProgram`。

### 6.3 Goal mask

每个 segment 的 target 会先转成 `goal_mask`：

- `position`：目标 cell。
- `room`：room cells 中可通行部分。
- `region`：region cells 中可通行部分。
- `entity`：object 周围半径内的可通行 cells。

Entity target 的半径由：

```text
OBJECT_GOAL_RADIUS_METERS = 0.30
```

转成 grid radius。目标 object 所在 room 会作为限制，避免 planner 为了接近 object 走到其他房间里贴近墙另一侧。

如果 entity/near-region 的 goal mask 为空，planner 会尝试找同 room 内离 reference center 最近的 traversable fallback cell。

### 6.4 Hard constraints

Planner 用 `_compile_hard()` 把 hard constraints 编译成：

- `forbidden` mask
- ordered `required` masks
- `required_smoothness_masks`
- debug details

`forbid` 会从 traversable mask 中扣掉 forbidden cells。

`require_visit` 和 `require_visit_in_order` 会变成必须经过的 stage targets。Planner 不再用一个 A* state 同时追踪 required index，而是当前主要走 `_plan_via_required_regions()`：

```text
start -> required_1 -> required_2 -> ... -> required_N -> final_goal
```

每一段 stage 都单独调用 cell-only A*。这样 path-shape/circle waypoint 的顺序更可控，也更容易在每个 stage 上记录 debug 信息。

### 6.5 A* state 和动作

当前 A* 使用 8-connected grid：

```text
up, down, left, right cost = 1
diagonal cost = sqrt(2)
```

每一步会检查：

- 是否在 grid 内。
- 是否 traversable。
- 是否被 hard forbidden mask 禁止。
- diagonal/neighbor transition 是否满足 `can_traverse_between(...)`，避免穿墙或斜穿障碍。

A* priority：

```text
f = g + heuristic
```

其中 heuristic 是当前位置到 goal mask 的 distance transform。每个 transition 的 g cost：

```text
step_cost + soft_field[next_cell] + smoothness_penalty
```

搜索上限由：

```text
MAX_EXPANSIONS = 250000
```

控制。

### 6.6 Soft cost fields

Planner 用 `_compile_soft()` 把 grounded soft constraints 编译成一个 additive cost field。

#### prefer_near

`prefer_near(ref)` 的 cost：

```text
distance_to_ref / near_radius
clip to [0, 1]
* WEIGHT_NEAR
* scope_mask
```

含义是：越远离 ref，cost 越高；超过 near radius 后达到最大惩罚。

相关参数：

```text
NEAR_RADIUS_METERS = 1.50
WEIGHT_NEAR = 8.00
```

#### prefer_far

`prefer_far(ref)` 的 cost：

```text
exp(-distance_to_ref / sigma)
* WEIGHT_FAR
* scope_mask
```

含义是：离 ref 越近，cost 越高；远离后指数衰减。

相关参数：

```text
FAR_SIGMA_METERS = 1.50
WEIGHT_FAR = 96.00
```

因为 grounder 已经把 multi-ref preference 拆成单 ref preference，所以多个 far preference 是独立 field 相加。默认情况下，object-level preference 的 scope 是该 object 所在 room，因此这些 field 不会无条件扩散到全局。

#### prefer_relative

`prefer_relative(first, relation, second)` 当前支持：

- `closer_to`
- `farther_from`

Planner 会计算当前位置到 `first` 和 `second` 的 distance field，然后根据 relation 计算 violation。

例如 `closer_to(first, second)` 的 violation 类似：

```text
max(0, distance_to_first - distance_to_second)
```

也就是当前位置如果比起 first 更靠近 second，就会被惩罚。

相关参数：

```text
RELATIVE_RADIUS_METERS = 1.50
WEIGHT_RELATIVE = 640.00
```

#### clearance

Planner 总是加入 clearance cost：

```text
clearance_field * WEIGHT_CLEARANCE
```

clearance field 来自 obstacle distance。离障碍物越近，惩罚越高；超出 `CLEARANCE_RADIUS_METERS` 后惩罚趋近于 0。

相关参数：

```text
CLEARANCE_RADIUS_METERS = 0.75
WEIGHT_CLEARANCE = 36.00
```

这个 cost 是 whole-segment scope，不依赖 instruction 中是否出现 soft preference。

### 6.7 Path-shape/circle planning

Path-shape-like preference 当前主要通过 `circle(...)` 表达。

Parser 会把 “circle around X / go around X / make a loop around X” 转成：

```text
require_visit_in_order(circle(target, fraction=1.0, direction=...))
```

Grounder 把 `circle(...)` 转成一串 ordered waypoint regions。

Planner 再按：

```text
start -> waypoint_1 -> waypoint_2 -> ... -> waypoint_N -> final target
```

分 stage 规划。这样算法不是直接追踪 GT trajectory，而是强制路线具备“绕一圈”的拓扑形状。

当前 circle waypoint 有两个重要改进：

1. Clearance-aware snapping：waypoint 不再简单贴着 reference 或 room 边缘，而是会偏向更可通行、更有 clearance 的位置。
2. Scoped smoothness：在 path-shape/circle region 内额外惩罚急转弯，减少局部锯齿。

### 6.8 Smoothness penalty

Smoothness penalty 当前只在 path-shape/circle 的 required stage 中生效，不做全局惩罚。

原因是：

- 全局 smoothness 可能伤害普通最短路径或绕障行为。
- path-shape preference 的主要问题通常出现在 waypoint 区域内路线太贴边、局部折返、形状不自然。
- 所以只在 circle/path-shape scope 里加 smoothness 更保守。

实现上，planner 在 `_compile_hard()` 中给 `circle_waypoint` 构造 `required_smoothness_masks`。A* transition 时如果当前 cell 或 next cell 在 smoothness mask 内，就加入：

```text
WEIGHT_SMOOTHNESS * turn_smoothness
```

其中：

```text
turn_smoothness = (turn_angle / pi) ** 2
```

如果方向不变，smoothness cost 是 0；转弯越大，惩罚越大。

当前参数：

```text
WEIGHT_SMOOTHNESS = 0.02
```

这个值很小，目标是去掉局部 zig-zag，而不是压过 semantic cost。

### 6.9 Failure behavior

Planner 可能返回的失败包括：

- `START_UNRESOLVED`
- `EMPTY_TARGET_REGION`
- `START_IN_FORBIDDEN_REGION`
- `GOAL_BLOCKED_BY_HARD_CONSTRAINT`
- `NO_FEASIBLE_PATH`

如果 planner 失败，pipeline 会返回 start-only trajectory，并在 metadata 中记录 failure status。因为 evaluator 现在对非空 trajectory 使用最小有效长度 1 grid，所以 start-only 不会造成 PLR 为 0。

### 6.10 Debug details

Planner 会把大量细节写入每个 `PlannedSegment.details`：

- target type
- target id/category
- goal radius/cell count
- hard constraints details
- soft constraints details
- required region count
- ordered stage planning details
- smoothness weight/scope

对于 path-shape/circle，stage details 里会包含：

- stage name
- status
- expanded states
- target cell count
- endpoint
- waypoint count
- smoothness active
- smoothness scope cell count

这些信息会进入 `instruction_xxxxxx.steps.json`，方便定位到底是 parser 没表达对、grounding 选错对象，还是 planner 没走出符合 soft/hard constraints 的路线。

### 6.11 当前实现边界

当前 Grounding2Route 已经把 parser、grounding 和 planner 明确拆开，但仍有几个重要边界：

- Grounding ranking 的 `geodesic` 目前实际还是 center-based euclidean distance，没有真正使用 room topology 或 traversable distance。
- Path-shape/circle 是 waypoint approximation，不是直接优化对 GT path 的 shape similarity。
- Soft preference 是 additive cost field，多个 preference 之间没有更高层的冲突协调器。
- LLM parser 的语义能力仍是主要瓶颈；JSON intent 降低了格式错误，但复杂指代、隐含比较和 scene-dependent disambiguation 仍可能失败。
- Grounding repair 可以修 parser 表达，但不会改变下游 planner 的策略。

因此，当前方法最值得继续做的大改方向不是继续微调权重，而是：

- 更强的 structured intent representation。
- 更强的 object/room relational grounding。
- 真正的 geodesic/topological grounding selector。
- 对 path-shape preference 使用更直接的 shape-aware planning objective。
