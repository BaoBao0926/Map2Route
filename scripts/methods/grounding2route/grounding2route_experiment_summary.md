# Grounding2Route 实验总结

这个文件整理 Grounding2Route 四次主要实验的变化。核心目标是说明：每一次 run 主要验证或修复了什么，以及最新的 intent parser 为什么有提升。

## 四次 Run 的定位

| Run | 结果目录 | 主要变化 | 简要结论 |
| --- | --- | --- | --- |
| 1st run | `resources/methods/grounding2route/previous/grounding2route_previous/1st_run` | 主要验证 planner 链路 | Planner 基本可以工作，能把 grounded task 转成路径；主要瓶颈开始暴露在 parser/grounding。 |
| 2nd run | `resources/methods/grounding2route/previous/grounding2route_previous/2nd_run` | 修复 parser 和 grounding 的基础问题 | Parser/grounding 基本连通，成功率明显提升，但复杂语义仍然容易解析错或 grounding 失败。 |
| 3rd run | `resources/methods/grounding2route/previous/grounding2route_previous/Grounding2Route` | 继续修复 grounding、cost field、path-shape、PLR/evaluation 记录 | 整体分数继续提升，尤其 PLR 很高；剩余问题主要是复杂目标选择和 path-shape。 |
| 4th run: Intent Mixed | `resources/methods/grounding2route/previous/grounding2route_previous/Grounding2Route_intent` | 初始 parser 输出 typed JSON intent graph，但 grounding repair 仍可能回到 DSL repair | HCS 和 OSS 明显提升，但这个实验不是严格纯 JSON，因为 repair 阶段混入了 DSL。 |
| 5th run: Pure JSON Intent | `resources/methods/grounding2route/previous/Grounding2Route_intent_pure` | 初始 parser 和 grounding repair 都输出 typed JSON intent graph；同时加入 `count_next_to/count_near` | 当前最好结果。说明提升不是因为 fallback 到 DSL，而是 JSON intent graph 和更明确的 deterministic operator 更稳。 |

## 指标对比

| Method | Overall N | PLR | HCS | SCS | OSS | Easy N | Easy OSS | Hard N | Hard OSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Grounding2Route-1st | 25 | 0.581 | 0.300 | 0.607 | 0.201 | 17 | 0.179 | 8 | 0.247 |
| Grounding2Route-2nd | 50 | 0.635 | 0.490 | 0.646 | 0.338 | 34 | 0.367 | 16 | 0.276 |
| Grounding2Route-3rd | 50 | 0.910 | 0.567 | 0.723 | 0.417 | 35 | 0.481 | 15 | 0.266 |
| Grounding2Route-Intent-Mixed | 50 | 0.862 | 0.683 | 0.780 | 0.492 | 35 | 0.533 | 15 | 0.396 |
| Grounding2Route-Intent-PureJSON | 50 | 0.863 | 0.723 | 0.783 | 0.519 | 35 | 0.557 | 15 | 0.432 |

## 当前判断

第三次 run 的主要收益来自 planner/cost/evaluation 侧的修复，特别是 PLR 已经很高，说明路径生成本身不是最大瓶颈。

第四次 mixed intent run 的主要收益来自 parser 输出格式变化，但这个实验有一个问题：它的初始解析是 JSON intent，grounding repair 却仍然使用原来的 DSL repair。因此它不能严格证明提升全部来自 JSON 格式。

第五次 pure JSON intent run 修复了这个问题：

```text
instruction
  -> LLM 输出 typed JSON intent graph
  -> 代码编译成现有 GPProgramSpec / DSL source
  -> grounding 失败时继续 repair JSON intent graph
  -> 原来的 grounding
  -> 原来的 planner
```

Grounding 和 planner 的核心逻辑没有换，所以第五次 run 是更干净的消融。HCS 从第三次的 `0.567` 提升到 `0.723`，OSS 从 `0.417` 提升到 `0.519`。这说明提升不是因为 JSON 失败后 fallback 到 DSL，而是因为 typed JSON intent graph 本身更稳定。

第五次还加入了两个 deterministic count operator：

```text
count_next_to(category, reference)
count_near(category, reference)
```

这解决了一个很典型的问题：`the dining table with exactly two chairs`。以前用嵌套 `where(..., self)` 容易让内层 `self` 覆盖外层 `self`，现在可以直接写成 `count_next_to("chair", self) == 2`。

## Intent JSON 长什么样

Intent parser 不再要求 LLM 直接写 DSL，而是要求它输出一个 JSON object，主要由两部分组成：

- `bindings`: 给对象、房间、区域、引用关系命名。
- `segments`: 表示导航顺序，每个 segment 有 `from`、`to` 和 constraints。

一个简化例子：

```json
{
  "bindings": [
    {
      "name": "start_room",
      "expr": {
        "op": "room_of",
        "args": ["start_position"]
      }
    },
    {
      "name": "target_table",
      "expr": {
        "op": "kth_nearest",
        "args": [
          {
            "op": "where",
            "args": [
              {"op": "entities", "args": ["dining_table"]},
              {
                "op": "compare",
                "args": [
                  {"op": "count_next_to", "args": ["chair", "self"]},
                  "==",
                  2
                ]
              }
            ]
          },
          "start_position"
        ],
        "kwargs": {
          "k": 1,
          "metric": "geodesic"
        }
      }
    },
    {
      "name": "cart",
      "expr": {
        "op": "kth_nearest",
        "args": [
          {"op": "entities", "args": ["cart"]},
          "start_position"
        ],
        "kwargs": {
          "k": 1,
          "metric": "geodesic"
        }
      }
    }
  ],
  "segments": [
    {
      "id": "s1",
      "from": "start_position",
      "to": "target_table",
      "constraints": [
        {
          "kind": "prefer_near",
          "args": ["target_table"],
          "spatial_scope": "start_room"
        },
        {
          "kind": "prefer_far",
          "args": ["cart"],
          "spatial_scope": {
            "op": "rooms",
            "args": ["living_room"]
          }
        }
      ]
    }
  ]
}
```

这个 JSON 会被代码编译成原来的 `GPProgramSpec`，再交给旧的 grounder/planner。完整真实例子可以看：

```text
resources/methods/grounding2route/previous/grounding2route_previous/Grounding2Route_intent/procthor/003_valunseen/instruction_000002.steps.json
```

历史 intent run 里的路径是：

```text
steps.parse.metadata.raw_intent
steps.parse.metadata.compiled_dsl
```

当前 API-program parser run 里的对应路径是：

```text
steps.parse.metadata.raw_api_program
<instruction>.artifacts/parser_api.py
```

## 为什么 JSON 比 DSL 稳

直接让 LLM 输出 DSL 时，很容易出现：

- `let` 放在 `task` 外面；
- segment 缺 `from`、`to` 或 `go_to`；
- `where/count/in/exclude` 写法不稳定；
- pronoun/reference 被写成无法 grounding 的字符串；
- repair 时重写整段 DSL，导致新错误。

JSON intent graph 的好处是：

- 每个 binding 都有固定 schema；
- 每个 segment 都必须有 `from` 和 `to`；
- 指代可以通过 binding name 表示，例如 `target_table`、`same_room`、`previous_target`；
- 代码可以局部检查和编译；
- grounding repair 也可以继续修 JSON intent graph，而不是退回 free-form DSL；
- grounding/planner 可以完全复用。

## 下一步

Intent parser 已经证明比直接 DSL 更稳，但它仍然依赖 LLM 做具体场景推理。下一步更好的方向是：

1. LLM 输出 intent graph。
2. 代码枚举所有 candidate object/room。
3. 代码执行 `exactly N`、`without X`、`contains X`、`same room`、`other` 等 filter。
4. 如果还有多个候选，再让 LLM 在小候选集合里消歧。

这样可以继续保留 JSON intent 的结构稳定性，同时把 LLM 最不擅长的精确场景计算交给 deterministic code。
