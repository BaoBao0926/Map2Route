# SemPathBench Benchmark-Native 方法设计

## 1. 总介绍

本文档定义一种面向 SemPathBench 的 benchmark-native 方法。该方法把自然语言导航指令转换为一条轨迹，轨迹运行在完整语义地图之上；语义地图包含房间、物体、占据栅格以及空间关系。

该方法包含三个概念模块：

1. **Parser:** 自然语言指令 → 未 grounding 的任务 DSL；
2. **Grounder:** 未 grounding 的 DSL 表达式 → 具体物体、房间和区域；
3. **Planner:** 已 grounding 的 segment → 满足硬约束、并考虑软偏好的路径。

```text
Natural-language instruction
        ↓
LLM parser
        ↓
Ungrounded task DSL / AST
        ↓
Restricted deterministic grounder
        ↓
Grounded Task IR
        ↓
Constraint-aware cost-based planner
        ↓
Final trajectory
```

概念上，这三个阶段相互独立，运行时也保持清晰的 `Parser → Grounder → Planner` 流程。Parser 先解析完整指令，Grounder 再一次性解析完整任务中的所有 symbolic reference，最后 Planner 按 segment 顺序生成连续路径：

```text
Parse the complete instruction once
        ↓
Ground the complete task program
        ↓
Plan segment s1 → plan segment s2 → ...
        ↓
Concatenate segment paths
```

对于 “go to A, then go to B” 这样的顺序指令，Grounder 不需要等待 Planner 产生第一段的实际 grid endpoint。它可以把第一段的 grounded target `A` 作为第二段的 semantic start reference，用于解析 “the nearest object from the second segment's starting point” 等表达式。Planner 阶段再使用第一段实际规划出的 endpoint 作为第二段的物理起点，从而保证最终轨迹连续。

### 1.1 设计原则

- 语言解释、场景 grounding、路径优化分别拥有独立接口。
- LLM 永远不预测 object ID、room ID、坐标或路径。
- 所有依赖具体场景的引用都通过人工定义的 map API 解析。
- 硬约束定义可行性，不能用很大的软惩罚替代。
- 软约束定义优化代价，不能被静默升级为必须满足的约束。
- 每个约束都属于明确的 segment，并具有显式 scope。
- 所有中间表示都应有类型、确定性、可检查、可序列化。
- 每个 DSL operator 都必须可 grounding，并被 planner 或 evaluator 支持。

### 1.2 场景表示

三个模块共享同一个语义地图抽象：

```python
SceneMap:
    occupancy_map
    traversable_map
    room_label_map
    object_label_map
    rooms
    objects
    doorways
    passages
    object_to_room
    object_footprints
    derived_area_masks
    map_resolution
```

Parser 不能访问 `SceneMap`。Grounder 只能通过受限 API 查询它。Planner 在所有相关引用完成 grounding 后，访问其中的几何字段。

---

## 2. Parser

### 2.1 总览

SemPathBench Parser 将自然语言导航指令转换为与具体场景无关、具有静态类型的任务程序。Parser 的输出是一个 **未 Grounding 的 DSL**：它保留指令的语义结构，但不包含具体场景中的 entity ID、room ID、坐标、目标栅格、cost map 或路径。

```text
自然语言指令
    ↓
LLM Parser
    ↓
Ungrounded DSL
    ↓
Grammar 与类型检查
    ↓
Typed AST
```

Parser 一次性解析完整指令，恢复有序导航 segment、符号化 entity 描述、跨 segment 引用、硬路径约束、软路径偏好及其作用域。随后，确定性的 Grounder 使用注册过的 map API 执行 typed AST。

Parser 永远不访问场景地图，也不决定某个表达式最终对应哪个具体 object 或 room。

---

### 2.2 Parser 的职责

#### 2.2.1 Parser 需要恢复的内容

Parser 必须恢复：

- 有序导航 segment；
- 每个 segment 的一个主要目的地；
- 符号化的 entity、room、position 和 region reference；
- entity filter、relation、count、ranking 和 set operation；
- task start 和先前 segment target 的引用；
- 必经区域和禁行区域；
- near、far、relative 和 path-shape preference；
- segment scope 和 spatial scope；
- 指令明确指定的 distance metric。

#### 2.2.2 Parser 禁止执行的操作

Parser 不得：

- 访问 occupancy map、semantic map、room map 或 topology map；
- 生成 object ID、room ID、坐标或 grid cell；
- 判断某个具体场景中的 candidate 数量；
- 执行 grounding query；
- 生成路径或选择物理 endpoint；
- 构造 cost field 或 planner weight；
- 使用未注册的 operator；
- 把显式 object-object relation 改写成 agent-object relation；
- 静默添加、删除、加强、弱化或重排指令要求。

---

### 2.3 为什么使用未 Grounding 的 DSL

SemPathBench 指令本质上是组合式程序，而不是若干彼此独立的固定 slot。一条指令可能同时包含：

- 多个有序目的地；
- 依赖先前目标的后续目标；
- 根据物体是否存在或数量筛选房间；
- 基于距离或属性的 ordinal selection；
- 必须经过或禁止经过的区域；
- 局部或跨 segment 的软偏好；
- 完整绕圈或部分绕圈要求。

固定的扁平 JSON schema 会逐渐产生大量 instruction-specific field 和深层嵌套。受限 DSL 则提供：

- 可复用且有类型的 operator；
- 显式 symbolic variable；
- 清晰的依赖结构；
- segment-local constraint；
- 可组合的场景查询；
- 可确定性转换的 typed AST；
- 基于封闭 operator registry 的自动验证。

DSL 应优先组合通用 primitive，避免为单条 instruction 创建专用 operator。

不允许：

```text
find_bed_near_window_in_three_bed_room()
```

推荐：

```text
let candidate_rooms = where(
    rooms("bedroom"),
    count(in(entities("bed"), self)) == 3
)
let target_room = unique(candidate_rooms)
let target_bed = unique(
    where(
        in(entities("bed"), target_room),
        near_to(in(entities("window"), target_room))
    )
)
```

禁止使用 `find_entity("the desired object")` 或 `apply_constraint("follow the instruction")` 等自由文本占位 operator。

---

### 2.4 DSL 程序结构

#### 2.4.1 标准结构

```text
task {
    let task_start = start_position

    let target_1 = ...
    let reference_1 = ...

    segment s1 {
        from task_start
        to target_1

        go_to(target_1)

        prefer_near(reference_1)
            on_segments({s1})
            within_rooms({room_of(reference_1)})
    }

    let target_2 = ...

    segment s2 {
        from target_1
        to target_2

        go_to(target_2)
    }
}
```

#### 2.4.2 结构规则

1. 所有 `let` 声明必须位于 segment block 外部。
2. 所有变量必须先定义后使用。
3. 第一个 segment 通常从 `task_start` 开始。
4. 后续 segment 通常使用前一 segment 的 target 作为 semantic start reference。
5. 一个新的目的地通常创建一个新的 segment。
6. Parser 必须保留目的地和约束的时间顺序。
7. Constraint 必须属于自然语言中描述它的 segment。
8. 如果合并事件会改变 dependency 或 scope，则不得合并。
9. 每个 segment 只能包含一个主要 `go_to` action。
10. `from` 是 Grounding 阶段使用的 semantic reference；Planner 使用上一段的实际 endpoint 作为下一段的物理起点。

#### 2.4.3 三角形路径

由三个 landmark 构成的三角形不需要特殊 path-shape operator，而应表示为三个连续目的地。

指令：

```text
以 A、B 和 C 为三个顶点走出一个三角形。
```

DSL：

```text
segment s1 {
    from task_start
    to A
    go_to(A)
}

segment s2 {
    from A
    to B
    go_to(B)
}

segment s3 {
    from B
    to C
    go_to(C)
}
```

如果指令明确要求闭合三角形，则增加第四个 segment，从 `C` 返回 `A`。

---

### 2.5 类型系统

#### 2.5.1 核心类型

```text
Position
Entity
EntitySet
Room
RoomSet
Region
RegionSet
SpatialReference
OrderedSet[T]
OrderedRegionList
Segment
SegmentSet
PathReference
Action
HardConstraint
SoftConstraint
PathShapeConstraint
SegmentScope
SpatialScope
Predicate
Direction
DistanceMetric
Attribute
Integer
Float
Boolean
```

`SpatialReference` 是以下类型的并集：

```text
SpatialReference := Position | Entity | Room | Region
```

Doorway、doorframe、wall、furniture 和 appliance 均表示为 `Entity`，通过 category 区分，不再设计独立 constructor。

Room 保留为独立类型，因为 room 同时是语义单元、topology node、container 和 spatial scope 单元。

#### 2.5.2 Collection 规则

- Set 不能直接作为单个 navigation target。
- 指令要求唯一 candidate 时使用 `unique(set)`。
- 指令明确允许任意 candidate 时使用 `choose_any(set)`。
- Ranking operator 返回单个 element。
- `order_by_distance` 返回有序集合。
- 空集合和错误的唯一性假设由 Grounder 报错，Parser 不得自行 fallback。

---

### 2.6 正式 Operator Registry

Parser 只能使用本节注册的 operator。

#### 2.6.1 Task Structure 与基础引用

| Operator | 签名 | 含义 |
|---|---|---|
| `task { ... }` | declarations × segments → `TaskProgram` | 完整任务程序 |
| `segment s_k { ... }` | start × target × constraints → `Segment` | 一个有序导航段 |
| `from reference` | `SpatialReference` | Segment 的 semantic start |
| `to target` | `SpatialReference` | Segment 的目的地 |
| `let x = expression` | `T → T` | Typed symbolic binding |
| `start_position` | `→ Position` | Task 初始位置 |
| `target_of(segment)` | `Segment → SpatialReference` | 引用先前 segment 的 target |
| `path_of(segment)` | `Segment → PathReference` | 引用先前 segment 的路径；为未来扩展保留 |

#### 2.6.2 Entity、Room、Containment 与 Topology

| Operator | 签名 | 含义 |
|---|---|---|
| `entities(category)` | `EntityCategory → EntitySet` | 获取指定 category 的所有语义实体 |
| `rooms(category?)` | `Optional[RoomCategory] → RoomSet` | 获取全部房间或指定类别房间 |
| `in(entity_set, room_or_rooms)` | `EntitySet × (Room | RoomSet) → EntitySet` | 获取指定房间中的实体 |
| `room_of(reference)` | `SpatialReference → Room` | 获取 position、entity 或 region 所在房间 |
| `contains(room, category)` | `Room × EntityCategory → Boolean` | 判断房间是否包含某类实体 |
| `adjacent_rooms(room)` | `Room → RoomSet` | 通过 topology map 查询相邻房间 |
| `passage_regions(room_a, room_b)` | `Room × Room → RegionSet` | 查询两个相邻房间之间的所有可通行区域 |

#### 2.6.3 Set Operation 与 Predicate Logic

| Operator | 签名 | 含义 |
|---|---|---|
| `union(A, B, ...)` | compatible sets → set | 并集 |
| `intersection(A, B, ...)` | compatible sets → set | 交集 |
| `exclude(set, elements)` | set × element/set → set | 排除指定候选 |
| `count(set)` | set → `Integer` | 集合大小 |
| `where(set, predicate)` | set × `Predicate` → set | 按条件过滤候选 |
| `not(predicate)` | `Predicate → Predicate` | 逻辑非 |
| `and(p1, p2, ...)` | predicates → `Predicate` | 逻辑与 |
| `or(p1, p2, ...)` | predicates → `Predicate` | 逻辑或 |
| `unique(set)` | set → element | 要求候选唯一 |
| `choose_any(set, reference=segment_start)` | set × optional reference → element | 从合法候选中确定性地选择 geodesic distance 最近者 |

Grammar 在 predicate 中提供基础比较符：

```text
==  !=  >  >=  <  <=
```

这些比较符属于 DSL grammar，不作为独立 semantic operator 注册。

#### 2.6.4 Relational Predicate

| Operator | 签名 | 含义 |
|---|---|---|
| `near_to(reference)` | `SpatialReference → Predicate` | Candidate 靠近 reference |
| `far_from(reference)` | `SpatialReference → Predicate` | Candidate 远离 reference |
| `next_to(reference)` | `SpatialReference → Predicate` | Candidate 与 reference 相邻 |
| `on_top_of(reference)` | `SpatialReference → Predicate` | Candidate 位于 reference 上方 |
| `in_corner(room?)` | `Optional[Room] → Predicate` | Candidate 位于房间角落 |
| `between(A, B)` | `SpatialReference × SpatialReference → Predicate` | Candidate 位于两个 reference 之间 |

这些 predicate 用于 entity grounding，不是 soft path preference：

```text
where(entities("bed"), near_to(window))  # 选择目标 bed
prefer_near(window)                       # 优化路径
```

#### 2.6.5 Selection 与 Ranking

| Operator | 签名 | 含义 |
|---|---|---|
| `kth_nearest(set, reference, k, metric=geodesic)` | set × reference × integer × metric → element | 第 k 近的 candidate |
| `kth_farthest(set, reference, k, metric=geodesic)` | set × reference × integer × metric → element | 第 k 远的 candidate |
| `kth_largest(set, attribute, k)` | set × attribute × integer → element | 按属性选择第 k 大的 candidate |
| `kth_smallest(set, attribute, k)` | set × attribute × integer → element | 按属性选择第 k 小的 candidate |
| `order_by_distance(set, reference, order, metric=geodesic)` | set × reference × order × metric → `OrderedSet` | 相对于固定 reference 对 object 或 room 排序 |
| `closest_pair_member(candidates, references, metric=euclidean)` | set × set × metric → element | 返回两组实体最近配对中 candidates 一侧的成员 |

`k=1` 分别表示最近、最远、最大或最小。DSL 不再注册 `nearest`、`second_nearest`、`largest` 和 `smallest` 等重复 operator。

#### 2.6.6 Derived Region

| Operator | 签名 | 含义 |
|---|---|---|
| `region_of(entity)` | `Entity → Region` | 获取 entity 对应的空间范围 |
| `room_region(room)` | `Room → Region` | 获取完整房间区域 |
| `midpoint_region(A, B)` | reference × reference → `Region` | 两个 reference 的中点区域 |
| `between_region(A, B)` | reference × reference → `Region` | 两个 reference 之间的可通行区域 |
| `near_region(reference, radius)` | reference × distance → `Region` | Reference 周围指定距离内的区域 |
| `side_region(center, side_reference)` | reference × reference → `Region` | Center 朝向指定 reference 的一侧区域 |
| `boundary_region(room)` | `Room → Region` | 房间边界附近区域 |
| `half_room(room, reference)` | room × reference → `Region` | 房间朝向 reference 的半侧区域 |

对于普通 object destination，Parser 必须使用：

```text
go_to(target_object)
```

不得自行构造 `near_region(target_object, 0.3 m)`。Planner 会根据 object footprint、object 所在房间、traversability 和 0.3 m 阈值自动构造 benchmark 定义的合法目标区域。

#### 2.6.7 Action 与 Hard Constraint

| Operator | 签名 | 含义 |
|---|---|---|
| `go_to(target)` | `(Position | Entity | Room | Region) → Action` | 唯一基础 navigation action |
| `require_visit(region)` | `Region → HardConstraint` | 路径必须进入指定区域 |
| `require_visit_in_order(regions)` | `OrderedRegionList → HardConstraint` | 路径必须按顺序进入指定区域 |
| `forbid(region)` | `Region → HardConstraint` | 路径不得进入指定区域 |

自然语言中的 return、enter、exit 和 pass through 不自动创建独立 action：

- `return to A`：创建新 segment，并使用 `go_to(A)`；
- `enter room A and go to B`：通常使用 `go_to(B)`，room A 用于确定或限制 B；
- `pass through A and B before reaching C`：当穿越顺序是硬要求时，使用 `go_to(C)` 和 `require_visit_in_order([room_region(A), room_region(B)])`；
- `exit room A while staying far from C`：使用 room A 作为 `prefer_far(C)` 的 spatial scope。

#### 2.6.8 Soft Constraint 与 Path-shape Preference

| Operator | 签名 | 含义 |
|---|---|---|
| `prefer_near(reference)` | `(Entity | EntitySet | Region) → SoftConstraint` | 鼓励路径靠近 reference |
| `prefer_far(reference)` | `(Entity | EntitySet | Region) → SoftConstraint` | 鼓励路径远离 reference |
| `prefer_relative(A, relation, B)` | references × relation → `SoftConstraint` | 鼓励满足 `closer_to` 或 `farther_from` 关系 |
| `circle(target, fraction=1.0, direction=counterclockwise, start_toward=optional_reference)` | `(Entity | Room) × fraction × direction × optional reference → PathShapeConstraint` | 完整/部分绕圈、沿房间边界或房间内部 loop 移动；`start_toward` 用来表达“先朝某物方向开始绕” |

`relation` 只允许：

```text
closer_to
farther_from
```

Clearance 不进入 Parser DSL。它是 Planner 对每个 segment 自动应用的固定全局 cost。

#### 2.6.9 Scope

Scope 分为两个相互独立的维度。

| 维度 | Operator | 签名 | 含义 |
|---|---|---|---|
| Segment | `on_segments(segments)` | segment/segment set → `SegmentScope` | 约束作用于指定 segment |
| Segment | `all_segments` | `→ SegmentScope` | 约束作用于所有 segment |
| Spatial | `within(region)` | region → `SpatialScope` | 约束在指定 region 内生效 |
| Spatial | `within_rooms(rooms)` | room/room set → `SpatialScope` | 约束在一个或多个 room 内生效 |
| Spatial | `whole_scene` | `→ SpatialScope` | 约束在整个 scene 生效 |

默认规则：

1. 除非语言明确扩展，constraint 只属于描述它的那个 segment。
2. Object-related near、far 或 relative preference 默认作用于 reference object 所在房间。
3. Relative preference 引用不同房间的 object 且 instruction 未指定 scope 时，Grounder 返回 invalid/ambiguous scope。
4. 当前基础实现把 entering/leaving scope 归一化为对应房间的 spatial scope。
5. 显式 scope 覆盖默认规则。

#### 2.6.10 Distance Metric

| Metric | 含义 |
|---|---|
| `euclidean` | 直线空间距离 |
| `geodesic` | Traversable map 上的可行走距离，可通过 A* 最短路径计算 |

默认规则：

- Entity 和 room selection 默认使用 `geodesic`；
- 静态几何关系和 derived region 可根据其注册定义使用 Euclidean boundary distance；
- 指令明确说 “straight-line distance” 时必须输出 `metric=euclidean`。

---

### 2.7 语义归一化规则

#### 2.7.1 Nearest、Farthest 与 Ordinal Selection

```text
nearest A          → kth_nearest(A, reference, k=1)
second-nearest A   → kth_nearest(A, reference, k=2)
farthest A         → kth_farthest(A, reference, k=1)
second-farthest A  → kth_farthest(A, reference, k=2)
largest A          → kth_largest(A, attribute=size, k=1)
smallest A         → kth_smallest(A, attribute=size, k=1)
```

未明确 reference 时，使用对应 segment 的 semantic start。

#### 2.7.2 “第几个遇到的物体”

Path-dependent encounter language 统一解释为相对于对应 segment start 的 geodesic distance ranking：

```text
the first box encountered
→ kth_nearest(entities("box"), segment_start, k=1, metric=geodesic)

the second dining table encountered
→ kth_nearest(entities("dining_table"), segment_start, k=2, metric=geodesic)
```

Parser 不预测未来路径实际遇到物体的先后顺序。

#### 2.7.3 任意合法 Candidate

指令明确允许任意 candidate 时使用 `choose_any`，而不是 `unique`：

```text
any painting in the kitchen
→ choose_any(
      in(entities("painting"), kitchen),
      reference=segment_start
  )
```

`choose_any` 不是随机选择，而是优先选择相对于指定 reference 的 geodesic distance 最近者。

#### 2.7.4 当前房间

不设置 `current_room` operator，通过组合得到：

```text
let current_room = room_of(segment_start)
```

第一个 segment 中可写为：

```text
let current_room = room_of(start_position)
```

#### 2.7.5 尚未访问的房间

“Unvisited”表示没有作为先前 segment 显式目标的房间，不包括生成路径偶然穿过的房间。

```text
let visited_rooms = {
    room_of(target_of(s1)),
    room_of(target_of(s2))
}

let unvisited_rooms = exclude(
    rooms(),
    visited_rooms
)
```

#### 2.7.6 Return

Return 不需要独立 action：

```text
return to the dining table near the starting point
```

创建新的 segment，并使用：

```text
go_to(reference_table)
```

#### 2.7.7 穿过房间

Room mention 可能具有不同作用。

仅用于目标消歧时，不需要创建 path event。例如目标已经被明确限定为某 bedroom 中的 sink。

如果穿越顺序是必须满足的要求：

```text
pass through living room A and bathroom B in sequence before reaching C
```

表示为：

```text
require_visit_in_order([
    room_region(living_room_A),
    room_region(bathroom_B)
])
```

#### 2.7.8 必经与禁行通道

“必须从 A 和 B 之间经过”：

```text
let required_passage = between_region(A, B)
require_visit(required_passage)
```

“禁止从 A 和 B 之间经过”：

```text
let forbidden_passage = between_region(A, B)
forbid(forbidden_passage)
```

“必须使用 passage P，不能使用其他入口”通过组合实现：

```text
let all_passages = passage_regions(room_a, room_b)
let selected_passage = ...
let other_passages = exclude(all_passages, selected_passage)

require_visit(selected_passage)
forbid(union(other_passages))
```

不需要 `require_only_passage` operator。

#### 2.7.9 Circle 与 Detour

```text
make one complete loop around A
→ circle(A, fraction=1.0)

go halfway around A
→ circle(A, fraction=0.5)

circle A toward C first
→ circle(
      A,
      fraction=1.0,
      start_toward=C
  )
```

绕房间表示沿房间 boundary 或房间内部 loop 行走。Detour-around language 在基础版本中近似为围绕 reference object 或 room 的部分 `circle`。
`circle(...)` 的 waypoint snapping 使用 clearance-aware selection：在理想几何 waypoint 附近选择
既接近目标形状、又尽量远离墙/障碍/房间边界的可通行 cell，避免 path-shape 被拉到过度贴边的轨迹。

---

### 2.8 特殊组合示例

#### 2.8.1 筛选不包含某物体的房间

```text
the sink in the bathroom without a laundry hamper
```

```text
let candidate_bathrooms = where(
    rooms("bathroom"),
    count(in(entities("laundry_hamper"), self)) == 0
)
let target_bathroom = unique(candidate_bathrooms)
let target_sink = unique(
    in(entities("sink"), target_bathroom)
)
```

#### 2.8.2 按物体数量筛选房间

```text
the bedroom containing three beds
```

```text
let target_room = unique(
    where(
        rooms("bedroom"),
        count(in(entities("bed"), self)) == 3
    )
)
```

在针对 room set 的 `where` predicate 内，`self` 表示当前正在测试的 room。

#### 2.8.3 保留 Object-object Distance

```text
the floor lamp closest to the dresser
```

```text
let target_lamp = kth_nearest(
    entities("floor_lamp"),
    target_dresser,
    k=1,
    metric=euclidean
)
```

不得把 reference 改成 agent 或 segment start。

#### 2.8.4 两个集合之间的最近配对

```text
the bed closest to a side table
```

```text
let target_bed = closest_pair_member(
    candidates=entities("bed"),
    references=entities("side_table"),
    metric=euclidean
)
```

#### 2.8.5 相邻房间中的目标

```text
the armchair in the adjacent bedroom
```

```text
let current_room = room_of(segment_start)
let adjacent_bedrooms = intersection(
    adjacent_rooms(current_room),
    rooms("bedroom")
)
let target_room = unique(adjacent_bedrooms)
let target_armchair = choose_any(
    in(entities("armchair"), target_room),
    reference=segment_start
)
```

#### 2.8.6 按距离访问整个集合

```text
visit all sinks from nearest to farthest
```

```text
let ordered_sinks = order_by_distance(
    entities("sink"),
    task_start,
    order=ascending,
    metric=geodesic
)
```

Parser 将 ordered set 展开成一系列 segment。排序使用固定 reference `task_start`，而不是到达每个目标后重新排序。

#### 2.8.7 跨 Segment 共享 Soft Preference

```text
Stay near the same dog bed during the second and third segments.
```

```text
prefer_near(reference_dog_bed)
    on_segments({s2, s3})
    within_rooms({room_of(reference_dog_bed)})
```

#### 2.8.8 Relative Preference

```text
stay closer to the dog bed than to the dining table
```

```text
prefer_relative(
    reference_dog_bed,
    relation=closer_to,
    reference_dining_table
)
```

这是 soft path preference，而不是用于选择 entity 的 ranking comparison。

#### 2.8.9 Midpoint Target

```text
go to the midpoint between the sofa and the TV
```

```text
let target_region = midpoint_region(target_sofa, target_tv)

segment s1 {
    from task_start
    to target_region
    go_to(target_region)
}
```

#### 2.8.10 指定唯一入口

```text
enter the kitchen through the door farther from the sofa and do not use the other door
```

```text
let start_room = room_of(task_start)
let kitchen = unique(rooms("kitchen"))
let entrances = passage_regions(start_room, kitchen)
let selected_entrance = kth_farthest(
    entrances,
    reference_sofa,
    k=1,
    metric=euclidean
)
let forbidden_entrances = exclude(
    entrances,
    selected_entrance
)

segment s1 {
    from task_start
    to target
    go_to(target)
    require_visit(selected_entrance)
    forbid(union(forbidden_entrances))
}
```

---

### 2.9 完整示例

指令：

> First, go to the sink in the bathroom without a laundry hamper. Along the way, stay close to the dining table nearest to the starting point. Then go to the bed farthest from the first target while staying far from the sofa in that bedroom.

DSL：

```text
task {
    let task_start = start_position

    let candidate_bathrooms = where(
        rooms("bathroom"),
        count(in(entities("laundry_hamper"), self)) == 0
    )
    let target_bathroom = unique(candidate_bathrooms)
    let target_sink = unique(
        in(entities("sink"), target_bathroom)
    )

    let reference_table = kth_nearest(
        entities("dining_table"),
        task_start,
        k=1,
        metric=geodesic
    )

    let target_bed = kth_farthest(
        entities("bed"),
        target_sink,
        k=1,
        metric=geodesic
    )
    let target_bedroom = room_of(target_bed)
    let reference_sofa = unique(
        in(entities("sofa"), target_bedroom)
    )

    segment s1 {
        from task_start
        to target_sink
        go_to(target_sink)

        prefer_near(reference_table)
            on_segments({s1})
            within_rooms({room_of(reference_table)})
    }

    segment s2 {
        from target_sink
        to target_bed
        go_to(target_bed)

        prefer_far(reference_sofa)
            on_segments({s2})
            within_rooms({target_bedroom})
    }
}
```

---

### 2.10 暂不支持或近似处理的语义

#### 2.10.1 暂不支持

- S-shaped path；
- U-shaped path；
- 精确 retrace 先前生成的路径；
- 显式禁止几何直线路径；
- 任意自由形状的 path drawing。

Parser 不得为这些要求发明新的 operator。根据 benchmark policy，可以返回 unsupported-semantics error，或者只忽略不支持的 clause，同时保留其余可支持要求。

#### 2.10.2 近似处理

- entering/leaving scope 近似为对应房间的 spatial scope；
- detour around entity 近似为 partial `circle`；
- follow room boundary 表示为 circle room；
- first/second encountered object 转换为相对 segment start 的 geodesic ranking；
- unvisited room 只排除作为先前 segment 显式目标的房间，不排除路径偶然经过的房间。

#### 2.10.3 Triangle

Triangle instruction 通过顺序分解支持：

```text
A → B → C
```

明确要求闭合时：

```text
A → B → C → A
```

不需要 triangle-specific path-shape operator。

---

### 2.11 Parser Validation

Parser 输出必须先解析成 AST 并通过以下验证，之后才能进入 Grounding。

#### 2.11.1 Syntax 与 Structure

- 只包含一个 `task { ... }` block；
- grammar 合法；
- segment ID 唯一且连续；
- 变量先定义后使用；
- segment 内不存在 `let`；
- 每个 segment 恰好包含一个 `go_to`；
- segment 顺序与原始 instruction 一致；
- 后续 segment 的 semantic start reference 合法。

#### 2.11.2 Type

- 每个 operator 都已注册；
- operator 参数类型符合签名；
- set-valued expression 不会被用作单个 target；
- ordered constraint 接收 ordered collection；
- segment scope 和 spatial scope 类型正确；
- distance/attribute selector 使用兼容的 metric 或 attribute。

#### 2.11.3 Semantic Restriction

- 不含 scene-specific ID；
- 不含坐标或 grid cell；
- 不访问地图；
- 不输出 Python grounding code；
- 不包含 planner weight、cost map 或 path；
- 不使用未注册或自由文本 operator；
- 保留显式 object-object reference；
- 不把 hard constraint 转成 soft preference；
- 不把 soft preference 转成 hard constraint；
- 按既定 policy 处理 unsupported path semantics。

---

### 2.12 Parser 输出契约

Parser 必须：

1. 只输出一个 `task { ... }` program；
2. 不输出 Markdown 或解释文本；
3. 只使用注册 operator；
4. 保留所有可支持的 destination、constraint、preference、reference 和 scope；
5. 保留时间顺序；
6. 每个变量先定义后使用；
7. 所有声明位于 segment 外；
8. 每个新目的地创建新 segment；
9. 除非 instruction 另有说明，后续 segment 使用前一 target 作为 semantic start；
10. 未明确 reference 的 distance selection 使用对应 segment start；
11. 默认使用 `geodesic`，显式 straight-line language 使用 `euclidean`；
12. 指令允许任意 candidate 时使用 `choose_any`，而不是 `unique`；
13. first/second encountered entity 使用相对于 segment start 的 geodesic ranking；
14. current room、unvisited room、exclusive passage 和 return action 必须通过 operator 组合表达；
15. triangle route 必须表示为连续 `go_to` segment；
16. 不生成 entity ID、room ID、坐标、path cell、grounding code 或 planner parameter。



## 3. Grounding

### 3.1 Grounding 目标

Grounding 模块接收已经通过语法检查和类型检查的 Ungrounded DSL AST，并在当前场景的语义地图上执行该程序。它将 DSL 中的符号化房间、物体、位置、区域和约束解析为当前场景中的具体 reference，最终构造供 Planner 使用的 Grounded Task IR。

整体流程为：

```text
Validated Ungrounded DSL AST
        ↓
Deterministic DSL Interpreter
        ↓
Restricted Grounding Operators
        ↓
Typed Grounded References
        ↓
Grounded Task IR
```

Grounding 负责：

* 查询满足符号描述的候选 object、room 和 region；
* 执行集合过滤、关系判断、排序和唯一性检查；
* 解析跨 segment 的 symbolic reference；
* 将 derived region 表达式构造为具体区域；
* 将 segment scope 和 spatial scope 解析为具体引用；
* 将 grounded action、hard constraint 和 soft preference 写入 Grounded Task IR。

Grounding 不负责：

* 重新解释原始自然语言指令；
* 修改 Parser 输出的语义结构；
* 创建新的目标、约束或偏好；
* 选择路径或最终 goal cell；
* 构造 Planner cost field；
* 设置 Planner 数值权重；
* 判断不同可行路径之间的优劣；
* 直接输出 grid path。

Grounding 是一个确定性的符号程序执行过程，而不是第二次自然语言推理。

---

### 3.2 Grounding 的输入和输出

Grounding 的输入包括：

```python
GroundingInput:
    program_ast: TypedTaskAST
    scene: SceneMap
    task_start: PositionRef
```

其中，`program_ast` 是 Parser 输出并经过验证的 typed AST；`scene` 提供语义地图和拓扑信息；`task_start` 是当前任务的初始位置。

Grounding 的输出为：

```python
GroundedProgram:
    segments: list[GroundedSegment]
```

Grounded Program 中的所有 symbolic expression 均已被解析为当前场景中的具体 reference，但尚未生成路径。

例如，Parser 输出：

```text
let target_sink = unique(
    in(entities("sink"), target_bathroom)
)
```

经过 Grounding 后，可能得到：

```python
ObjectRef(
    id="object_17",
    category="sink",
    room_id="room_3",
)
```

Grounding 输出中的 `id` 只由 Grounding runtime 从场景中获得，Parser 和 LLM 均不能生成或猜测具体 ID。

---

### 3.3 Grounding 执行模型

Grounder 通过一个确定性的 DSL interpreter 执行 AST。每个 DSL node 都被映射到一个已注册的 grounding implementation。

```text
DSL AST Node
        ↓
Registered Node Evaluator
        ↓
Grounding Operator
        ↓
Typed Grounded Value
```

例如：

```text
in(entities("sink"), target_room)
```

可以确定性地执行为：

```python
all_sinks = api.get_entities(category="sink")
target_sinks = api.filter_entities_by_room(
    entities=all_sinks,
    room=target_room,
)
```

也可以在底层实现中使用等价的组合查询：

```python
target_sinks = api.get_entities_in_room(
    room=target_room,
    category="sink",
)
```

两种实现必须返回相同的 typed result。Grounding operator 是底层执行接口，不要求与 DSL operator 严格一一对应。一个 DSL expression 可以由多个 grounding operator 组合执行。

Grounder 不再读取原始 instruction，也不允许使用另一个 LLM重新解释指令。

---

### 3.4 Grounding 类型系统

Grounding runtime 使用以下核心类型：

```text
PositionRef
EntityRef
EntitySet
RoomRef
RoomSet
RegionRef
RegionSet
OrderedRefList[T]
Boolean
Integer
Float
AttributeValue
DistanceValue
```

空间引用的联合类型定义为：

```text
GroundedSpatialRef :=
    PositionRef
    | EntityRef
    | RoomRef
    | RegionRef
```

集合类型和单个 reference 必须严格区分：

```text
EntitySet ≠ EntityRef
RoomSet   ≠ RoomRef
RegionSet ≠ RegionRef
```

集合不能直接作为单个 navigation target。集合必须经过以下操作之一转换为单个 reference：

* `require_unique`；
* `choose_any`；
* ranking operator；
* 其他返回单个 reference 的 selection operator。

Grounding runtime 禁止隐式执行：

```python
target = candidates[0]
```

候选集合为空、候选不唯一或 ordinal 超出范围时，必须返回明确的 grounding failure。

---

### 3.5 Grounding Operator Registry

Grounding operator 是 Grounder 可以调用的受限场景查询和几何计算接口。所有 operator 必须具有固定签名、确定性行为和严格类型。

Grounding operator 分为以下八类：

1. 基础 reference 查询；
2. 集合操作；
3. 属性和关系判断；
4. 距离计算；
5. selection 和 ranking；
6. 房间拓扑与 passage 查询；
7. derived region 构造；
8. Grounded IR 构造。

---

#### 3.5.1 基础 Reference 查询

基础查询 operator 从 SceneMap 中读取 object、room 和初始位置等结构化信息。

| Operator                                    | 签名                                               | 含义                |
| ------------------------------------------- | ------------------------------------------------ | ----------------- |
| `get_start_position()`                      | `→ PositionRef`                                  | 返回任务初始位置          |
| `get_rooms(category=None)`                  | `Optional[RoomCategory] → RoomSet`               | 返回全部房间或指定类别房间     |
| `get_entities(category)`                    | `EntityCategory → EntitySet`                     | 返回指定类别的所有实体       |
| `get_entities_in_room(room, category=None)` | `RoomRef × Optional[EntityCategory] → EntitySet` | 返回指定房间内的实体        |
| `get_room_of(reference)`                    | `GroundedSpatialRef → RoomRef`                   | 返回 reference 所在房间 |
| `get_entity_category(entity)`               | `EntityRef → EntityCategory`                     | 返回实体类别            |
| `get_room_category(room)`                   | `RoomRef → RoomCategory`                         | 返回房间类别            |

代表性接口：

```python
class GroundingAPI:
    def get_start_position(self) -> PositionRef:
        ...

    def get_rooms(
        self,
        category: str | None = None,
    ) -> RoomSet:
        ...

    def get_entities(
        self,
        category: str,
    ) -> EntitySet:
        ...

    def get_entities_in_room(
        self,
        room: RoomRef,
        category: str | None = None,
    ) -> EntitySet:
        ...

    def get_room_of(
        self,
        reference: GroundedSpatialRef,
    ) -> RoomRef:
        ...
```

这些 operator 只返回 typed reference，不向调用者暴露 raw semantic map、raw occupancy array 或具体 grid coordinates。

---

#### 3.5.2 集合操作

集合 operator 负责组合和过滤候选 reference。

| Operator                                 | 签名                                | 含义               |
| ---------------------------------------- | --------------------------------- | ---------------- |
| `union_sets(A, B, ...)`                  | compatible sets → set             | 返回集合并集           |
| `intersect_sets(A, B, ...)`              | compatible sets → set             | 返回集合交集           |
| `exclude_from_set(candidates, excluded)` | set × element/set → set           | 排除指定元素           |
| `count_set(candidates)`                  | set → `Integer`                   | 返回集合大小           |
| `filter_set(candidates, predicate)`      | `Set[T] × (T → Boolean) → Set[T]` | 按 predicate 过滤候选 |

接口示例：

```python
def union_sets(
    *sets: RefSet,
) -> RefSet:
    ...

def intersect_sets(
    *sets: RefSet,
) -> RefSet:
    ...

def exclude_from_set(
    candidates: RefSet,
    excluded: GroundedRef | RefSet,
) -> RefSet:
    ...

def count_set(
    candidates: RefSet,
) -> int:
    ...

def filter_set(
    candidates: RefSet,
    predicate: GroundedPredicate,
) -> RefSet:
    ...
```

DSL 中的 `where(set, predicate)` 由 `filter_set` 执行。

在 `where` 的 predicate scope 中，`self` 被绑定为当前测试元素：

```text
where(S: Set[T], predicate: T → Boolean)
```

因此：

```text
where(
    rooms("bedroom"),
    count(in(entities("bed"), self)) == 3
)
```

执行时，`self` 的 Grounding 类型为 `RoomRef`。

---

#### 3.5.3 属性和关系判断

这一类 operator 用于执行 DSL predicate，并返回 Boolean 或属性值。

| Operator                                 | 签名                                          | 含义                         |
| ---------------------------------------- | ------------------------------------------- | -------------------------- |
| `get_attribute(reference, attribute)`    | reference × attribute → value               | 获取 size、area 等属性           |
| `room_contains(room, category)`          | room × category → Boolean                   | 判断房间是否包含某类实体               |
| `count_entities_in_room(room, category)` | room × category → Integer                   | 统计房间内指定类别实体数量              |
| `is_near(candidate, reference)`          | reference × reference → Boolean             | 判断两个 reference 是否接近        |
| `is_far(candidate, reference)`           | reference × reference → Boolean             | 判断两个 reference 是否远离        |
| `is_next_to(candidate, reference)`       | reference × reference → Boolean             | 判断是否相邻                     |
| `is_on_top_of(candidate, reference)`     | entity × entity → Boolean                   | 判断上下关系                     |
| `is_in_corner(candidate, room)`          | reference × room → Boolean                  | 判断是否位于房间角落                 |
| `is_between(candidate, A, B)`            | reference × reference × reference → Boolean | 判断 candidate 是否位于 A 和 B 之间 |

代表性接口：

```python
def get_attribute(
    reference: EntityRef | RoomRef,
    attribute: str,
) -> AttributeValue:
    ...

def room_contains(
    room: RoomRef,
    category: str,
) -> bool:
    ...

def count_entities_in_room(
    room: RoomRef,
    category: str,
) -> int:
    ...

def is_near(
    candidate: GroundedSpatialRef,
    reference: GroundedSpatialRef,
) -> bool:
    ...

def is_in_corner(
    candidate: GroundedSpatialRef,
    room: RoomRef,
) -> bool:
    ...
```

所有关系 operator 的几何定义和阈值由 benchmark 全局配置固定，不由 Parser 或 LLM 生成。

默认规则为：

* object-object 静态关系使用 footprint boundary 的 Euclidean distance；
* nearest 和 farthest selection 默认使用 geodesic distance；
* `next_to` 使用比 `near` 更严格的固定距离阈值；
* `in_corner` 根据 room boundary 和 corner region 的固定定义计算；
* `between` 根据两个 reference 之间的几何 corridor 或中心连线邻域计算。

Grounding implementation 必须对这些定义进行版本化和记录，确保所有方法和样本使用相同的判定标准。

---

#### 3.5.4 距离计算

距离 operator 提供统一的 Euclidean 和 geodesic distance 查询。

| Operator                                           | 签名                                     | 含义                     |
| -------------------------------------------------- | -------------------------------------- | ---------------------- |
| `compute_distance(A, B, metric)`                   | reference × reference × metric → Float | 计算两个 reference 之间的距离   |
| `compute_distances(candidates, reference, metric)` | set × reference × metric → mapping     | 计算候选集合相对 reference 的距离 |
| `compute_pairwise_distances(A, B, metric)`         | set × set × metric → matrix            | 计算两个集合的两两距离            |

接口示例：

```python
def compute_distance(
    reference_a: GroundedSpatialRef,
    reference_b: GroundedSpatialRef,
    metric: DistanceMetric,
) -> float:
    ...

def compute_distances(
    candidates: RefSet,
    reference: GroundedSpatialRef,
    metric: DistanceMetric,
) -> dict[GroundedRef, float]:
    ...

def compute_pairwise_distances(
    candidates: RefSet,
    references: RefSet,
    metric: DistanceMetric,
) -> dict[tuple[GroundedRef, GroundedRef], float]:
    ...
```

距离 metric 只允许：

```text
euclidean
geodesic
```

默认规则：

* entity 和 room ranking 使用 `geodesic`；
* object-object 静态空间关系使用 `euclidean`；
* instruction 明确指定 straight-line distance 时使用 `euclidean`；
* geodesic distance 在 traversable map 上计算；
* entity distance 从 footprint 或合法 interaction region 计算，而不是仅使用 centroid。

Grounding 中的 geodesic distance 只用于 reference selection 和语义关系计算，不生成最终导航路径。

---

#### 3.5.5 Selection 和 Ranking

Selection operator 将候选集合确定性地转换为单个 reference 或有序列表。

| Operator                                                     | 签名                                              | 含义                       |
| ------------------------------------------------------------ | ----------------------------------------------- | ------------------------ |
| `require_unique(candidates)`                                 | set → element                                   | 要求候选唯一                   |
| `choose_any(candidates, reference, metric)`                  | set × reference × metric → element              | 确定性选择离 reference 最近的合法候选 |
| `select_kth_nearest(candidates, reference, k, metric)`       | set × reference × integer × metric → element    | 选择第 k 近候选                |
| `select_kth_farthest(candidates, reference, k, metric)`      | set × reference × integer × metric → element    | 选择第 k 远候选                |
| `select_kth_largest(candidates, attribute, k)`               | set × attribute × integer → element             | 按属性选择第 k 大候选             |
| `select_kth_smallest(candidates, attribute, k)`              | set × attribute × integer → element             | 按属性选择第 k 小候选             |
| `sort_by_distance(candidates, reference, order, metric)`     | set × reference × order × metric → ordered list | 按相对固定 reference 的距离排序    |
| `select_closest_pair_member(candidates, references, metric)` | set × set × metric → element                    | 返回最近配对中 candidates 一侧成员  |

接口示例：

```python
def require_unique(
    candidates: RefSet,
) -> GroundedRef:
    ...

def choose_any(
    candidates: RefSet,
    reference: GroundedSpatialRef,
    metric: DistanceMetric = "geodesic",
) -> GroundedRef:
    ...

def select_kth_nearest(
    candidates: RefSet,
    reference: GroundedSpatialRef,
    k: int,
    metric: DistanceMetric = "geodesic",
) -> GroundedRef:
    ...

def select_kth_farthest(
    candidates: RefSet,
    reference: GroundedSpatialRef,
    k: int,
    metric: DistanceMetric = "geodesic",
) -> GroundedRef:
    ...

def select_kth_largest(
    candidates: RefSet,
    attribute: str,
    k: int,
) -> GroundedRef:
    ...

def select_kth_smallest(
    candidates: RefSet,
    attribute: str,
    k: int,
) -> GroundedRef:
    ...

def sort_by_distance(
    candidates: RefSet,
    reference: GroundedSpatialRef,
    order: Literal["ascending", "descending"],
    metric: DistanceMetric = "geodesic",
) -> OrderedRefList:
    ...

def select_closest_pair_member(
    candidates: RefSet,
    references: RefSet,
    metric: DistanceMetric = "euclidean",
) -> GroundedRef:
    ...
```

Tie-breaking 必须确定性执行。若多个 candidate 的主排序值相同，则按照以下顺序打破平局：

1. secondary geometric value；
2. category-specific stable attribute；
3. scene-provided stable ID。

Grounder 不允许随机选择候选。

`choose_any` 不表示随机选择，而是选择相对于指定 reference 的 geodesic distance 最近候选。只有 instruction 明确允许任意合法 candidate 时，Parser 才能调用该操作。

---

#### 3.5.6 房间拓扑与 Passage 查询

Topology operator 用于查询房间连接关系和房间之间的可通行区域。

| Operator                     | 签名                       | 含义                         |
| ---------------------------- | ------------------------ | -------------------------- |
| `get_adjacent_rooms(room)`   | room → room set          | 返回与指定房间相邻的房间               |
| `are_rooms_adjacent(A, B)`   | room × room → Boolean    | 判断两个房间是否相邻                 |
| `get_passages_between(A, B)` | room × room → region set | 返回两个房间之间的所有 passage region |
| `get_room_neighbors(room)`   | room → ordered room list | 返回稳定排序的相邻房间列表              |

接口示例：

```python
def get_adjacent_rooms(
    room: RoomRef,
) -> RoomSet:
    ...

def are_rooms_adjacent(
    room_a: RoomRef,
    room_b: RoomRef,
) -> bool:
    ...

def get_passages_between(
    room_a: RoomRef,
    room_b: RoomRef,
) -> RegionSet:
    ...
```

`passage region` 表示机器人可以实际穿越的空间区域，与 door、doorframe 等语义实体不同：

* door 或 doorframe 是 `EntityRef`；
* passage 是 `RegionRef`；
* 一个 passage 可以由一个或多个 door-related entity 推导得到；
* hard constraint 中的必须经过或禁止经过对象应最终转换为 `RegionRef`。

---

#### 3.5.7 Derived Region 构造

部分 DSL expression 不直接对应已有的 object 或 room，而需要 Grounder 根据具体场景构造 derived region。

| Operator                                    | 签名                             | 含义                                |
| ------------------------------------------- | ------------------------------ | --------------------------------- |
| `get_entity_region(entity)`                 | entity → region                | 获取实体 footprint 对应区域               |
| `get_room_region(room)`                     | room → region                  | 获取完整房间区域                          |
| `build_midpoint_region(A, B)`               | reference × reference → region | 构造两个 reference 的中点区域              |
| `build_between_region(A, B)`                | reference × reference → region | 构造两个 reference 之间的通行区域            |
| `build_near_region(reference, radius)`      | reference × float → region     | 构造 reference 周围指定半径区域             |
| `build_side_region(center, side_reference)` | reference × reference → region | 构造 center 朝向 side reference 的一侧区域 |
| `build_boundary_region(room)`               | room → region                  | 构造房间边界附近区域                        |
| `build_half_room_region(room, reference)`   | room × reference → region      | 构造房间朝向 reference 的半侧区域            |

接口示例：

```python
def get_entity_region(
    entity: EntityRef,
) -> RegionRef:
    ...

def get_room_region(
    room: RoomRef,
) -> RegionRef:
    ...

def build_midpoint_region(
    reference_a: GroundedSpatialRef,
    reference_b: GroundedSpatialRef,
) -> RegionRef:
    ...

def build_between_region(
    reference_a: GroundedSpatialRef,
    reference_b: GroundedSpatialRef,
) -> RegionRef:
    ...

def build_near_region(
    reference: GroundedSpatialRef,
    radius: float,
) -> RegionRef:
    ...

def build_side_region(
    center: GroundedSpatialRef,
    side_reference: GroundedSpatialRef,
) -> RegionRef:
    ...

def build_boundary_region(
    room: RoomRef,
) -> RegionRef:
    ...

def build_half_room_region(
    room: RoomRef,
    reference: GroundedSpatialRef,
) -> RegionRef:
    ...
```

每个 derived region 必须保存构造来源：

```python
RegionRef:
    id: str
    mask_handle: RegionMaskHandle
    construction: RegionConstruction
```

例如：

```python
RegionConstruction(
    type="between",
    references=[
        ObjectRef(id="object_12"),
        ObjectRef(id="object_19"),
    ],
    parameters={},
)
```

保存 construction provenance 可以支持：

* evaluator 重建；
* 可视化；
* grounding debugging；
* operator-level accuracy evaluation；
* 不同版本 region definition 的追踪。

Derived region 必须满足：

* 与 SceneMap 坐标系一致；
* 至少包含一个有效 cell；
* 使用固定全局参数；
* 不由 LLM 指定 arbitrary mask；
* 不包含 Grounder 自行添加的路径偏好。

---

#### 3.5.8 Grounded IR 构造 Operator

场景查询 operator 和 Grounded IR 构造 operator 应当分离。

`GroundingAPI` 负责：

* 查询场景；
* 计算关系；
* 执行 selection；
* 构造 region。

`GroundedIRBuilder` 负责：

* 构造 action；
* 构造 grounded constraint；
* 构造 grounded scope；
* 组织 segment；
* 输出 GroundedProgram。

代表性接口如下：

```python
class GroundedIRBuilder:
    def make_go_to(
        self,
        target: GroundedSpatialRef,
    ) -> GroundedAction:
        ...

    def make_require_visit(
        self,
        region: RegionRef,
    ) -> GroundedHardConstraint:
        ...

    def make_require_visit_in_order(
        self,
        regions: list[RegionRef],
    ) -> GroundedHardConstraint:
        ...

    def make_forbid(
        self,
        region: RegionRef,
    ) -> GroundedHardConstraint:
        ...

    def make_near_preference(
        self,
        reference: EntityRef | EntitySet | RegionRef,
        spatial_scope: GroundedSpatialScope,
    ) -> GroundedSoftConstraint:
        ...

    def make_far_preference(
        self,
        reference: EntityRef | EntitySet | RegionRef,
        spatial_scope: GroundedSpatialScope,
    ) -> GroundedSoftConstraint:
        ...

    def make_relative_preference(
        self,
        reference_a: GroundedSpatialRef,
        relation: Literal["closer_to", "farther_from"],
        reference_b: GroundedSpatialRef,
        spatial_scope: GroundedSpatialScope,
    ) -> GroundedSoftConstraint:
        ...

    def add_segment(
        self,
        segment_id: str,
        start_reference: GroundedSpatialRef,
        action: GroundedAction,
        hard_constraints: list[GroundedHardConstraint],
        soft_constraints: list[GroundedSoftConstraint],
    ) -> None:
        ...

    def build_program(
        self,
    ) -> GroundedProgram:
        ...
```

Scope 构造接口包括：

```python
def make_room_scope(
    rooms: RoomRef | RoomSet,
) -> GroundedSpatialScope:
    ...

def make_region_scope(
    region: RegionRef,
) -> GroundedSpatialScope:
    ...

def make_whole_scene_scope(
) -> GroundedSpatialScope:
    ...

def make_segment_scope(
    segment_ids: list[str],
) -> GroundedSegmentScope:
    ...
```

Grounded IR builder 不访问 SceneMap，也不计算距离或区域。它只将已经 grounded 的 typed reference 组织成 Planner 所需的结构。

---

### 3.6 DSL Operator 与 Grounding Operator 的映射

DSL operator 和 grounding operator 不需要严格一一对应。它们之间存在三种关系。

#### 3.6.1 直接映射

部分 DSL operator 可以直接映射到底层 grounding operator：

| DSL Operator            | Grounding Operator            |
| ----------------------- | ----------------------------- |
| `rooms(category)`       | `get_rooms(category)`         |
| `entities(category)`    | `get_entities(category)`      |
| `room_of(reference)`    | `get_room_of(reference)`      |
| `adjacent_rooms(room)`  | `get_adjacent_rooms(room)`    |
| `passage_regions(A, B)` | `get_passages_between(A, B)`  |
| `kth_nearest(...)`      | `select_kth_nearest(...)`     |
| `kth_farthest(...)`     | `select_kth_farthest(...)`    |
| `kth_largest(...)`      | `select_kth_largest(...)`     |
| `kth_smallest(...)`     | `select_kth_smallest(...)`    |
| `midpoint_region(A, B)` | `build_midpoint_region(A, B)` |
| `between_region(A, B)`  | `build_between_region(A, B)`  |
| `boundary_region(room)` | `build_boundary_region(room)` |

#### 3.6.2 Interpreter 内部执行

以下 DSL structure 和逻辑操作不需要访问 SceneMap，由 interpreter 本身执行：

```text
task
segment
let
from
to
and
or
not
==
!=
>
>=
<
<=
```

例如：

```text
count(candidate_set) == 3
```

由 interpreter 对 `count_set(candidate_set)` 的结果执行整数比较。

#### 3.6.3 组合执行

部分 DSL expression 会被编译为多个 grounding operator。

例如：

```text
in(entities("sink"), target_room)
```

可以执行为：

```python
candidate_sinks = api.get_entities("sink")

target_sinks = api.filter_set(
    candidate_sinks,
    predicate=lambda entity:
        api.get_room_of(entity) == target_room,
)
```

或者使用等价的优化实现：

```python
target_sinks = api.get_entities_in_room(
    target_room,
    category="sink",
)
```

类似地：

```text
where(
    rooms("bedroom"),
    count(in(entities("bed"), self)) == 3
)
```

可以执行为：

```python
candidate_rooms = api.get_rooms("bedroom")

valid_rooms = api.filter_set(
    candidate_rooms,
    predicate=lambda room:
        api.count_entities_in_room(
            room,
            category="bed",
        ) == 3,
)
```

不同执行方式必须满足相同的 DSL semantics。

---

### 3.7 Grounding Context 和跨 Segment Reference

Grounder 一次性执行完整 task program，并维护一个 Grounding Context：

```python
GroundingContext:
    task_start: PositionRef
    variable_bindings: dict[str, GroundedValue]
    segment_targets: dict[str, GroundedSpatialRef]
    segment_start_references: dict[str, GroundedSpatialRef]
```

第一个 segment 的 semantic start 通常是：

```python
context.task_start
```

后续 segment 的 semantic start 通常是前一 segment 的 grounded target：

```python
context.segment_targets["s1"]
```

例如：

```text
segment s1 {
    from task_start
    to target_sink
    go_to(target_sink)
}

segment s2 {
    from target_sink
    to target_bed
    go_to(target_bed)
}
```

Grounding 阶段中，`s2` 的 semantic start reference 是具体的 `target_sink ObjectRef`。

因此，下列表达式可以在完整任务 Grounding 时直接解析：

```text
the bed farthest from the first target
```

```text
the nearest table from the second segment's starting point
```

Grounding 不需要等待 Planner 完成第一段路径。

需要区分：

* semantic start reference：用于解析自然语言中的距离和关系；
* physical segment start：Planner 实际规划时使用的上一段 endpoint。

Planner 按 segment 顺序执行时，使用上一段的实际路径 endpoint 作为下一段物理起点。Grounder 不根据该 endpoint 重新执行 grounding。

---

### 3.8 Scope Grounding

Grounding 需要将 Parser 输出的 symbolic scope 转换为具体 spatial reference。

#### 3.8.1 显式 Room Scope

```text
within_rooms({target_room})
```

转换为：

```python
RoomScope(
    rooms=[RoomRef(id="room_3")]
)
```

#### 3.8.2 显式 Region Scope

```text
within(target_region)
```

转换为：

```python
RegionScope(
    region=RegionRef(id="region_7")
)
```

#### 3.8.3 默认 Object-related Scope

对于 object-related near、far 或 relative preference，如果 instruction 未显式提供 spatial scope，则默认使用 reference object 所在房间：

```python
reference_room = api.get_room_of(reference_object)

scope = builder.make_room_scope(
    reference_room
)
```

例如：

```text
prefer_near(reference_table)
```

Grounding 为：

```python
NearPreference(
    reference=ObjectRef(id="object_18"),
    spatial_scope=RoomScope(
        rooms=[RoomRef(id="room_5")]
    ),
)
```

#### 3.8.4 Relative Preference Scope

对于：

```text
prefer_relative(A, closer_to, B)
```

如果 A 和 B 位于同一房间，则默认 scope 为该房间。

如果 A 和 B 位于不同房间，且 instruction 没有显式指定 scope，则 Grounder 返回：

```text
INVALID_SCOPE
```

Grounder不能静默选择 A 或 B 所在房间，也不能默认升级为 whole-scene scope。

#### 3.8.5 Segment Scope

Segment scope 已由 Parser 恢复。Grounder 只需要验证引用的 segment 存在，并将 symbolic segment ID 写入 Grounded IR。

---

### 3.9 Grounded Task IR

Grounding 的最终输出是 Grounded Task IR。IR 是 Grounder 和 Planner 之间的结构化接口。

建议定义如下：

```python
@dataclass
class GroundedProgram:
    segments: list["GroundedSegment"]


@dataclass
class GroundedSegment:
    id: str
    start_reference: "GroundedSpatialRef"
    action: "GroundedAction"
    hard_constraints: list["GroundedHardConstraint"]
    soft_constraints: list["GroundedSoftConstraint"]


@dataclass
class GroundedAction:
    type: Literal["go_to"]
    target: "GroundedSpatialRef"


@dataclass
class GroundedHardConstraint:
    type: Literal[
        "require_visit",
        "require_visit_in_order",
        "forbid",
    ]
    regions: list["RegionRef"]


@dataclass
class GroundedSoftConstraint:
    type: Literal[
        "near",
        "far",
        "relative",
        "circle",
    ]
    references: list["GroundedSpatialRef"]
    relation: str | None
    spatial_scope: "GroundedSpatialScope"
    segment_scope: "GroundedSegmentScope"


@dataclass
class GroundedSpatialScope:
    type: Literal[
        "room",
        "region",
        "whole_scene",
    ]
    references: list["RoomRef | RegionRef"]


@dataclass
class GroundedSegmentScope:
    segment_ids: list[str]
```

如果基础 Planner 暂不支持 `circle`，则 `circle` 不应出现在当前版本的有效 `GroundedSoftConstraint.type` 中，而应在 Parser validation 阶段返回 unsupported-semantics error。

Grounded IR 中不包含：

* 自然语言文本；
* DSL expression；
* 未解析变量；
* raw map；
* grid path；
* cost map；
* planner weight；
* 最终目标 cell。

Grounded IR 只表达：

* 具体目标；
* 具体 reference；
* 具体 hard region；
* 具体 soft preference；
* 具体作用域；
* segment 顺序。

---

### 3.10 Grounding 示例

输入指令：

```text
First, go to the sink in the bathroom without a laundry hamper.
Along the way, stay close to the dining table nearest to the
starting point. Then go to the bed farthest from the first target
while staying far from the sofa in that bedroom.
```

Parser 输出：

```text
task {
    let task_start = start_position

    let candidate_bathrooms = where(
        rooms("bathroom"),
        count(in(entities("laundry_hamper"), self)) == 0
    )

    let target_bathroom = unique(candidate_bathrooms)

    let target_sink = unique(
        in(entities("sink"), target_bathroom)
    )

    let reference_table = kth_nearest(
        entities("dining_table"),
        task_start,
        k=1,
        metric=geodesic
    )

    let target_bed = kth_farthest(
        entities("bed"),
        target_sink,
        k=1,
        metric=geodesic
    )

    let target_bedroom = room_of(target_bed)

    let reference_sofa = unique(
        in(entities("sofa"), target_bedroom)
    )

    segment s1 {
        from task_start
        to target_sink
        go_to(target_sink)

        prefer_near(reference_table)
            on_segments({s1})
            within_rooms({room_of(reference_table)})
    }

    segment s2 {
        from target_sink
        to target_bed
        go_to(target_bed)

        prefer_far(reference_sofa)
            on_segments({s2})
            within_rooms({target_bedroom})
    }
}
```

Grounding 执行：

```python
def ground_program(
    api: GroundingAPI,
    builder: GroundedIRBuilder,
) -> GroundedProgram:
    task_start = api.get_start_position()

    bathrooms = api.get_rooms("bathroom")

    valid_bathrooms = api.filter_set(
        bathrooms,
        predicate=lambda room:
            api.count_entities_in_room(
                room,
                category="laundry_hamper",
            ) == 0,
    )

    target_bathroom = api.require_unique(
        valid_bathrooms
    )

    target_sink = api.require_unique(
        api.get_entities_in_room(
            target_bathroom,
            category="sink",
        )
    )

    reference_table = api.select_kth_nearest(
        candidates=api.get_entities("dining_table"),
        reference=task_start,
        k=1,
        metric="geodesic",
    )

    target_bed = api.select_kth_farthest(
        candidates=api.get_entities("bed"),
        reference=target_sink,
        k=1,
        metric="geodesic",
    )

    target_bedroom = api.get_room_of(
        target_bed
    )

    reference_sofa = api.require_unique(
        api.get_entities_in_room(
            target_bedroom,
            category="sofa",
        )
    )

    table_room = api.get_room_of(
        reference_table
    )

    s1_action = builder.make_go_to(
        target_sink
    )

    s1_near = builder.make_near_preference(
        reference=reference_table,
        spatial_scope=builder.make_room_scope(
            table_room
        ),
    )

    builder.add_segment(
        segment_id="s1",
        start_reference=task_start,
        action=s1_action,
        hard_constraints=[],
        soft_constraints=[s1_near],
    )

    s2_action = builder.make_go_to(
        target_bed
    )

    s2_far = builder.make_far_preference(
        reference=reference_sofa,
        spatial_scope=builder.make_room_scope(
            target_bedroom
        ),
    )

    builder.add_segment(
        segment_id="s2",
        start_reference=target_sink,
        action=s2_action,
        hard_constraints=[],
        soft_constraints=[s2_far],
    )

    return builder.build_program()
```

可能的 Grounded Task IR 为：

```json
{
  "segments": [
    {
      "id": "s1",
      "start_reference": {
        "kind": "position",
        "id": "task_start"
      },
      "action": {
        "type": "go_to",
        "target": {
          "kind": "entity",
          "id": "object_17",
          "category": "sink",
          "room_id": "room_3"
        }
      },
      "hard_constraints": [],
      "soft_constraints": [
        {
          "type": "near",
          "references": [
            {
              "kind": "entity",
              "id": "object_18",
              "category": "dining_table",
              "room_id": "room_5"
            }
          ],
          "spatial_scope": {
            "type": "room",
            "references": [
              {
                "kind": "room",
                "id": "room_5"
              }
            ]
          },
          "segment_scope": {
            "segment_ids": ["s1"]
          }
        }
      ]
    },
    {
      "id": "s2",
      "start_reference": {
        "kind": "entity",
        "id": "object_17"
      },
      "action": {
        "type": "go_to",
        "target": {
          "kind": "entity",
          "id": "object_42",
          "category": "bed",
          "room_id": "room_8"
        }
      },
      "hard_constraints": [],
      "soft_constraints": [
        {
          "type": "far",
          "references": [
            {
              "kind": "entity",
              "id": "object_46",
              "category": "sofa",
              "room_id": "room_8"
            }
          ],
          "spatial_scope": {
            "type": "room",
            "references": [
              {
                "kind": "room",
                "id": "room_8"
              }
            ]
          },
          "segment_scope": {
            "segment_ids": ["s2"]
          }
        }
      ]
    }
  ]
}
```

---

### 3.11 Grounding Failure

Grounding 必须显式报告失败，而不是静默修改语义或选择 arbitrary fallback。

推荐 failure state：

```text
EMPTY_CANDIDATE
AMBIGUOUS_CANDIDATE
ORDINAL_OUT_OF_RANGE
INVALID_REFERENCE
INVALID_TYPE
INVALID_ATTRIBUTE
INVALID_RELATION
INVALID_SCOPE
INVALID_REGION
INVALID_TOPOLOGY_QUERY
UNSUPPORTED_OPERATOR
UNSUPPORTED_SEMANTICS
INVALID_SEGMENT_REFERENCE
EMPTY_TARGET_REGION
```

各 failure 的含义如下：

| Failure                     | 含义                                     |
| --------------------------- | -------------------------------------- |
| `EMPTY_CANDIDATE`           | 候选集合为空                                 |
| `AMBIGUOUS_CANDIDATE`       | `require_unique` 接收到多个候选               |
| `ORDINAL_OUT_OF_RANGE`      | 第 k 个候选不存在                             |
| `INVALID_REFERENCE`         | 引用了不存在或未绑定的变量                          |
| `INVALID_TYPE`              | operator 参数类型不匹配                       |
| `INVALID_ATTRIBUTE`         | reference 不支持指定属性                      |
| `INVALID_RELATION`          | relation 不在注册集合中                       |
| `INVALID_SCOPE`             | scope 无法确定或语义冲突                        |
| `INVALID_REGION`            | derived region 为空或无法构造                 |
| `INVALID_TOPOLOGY_QUERY`    | 对非相邻房间查询 passage 等非法拓扑操作               |
| `UNSUPPORTED_OPERATOR`      | AST 包含未注册 operator                     |
| `UNSUPPORTED_SEMANTICS`     | Parser 保留了当前版本无法执行的语义                  |
| `INVALID_SEGMENT_REFERENCE` | 引用了不存在或未来的 segment                     |
| `EMPTY_TARGET_REGION`       | target 附近不存在合法 traversable goal region |

`NO_FEASIBLE_PATH` 不属于 Grounding failure。它应由 Planner 在考虑 occupancy、hard constraint 和完整路径搜索后返回。

Grounder 禁止：

* 返回候选集合中的第一个元素作为 fallback；
* 随机选择候选；
* 删除无法 grounding 的约束；
* 将 hard constraint 转换为 soft preference；
* 将 soft preference 转换为 hard constraint；
* 修改 segment 顺序；
* 用当前 agent position 替换显式 object-object reference；
* 构造未在 DSL 中出现的新 target。

---

### 3.12 Grounding Validation

在 Grounded Task IR 进入 Planner 前，必须执行以下验证。

#### 3.12.1 Reference Validation

* 所有 reference 均存在于当前 SceneMap；
* 所有 entity 和 room ID 有效；
* 所有 variable binding 已解析；
* 所有 target 均为单个 reference；
* 不存在未解析的 symbolic expression。

#### 3.12.2 Type Validation

* action target 类型合法；
* hard constraint 只引用 RegionRef；
* soft constraint reference 类型符合定义；
* room scope 只包含 RoomRef；
* region scope 只包含 RegionRef；
* ranking result 与 DSL 预期类型一致。

#### 3.12.3 Segment Validation

* segment ID 唯一；
* segment 顺序与 Parser 输出一致；
* 每个 segment 恰好包含一个 `go_to` action；
* 后续 segment 的 semantic start reference 合法；
* constraint 的 segment scope 只引用已有 segment；
* 不存在从未来 segment 到当前 segment 的非法依赖。

#### 3.12.4 Region Validation

* 所有 derived region 可确定性重建；
* region mask 非空；
* region 位于有效场景范围内；
* passage region 对应合法房间连接；
* hard constraint region 具有可解释 construction provenance。

#### 3.12.5 Semantic Validation

* 没有添加、删除或重排 Parser 输出的要求；
* hard 和 soft modality 保持不变；
* object-object relation 没有被改写为 agent-object relation；
* scope 与 Parser 输出和默认规则一致；
* distance metric 与 DSL 声明一致；
* tie-breaking 符合固定 benchmark policy。

---

### 3.13 Grounding Operator 的限制

Grounding API 不得提供以下接口：

```python
get_raw_occupancy_map()
get_raw_semantic_map()
get_raw_room_label_map()
get_grid_coordinates()
construct_object_id()
construct_room_id()
select_goal_cell()
build_cost_map()
plan_path()
run_astar()
set_cost_weight()
rank_complete_paths()
```

原因是这些接口会破坏 Grounding 和 Planner 的职责边界。

Grounder 可以查询：

* typed entity；
* typed room；
* typed region；
* fixed distance；
* fixed attribute；
* fixed topology relation。

Grounder不能执行：

* 路径优化；
* endpoint selection；
* cost-field construction；
* full trajectory comparison；
* planner hyperparameter selection。

如果 Grounding implementation 为了效率需要访问底层 map array，该访问必须封装在 grounding operator 内部。DSL interpreter、LLM 和生成代码不能直接读取这些数组。

---

### 3.14 Grounding Operator 最小实现集合

基础版本至少需要实现以下 Grounding operator：

```python
class GroundingAPI:
    # Basic reference queries
    def get_start_position(self) -> PositionRef: ...
    def get_rooms(self, category=None) -> RoomSet: ...
    def get_entities(self, category) -> EntitySet: ...
    def get_entities_in_room(
        self,
        room,
        category=None,
    ) -> EntitySet: ...
    def get_room_of(self, reference) -> RoomRef: ...

    # Set operations
    def union_sets(self, *sets) -> RefSet: ...
    def intersect_sets(self, *sets) -> RefSet: ...
    def exclude_from_set(
        self,
        candidates,
        excluded,
    ) -> RefSet: ...
    def count_set(self, candidates) -> int: ...
    def filter_set(
        self,
        candidates,
        predicate,
    ) -> RefSet: ...

    # Attributes and predicates
    def get_attribute(
        self,
        reference,
        attribute,
    ) -> AttributeValue: ...
    def room_contains(
        self,
        room,
        category,
    ) -> bool: ...
    def count_entities_in_room(
        self,
        room,
        category,
    ) -> int: ...
    def is_near(self, candidate, reference) -> bool: ...
    def is_far(self, candidate, reference) -> bool: ...
    def is_next_to(self, candidate, reference) -> bool: ...
    def is_on_top_of(self, candidate, reference) -> bool: ...
    def is_in_corner(self, candidate, room) -> bool: ...
    def is_between(self, candidate, a, b) -> bool: ...

    # Distance
    def compute_distance(
        self,
        a,
        b,
        metric,
    ) -> float: ...
    def compute_distances(
        self,
        candidates,
        reference,
        metric,
    ) -> dict: ...

    # Selection and ranking
    def require_unique(self, candidates): ...
    def choose_any(
        self,
        candidates,
        reference,
        metric="geodesic",
    ): ...
    def select_kth_nearest(
        self,
        candidates,
        reference,
        k,
        metric="geodesic",
    ): ...
    def select_kth_farthest(
        self,
        candidates,
        reference,
        k,
        metric="geodesic",
    ): ...
    def select_kth_largest(
        self,
        candidates,
        attribute,
        k,
    ): ...
    def select_kth_smallest(
        self,
        candidates,
        attribute,
        k,
    ): ...
    def sort_by_distance(
        self,
        candidates,
        reference,
        order,
        metric="geodesic",
    ) -> OrderedRefList: ...
    def select_closest_pair_member(
        self,
        candidates,
        references,
        metric="euclidean",
    ): ...

    # Topology
    def get_adjacent_rooms(self, room) -> RoomSet: ...
    def are_rooms_adjacent(self, room_a, room_b) -> bool: ...
    def get_passages_between(
        self,
        room_a,
        room_b,
    ) -> RegionSet: ...

    # Derived regions
    def get_entity_region(self, entity) -> RegionRef: ...
    def get_room_region(self, room) -> RegionRef: ...
    def build_midpoint_region(self, a, b) -> RegionRef: ...
    def build_between_region(self, a, b) -> RegionRef: ...
    def build_near_region(
        self,
        reference,
        radius,
    ) -> RegionRef: ...
    def build_side_region(
        self,
        center,
        side_reference,
    ) -> RegionRef: ...
    def build_boundary_region(self, room) -> RegionRef: ...
    def build_half_room_region(
        self,
        room,
        reference,
    ) -> RegionRef: ...
```

这组 operator 足以支持基础版本中的：

* object 和 room 查询；
* room containment；
* count-based room filtering；
* nearest、farthest 和 ordinal selection；
* object-object relation；
* adjacent-room query；
* passage constraint；
* midpoint 和 between target；
* near、far 和 relative preference grounding；
* must-pass 和 forbidden region grounding。

后续新增 DSL operator 时，必须同时满足以下条件之一：

1. 已有 Grounding operator 可以组合执行；
2. 新增一个具有明确类型和确定性定义的 Grounding operator。

不能只在 Parser 中添加 operator，而不定义其 Grounding 和 Planner semantics。


## 4. Planner

### 4.1 Planner 目标

Planner 接收一个 grounded segment，并生成一条路径，使其：

1. 在 benchmark success definition 下到达 grounded target；
2. 满足所有 hard constraint；
3. 最小化 path length 和 soft-constraint cost。

基础实现使用 sequential segment planning，并对每个 segment 执行一次 cost-aware A* search。

### 4.2 Target Goal Region

对于 target object \(A\)，令：

- \(O_A\) 为它的二维 footprint；
- \(R_A\) 为其所在房间的 mask；
- \(\mathcal F\) 为 robot-traversable free-space mask。

有效 target region 为：

\[
G(A)=
\operatorname{Dilate}(O_A,0.3\text{ m})
\cap R_A
\cap \mathcal F
\setminus O_A.
\]

因此，机器人最终位置必须：

- 距离 object 外部 footprint 不超过 0.3 m；
- 保持在 object 所在房间内；
- collision-free 且 traversable。

距离从 object footprint 测量，而不是从 centroid 测量。

```python
distance_to_target = footprint_distance_transform(
    target_object.footprint,
    resolution=scene.map_resolution,
)

goal_region = (
    (distance_to_target > 0.0)
    & (distance_to_target <= 0.3)
    & target_room_mask
    & scene.traversable_map
)
```

所有合法 cell 都保留为 goal set。搜索前不会预先选择单个 endpoint。

### 4.3 Hard Constraint

Hard constraint 定义搜索可行性。

#### 4.3.1 Occupancy 和 Traversability

```python
if not scene.traversable_map[next_cell]:
    continue
```

Traversable map 应考虑 robot footprint 和所需 collision margin。

#### 4.3.2 Must Avoid

对于 `MustAvoid(area)`，将 grounded area 加入 forbidden mask：

```python
forbidden_mask = ~scene.traversable_map

for constraint in must_avoid_constraints:
    forbidden_mask |= scene.get_area_mask(
        constraint.reference
    )
```

Forbidden cell 永远不会被扩展。

#### 4.3.3 Must Pass

对于 `MustPass(area)`，扩展 search state：

```python
SearchState(
    position=(x, y),
    passed_required_area=False,
)
```

当路径进入 required area 时更新 state。只有同时满足 goal 和 required event，segment 才能终止。多个 required area 可以使用 bit mask 或 finite-state progress index。

#### 4.3.4 有序 Path Event

有序 event 使用离散 progress state：

```python
SearchState(
    position=(x, y),
    heading=heading,
    event_index=k,
)
```

只有当前 required region 或 event 被满足时，event index 才会前进。完整 circling 等复杂 event 需要专门的 state machine，在基础实现中暂缓。

### 4.4 Soft Constraint Field

所有 soft spatial cost 都归一化到 \([0,1]\)。Object-related preference 只在其 grounded room scope 内生效。

对于 reference object \(C\)，定义：

\[
M_C(x)=
\begin{cases}
1,&x\in R_C,\\
0,&x\notin R_C.
\end{cases}
\]

不额外引入鼓励进入该房间的 cost；在 benchmark task 中，规划路径预计会因为到达其 target 而进入相关房间。

#### 4.4.1 Near Preference

令 \(d_C(x)\) 为 cell \(x\) 到 reference object \(C\) footprint 的欧氏距离：

\[
C_{\mathrm{near}}(x;C)
=
M_C(x)
\min\left(
\frac{d_C(x)}{r_{\mathrm{near}}},
1
\right).
\]

```python
near_cost = np.clip(
    distance_to_reference / near_radius,
    0.0,
    1.0,
)
near_cost *= reference_room_mask
```

#### 4.4.2 Far Preference

\[
C_{\mathrm{far}}(x;C)
=
M_C(x)
\exp\left(
-\frac{d_C(x)}{\sigma_{\mathrm{far}}}
\right).
\]

```python
far_cost = np.exp(
    -distance_to_reference / far_sigma
)
far_cost *= reference_room_mask
```

#### 4.4.3 Relative Preference

对于 “remain closer to \(A\) than to \(B\)”：

\[
C_{\mathrm{relative}}(x;A,B)
=
M_R(x)
\operatorname{clip}
\left(
\frac{\max(0,d_A(x)-d_B(x))}
{r_{\mathrm{relative}}},
0,
1
\right).
\]

当 \(d_A(x)\le d_B(x)\) 时 cost 为零；否则 cost 随违反程度增大。

#### 4.4.4 Clearance Preference

令 \(d_{\mathrm{occ}}(x)\) 为 cell \(x\) 到最近 occupied region 的距离。定义平方 clearance-deficit penalty：

\[
C_{\mathrm{clear}}(x)
=
\left[
\max\left(
0,
\frac{r_{\mathrm{clear}}-d_{\mathrm{occ}}(x)}
{r_{\mathrm{clear}}}
\right)
\right]^2.
\]

```python
clearance_deficit = np.maximum(
    0.0,
    clear_radius - distance_to_occupancy,
)

clearance_cost = (
    clearance_deficit / clear_radius
) ** 2
```

当路径达到或超过期望 clearance 时 cost 为零；靠近 occupancy 时 cost 以二次形式增长。Clearance 通常应用于整个 segment。Occupancy 包含 target object，因此 planner 会倾向于停在有效 0.3 m goal region 的外边界附近，而不是不必要地贴近 object footprint。

#### 4.4.5 Path Shape 和 Smoothness

简单 smoothness 是 transition-dependent cost，而不是二维 cost field。当前实现只在
`circle(...)` 降成的 path-shape waypoint 阶段中启用，普通 navigation segment 不加这项。
为了避免 heading-state A* 的状态空间膨胀，实现使用当前 cell 的最佳 predecessor 推断
incoming heading，并在下一步转向时加轻量惩罚：

```python
transition_cost += WEIGHT_SMOOTHNESS * turn_cost(previous_heading, next_heading)
```

对于连续 heading：

\[
C_{\mathrm{shape}}(s_t,s_{t+1})
=
\left(
\frac{\operatorname{wrap}(\theta_{t+1}-\theta_t)}{\pi}
\right)^2.
\]

指定形状，如 circling、retracing 或 S-shaped motion，需要 ordered region 或 event-state automata，不能简化为静态 cost field。

### 4.5 Combined Path Cost

对于 transition \(s_t\rightarrow s_{t+1}\)：

\[
c(s_t,s_{t+1})
=
c_{\mathrm{move}}
+\lambda_{\mathrm{near}}C_{\mathrm{near}}(x_{t+1})
+\lambda_{\mathrm{far}}C_{\mathrm{far}}(x_{t+1})
+\lambda_{\mathrm{relative}}C_{\mathrm{relative}}(x_{t+1})
+\lambda_{\mathrm{clear}}C_{\mathrm{clear}}(x_{t+1})
+\lambda_{\mathrm{shape}}C_{\mathrm{shape}}(s_t,s_{t+1}).
\]

当同一类型有多个 active constraint 时，先平均它们的 normalized field，再应用 type-level weight。Planner weight 是固定 global hyperparameter，永远不由 LLM 生成。

### 4.6 Cost-Aware A*

累计 cost 为：

\[
g(n)=
\sum_{(s_t,s_{t+1})\in\tau_n}
c(s_t,s_{t+1}).
\]

搜索 priority 为：

\[
f(n)=g(n)+h(n),
\]

其中 \(h(n)\) 是到 goal region 中任意 cell 的剩余 movement distance 的 lower bound。

该算法最好称为 **cost-aware A***、**semantic cost A*** 或 **multi-cost A***。它使用 path-cost component 的 weighted sum。除非显式给 heuristic 加权，否则它不是经典 Weighted A*：

\[
f(n)=g(n)+w_hh(n),\qquad w_h>1.
\]

### 4.7 基础 Planner Interface

```python
class SemPathPlanner:
    def plan_program(
        self,
        scene,
        start,
        grounded_program,
    ) -> PlannedProgram:
        current_position = start
        planned_segments = []

        for segment in grounded_program.segments:
            planned_segment = self.plan_segment(
                scene,
                segment,
                start=current_position,
            )
            planned_segments.append(planned_segment)
            current_position = planned_segment.endpoint

        return PlannedProgram(
            segments=planned_segments,
            path=concatenate_without_duplicate(
                [s.path for s in planned_segments]
            ),
        )

    def plan_segment(
        self,
        scene,
        grounded_segment,
        start,
    ) -> PlannedSegment:
        goal_region = self.compile_goal_region(
            scene,
            grounded_segment.action,
        )

        hard_model = self.compile_hard_constraints(
            scene,
            grounded_segment.constraints,
        )

        soft_model = self.compile_soft_costs(
            scene,
            grounded_segment.constraints,
        )

        path = self.cost_aware_astar(
            start=start,
            goal_region=goal_region,
            hard_model=hard_model,
            soft_model=soft_model,
        )

        return PlannedSegment(
            id=grounded_segment.id,
            path=path,
        )
```

### 4.8 Planner 输出

```python
PlannedSegment:
    id: str
    path: list[GridCell]
    endpoint: GridCell
    target_reached: bool
    hard_constraints_satisfied: bool
    path_length: float
    soft_costs: dict[str, float]
```

完整路径通过拼接 segment path 构成，并避免重复共享 endpoint。

### 4.9 基础实现范围

初始实现支持：

1. 将完整指令解析为 typed DSL；
2. 通过 restricted map API 对完整 task program 进行确定性 DSL grounding；
3. 使用前一 grounded target 作为后续 segment 的 semantic start reference；
4. sequential segment planning；
5. 0.3 m room-conditioned object goal region；
6. occupancy 和 must-avoid mask；
7. must-pass state tracking；
8. room-scoped near、far 和 relative cost；
9. quadratic clearance cost；
10. heading-based smoothness cost；
11. 每个 segment 一次 cost-aware A* search。

暂缓扩展包括：

- joint planning across all segments；
- endpoint lookahead 和 grounding-planning 交替优化；
- multiple path candidates and reranking；
- Pareto 或 length-budgeted search；
- learned cost weights；
- 用于 circle、retrace、S-shape 和其他复杂 path event 的完整 automata。

### 4.10 End-to-End 示例

指令：

> Go to the sink in the bathroom without a laundry hamper while staying near the dining table closest to the starting point. Then go to the bed farthest from the first target while staying far from the sofa in that bedroom.

#### 4.10.1 Parser 输出

```text
task {
    let task_start = start_position

    let bathrooms = rooms("bathroom")
    let valid_bathrooms = bathrooms.where(
        count("laundry_hamper") == 0
    )
    let target_room = unique(valid_bathrooms)
    let target_sink = unique(objects("sink").in(target_room))
    let reference_table = nearest(
        objects("dining_table"),
        to=task_start
    )

    segment s1 {
        from task_start
        to target_sink
        go_to(target_sink)
        prefer near(reference_table)
            scope default_reference_room
    }

    let s2_start = target_sink
    let target_bed = farthest(
        objects("bed"),
        from=target_sink
    )
    let target_bedroom = room_containing(target_bed)
    let reference_sofa = unique(
        objects("sofa").in(target_bedroom)
    )

    segment s2 {
        from s2_start
        to target_bed
        go_to(target_bed)
        prefer far_from(reference_sofa)
            scope default_reference_room
    }
}
```

#### 4.10.2 Segment 1 Grounding

```text
target_room      → room_3
target_sink      → object_17
reference_table  → object_18
room(reference_table) → room_2
```

```python
GroundedSegment(
    id="s1",
    start_reference=task_start,
    action=GoTo(ObjectRef(17)),
    constraints=[
        NearPreference(
            reference=ObjectRef(18),
            scope=InsideRoom(RoomRef(2)),
        )
    ],
)
```

#### 4.10.3 Segment 1 Planning

```text
Goal = cells within 0.3 m of object_17 footprint
       ∩ room_3
       ∩ traversable space

Cost = movement
       + near(object_18) inside room_2
       + clearance
       + smoothness
```

Planner 返回 `path_s1` 以及它的实际最终 cell `endpoint_s1`。这个 endpoint 只在 Planner 内部用于下一段物理起点，不参与前面的 grounding。

#### 4.10.4 Segment 2 Grounding

```text
s2_start         → object_17
target_bed       → object_21
target_bedroom   → room_5
reference_sofa   → object_30
```

```python
GroundedSegment(
    id="s2",
    start_reference=ObjectRef(17),
    action=GoTo(ObjectRef(21)),
    constraints=[
        FarPreference(
            reference=ObjectRef(30),
            scope=InsideRoom(RoomRef(5)),
        )
    ],
)
```

#### 4.10.5 Segment 2 Planning

```text
Goal = cells within 0.3 m of object_21 footprint
       ∩ room_5
       ∩ traversable space

Cost = movement
       + far(object_30) inside room_5
       + clearance
       + smoothness
```

最终 trajectory 为：

```python
full_path = concatenate(path_s1, path_s2)
```

### 4.11 Operator Coverage 要求

实现完成前，应维护一张显式 coverage table：

| DSL meaning | Grounded representation | Planner mechanism |
|---|---|---|
| `go_to(Object)` | `ObjectRef` | 0.3 m room-conditioned goal region |
| `go_to(Room)` | `RoomRef` | room goal region |
| `go_to(Region)` | `AreaRef` | area goal region |
| `near(A)` | object + room scope | near cost field |
| `far_from(A)` | object + room scope | far cost field |
| `closer_to(A,B)` | two objects + room scope | relative cost field |
| `high_clearance()` | whole-segment scope | quadratic clearance cost |
| `smooth_path()` | whole-segment scope | heading transition cost |
| `require pass_between(A,B)` | derived `AreaRef` | must-pass state |
| `forbid pass_between(A,B)` | derived `AreaRef` | forbidden mask |
| `circle(A)` | object + loop specification | event automaton, deferred |
| `retrace(path)` | grounded path reference | path-following state, deferred |

有效 operator inventory 是以下三者的交集：

```text
language-expressible semantics
∩ map-groundable semantics
∩ planner/evaluator-supported semantics.
```

### 4.12 总结

该方法把导航视为结构化编译与规划问题：

```text
Natural language
    → typed ungrounded task program
    → grounded segment-level task representation
    → hard-constrained semantic-cost search
    → trajectory
```

Parser 决定指令是什么意思。Grounder 决定这些表达式指代场景中的哪些 entity 和 region。Planner 决定如何在满足 mandatory requirement 的同时，优化局部 semantic preference、clearance、smoothness 和 path length，并在地图中移动。
