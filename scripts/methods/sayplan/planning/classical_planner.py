"""Classical grid path completion for SayPlan high-level plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scripts.methods.util.grid_astar import TraversableGrid, astar_path
from scripts.methods.sayplan.graph.scene_graph import SceneGraph
from scripts.methods.sayplan.llm.schemas import HighLevelAction
from scripts.methods.sayplan.utils.geometry import Cell, path_length


@dataclass
class PathSegmentResult:
    from_node: str
    to_node: str
    start: Cell
    goal: Cell | None
    reachable: bool
    path_length: float
    failure_reason: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "start": [self.start[0], self.start[1]],
            "goal": [self.goal[0], self.goal[1]] if self.goal else None,
            "reachable": self.reachable,
            "path_length": self.path_length,
            "failure_reason": self.failure_reason,
        }


@dataclass
class PathCompletionResult:
    trajectory: list[list[int]]
    segments: list[PathSegmentResult]
    success: bool
    failure_reason: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "reachable": self.success,
            "length": path_length(self.trajectory),
            "failure_reason": self.failure_reason,
            "segments": [segment.to_json() for segment in self.segments],
        }


def complete_path(
    *,
    graph: SceneGraph,
    traversable: TraversableGrid,
    start_cell: Cell | None,
    high_level_plan: Sequence[HighLevelAction],
    verbose: bool = False,
) -> PathCompletionResult:
    if start_cell is None:
        if verbose:
            print("[SayPlan][path] failed failure_reason=no_start_cell", flush=True)
        return PathCompletionResult([], [], False, "no_start_cell")

    trajectory = [[start_cell[0], start_cell[1]]]
    current = start_cell
    current_node = "agent_0"
    segments: list[PathSegmentResult] = []
    success = True
    failure_reason: str | None = None
    if verbose:
        print(
            f"[SayPlan][path] start={start_cell} "
            f"high_level_plan={[action.to_plan_string() for action in high_level_plan]}",
            flush=True,
        )

    for action in high_level_plan:
        if action.action == "done":
            if verbose:
                print("[SayPlan][path] done", flush=True)
            break
        if action.action != "goto" or not action.target:
            success = False
            failure_reason = "invalid_action"
            segments.append(
                PathSegmentResult(current_node, action.target or "", current, None, False, 0.0, failure_reason)
            )
            if verbose:
                print(
                    f"[SayPlan][path] invalid_action action={action.to_plan_string()}",
                    flush=True,
                )
            continue
        if not graph.has_node(action.target):
            success = False
            failure_reason = "invalid_node_id"
            segments.append(
                PathSegmentResult(current_node, action.target, current, None, False, 0.0, failure_reason)
            )
            if verbose:
                print(
                    f"[SayPlan][path] invalid_node target={action.target}",
                    flush=True,
                )
            continue
        goal, segment_path = _path_to_node(graph, traversable, current, action.target)
        if goal is None or not segment_path:
            success = False
            failure_reason = "no_path_to_target"
            segments.append(
                PathSegmentResult(current_node, action.target, current, goal, False, 0.0, failure_reason)
            )
            if verbose:
                print(
                    f"[SayPlan][path] no_path from={current_node} start={current} "
                    f"to={action.target} goal={goal}",
                    flush=True,
                )
            continue
        trajectory.extend(segment_path[1:])
        segment_length = path_length(segment_path)
        segments.append(
            PathSegmentResult(current_node, action.target, current, goal, True, segment_length)
        )
        if verbose:
            print(
                f"[SayPlan][path] segment from={current_node} start={current} "
                f"to={action.target} goal={goal} waypoints={len(segment_path)} "
                f"length={segment_length:.3f}",
                flush=True,
            )
        current = goal
        current_node = action.target

    if verbose:
        print(
            f"[SayPlan][path] completed success={success} "
            f"failure_reason={failure_reason} trajectory_length={len(trajectory)}",
            flush=True,
        )
    return PathCompletionResult(trajectory, segments, success, failure_reason)


def _path_to_node(
    graph: SceneGraph,
    traversable: TraversableGrid,
    current: Cell,
    node_id: str,
) -> tuple[Cell | None, list[list[int]] | None]:
    node = graph.get_node(node_id)
    candidates: list[Cell] = []
    if node.type == "object":
        raw = node.metadata.get("approach_cells")
        if isinstance(raw, list):
            candidates = [
                (int(cell[0]), int(cell[1]))
                for cell in raw
                if isinstance(cell, tuple) and len(cell) == 2
            ]
            if not candidates:
                candidates = [
                    (int(cell[0]), int(cell[1]))
                    for cell in raw
                    if isinstance(cell, list) and len(cell) == 2
                ]
    else:
        target = node.metadata.get("navigation_target")
        if isinstance(target, tuple) and len(target) == 2:
            candidates = [(int(target[0]), int(target[1]))]
        elif isinstance(target, list) and len(target) == 2:
            candidates = [(int(target[0]), int(target[1]))]
        elif node.position is not None:
            candidates = [node.position]

    candidates = sorted(
        set(candidates),
        key=lambda cell: math.hypot(cell[0] - current[0], cell[1] - current[1]),
    )
    for candidate in candidates:
        path = astar_path(traversable, current, candidate)
        if path:
            return candidate, path
    return (candidates[0] if candidates else None), None
