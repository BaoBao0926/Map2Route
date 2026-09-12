"""Experimental hierarchy-aware planner for OSG-LLM.

This is not a full AMRA* implementation. It makes room/object/floor hierarchy
operational by searching a high-level room/object skeleton, then refining it to
an executable grid path with AP-level A*.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from collections.abc import Sequence

from scripts.methods.osgllm.planning.sequential_fallback import sequential_ap_astar
from scripts.methods.osgllm.scene_graph.graph_types import Cell, SceneGraph
from scripts.methods.osgllm.scene_graph.propositions import PropositionModel
from scripts.methods.util.grid_astar import TraversableGrid


@dataclass(frozen=True)
class HierarchicalPlanningResult:
    status: str
    trajectory: list[list[int]]
    details: dict[str, object]


def _room_for_cell(graph: SceneGraph, cell: Cell) -> str | None:
    for room in graph.by_kind("room"):
        if cell in set(room.cells):
            return room.node_id
    return None


def _room_for_ap(graph: SceneGraph, ap_name: str) -> str | None:
    node_id = graph.ap_inventory().get(ap_name)
    if node_id is None:
        return None
    region = graph.regions.get(node_id)
    if region is None:
        return None
    if region.kind == "room":
        return region.node_id
    if region.kind == "object":
        return region.parent_id if isinstance(region.parent_id, str) and region.parent_id.startswith("room_") else None
    return None


def _bfs_rooms(graph: SceneGraph, start_room: str | None, goal_room: str | None) -> list[str]:
    if start_room is None or goal_room is None:
        return []
    if start_room == goal_room:
        return [start_room]
    queue: deque[str] = deque([start_room])
    parent: dict[str, str | None] = {start_room: None}
    while queue:
        current = queue.popleft()
        for neighbor in graph.room_adjacency.get(current, ()):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor == goal_room:
                path = [neighbor]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])  # type: ignore[arg-type]
                path.reverse()
                return path
            queue.append(neighbor)
    return []


def _augment_goals_with_room_path(
    graph: SceneGraph,
    start: Cell,
    goals: Sequence[str],
    *,
    preferred_nodes: Sequence[str] = (),
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    current_room = _room_for_cell(graph, start)
    augmented: list[str] = []
    expansions: list[dict[str, object]] = []
    preferred_rooms = tuple(
        node_id
        for node_id in preferred_nodes
        if node_id.startswith("room_") and node_id in graph.regions
    )
    for goal in goals:
        goal_room = _room_for_ap(graph, goal)
        path = _bfs_rooms_with_preference(graph, current_room, goal_room, preferred_rooms)
        expansions.append(
            {
                "from_room": current_room,
                "goal_ap": goal,
                "goal_room": goal_room,
                "room_path": path,
                "expanded_room_nodes": len(path),
            }
        )
        for room_id in path[1:]:
            ap = graph.regions[room_id].ap
            if ap not in augmented:
                augmented.append(ap)
        if goal not in augmented:
            augmented.append(goal)
        current_room = goal_room or current_room
    return tuple(dict.fromkeys(augmented)), expansions


def _bfs_rooms_with_preference(
    graph: SceneGraph,
    start_room: str | None,
    goal_room: str | None,
    preferred_rooms: Sequence[str],
) -> list[str]:
    if not preferred_rooms:
        return _bfs_rooms(graph, start_room, goal_room)
    if start_room is None or goal_room is None:
        return []
    if start_room == goal_room:
        return [start_room]
    preferred_rank = {room: index for index, room in enumerate(preferred_rooms)}
    queue: deque[str] = deque([start_room])
    parent: dict[str, str | None] = {start_room: None}
    while queue:
        current = queue.popleft()
        neighbors = sorted(
            graph.room_adjacency.get(current, ()),
            key=lambda room: preferred_rank.get(room, 10_000),
        )
        for neighbor in neighbors:
            if neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor == goal_room:
                path = [neighbor]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])  # type: ignore[arg-type]
                path.reverse()
                return path
            queue.append(neighbor)
    return []


def hierarchy_guided_plan(
    graph: SceneGraph,
    traversable: TraversableGrid,
    proposition_model: PropositionModel,
    start: Cell,
    goals: Sequence[str],
    *,
    preferred_nodes: Sequence[str] = (),
) -> HierarchicalPlanningResult:
    augmented_goals, hierarchy_expansions = _augment_goals_with_room_path(
        graph,
        start,
        goals,
        preferred_nodes=preferred_nodes,
    )
    fallback = sequential_ap_astar(
        traversable,
        proposition_model,
        start,
        augmented_goals,
    )
    details = {
        "planner_type": "experimental_hierarchy_guided_refinement",
        "status": fallback.details.get("status"),
        "original_goals": list(goals),
        "augmented_goals": list(augmented_goals),
        "hierarchy_levels": ["room", "object", "floor"],
        "llm_preferred_nodes_used": list(preferred_nodes),
        "expanded_states": {
            "room": sum(int(item.get("expanded_room_nodes", 0)) for item in hierarchy_expansions),
            "object": sum(1 for goal in goals if goal.startswith("reach(object_")),
            "floor": 1,
        },
        "hierarchy_expansions": hierarchy_expansions,
        "refinement": fallback.details,
    }
    return HierarchicalPlanningResult(fallback.status, fallback.trajectory, details)
