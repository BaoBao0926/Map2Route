"""Cost-aware ILN-style planning over the derived area-passage graph."""

from __future__ import annotations

import heapq
import math
from typing import Iterable, Mapping

from scripts.methods.iln.graph_types import AreaPassageGraph, RouteResult


def _room_ids(values: object) -> set[int]:
    result: set[int] = set()
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        return result
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            result.add(value)
        elif isinstance(value, str):
            text = value.removeprefix("room_")
            try:
                result.add(int(text))
            except ValueError:
                continue
    return result


def _passage_ids(values: object) -> set[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        return set()
    return {str(value) for value in values}


def _edge_metric(graph: AreaPassageGraph, from_room: int, to_room: int, passage_id: str) -> float:
    from_area = graph.areas[from_room]
    to_area = graph.areas[to_room]
    passage = graph.passages[passage_id]
    return (
        math.hypot(from_area.centroid[0] - passage.anchor[0], from_area.centroid[1] - passage.anchor[1])
        + math.hypot(to_area.centroid[0] - passage.anchor[0], to_area.centroid[1] - passage.anchor[1])
    )


def plan_route(
    graph: AreaPassageGraph,
    *,
    start_area: int,
    goal_area: int,
    door_costs: Mapping[str, float],
    adapter_graph_constraints: Mapping[str, object] | None = None,
    event_constraints: Mapping[str, object] | None = None,
    unknown_passage_cost: float = 10.0,
    failed_passages: set[str] | None = None,
) -> RouteResult:
    if start_area == goal_area:
        return RouteResult("same_area_direct_realization", [start_area], [], 0.0)
    adapter_graph_constraints = adapter_graph_constraints or {}
    event_constraints = event_constraints or {}
    forbidden_areas = _room_ids(adapter_graph_constraints.get("forbidden_areas")) | _room_ids(
        event_constraints.get("areas_to_Avoid")
    )
    soft_avoid_areas = _room_ids(adapter_graph_constraints.get("soft_avoid_areas")) | _room_ids(
        event_constraints.get("areas_try_to_Avoid")
    )
    forbidden_passages = _passage_ids(adapter_graph_constraints.get("forbidden_passages")) | (
        failed_passages or set()
    )
    diagnostics: list[str] = []
    if goal_area in forbidden_areas:
        return RouteResult("NO_GRAPH_PATH", [start_area], [], math.inf, ["destination_area_forbidden"])

    heap: list[tuple[float, int, int]] = [(0.0, 0, start_area)]
    counter = 1
    best = {start_area: 0.0}
    previous: dict[int, tuple[int, str]] = {}
    closed: set[int] = set()

    while heap:
        cost, _index, room_id = heapq.heappop(heap)
        if room_id in closed:
            continue
        if room_id == goal_area:
            areas = [room_id]
            passages: list[str] = []
            while areas[-1] in previous:
                prev_room, passage_id = previous[areas[-1]]
                passages.append(passage_id)
                areas.append(prev_room)
            areas.reverse()
            passages.reverse()
            return RouteResult("success", areas, passages, cost, diagnostics)
        closed.add(room_id)
        for passage_id in graph.room_passages.get(room_id, []):
            if passage_id in forbidden_passages:
                continue
            passage = graph.passages.get(passage_id)
            if passage is None:
                continue
            next_room = passage.other_room(room_id)
            if next_room is None or next_room in forbidden_areas or next_room not in graph.areas:
                continue
            passage_cost = float(door_costs.get(passage_id, unknown_passage_cost))
            area_penalty = 10.0 if next_room in soft_avoid_areas else 0.0
            edge_cost = _edge_metric(graph, room_id, next_room, passage_id) + passage_cost + passage_cost + area_penalty
            next_cost = cost + edge_cost
            if next_cost >= best.get(next_room, math.inf):
                continue
            best[next_room] = next_cost
            previous[next_room] = (room_id, passage_id)
            heapq.heappush(heap, (next_cost, counter, next_room))
            counter += 1

    return RouteResult("NO_GRAPH_PATH", [start_area], [], math.inf, diagnostics)

