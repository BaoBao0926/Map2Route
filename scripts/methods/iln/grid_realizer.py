"""Realize selected ILN passage routes as dense grid trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.methods.iln.graph_types import AreaPassageGraph, Cell, RouteResult
from scripts.methods.util.grid_astar import TraversableGrid, astar_path, is_traversable


@dataclass
class RealizationResult:
    trajectory: list[list[int]]
    status: str
    details: list[dict[str, object]] = field(default_factory=list)
    failed_passage: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "trajectory_length": len(self.trajectory),
            "failed_passage": self.failed_passage,
            "details": self.details,
        }


def _append_segment(
    trajectory: list[list[int]],
    segment: list[list[int]],
) -> None:
    if not trajectory:
        trajectory.extend(segment)
    else:
        trajectory.extend(segment[1:])


def _valid_trajectory(traversable: TraversableGrid, trajectory: list[list[int]]) -> bool:
    if not trajectory:
        return False
    for point in trajectory:
        if len(point) != 2 or not is_traversable(traversable, int(point[0]), int(point[1])):
            return False
    return True


def realize_route(
    graph: AreaPassageGraph,
    traversable: TraversableGrid,
    *,
    start_cell: Cell,
    endpoint: Cell,
    route: RouteResult,
) -> RealizationResult:
    waypoints: list[tuple[str, Cell, str | None]] = []
    for passage_id in route.passage_sequence:
        passage = graph.passages.get(passage_id)
        if passage is None:
            return RealizationResult([[start_cell[0], start_cell[1]]], "GRID_REALIZATION_FAILED")
        waypoints.append(("passage_anchor", passage.anchor, passage_id))
    waypoints.append(("endpoint", endpoint, None))

    trajectory: list[list[int]] = [[start_cell[0], start_cell[1]]]
    current = start_cell
    details: list[dict[str, object]] = []
    for kind, waypoint, passage_id in waypoints:
        segment = astar_path(traversable, current, waypoint)
        detail = {
            "kind": kind,
            "from": [current[0], current[1]],
            "to": [waypoint[0], waypoint[1]],
            "passage_id": passage_id,
        }
        if not segment:
            detail["status"] = "failed_no_path"
            details.append(detail)
            return RealizationResult(
                trajectory,
                "GRID_REALIZATION_FAILED",
                details,
                failed_passage=passage_id,
            )
        detail["status"] = "connected"
        detail["waypoint_count"] = len(segment)
        details.append(detail)
        _append_segment(trajectory, segment)
        current = waypoint

    if not _valid_trajectory(traversable, trajectory):
        return RealizationResult(trajectory, "GRID_REALIZATION_FAILED", details)
    return RealizationResult(trajectory, "success", details)

