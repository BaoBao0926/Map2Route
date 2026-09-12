"""NavigationEventMonitor interface for the ILN SemPathBench adapter."""

from __future__ import annotations

from typing import Mapping

from scripts.methods.iln.graph_types import AreaPassageGraph, RouteResult


def monitor_events(
    graph: AreaPassageGraph,
    route: RouteResult | None = None,
    *,
    mode: str = "empty",
    event_file: object | None = None,
) -> dict[str, object]:
    if mode != "empty" and event_file:
        return {
            "mode": mode,
            "current_status": {"current_events": []},
            "route_approval": {
                "is_Valid": True,
                "areas_to_Avoid": [],
                "areas_try_to_Avoid": [],
            },
            "notes": [
                "explicit event files are reserved for future non-benchmark experiments",
                "benchmark run used no instruction constraint parsing",
            ],
        }
    return {
        "mode": "empty",
        "current_status": {"current_events": []},
        "route_approval": {
            "is_Valid": True,
            "areas_to_Avoid": [],
            "areas_try_to_Avoid": [],
        },
        "notes": ["no external event stream in SemPathBench benchmark input"],
    }


def route_constraints(monitor_payload: Mapping[str, object]) -> dict[str, object]:
    approval = monitor_payload.get("route_approval")
    if not isinstance(approval, Mapping):
        return {"areas_to_Avoid": [], "areas_try_to_Avoid": []}
    return {
        "areas_to_Avoid": approval.get("areas_to_Avoid", []),
        "areas_try_to_Avoid": approval.get("areas_try_to_Avoid", []),
    }

