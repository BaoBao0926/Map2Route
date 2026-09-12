"""Navigation scene-graph verifier for SayPlan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.methods.sayplan.graph.scene_graph_view import SceneGraphView
from scripts.methods.sayplan.llm.schemas import HighLevelAction
from scripts.methods.sayplan.planning.classical_planner import PathCompletionResult


@dataclass
class VerificationResult:
    success: bool
    feedback: str
    failures: list[dict[str, object]]


def verify_plan(
    *,
    task_graph_view: SceneGraphView,
    high_level_plan: Sequence[HighLevelAction],
    path_result: PathCompletionResult,
) -> VerificationResult:
    failures: list[dict[str, object]] = []
    graph = task_graph_view.full_graph
    visible = task_graph_view.visible_nodes

    for action in high_level_plan:
        if action.action == "done":
            continue
        if action.action != "goto":
            failures.append(
                {
                    "kind": "invalid_action",
                    "feedback": f"Action {action.action} is invalid for this navigation-only adaptation.",
                }
            )
            continue
        if not action.target or not graph.has_node(action.target):
            failures.append(
                {
                    "kind": "invalid_node",
                    "feedback": f"Target node {action.target} does not exist.",
                }
            )
            continue
        if action.target not in visible:
            failures.append(
                {
                    "kind": "target_not_visible",
                    "feedback": f"Target {action.target} is not visible in the task-relevant graph.",
                }
            )
        node = graph.get_node(action.target)
        if node.type == "object" and not node.metadata.get("approach_cells"):
            failures.append(
                {
                    "kind": "no_approach_cell",
                    "feedback": f"No traversable approach cell exists near {action.target}.",
                }
            )

    if not path_result.success:
        failures.append(
            {
                "kind": path_result.failure_reason or "path_failed",
                "feedback": path_result.failure_reason or "The path planner failed.",
            }
        )

    if failures:
        feedback = " ".join(str(item["feedback"]) for item in failures if item.get("feedback"))
        return VerificationResult(False, feedback, failures)
    return VerificationResult(True, "success", [])

