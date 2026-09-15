"""ToolCall grounding plus the fixed Grounding2Route planner handoff."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

from scripts.methods.grounding2route.constraint_evaluation import apply_scope_ablation
from scripts.methods.grounding2route.grounding.scene import SceneMap
from scripts.methods.grounding2route.grounding.tool_call import ToolCallGrounder
from scripts.methods.grounding2route.ir import GroundedProgram, PlannedProgram
from scripts.methods.grounding2route.llm_client import GeminiClient
from scripts.methods.grounding2route.parse import scene_category_summary, supported_categories
from scripts.methods.grounding2route.planner import plan_grounded_program


@dataclass(frozen=True)
class ToolPipelineRun:
    trajectory: list[list[int]]
    grounded: GroundedProgram | None
    planned: PlannedProgram | None
    success: bool
    failure_reason: str | None


def run_tool_call_pipeline(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    *,
    config: object,
    llm_client: GeminiClient | None,
    metadata: dict[str, object],
    steps: dict[str, object],
) -> ToolPipelineRun:
    scene = SceneMap(map_state, instruction)
    steps["parse"] = {
        "status": "skipped",
        "mode": "tool_call",
        "reason": "ToolCall grounds the free-form instruction interactively without GT decomposition or generated code.",
    }
    steps["parse_attempts"] = [dict(steps["parse"])]
    metadata["parser"] = dict(steps["parse"])
    if llm_client is None:
        return _failure(map_state, instruction, metadata, steps, scene, None, "LLM_UNAVAILABLE")

    entities, rooms = supported_categories(map_state)
    grounder = ToolCallGrounder(
        llm_client,
        max_tool_calls=int(getattr(config, "max_tool_calls")),
        max_llm_turns=int(getattr(config, "max_tool_llm_turns")),
        max_failed_query_attempts=min(3, max(1, int(getattr(config, "max_tool_query_retries")))),
    )
    result = grounder.ground(
        str(instruction.get("instruction") or ""),
        scene,
        instruction.get("start_pose"),
        entity_categories=entities,
        room_categories=rooms,
        scene_summary=scene_category_summary(map_state),
        max_repairs=(
            max(0, int(getattr(config, "max_grounding_repairs")))
            if getattr(config, "execution_repair") != "off"
            else 0
        ),
    )
    steps["tool_call_trace"] = [dict(item) for item in result.tool_trace]
    steps["tool_call_candidates"] = [dict(item) for item in result.candidate_attempts]
    steps["tool_call_statistics"] = dict(result.statistics)
    if result.grounded is not None:
        steps["grounding"] = result.grounded.to_json()
    metadata["grounding"] = {
        "status": "success" if result.failure_reason is None else "failed",
        "mode": "native_function_calling",
        "failure_reason": result.failure_reason,
        "verification": dict(result.verification),
        "statistics": dict(result.statistics),
        "segment_count": len(result.grounded.segments) if result.grounded is not None else 0,
    }
    metadata["grounding_efficiency"] = dict(result.statistics)
    metadata["tool_call_config"] = {
        "repair_budget": max(0, int(getattr(config, "max_grounding_repairs"))),
        "max_tool_calls": int(getattr(config, "max_tool_calls")),
        "max_llm_turns": int(getattr(config, "max_tool_llm_turns")),
        "max_failed_query_attempts": min(3, max(1, int(getattr(config, "max_tool_query_retries")))),
        "generated_code": False,
    }
    grounded = result.grounded
    if result.failure_reason is not None or grounded is None:
        return _failure(map_state, instruction, metadata, steps, scene, grounded, result.failure_reason or "GROUNDING_FAILED")

    scope_ablation = str(getattr(config, "scope_ablation"))
    if scope_ablation != "full":
        grounded = apply_scope_ablation(grounded, scope_ablation)
        steps["grounded_program_ablation"] = {"mode": scope_ablation, "grounded": grounded.to_json()}
        steps["grounding"] = grounded.to_json()
        metadata["grounded_program_ablation"] = {"mode": scope_ablation}

    try:
        planner_started = time.perf_counter()
        planned = plan_grounded_program(
            scene,
            grounded,
            max_expansions=int(getattr(config, "max_expansions")),
            planner_mode=str(getattr(config, "planner_mode")),
            heuristic_weight=float(getattr(config, "planner_heuristic_weight")),
            relative_cost_mode=str(getattr(config, "relative_cost_mode")),
            relative_weight=float(getattr(config, "relative_weight")),
            soft_weight_scale=(0.0 if scope_ablation == "no_soft_constraints" else float(getattr(config, "soft_weight_scale"))),
            include_clearance=scope_ablation != "no_soft_constraints",
            global_min_path_improvement_m=getattr(config, "global_min_path_improvement_m"),
        )
        wall = time.perf_counter() - planner_started
        planner_payload = planned.to_json()
        planner_payload["planning_wall_seconds"] = wall
        steps["planner"] = planner_payload
        metadata["planner"] = {
            "status": planned.status,
            "failure_reason": planned.failure_reason,
            "mode": getattr(config, "planner_mode"),
            "heuristic_weight": getattr(config, "planner_heuristic_weight"),
            "details": dict(planned.details),
            "segment_count": len(planned.segments),
            "trajectory_length": len(planned.trajectory),
            "planning_wall_seconds": wall,
            "input": "verified_grounded_program",
        }
        metadata["grounding_efficiency"]["planning_success"] = planned.status == "success"  # type: ignore[index]
    except Exception as exc:
        metadata["grounding_efficiency"]["planning_success"] = False  # type: ignore[index]
        metadata["status"] = "PLANNING_FAILED"
        metadata["planner"] = {"status": "failed", "error": str(exc)}
        steps["planner"] = dict(metadata["planner"])  # type: ignore[arg-type]
        return ToolPipelineRun(_start_only(map_state, instruction), grounded, None, False, "PLANNING_FAILED")

    trajectory = planned.trajectory or _start_only(map_state, instruction)
    metadata["status"] = "success" if planned.status == "success" else str(planned.failure_reason or planned.status)
    return ToolPipelineRun(trajectory, grounded, planned, planned.status == "success", None if planned.status == "success" else planned.failure_reason)


def _failure(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    metadata: dict[str, object],
    steps: dict[str, object],
    scene: SceneMap,
    grounded: GroundedProgram | None,
    reason: str,
) -> ToolPipelineRun:
    del scene
    metadata["status"] = "GROUNDING_FAILED"
    grounding = metadata.setdefault("grounding", {})
    if isinstance(grounding, dict):
        grounding.setdefault("status", "failed")
        grounding.setdefault("mode", "native_function_calling")
        grounding["failure_reason"] = reason
    steps.setdefault("grounding", dict(grounding) if isinstance(grounding, Mapping) else {"failure_reason": reason})
    return ToolPipelineRun(_start_only(map_state, instruction), grounded, None, False, "GROUNDING_FAILED")


def _start_only(map_state: Mapping[str, object], instruction: Mapping[str, object]) -> list[list[int]]:
    start = instruction.get("start_pose")
    if isinstance(start, Mapping) and isinstance(start.get("row"), int) and isinstance(start.get("col"), int):
        return [[int(start["row"]), int(start["col"])]]
    center = max(0, int(map_state.get("grid_size", 1)) // 2)
    return [[center, center]]
