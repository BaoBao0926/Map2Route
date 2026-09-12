"""End-to-end ILN SemPathBench planning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from scripts.methods.iln.event_monitor import monitor_events, route_constraints
from scripts.methods.iln.goal_resolver import resolve_stage_goal
from scripts.methods.iln.graph_builder import build_area_passage_graph
from scripts.methods.iln.graph_planner import plan_route
from scripts.methods.iln.graph_serializer import serialize_graph_for_prompt
from scripts.methods.iln.graph_types import Cell
from scripts.methods.iln.grid_realizer import realize_route
from scripts.methods.iln.grounding_adapter import ground_instruction
from scripts.methods.iln.llm_client import GeminiClient
from scripts.methods.iln.passage_cost_evaluator import evaluate_passage_costs
from scripts.methods.util.grid_astar import build_traversable_grid


@dataclass
class ILNRunConfig:
    grounding_mode: str = "heuristic"
    event_mode: str = "empty"
    history_mode: str = "empty"
    passage_extraction: str = "boundary"
    max_prompt_objects: int = 120
    unknown_passage_cost: float = 10.0
    allow_grid_fallback: bool = False
    history_file: object | None = None
    event_file: object | None = None
    graph_cache_root: object | None = None


@dataclass
class ILNPipelineResult:
    trajectory: list[list[int]]
    metadata: dict[str, object]
    success: bool
    failure_reason: str | None = None


def _append_trajectory(base: list[list[int]], segment: list[list[int]]) -> None:
    if not segment:
        return
    if not base:
        base.extend(segment)
    else:
        base.extend(segment[1:])


def _passage_history(config: ILNRunConfig) -> object:
    if config.history_mode == "file" and config.history_file:
        return {"history_file": str(config.history_file)}
    return "no experience"


def run_iln_pipeline(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    *,
    config: ILNRunConfig,
    llm_client: GeminiClient | None = None,
    verbose: bool = False,
) -> ILNPipelineResult:
    traversable = build_traversable_grid(map_state)
    graph = build_area_passage_graph(
        map_state,
        instruction,
        passage_extraction=config.passage_extraction,
        graph_cache_root=config.graph_cache_root,  # type: ignore[arg-type]
    )
    instruction_text = str(instruction.get("instruction") or "")
    graph_prompt = serialize_graph_for_prompt(
        graph,
        map_state=map_state,
        instruction_text=instruction_text,
        max_prompt_objects=config.max_prompt_objects,
    )
    metadata: dict[str, object] = {
        "status": "running",
        "capability_profile": (
            "single area destination; area-passage costs; graph A*; "
            "dense route realization"
        ),
        "benchmark_adaptations": [
            "benchmark_layers_to_area_passage_graph",
            "single_destination_area_selector",
            "passage_route_to_dense_grid_realization",
        ],
        "input_contract": (
            "Uses instruction text, start_pose, occupancy/room/object layers, room_instances, "
            "object_instances, and derived graph/geometry. Does not use hard_constraints, "
            "soft_constraints, instruction.objects, human_expert_trajectory, or evaluator feedback."
        ),
        "area_passage_graph": graph.to_json(),
        "graph_prompt_summary": {
            "area_line_count": len(graph_prompt.get("areas", [])),
            "passage_line_count": len(graph_prompt.get("passages", [])),
            "truncation": graph_prompt.get("truncation"),
        },
        "goal_resolver": {"stages": []},
        "grounding_adapter": {},
        "passage_cost_evaluator": {"stages": []},
        "navigation_event_monitor": {},
        "graph_planner": {"stages": []},
        "grid_realizer": {"stages": []},
        "method_level_deviations": [],
    }

    if graph.start_cell is None:
        metadata["status"] = "START_AREA_UNRESOLVED"
        return ILNPipelineResult([], metadata, False, "START_AREA_UNRESOLVED")
    if graph.start_area is None:
        metadata["status"] = "START_AREA_UNRESOLVED"
        return ILNPipelineResult([[graph.start_cell[0], graph.start_cell[1]]], metadata, False, "START_AREA_UNRESOLVED")

    try:
        grounding = ground_instruction(
            map_state,
            graph,
            instruction,
            mode=config.grounding_mode,
            llm_client=llm_client,
            max_prompt_objects=config.max_prompt_objects,
        )
    except Exception as exc:
        metadata["status"] = "GROUNDING_FAILED"
        metadata["grounding_adapter"] = {"error": str(exc)}
        return ILNPipelineResult([[graph.start_cell[0], graph.start_cell[1]]], metadata, False, "GROUNDING_FAILED")
    metadata["grounding_adapter"] = grounding.to_json()
    if (
        len(grounding.stages) != 1
        or grounding.stages[0].target_type != "area"
        or grounding.stages[0].object_id is not None
        or bool(grounding.adapter_graph_constraints)
    ):
        metadata["status"] = "ILN_CAPABILITY_VIOLATION"
        return ILNPipelineResult([[graph.start_cell[0], graph.start_cell[1]]], metadata, False, "ILN_CAPABILITY_VIOLATION")
    if not grounding.stages:
        metadata["status"] = "GROUNDING_FAILED"
        return ILNPipelineResult([[graph.start_cell[0], graph.start_cell[1]]], metadata, False, "GROUNDING_FAILED")

    trajectory: list[list[int]] = [[graph.start_cell[0], graph.start_cell[1]]]
    current_cell: Cell = graph.start_cell
    current_area = graph.start_area
    failed_passages: set[str] = set()
    monitor_payload = monitor_events(graph, None, mode=config.event_mode, event_file=config.event_file)
    metadata["navigation_event_monitor"] = monitor_payload
    event_constraint_payload = route_constraints(monitor_payload)
    adapter_graph_constraints: dict[str, object] = {}

    for stage_index, stage in enumerate(grounding.stages, start=1):
        if verbose:
            print(
                f"[iln:pipeline] stage={stage.stage_id} current_area=room_{current_area} "
                f"target={stage.to_json()}",
                flush=True,
            )
        goal = resolve_stage_goal(
            map_state,
            traversable,
            graph,
            stage,
            current_cell=current_cell,
            allow_grid_fallback=config.allow_grid_fallback,
        )
        metadata["goal_resolver"]["stages"].append(goal.to_json())  # type: ignore[index]
        if goal.endpoint is None:
            metadata["status"] = goal.status
            return ILNPipelineResult(trajectory, metadata, False, goal.status)

        pce_mode = config.grounding_mode
        try:
            passage_costs = evaluate_passage_costs(
                map_state,
                graph,
                instruction,
                stage,
                current_area_id=current_area,
                mode=pce_mode,
                llm_client=llm_client,
                max_prompt_objects=config.max_prompt_objects,
                passage_history=_passage_history(config),
                unknown_passage_cost=config.unknown_passage_cost,
            )
        except Exception as exc:
            metadata["status"] = "LLM_OUTPUT_INVALID"
            metadata["passage_cost_evaluator"]["stages"].append(  # type: ignore[index]
                {"stage_id": stage.stage_id, "error": str(exc)}
            )
            return ILNPipelineResult(trajectory, metadata, False, "LLM_OUTPUT_INVALID")
        metadata["passage_cost_evaluator"]["stages"].append(  # type: ignore[index]
            {"stage_id": stage.stage_id, **passage_costs.to_json()}
        )

        route = plan_route(
            graph,
            start_area=current_area,
            goal_area=stage.area_id,
            door_costs=passage_costs.door_costs,
            adapter_graph_constraints=adapter_graph_constraints,
            event_constraints=event_constraint_payload,
            unknown_passage_cost=config.unknown_passage_cost,
            failed_passages=failed_passages,
        )
        route_payload = route.to_json(graph)
        route_payload["stage_id"] = stage.stage_id
        metadata["graph_planner"]["stages"].append(route_payload)  # type: ignore[index]
        if route.status not in {"success", "same_area_direct_realization"}:
            metadata["status"] = route.status
            return ILNPipelineResult(trajectory, metadata, False, route.status)

        realization = realize_route(
            graph,
            traversable,
            start_cell=current_cell,
            endpoint=goal.endpoint,
            route=route,
        )
        realization_payload = realization.to_json()
        realization_payload["stage_id"] = stage.stage_id
        metadata["grid_realizer"]["stages"].append(realization_payload)  # type: ignore[index]
        if realization.status != "success" and realization.failed_passage:
            failed_passages.add(realization.failed_passage)
            retry_route = plan_route(
                graph,
                start_area=current_area,
                goal_area=stage.area_id,
                door_costs=passage_costs.door_costs,
                adapter_graph_constraints=adapter_graph_constraints,
                event_constraints=event_constraint_payload,
                unknown_passage_cost=config.unknown_passage_cost,
                failed_passages=failed_passages,
            )
            retry_payload = retry_route.to_json(graph)
            retry_payload["stage_id"] = stage.stage_id
            retry_payload["retry_after_failed_passage"] = realization.failed_passage
            metadata["graph_planner"]["stages"].append(retry_payload)  # type: ignore[index]
            if retry_route.status in {"success", "same_area_direct_realization"}:
                realization = realize_route(
                    graph,
                    traversable,
                    start_cell=current_cell,
                    endpoint=goal.endpoint,
                    route=retry_route,
                )
                retry_realization_payload = realization.to_json()
                retry_realization_payload["stage_id"] = stage.stage_id
                retry_realization_payload["retry"] = True
                metadata["grid_realizer"]["stages"].append(retry_realization_payload)  # type: ignore[index]
                route = retry_route
        if realization.status != "success":
            metadata["status"] = realization.status
            return ILNPipelineResult(trajectory, metadata, False, realization.status)

        _append_trajectory(trajectory, realization.trajectory)
        current_cell = (trajectory[-1][0], trajectory[-1][1])
        current_area = stage.area_id

    metadata["status"] = "success"
    return ILNPipelineResult(trajectory, metadata, True)

