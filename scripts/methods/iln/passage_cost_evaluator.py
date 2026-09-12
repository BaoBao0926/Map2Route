"""ILN-style PassageCostEvaluator for SemPathBench."""

from __future__ import annotations

import json
from typing import Mapping

from scripts.methods.iln.graph_serializer import serialize_graph_for_prompt
from scripts.methods.iln.graph_types import AreaPassageGraph, GroundedStage, PassageCostResult
from scripts.methods.iln.llm_client import GeminiClient, extract_json_object
from scripts.methods.iln.prompts import PASSAGE_COST_SYSTEM_PROMPT, passage_cost_user_prompt


def _parse_door_costs(payload: Mapping[str, object], graph: AreaPassageGraph) -> tuple[dict[str, float], list[str]]:
    diagnostics: list[str] = []
    costs: dict[str, float] = {}
    raw = payload.get("door_costs")
    if not isinstance(raw, Mapping):
        return costs, ["PCE_EMPTY_DOOR_COSTS"]
    for passage_id, value in raw.items():
        key = str(passage_id)
        if key not in graph.passages:
            diagnostics.append(f"UNKNOWN_PASSAGE_COST_KEYS:{key}")
            continue
        if isinstance(value, Mapping):
            raw_cost = value.get("cost")
        else:
            raw_cost = value
        if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
            costs[key] = float(raw_cost)
    if not costs:
        diagnostics.append("PCE_EMPTY_DOOR_COSTS")
    return costs, diagnostics


def _navigation_task(payload: Mapping[str, object], current_area: str, destination_area: str) -> tuple[list[str], list[str]]:
    diagnostics: list[str] = []
    raw = payload.get("navigation_task")
    if isinstance(raw, list) and len(raw) >= 2:
        task = [str(raw[0]), str(raw[1])]
    else:
        task = [current_area, destination_area]
        diagnostics.append("PCE_MISSING_NAVIGATION_TASK")
    if task != [current_area, destination_area]:
        diagnostics.append("PCE_NAVIGATION_TASK_MISMATCH")
        task = [current_area, destination_area]
    return task, diagnostics


def heuristic_passage_costs(
    graph: AreaPassageGraph,
    *,
    current_area: str,
    destination_area: str,
    unknown_passage_cost: float = 10.0,
) -> PassageCostResult:
    return PassageCostResult(
        mode="heuristic",
        navigation_task=[current_area, destination_area],
        door_costs={passage_id: float(unknown_passage_cost) for passage_id in graph.passages},
        raw_response=None,
        diagnostics=["heuristic_unknown_passage_costs"],
    )


def evaluate_passage_costs(
    map_state: Mapping[str, object],
    graph: AreaPassageGraph,
    instruction: Mapping[str, object],
    stage: GroundedStage,
    *,
    current_area_id: int,
    mode: str,
    llm_client: GeminiClient | None = None,
    max_prompt_objects: int = 120,
    passage_history: object = "no experience",
    unknown_passage_cost: float = 10.0,
) -> PassageCostResult:
    current_area = f"room_{current_area_id}"
    destination_area = f"room_{stage.area_id}"
    if mode == "heuristic":
        return heuristic_passage_costs(
            graph,
            current_area=current_area,
            destination_area=destination_area,
            unknown_passage_cost=unknown_passage_cost,
        )
    instruction_text = str(instruction.get("instruction") or "")
    graph_payload = serialize_graph_for_prompt(
        graph,
        map_state=map_state,
        instruction_text=instruction_text,
        max_prompt_objects=max_prompt_objects,
    )
    try:
        if llm_client is None:
            raise RuntimeError("llm_client is required for LLM PassageCostEvaluator.")
        text, cache_info = llm_client.response_text(
            system_prompt=PASSAGE_COST_SYSTEM_PROMPT,
            user_prompt=passage_cost_user_prompt(
                instruction_text=instruction_text,
                graph_payload=graph_payload,
                current_area=current_area,
                destination_area=destination_area,
                passage_history=passage_history,
            ),
            cache_namespace="passage_cost_evaluator_v1",
        )
        payload = extract_json_object(text)
        costs, diagnostics = _parse_door_costs(payload, graph)
        task, task_diagnostics = _navigation_task(payload, current_area, destination_area)
        diagnostics.extend(task_diagnostics)
        diagnostics.append(json.dumps(cache_info, sort_keys=True))
        return PassageCostResult(
            mode="llm",
            navigation_task=task,
            door_costs=costs,
            raw_response=payload,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        if mode == "auto":
            result = heuristic_passage_costs(
                graph,
                current_area=current_area,
                destination_area=destination_area,
                unknown_passage_cost=unknown_passage_cost,
            )
            result.mode = "auto_heuristic_fallback"
            result.diagnostics.append(f"llm_passage_cost_failed: {exc}")
            return result
        raise

