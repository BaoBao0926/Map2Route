"""LLM high-level grounded navigation planning for SayPlan."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.methods.sayplan.config import DEFAULT_MAX_JSON_RETRIES
from scripts.methods.sayplan.graph.scene_graph_view import SceneGraphView
from scripts.methods.sayplan.llm.client import gemini_response_json
from scripts.methods.sayplan.llm.prompts import planning_prompt
from scripts.methods.sayplan.llm.schemas import (
    HighLevelAction,
    SayPlanLLMResponse,
    SayPlanResponseError,
    parse_plan_items,
    validate_response,
)


@dataclass
class HighLevelPlanResult:
    actions: list[HighLevelAction]
    plan_strings: list[str]
    response: SayPlanLLMResponse | None
    failure_reason: str | None = None


def generate_high_level_plan(
    *,
    view: SceneGraphView,
    instruction: dict[str, object],
    model: str,
    feedback: str = "",
    previous_plan: list[str] | None = None,
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
    verbose: bool = False,
) -> HighLevelPlanResult:
    last_error: str | None = None
    local_feedback = feedback
    for _attempt in range(max_json_retries + 1):
        if verbose:
            print(
                f"[SayPlan][planner] attempt={_attempt + 1} "
                f"task_nodes={len(view.visible_nodes)} feedback={local_feedback!r} "
                f"previous_plan={previous_plan}",
                flush=True,
            )
        system_prompt, user_prompt = planning_prompt(
            instruction=instruction,
            task_graph=view.task_subgraph_json(),
            memory=view.memory_json(),
            feedback=local_feedback,
            previous_plan=previous_plan,
        )
        try:
            payload, raw_text = gemini_response_json(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = validate_response(payload, raw_text)
            actions = parse_plan_items(response.command.plan)
            if not actions or actions[-1].action != "done":
                actions.append(HighLevelAction("done"))
            plan_strings = [action.to_plan_string() for action in actions]
            if verbose:
                print(
                    f"[SayPlan][planner] plan={plan_strings} "
                    f"reasoning={response.reasoning!r}",
                    flush=True,
                )
            return HighLevelPlanResult(actions, plan_strings, response)
        except SayPlanResponseError as exc:
            last_error = exc.reason
            local_feedback = f"Previous response was invalid: {exc.reason}. Return only goto(node_id) and done()."
            if verbose:
                print(
                    f"[SayPlan][planner] invalid_plan reason={exc.reason}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - reported to caller
            last_error = "llm_json_failed"
            local_feedback = f"Previous Gemini call or JSON parse failed: {exc}."
            if verbose:
                print(
                    f"[SayPlan][planner] llm_failed error={type(exc).__name__}: {exc}",
                    flush=True,
                )
    if verbose:
        print(
            f"[SayPlan][planner] failed failure_reason={last_error or 'llm_invalid_plan'}",
            flush=True,
        )
    return HighLevelPlanResult([], [], None, last_error or "llm_invalid_plan")
