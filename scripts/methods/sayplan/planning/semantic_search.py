"""SayPlan semantic search over collapsed scene graphs."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.methods.sayplan.config import DEFAULT_MAX_JSON_RETRIES
from scripts.methods.sayplan.graph.scene_graph import SceneGraph
from scripts.methods.sayplan.graph.scene_graph_view import SceneGraphView
from scripts.methods.sayplan.llm.client import gemini_response_json
from scripts.methods.sayplan.llm.prompts import semantic_search_prompt
from scripts.methods.sayplan.llm.schemas import SayPlanResponseError, validate_response


@dataclass
class SemanticSearchResult:
    view: SceneGraphView
    trace: list[dict[str, object]]
    failure_reason: str | None = None


def run_semantic_search(
    *,
    graph: SceneGraph,
    instruction: dict[str, object],
    model: str,
    max_search_steps: int,
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
    verbose: bool = False,
) -> SemanticSearchResult:
    view = SceneGraphView.collapse(graph)
    trace: list[dict[str, object]] = []
    feedback = ""
    failure_reason: str | None = None

    for step in range(1, max_search_steps + 1):
        response = None
        last_error: str | None = None
        for _attempt in range(max_json_retries + 1):
            if verbose:
                print(
                    f"[SayPlan][semantic] step={step} attempt={_attempt + 1} "
                    f"visible_nodes={len(view.visible_nodes)} feedback={feedback!r}",
                    flush=True,
                )
            system_prompt, user_prompt = semantic_search_prompt(
                instruction=instruction,
                visible_graph=view.visible_json(),
                memory=view.memory_json(),
                feedback=feedback,
            )
            try:
                payload, raw_text = gemini_response_json(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                response = validate_response(payload, raw_text)
                if verbose:
                    print(
                        "[SayPlan][semantic] llm_response "
                        f"mode={response.mode} "
                        f"command={response.command.command_name} "
                        f"node={response.command.node_name!r} "
                        f"reasoning={response.reasoning!r}",
                        flush=True,
                    )
                break
            except SayPlanResponseError as exc:
                last_error = exc.reason
                feedback = f"Previous response was invalid: {exc.reason}. Return the required JSON schema."
                if verbose:
                    print(
                        f"[SayPlan][semantic] invalid_json reason={exc.reason}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001 - recorded as LLM failure
                last_error = "llm_json_failed"
                feedback = f"Previous Gemini call or JSON parse failed: {exc}."
                if verbose:
                    print(
                        f"[SayPlan][semantic] llm_failed error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
        if response is None:
            failure_reason = last_error or "llm_json_failed"
            trace.append({"step": step, "status": "failed", "failure_reason": failure_reason})
            if verbose:
                print(
                    f"[SayPlan][semantic] failed failure_reason={failure_reason}",
                    flush=True,
                )
            break

        command = response.command
        trace_item: dict[str, object] = {
            "step": step,
            "mode": response.mode,
            "command": {
                "command_name": command.command_name,
                "node_name": command.node_name,
                "plan": command.plan,
            },
            "reasoning": response.reasoning,
            "visible_node_count": len(view.visible_nodes),
            "raw_llm_response": response.raw_json,
        }

        if response.mode == "planning" or command.command_name == "none":
            trace_item["status"] = "terminated"
            trace.append(trace_item)
            if verbose:
                print(
                    f"[SayPlan][semantic] terminated step={step} "
                    f"visible_nodes={len(view.visible_nodes)}",
                    flush=True,
                )
            return SemanticSearchResult(view=view, trace=trace, failure_reason=failure_reason)

        node_id = command.node_name
        if not node_id or not graph.has_node(node_id):
            feedback = f"Node {node_id!r} does not exist in the current scene graph. Use only visible node IDs."
            trace_item["status"] = "invalid_node"
            trace.append(trace_item)
            failure_reason = "llm_invalid_command"
            if verbose:
                print(
                    f"[SayPlan][semantic] invalid_node node={node_id!r}",
                    flush=True,
                )
            continue
        if node_id not in view.visible_nodes:
            feedback = f"Node {node_id!r} is not currently visible. Use only node IDs from the visible graph."
            trace_item["status"] = "hidden_node"
            trace.append(trace_item)
            failure_reason = "llm_invalid_command"
            if verbose:
                print(
                    f"[SayPlan][semantic] hidden_node node={node_id!r}",
                    flush=True,
                )
            continue

        try:
            if command.command_name == "expand_node":
                view.expand(node_id)
            elif command.command_name == "contract_node":
                view.contract(node_id)
            else:
                feedback = f"Unsupported command {command.command_name}."
                trace_item["status"] = "invalid_command"
                failure_reason = "llm_invalid_command"
                trace.append(trace_item)
                if verbose:
                    print(
                        f"[SayPlan][semantic] invalid_command command={command.command_name!r}",
                        flush=True,
                    )
                continue
        except KeyError:
            feedback = f"Node {node_id!r} does not exist in the current scene graph."
            trace_item["status"] = "invalid_node"
            failure_reason = "llm_invalid_command"
            trace.append(trace_item)
            if verbose:
                print(
                    f"[SayPlan][semantic] invalid_node node={node_id!r}",
                    flush=True,
                )
            continue

        trace_item["status"] = "applied"
        trace_item["visible_node_count_after"] = len(view.visible_nodes)
        trace.append(trace_item)
        if verbose:
            print(
                f"[SayPlan][semantic] applied command={command.command_name} "
                f"node={node_id} visible_nodes={len(view.visible_nodes)}",
                flush=True,
            )
        feedback = ""

    if failure_reason is None:
        failure_reason = "semantic_search_max_steps"
    if verbose:
        print(
            f"[SayPlan][semantic] exhausted failure_reason={failure_reason} "
            f"visible_nodes={len(view.visible_nodes)}",
            flush=True,
        )
    return SemanticSearchResult(view=view, trace=trace, failure_reason=failure_reason)
