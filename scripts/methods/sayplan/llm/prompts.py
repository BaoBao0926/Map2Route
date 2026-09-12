"""Prompt builders following the SayPlan paper's prompt structure."""

from __future__ import annotations

import json
from collections.abc import Mapping


SYSTEM_PROMPT = """You are an excellent graph planning agent. You must return only JSON parseable by Python json.loads."""


def instruction_context(instruction: Mapping[str, object]) -> dict[str, object]:
    """Expose only free-form language to SayPlan LLM calls.

    Benchmark annotations are evaluation data, not inference inputs. Keeping
    this projection narrow prevents a complete record from leaking GT fields.
    """
    return {
        "instruction": instruction.get("instruction", ""),
    }


def semantic_search_prompt(
    *,
    instruction: Mapping[str, object],
    visible_graph: Mapping[str, object],
    memory: Mapping[str, object],
    feedback: str = "",
) -> tuple[str, str]:
    user_prompt = f"""
Agent Role:
You are an excellent graph planning agent. Given a graph representation of an
environment, you can explore the graph by expanding nodes to find the items of
interest. You can then use this graph to generate a step-by-step navigation plan
that the agent can follow to solve a given instruction.

Environment Functions:
goto(<node>): Move the agent to any room, object approach node, or pose node.
done(): Call when the navigation task is completed.

Environment State:
located_at(<node>): The agent is currently located at a node.
reachable(<node>): The node can be reached by the classical path planner.
contains(<room>, <object>): The object is located in the room.
adjacent_to(<node>, <node>): Two rooms or poses are topologically connected.

Environment API:
expand_node(<node>): Reveal objects or lower-level nodes connected to a room/scene node.
contract_node(<node>): Hide lower-level nodes, reducing graph size for memory constraints.
verify_plan(): Verify generated plan in the scene graph navigation environment.

Rules:
- Use only node IDs in the visible graph.
- Expand a node if its hidden children may contain task-relevant entities.
- Contract a node if its visible children are irrelevant.
- Do not invent nodes.
- Switch to planning mode only when enough grounded entities are visible.
- Do not output grid coordinates or dense paths.

Output Response Format:
{{
  "reasoning": "brief reason for the next command",
  "mode": "exploring" OR "planning",
  "command": {{
    "command_name": "expand_node" OR "contract_node" OR "none",
    "node_name": "node_id or null",
    "plan": null
  }}
}}

Instruction Context:
{json.dumps(instruction_context(instruction), ensure_ascii=True)}

3D Scene Graph:
{json.dumps(visible_graph, ensure_ascii=True)}

Memory:
{json.dumps(memory, ensure_ascii=True)}

Feedback:
{feedback}
""".strip()
    return SYSTEM_PROMPT, user_prompt


def planning_prompt(
    *,
    instruction: Mapping[str, object],
    task_graph: Mapping[str, object],
    memory: Mapping[str, object],
    feedback: str = "",
    previous_plan: list[str] | None = None,
) -> tuple[str, str]:
    previous = ""
    if previous_plan:
        previous = "\nPrevious Plan:\n" + json.dumps(previous_plan, ensure_ascii=True)
    user_prompt = f"""
Agent Role:
You are an excellent graph planning agent. Given a graph representation of an
environment, generate a high-level grounded navigation plan that the agent can
follow to solve the instruction.

Environment Functions:
goto(<node>): Move the agent to any room, object approach node, or pose node.
done(): Call when the navigation task is completed.

Environment API:
verify_plan(): Verify generated plan in the scene graph navigation environment.

Rules:
- Generate a short plan using only node IDs present in the task-relevant graph.
- Do not output grid coordinates.
- Do not output a dense path.
- Do not invent node IDs.
- A classical path planner will connect the selected targets.
- End the plan with done().

Output Response Format:
{{
  "reasoning": "brief reason for the selected high-level plan",
  "mode": "planning",
  "command": {{
    "command_name": "none",
    "node_name": null,
    "plan": [
      "goto(node_id)",
      "done()"
    ]
  }}
}}

Instruction Context:
{json.dumps(instruction_context(instruction), ensure_ascii=True)}

3D Scene Graph:
{json.dumps(task_graph, ensure_ascii=True)}

Memory:
{json.dumps(memory, ensure_ascii=True)}
{previous}

Feedback:
{feedback}
""".strip()
    return SYSTEM_PROMPT, user_prompt

