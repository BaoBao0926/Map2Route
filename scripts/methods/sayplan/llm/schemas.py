"""Validation and parsing for SayPlan LLM responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


class SayPlanResponseError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class SayPlanCommand:
    command_name: str
    node_name: str | None
    plan: list[str] | None


@dataclass
class SayPlanLLMResponse:
    reasoning: str
    mode: str
    command: SayPlanCommand
    raw_text: str
    raw_json: dict[str, object]


@dataclass
class HighLevelAction:
    action: str
    target: str | None = None

    def to_plan_string(self) -> str:
        if self.action == "goto" and self.target:
            return f"goto({self.target})"
        return "done()"


_GOTO_RE = re.compile(r"^goto\(([^()]+)\)$")


def validate_response(
    payload: dict[str, object],
    raw_text: str,
) -> SayPlanLLMResponse:
    reasoning = payload.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    mode = payload.get("mode")
    if mode not in {"exploring", "planning"}:
        raise SayPlanResponseError("invalid_mode", f"Invalid SayPlan mode: {mode!r}")
    command = payload.get("command")
    if not isinstance(command, Mapping):
        raise SayPlanResponseError("missing_command", "Missing command object.")
    command_name = command.get("command_name")
    if command_name not in {"expand_node", "contract_node", "none"}:
        raise SayPlanResponseError(
            "invalid_command_name",
            f"Invalid command_name: {command_name!r}",
        )
    node_name = command.get("node_name")
    if node_name is not None and not isinstance(node_name, str):
        raise SayPlanResponseError("invalid_node_id", "node_name must be a string or null.")
    raw_plan = command.get("plan")
    plan: list[str] | None = None
    if raw_plan is not None:
        if not isinstance(raw_plan, list) or not all(isinstance(item, str) for item in raw_plan):
            raise SayPlanResponseError("invalid_plan_item", "plan must be a list of strings.")
        plan = [item.strip() for item in raw_plan if item.strip()]
    return SayPlanLLMResponse(
        reasoning=reasoning,
        mode=mode,
        command=SayPlanCommand(
            command_name=str(command_name),
            node_name=node_name,
            plan=plan,
        ),
        raw_text=raw_text,
        raw_json=payload,
    )


def parse_plan_items(items: list[str] | None) -> list[HighLevelAction]:
    if not items:
        return []
    actions: list[HighLevelAction] = []
    for item in items:
        stripped = item.strip()
        if stripped == "done()":
            actions.append(HighLevelAction("done"))
            continue
        match = _GOTO_RE.match(stripped)
        if not match:
            raise SayPlanResponseError("invalid_plan_item", f"Invalid plan item: {item}")
        actions.append(HighLevelAction("goto", match.group(1).strip()))
    return actions

