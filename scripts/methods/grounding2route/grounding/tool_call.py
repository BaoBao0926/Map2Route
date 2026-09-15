"""Native function-calling grounder over the existing Grounding2Route runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Mapping, Sequence

from scripts.methods.grounding2route.grounding.code import GroundingRuntime, verify_grounded_program
from scripts.methods.grounding2route.grounding.grounder import GroundingFailure
from scripts.methods.grounding2route.grounding.scene import SceneMap
from scripts.methods.grounding2route.grounding.tool_client import tool_turn
from scripts.methods.grounding2route.grounding.semantic_api import (
    PREDICATE_NAMES,
    SEMANTIC_API_NAMES,
    gemini_function_declarations,
)
from scripts.methods.grounding2route.ir import GPRef, GroundedProgram
from scripts.methods.grounding2route.llm_client import GeminiClient


TOOL_CALL_SYSTEM_PROMPT = """You are a semantic route grounding agent.

Convert the free-form route instruction into Grounding2Route's existing RouteIR by
calling the provided read-only semantic-map functions. You may use multiple
turns and compose prior tool results through their opaque handles.

Rules:
1. Resolve every map-dependent room, entity, relation, destination, constraint,
   temporal scope, and spatial scope only through the provided tools.
2. Never invent an ID, handle, region, category, coordinate, or map fact.
3. Do not write Python, helper functions, code, or any executable program.
4. Do not request raw grid data, planner/search internals, costs, or a trajectory.
5. Use start_position as the first segment start and each prior target as the
   next segment start. Call set_segment_context before constraints for a segment.
6. Build Visit/Avoid/Near/Far/Relative/PathShape requirements with constraint,
   preserve their scopes, build ordered segments, then call task exactly once.
7. A singular phrase is not proof of uniqueness; use relational filtering or
   deterministic ranking. Use only categories listed in the user message.
8. An exact query that returns an error or empty result may be attempted at most
   three times. After that, use a different query or finish grounding by returning
   a brief plain-text explanation instead of calling another tool.

The task tool is the only valid submission mechanism. Text is not RouteIR.
"""


@dataclass(frozen=True)
class ToolCallGroundingResult:
    grounded: GroundedProgram | None
    verification: Mapping[str, object]
    tool_trace: tuple[Mapping[str, object], ...]
    candidate_attempts: tuple[Mapping[str, object], ...]
    statistics: Mapping[str, object]
    failure_reason: str | None = None


@dataclass
class _ToolSession:
    scene: SceneMap
    max_failed_query_attempts: int = 3
    runtime: GroundingRuntime = field(init=False)
    values: dict[str, object] = field(init=False)
    trace: list[dict[str, object]] = field(default_factory=list)
    failed_query_attempts: dict[str, int] = field(default_factory=dict)
    executed_calls: int = 0
    blocked_calls: int = 0
    _counter: int = 0

    def __post_init__(self) -> None:
        self.max_failed_query_attempts = min(3, max(1, self.max_failed_query_attempts))
        self.runtime = GroundingRuntime(self.scene)
        self.values = {
            "start_position": self.scene.start,
            "task_start": self.scene.start,
        }
        missing = SEMANTIC_API_NAMES - set(self.runtime.namespace())
        if missing:
            raise RuntimeError(f"Grounding runtime is missing registered APIs: {sorted(missing)}")

    def call(self, name: str, arguments: Mapping[str, object]) -> tuple[object, dict[str, object]]:
        started = time.perf_counter()
        query_key = _query_key(name, arguments)
        previous_failures = self.failed_query_attempts.get(query_key, 0)
        if previous_failures >= self.max_failed_query_attempts:
            self.blocked_calls += 1
            payload = {
                "error": "QUERY_RETRY_LIMIT_REACHED",
                "message": (
                    "This exact semantic query already produced an error or empty result "
                    f"{previous_failures} times. Use a different query or finish grounding."
                ),
                "failed_attempts": previous_failures,
                "max_failed_query_attempts": self.max_failed_query_attempts,
            }
            self.trace.append({
                "call_index": len(self.trace), "name": name,
                "arguments": dict(arguments), "query_key": query_key,
                "status": "blocked_retry_limit", "executed": False,
                "result": payload, "wall_seconds": time.perf_counter() - started,
            })
            return None, payload
        try:
            if name not in SEMANTIC_API_NAMES:
                raise GroundingFailure(f"UNKNOWN_TOOL: {name}")
            self.executed_calls += 1
            result = self._invoke(name, arguments)
            payload = self._record_value(result)
            if _is_empty_result(result):
                failure_count = previous_failures + 1
                self.failed_query_attempts[query_key] = failure_count
                payload.update({
                    "empty_result": True,
                    "failed_attempts": failure_count,
                    "remaining_failed_attempts": max(0, self.max_failed_query_attempts - failure_count),
                })
                status = "empty"
            else:
                self.failed_query_attempts.pop(query_key, None)
                status = "success"
            entry = {
                "call_index": len(self.trace), "name": name,
                "arguments": dict(arguments), "query_key": query_key,
                "status": status, "executed": True, "result": payload,
                "wall_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            result = None
            failure_count = previous_failures + 1
            self.failed_query_attempts[query_key] = failure_count
            payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "failed_attempts": failure_count,
                "remaining_failed_attempts": max(0, self.max_failed_query_attempts - failure_count),
            }
            entry = {
                "call_index": len(self.trace), "name": name,
                "arguments": dict(arguments), "query_key": query_key,
                "status": "failed", "executed": name in SEMANTIC_API_NAMES,
                "result": payload, "wall_seconds": time.perf_counter() - started,
            }
        self.trace.append(entry)
        return result, payload

    def _resolve(self, value: object) -> object:
        if isinstance(value, str) and value in self.values:
            return self.values[value]
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): self._resolve(item) for key, item in value.items()}
        return value

    def _invoke(self, name: str, raw: Mapping[str, object]) -> object:
        api = self.runtime.namespace()[name]
        if not callable(api):
            raise GroundingFailure(f"INVALID_TOOL: {name} is not callable")
        a = {key: self._resolve(value) for key, value in raw.items()}
        if name in {"union", "intersection"}:
            return api(*_sequence(a.get("sets"), "sets"))
        if name == "where":
            predicate_name = str(a.get("predicate") or "")
            predicate = self.runtime.namespace().get(predicate_name)
            if predicate_name not in PREDICATE_NAMES or not callable(predicate):
                raise GroundingFailure(f"INVALID_PREDICATE: {predicate_name}")
            predicate_args = _sequence(a.get("arguments", ()), "arguments")
            return api(a.get("candidates"), lambda candidate: predicate(candidate, *predicate_args))
        if name in {"all_of", "any_of"}:
            return api(*_sequence(a.get("values"), "values"))
        if name == "not_":
            return api(a.get("value"))
        if name == "constraint":
            refs = list(_sequence(a.get("references"), "references"))
            relation = a.get("relation")
            if relation is not None:
                if len(refs) != 2:
                    raise GroundingFailure("INVALID_ARGUMENT: a relative relation requires exactly two references")
                refs.insert(1, relation)
            segment_scope: object | None = None
            if bool(a.get("global_scope")):
                segment_scope = "global"
            elif "segment_scope" in a:
                segment_scope = a["segment_scope"]
            kwargs = {
                key: a[key]
                for key in ("spatial_scope", "source_text")
                if key in a
            }
            if segment_scope is not None:
                kwargs["segment_scope"] = segment_scope
            return api(str(a.get("kind") or ""), *refs, **kwargs)
        if name == "segment":
            return api(a.get("segment_id"), a.get("start"), a.get("target"), _sequence(a.get("constraints", ()), "constraints"))
        if name == "task":
            bindings: dict[str, object] = {}
            for item in _sequence(a.get("bindings", ()), "bindings"):
                if not isinstance(item, Mapping):
                    raise GroundingFailure("INVALID_BINDING: task bindings must contain name/value objects")
                bindings[str(item.get("name") or "")] = item.get("value")
            return api(_sequence(a.get("segments"), "segments"), bindings)

        positional_names: dict[str, tuple[str, ...]] = {
            "entities": ("category",), "rooms": ("category",), "target_of": ("segment",),
            "in_room": ("entities", "rooms"), "room_of": ("ref",), "contains": ("room", "category"),
            "adjacent_rooms": ("room",), "passage_regions": ("first_room", "second_room"),
            "exclude": ("candidates", "excluded"), "count": ("candidates",), "unique": ("candidates",),
            "choose_any": ("candidates",), "kth_nearest": ("candidates", "reference"),
            "kth_farthest": ("candidates", "reference"), "kth_largest": ("candidates",),
            "kth_smallest": ("candidates",), "order_by_distance": ("candidates", "reference"),
            "closest_pair_member": ("candidates", "references"), "region_of": ("entity",),
            "room_region": ("room",), "midpoint_region": ("first", "second"),
            "between_region": ("first", "second"), "near_region": ("reference",),
            "side_region": ("reference", "relative_to"), "boundary_region": ("room",),
            "half_room": ("room", "relative_to"), "relative_waypoint": ("first_anchor", "second_anchor"),
            "circle": ("reference",), "follow_wall": ("reference",), "position": ("ref",),
            "distance": ("first", "second"), "geodesic_distance": ("first", "second"),
            "inside": ("child", "container"), "intersects": ("first", "second"),
            "object_bbox": ("ref",), "room_polygon": ("room",), "compare": ("left", "operator", "right"),
            "near_to": ("candidate", "reference"), "far_from": ("candidate", "reference"),
            "next_to": ("candidate", "reference"), "on_top_of": ("candidate", "reference"),
            "in_corner": ("candidate",), "between": ("candidate", "first", "second"),
            "set_segment_context": ("segment_id", "start"),
            "count_next_to": ("category", "reference"), "count_near": ("category", "reference"),
        }
        positional = positional_names.get(name)
        if positional is None:
            raise GroundingFailure(f"UNSUPPORTED_TOOL_ADAPTER: {name}")
        args = [a.get(key) for key in positional]
        optional = {
            key: value for key, value in a.items()
            if key not in positional and value is not None
        }
        return api(*args, **optional)

    def _record_value(self, value: object) -> dict[str, object]:
        if value is None:
            return {"value": None}
        self._counter += 1
        prefix = "ref" if isinstance(value, GPRef) else "value"
        handle = f"{prefix}_{self._counter:04d}"
        self.values[handle] = value
        return {"handle": handle, "value": self._preview(value)}

    def _preview(self, value: object) -> object:
        if isinstance(value, GPRef):
            payload: dict[str, object] = {"kind": value.kind}
            if value.kind != "region":
                payload["id"] = value.id
            if value.category:
                payload["category"] = value.category
            if value.room_id:
                payload["room_id"] = value.room_id
            if value.kind == "region":
                payload["construction"] = dict(value.construction)
            return payload
        if isinstance(value, GroundedProgram):
            return value.to_json()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return {"count": len(value), "items": [self._preview(item) for item in value]}
        if isinstance(value, Mapping):
            return {str(key): self._preview(item) for key, item in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return {"type": type(value).__name__}


class ToolCallGrounder:
    """Compose the fixed grounding API through Gemini native function calls."""

    def __init__(
        self,
        llm_client: GeminiClient,
        *,
        max_tool_calls: int = 32,
        max_llm_turns: int = 40,
        max_failed_query_attempts: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.max_tool_calls = max(1, max_tool_calls)
        self.max_llm_turns = max(1, max_llm_turns)
        self.max_failed_query_attempts = min(3, max(1, max_failed_query_attempts))

    def ground(
        self,
        instruction: str,
        semantic_map: SceneMap,
        start_pose: object | None = None,
        *,
        entity_categories: Sequence[str],
        room_categories: Sequence[str],
        scene_summary: Mapping[str, object],
        max_repairs: int = 0,
    ) -> ToolCallGroundingResult:
        del start_pose  # SceneMap already owns the validated start reference.
        started = time.perf_counter()
        session = _ToolSession(
            semantic_map, max_failed_query_attempts=self.max_failed_query_attempts,
        )
        contents: list[dict[str, object]] = [{
            "role": "user",
            "parts": [{"text": self._user_prompt(instruction, entity_categories, room_categories, scene_summary)}],
        }]
        candidates: list[dict[str, object]] = []
        verification: Mapping[str, object] = {"status": "not_run", "errors": [], "warnings": []}
        grounded: GroundedProgram | None = None
        repairs = 0
        llm_turns = 0
        input_tokens = 0
        output_tokens = 0
        failure_reason: str | None = None
        budget_exhausted = False

        while llm_turns < self.max_llm_turns:
            if len(session.trace) >= self.max_tool_calls:
                budget_exhausted = True
                failure_reason = "TOOL_BUDGET_EXHAUSTED"
                break
            try:
                content, call_meta = tool_turn(
                    self.llm_client,
                    system_prompt=TOOL_CALL_SYSTEM_PROMPT,
                    contents=contents,
                    function_declarations=gemini_function_declarations(),
                    cache_namespace=f"grounding2route_tool_call_turn_{llm_turns}",
                )
            except Exception as exc:
                failure_reason = f"LLM_TOOL_CALL_FAILED: {type(exc).__name__}: {exc}"
                break
            llm_turns += 1
            usage = call_meta.get("usage_metadata")
            if isinstance(usage, Mapping):
                input_tokens += int(usage.get("promptTokenCount") or 0)
                output_tokens += int(usage.get("candidatesTokenCount") or 0)
            contents.append(dict(content))
            calls = _function_calls(content)
            if not calls:
                failure_reason = "MODEL_RETURNED_TEXT_WITHOUT_TASK"
                break
            response_parts: list[dict[str, object]] = []
            submitted: GroundedProgram | None = None
            for call in calls:
                if len(session.trace) >= self.max_tool_calls:
                    budget_exhausted = True
                    failure_reason = "TOOL_BUDGET_EXHAUSTED"
                    break
                name = str(call.get("name") or "")
                arguments = call.get("args")
                arguments = arguments if isinstance(arguments, Mapping) else {}
                result, payload = session.call(name, arguments)
                function_response: dict[str, object] = {"name": name, "response": payload}
                call_id = call.get("id")
                if isinstance(call_id, str) and call_id:
                    function_response["id"] = call_id
                response_parts.append({"functionResponse": function_response})
                if name == "task" and isinstance(result, GroundedProgram):
                    submitted = result
            if response_parts:
                contents.append({"role": "user", "parts": response_parts})
            if budget_exhausted:
                break
            if submitted is None:
                continue
            grounded = submitted
            verification = verify_grounded_program(grounded, semantic_map, expected_program=None)
            candidates.append({
                "attempt": len(candidates),
                "route_ir": grounded.to_json(),
                "verification": dict(verification),
            })
            if verification.get("status") == "success":
                failure_reason = None
                break
            if repairs >= max(0, max_repairs):
                failure_reason = "ROUTE_IR_VERIFICATION_FAILED"
                break
            repairs += 1
            contents.append({
                "role": "user",
                "parts": [{"text": "The submitted RouteIR failed the fixed verifier. Continue using tools and submit a corrected task. Structured verifier errors:\n" + json.dumps(verification, ensure_ascii=False)}],
            })
        else:
            budget_exhausted = True
            failure_reason = "LLM_TURN_BUDGET_EXHAUSTED"

        statistics = {
            "llm_invocations": llm_turns,
            "llm_input_tokens": input_tokens,
            "llm_output_tokens": output_tokens,
            "tool_call_requests": len(session.trace),
            "semantic_api_calls": session.executed_calls,
            "query_retry_limit_blocks": session.blocked_calls,
            "verifier_repair_rounds": repairs,
            "grounding_wall_seconds": time.perf_counter() - started,
            "verification_success": verification.get("status") == "success",
            "tool_budget_exhausted": budget_exhausted,
            "max_tool_calls": self.max_tool_calls,
            "max_llm_turns": self.max_llm_turns,
            "max_failed_query_attempts": self.max_failed_query_attempts,
        }
        return ToolCallGroundingResult(
            grounded,
            verification,
            tuple(session.trace),
            tuple(candidates),
            statistics,
            failure_reason,
        )

    @staticmethod
    def _user_prompt(instruction: str, entities: Sequence[str], rooms: Sequence[str], summary: Mapping[str, object]) -> str:
        return "\n\n".join((
            f"Instruction:\n{instruction.strip()}",
            "Object categories:\n" + ", ".join(entities),
            "Room categories:\n" + ", ".join(rooms),
            "Category-only scene summary (not object IDs or grid data):\n" + json.dumps(summary, ensure_ascii=False),
            "Initial opaque handles: start_position and task_start.",
        ))


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GroundingFailure(f"INVALID_ARGUMENT: {label} must be an array")
    return value


def _query_key(name: str, arguments: Mapping[str, object]) -> str:
    def normalize(value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return json.dumps(
        {"name": name.strip().lower(), "arguments": normalize(arguments)},
        sort_keys=True, ensure_ascii=False, default=str,
    )


def _is_empty_result(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 0
    )


def _function_calls(content: Mapping[str, object]) -> list[Mapping[str, object]]:
    parts = content.get("parts")
    if not isinstance(parts, list):
        return []
    return [part["functionCall"] for part in parts if isinstance(part, Mapping) and isinstance(part.get("functionCall"), Mapping)]
