"""JSON intent-graph parser for Grounding2Route.

This module keeps the downstream contract unchanged: it converts a typed JSON
intent graph into the same ``GPProgramSpec`` consumed by the existing grounder
and planner.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from scripts.methods.grounding2route.config import PROMPT_DIR
from scripts.methods.grounding2route.ir import (
    GPBinding,
    GPConstraintSpec,
    GPExpr,
    GPProgramSpec,
    GPSegmentSpec,
)
from scripts.methods.grounding2route.llm_client import GeminiClient


class IntentParseError(ValueError):
    pass


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONSTRAINT_KINDS = {
    "require_visit",
    "require_visit_in_order",
    "forbid",
    "prefer_near",
    "prefer_far",
    "prefer_relative",
    "prefer_path_shape",
}


def llm_parse_intent_instruction(
    instruction_text: str,
    *,
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    llm_client: GeminiClient,
    max_parse_repairs: int = 1,
) -> tuple[GPProgramSpec, dict[str, object]]:
    prompt_template = (PROMPT_DIR / "parse" / "gemini_intent_parser_prompt.md").read_text(encoding="utf-8")
    user_prompt = (
        prompt_template.replace("{{INSTRUCTION}}", instruction_text.strip())
        .replace("{{ENTITY_CATEGORIES}}", ", ".join(entity_categories))
        .replace("{{ROOM_CATEGORIES}}", ", ".join(room_categories))
    )
    text, meta = llm_client.response_text(
        system_prompt="You are the SemPathBench Grounding2Route intent parser. Return JSON only.",
        user_prompt=user_prompt,
        cache_namespace="grounding2route_intent_parser",
    )
    parse_attempts: list[dict[str, object]] = [
        {
            "attempt": 0,
            "kind": "initial_intent",
            "llm": meta,
            "raw_intent": text,
        }
    ]
    try:
        intent = parse_intent_json(text)
        program = intent_to_program(intent, parser_mode="intent")
    except IntentParseError as exc:
        parse_attempts[0]["status"] = "failed"
        parse_attempts[0]["error"] = str(exc)
        if max_parse_repairs <= 0:
            setattr(exc, "raw_intent", text)
            setattr(exc, "llm", meta)
            setattr(exc, "parse_attempts", parse_attempts)
            raise
        try:
            repaired_program, repair_meta = llm_repair_after_intent_parse_failure(
                instruction_text,
                previous_intent=text,
                parse_error=str(exc),
                entity_categories=entity_categories,
                room_categories=room_categories,
                llm_client=llm_client,
                attempt=1,
            )
        except IntentParseError as repair_exc:
            parse_attempts.append(
                {
                    "attempt": 1,
                    "kind": "intent_parse_repair",
                    "status": "failed",
                    "error": str(repair_exc),
                    "llm": getattr(repair_exc, "llm", None),
                    "raw_intent": getattr(repair_exc, "raw_intent", None),
                }
            )
            setattr(repair_exc, "parse_attempts", parse_attempts)
            raise
        parse_attempts.append(
            {
                "attempt": 1,
                "kind": "intent_parse_repair",
                "status": "success",
                "metadata": repair_meta,
                "raw_intent": repair_meta.get("raw_intent"),
            }
        )
        return repaired_program, {
            "llm": meta,
            "raw_intent": repair_meta.get("raw_intent"),
            "initial_raw_intent": text,
            "compiled_dsl": repaired_program.source,
            "intent_parse_repair": repair_meta,
            "parse_attempts": parse_attempts,
        }
    parse_attempts[0]["status"] = "success"
    return program, {
        "llm": meta,
        "raw_intent": text,
        "compiled_dsl": program.source,
        "parse_attempts": parse_attempts,
    }


def llm_repair_after_intent_parse_failure(
    instruction_text: str,
    *,
    previous_intent: str,
    parse_error: str,
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    llm_client: GeminiClient,
    attempt: int,
) -> tuple[GPProgramSpec, dict[str, object]]:
    prompt_template = (PROMPT_DIR / "parse" / "gemini_intent_parser_prompt.md").read_text(encoding="utf-8")
    base_prompt = (
        prompt_template.replace("{{INSTRUCTION}}", instruction_text.strip())
        .replace("{{ENTITY_CATEGORIES}}", ", ".join(entity_categories))
        .replace("{{ROOM_CATEGORIES}}", ", ".join(room_categories))
    )
    repair_prompt = f"""
{base_prompt}

## Intent Parse Repair Context

The previous JSON intent graph could not be parsed or validated. Rewrite it as
one complete valid JSON intent graph while preserving the instruction semantics.

Parse error:

```text
{parse_error}
```

Previous intent JSON:

```json
{previous_intent}
```

Repair rules:

- Return exactly one JSON object.
- Do not output Markdown fences, comments, explanations, object IDs, room IDs,
  coordinates, cells, paths, or planner parameters.
- Every segment must have `id`, `from`, and `to`.
- Every operator object must have an `op` string. Use `args` and `kwargs`.
""".strip()
    text, meta = llm_client.response_text(
        system_prompt="You repair SemPathBench Grounding2Route intent JSON. Return JSON only.",
        user_prompt=repair_prompt,
        cache_namespace=f"grounding2route_intent_parser_parse_repair_v{attempt}",
    )
    try:
        intent = parse_intent_json(text)
        program = intent_to_program(intent, parser_mode="intent_parse_repair")
    except IntentParseError as exc:
        setattr(exc, "raw_intent", text)
        setattr(exc, "llm", meta)
        raise
    return program, {
        "mode": "intent_parse_repair",
        "repair_attempt": attempt,
        "parse_error": parse_error,
        "llm": meta,
        "raw_intent": text,
        "compiled_dsl": program.source,
        "status": "success",
    }


def llm_repair_intent_after_grounding_failure(
    instruction_text: str,
    *,
    previous_intent: str,
    previous_compiled_dsl: str,
    grounding_error: str,
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    scene_summary: Mapping[str, object],
    llm_client: GeminiClient,
    attempt: int,
) -> tuple[GPProgramSpec, dict[str, object]]:
    prompt_template = (PROMPT_DIR / "parse" / "gemini_intent_parser_prompt.md").read_text(encoding="utf-8")
    base_prompt = (
        prompt_template.replace("{{INSTRUCTION}}", instruction_text.strip())
        .replace("{{ENTITY_CATEGORIES}}", ", ".join(entity_categories))
        .replace("{{ROOM_CATEGORIES}}", ", ".join(room_categories))
    )
    repair_prompt = f"""
{base_prompt}

## Intent Grounding Repair Context

The previous JSON intent graph was syntactically valid and compiled to the
Grounding2Route DSL, but deterministic grounding failed. Rewrite the JSON intent
graph, not DSL, so that it still represents the original instruction while
avoiding the grounding error.

Grounding error:

```text
{grounding_error}
```

Scene category count summary:

```json
{json.dumps(scene_summary, indent=2, ensure_ascii=False)}
```

Previous intent JSON:

```json
{previous_intent}
```

Compiled DSL from the previous intent, for debugging only:

```text
{previous_compiled_dsl}
```

Repair rules:

- Return exactly one JSON object using the intent schema.
- Do not output DSL, Markdown fences, comments, explanations, object IDs, room
  IDs, coordinates, cells, paths, or planner parameters.
- Preserve all destinations, ordering, hard requirements, and soft preferences
  unless the failed expression made them impossible to ground.
- If a candidate set is empty, weaken only the failed filter or replace a hard
  threshold relation with a ranking expression that preserves the same noun
  phrase.
- If `unique(...)` was ambiguous, replace only that expression with a symbolic
  filter or contextual ranking. A singular English noun phrase does not imply
  scene uniqueness.
- If a room/object filter refers to counts, containment, absence, "same room",
  "other", or "remaining", express it with `where`, `count`, `in`, `exclude`,
  and binding references instead of guessing a nearest object.
- If a relative preference failed because the spatial scope forced references
  into different rooms, repair the scope or split the preference. Do not delete
  the destination.
- Keep the output as JSON intent so it can be compiled by code.
""".strip()
    text, meta = llm_client.response_text(
        system_prompt="You repair SemPathBench Grounding2Route intent JSON after deterministic grounding errors. Return JSON only.",
        user_prompt=repair_prompt,
        cache_namespace=f"grounding2route_intent_parser_grounding_repair_v{attempt}",
    )
    try:
        intent = parse_intent_json(text)
        program = intent_to_program(intent, parser_mode="intent_grounding_repair")
    except IntentParseError as exc:
        setattr(exc, "raw_intent", text)
        setattr(exc, "llm", meta)
        raise
    return program, {
        "mode": "intent_grounding_repair",
        "repair_attempt": attempt,
        "grounding_error": grounding_error,
        "scene_summary": dict(scene_summary),
        "llm": meta,
        "raw_intent": text,
        "compiled_dsl": program.source,
        "status": "success",
    }


def parse_intent_json(text: str) -> Mapping[str, object]:
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise IntentParseError(f"INTENT_JSON_ERROR: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise IntentParseError("Intent payload must be a JSON object.")
    return payload


def intent_to_program(intent: Mapping[str, object], *, parser_mode: str) -> GPProgramSpec:
    raw_bindings = intent.get("bindings")
    raw_segments = intent.get("segments")
    if not isinstance(raw_bindings, list):
        raise IntentParseError("Intent must contain a bindings array.")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise IntentParseError("Intent must contain a non-empty segments array.")

    binding_names: set[str] = set()
    bindings: list[GPBinding] = []
    for index, raw_binding in enumerate(raw_bindings):
        if not isinstance(raw_binding, Mapping):
            raise IntentParseError(f"Binding {index} must be an object.")
        name = _required_name(raw_binding.get("name"), f"binding {index} name")
        if name in binding_names:
            raise IntentParseError(f"Duplicate binding name: {name}")
        binding_names.add(name)
        if "expr" not in raw_binding:
            raise IntentParseError(f"Binding {name} must contain expr.")
        bindings.append(GPBinding(name, _parse_expr_value(raw_binding["expr"])))

    segment_ids: set[str] = set()
    segments: list[GPSegmentSpec] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise IntentParseError(f"Segment {index} must be an object.")
        segment_id = _required_name(raw_segment.get("id", f"s{index + 1}"), f"segment {index} id")
        if segment_id in segment_ids:
            raise IntentParseError(f"Duplicate segment id: {segment_id}")
        segment_ids.add(segment_id)
        if "from" not in raw_segment or "to" not in raw_segment:
            raise IntentParseError(f"Segment {segment_id} must contain from and to.")
        constraints = tuple(_parse_constraint(item, segment_id) for item in _list(raw_segment.get("constraints", []), f"segment {segment_id} constraints"))
        segments.append(
            GPSegmentSpec(
                id=segment_id,
                start=_parse_expr_value(raw_segment["from"]),
                target=_parse_expr_value(raw_segment["to"]),
                constraints=constraints,
            )
        )

    program = GPProgramSpec(
        bindings=tuple(bindings),
        segments=tuple(segments),
        source=_program_source_from_ast(tuple(bindings), tuple(segments)),
        parser_mode=parser_mode,
        diagnostics=("intent_json_parser_mode",),
    )
    return program


def _parse_constraint(raw_constraint: object, segment_id: str) -> GPConstraintSpec:
    if not isinstance(raw_constraint, Mapping):
        raise IntentParseError(f"Constraint in {segment_id} must be an object.")
    kind = raw_constraint.get("kind")
    if not isinstance(kind, str) or kind not in _CONSTRAINT_KINDS:
        raise IntentParseError(f"Constraint in {segment_id} has unsupported kind: {kind!r}")
    raw_args = raw_constraint.get("args", raw_constraint.get("exprs", []))
    args = tuple(_parse_expr_value(item) for item in _list(raw_args, f"constraint {kind} args"))
    spatial_scope = None
    if "spatial_scope" in raw_constraint:
        spatial_scope = _parse_expr_value(raw_constraint["spatial_scope"])
    source_text = raw_constraint.get("source_text")
    return GPConstraintSpec(
        kind=kind,  # type: ignore[arg-type]
        exprs=args,
        segment_scope=(segment_id,),
        spatial_scope=spatial_scope,
        source_text=str(source_text) if source_text is not None else None,
    )


def _parse_expr_value(value: object) -> object:
    if isinstance(value, Mapping):
        op = value.get("op")
        if not isinstance(op, str) or not op.strip():
            raise IntentParseError(f"Expression object must contain op: {value!r}")
        args = tuple(_parse_expr_value(item) for item in _list(value.get("args", []), f"{op}.args"))
        raw_kwargs = value.get("kwargs", {})
        if not isinstance(raw_kwargs, Mapping):
            raise IntentParseError(f"{op}.kwargs must be an object.")
        kwargs = {str(key): _parse_expr_value(item) for key, item in raw_kwargs.items()}
        return GPExpr(op.strip(), args, kwargs)
    if isinstance(value, list):
        return [_parse_expr_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_parse_expr_value(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise IntentParseError(f"Unsupported expression value: {value!r}")


def _required_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise IntentParseError(f"{label} must be a valid identifier.")
    return value


def _list(value: object, label: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise IntentParseError(f"{label} must be an array.")
    return value


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise IntentParseError("Intent response did not contain a JSON object.")
    return stripped[start : end + 1]


def _program_source_from_ast(bindings: tuple[GPBinding, ...], segments: tuple[GPSegmentSpec, ...]) -> str:
    lines = ["task {"]
    for binding in bindings:
        lines.append(f"    let {binding.name} = {_expr_source(binding.expr)}")
    for segment in segments:
        lines.append("")
        lines.append(f"    segment {segment.id} {{")
        lines.append(f"        from: {_expr_source(segment.start)}")
        lines.append(f"        to: {_expr_source(segment.target)}")
        lines.append(f"        go_to({_expr_source(segment.target)})")
        for constraint in segment.constraints:
            args = ", ".join(_expr_source(expr) for expr in constraint.exprs)
            line = f"        {constraint.kind}({args})"
            if constraint.spatial_scope is not None:
                line += f" within({_expr_source(constraint.spatial_scope)})"
            lines.append(line)
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


def _expr_source(expr: object) -> str:
    if isinstance(expr, GPExpr):
        args = [_expr_source(arg) for arg in expr.args]
        args.extend(f"{key}={_expr_source(value)}" for key, value in expr.kwargs.items())
        return f"{expr.op}({', '.join(args)})"
    if isinstance(expr, str):
        if expr == "start_position" or _NAME_RE.fullmatch(expr):
            return expr
        return json.dumps(expr)
    if isinstance(expr, bool):
        return "true" if expr else "false"
    if expr is None:
        return "null"
    if isinstance(expr, list):
        return "[" + ", ".join(_expr_source(item) for item in expr) + "]"
    if isinstance(expr, tuple):
        return "{" + ", ".join(_expr_source(item) for item in expr) + "}"
    return str(expr)
