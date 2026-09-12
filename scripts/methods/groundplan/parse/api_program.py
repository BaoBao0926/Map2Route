"""Restricted API-program parser for GroundPlan."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from scripts.methods.groundplan.config import PROMPT_DIR
from scripts.methods.groundplan.ir import (
    GPBinding,
    GPConstraintSpec,
    GPExpr,
    GPProgramSpec,
    GPSegmentSpec,
)
from scripts.methods.groundplan.llm_client import GeminiClient


class APIProgramParseError(ValueError):
    pass


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPR_OPERATORS = {
    "target_of": "target_of",
    "entities": "entities",
    "rooms": "rooms",
    "in_": "in",
    "in_room": "in",
    "room_of": "room_of",
    "contains": "contains",
    "count_next_to": "count_next_to",
    "count_near": "count_near",
    "adjacent_rooms": "adjacent_rooms",
    "passage_regions": "passage_regions",
    "union": "union",
    "intersection": "intersection",
    "exclude": "exclude",
    "count": "count",
    "where": "where",
    "unique": "unique",
    "choose_any": "choose_any",
    "near_to": "near_to",
    "far_from": "far_from",
    "next_to": "next_to",
    "on_top_of": "on_top_of",
    "in_corner": "in_corner",
    "between": "between",
    "compare": "compare",
    "and_": "and",
    "or_": "or",
    "not_": "not",
    "kth_nearest": "kth_nearest",
    "kth_farthest": "kth_farthest",
    "kth_largest": "kth_largest",
    "kth_smallest": "kth_smallest",
    "order_by_distance": "order_by_distance",
    "closest_pair_member": "closest_pair_member",
    "region_of": "region_of",
    "room_region": "room_region",
    "midpoint_region": "midpoint_region",
    "between_region": "between_region",
    "near_region": "near_region",
    "side_region": "side_region",
    "boundary_region": "boundary_region",
    "half_room": "half_room",
    "relative_waypoint": "relative_waypoint",
    "circle": "circle",
    "follow_wall": "follow_wall",
}
_CONSTRAINT_OPERATORS = {
    "require_visit",
    "require_visit_in_order",
    "forbid",
    "prefer_near",
    "prefer_far",
    "prefer_relative",
    "prefer_path_shape",
}
_SPECIAL_OPERATORS = {"go_to"}
_ALLOWED_API_FUNCTIONS = set(_EXPR_OPERATORS) | _CONSTRAINT_OPERATORS | _SPECIAL_OPERATORS


@dataclass
class _PendingSegment:
    id: str
    start: object
    target: object
    constraints: list[GPConstraintSpec] = field(default_factory=list)


def llm_parse_api_program_instruction(
    instruction_text: str,
    *,
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    llm_client: GeminiClient,
    max_parse_repairs: int = 1,
) -> tuple[GPProgramSpec, dict[str, object]]:
    prompt_template = (PROMPT_DIR / "parse" / "gemini_api_program_prompt.md").read_text(encoding="utf-8")
    user_prompt = (
        prompt_template.replace("{{INSTRUCTION}}", instruction_text.strip())
        .replace("{{ENTITY_CATEGORIES}}", ", ".join(entity_categories))
        .replace("{{ROOM_CATEGORIES}}", ", ".join(room_categories))
    )
    text, meta = llm_client.response_text(
        system_prompt="You are the SemPathBench GroundPlan API-program parser. Return code only.",
        user_prompt=user_prompt,
        cache_namespace="groundplan_api_program_parser",
    )
    parse_attempts: list[dict[str, object]] = [
        {
            "attempt": 0,
            "kind": "initial_api_program",
            "llm": meta,
            "raw_api_program": text,
        }
    ]
    try:
        program = parse_api_program(text, parser_mode="api")
    except APIProgramParseError as exc:
        parse_attempts[0]["status"] = "failed"
        parse_attempts[0]["error"] = str(exc)
        if max_parse_repairs <= 0:
            _attach_api_parse_context(exc, text, meta, parse_attempts)
            raise
        try:
            repaired_program, repair_meta = llm_repair_after_api_parse_failure(
                instruction_text,
                previous_api_program=text,
                parse_error=str(exc),
                entity_categories=entity_categories,
                room_categories=room_categories,
                llm_client=llm_client,
                attempt=1,
            )
        except APIProgramParseError as repair_exc:
            parse_attempts.append(
                {
                    "attempt": 1,
                    "kind": "api_program_parse_repair",
                    "status": "failed",
                    "error": str(repair_exc),
                    "llm": getattr(repair_exc, "llm", None),
                    "raw_api_program": getattr(repair_exc, "raw_api_program", None),
                }
            )
            setattr(repair_exc, "parse_attempts", parse_attempts)
            raise
        parse_attempts.append(
            {
                "attempt": 1,
                "kind": "api_program_parse_repair",
                "status": "success",
                "metadata": repair_meta,
                "raw_api_program": repair_meta.get("raw_api_program"),
            }
        )
        return repaired_program, {
            "llm": meta,
            "raw_api_program": repair_meta.get("raw_api_program"),
            "initial_raw_api_program": text,
            "api_parse_repair": repair_meta,
            "parse_attempts": parse_attempts,
        }
    parse_attempts[0]["status"] = "success"
    return program, {"llm": meta, "raw_api_program": text, "parse_attempts": parse_attempts}


def llm_repair_after_api_parse_failure(
    instruction_text: str,
    *,
    previous_api_program: str,
    parse_error: str,
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    llm_client: GeminiClient,
    attempt: int,
) -> tuple[GPProgramSpec, dict[str, object]]:
    prompt_template = (PROMPT_DIR / "parse" / "gemini_api_program_prompt.md").read_text(encoding="utf-8")
    base_prompt = (
        prompt_template.replace("{{INSTRUCTION}}", instruction_text.strip())
        .replace("{{ENTITY_CATEGORIES}}", ", ".join(entity_categories))
        .replace("{{ROOM_CATEGORIES}}", ", ".join(room_categories))
    )
    repair_prompt = f"""
{base_prompt}

## API Program Parse Repair Context

The previous restricted API program could not be parsed or validated. Rewrite it
as one complete valid API program while preserving the instruction semantics.

Parse error:

```text
{parse_error}
```

Previous API program:

```python
{previous_api_program}
```

Return code only.
""".strip()
    text, meta = llm_client.response_text(
        system_prompt="You repair SemPathBench GroundPlan restricted API programs. Return code only.",
        user_prompt=repair_prompt,
        cache_namespace=f"groundplan_api_program_parse_repair_v{attempt}",
    )
    try:
        program = parse_api_program(text, parser_mode="api_parse_repair")
    except APIProgramParseError as exc:
        _attach_api_parse_context(exc, text, meta, [])
        raise
    return program, {
        "mode": "api_parse_repair",
        "repair_attempt": attempt,
        "parse_error": parse_error,
        "llm": meta,
        "raw_api_program": text,
        "status": "success",
    }


def llm_repair_api_after_grounding_failure(
    instruction_text: str,
    *,
    previous_api_program: str,
    grounding_error: str,
    entity_categories: Sequence[str],
    room_categories: Sequence[str],
    scene_summary: Mapping[str, object],
    llm_client: GeminiClient,
    attempt: int,
) -> tuple[GPProgramSpec, dict[str, object]]:
    prompt_template = (PROMPT_DIR / "parse" / "gemini_api_program_prompt.md").read_text(encoding="utf-8")
    base_prompt = (
        prompt_template.replace("{{INSTRUCTION}}", instruction_text.strip())
        .replace("{{ENTITY_CATEGORIES}}", ", ".join(entity_categories))
        .replace("{{ROOM_CATEGORIES}}", ", ".join(room_categories))
    )
    repair_prompt = f"""
{base_prompt}

## API Program Grounding Repair Context

The previous API program was syntactically valid, but deterministic grounding
failed. Rewrite the API program so that it still represents the original
instruction while avoiding the grounding error.

Grounding error:

```text
{grounding_error}
```

Scene category count summary:

```text
{dict(scene_summary)}
```

Previous API program:

```python
{previous_api_program}
```

Return code only.
""".strip()
    text, meta = llm_client.response_text(
        system_prompt="You repair SemPathBench GroundPlan API programs after grounding errors. Return code only.",
        user_prompt=repair_prompt,
        cache_namespace=f"groundplan_api_program_grounding_repair_v{attempt}",
    )
    try:
        program = parse_api_program(text, parser_mode="api_grounding_repair")
    except APIProgramParseError as exc:
        _attach_api_parse_context(exc, text, meta, [])
        raise
    return program, {
        "mode": "api_grounding_repair",
        "repair_attempt": attempt,
        "grounding_error": grounding_error,
        "scene_summary": dict(scene_summary),
        "llm": meta,
        "raw_api_program": text,
        "status": "success",
    }


def parse_api_program(text: str, *, parser_mode: str = "api") -> GPProgramSpec:
    source = _extract_code(text)
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise APIProgramParseError(f"API_PROGRAM_SYNTAX_ERROR: {exc}") from exc
    compiler = _APIProgramCompiler(source)
    return compiler.compile(module, parser_mode=parser_mode)


class _APIProgramCompiler:
    def __init__(self, source: str) -> None:
        self.source = source
        self.bindings: list[GPBinding] = []
        self.binding_names: set[str] = {"start_position", "task_start"}
        self.segments: list[_PendingSegment] = []
        self.current_start: object = "start_position"

    def compile(self, module: ast.Module, *, parser_mode: str) -> GPProgramSpec:
        for statement in module.body:
            if isinstance(statement, ast.Assign):
                self._compile_assign(statement)
            elif isinstance(statement, ast.Expr):
                value = statement.value
                if not isinstance(value, ast.Call):
                    raise self._error(statement, "Only api function calls may appear as expression statements.")
                self._compile_expr_statement(value)
            else:
                raise self._error(
                    statement,
                    "Only assignment statements and api function call expression statements are allowed.",
                )
        if not self.segments:
            raise APIProgramParseError("API program must create at least one segment with api.go_to(...).")
        return GPProgramSpec(
            bindings=tuple(self.bindings),
            segments=tuple(
                GPSegmentSpec(
                    id=segment.id,
                    start=segment.start,
                    target=segment.target,
                    constraints=tuple(
                        GPConstraintSpec(
                            kind=constraint.kind,
                            exprs=constraint.exprs,
                            segment_scope=constraint.segment_scope or (segment.id,),
                            spatial_scope=constraint.spatial_scope,
                            source_text=constraint.source_text,
                        )
                        for constraint in segment.constraints
                    ),
                )
                for segment in self.segments
            ),
            source=self.source,
            parser_mode=parser_mode,
            diagnostics=("api_program_parser_mode",),
        )

    def _compile_assign(self, statement: ast.Assign) -> None:
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise self._error(statement, "Assignments must be simple: name = api.function(...).")
        name = statement.targets[0].id
        self._validate_name(name, statement)
        if isinstance(statement.value, ast.Call) and self._api_function_name(statement.value) == "go_to":
            self._add_segment_from_call(statement.value, segment_name=name)
            self.binding_names.add(name)
            return
        value = self._parse_value(statement.value)
        self.bindings.append(GPBinding(name, value))
        self.binding_names.add(name)

    def _compile_expr_statement(self, call: ast.Call) -> None:
        function_name = self._api_function_name(call)
        if function_name == "go_to":
            self._add_segment_from_call(call, segment_name=None)
            return
        if function_name in _CONSTRAINT_OPERATORS:
            if not self.segments:
                raise self._error(call, "Standalone constraints must follow api.go_to(...).")
            self.segments[-1].constraints.append(self._parse_constraint_call(call))
            return
        raise self._error(call, "Expression statements must be api.go_to(...) or a constraint call.")

    def _add_segment_from_call(self, call: ast.Call, *, segment_name: str | None) -> None:
        args = [self._parse_value(arg) for arg in call.args]
        kwargs = self._parse_keywords(call)
        explicit_id = kwargs.pop("id", None)
        if explicit_id is not None and not isinstance(explicit_id, str):
            raise self._error(call, "api.go_to(..., id=...) must use a string id.")
        segment_id = explicit_id or segment_name or f"s{len(self.segments) + 1}"
        if not _NAME_RE.fullmatch(segment_id):
            raise self._error(call, f"Invalid segment id: {segment_id!r}.")
        if len(args) == 1:
            start = kwargs.pop("start", self.current_start)
            target = args[0]
        elif len(args) == 2:
            start, target = args
            if "start" in kwargs:
                raise self._error(call, "api.go_to received both positional start and start=.")
        else:
            raise self._error(call, "api.go_to expects target or start, target.")
        raw_constraints = kwargs.pop("constraints", [])
        if kwargs:
            raise self._error(call, f"Unsupported api.go_to keyword(s): {', '.join(sorted(kwargs))}.")
        constraints = self._parse_constraint_list(raw_constraints)
        self.segments.append(_PendingSegment(segment_id, start, target, constraints))
        self.current_start = target
        self.binding_names.add(segment_id)

    def _parse_value(self, node: ast.AST) -> object:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool)) or node.value is None:
                return node.value
            raise self._error(node, "Only string, number, boolean, and None literals are allowed.")
        if isinstance(node, ast.List):
            return [self._parse_value(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._parse_value(item) for item in node.elts)
        if isinstance(node, ast.Call):
            function_name = self._api_function_name(node)
            if function_name in _CONSTRAINT_OPERATORS:
                return self._parse_constraint_call(node)
            if function_name in _SPECIAL_OPERATORS:
                raise self._error(node, f"api.{function_name} cannot be nested inside expressions.")
            op = _EXPR_OPERATORS.get(function_name)
            if op is None:
                raise self._error(node, f"Unsupported api function: {function_name}.")
            args = tuple(self._parse_value(arg) for arg in node.args)
            kwargs = self._parse_keywords(node)
            return GPExpr(op, args, kwargs)
        raise self._error(node, "Unsupported syntax in API program.")

    def _parse_constraint_call(self, call: ast.Call) -> GPConstraintSpec:
        function_name = self._api_function_name(call)
        if function_name not in _CONSTRAINT_OPERATORS:
            raise self._error(call, f"Unsupported constraint function: {function_name}.")
        kwargs = self._parse_keywords(call)
        spatial_scope = kwargs.pop("spatial_scope", kwargs.pop("within", kwargs.pop("scope", None)))
        source_text = kwargs.pop("source_text", None)
        if kwargs:
            raise self._error(call, f"Unsupported constraint keyword(s): {', '.join(sorted(kwargs))}.")
        if source_text is not None and not isinstance(source_text, str):
            raise self._error(call, "source_text must be a string.")
        return GPConstraintSpec(
            kind=function_name,  # type: ignore[arg-type]
            exprs=tuple(self._parse_value(arg) for arg in call.args),
            spatial_scope=spatial_scope,
            source_text=source_text or ast.get_source_segment(self.source, call),
        )

    def _parse_constraint_list(self, value: object) -> list[GPConstraintSpec]:
        if value is None:
            return []
        if isinstance(value, GPConstraintSpec):
            return [value]
        if isinstance(value, list):
            constraints: list[GPConstraintSpec] = []
            for item in value:
                if not isinstance(item, GPConstraintSpec):
                    raise APIProgramParseError("api.go_to constraints must be constraint API calls.")
                constraints.append(item)
            return constraints
        raise APIProgramParseError("api.go_to constraints= must be a list of constraint API calls.")

    def _parse_keywords(self, call: ast.Call) -> dict[str, object]:
        result: dict[str, object] = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise self._error(call, "Starred keyword arguments are not allowed.")
            result[keyword.arg] = self._parse_value(keyword.value)
        return result

    def _api_function_name(self, call: ast.Call) -> str:
        func = call.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "api":
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            raise self._error(call, "Calls must be to api.<function>(...) only.")
        if name not in _ALLOWED_API_FUNCTIONS:
            raise self._error(call, f"Unsupported api function: {name}.")
        return name

    def _validate_name(self, name: str, node: ast.AST) -> None:
        if name == "api" or name.startswith("__") or not _NAME_RE.fullmatch(name):
            raise self._error(node, f"Invalid binding name: {name!r}.")

    def _error(self, node: ast.AST, message: str) -> APIProgramParseError:
        line = getattr(node, "lineno", "?")
        return APIProgramParseError(f"{message} (line {line})")


def _extract_code(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    lines = stripped.splitlines()
    if lines and lines[0].strip().lower() in {"python", "py"}:
        stripped = "\n".join(lines[1:]).strip()
    if not stripped:
        raise APIProgramParseError("API program is empty.")
    return stripped


def _attach_api_parse_context(
    exc: APIProgramParseError,
    raw_api_program: str,
    llm_meta: Mapping[str, object],
    parse_attempts: list[dict[str, object]],
) -> None:
    setattr(exc, "raw_api_program", raw_api_program)
    setattr(exc, "llm", llm_meta)
    setattr(exc, "parse_attempts", parse_attempts)
