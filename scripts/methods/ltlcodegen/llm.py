"""Native-style LTLCodeGen code generation and execution with Gemini."""

from __future__ import annotations

import json
import os
import re
import time
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping

from scripts.methods.ltlcodegen.ltl import (
    Formula,
    atomic_propositions,
    ap,
    ltl_always,
    ltl_and,
    ltl_eventually,
    ltl_imply,
    ltl_next,
    ltl_not,
    ltl_or,
    ltl_until,
)


DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)


SYSTEM_PROMPT = """You are implementing the original LTLCodeGen mechanism for robot navigation.

Generate Python code, not JSON. The code must define:

def question():
    ...
    return formula

Available functions are already imported in the execution template:
from ltl_operators import ap
from ltl_operators import ltl_and, ltl_or, ltl_not, ltl_until, ltl_eventually, ltl_always, ltl_imply

Use only atomic propositions from the AP inventory. The inventory is category-level
and may contain duplicate categories. If an instruction mentions a category with
multiple instances, use the first matching AP listed in the inventory. Do not solve
nearest, farthest, relational, room-content, midpoint, or comparative references.
Ignore soft preferences that cannot be represented as hard LTL constraints.

The planner already starts from the benchmark start_pose. Do not encode the
starting pose, starting room, or starting object as an atomic proposition or as
a precondition. Phrases such as "start from", "starting at", "beside you", or
"near you" describe only the initial robot pose and must be ignored when
constructing the LTL formula.

Do not output markdown explanation. Output only Python code.
"""


def _extract_gemini_text(response: Mapping[str, object]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Gemini response did not contain candidates.")
    chunks: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        content = candidate.get("content")
        if not isinstance(content, Mapping):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    if not chunks:
        raise ValueError("Gemini response did not contain text parts.")
    return "".join(chunks)


def _strip_markdown_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _remove_allowed_imports(code: str) -> str:
    kept: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("from ltl_operators import "):
            continue
        if stripped.startswith("from available_actions import "):
            continue
        if stripped.startswith("from speech_to_ltl."):
            continue
        kept.append(line)
    return "\n".join(kept)


def _remove_initial_ap_conjuncts(formula: Formula) -> tuple[Formula, tuple[str, ...]]:
    """Drop top-level AP preconditions that duplicate the external start pose."""
    formula = formula.simplify()
    if formula.op != "and":
        return formula, ()

    kept: list[Formula] = []
    removed: list[str] = []
    for arg in formula.args:
        if arg.op == "ap" and arg.value:
            removed.append(arg.value)
        else:
            kept.append(arg)

    if not removed or not kept:
        return formula, ()
    normalized = kept[0]
    for arg in kept[1:]:
        normalized = ltl_and(normalized, arg)
    return normalized.simplify(), tuple(removed)


def _api_key_module_value(*names: str) -> str | None:
    try:
        from scripts.methods import api_key
    except ImportError:
        return None
    for name in names:
        value = getattr(api_key, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def configured_model() -> str:
    return (
        _api_key_module_value("MODEL")
        or os.environ.get("GEMINI_MODEL")
        or DEFAULT_MODEL
    )


def configured_api_key() -> str | None:
    return (
        _api_key_module_value("API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )


def _verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[ltlcodegen:llm] {message}", flush=True)


def gemini_response_text(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> str:
    key = api_key or configured_api_key()
    if not key:
        raise RuntimeError(
            "API_KEY in scripts/methods/api_key.py or GEMINI_API_KEY is required "
            "for Gemini LTLCodeGen translation."
        )

    model_name = model.removeprefix("models/")
    encoded_model = urllib.parse.quote(model_name, safe="")
    url = GEMINI_GENERATE_CONTENT_URL.format(model=encoded_model)
    url = f"{url}?key={urllib.parse.quote(key, safe='')}"
    request_payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
        },
    }
    data = json.dumps(request_payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: {error.code} {body}") from error

    response_payload = json.loads(raw)
    if not isinstance(response_payload, dict):
        raise ValueError("Gemini response body must be a JSON object.")
    return _strip_markdown_code(_extract_gemini_text(response_payload))


def build_translation_prompt(
    *,
    instruction: str,
    inventory: Mapping[str, object],
    previous_answer: str = "",
    failure_reason: str = "",
) -> str:
    retry_block = ""
    if previous_answer or failure_reason:
        retry_block = (
            "\nPrevious generated code:\n"
            f"{previous_answer}\n\n"
            "Failure reason:\n"
            f"{failure_reason}\n\n"
            "Regenerate a corrected question() implementation.\n"
        )
    return (
        "Translate the instruction to LTL using Python code.\n\n"
        "Important grounding rule:\n"
        "- The planner already starts from start_pose.\n"
        "- Do not represent the starting location, starting room, or starting object.\n"
        "- Ignore phrases like 'start from', 'beside you', and 'near you' unless they "
        "refer to a future navigation target.\n\n"
        "Instruction:\n"
        f"{instruction.strip()}\n\n"
        "Available atomic propositions:\n"
        f"{json.dumps(inventory, indent=2, ensure_ascii=False)}\n\n"
        "Use only AP names that appear explicitly in the rooms or objects lists. "
        "Do not invent AP names from object categories.\n\n"
        "Example style:\n"
        "def question():\n"
        "    target = ap(\"object_12\")\n"
        "    return ltl_eventually(target)\n"
        f"{retry_block}"
    )


def _inventory_aps(inventory: Mapping[str, object]) -> set[str]:
    aps: set[str] = set()
    for key in ("rooms", "objects"):
        raw_entries = inventory.get(key, [])
        if not isinstance(raw_entries, list):
            continue
        for entry in raw_entries:
            if not isinstance(entry, str):
                continue
            ap_name = entry.split(":", 1)[0].strip()
            if ap_name:
                aps.add(ap_name)
    return aps


def validate_formula_aps(formula: Formula, inventory: Mapping[str, object]) -> None:
    valid_aps = _inventory_aps(inventory)
    unknown = sorted(atomic_propositions(formula) - valid_aps)
    if unknown:
        raise ValueError(
            "Formula uses AP(s) not listed in the inventory: "
            + ", ".join(unknown)
            + ". Use only AP names from the provided rooms/objects lists."
        )


def execute_question(code: str) -> Formula:
    cleaned = _remove_allowed_imports(_strip_markdown_code(code))
    namespace: dict[str, object] = {
        "__builtins__": {
            "False": False,
            "True": True,
            "None": None,
            "range": range,
            "len": len,
        },
        "reach": "reach",
        "ap": ap,
        "ltl_and": ltl_and,
        "ltl_or": ltl_or,
        "ltl_not": ltl_not,
        "ltl_until": ltl_until,
        "ltl_eventually": ltl_eventually,
        "ltl_always": ltl_always,
        "ltl_imply": ltl_imply,
        "ltl_next": ltl_next,
    }
    exec(compile(cleaned, "<ltlcodegen_generated>", "exec"), namespace)
    question = namespace.get("question")
    if not callable(question):
        raise ValueError("Generated code must define callable question().")
    result = question()
    if not isinstance(result, Formula):
        raise TypeError(
            "question() must return an LTL Formula built with ap/ltl_* helpers."
        )
    return result.simplify()


def translate_instruction_to_ltl(
    *,
    instruction: str,
    inventory: Mapping[str, object],
    model: str | None = None,
    api_key: str | None = None,
    cache_path: Path | None = None,
    overwrite_cache: bool = False,
    max_retries: int = 3,
    verbose: bool = False,
) -> dict[str, object]:
    model = model or configured_model()
    _verbose_print(verbose, f"model={model}")
    previous_answer = ""
    failure_reason = ""
    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        _verbose_print(verbose, f"checking cache {cache_path}")
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("formula_ast"), dict):
            cached_model = payload.get("model")
            if cached_model == model:
                formula = Formula.from_json(payload["formula_ast"])  # type: ignore[arg-type]
                formula, removed_initial_aps = _remove_initial_ap_conjuncts(formula)
                try:
                    validate_formula_aps(formula, inventory)
                except ValueError as exc:
                    _verbose_print(verbose, f"cache invalid: {exc}")
                    previous_answer = str(payload.get("generated_code", ""))
                    failure_reason = str(exc)
                else:
                    _verbose_print(verbose, "cache hit")
                    if removed_initial_aps:
                        _verbose_print(
                            verbose,
                            "normalized cached formula by removing initial AP precondition(s): "
                            + ", ".join(removed_initial_aps),
                        )
                        payload["ltl_formula"] = str(formula)
                        payload["formula_ast"] = formula.to_json()
                        payload["removed_initial_aps"] = list(removed_initial_aps)
                        cache_path.write_text(
                            json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        _verbose_print(verbose, f"rewrote normalized cache {cache_path}")
                    payload["formula"] = formula
                    return payload
                if removed_initial_aps:
                    payload["ltl_formula"] = str(formula)
                    payload["formula_ast"] = formula.to_json()
                    payload["removed_initial_aps"] = list(removed_initial_aps)
            else:
                _verbose_print(
                    verbose,
                    f"cache model mismatch cached={cached_model!r} current={model!r}",
                )
    elif cache_path is not None:
        _verbose_print(verbose, f"cache miss {cache_path}")

    attempts: list[dict[str, str]] = []
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        prompt = build_translation_prompt(
            instruction=instruction,
            inventory=inventory,
            previous_answer=previous_answer,
            failure_reason=failure_reason,
        )
        _verbose_print(verbose, f"Gemini request attempt={attempt}/{max_retries}")
        code = ""
        try:
            code = gemini_response_text(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                api_key=api_key,
            )
            _verbose_print(verbose, "generated code:")
            _verbose_print(verbose, textwrap.dedent(_strip_markdown_code(code)).strip())
            formula = execute_question(code)
            formula, removed_initial_aps = _remove_initial_ap_conjuncts(formula)
            validate_formula_aps(formula, inventory)
            if removed_initial_aps:
                _verbose_print(
                    verbose,
                    "removed initial AP precondition(s): "
                    + ", ".join(removed_initial_aps),
                )
            _verbose_print(verbose, f"executed question() -> {formula}")
            payload: dict[str, object] = {
                "model": model,
                "generated_code": textwrap.dedent(_strip_markdown_code(code)).strip(),
                "ltl_formula": str(formula),
                "formula_ast": formula.to_json(),
                "formula": formula,
                "removed_initial_aps": list(removed_initial_aps),
                "attempts": attempts
                + [{"attempt": str(attempt), "status": "success"}],
            }
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                serializable = dict(payload)
                serializable.pop("formula", None)
                cache_path.write_text(
                    json.dumps(serializable, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                _verbose_print(verbose, f"wrote cache {cache_path}")
            return payload
        except Exception as exc:
            _verbose_print(verbose, f"attempt failed: {exc}")
            last_error = exc
            attempts.append(
                {
                    "attempt": str(attempt),
                    "status": "failed",
                    "failure_reason": str(exc),
                }
            )
            if code:
                previous_answer = code
                failure_reason = str(exc)
            else:
                previous_answer = ""
                failure_reason = ""
            if isinstance(exc, urllib.error.URLError) and attempt < max_retries:
                delay_seconds = min(2 ** (attempt - 1), 10)
                _verbose_print(verbose, f"retrying after {delay_seconds}s")
                time.sleep(delay_seconds)

    raise RuntimeError(f"LTLCodeGen code generation failed: {last_error}")
