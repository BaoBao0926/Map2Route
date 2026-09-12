"""LIMP two-stage translation for SemPathBench."""

from __future__ import annotations

import re
from collections.abc import Mapping

from scripts.methods.limp.grounding.scene_adapter import CandidateRegistry
from scripts.methods.limp.language.cache import cache_key, read_cache, write_cache
from scripts.methods.limp.language.client import call_gemini
from scripts.methods.limp.language.predicate_parser import encode_lifted_predicates, extract_ordered_symbols, unsupported_predicates
from scripts.methods.limp.language.prompts import (
    STAGE1_PROMPT_VERSION,
    STAGE1_SYSTEM_PROMPT,
    STAGE2_PROMPT_VERSION,
    STAGE2_SYSTEM_PROMPT,
)


def _phrase_for_category(category: str) -> str:
    return category.replace("_", " ")


def _find_phrase(text: str, phrase: str) -> tuple[int, int]:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    match = re.search(pattern, text.lower())
    return (match.start(), match.end()) if match else (-1, -1)


def heuristic_stage2_formula(instruction: Mapping[str, object], registry: CandidateRegistry) -> tuple[str, dict[str, object]]:
    text = str(instruction.get("instruction", "") or "")
    raw_mentions: list[tuple[int, int, str, str]] = []
    categories = sorted(registry.by_category, key=lambda item: len(item), reverse=True)
    for category in categories:
        phrase = _phrase_for_category(category)
        index, end = _find_phrase(text, phrase)
        if index < 0 and "_" in category:
            index, end = _find_phrase(text, category.replace("_", ""))
        if index >= 0:
            raw_mentions.append((index, end, category, category))

    accepted: list[tuple[int, int, str, str]] = []
    occupied: set[int] = set()
    for start, end, category, referent in sorted(raw_mentions, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
        span = set(range(start, end))
        if span & occupied:
            continue
        occupied.update(span)
        accepted.append((start, end, category, referent))

    mentions = sorted(accepted, key=lambda item: (item[0], item[2]))
    seen: set[str] = set()
    referents: list[str] = []
    for _index, _end, category, referent in mentions:
        if category in seen:
            continue
        seen.add(category)
        referents.append(referent)

    if not referents:
        # Keep the failure explicit while still returning syntactically valid metadata.
        return "False", {"mode": "heuristic", "status": "no_category_mentions"}

    formula = ""
    for referent in reversed(referents):
        atom = f"near[{referent}]"
        if not formula:
            formula = f"F {atom}"
        else:
            formula = f"F ( {atom} & {formula} )"
    return formula, {
        "mode": "heuristic",
        "status": "category_order_from_instruction_text",
        "referents": referents,
    }


def _llm_translate(instruction_text: str, model: str) -> tuple[str, str]:
    stage1_user = f"Input: {instruction_text}\nOutput:"
    stage1 = call_gemini(model, STAGE1_SYSTEM_PROMPT, stage1_user)
    stage2_user = (
        f"Input_instruction: {instruction_text}\n"
        f"Input_ltl: {stage1}\n"
        "Output:"
    )
    stage2 = call_gemini(model, STAGE2_SYSTEM_PROMPT, stage2_user)
    return stage1, stage2


def _llm_repair_stage2(
    instruction_text: str,
    stage1_ltl: str,
    bad_stage2_ltl: str,
    error: str,
    model: str,
) -> str:
    repair_prompt = (
        f"Input_instruction: {instruction_text}\n"
        f"Input_ltl: {stage1_ltl}\n"
        f"Previous_invalid_output: {bad_stage2_ltl}\n"
        f"Validation_error: {error}\n\n"
        "Return only one complete LTL formula. Use only complete near[...] "
        "predicates. Do not output prose, markdown, explanations, or partial "
        "predicates."
    )
    return call_gemini(model, STAGE2_SYSTEM_PROMPT, repair_prompt)


def _encode_and_validate_stage2(stage2_ltl: str) -> tuple[str, dict[str, str], list[str]]:
    errors: list[str] = []
    try:
        encoded_ltl, encoding_map = encode_lifted_predicates(stage2_ltl)
    except ValueError as exc:
        return "", {}, [str(exc)]
    if not encoding_map:
        errors.append("No complete lifted predicates were found.")
    if "[" in encoded_ltl or "]" in encoded_ltl:
        errors.append(
            "Encoded LTL still contains bracketed predicate text; the Stage-2 "
            "formula is likely malformed or truncated."
        )
    if "near[" in encoded_ltl or "pick[" in encoded_ltl or "release[" in encoded_ltl:
        errors.append("Encoded LTL still contains unencoded lifted predicates.")
    return encoded_ltl, encoding_map, errors


def translate_instruction(
    instruction: Mapping[str, object],
    registry: CandidateRegistry,
    *,
    model: str,
    mode: str,
    cache_root,
    overwrite_cache: bool = False,
) -> dict[str, object]:
    text = str(instruction.get("instruction", "") or "")
    key = cache_key(
        "limp",
        "translator_v2",
        STAGE1_PROMPT_VERSION,
        STAGE2_PROMPT_VERSION,
        model,
        mode,
        text,
    )
    if not overwrite_cache:
        cached = read_cache(cache_root, key)
        if cached is not None:
            cached["cache_status"] = "hit"
            return cached

    errors: list[str] = []
    stage1_ltl = ""
    stage2_ltl = ""
    actual_mode = mode
    if mode == "llm":
        try:
            stage1_ltl, stage2_ltl = _llm_translate(text, model)
            actual_mode = "llm"
        except Exception as exc:
            errors.append(str(exc))
            payload = {
                "requested_mode": mode,
                "mode": "llm",
                "status": "TRANSLATION_FAILED",
                "model": model,
                "stage1_prompt_version": STAGE1_PROMPT_VERSION,
                "stage2_prompt_version": STAGE2_PROMPT_VERSION,
                "raw_stage1_response": stage1_ltl,
                "raw_stage2_response": stage2_ltl,
                "parsed_stage1_ltl": stage1_ltl,
                "parsed_stage2_ltl": stage2_ltl,
                "stage1_ltl": stage1_ltl,
                "stage2_ltl": stage2_ltl,
                "encoded_ltl": "",
                "encoding_map": {},
                "ordered_symbols": [],
                "unsupported_predicates": [],
                "translation_failure_reason": str(exc),
                "errors": errors,
                "cache_status": "miss",
            }
            write_cache(cache_root, key, payload)
            return payload

    heuristic_info: dict[str, object] = {}
    if mode == "heuristic":
        stage2_ltl, heuristic_info = heuristic_stage2_formula(instruction, registry)
        stage1_ltl = heuristic_info.get("status", "heuristic")  # type: ignore[assignment]
        actual_mode = "heuristic"
    elif not stage2_ltl:
        payload = {
            "requested_mode": mode,
            "mode": mode,
            "status": "TRANSLATION_FAILED",
            "model": model,
            "stage1_prompt_version": STAGE1_PROMPT_VERSION,
            "stage2_prompt_version": STAGE2_PROMPT_VERSION,
            "raw_stage1_response": stage1_ltl,
            "raw_stage2_response": stage2_ltl,
            "parsed_stage1_ltl": stage1_ltl,
            "parsed_stage2_ltl": stage2_ltl,
            "stage1_ltl": stage1_ltl,
            "stage2_ltl": stage2_ltl,
            "encoded_ltl": "",
            "encoding_map": {},
            "ordered_symbols": [],
            "unsupported_predicates": [],
            "translation_failure_reason": "No Stage-2 LTL was produced.",
            "errors": errors,
            "cache_status": "miss",
        }
        write_cache(cache_root, key, payload)
        return payload

    encoded_ltl, encoding_map, validation_errors = _encode_and_validate_stage2(stage2_ltl)
    errors.extend(validation_errors)
    if mode == "llm" and validation_errors:
        try:
            repaired_stage2 = _llm_repair_stage2(
                text,
                stage1_ltl,
                stage2_ltl,
                "; ".join(validation_errors),
                model,
            )
            repaired_encoded, repaired_map, repaired_errors = _encode_and_validate_stage2(
                repaired_stage2
            )
            if not repaired_errors:
                stage2_ltl = repaired_stage2
                encoded_ltl = repaired_encoded
                encoding_map = repaired_map
                errors = []
            else:
                errors.extend(f"repair: {error}" for error in repaired_errors)
        except Exception as exc:
            errors.append(f"repair_failed: {exc}")

    translation_ok = bool(encoding_map) and not errors
    payload = {
        "requested_mode": mode,
        "mode": actual_mode,
        "status": "ok" if translation_ok else "TRANSLATION_FAILED",
        "model": model,
        "stage1_prompt_version": STAGE1_PROMPT_VERSION,
        "stage2_prompt_version": STAGE2_PROMPT_VERSION,
        "raw_stage1_response": stage1_ltl,
        "raw_stage2_response": stage2_ltl,
        "parsed_stage1_ltl": stage1_ltl,
        "parsed_stage2_ltl": stage2_ltl,
        "stage1_ltl": stage1_ltl,
        "stage2_ltl": stage2_ltl,
        "encoded_ltl": encoded_ltl,
        "encoding_map": encoding_map,
        "ordered_symbols": extract_ordered_symbols(encoded_ltl, encoding_map),
        "unsupported_predicates": unsupported_predicates(encoding_map),
        "translation_failure_reason": None if translation_ok else "; ".join(errors) or "No lifted predicates encoded.",
        "errors": errors,
        "heuristic_info": heuristic_info,
        "cache_status": "miss",
    }
    write_cache(cache_root, key, payload)
    return payload
