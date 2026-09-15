"""LLM-assisted learned aliases for unsupported Grounding2Route object categories."""

from __future__ import annotations

import json
import re
from typing import Mapping, Sequence

from scripts.methods.grounding2route.alias_utils import (
    normalize_category_name,
    register_entity_alias,
)
from scripts.methods.grounding2route.llm_client import GeminiClient


UNKNOWN_ENTITY_CATEGORY_RE = re.compile(
    r"UNKNOWN_ENTITY_CATEGORY:\s*([A-Za-z0-9_ -]+)\s+is not in the current object category inventory"
)


def unknown_entity_category_from_failure(failure_reason: object) -> str | None:
    if not isinstance(failure_reason, str):
        return None
    match = UNKNOWN_ENTITY_CATEGORY_RE.search(failure_reason)
    if not match:
        return None
    category = normalize_category_name(match.group(1))
    return category or None


def resolve_unknown_entity_alias(
    *,
    unknown_category: str,
    object_categories: Sequence[str],
    instruction_text: str,
    program_source: str,
    failure_reason: str,
    llm_client: GeminiClient,
) -> dict[str, object]:
    normalized_unknown = normalize_category_name(unknown_category)
    normalized_categories = sorted(
        {
            category
            for category in (normalize_category_name(item) for item in object_categories)
            if category
        }
    )
    if not normalized_unknown:
        return {"status": "skipped_empty_unknown_category"}
    if normalized_unknown in normalized_categories:
        return {"status": "skipped_category_already_supported", "unknown_category": normalized_unknown}

    prompt = f"""
You are helping maintain Grounding2Route object category aliases.

An LLM parser used an object category that is not in the current scene object
category inventory. Choose the closest supported object category ONLY if the
unknown category clearly refers to one of the supported categories.

Unknown object category:

```text
{normalized_unknown}
```

Supported object categories:

```text
{", ".join(normalized_categories)}
```

Original instruction:

```text
{instruction_text}
```

Compiled program source that failed:

```text
{program_source}
```

Grounding failure:

```text
{failure_reason}
```

Return exactly one JSON object:

```json
{{"target_category": "supported_category_or_null", "confidence": 0.0, "reason": "short reason"}}
```

Rules:

- `target_category` must be one of the supported object categories or null.
- Use null if the unknown category is genuinely absent or the match is weak.
- Do not invent categories.
- Do not return room categories.
- Prefer null over a semantically distant match.
""".strip()
    text, meta = llm_client.response_text(
        system_prompt="You map unknown Grounding2Route object categories to supported object categories. Return JSON only.",
        user_prompt=prompt,
        cache_namespace="grounding2route_entity_alias_resolver",
    )
    try:
        payload = _extract_json_object(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "llm_output_invalid",
            "unknown_category": normalized_unknown,
            "llm": meta,
            "raw_response": text,
            "error": str(exc),
        }

    target_raw = payload.get("target_category")
    target = normalize_category_name(target_raw) if isinstance(target_raw, str) else ""
    confidence = payload.get("confidence")
    if target in {"", "none", "null"}:
        target = ""
    if target not in normalized_categories:
        return {
            "status": "no_supported_alias",
            "unknown_category": normalized_unknown,
            "llm": meta,
            "raw_response": text,
            "payload": payload,
        }

    register_entity_alias(normalized_unknown, target)
    return {
        "status": "resolved",
        "unknown_category": normalized_unknown,
        "target_category": target,
        "confidence": confidence,
        "reason": payload.get("reason"),
        "llm": meta,
        "raw_response": text,
    }


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json|text)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Alias resolver response must be a JSON object.")
    return payload
