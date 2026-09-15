"""Gemini native function-calling transport for Grounding2Route ToolCall."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Mapping, Sequence

from scripts.methods.grounding2route.llm_client import (
    GEMINI_GENERATE_CONTENT_URL,
    GeminiClient,
    configured_api_key,
    prompt_hash,
)


def tool_turn(
    client: GeminiClient,
    *,
    system_prompt: str,
    contents: Sequence[Mapping[str, object]],
    function_declarations: Sequence[Mapping[str, object]],
    cache_namespace: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run one cached Gemini tool turn with the client's model/configuration."""

    # Test doubles may implement the same small interface without networking.
    native = getattr(client, "tool_turn", None)
    if callable(native):
        return native(
            system_prompt=system_prompt,
            contents=contents,
            function_declarations=function_declarations,
            cache_namespace=cache_namespace,
        )

    key = prompt_hash(cache_namespace, client.model, system_prompt, list(contents), list(function_declarations))
    cache_path = client.cache_root / f"{key}.json" if client.cache_root else None
    if cache_path and cache_path.exists() and not client.overwrite_cache:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        content = cached.get("content") if isinstance(cached, Mapping) else None
        if isinstance(content, Mapping):
            normalized = dict(content)
            _append_trace(client, cache_namespace, key, cache_path, "hit", system_prompt, contents, function_declarations, normalized, cached.get("raw_response"))
            usage = cached.get("usage_metadata")
            return normalized, {
                "cache": "hit", "cache_key": key,
                "trace_index": len(client.trace()) - 1,
                "usage_metadata": dict(usage) if isinstance(usage, Mapping) else {},
            }

    api_key = configured_api_key()
    if not api_key:
        raise RuntimeError("Gemini API key is required for Grounding2Route ToolCall mode.")
    model_name = client.model.removeprefix("models/")
    url = GEMINI_GENERATE_CONTENT_URL.format(model=urllib.parse.quote(model_name, safe=""))
    url = f"{url}?key={urllib.parse.quote(api_key, safe='')}"
    request_payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": list(contents),
        "tools": [{"functionDeclarations": list(function_declarations)}],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {"temperature": 0},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=client.timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: {error.code} {body}") from error
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("Gemini response body must be a JSON object.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], Mapping):
        raise ValueError("Gemini tool response did not contain a candidate.")
    content = candidates[0].get("content")
    if not isinstance(content, Mapping):
        raise ValueError("Gemini tool response candidate did not contain content.")
    normalized = dict(content)
    usage = payload.get("usageMetadata")
    usage_metadata = dict(usage) if isinstance(usage, Mapping) else {}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "model": client.model, "cache_namespace": cache_namespace,
            "system_prompt": system_prompt, "contents": list(contents),
            "function_declarations": list(function_declarations),
            "content": normalized, "usage_metadata": usage_metadata,
            "raw_response": payload,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    _append_trace(client, cache_namespace, key, cache_path, "miss", system_prompt, contents, function_declarations, normalized, payload)
    return normalized, {
        "cache": "miss", "cache_key": key,
        "trace_index": len(client.trace()) - 1,
        "usage_metadata": usage_metadata,
    }


def _append_trace(
    client: GeminiClient,
    namespace: str,
    key: str,
    cache_path: object,
    cache: str,
    system_prompt: str,
    contents: Sequence[Mapping[str, object]],
    declarations: Sequence[Mapping[str, object]],
    content: Mapping[str, object],
    raw_response: object,
) -> None:
    # GeminiClient deliberately owns this per-episode trace; this adapter only
    # adds the native-tool turn in the same persisted format.
    client._trace.append({  # noqa: SLF001 - companion transport for GeminiClient.
        "call_index": len(client._trace),  # noqa: SLF001
        "model": client.model,
        "cache_namespace": namespace,
        "cache": cache,
        "cache_key": key,
        "cache_path": str(cache_path) if cache_path else None,
        "system_prompt": system_prompt,
        "contents": list(contents),
        "function_declarations": list(declarations),
        "response_content": dict(content),
        "raw_response": raw_response,
    })
