"""Gemini REST client helpers for SayPlan."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping

from scripts.methods.sayplan.config import (
    DEFAULT_GEMINI_TEMPERATURE,
    DEFAULT_GEMINI_TIMEOUT_SECONDS,
)


GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)


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
    model = _api_key_module_value("MODEL")
    if not model:
        raise RuntimeError("MODEL in scripts/methods/api_key.py is required for SayPlan.")
    return model


def configured_api_key() -> str | None:
    return _api_key_module_value("API_KEY") or os.environ.get("GEMINI_API_KEY")


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
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                chunks.append(str(part["text"]))
    if not chunks:
        raise ValueError("Gemini response did not contain text parts.")
    return "".join(chunks).strip()


def strip_fenced_text(text: str) -> str:
    fenced = re.search(r"```(?:json|text)?\s*(.*?)```", text, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def extract_json_object(text: str) -> dict[str, object]:
    cleaned = strip_fenced_text(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("SayPlan Gemini response must be a JSON object.")
    return payload


def gemini_response_text(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    timeout: float = DEFAULT_GEMINI_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_GEMINI_TEMPERATURE,
) -> str:
    key = api_key or configured_api_key()
    if not key:
        raise RuntimeError(
            "API_KEY in scripts/methods/api_key.py or GEMINI_API_KEY is required "
            "for SayPlan."
        )
    model_name = model.removeprefix("models/")
    encoded_model = urllib.parse.quote(model_name, safe="")
    url = GEMINI_GENERATE_CONTENT_URL.format(model=encoded_model)
    url = f"{url}?key={urllib.parse.quote(key, safe='')}"
    request_payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: {error.code} {body}") from error
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Gemini response body must be a JSON object.")
    return strip_fenced_text(_extract_gemini_text(payload))


def gemini_response_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> tuple[dict[str, object], str]:
    text = gemini_response_text(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_key=api_key,
    )
    return extract_json_object(text), text

