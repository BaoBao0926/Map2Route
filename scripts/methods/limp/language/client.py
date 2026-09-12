"""Gemini REST client for LIMP translation.

This intentionally mirrors ``scripts.methods.lang2ltl.translator``: it uses the
Gemini REST API directly instead of requiring the optional ``google-generativeai``
package to be installed in the active environment.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping


GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)


def configured_model(default: str = "gemini-2.5-flash") -> str:
    try:
        from scripts.methods import api_key  # type: ignore

        model = getattr(api_key, "MODEL", None)
        if isinstance(model, str) and model.strip():
            return model.strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_MODEL", default)


def configured_api_key() -> str | None:
    try:
        from scripts.methods import api_key  # type: ignore

        key = getattr(api_key, "API_KEY", None)
        if isinstance(key, str) and key.strip():
            return key.strip()
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    return key.strip() if key and key.strip() else None


def call_gemini(model: str, system_prompt: str, user_prompt: str, *, timeout: int = 90) -> str:
    api_key = configured_api_key()
    if not api_key:
        raise RuntimeError("No Gemini API key configured")

    model_name = model.removeprefix("models/")
    encoded_model = urllib.parse.quote(model_name, safe="")
    url = GEMINI_GENERATE_CONTENT_URL.format(model=encoded_model)
    url = f"{url}?key={urllib.parse.quote(api_key, safe='')}"
    request_payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 1.0,
            "maxOutputTokens": 8192,
        },
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
    if not isinstance(payload, Mapping):
        raise RuntimeError("Gemini response body must be a JSON object")

    text = _extract_gemini_text(payload)
    fenced = re.search(r"```(?:json|text|ltl)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    text = text.replace("Output:", "").strip()
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Gemini returned an empty response")
    return text


def _extract_gemini_text(response: Mapping[str, object]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("Gemini response did not contain candidates")
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
        raise RuntimeError("Gemini response did not contain text parts")
    return "".join(chunks).strip()
