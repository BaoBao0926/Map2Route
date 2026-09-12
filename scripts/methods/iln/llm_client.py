"""Gemini helper with file cache for the ILN adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping

from scripts.methods.iln.config import DEFAULT_MODEL


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
    return _api_key_module_value("MODEL") or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


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
    text = "".join(chunks).strip()
    fenced = re.search(r"```(?:json|text)?\s*(.*?)```", text, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def prompt_hash(*parts: object) -> str:
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GeminiClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        cache_root: Path | None = None,
        overwrite_cache: bool = False,
        timeout: float = 90.0,
    ) -> None:
        self.model = model or configured_model()
        self.cache_root = cache_root
        self.overwrite_cache = overwrite_cache
        self.timeout = timeout

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_root is None:
            return None
        return self.cache_root / f"{key}.json"

    def response_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        cache_namespace: str,
    ) -> tuple[str, dict[str, object]]:
        key = prompt_hash(cache_namespace, self.model, system_prompt, user_prompt)
        cache_path = self._cache_path(key)
        if cache_path is not None and cache_path.exists() and not self.overwrite_cache:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and isinstance(payload.get("text"), str):
                return str(payload["text"]), {"cache": "hit", "cache_key": key}

        api_key = configured_api_key()
        if not api_key:
            raise RuntimeError("Gemini API key is required for ILN LLM mode.")
        model_name = self.model.removeprefix("models/")
        url = GEMINI_GENERATE_CONTENT_URL.format(model=urllib.parse.quote(model_name, safe=""))
        url = f"{url}?key={urllib.parse.quote(api_key, safe='')}"
        request_payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(request_payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API request failed: {error.code} {body}") from error
        response_payload = json.loads(raw)
        if not isinstance(response_payload, Mapping):
            raise ValueError("Gemini response body must be a JSON object.")
        text = _extract_gemini_text(response_payload)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "cache_namespace": cache_namespace,
                        "text": text,
                        "raw_response": response_payload,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return text, {"cache": "miss", "cache_key": key}


def extract_json_object(text: str) -> dict[str, object]:
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
        raise ValueError("LLM JSON response must be an object.")
    return payload

