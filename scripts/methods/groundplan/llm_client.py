"""Gemini REST helper for GroundPlan parsing."""

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

from scripts.methods.groundplan.config import DEFAULT_MODEL


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


def prompt_hash(*parts: object) -> str:
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_text(payload: Mapping[str, object]) -> str:
    candidates = payload.get("candidates")
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
    fenced = re.search(r"```(?:text|json)?\s*(.*?)```", text, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else text


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
        self._trace: list[dict[str, object]] = []

    def trace(self) -> list[dict[str, object]]:
        """Return the complete prompt/response trace for one episode.

        The runner persists this in the episode's ``.steps.json`` so future
        analyses do not need to reverse-engineer a cached prompt hash.
        """

        return [dict(entry) for entry in self._trace]

    def response_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        cache_namespace: str,
    ) -> tuple[str, dict[str, object]]:
        key = prompt_hash(cache_namespace, self.model, system_prompt, user_prompt)
        cache_path = self.cache_root / f"{key}.json" if self.cache_root else None
        if cache_path and cache_path.exists() and not self.overwrite_cache:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, Mapping) and isinstance(cached.get("text"), str):
                text = str(cached["text"])
                trace_entry = self._trace_entry(
                    cache_namespace=cache_namespace,
                    cache_key=key,
                    cache_path=cache_path,
                    cache="hit",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    text=text,
                    raw_response=cached.get("raw_response"),
                )
                self._trace.append(trace_entry)
                return text, {"cache": "hit", "cache_key": key, "trace_index": len(self._trace) - 1}

        api_key = configured_api_key()
        if not api_key:
            raise RuntimeError("Gemini API key is required for GroundPlan LLM parser mode.")
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
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("Gemini response body must be a JSON object.")
        text = _extract_text(payload)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "cache_namespace": cache_namespace,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "text": text,
                        "raw_response": payload,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        trace_entry = self._trace_entry(
            cache_namespace=cache_namespace,
            cache_key=key,
            cache_path=cache_path,
            cache="miss",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            text=text,
            raw_response=payload,
        )
        self._trace.append(trace_entry)
        return text, {"cache": "miss", "cache_key": key, "trace_index": len(self._trace) - 1}

    def _trace_entry(
        self,
        *,
        cache_namespace: str,
        cache_key: str,
        cache_path: Path | None,
        cache: str,
        system_prompt: str,
        user_prompt: str,
        text: str,
        raw_response: object,
    ) -> dict[str, object]:
        return {
            "call_index": len(self._trace),
            "model": self.model,
            "cache_namespace": cache_namespace,
            "cache": cache,
            "cache_key": cache_key,
            "cache_path": str(cache_path) if cache_path else None,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_text": text,
            "raw_response": raw_response,
        }
