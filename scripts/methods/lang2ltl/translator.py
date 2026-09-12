"""Self-contained Lang2LTL-style translation for SemPathBench."""

from __future__ import annotations

import json
import math
import os
import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

from scripts.methods.lang2ltl.ltl import Formula, atomic_propositions, ap, ltl_and, ltl_eventually
from scripts.methods.lang2ltl.ltl_parser import (
    parse_prefix_ltl,
    substitute_formula_aps,
)
from scripts.methods.lang2ltl.map_features import MapSymbols, SemanticEntity, obj2sem


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4"
GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
GEMINI_EMBED_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:embedContent"
)
TRANSLATION_MODES = ("auto", "llm", "heuristic")
LLM_BACKENDS = ("gemini", "openai")
GROUNDING_MODES = ("embedding", "local")
EMBEDDING_BACKENDS = ("gemini", "openai")
SYMBOLIC_TRANSLATORS = ("llm", "t5")
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
T5_PREFIX = "translate English to Linear Temporal Logic: "
# Match the original Lang2LTL proposition pool and avoid Spot prefix operators
# such as i/e/U/F/G/X.
SYMBOLIC_PROPS = ("a", "b", "c", "d", "h", "j", "k", "l", "n", "o", "p", "q", "r", "s", "y", "z")

# A Lang2LTL batch runs episode work in threads.  Transformers model loading
# mutates process-global state and is not safe when several threads call
# ``from_pretrained``/``.to(device)`` at once; that race leaves some modules on
# the meta device.  Keep one runtime per checkpoint and serialize inference on
# it, while LLM requests, grounding, planning, and metrics remain parallel.
_T5_RUNTIME_CACHE: dict[Path, tuple[object, object, object]] = {}
_T5_RUNTIME_LOCK = threading.Lock()
_T5_INFERENCE_LOCK = threading.Lock()

RER_SYSTEM_PROMPT = """You are the referring-expression recognizer in a Lang2LTL-style robot navigation system.

Given a route instruction, extract exact noun phrases from the instruction that refer to target rooms, objects, regions, or landmarks.

Return JSON only:
{"referring_expressions": ["exact phrase 1", "exact phrase 2"]}

Rules:
- Use exact text spans from the instruction when possible.
- Include target objects and rooms.
- Exclude pure spatial relation words such as left, right, near, beside, before, after unless they are part of the noun phrase.
- Exclude starting-pose phrases such as "start", "beside you", "near you".
- Preserve the order in which expressions appear.
"""

SYMBOLIC_TRANSLATION_SYSTEM_PROMPT = """You are the symbolic translation module of Lang2LTL.

Translate a symbolic navigation instruction into a Spot-style prefix LTL formula.
Use only the symbolic propositions listed in the placeholder map, such as a, b, c.
Return only the LTL formula, no markdown and no explanation.

Useful patterns:
- "go to a" -> F a
- "visit a then b" -> F & a F b
- "go to a before b" -> & U ! b a F b
- "avoid a" -> G ! a
- "go to a and avoid b" -> & F a G ! b
- "either a or b" -> | F a F b

If the instruction contains soft preferences that cannot be represented exactly, ignore those preferences and translate the hard visit/avoid requirements.
"""

SYMBOLIC_REPAIR_SYSTEM_PROMPT = """You repair malformed Spot-style prefix LTL.

Return only one syntactically complete prefix LTL formula.
Use only the allowed symbolic propositions.
Do not output markdown or explanation.

Valid examples:
- F a
- F & a F b
- & F a G ! b
- | F a F b
"""


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
    return _api_key_module_value("API_KEY") or os.environ.get("GEMINI_API_KEY")


def configured_openai_api_key() -> str | None:
    return _api_key_module_value("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")


def configured_embedding_model() -> str:
    return (
        _api_key_module_value("EMBEDDING_MODEL")
        or os.environ.get("GEMINI_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )


def _verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[lang2ltl:translator] {message}", flush=True)


class SymbolicLTLParseError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, str]],
        last_formula: str,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_formula = last_formula


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
            "for Lang2LTL LLM translation."
        )
    model_name = model.removeprefix("models/")
    encoded_model = urllib.parse.quote(model_name, safe="")
    url = GEMINI_GENERATE_CONTENT_URL.format(model=encoded_model)
    url = f"{url}?key={urllib.parse.quote(key, safe='')}"
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
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: {error.code} {body}") from error
    except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
        raise RuntimeError(f"Gemini API request failed: {error}") from error
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Gemini response body must be a JSON object.")
    text = _extract_gemini_text(payload)
    fenced = re.search(r"```(?:json|text|ltl)?\s*(.*?)```", text, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def openai_response_text(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> str:
    key = api_key or configured_openai_api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for the original Lang2LTL OpenAI "
            "LLM backend."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for --llm-backend openai. "
            "Install it with `pip install openai`."
        ) from exc

    client = OpenAI(api_key=key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content or ""
    fenced = re.search(r"```(?:json|text|ltl)?\s*(.*?)```", text, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def llm_response_text(
    *,
    backend: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> str:
    if backend == "gemini":
        return gemini_response_text(
            model=model,
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    if backend == "openai":
        return openai_response_text(
            model=model,
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    raise ValueError(f"Unsupported LLM backend: {backend}")


def _clean_token(value: str) -> str:
    return "_".join(part for part in re.split(r"[^a-z0-9]+", value.lower()) if part)


def _word_set(text: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", text.lower()) if part}


def _entity_text(entity: SemanticEntity, semantics: Mapping[str, object]) -> str:
    fields = [
        entity.ap,
        entity.category,
        entity.label,
        entity.room_category or "",
        " ".join(str(item) for item in semantics.get("attributes", []) if item),
    ]
    return " ".join(fields)


def _extract_json_list(text: str, key: str) -> list[str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[(.*?)\]", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(f"[{match.group(1)}]")
        except json.JSONDecodeError:
            return []
        return [str(item).strip() for item in payload if str(item).strip()]
    if not isinstance(payload, Mapping):
        return []
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def extract_referring_expressions_llm(
    instruction: str,
    *,
    model: str,
    llm_backend: str = "gemini",
    api_key: str | None = None,
) -> tuple[list[str], str]:
    response = llm_response_text(
        backend=llm_backend,
        model=model,
        api_key=api_key,
        system_prompt=RER_SYSTEM_PROMPT,
        user_prompt=f"Instruction:\n{instruction.strip()}",
    )
    expressions = _extract_json_list(response, "referring_expressions")
    deduped: list[str] = []
    seen: set[str] = set()
    for expression in expressions:
        key = expression.lower()
        if key not in seen:
            deduped.append(expression)
            seen.add(key)
    return deduped, response


def extract_referring_expressions_heuristic(
    instruction: str,
    symbols: MapSymbols,
) -> list[str]:
    lowered = instruction.lower()
    matches: list[tuple[int, str]] = []
    categories = {
        entity.category.replace("_", " ")
        for entity in (*symbols.room_entities, *symbols.object_entities)
    }
    for category in sorted(categories, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(category)}s?\b")
        for match in pattern.finditer(lowered):
            matches.append((match.start(), match.group(0)))
    matches.sort(key=lambda item: item[0])
    deduped: list[str] = []
    seen: set[str] = set()
    for _position, expression in matches:
        if expression not in seen:
            deduped.append(expression)
            seen.add(expression)
    return deduped


def serialize_ap_description(entity: SemanticEntity) -> str:
    """Serialize a map AP as natural language for embedding retrieval."""
    if entity.kind == "room":
        return f"a {entity.category.replace('_', ' ')}"

    category = entity.category.replace("_", " ")
    room = entity.room_category.replace("_", " ") if entity.room_category else ""
    label = entity.label.replace("|", " ") if entity.label else ""
    attributes = " ".join(
        str(attribute).replace("_", " ").replace("=", " ")
        for attribute in entity.attributes
        if attribute
    )
    parts = [f"a {category}"]
    if room:
        parts.append(f"located in a {room}")
    if label and label.lower() != category:
        parts.append(f"with label {label}")
    if attributes:
        parts.append(f"with attributes {attributes}")
    return " ".join(parts)


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    dot = 0.0
    norm_first = 0.0
    norm_second = 0.0
    for a, b in zip(first, second):
        dot += float(a) * float(b)
        norm_first += float(a) * float(a)
        norm_second += float(b) * float(b)
    if norm_first <= 0.0 or norm_second <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_first * norm_second)


def gemini_embed_text(
    text: str,
    *,
    model: str,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> list[float]:
    key = api_key or configured_api_key()
    if not key:
        raise RuntimeError(
            "API_KEY in scripts/methods/api_key.py or GEMINI_API_KEY is required "
            "for embedding-based Lang2LTL grounding."
        )
    model_name = model.removeprefix("models/")
    encoded_model = urllib.parse.quote(model_name, safe="")
    url = GEMINI_EMBED_CONTENT_URL.format(model=encoded_model)
    url = f"{url}?key={urllib.parse.quote(key, safe='')}"
    request_payload = {
        "model": f"models/{model_name}",
        "content": {"parts": [{"text": text}]},
        "taskType": "SEMANTIC_SIMILARITY",
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
        raise RuntimeError(f"Gemini embedding request failed: {error.code} {body}") from error
    except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
        raise RuntimeError(f"Gemini embedding request failed: {error}") from error
    payload = json.loads(raw)
    embedding = payload.get("embedding") if isinstance(payload, Mapping) else None
    values = embedding.get("values") if isinstance(embedding, Mapping) else None
    if not isinstance(values, list):
        raise ValueError("Gemini embedding response did not contain embedding.values.")
    return [float(value) for value in values]


def openai_embed_text(
    text: str,
    *,
    model: str,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> list[float]:
    key = api_key or configured_openai_api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for the original Lang2LTL OpenAI "
            "embedding backend."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for --embedding-backend openai. "
            "Install it with `pip install openai`."
        ) from exc

    client = OpenAI(api_key=key, timeout=timeout)
    response = client.embeddings.create(model=model, input=text)
    return [float(value) for value in response.data[0].embedding]


def _load_embedding_cache(
    cache_path: Path | None,
    *,
    backend: str,
    model: str,
) -> dict[str, object]:
    if cache_path is None or not cache_path.exists():
        return {"backend": backend, "model": model, "items": {}}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"backend": backend, "model": model, "items": {}}
    if not isinstance(payload, dict):
        return {"backend": backend, "model": model, "items": {}}
    if payload.get("backend") != backend or payload.get("model") != model:
        return {"backend": backend, "model": model, "items": {}}
    if not isinstance(payload.get("items"), dict):
        payload["items"] = {}
    return payload


def _save_embedding_cache(cache_path: Path | None, payload: Mapping[str, object]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _embedding_from_cache_or_backend(
    cache_payload: dict[str, object],
    key: str,
    text: str,
    *,
    backend: str,
    model: str,
    api_key: str | None = None,
) -> list[float]:
    items = cache_payload.setdefault("items", {})
    if not isinstance(items, dict):
        items = {}
        cache_payload["items"] = items
    cached = items.get(key)
    if isinstance(cached, Mapping) and cached.get("text") == text:
        embedding = cached.get("embedding")
        if isinstance(embedding, list) and embedding:
            return [float(value) for value in embedding]

    if backend == "gemini":
        embedding = gemini_embed_text(text, model=model, api_key=api_key)
    elif backend == "openai":
        embedding = openai_embed_text(text, model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")
    items[key] = {"text": text, "embedding": embedding}
    return embedding


def ground_referring_expressions_embedding(
    expressions: Sequence[str],
    symbols: MapSymbols,
    *,
    topk: int = 3,
    embedding_backend: str = "gemini",
    embedding_model: str | None = None,
    embedding_cache_path: Path | None = None,
    api_key: str | None = None,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, str], dict[str, object]]:
    if embedding_backend not in EMBEDDING_BACKENDS:
        raise ValueError(f"Unsupported embedding backend: {embedding_backend}")
    embedding_model = embedding_model or configured_embedding_model()
    cache_payload = _load_embedding_cache(
        embedding_cache_path,
        backend=embedding_backend,
        model=embedding_model,
    )

    entities = tuple(symbols.room_entities) + tuple(symbols.object_entities)
    descriptions = {entity.ap: serialize_ap_description(entity) for entity in entities}
    entity_by_ap = {entity.ap: entity for entity in entities}
    ap_embeddings = {
        entity.ap: _embedding_from_cache_or_backend(
            cache_payload,
            f"ap:{entity.ap}",
            descriptions[entity.ap],
            backend=embedding_backend,
            model=embedding_model,
            api_key=api_key,
        )
        for entity in entities
    }

    result: dict[str, list[dict[str, object]]] = {}
    for expression in expressions:
        re_embedding = _embedding_from_cache_or_backend(
            cache_payload,
            f"re:{expression.strip().lower()}",
            expression,
            backend=embedding_backend,
            model=embedding_model,
            api_key=api_key,
        )
        ranked: list[dict[str, object]] = []
        for ap_name, ap_embedding in ap_embeddings.items():
            entity = entity_by_ap[ap_name]
            ranked.append(
                {
                    "ap": ap_name,
                    "kind": entity.kind,
                    "category": entity.category,
                    "label": entity.label,
                    "room_category": entity.room_category,
                    "score": cosine_similarity(re_embedding, ap_embedding),
                    "semantic_description": descriptions[ap_name],
                }
            )
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["ap"])))
        result[expression] = ranked[:topk]

    _save_embedding_cache(embedding_cache_path, cache_payload)
    backend_info = {
        "grounding_mode": "embedding",
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "embedding_cache_path": str(embedding_cache_path) if embedding_cache_path else None,
        "candidate_count": len(entities),
    }
    return result, descriptions, backend_info


def ground_referring_expressions_local(
    expressions: Sequence[str],
    symbols: MapSymbols,
    *,
    topk: int = 3,
) -> dict[str, list[dict[str, object]]]:
    semantics = obj2sem(symbols)
    entities = tuple(symbols.room_entities) + tuple(symbols.object_entities)
    result: dict[str, list[dict[str, object]]] = {}
    for expression in expressions:
        expression_words = _word_set(expression)
        ranked: list[dict[str, object]] = []
        for entity in entities:
            sem = semantics.get(entity.ap, {})
            entity_text = _entity_text(entity, sem)
            entity_words = _word_set(entity_text)
            overlap = len(expression_words & entity_words)
            category_phrase = entity.category.replace("_", " ")
            score = float(overlap)
            if category_phrase in expression.lower():
                score += 4.0
            if entity.room_category and entity.room_category.replace("_", " ") in expression.lower():
                score += 2.0
            if _clean_token(expression) and _clean_token(expression) in _clean_token(entity_text):
                score += 1.0
            distance_tiebreaker = math.hypot(entity.center[0], entity.center[1]) * 1e-9
            ranked.append(
                {
                    "ap": entity.ap,
                    "kind": entity.kind,
                    "category": entity.category,
                    "label": entity.label,
                    "room_category": entity.room_category,
                    "score": score,
                    "_sort": -score + distance_tiebreaker,
                }
            )
        ranked.sort(key=lambda item: (float(item["_sort"]), str(item["ap"])))
        result[expression] = [
            {key: value for key, value in item.items() if key != "_sort"}
            for item in ranked[:topk]
        ]
    return result


def replace_phrases(text: str, replacements: Mapping[str, str]) -> str:
    output = text
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        output = re.sub(re.escape(source), target, output, flags=re.IGNORECASE)
    return output


def build_placeholder_maps(grounded_aps: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    unique_aps = list(dict.fromkeys(grounded_aps))
    if len(unique_aps) > len(SYMBOLIC_PROPS):
        raise ValueError(f"Too many grounded APs for symbolic props: {len(unique_aps)}")
    placeholder = {ap_name: SYMBOLIC_PROPS[index] for index, ap_name in enumerate(unique_aps)}
    inverse = {value: key for key, value in placeholder.items()}
    return placeholder, inverse


def symbolic_translate_llm(
    *,
    symbolic_utterance: str,
    placeholder_map: Mapping[str, str],
    grounded_metadata: Mapping[str, object],
    model: str,
    llm_backend: str = "gemini",
    api_key: str | None = None,
) -> tuple[str, str]:
    user_prompt = (
        "Symbolic utterance:\n"
        f"{symbolic_utterance.strip()}\n\n"
        "Placeholder map from grounded AP to symbolic proposition:\n"
        f"{json.dumps(placeholder_map, indent=2, ensure_ascii=False)}\n\n"
        "Grounding metadata:\n"
        f"{json.dumps(grounded_metadata, indent=2, ensure_ascii=False)}\n\n"
        "Return only one prefix LTL formula over the symbolic propositions."
    )
    response = llm_response_text(
        backend=llm_backend,
        model=model,
        api_key=api_key,
        system_prompt=SYMBOLIC_TRANSLATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return response.strip(), response


def symbolic_translate_t5(
    *,
    symbolic_utterance: str,
    model_path: Path,
) -> tuple[str, str]:
    """Run the original Lang2LTL T5 symbolic translator checkpoint."""
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Lang2LTL T5 checkpoint not found: {model_path}")

    with _T5_RUNTIME_LOCK:
        runtime = _T5_RUNTIME_CACHE.get(model_path)
        if runtime is None:
            # Avoid importing torchvision through recent transformers versions.
            # The downloaded Lang2LTL T5 checkpoint is text-only.
            os.environ.setdefault("USE_TORCHVISION", "0")
            os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
            try:
                import torch
                import transformers.utils.import_utils as transformers_import_utils
            except ImportError as exc:
                raise RuntimeError(
                    "Could not import the Lang2LTL T5 inference dependencies. It "
                    "requires torch, transformers, and a compatible tokenizer backend."
                ) from exc

            transformers_import_utils._torchvision_available = False
            transformers_import_utils._torchvision_version = None
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config
                from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
            except ImportError as exc:
                raise RuntimeError(
                    "Could not import the Lang2LTL T5 inference dependencies after "
                    "disabling torchvision. Check the installed transformers package."
                ) from exc
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            except ValueError as exc:
                model_file = model_path / "pytorch_model.bin"
                if not model_file.is_file() or "torch.load" not in str(exc):
                    raise
                config = T5Config.from_pretrained(str(model_path))
                model = T5ForConditionalGeneration(config)
                state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
                # Some Transformers/PyTorch combinations construct this
                # fallback model on the meta device.  ``assign=True`` replaces
                # those meta parameters with checkpoint tensors instead of
                # performing a no-op copy (and avoids the subsequent .to()
                # meta-tensor failure).
                model.load_state_dict(state_dict, assign=True)
            model = model.to(device)
            model.eval()
            runtime = (torch, tokenizer, model)
            _T5_RUNTIME_CACHE[model_path] = runtime

    torch, tokenizer, model = runtime
    with _T5_INFERENCE_LOCK:
        inputs = tokenizer(
            [f"{T5_PREFIX}{symbolic_utterance.strip()}"],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                max_new_tokens=256,
            )
        output = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    if not output or not str(output[0]).strip():
        raise ValueError("Lang2LTL T5 translator returned no formula.")
    formula = str(output[0]).strip()
    return formula, formula


def repair_symbolic_ltl_llm(
    *,
    symbolic_utterance: str,
    placeholder_map: Mapping[str, str],
    bad_ltl: str,
    parse_error: str,
    model: str,
    llm_backend: str = "gemini",
    api_key: str | None = None,
) -> tuple[str, str]:
    user_prompt = (
        "The previous symbolic LTL was malformed.\n\n"
        "Symbolic utterance:\n"
        f"{symbolic_utterance.strip()}\n\n"
        "Allowed propositions:\n"
        f"{json.dumps(sorted(placeholder_map.values()), ensure_ascii=False)}\n\n"
        "Placeholder map from grounded AP to symbolic proposition:\n"
        f"{json.dumps(placeholder_map, indent=2, ensure_ascii=False)}\n\n"
        "Malformed LTL:\n"
        f"{bad_ltl.strip()}\n\n"
        "Parser error:\n"
        f"{parse_error}\n\n"
        "Return a corrected complete prefix LTL formula only."
    )
    response = llm_response_text(
        backend=llm_backend,
        model=model,
        api_key=api_key,
        system_prompt=SYMBOLIC_REPAIR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return response.strip(), response


def parse_symbolic_ltl_with_repair(
    *,
    symbolic_ltl: str,
    symbolic_utterance: str,
    placeholder_map: Mapping[str, str],
    model: str,
    llm_backend: str = "gemini",
    api_key: str | None = None,
    max_repairs: int = 2,
    verbose: bool = False,
) -> tuple[str, str, object, list[dict[str, str]]]:
    attempts: list[dict[str, str]] = []
    candidate = symbolic_ltl
    raw_response = symbolic_ltl
    for attempt_index in range(max_repairs + 1):
        try:
            parse_result = parse_prefix_ltl(candidate)
            return candidate, raw_response, parse_result, attempts
        except Exception as exc:
            attempts.append(
                {
                    "attempt": str(attempt_index),
                    "bad_ltl": candidate,
                    "error": str(exc),
                }
            )
            _verbose_print(
                verbose,
                f"symbolic LTL parse failed attempt={attempt_index}: {exc}; ltl={candidate}",
            )
            if attempt_index >= max_repairs:
                raise SymbolicLTLParseError(
                    "Failed to parse symbolic LTL after repair attempts. "
                    f"Last formula: {candidate!r}; last error: {exc}",
                    attempts=attempts,
                    last_formula=candidate,
                ) from exc
            try:
                candidate, raw_response = repair_symbolic_ltl_llm(
                    symbolic_utterance=symbolic_utterance,
                    placeholder_map=placeholder_map,
                    bad_ltl=candidate,
                    parse_error=str(exc),
                    model=model,
                    llm_backend=llm_backend,
                    api_key=api_key,
                )
            except RuntimeError as repair_exc:
                attempts.append(
                    {
                        "attempt": str(attempt_index),
                        "bad_ltl": candidate,
                        "error": f"repair_failed: {repair_exc}",
                    }
                )
                raise SymbolicLTLParseError(
                    "Failed to repair malformed symbolic LTL. "
                    f"Last formula: {candidate!r}; repair error: {repair_exc}",
                    attempts=attempts,
                    last_formula=candidate,
                ) from repair_exc


def _sequential_eventual_formula(aps: Sequence[str]) -> Formula:
    if not aps:
        return Formula("true")
    result = ap(aps[-1])
    for ap_name in reversed(aps[:-1]):
        result = ltl_and(ap(ap_name), ltl_eventually(result))
    return ltl_eventually(result).simplify() if len(aps) == 1 else ltl_eventually(result).simplify()


def heuristic_translate(
    instruction: str,
    symbols: MapSymbols,
    *,
    topk: int = 3,
) -> dict[str, object]:
    expressions = extract_referring_expressions_heuristic(instruction, symbols)
    grounding = ground_referring_expressions_local(expressions, symbols, topk=topk)
    selected_aps = [
        candidates[0]["ap"]
        for candidates in grounding.values()
        if candidates and float(candidates[0].get("score", 0.0)) > 0.0
    ]
    if not selected_aps:
        first_object = next(iter(symbols.object_entities), None)
        first_room = next(iter(symbols.room_entities), None)
        fallback = first_object or first_room
        if fallback is not None:
            selected_aps = [fallback.ap]
    formula = _sequential_eventual_formula(selected_aps)
    return {
        "mode": "heuristic",
        "model": None,
        "referring_expressions": expressions,
        "rer_raw_response": None,
        "grounding": grounding,
        "grounding_selected": selected_aps,
        "grounding_mode": "local",
        "grounding_backend": {
            "grounding_mode": "local",
            "embedding_backend": None,
            "embedding_model": None,
            "embedding_cache_path": None,
        },
        "ap_semantic_descriptions": {
            entity.ap: serialize_ap_description(entity)
            for entity in (*symbols.room_entities, *symbols.object_entities)
        },
        "grounded_utterance": instruction,
        "placeholder_map": {},
        "symbolic_utterance": instruction,
        "symbolic_ltl": str(formula),
        "grounded_ltl": str(formula),
        "formula": formula,
        "formula_ast": formula.to_json(),
        "parse_warnings": [],
        "fallback_reason": "local category-match heuristic",
    }


def validate_formula_aps(formula: Formula, symbols: MapSymbols) -> None:
    valid_aps = {entity.ap for entity in (*symbols.room_entities, *symbols.object_entities)}
    unknown = sorted(atomic_propositions(formula) - valid_aps)
    if unknown:
        raise ValueError("Formula contains unknown APs: " + ", ".join(unknown))


def _load_cached_translation(
    cache_path: Path,
    *,
    model: str,
    llm_backend: str,
    symbolic_translator: str,
    symbolic_model_path: Path | None,
    translation_mode: str,
    grounding_mode: str,
    embedding_backend: str,
    embedding_model: str,
    symbols: MapSymbols,
) -> dict[str, object] | None:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("model") != model:
        return None
    cached_mode = payload.get("mode")
    if translation_mode == "llm" and cached_mode not in {"llm", "llm_symbolic_parse_fallback"}:
        return None
    if translation_mode == "heuristic" and cached_mode not in {"heuristic", "auto_heuristic_fallback"}:
        return None
    if cached_mode in {"llm", "llm_symbolic_parse_fallback"}:
        if payload.get("llm_backend", "gemini") != llm_backend:
            return None
        if payload.get("symbolic_translator", "llm") != symbolic_translator:
            return None
        cached_model_path = payload.get("symbolic_model_path")
        expected_model_path = str(symbolic_model_path.expanduser().resolve()) if symbolic_model_path else None
        if symbolic_translator == "t5" and cached_model_path != expected_model_path:
            return None
        if payload.get("grounding_mode") != grounding_mode:
            return None
        backend = payload.get("grounding_backend")
        if isinstance(backend, Mapping) and grounding_mode == "embedding":
            if backend.get("embedding_backend") != embedding_backend:
                return None
            if backend.get("embedding_model") != embedding_model:
                return None
    formula_ast = payload.get("formula_ast")
    if not isinstance(formula_ast, Mapping):
        return None
    formula = Formula.from_json(formula_ast)
    validate_formula_aps(formula, symbols)
    payload["formula"] = formula
    return payload


def _cacheable(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "formula"}


def translate_instruction_to_ltl(
    *,
    instruction: str,
    symbols: MapSymbols,
    model: str | None = None,
    llm_backend: str = "gemini",
    symbolic_translator: str = "llm",
    symbolic_model_path: Path | None = None,
    api_key: str | None = None,
    cache_path: Path | None = None,
    overwrite_cache: bool = False,
    translation_mode: str = "auto",
    grounding_mode: str = "embedding",
    embedding_backend: str = "gemini",
    embedding_model: str | None = None,
    embedding_cache_path: Path | None = None,
    topk_groundings: int = 3,
    verbose: bool = False,
) -> dict[str, object]:
    if translation_mode not in TRANSLATION_MODES:
        raise ValueError(f"Unsupported translation mode: {translation_mode}")
    if llm_backend not in LLM_BACKENDS:
        raise ValueError(f"Unsupported LLM backend: {llm_backend}")
    if symbolic_translator not in SYMBOLIC_TRANSLATORS:
        raise ValueError(f"Unsupported symbolic translator: {symbolic_translator}")
    if grounding_mode not in GROUNDING_MODES:
        raise ValueError(f"Unsupported grounding mode: {grounding_mode}")
    model = model or configured_model()
    embedding_model = embedding_model or configured_embedding_model()
    if symbolic_model_path is not None:
        symbolic_model_path = symbolic_model_path.expanduser().resolve()
    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        try:
            cached = _load_cached_translation(
                cache_path,
                model=model,
                llm_backend=llm_backend,
                symbolic_translator=symbolic_translator,
                symbolic_model_path=symbolic_model_path,
                translation_mode=translation_mode,
                grounding_mode=grounding_mode,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                symbols=symbols,
            )
            if cached is not None:
                _verbose_print(verbose, f"using cache {cache_path}")
                return cached
        except Exception as exc:
            _verbose_print(verbose, f"cache ignored: {exc}")

    if translation_mode == "heuristic":
        payload = heuristic_translate(instruction, symbols, topk=topk_groundings)
        payload["model"] = model
    else:
        try:
            rer, rer_raw = extract_referring_expressions_llm(
                instruction,
                model=model,
                llm_backend=llm_backend,
                api_key=api_key,
            )
            if not rer:
                rer = extract_referring_expressions_heuristic(instruction, symbols)
            if grounding_mode == "embedding":
                grounding, ap_descriptions, grounding_backend_info = ground_referring_expressions_embedding(
                    rer,
                    symbols,
                    topk=topk_groundings,
                    embedding_backend=embedding_backend,
                    embedding_model=embedding_model,
                    embedding_cache_path=embedding_cache_path,
                    api_key=api_key,
                )
            else:
                grounding = ground_referring_expressions_local(
                    rer,
                    symbols,
                    topk=topk_groundings,
                )
                ap_descriptions = {
                    entity.ap: serialize_ap_description(entity)
                    for entity in (*symbols.room_entities, *symbols.object_entities)
                }
                grounding_backend_info = {
                    "grounding_mode": "local",
                    "embedding_backend": None,
                    "embedding_model": None,
                    "embedding_cache_path": None,
                    "candidate_count": len(ap_descriptions),
                }
            selected = {
                expression: str(candidates[0]["ap"])
                for expression, candidates in grounding.items()
                if candidates
            }
            grounded_utterance = replace_phrases(instruction, selected)
            grounded_aps = list(selected.values())
            placeholder_map, inverse_placeholder_map = build_placeholder_maps(grounded_aps)
            symbolic_utterance = replace_phrases(grounded_utterance, placeholder_map)
            grounding_metadata = {
                "selected_groundings": selected,
                "topk_groundings": grounding,
                "ap_semantic_descriptions": ap_descriptions,
                "grounding_backend": grounding_backend_info,
            }
            if symbolic_translator == "t5":
                if symbolic_model_path is None:
                    raise ValueError(
                        "--symbolic-model-path is required when "
                        "--symbolic-translator t5 is selected."
                    )
                symbolic_ltl, symbolic_raw = symbolic_translate_t5(
                    symbolic_utterance=symbolic_utterance,
                    model_path=symbolic_model_path,
                )
            else:
                symbolic_ltl, symbolic_raw = symbolic_translate_llm(
                    symbolic_utterance=symbolic_utterance,
                    placeholder_map=placeholder_map,
                    grounded_metadata=grounding_metadata,
                    model=model,
                    llm_backend=llm_backend,
                    api_key=api_key,
                )
            symbolic_parse_fallback = None
            try:
                symbolic_ltl, symbolic_raw, parse_result, repair_attempts = parse_symbolic_ltl_with_repair(
                    symbolic_ltl=symbolic_ltl,
                    symbolic_utterance=symbolic_utterance,
                    placeholder_map=placeholder_map,
                    model=model,
                    llm_backend=llm_backend,
                    api_key=api_key,
                    verbose=verbose,
                )
                grounded_formula = substitute_formula_aps(
                    parse_result.formula,
                    inverse_placeholder_map,
                )
                parse_warnings = list(parse_result.warnings)
            except SymbolicLTLParseError as parse_exc:
                repair_attempts = parse_exc.attempts
                symbolic_parse_fallback = {
                    "reason": str(parse_exc),
                    "last_bad_symbolic_ltl": parse_exc.last_formula,
                    "strategy": "sequential_eventual_from_selected_groundings",
                }
                _verbose_print(
                    verbose,
                    "symbolic LTL remained malformed; using deterministic "
                    f"sequential-eventual fallback over {grounded_aps}",
                )
                grounded_formula = _sequential_eventual_formula(grounded_aps)
                symbolic_ltl = parse_exc.last_formula
                parse_warnings = [
                    "symbolic_ltl_parse_failed; used sequential-eventual fallback"
                ]
            validate_formula_aps(grounded_formula, symbols)
            payload = {
                "mode": "llm" if symbolic_parse_fallback is None else "llm_symbolic_parse_fallback",
                "model": model,
                "llm_backend": llm_backend,
                "symbolic_translator": symbolic_translator,
                "symbolic_model_path": str(symbolic_model_path) if symbolic_model_path else None,
                "referring_expressions": rer,
                "rer_raw_response": rer_raw,
                "grounding": grounding,
                "grounding_selected": selected,
                "grounding_mode": grounding_mode,
                "grounding_backend": grounding_backend_info,
                "ap_semantic_descriptions": ap_descriptions,
                "grounded_utterance": grounded_utterance,
                "placeholder_map": placeholder_map,
                "symbolic_utterance": symbolic_utterance,
                "symbolic_ltl": symbolic_ltl,
                "symbolic_translation_raw_response": symbolic_raw,
                "symbolic_ltl_repair_attempts": repair_attempts,
                "grounded_ltl": str(grounded_formula),
                "formula": grounded_formula,
                "formula_ast": grounded_formula.to_json(),
                "parse_warnings": parse_warnings,
                "symbolic_parse_fallback": symbolic_parse_fallback,
            }
        except Exception as exc:
            if translation_mode == "llm":
                raise
            _verbose_print(verbose, f"LLM translation failed; fallback heuristic: {exc}")
            payload = heuristic_translate(instruction, symbols, topk=topk_groundings)
            payload["mode"] = "auto_heuristic_fallback"
            payload["model"] = model
            payload["fallback_error"] = str(exc)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(_cacheable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return payload
