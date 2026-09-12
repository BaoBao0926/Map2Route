"""Lang2LTL-2 grounding/LT wrapper for SemPathBench."""

from __future__ import annotations

import importlib
import contextlib
import csv
import io
import json
import os
import pickle
import re
import sys
import threading
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from scripts.methods.lang2ltl.ltl import atomic_propositions
from scripts.methods.lang2ltl.ltl_parser import parse_prefix_ltl, tokenize_prefix_ltl
from scripts.methods.lang2ltl.map_features import MapSymbols
from scripts.methods.lang2ltl.translator import (
    configured_api_key,
    configured_embedding_model,
    configured_model,
    gemini_embed_text,
    gemini_response_text,
    heuristic_translate,
)


LANG2LTL2_ROOT = Path(__file__).resolve().parent / "Lang2LTL-2"


def _prepare_text_only_transformers() -> None:
    """Keep the original T5 LT module independent of torchvision.

    Lang2LTL-2's LT stage is text-only, but recent Transformers releases may
    import torchvision while resolving T5 classes. Some installed
    torch/torchvision combinations fail during that unrelated import before
    the original ``lt.Seq2Seq`` code can load its checkpoint. Match the
    compatibility handling used by the Lang2LTL adapter without changing the
    cloned upstream implementation.
    """
    os.environ.setdefault("USE_TORCHVISION", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    try:
        import transformers.utils.import_utils as transformers_import_utils
    except ImportError:
        return
    transformers_import_utils._torchvision_available = False
    transformers_import_utils._torchvision_version = None


_LT_MODEL_CACHE: dict[Path, object] = {}
_LT_MODEL_LOCK = threading.Lock()
_LT_INFERENCE_LOCK = threading.Lock()


def _get_original_lt_model(lt_module: object, model_path: Path) -> object:
    """Load one original Lang2LTL-2 T5 runtime per checkpoint path."""
    resolved_path = model_path.expanduser().resolve()
    with _LT_MODEL_LOCK:
        model = _LT_MODEL_CACHE.get(resolved_path)
        if model is None:
            constructor = getattr(lt_module, "Seq2Seq")
            model = constructor(str(resolved_path), "t5-base")
            raw_model = getattr(model, "model", None)
            if raw_model is not None and hasattr(raw_model, "eval"):
                raw_model.eval()
            _LT_MODEL_CACHE[resolved_path] = model
    return model


SRER_SYSTEM_PROMPT = """You are the spatial referring expression recognition module of Lang2LTL-2.

Given a navigation command, extract:
1. Referring Expressions: spatial referring expressions or landmark/object target phrases.
2. Spatial Predicates: a list of dictionaries mapping spatial relation text to the target/anchor referring expressions involved in that relation.
3. Lifted Command: the original command with each spatial referring expression replaced by symbols a, b, c, d, h, j, k in order.

Return exactly three lines and no markdown:
Referring Expressions: [...]
Spatial Predicates: [...]
Lifted Command: "..."

Use Python literal syntax for the lists/dictionaries/strings, because the original parser reads the output with eval().
If no spatial predicates exist, return Spatial Predicates: [].
"""

RAG_TRANSLATION_SYSTEM_PROMPT = """You are an expert at translating lifted navigation commands to Spot-style prefix Linear Temporal Logic (LTL).

Return exactly one line:
LTL formula: "<formula>"

Use symbols already present in the command, such as a, b, c, d, h, j, k.
Do not output markdown or explanation.
"""


RETRYABLE_GEMINI_MARKERS = (
    " 429 ",
    " 500 ",
    " 502 ",
    " 503 ",
    " 504 ",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
)


@dataclass
class Lang2LTL2Result:
    srer_output: dict[str, object] | None
    reg_output: dict[str, object] | None
    spg_output: dict[str, object] | None
    lifted_utterance: str | None
    lifted_symbol_map: dict[str, str]
    lifted_ltl: str | None
    grounded_ltl: str | None
    symbol_to_ap: dict[str, str]
    target_sequence: list[str]
    unsupported: list[dict[str, object]]
    failures: list[dict[str, object]]
    cache_path: str | None = None
    llm_backend: str = "gemini"
    model: str | None = None
    embedding_model: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_cache(cache_path: Path, result: Lang2LTL2Result) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(_json_safe(result.to_metadata()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _str_dict(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _normalize_srer_predicate_references(srer_out: dict[str, object]) -> None:
    """Ensure each SRER relation has a sequence of referring expressions.

    The original REG module enumerates the relation value. Gemini can emit a
    single string (for example ``{"in": "the starting room"}``), which would
    otherwise be enumerated character-by-character. SPG then attempts a
    Cartesian product over those character-level candidate lists.
    """
    sre_to_preds = srer_out.get("sre_to_preds")
    if not isinstance(sre_to_preds, dict):
        return
    for spatial_predicate in sre_to_preds.values():
        if not isinstance(spatial_predicate, dict):
            continue
        for relation, references in list(spatial_predicate.items()):
            if isinstance(references, str):
                spatial_predicate[relation] = [references]


def _load_cache(cache_path: Path) -> Lang2LTL2Result | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    lifted_utterance = payload.get("lifted_utterance") if isinstance(payload.get("lifted_utterance"), str) else None
    lifted_symbol_map = _str_dict(payload.get("lifted_symbol_map"))
    symbol_to_ap = _str_dict(payload.get("symbol_to_ap"))
    target_sequence = [
        str(value)
        for value in payload.get("target_sequence", [])
        if isinstance(value, str)
    ] if isinstance(payload.get("target_sequence"), list) else []
    if not target_sequence:
        target_sequence = derive_target_sequence(
            lifted_utterance=lifted_utterance,
            lifted_symbol_map=lifted_symbol_map,
            symbol_to_ap=symbol_to_ap,
        )
    return Lang2LTL2Result(
        srer_output=payload.get("srer_output") if isinstance(payload.get("srer_output"), dict) else None,
        reg_output=payload.get("reg_output") if isinstance(payload.get("reg_output"), dict) else None,
        spg_output=payload.get("spg_output") if isinstance(payload.get("spg_output"), dict) else None,
        lifted_utterance=lifted_utterance,
        lifted_symbol_map=lifted_symbol_map,
        lifted_ltl=payload.get("lifted_ltl") if isinstance(payload.get("lifted_ltl"), str) else None,
        grounded_ltl=payload.get("grounded_ltl") if isinstance(payload.get("grounded_ltl"), str) else None,
        symbol_to_ap=symbol_to_ap,
        target_sequence=target_sequence,
        unsupported=payload.get("unsupported") if isinstance(payload.get("unsupported"), list) else [],
        failures=payload.get("failures") if isinstance(payload.get("failures"), list) else [],
        cache_path=str(cache_path),
        llm_backend=payload.get("llm_backend") if isinstance(payload.get("llm_backend"), str) else "gemini",
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        embedding_model=payload.get("embedding_model") if isinstance(payload.get("embedding_model"), str) else None,
    )


def _is_retryable_gemini_error(exc: Exception) -> bool:
    message = f" {exc} "
    return any(marker in message for marker in RETRYABLE_GEMINI_MARKERS)


def _with_gemini_retries(operation: str, callback: object, *, attempts: int = 3) -> object:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return callback()
        except Exception as exc:
            last_exc = exc
            if attempt == attempts - 1 or not _is_retryable_gemini_error(exc):
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{operation} failed") from last_exc


def _gemini_response_text_with_retry(
    *,
    model: str,
    api_key: str | None,
    system_prompt: str,
    user_prompt: str,
) -> str:
    return str(
        _with_gemini_retries(
            "Gemini text generation",
            lambda: gemini_response_text(
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ),
        )
    )


def _gemini_embed_text_with_retry(
    text: str,
    *,
    model: str,
    api_key: str | None,
) -> list[float]:
    result = _with_gemini_retries(
        "Gemini embedding",
        lambda: gemini_embed_text(text, model=model, api_key=api_key),
    )
    return [float(value) for value in result]  # type: ignore[union-attr]


def _install_gemini_openai_models_module(
    *,
    model: str,
    embedding_model: str,
    api_key: str | None,
) -> None:
    """Install a Gemini-backed module compatible with Lang2LTL-2 openai_models."""

    def extract(command: str) -> str:
        return _gemini_response_text_with_retry(
            model=model,
            api_key=api_key,
            system_prompt=SRER_SYSTEM_PROMPT,
            user_prompt=(
                "Extract the referring expressions to predicates map, lifted "
                "command, and symbol map for the following command:\n\n"
                f"Command: {command}"
            ),
        )

    def get_embed(txt: object) -> list[float]:
        text = json.dumps(txt, ensure_ascii=False).replace("\n", " ")
        return _gemini_embed_text_with_retry(text, model=embedding_model, api_key=api_key)

    def translate(query: str, examples: object) -> tuple[str, int]:
        response = _gemini_response_text_with_retry(
            model=model,
            api_key=api_key,
            system_prompt=(
                RAG_TRANSLATION_SYSTEM_PROMPT
                + "\n\nHere are examples:\n"
                + str(examples)
            ),
            user_prompt=f'Translate the following command to an LTL formula:\n\nCommand: "{query}"',
        )
        cleaned = response.strip().replace('"', "")
        if ":" in cleaned:
            cleaned = cleaned.split(":", 1)[1].strip()
        return cleaned, 0

    class GeminiVisionStub:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def caption(self, img_fpath: str) -> str:
            # The SemPathBench adapter is text-only by default, so this path should
            # not be used. Return a minimal deterministic caption if called.
            return f"an image landmark from {Path(img_fpath).stem}"

    module = types.ModuleType("openai_models")
    module.extract = extract  # type: ignore[attr-defined]
    module.get_embed = get_embed  # type: ignore[attr-defined]
    module.translate = translate  # type: ignore[attr-defined]
    module.GPT4V = GeminiVisionStub  # type: ignore[attr-defined]
    sys.modules["openai_models"] = module


def _install_original_dependency_shims() -> None:
    """Install lightweight shims for original utility modules used by Lang2LTL-2."""

    def load_from_file(fpath: str, noheader: bool = True, use_pandas: bool = False) -> object:
        path = Path(fpath)
        suffix = path.suffix.lower()
        if suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8")
            if "prompt" in str(path):
                return text
            return [line.strip() for line in text.splitlines() if line.strip()]
        if suffix == ".pkl":
            with path.open("rb") as handle:
                return pickle.load(handle)
        if suffix == ".csv":
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                if noheader:
                    next(reader, None)
                return [row for row in reader]
        raise ValueError(f"Unsupported file type for shim load_from_file: {suffix}")

    def save_to_file(data: object, fpath: str, mode: str | None = None) -> None:
        path = Path(fpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".json":
            path.write_text(json.dumps(_json_safe(data), indent=4, ensure_ascii=False), encoding="utf-8")
            return
        if suffix == ".txt":
            path.write_text(str(data), encoding="utf-8")
            return
        if suffix == ".pkl":
            with path.open(mode or "wb") as handle:
                pickle.dump(data, handle)
            return
        if suffix == ".csv":
            with path.open(mode or "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerows(data)  # type: ignore[arg-type]
            return
        raise ValueError(f"Unsupported file type for shim save_to_file: {suffix}")

    utils_module = types.ModuleType("utils")
    utils_module.load_from_file = load_from_file  # type: ignore[attr-defined]
    utils_module.save_to_file = save_to_file  # type: ignore[attr-defined]
    sys.modules["utils"] = utils_module

    def load_map(_path: str) -> object:
        raise FileNotFoundError("Spot graph files are not used by the SemPathBench text-only bridge.")

    def extract_waypoints(_graph: object) -> dict[str, object]:
        return {}

    load_map_module = types.ModuleType("load_map")
    load_map_module.load_map = load_map  # type: ignore[attr-defined]
    load_map_module.extract_waypoints = extract_waypoints  # type: ignore[attr-defined]
    sys.modules["load_map"] = load_map_module

    utm_module = types.ModuleType("utm")
    utm_module.from_latlon = lambda _lat, _long: (0.0, 0.0, 31, "N")  # type: ignore[attr-defined]
    sys.modules.setdefault("utm", utm_module)

    class _Transformer:
        @staticmethod
        def from_crs(*_args: object, **_kwargs: object) -> "_Transformer":
            return _Transformer()

        def transform(
            self,
            longitude: float,
            latitude: float,
            altitude: float = 0.0,
            *,
            radians: bool = False,
        ) -> tuple[float, float, float]:
            return float(longitude), float(latitude), float(altitude)

    pyproj_module = types.ModuleType("pyproj")
    pyproj_module.Transformer = _Transformer  # type: ignore[attr-defined]
    sys.modules.setdefault("pyproj", pyproj_module)


def _import_original_modules(
    *,
    model: str,
    embedding_model: str,
    api_key: str | None,
) -> tuple[object, object, object]:
    _install_gemini_openai_models_module(
        model=model,
        embedding_model=embedding_model,
        api_key=api_key,
    )
    _install_original_dependency_shims()
    if str(LANG2LTL2_ROOT) not in sys.path:
        sys.path.insert(0, str(LANG2LTL2_ROOT))
    return (
        importlib.import_module("srer"),
        importlib.import_module("reg"),
        importlib.import_module("spg"),
    )


def _lookup_grounded_sre(grounded_sps: Mapping[str, object], sre: str) -> object | None:
    if sre in grounded_sps:
        return grounded_sps[sre]
    lowered = sre.lower()
    for key, value in grounded_sps.items():
        if str(key).lower() == lowered:
            return value
    return None


def _extract_symbol_to_ap(
    *,
    lifted_symbol_map: Mapping[str, object],
    grounded_sps: Mapping[str, object],
    valid_aps: set[str],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    symbol_to_ap: dict[str, str] = {}
    failures: list[dict[str, object]] = []
    for symbol, raw_sre in lifted_symbol_map.items():
        sre = str(raw_sre)
        candidates = _lookup_grounded_sre(grounded_sps, sre)
        if not isinstance(candidates, list) or not candidates:
            failures.append({"status": "no_grounding", "symbol": symbol, "sre": sre})
            continue
        top = candidates[0]
        if not isinstance(top, Mapping):
            failures.append({"status": "malformed_spg_grounding", "symbol": symbol, "sre": sre})
            continue
        target = top.get("target")
        if not isinstance(target, str) or not target:
            failures.append({"status": "missing_spg_target", "symbol": symbol, "sre": sre})
            continue
        if target not in valid_aps:
            failures.append(
                {
                    "status": "grounding_not_sembench_ap",
                    "symbol": symbol,
                    "sre": sre,
                    "target": target,
                }
            )
            continue
        symbol_to_ap[str(symbol)] = target
    return symbol_to_ap, failures


def ground_lifted_ltl(lifted_ltl: str, symbol_to_ap: Mapping[str, str]) -> str:
    tokens = tokenize_prefix_ltl(lifted_ltl)
    grounded = [symbol_to_ap.get(token, token) for token in tokens]
    return " ".join(grounded)


def _is_start_symbol_context(lifted_utterance: str, symbol_start: int) -> bool:
    prefix = lifted_utterance[max(0, symbol_start - 40):symbol_start].lower()
    return bool(
        re.search(
            r"(?:^|[\s.])(?:start|starting|begin|beginning|depart)\s+(?:from|at|near|beside|by)?\s*$",
            prefix,
        )
    )


def derive_target_sequence(
    *,
    lifted_utterance: str | None,
    lifted_symbol_map: Mapping[str, str],
    symbol_to_ap: Mapping[str, str],
) -> list[str]:
    """Derive a conservative debug target order from grounded lifted symbols."""
    if not symbol_to_ap:
        return []
    ordered_symbols: list[str] = []
    if lifted_utterance:
        symbol_pattern = "|".join(
            re.escape(symbol)
            for symbol in sorted(symbol_to_ap, key=len, reverse=True)
        )
        if symbol_pattern:
            for match in re.finditer(rf"\b({symbol_pattern})\b", lifted_utterance):
                symbol = match.group(1)
                if _is_start_symbol_context(lifted_utterance, match.start()):
                    continue
                ordered_symbols.append(symbol)
    if not ordered_symbols:
        ordered_symbols = [
            symbol
            for symbol in lifted_symbol_map
            if symbol in symbol_to_ap
        ]

    sequence: list[str] = []
    seen: set[str] = set()
    for symbol in ordered_symbols:
        ap_name = symbol_to_ap.get(symbol)
        if ap_name and ap_name not in seen:
            sequence.append(ap_name)
            seen.add(ap_name)
    return sequence


def _formula_to_prefix(formula: object) -> str:
    """Serialize the local Formula object into the prefix syntax used here."""
    op = getattr(formula, "op", None)
    args = tuple(getattr(formula, "args", ()) or ())
    value = getattr(formula, "value", None)
    if op == "true":
        return "true"
    if op == "false":
        return "false"
    if op == "ap":
        return str(value)
    unary_ops = {
        "not": "!",
        "eventually": "F",
        "always": "G",
        "next": "X",
    }
    if op in unary_ops and len(args) == 1:
        return f"{unary_ops[op]} {_formula_to_prefix(args[0])}"
    binary_ops = {
        "and": "&",
        "or": "|",
        "until": "U",
        "imply": "i",
    }
    if op in binary_ops and len(args) >= 2:
        operator = binary_ops[op]
        serialized = _formula_to_prefix(args[-1])
        for child in reversed(args[:-1]):
            serialized = f"{operator} {_formula_to_prefix(child)} {serialized}"
        return serialized
    return str(formula)


def _heuristic_lang2ltl_fallback(
    *,
    instruction_text: str,
    symbols: MapSymbols | None,
    valid_aps: set[str],
    topk_groundings: int,
    reason: str,
) -> tuple[str | None, list[str], dict[str, object] | None, list[dict[str, object]]]:
    if symbols is None:
        return None, [], None, [
            {
                "status": "local_lang2ltl_fallback_unavailable",
                "reason": "map_symbols_not_provided",
                "trigger": reason,
            }
        ]
    try:
        payload = heuristic_translate(
            instruction_text,
            symbols,
            topk=topk_groundings,
        )
        selected = [
            str(ap_name)
            for ap_name in payload.get("grounding_selected", [])
            if isinstance(ap_name, str) and ap_name in valid_aps
        ]
        formula = payload.get("formula")
        grounded_ltl = _formula_to_prefix(formula) if formula is not None else None
        validation_failures: list[dict[str, object]] = []
        if grounded_ltl is not None:
            validation_failures = _validate_grounded_ltl(grounded_ltl, valid_aps)
            if validation_failures:
                grounded_ltl = None
        fallback_payload = {
            key: _json_safe(value)
            for key, value in payload.items()
            if key != "formula"
        }
        fallback_payload.update(
            {
                "mode": "local_lang2ltl_heuristic_fallback",
                "trigger": reason,
                "grounded_ltl_prefix": grounded_ltl,
                "target_sequence": selected,
                "validation_failures": validation_failures,
            }
        )
        return grounded_ltl, selected, fallback_payload, [
            {
                "status": "used_local_lang2ltl_heuristic_fallback",
                "trigger": reason,
                "target_sequence": selected,
                "grounded_ltl": grounded_ltl,
                "validation_failures": validation_failures,
            }
        ]
    except Exception as exc:
        return None, [], None, [
            {
                "status": "local_lang2ltl_fallback_failure",
                "reason": str(exc),
                "trigger": reason,
            }
        ]


def _validate_grounded_ltl(grounded_ltl: str, valid_aps: set[str]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    try:
        parse_result = parse_prefix_ltl(grounded_ltl)
    except Exception as exc:
        return [{"status": "grounded_ltl_parse_failure", "reason": str(exc)}]
    unknown = sorted(atomic_propositions(parse_result.formula) - valid_aps)
    if unknown:
        failures.append({"status": "grounded_ltl_unknown_aps", "unknown_aps": unknown})
    return failures


def run_original_lang2ltl2_pipeline(
    *,
    instruction_text: str,
    graph_dpath: Path,
    osm_fpath: Path,
    valid_aps: set[str],
    cache_path: Path | None,
    overwrite_cache: bool = False,
    topk_groundings: int = 10,
    text_only: bool = True,
    model: str | None = None,
    embedding_model: str | None = None,
    api_key: str | None = None,
    lt_model_path: Path | None = None,
    rel_embeds_path: Path | None = None,
    reg_query_cache_path: Path | None = None,
    symbols: MapSymbols | None = None,
    verbose: bool = False,
) -> Lang2LTL2Result:
    model = model or configured_model()
    embedding_model = embedding_model or configured_embedding_model()
    api_key = api_key or configured_api_key()
    if cache_path is not None and not overwrite_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    failures: list[dict[str, object]] = []
    unsupported: list[dict[str, object]] = []
    srer_output: dict[str, object] | None = None
    reg_output: dict[str, object] | None = None
    spg_output: dict[str, object] | None = None
    lifted_utterance: str | None = None
    lifted_symbol_map: dict[str, str] = {}
    lifted_ltl: str | None = None
    grounded_ltl: str | None = None
    symbol_to_ap: dict[str, str] = {}
    target_sequence: list[str] = []

    try:
        srer_mod, reg_mod, spg_mod = _import_original_modules(
            model=model,
            embedding_model=embedding_model,
            api_key=api_key,
        )
    except Exception as exc:
        failures.append({"status": "original_module_import_failure", "reason": str(exc)})
        grounded_ltl, target_sequence, fallback_payload, fallback_failures = _heuristic_lang2ltl_fallback(
            instruction_text=instruction_text,
            symbols=symbols,
            valid_aps=valid_aps,
            topk_groundings=topk_groundings,
            reason="original_module_import_failure",
        )
        unsupported.extend(fallback_failures)
        result = Lang2LTL2Result(
            srer_output=None,
            reg_output=fallback_payload,
            spg_output=None,
            lifted_utterance=None,
            lifted_symbol_map={},
            lifted_ltl=None,
            grounded_ltl=grounded_ltl,
            symbol_to_ap={},
            target_sequence=target_sequence,
            unsupported=unsupported,
            failures=failures,
            cache_path=str(cache_path) if cache_path else None,
            llm_backend="gemini",
            model=model,
            embedding_model=embedding_model,
        )
        if cache_path is not None:
            _write_cache(cache_path, result)
        return result

    try:
        _raw, srer_out = srer_mod.srer(instruction_text)
        _normalize_srer_predicate_references(srer_out)
        srer_output = _json_safe(srer_out)
    except Exception as exc:
        failures.append({"status": "srer_failure", "reason": str(exc)})
        srer_out = {"utt": instruction_text}

    if "sre_to_preds" in srer_out:
        try:
            ablate = "image" if text_only else None
            reg_cache = reg_query_cache_path or (graph_dpath.parent / "reg_query_cache.pkl")
            reg_mod.reg(
                str(graph_dpath),
                str(osm_fpath),
                [srer_out],
                topk_groundings,
                ablate,
                str(reg_cache),
            )
            reg_output = _json_safe(srer_out)
        except Exception as exc:
            failures.append({"status": "reg_failure", "reason": str(exc)})
    else:
        failures.append({"status": "srer_missing_sre_to_preds"})

    if "grounded_sre_to_preds" in srer_out:
        try:
            rel_embeds = rel_embeds_path or (graph_dpath.parent / "known_rel_embeds.json")
            with contextlib.redirect_stdout(io.StringIO()):
                landmarks = spg_mod.load_lmks(str(graph_dpath), str(osm_fpath))
            srer_out["grounded_sps"] = spg_mod.spg(
                landmarks,
                srer_out,
                topk_groundings,
                str(rel_embeds),
            )
            spg_output = _json_safe(srer_out)
        except Exception as exc:
            failures.append({"status": "spg_failure", "reason": str(exc)})
    else:
        failures.append({"status": "reg_missing_grounded_sre_to_preds"})

    lifted_utterance = srer_out.get("lifted_utt") if isinstance(srer_out.get("lifted_utt"), str) else None
    if isinstance(srer_out.get("lifted_symbol_map"), Mapping):
        lifted_symbol_map = {
            str(key): str(value)
            for key, value in srer_out["lifted_symbol_map"].items()
        }

    if lt_model_path is None:
        lt_model_path = Path.home() / "ground" / "models" / "checkpoint-best"
    try:
        _prepare_text_only_transformers()
        import torch

        lt_mod = importlib.import_module("lt")
        if not lt_model_path.exists():
            raise FileNotFoundError(f"LT model path does not exist: {lt_model_path}")
        lt_model = _get_original_lt_model(lt_mod, lt_model_path)
        # The original constrained decoder performs up to 256 T5 forwards.
        # Its upstream implementation does not enter inference mode, so each
        # evaluation would otherwise build autograd state unnecessarily.
        with _LT_INFERENCE_LOCK, torch.inference_mode():
            lt_mod.lt(srer_out, lt_model)
        lifted_ltl = srer_out.get("lifted_ltl") if isinstance(srer_out.get("lifted_ltl"), str) else None
    except Exception as exc:
        failures.append({"status": "lt_failure", "reason": str(exc)})

    grounded_sps = srer_out.get("grounded_sps")
    if isinstance(grounded_sps, Mapping) and lifted_symbol_map:
        symbol_to_ap, grounding_failures = _extract_symbol_to_ap(
            lifted_symbol_map=lifted_symbol_map,
            grounded_sps=grounded_sps,
            valid_aps=valid_aps,
        )
        failures.extend(grounding_failures)
    elif lifted_symbol_map:
        failures.append({"status": "spg_missing_grounded_sps"})

    target_sequence = derive_target_sequence(
        lifted_utterance=lifted_utterance,
        lifted_symbol_map=lifted_symbol_map,
        symbol_to_ap=symbol_to_ap,
    )

    if lifted_ltl and symbol_to_ap:
        grounded_ltl = ground_lifted_ltl(lifted_ltl, symbol_to_ap)
        failures.extend(_validate_grounded_ltl(grounded_ltl, valid_aps))
    elif lifted_ltl and not symbol_to_ap:
        failures.append({"status": "no_symbol_groundings_for_lifted_ltl"})

    if not grounded_ltl and not target_sequence:
        grounded_ltl, target_sequence, fallback_payload, fallback_failures = _heuristic_lang2ltl_fallback(
            instruction_text=instruction_text,
            symbols=symbols,
            valid_aps=valid_aps,
            topk_groundings=topk_groundings,
            reason="empty_original_lang2ltl2_output",
        )
        unsupported.extend(fallback_failures)
        if fallback_payload is not None:
            reg_output = fallback_payload if reg_output is None else {
                "original": reg_output,
                "fallback": fallback_payload,
            }

    if failures and verbose:
        print(f"[lang2ltlv2:translator] failures={failures}", flush=True)

    result = Lang2LTL2Result(
        srer_output=srer_output,
        reg_output=reg_output,
        spg_output=spg_output,
        lifted_utterance=lifted_utterance,
        lifted_symbol_map=lifted_symbol_map,
        lifted_ltl=lifted_ltl,
        grounded_ltl=grounded_ltl,
        symbol_to_ap=symbol_to_ap,
        target_sequence=target_sequence,
        unsupported=unsupported,
        failures=failures,
        cache_path=str(cache_path) if cache_path else None,
        llm_backend="gemini",
        model=model,
        embedding_model=embedding_model,
    )
    if cache_path is not None:
        _write_cache(cache_path, result)
    return result
