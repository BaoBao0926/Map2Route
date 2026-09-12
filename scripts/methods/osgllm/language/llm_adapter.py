"""Optional LLM grounding and LTL translation for the OSG-LLM adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from pathlib import Path

from scripts.methods.lang2ltl.ltl import Formula, atomic_propositions
from scripts.methods.lang2ltl.ltl_parser import parse_prefix_ltl
from scripts.methods.lang2ltl.translator import configured_api_key, configured_model, gemini_response_text
from scripts.methods.osgllm.language.translator import (
    TranslationResult,
    _sequence_formula,
    _unsupported_constraints,
    _with_avoids,
)
from scripts.methods.osgllm.scene_graph.graph_types import AttributeRegion, Cell, SceneGraph


GROUNDING_SYSTEM_PROMPT = """You ground route instructions to a SemPathBench scene graph.

Return JSON only. Do not invent ids. Use only room_*, object_* ids present in the scene graph.
Soft preferences should be listed as unsupported_phrases, not hard goals.

Schema:
{
  "ordered_goal_entities": ["room_1", "object_2"],
  "avoid_entities": ["room_3"],
  "entity_bindings": [
    {"text_span": "exact phrase", "entity_id": "object_2", "reason": "short reason"}
  ],
  "unsupported_phrases": ["phrase"]
}
"""

LTL_SYSTEM_PROMPT = """You translate grounded navigation goals into prefix LTL.

Return JSON only:
{
  "ltl_prefix": "F & enter(room_1) F reach(object_2)",
  "notes": "short note"
}

Allowed operators: F, G, !, &, |, U, ->.
Use only allowed AP names exactly as provided. Ignore soft preferences.
For ordered goals a then b, use: F & a F b
For avoid c, combine with: & <goals> G ! c
"""


def _cache_key(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _read_cache(root: Path | None, namespace: str, key: str) -> dict[str, object] | None:
    if root is None:
        return None
    path = root / namespace / f"{key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_cache(root: Path | None, namespace: str, key: str, payload: Mapping[str, object]) -> None:
    if root is None:
        return
    path = root / namespace / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_from_text(text: str) -> dict[str, object]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    return payload


def _compact_scene_graph(graph: SceneGraph, *, max_objects_per_room: int = 80) -> dict[str, object]:
    rooms: list[dict[str, object]] = []
    objects_by_room: dict[str, list[AttributeRegion]] = {}
    for region in graph.by_kind("object"):
        objects_by_room.setdefault(region.parent_id or "floor_0", []).append(region)
    for room in graph.by_kind("room"):
        objects = [
            {
                "id": obj.node_id,
                "category": obj.category,
                "name": obj.name,
                "ap": obj.ap,
            }
            for obj in objects_by_room.get(room.node_id, ())[:max_objects_per_room]
        ]
        rooms.append(
            {
                "id": room.node_id,
                "category": room.category,
                "name": room.name,
                "ap": room.ap,
                "connected_rooms": list(graph.room_adjacency.get(room.node_id, ())),
                "objects": objects,
            }
        )
    return {
        "floor": {"id": "floor_0", "ap": "enter(floor_0)"},
        "rooms": rooms,
        "object_count": len(graph.by_kind("object")),
    }


def _entity_to_ap(graph: SceneGraph, entity_id: str) -> str | None:
    region = graph.regions.get(entity_id)
    return region.ap if region is not None else None


def _binding_record(graph: SceneGraph, item: Mapping[str, object], ap_name: str) -> dict[str, object]:
    entity_id = str(item.get("entity_id", ""))
    region = graph.regions.get(entity_id)
    return {
        "text_span": str(item.get("text_span", "")),
        "entity_id": entity_id,
        "ap": ap_name,
        "kind": region.kind if region else "",
        "category": region.category if region else "",
        "parent_id": region.parent_id if region else None,
        "selection_rule": "llm_validated",
        "reason": str(item.get("reason", "")),
    }


def _validate_grounding_payload(
    graph: SceneGraph,
    payload: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[dict[str, object], ...], dict[str, object]]:
    goals: list[str] = []
    avoids: list[str] = []
    bindings: list[dict[str, object]] = []
    errors: list[str] = []
    raw_bindings = payload.get("entity_bindings", [])
    binding_by_id: dict[str, Mapping[str, object]] = {}
    if isinstance(raw_bindings, Sequence) and not isinstance(raw_bindings, (str, bytes)):
        for item in raw_bindings:
            if isinstance(item, Mapping) and isinstance(item.get("entity_id"), str):
                binding_by_id[str(item["entity_id"])] = item
    for field, target in (("ordered_goal_entities", goals), ("avoid_entities", avoids)):
        raw_entities = payload.get(field, [])
        if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
            errors.append(f"{field} must be a list")
            continue
        for raw_entity in raw_entities:
            entity_id = str(raw_entity)
            ap_name = _entity_to_ap(graph, entity_id)
            if ap_name is None:
                errors.append(f"unknown entity id: {entity_id}")
                continue
            target.append(ap_name)
            bindings.append(_binding_record(graph, binding_by_id.get(entity_id, {"entity_id": entity_id}), ap_name))
    validation = {
        "valid": not errors and bool(goals),
        "errors": errors,
        "source": "llm_grounding",
    }
    return tuple(dict.fromkeys(goals)), tuple(dict.fromkeys(avoids)), tuple(bindings), validation


def _formula_from_ast(payload: Mapping[str, object]) -> Formula:
    op = str(payload.get("op", "")).lower()
    if op == "ap":
        value = payload.get("value")
        if not isinstance(value, str) or not value:
            raise ValueError("AST ap node requires string value")
        return Formula("ap", value=value)
    raw_args = payload.get("args", [])
    if not isinstance(raw_args, Sequence) or isinstance(raw_args, (str, bytes)):
        raise ValueError("AST node args must be a list")
    args = tuple(_formula_from_ast(item) for item in raw_args if isinstance(item, Mapping))
    if op in {"true", "false"}:
        return Formula(op)
    if op in {"not", "eventually", "always", "next"} and len(args) == 1:
        return Formula({"not": "not", "eventually": "eventually", "always": "always", "next": "next"}[op], args).simplify()
    if op in {"and", "or", "until", "imply"} and len(args) == 2:
        return Formula(op, args).simplify()
    raise ValueError(f"Unsupported AST node: {op}")


def _validate_formula(formula: Formula, inventory: set[str]) -> dict[str, object]:
    used = atomic_propositions(formula)
    missing = sorted(used - inventory)
    return {
        "valid": not missing and formula.op != "true",
        "used_atomic_propositions": sorted(used),
        "missing_atomic_propositions": missing,
        "backend": "local_formula_progression",
        "source": "llm_ltl",
    }


def _restore_function_ap_names(formula: Formula, graph: SceneGraph) -> Formula:
    formula = formula.simplify()
    if formula.op == "ap":
        value = str(formula.value)
        if value in graph.regions:
            return Formula("ap", value=graph.regions[value].ap)
        return formula
    if formula.op in {"true", "false"}:
        return formula
    return Formula(
        formula.op,
        tuple(_restore_function_ap_names(child, graph) for child in formula.args),
        formula.value,
    ).simplify()


def _replace_formula_aps(formula: Formula, ap_map: Mapping[str, str]) -> Formula:
    formula = formula.simplify()
    if formula.op == "ap":
        return Formula("ap", value=ap_map.get(str(formula.value), str(formula.value)))
    if formula.op in {"true", "false"}:
        return formula
    return Formula(
        formula.op,
        tuple(_replace_formula_aps(child, ap_map) for child in formula.args),
        formula.value,
    ).simplify()


def _parse_llm_prefix(prefix: str, graph: SceneGraph) -> Formula:
    alias_to_ap: dict[str, str] = {}
    safe_prefix = prefix
    for index, ap_name in enumerate(sorted(graph.ap_inventory(), key=len, reverse=True)):
        alias = f"p{index}"
        if ap_name in safe_prefix:
            safe_prefix = safe_prefix.replace(ap_name, alias)
            alias_to_ap[alias] = ap_name
    formula = parse_prefix_ltl(safe_prefix).formula
    return _replace_formula_aps(_restore_function_ap_names(formula, graph), alias_to_ap)


def translate_instruction_llm(
    graph: SceneGraph,
    instruction: Mapping[str, object],
    *,
    start: Cell,
    model: str | None,
    cache_root: Path | None,
    overwrite_cache: bool,
    cache_only: bool = False,
) -> TranslationResult:
    del start
    instruction_text = str(instruction.get("instruction", ""))
    scene_graph = _compact_scene_graph(graph)
    cache_key = _cache_key("ground_ltl_v1", instruction_text, scene_graph)
    if not overwrite_cache:
        cached = _read_cache(cache_root, "ground_ltl", cache_key)
        if cached is not None:
            payload = cached
            cache_status = "hit"
        else:
            payload = None
            cache_status = "miss"
    else:
        payload = None
        cache_status = "overwrite"

    if payload is None:
        if cache_only:
            raise RuntimeError(
                "Required cached LLM grounding/LTL translation was not found "
                f"for cache key {cache_key}."
            )
        if not configured_api_key():
            raise RuntimeError("Gemini API key is not configured.")
        chosen_model = model or configured_model()
        grounding_prompt = json.dumps(
            {"instruction": instruction_text, "scene_graph": scene_graph},
            ensure_ascii=False,
        )
        raw_grounding = gemini_response_text(
            model=chosen_model,
            system_prompt=GROUNDING_SYSTEM_PROMPT,
            user_prompt=grounding_prompt,
        )
        grounding_payload = _json_from_text(raw_grounding)
        goals, avoids, bindings, grounding_validation = _validate_grounding_payload(graph, grounding_payload)
        if not grounding_validation.get("valid"):
            raise ValueError(f"LLM grounding validation failed: {grounding_validation}")
        allowed_aps = sorted(set(graph.ap_inventory()) & set(goals + avoids))
        ltl_prompt = json.dumps(
            {
                "instruction": instruction_text,
                "ordered_goal_aps": list(goals),
                "avoid_aps": list(avoids),
                "allowed_atomic_propositions": allowed_aps,
            },
            ensure_ascii=False,
        )
        raw_ltl = gemini_response_text(
            model=chosen_model,
            system_prompt=LTL_SYSTEM_PROMPT,
            user_prompt=ltl_prompt,
        )
        ltl_payload = _json_from_text(raw_ltl)
        payload = {
            "model": chosen_model,
            "raw_grounding_response": raw_grounding,
            "grounding_payload": grounding_payload,
            "grounding_validation": grounding_validation,
            "raw_ltl_response": raw_ltl,
            "ltl_payload": ltl_payload,
            "goals": list(goals),
            "avoids": list(avoids),
            "bindings": list(bindings),
        }
        _write_cache(cache_root, "ground_ltl", cache_key, payload)

    goals = tuple(str(item) for item in payload.get("goals", []) if isinstance(item, str))
    avoids = tuple(str(item) for item in payload.get("avoids", []) if isinstance(item, str))
    ltl_payload = payload.get("ltl_payload", {})
    formula: Formula
    if isinstance(ltl_payload, Mapping) and isinstance(ltl_payload.get("ltl_ast"), Mapping):
        formula = _formula_from_ast(ltl_payload["ltl_ast"])  # type: ignore[arg-type]
    elif isinstance(ltl_payload, Mapping) and isinstance(ltl_payload.get("ltl_prefix"), str):
        formula = _parse_llm_prefix(str(ltl_payload["ltl_prefix"]), graph)
    else:
        formula = _with_avoids(_sequence_formula(goals), avoids)
    validation = _validate_formula(formula, set(graph.ap_inventory()))
    validation["grounding"] = payload.get("grounding_validation", {})
    status = "SUCCESS" if validation.get("valid") else "LTL_VALIDATION_FAILED"
    bindings = tuple(
        item for item in payload.get("bindings", []) if isinstance(item, dict)
    )
    return TranslationResult(
        status=status,
        formula=formula,
        raw_ltl=json.dumps(payload.get("ltl_payload", {}), ensure_ascii=False),
        normalized_ltl=str(formula.simplify()),
        goals=goals,
        avoids=avoids,
        entity_bindings=bindings,
        ambiguous_bindings=(),
        unsupported_constraints=_unsupported_constraints(str(instruction.get("instruction", "")).lower()),
        validation={
            **validation,
            "cache_key": cache_key,
            "cache_status": cache_status,
            "raw_grounding_response": payload.get("raw_grounding_response"),
            "raw_ltl_response": payload.get("raw_ltl_response"),
            "llm_model": payload.get("model"),
        },
    )
