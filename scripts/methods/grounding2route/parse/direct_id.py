"""Direct instance-ID grounding from a compact semantic scene catalog."""

from __future__ import annotations

import json
import re
from typing import Mapping, Sequence

from scripts.methods.grounding2route.config import PROMPT_DIR
from scripts.methods.grounding2route.grounding.scene import SceneMap
from scripts.methods.grounding2route.ir import (
    GPConstraintSpec,
    GPExpr,
    GPProgramSpec,
    GPRef,
    GPSegmentSpec,
)
from scripts.methods.grounding2route.llm_client import GeminiClient
from scripts.methods.grounding2route.parse.scene_catalog import (
    catalog_reference,
    compact_scene_catalog,
)


class DirectIDParseError(ValueError):
    pass


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONSTRAINT_KINDS = {
    "require_visit",
    "require_visit_in_order",
    "forbid",
    "prefer_near",
    "prefer_far",
    "prefer_relative",
    "prefer_path_shape",
}


def llm_parse_direct_id_instruction(
    instruction_text: str,
    *,
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    llm_client: GeminiClient,
    max_parse_repairs: int = 1,
) -> tuple[GPProgramSpec, dict[str, object]]:
    """Ask the LLM to select final IDs from a compact, annotation-free map."""

    scene = SceneMap(map_state, instruction)
    catalog = compact_scene_catalog(map_state, instruction, scene=scene)
    prompt = _direct_id_prompt(instruction_text, catalog)
    text, meta = llm_client.response_text(
        system_prompt=(
            "You ground route instructions to IDs in a compact semantic scene "
            "catalog. Return JSON only."
        ),
        user_prompt=prompt,
        cache_namespace="grounding2route_direct_id_catalog_v1",
    )
    attempts: list[dict[str, object]] = [
        {"attempt": 0, "kind": "initial_direct_id", "llm": meta, "raw_direct_id": text}
    ]
    try:
        payload = parse_direct_id_json(text)
        program = direct_id_to_program(payload, scene=scene, source=text)
    except DirectIDParseError as exc:
        attempts[0].update({"status": "failed", "error": str(exc)})
        if max_parse_repairs <= 0:
            _attach_error_context(exc, text=text, meta=meta, attempts=attempts)
            raise
        repair_prompt = (
            f"{prompt}\n\n"
            "## Validation repair\n\n"
            f"The previous response failed validation:\n{exc}\n\n"
            f"Previous response:\n{text}\n\n"
            "Return one complete corrected JSON object and nothing else."
        )
        repaired, repair_meta = llm_client.response_text(
            system_prompt="Repair the Direct-ID grounding JSON. Return JSON only.",
            user_prompt=repair_prompt,
            cache_namespace="grounding2route_direct_id_catalog_parse_repair_v1",
        )
        try:
            payload = parse_direct_id_json(repaired)
            program = direct_id_to_program(payload, scene=scene, source=repaired)
        except DirectIDParseError as repair_exc:
            attempts.append(
                {
                    "attempt": 1,
                    "kind": "direct_id_parse_repair",
                    "status": "failed",
                    "error": str(repair_exc),
                    "llm": repair_meta,
                    "raw_direct_id": repaired,
                }
            )
            _attach_error_context(
                repair_exc,
                text=repaired,
                meta=repair_meta,
                attempts=attempts,
            )
            raise
        attempts.append(
            {
                "attempt": 1,
                "kind": "direct_id_parse_repair",
                "status": "success",
                "llm": repair_meta,
                "raw_direct_id": repaired,
            }
        )
        text = repaired
    attempts[-1]["status"] = "success"
    return program, {
        "mode": "direct_id",
        "llm": meta,
        "raw_direct_id": text,
        "catalog_schema_version": catalog["schema_version"],
        "catalog_room_count": len(catalog["rooms"]),  # type: ignore[arg-type]
        "catalog_object_count": len(catalog["objects"]),  # type: ignore[arg-type]
        "catalog_serialized_bytes": len(
            json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ),
        "parse_attempts": attempts,
    }


def parse_direct_id_json(text: str) -> Mapping[str, object]:
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise DirectIDParseError(f"DIRECT_ID_JSON_ERROR: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DirectIDParseError("Direct-ID response must be one JSON object.")
    return payload


def direct_id_to_program(
    payload: Mapping[str, object],
    *,
    scene: SceneMap,
    source: str = "",
) -> GPProgramSpec:
    """Validate final IDs and compile them to the shared Grounding2Route IR."""

    raw_segments = payload.get("segments")
    raw_constraints = payload.get("constraints", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        raise DirectIDParseError("Direct-ID output needs a non-empty segments array.")
    if not isinstance(raw_constraints, list):
        raise DirectIDParseError("Direct-ID constraints must be an array.")

    segment_records: list[tuple[str, GPRef]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, Mapping):
            raise DirectIDParseError(f"Segment {index} must be an object.")
        segment_id = _name(raw.get("id", f"s{index}"), f"segment {index} id")
        if segment_id in seen_ids:
            raise DirectIDParseError(f"Duplicate segment ID: {segment_id}")
        seen_ids.add(segment_id)
        try:
            target = catalog_reference(scene, raw.get("target_id"))
        except ValueError as exc:
            raise DirectIDParseError(f"Segment {segment_id}: {exc}") from exc
        if target.kind == "position":
            raise DirectIDParseError(f"Segment {segment_id} cannot target task_start.")
        segment_records.append((segment_id, target))

    constraints_by_segment: dict[str, list[GPConstraintSpec]] = {
        segment_id: [] for segment_id, _target in segment_records
    }
    for index, raw in enumerate(raw_constraints, start=1):
        if not isinstance(raw, Mapping):
            raise DirectIDParseError(f"Constraint {index} must be an object.")
        kind = raw.get("kind")
        if not isinstance(kind, str) or kind not in _CONSTRAINT_KINDS:
            raise DirectIDParseError(f"Constraint {index} has unsupported kind: {kind!r}")
        refs = _references(scene, raw.get("ref_ids"), f"constraint {index} ref_ids")
        _validate_constraint_arity(kind, refs, index)
        raw_segment_ids = raw.get("segment_ids")
        if raw_segment_ids is None or raw_segment_ids == []:
            segment_ids = [segment_id for segment_id, _target in segment_records]
        else:
            segment_ids = _string_list(raw_segment_ids, f"constraint {index} segment_ids")
            unknown = [item for item in segment_ids if item not in constraints_by_segment]
            if unknown:
                raise DirectIDParseError(
                    f"Constraint {index} references unknown segment IDs: {unknown}"
                )
        spatial_scope = _spatial_scope(scene, raw.get("spatial_scope_ids"), index)
        exprs: tuple[object, ...] = tuple(refs)
        if kind == "prefer_relative":
            relation = str(raw.get("relation") or "closer_to")
            exprs = (refs[0], relation, refs[1])
        elif kind == "prefer_path_shape":
            exprs = _path_shape_exprs(refs, raw.get("path_shape"), index)
        source_text = raw.get("source_text")
        for segment_id in segment_ids:
            constraints_by_segment[segment_id].append(
                GPConstraintSpec(
                    kind=kind,  # type: ignore[arg-type]
                    exprs=exprs,
                    segment_scope=(segment_id,),
                    spatial_scope=spatial_scope,
                    source_text=(
                        str(source_text)
                        if source_text is not None
                        else f"direct_id_constraint_{index}"
                    ),
                )
            )

    segments: list[GPSegmentSpec] = []
    previous: GPRef = scene.start
    for segment_id, target in segment_records:
        segments.append(
            GPSegmentSpec(
                id=segment_id,
                start=previous,
                target=target,
                constraints=tuple(constraints_by_segment[segment_id]),
            )
        )
        previous = target
    return GPProgramSpec(
        bindings=(),
        segments=tuple(segments),
        source=source or json.dumps(payload, ensure_ascii=False),
        parser_mode="direct_id",
        diagnostics=(
            "direct_id_compact_scene_catalog",
            "no_dense_map_arrays_in_llm_prompt",
        ),
    )


def _references(scene: SceneMap, value: object, label: str) -> tuple[GPRef, ...]:
    ids = _string_list(value, label)
    refs: list[GPRef] = []
    for reference_id in ids:
        try:
            refs.append(catalog_reference(scene, reference_id))
        except ValueError as exc:
            raise DirectIDParseError(str(exc)) from exc
    return tuple(refs)


def _spatial_scope(scene: SceneMap, value: object, index: int) -> object | None:
    if value is None or value == []:
        return None
    refs = _references(scene, value, f"constraint {index} spatial_scope_ids")
    if not all(ref.kind in {"room", "entity"} for ref in refs):
        raise DirectIDParseError(f"Constraint {index} has invalid spatial scope.")
    return refs[0] if len(refs) == 1 else list(refs)


def _path_shape_exprs(
    refs: tuple[GPRef, ...],
    value: object,
    index: int,
) -> tuple[object, ...]:
    if value is None:
        return tuple(refs)
    if not isinstance(value, Mapping):
        raise DirectIDParseError(f"Constraint {index} path_shape must be an object.")
    shape_type = str(value.get("type") or "through").strip().lower()
    if shape_type == "through":
        return tuple(refs)
    if shape_type not in {"circle", "follow_wall"}:
        raise DirectIDParseError(
            f"Constraint {index} has unsupported path-shape type: {shape_type}"
        )
    if len(refs) != 1:
        raise DirectIDParseError(
            f"Constraint {index} path-shape {shape_type} requires one reference ID."
        )
    kwargs = {
        str(key): item
        for key, item in value.items()
        if key in {"fraction", "direction"}
    }
    return (GPExpr(shape_type, (refs[0],), kwargs),)


def _validate_constraint_arity(kind: str, refs: Sequence[GPRef], index: int) -> None:
    if not refs:
        raise DirectIDParseError(f"Constraint {index} requires at least one ref_id.")
    if kind in {"prefer_near", "prefer_far"} and len(refs) != 1:
        raise DirectIDParseError(f"Constraint {index} {kind} requires exactly one ref_id.")
    if kind == "prefer_relative" and len(refs) != 2:
        raise DirectIDParseError(
            f"Constraint {index} prefer_relative requires exactly two ref_ids."
        )


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise DirectIDParseError(f"{label} must be a non-empty string array.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DirectIDParseError(f"{label} must contain only non-empty strings.")
        result.append(item.strip())
    return result


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise DirectIDParseError(f"{label} must be a valid identifier.")
    return value


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise DirectIDParseError("Direct-ID response did not contain a JSON object.")
    return stripped[start : end + 1]


def _direct_id_prompt(instruction_text: str, catalog: Mapping[str, object]) -> str:
    template = (PROMPT_DIR / "parse" / "gemini_direct_id_parser_prompt.md").read_text(
        encoding="utf-8"
    )
    return (
        template.replace("{{INSTRUCTION}}", instruction_text.strip()).replace(
            "{{SCENE_CATALOG}}",
            json.dumps(catalog, ensure_ascii=False, separators=(",", ":")),
        )
    )


def _attach_error_context(
    error: DirectIDParseError,
    *,
    text: str,
    meta: Mapping[str, object],
    attempts: list[dict[str, object]],
) -> None:
    setattr(error, "raw_direct_id", text)
    setattr(error, "llm", dict(meta))
    setattr(error, "parse_attempts", attempts)
