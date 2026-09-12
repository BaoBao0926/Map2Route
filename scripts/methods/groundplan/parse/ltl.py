"""Restricted LTL representation over compact-catalog propositions.

Standard LTL represents the ordered hard task.  Quantitative soft preferences
remain typed proposition annotations because Near/Far/Path-Shape are not
Boolean temporal operators.  Both parts compile to the same GroundPlan IR used
by every other representation experiment.
"""

from __future__ import annotations

import json
import re
from typing import Mapping

from scripts.methods.groundplan.config import PROMPT_DIR
from scripts.methods.groundplan.grounding.scene import SceneMap
from scripts.methods.groundplan.ir import GPProgramSpec
from scripts.methods.groundplan.llm_client import GeminiClient
from scripts.methods.groundplan.parse.direct_id import direct_id_to_program
from scripts.methods.groundplan.parse.scene_catalog import compact_scene_catalog


class LTLParseError(ValueError):
    pass


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROPOSITION_KINDS = {
    "visit",
    "avoid",
    "prefer_near",
    "prefer_far",
    "prefer_relative",
    "prefer_path_shape",
}

def llm_parse_ltl_instruction(
    instruction_text: str,
    *,
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    llm_client: GeminiClient,
    max_parse_repairs: int = 1,
) -> tuple[GPProgramSpec, dict[str, object]]:
    scene = SceneMap(map_state, instruction)
    catalog = compact_scene_catalog(map_state, instruction, scene=scene)
    prompt = _ltl_prompt(instruction_text, catalog)
    text, meta = llm_client.response_text(
        system_prompt=(
            "Translate route instructions into restricted LTL over grounded "
            "compact-catalog propositions. Return JSON only."
        ),
        user_prompt=prompt,
        cache_namespace="groundplan_ltl_catalog_v1",
    )
    attempts: list[dict[str, object]] = [
        {"attempt": 0, "kind": "initial_ltl", "llm": meta, "raw_ltl": text}
    ]
    try:
        payload = parse_ltl_json(text)
        program, formula = ltl_to_program(payload, scene=scene, source=text)
    except LTLParseError as exc:
        attempts[0].update({"status": "failed", "error": str(exc)})
        if max_parse_repairs <= 0:
            _attach_error_context(exc, text=text, meta=meta, attempts=attempts)
            raise
        repaired, repair_meta = llm_client.response_text(
            system_prompt="Repair the restricted GroundPlan LTL JSON. Return JSON only.",
            user_prompt=(
                f"{prompt}\n\n## Validation repair\n\n"
                f"Validation error:\n{exc}\n\nPrevious response:\n{text}\n\n"
                "Return one complete corrected JSON object and nothing else."
            ),
            cache_namespace="groundplan_ltl_catalog_parse_repair_v1",
        )
        try:
            payload = parse_ltl_json(repaired)
            program, formula = ltl_to_program(payload, scene=scene, source=repaired)
        except LTLParseError as repair_exc:
            attempts.append(
                {
                    "attempt": 1,
                    "kind": "ltl_parse_repair",
                    "status": "failed",
                    "error": str(repair_exc),
                    "llm": repair_meta,
                    "raw_ltl": repaired,
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
                "kind": "ltl_parse_repair",
                "status": "success",
                "llm": repair_meta,
                "raw_ltl": repaired,
            }
        )
        text = repaired
    attempts[-1]["status"] = "success"
    return program, {
        "mode": "ltl",
        "llm": meta,
        "raw_ltl": text,
        "ltl_formula": formula,
        "ltl_dialect": "restricted_F_G_X_boolean_with_typed_soft_propositions",
        "catalog_schema_version": catalog["schema_version"],
        "catalog_room_count": len(catalog["rooms"]),  # type: ignore[arg-type]
        "catalog_object_count": len(catalog["objects"]),  # type: ignore[arg-type]
        "catalog_serialized_bytes": len(
            json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ),
        "parse_attempts": attempts,
    }


def parse_ltl_json(text: str) -> Mapping[str, object]:
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise LTLParseError(f"LTL_JSON_ERROR: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LTLParseError("LTL response must be one JSON object.")
    return payload


def ltl_to_program(
    payload: Mapping[str, object],
    *,
    scene: SceneMap,
    source: str = "",
) -> tuple[GPProgramSpec, str]:
    """Validate restricted LTL and compile its propositions to GroundPlan IR."""

    formula = payload.get("formula")
    raw_propositions = payload.get("propositions")
    if not isinstance(formula, str) or not formula.strip():
        raise LTLParseError("LTL output needs a non-empty formula string.")
    if not isinstance(raw_propositions, list) or not raw_propositions:
        raise LTLParseError("LTL output needs a non-empty propositions array.")

    propositions: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(raw_propositions, start=1):
        if not isinstance(raw, Mapping):
            raise LTLParseError(f"Proposition {index} must be an object.")
        name = raw.get("name")
        kind = raw.get("kind")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise LTLParseError(f"Proposition {index} has an invalid name.")
        if name in propositions:
            raise LTLParseError(f"Duplicate proposition name: {name}")
        if not isinstance(kind, str) or kind not in _PROPOSITION_KINDS:
            raise LTLParseError(f"Proposition {name} has unsupported kind: {kind!r}")
        propositions[name] = raw

    ast = _LTLParser(formula).parse()
    formula_atoms = _atoms(ast)
    unknown = sorted(formula_atoms - set(propositions))
    if unknown:
        raise LTLParseError(f"LTL formula uses undefined propositions: {unknown}")
    visit_names = {
        name for name, proposition in propositions.items() if proposition.get("kind") == "visit"
    }
    visit_order = _ordered_visits(ast, visit_names)
    if not visit_order:
        raise LTLParseError("LTL formula must contain at least one eventual visit proposition.")
    if set(visit_order) != visit_names or len(visit_order) != len(visit_names):
        raise LTLParseError(
            "Every visit proposition must occur exactly once in one nested eventuality chain."
        )

    global_avoids = _global_avoid_atoms(ast)
    direct_segments: list[dict[str, object]] = []
    for visit_name in visit_order:
        proposition = propositions[visit_name]
        ref_ids = _ref_ids(proposition, visit_name)
        if len(ref_ids) != 1:
            raise LTLParseError(f"Visit proposition {visit_name} requires exactly one ref_id.")
        direct_segments.append({"id": visit_name, "target_id": ref_ids[0]})

    direct_constraints: list[dict[str, object]] = []
    for name, proposition in propositions.items():
        kind = str(proposition["kind"])
        if kind == "visit":
            continue
        ref_ids = _ref_ids(proposition, name)
        raw_during = proposition.get("during")
        if raw_during is None:
            during = list(visit_order)
        else:
            during = _name_list(raw_during, f"proposition {name} during")
            invalid = [item for item in during if item not in visit_names]
            if invalid:
                raise LTLParseError(
                    f"Proposition {name} has unknown visit scopes: {invalid}"
                )
        if kind == "avoid" and raw_during is None and name not in formula_atoms:
            raise LTLParseError(f"Global avoid proposition {name} is absent from the LTL formula.")
        if kind == "avoid" and raw_during is None and name not in global_avoids:
            raise LTLParseError(
                f"Global avoid proposition {name} must appear as G(!{name})."
            )
        direct_kind = "forbid" if kind == "avoid" else kind
        record: dict[str, object] = {
            "kind": direct_kind,
            "ref_ids": ref_ids,
            "segment_ids": during,
            "source_text": proposition.get("source_text", f"ltl_proposition_{name}"),
        }
        for key in ("spatial_scope_ids", "relation", "path_shape"):
            if key in proposition:
                record[key] = proposition[key]
        direct_constraints.append(record)

    direct_payload = {
        "segments": direct_segments,
        "constraints": direct_constraints,
    }
    try:
        program = direct_id_to_program(direct_payload, scene=scene, source=source)
    except Exception as exc:
        raise LTLParseError(f"LTL_GROUNDING_SCHEMA_ERROR: {exc}") from exc
    program = GPProgramSpec(
        bindings=program.bindings,
        segments=program.segments,
        source=source or formula,
        parser_mode="ltl",
        diagnostics=(
            "restricted_ltl_compiled_to_groundplan_ir",
            "soft_preferences_are_typed_annotations_outside_boolean_ltl",
            "no_dense_map_arrays_in_llm_prompt",
        ),
    )
    return program, formula.strip()


class _LTLParser:
    def __init__(self, formula: str) -> None:
        self.tokens = _tokenize(formula)
        self.index = 0

    def parse(self) -> tuple[object, ...]:
        expression = self._or()
        if self.index != len(self.tokens):
            raise LTLParseError(f"Unexpected LTL token: {self.tokens[self.index]}")
        return expression

    def _or(self) -> tuple[object, ...]:
        node = self._and()
        while self._peek("|"):
            self.index += 1
            node = ("or", node, self._and())
        return node

    def _and(self) -> tuple[object, ...]:
        node = self._unary()
        while self._peek("&"):
            self.index += 1
            node = ("and", node, self._unary())
        return node

    def _unary(self) -> tuple[object, ...]:
        if self.index >= len(self.tokens):
            raise LTLParseError("Unexpected end of LTL formula.")
        token = self.tokens[self.index]
        if token in {"!", "F", "G", "X"}:
            self.index += 1
            return ({"!": "not", "F": "F", "G": "G", "X": "X"}[token], self._unary())
        if token == "(":
            self.index += 1
            node = self._or()
            if not self._peek(")"):
                raise LTLParseError("Unclosed parenthesis in LTL formula.")
            self.index += 1
            return node
        if token == ")":
            raise LTLParseError("Unexpected ')' in LTL formula.")
        self.index += 1
        return ("atom", token)

    def _peek(self, token: str) -> bool:
        return self.index < len(self.tokens) and self.tokens[self.index] == token


def _tokenize(formula: str) -> list[str]:
    normalized = (
        formula.strip().replace("◇", "F")
        .replace("□", "G")
        .replace("¬", "!")
        .replace("∧", "&")
        .replace("∨", "|")
        .replace("<>", "F")
        .replace("[]", "G")
        .replace("&&", "&")
        .replace("||", "|")
        .replace("~", "!")
    )
    tokens: list[str] = []
    position = 0
    pattern = re.compile(r"\s*([()!&|]|[A-Za-z_][A-Za-z0-9_]*)")
    while position < len(normalized):
        match = pattern.match(normalized, position)
        if match is None:
            raise LTLParseError(
                f"Unsupported LTL syntax near: {normalized[position:position + 20]!r}"
            )
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        raise LTLParseError("LTL formula is empty.")
    return tokens


def _flatten_and(node: tuple[object, ...]) -> list[tuple[object, ...]]:
    if node[0] == "and":
        return _flatten_and(node[1]) + _flatten_and(node[2])  # type: ignore[arg-type]
    return [node]


def _ordered_visits(
    ast: tuple[object, ...],
    visit_names: set[str],
) -> list[str]:
    candidates: list[list[str]] = []
    for conjunct in _flatten_and(ast):
        sequence = _eventual_chain(conjunct, visit_names)
        if sequence:
            candidates.append(sequence)
    if len(candidates) != 1:
        raise LTLParseError(
            "Use one ordered eventuality chain such as F(v1 & F(v2))."
        )
    return candidates[0]


def _eventual_chain(node: tuple[object, ...], visit_names: set[str]) -> list[str]:
    if node[0] != "F":
        return []
    return _chain_body(node[1], visit_names)  # type: ignore[arg-type]


def _chain_body(node: tuple[object, ...], visit_names: set[str]) -> list[str]:
    if node[0] == "atom":
        name = str(node[1])
        return [name] if name in visit_names else []
    if node[0] == "F":
        return _chain_body(node[1], visit_names)  # type: ignore[arg-type]
    if node[0] == "and":
        result: list[str] = []
        for child in _flatten_and(node):
            if child[0] == "atom" and str(child[1]) in visit_names:
                result.append(str(child[1]))
            elif child[0] == "F":
                result.extend(_chain_body(child[1], visit_names))  # type: ignore[arg-type]
        return result
    return []


def _atoms(node: tuple[object, ...]) -> set[str]:
    if node[0] == "atom":
        name = str(node[1])
        return set() if name.lower() in {"true", "false"} else {name}
    result: set[str] = set()
    for child in node[1:]:
        if isinstance(child, tuple):
            result.update(_atoms(child))
    return result


def _global_avoid_atoms(ast: tuple[object, ...]) -> set[str]:
    result: set[str] = set()
    for conjunct in _flatten_and(ast):
        if (
            conjunct[0] == "G"
            and isinstance(conjunct[1], tuple)
            and conjunct[1][0] == "not"  # type: ignore[index]
            and isinstance(conjunct[1][1], tuple)  # type: ignore[index]
            and conjunct[1][1][0] == "atom"  # type: ignore[index]
        ):
            result.add(str(conjunct[1][1][1]))  # type: ignore[index]
    return result


def _ref_ids(proposition: Mapping[str, object], name: str) -> list[str]:
    return _name_list(proposition.get("ref_ids"), f"proposition {name} ref_ids")


def _name_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise LTLParseError(f"{label} must be a non-empty string array.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LTLParseError(f"{label} must contain only non-empty strings.")
        result.append(item.strip())
    return result


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise LTLParseError("LTL response did not contain a JSON object.")
    return stripped[start : end + 1]


def _ltl_prompt(instruction_text: str, catalog: Mapping[str, object]) -> str:
    template = (PROMPT_DIR / "parse" / "gemini_ltl_parser_prompt.md").read_text(
        encoding="utf-8"
    )
    return (
        template.replace("{{INSTRUCTION}}", instruction_text.strip()).replace(
            "{{SCENE_CATALOG}}",
            json.dumps(catalog, ensure_ascii=False, separators=(",", ":")),
        )
    )


def _attach_error_context(
    error: LTLParseError,
    *,
    text: str,
    meta: Mapping[str, object],
    attempts: list[dict[str, object]],
) -> None:
    setattr(error, "raw_ltl", text)
    setattr(error, "llm", dict(meta))
    setattr(error, "parse_attempts", attempts)

