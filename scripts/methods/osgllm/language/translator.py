"""Deterministic first-pass grounding and LTL translation.

This module is deliberately conservative: it uses only instruction text and
map-derived scene-graph categories/attributes. It does not read benchmark target
annotations such as hard constraints or human trajectories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re

from scripts.methods.lang2ltl.ltl import (
    Formula,
    TRUE,
    atomic_propositions,
    ltl_always,
    ltl_and,
    ltl_eventually,
    ltl_not,
)
from scripts.methods.limp.grounding.candidates import ALIASES
from scripts.methods.limp.utils.geometry import normalize_name
from scripts.methods.osgllm.scene_graph.graph_types import AttributeRegion, Cell, SceneGraph
from scripts.methods.util.grid_astar import octile_heuristic


SOFT_UNSUPPORTED_PATTERNS = (
    ("near_preference", r"\btry to (?:be |walk )?(?:closer|nearer|close)\b|\bshould be closer\b"),
    ("far_preference", r"\btry to (?:be |walk )?(?:farther|further)\b|\bstay farther away\b"),
    ("relative_preference", r"\bcloser to .+ than\b|\bfurther from .+ than\b"),
    ("path_shape_preference", r"\bwalk around\b|\bcircl(?:e|ing)\b|\balong the wall\b|\bhuman-like\b"),
    ("door_preference", r"\bdoor (?:closer|further|farther|at the other end)\b"),
)


@dataclass(frozen=True)
class TranslationResult:
    status: str
    formula: Formula
    raw_ltl: str
    normalized_ltl: str
    goals: tuple[str, ...]
    avoids: tuple[str, ...]
    entity_bindings: tuple[dict[str, object], ...]
    ambiguous_bindings: tuple[dict[str, object], ...]
    unsupported_constraints: tuple[dict[str, object], ...]
    validation: dict[str, object]


def _ap(region: AttributeRegion) -> str:
    return region.ap


def _formula_ap(ap_name: str) -> Formula:
    return Formula("ap", value=ap_name)


def _sequence_formula(goals: tuple[str, ...]) -> Formula:
    if not goals:
        return TRUE
    result = ltl_eventually(_formula_ap(goals[-1]))
    for goal in reversed(goals[:-1]):
        result = ltl_eventually(ltl_and(_formula_ap(goal), result))
    return result.simplify()


def _with_avoids(goal_formula: Formula, avoids: tuple[str, ...]) -> Formula:
    result = goal_formula
    for avoid in avoids:
        result = ltl_and(result, ltl_always(ltl_not(_formula_ap(avoid))))
    return result.simplify()


def _text_terms(region: AttributeRegion) -> set[str]:
    terms = {
        region.category.replace("_", " "),
        normalize_name(region.category).replace("_", " "),
        normalize_name(region.name).replace("_", " "),
    }
    for attr in region.attributes:
        cleaned = re.sub(r"^[a-z_]+=", "", str(attr))
        terms.add(normalize_name(cleaned).replace("_", " "))
        terms.add(str(cleaned).strip().lower())
    result: set[str] = set()
    for term in terms:
        term = re.sub(r"\s+", " ", term.strip().lower())
        if len(term) < 3:
            continue
        result.add(term)
        normalized = normalize_name(term)
        alias = ALIASES.get(normalized)
        if alias:
            result.add(alias.replace("_", " "))
        if term.endswith("s"):
            result.add(term[:-1])
        else:
            result.add(term + "s")
    return result


def _compact_text(value: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def _region_distance(start: Cell, region: AttributeRegion) -> float:
    if not region.cells:
        return math.inf
    center = (int(round(region.center[0])), int(round(region.center[1])))
    best = octile_heuristic(start, center)
    sample_stride = max(1, len(region.cells) // 200)
    for cell in region.cells[::sample_stride]:
        best = min(best, octile_heuristic(start, cell))
    return best


def _choose_region(
    candidates: list[AttributeRegion],
    *,
    start: Cell,
    mode: str,
    already_selected: set[str],
) -> tuple[AttributeRegion | None, dict[str, object] | None]:
    if not candidates:
        return None, None
    scored = [
        (_region_distance(start, region), region.instance_id or 0, region)
        for region in candidates
    ]
    if mode == "farthest":
        scored.sort(key=lambda item: (-item[0], item[1]))
    else:
        scored.sort(key=lambda item: (item[0], item[1]))
    unused = [item for item in scored if item[2].node_id not in already_selected]
    chosen = (unused or scored)[0][2]
    ambiguity = None
    if len(scored) > 1:
        ambiguity = {
            "status": "tie_broken_or_ranked",
            "chosen": chosen.node_id,
            "candidates": [item[2].node_id for item in scored[:10]],
            "rule": f"{mode}_octile_distance_then_instance_id",
        }
    return chosen, ambiguity


def _mention_mode(text: str, start_index: int, end_index: int) -> str:
    window = text[max(0, start_index - 80) : min(len(text), end_index + 80)]
    if re.search(r"\bfarthest\b|\bfurthest\b|\bfurther away\b|\bfarther away\b", window):
        return "farthest"
    return "nearest"


def _room_context(
    text: str,
    regions: Sequence[AttributeRegion],
    start_index: int,
    end_index: int,
) -> set[str]:
    window = text[max(0, start_index - 80) : min(len(text), end_index + 100)]
    result: set[str] = set()
    for room in regions:
        for term in _text_terms(room):
            if re.search(rf"\b{re.escape(term)}\b", window):
                result.add(room.node_id)
    return result


def _find_mentions(
    text: str,
    graph: SceneGraph,
    *,
    start: Cell,
    ignored_spans: Sequence[tuple[int, int]] = (),
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    rooms = list(graph.by_kind("room"))
    objects = list(graph.by_kind("object"))
    entries: list[tuple[int, int, str, str, list[AttributeRegion]]] = []
    category_groups: dict[tuple[str, str], list[AttributeRegion]] = {}
    for region in rooms + objects:
        category_groups.setdefault((region.kind, region.category), []).append(region)
    for (kind, category), candidates in category_groups.items():
        terms: set[str] = set()
        for region in candidates:
            terms.update(_text_terms(region))
        for term in terms:
            if term in {"room", "object", "area"}:
                continue
            for match in re.finditer(rf"\b{re.escape(term)}\b", text):
                entries.append((match.start(), match.end(), kind, term, candidates))

    entries.sort(key=lambda item: (item[0], -(item[1] - item[0]), 0 if item[2] == "room" else 1))
    selected: list[str] = []
    bindings: list[dict[str, object]] = []
    ambiguities: list[dict[str, object]] = []
    occupied_spans: list[tuple[int, int]] = []
    selected_nodes: set[str] = set()
    for start_index, end_index, kind, term, candidates in entries:
        if any(start_index < span_end and end_index > span_start for span_start, span_end in ignored_spans):
            continue
        if any(start_index < used_end and end_index > used_start for used_start, used_end in occupied_spans):
            continue
        filtered = list(candidates)
        if kind == "object":
            context_rooms = _room_context(text, rooms, start_index, end_index)
            if context_rooms:
                scoped = [
                    region
                    for region in filtered
                    if region.parent_id in context_rooms
                ]
                if scoped:
                    filtered = scoped
        mode = _mention_mode(text, start_index, end_index)
        chosen, ambiguity = _choose_region(
            filtered,
            start=start,
            mode=mode,
            already_selected=selected_nodes,
        )
        if chosen is None:
            continue
        selected.append(_ap(chosen))
        selected_nodes.add(chosen.node_id)
        occupied_spans.append((start_index, end_index))
        bindings.append(
            {
                "text_span": term,
                "entity_id": chosen.node_id,
                "ap": _ap(chosen),
                "kind": chosen.kind,
                "category": chosen.category,
                "parent_id": chosen.parent_id,
                "selection_rule": mode,
            }
        )
        if ambiguity is not None:
            ambiguities.append(ambiguity)
    return selected, bindings, ambiguities


def _hard_goal_ignored_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(
        r"\bstart(?:ing)? from\b.{0,120}?(?=\byour task\b|\byou need\b|\bneed to\b|\bnow you need\b|$)",
        text,
    ):
        spans.append((match.start(), match.end()))
    soft_pattern = (
        r"\b(?:try to|closer to|farther away|further away|walk around|"
        r"circl\w*|along the wall|door closer|door further|door farther|stand closer)\b"
    )
    for match in re.finditer(soft_pattern, text):
        spans.append((max(0, match.start() - 40), min(len(text), match.end() + 140)))
    return tuple(spans)


def _sequence_clause_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\bpass through\b.{0,180}?\bin sequence\b", text):
        spans.append((match.start(), match.end()))
    return tuple(spans)


def _dedupe_goals(goals: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(goals))


def _remove_context_room_goals(
    goals: Sequence[str],
    bindings: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    object_parent_aps: set[str] = set()
    for binding in bindings:
        if binding.get("kind") != "object":
            continue
        parent_id = binding.get("parent_id")
        if isinstance(parent_id, str) and parent_id.startswith("room_"):
            object_parent_aps.add(f"enter({parent_id})")
    return tuple(goal for goal in goals if goal not in object_parent_aps)


def _find_avoids(
    text: str,
    graph: SceneGraph,
    *,
    start: Cell,
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    avoid_spans = list(re.finditer(r"\b(?:avoid|not allowed to enter|must not enter|do not enter)\b.{0,100}", text))
    avoids: list[str] = []
    bindings: list[dict[str, object]] = []
    ambiguities: list[dict[str, object]] = []
    if not avoid_spans:
        return avoids, bindings, ambiguities
    for span in avoid_spans:
        snippet = span.group(0)
        scoped_graph_text = _compact_text(snippet)
        goals, scoped_bindings, scoped_ambiguities = _find_mentions(scoped_graph_text, graph, start=start)
        avoids.extend(goals)
        bindings.extend(
            {
                **binding,
                "avoid_context": True,
            }
            for binding in scoped_bindings
        )
        ambiguities.extend(scoped_ambiguities)
    return avoids, bindings, ambiguities


def _unsupported_constraints(text: str) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for case, pattern in SOFT_UNSUPPORTED_PATTERNS:
        if re.search(pattern, text):
            records.append(
                {
                    "case": case,
                    "status": "unsupported",
                    "reason": "Boolean OSG-LTL planner does not model this soft/path-shape preference in the first local adapter.",
                }
            )
    return tuple(records)


def translate_instruction_heuristic(
    graph: SceneGraph,
    instruction: Mapping[str, object],
    *,
    start: Cell,
) -> TranslationResult:
    text = _compact_text(str(instruction.get("instruction", "")))
    sequence_spans = _sequence_clause_spans(text)
    sequence_goals: list[str] = []
    sequence_bindings: list[dict[str, object]] = []
    sequence_ambiguities: list[dict[str, object]] = []
    for start_index, end_index in sequence_spans:
        snippet = text[start_index:end_index]
        clause_goals, clause_bindings, clause_ambiguities = _find_mentions(
            snippet,
            graph,
            start=start,
        )
        sequence_goals.extend(clause_goals)
        sequence_bindings.extend(clause_bindings)
        sequence_ambiguities.extend(clause_ambiguities)

    ignored_spans = _hard_goal_ignored_spans(text) + sequence_spans
    goals, bindings, ambiguities = _find_mentions(
        text,
        graph,
        start=start,
        ignored_spans=ignored_spans,
    )
    avoids, avoid_bindings, avoid_ambiguities = _find_avoids(text, graph, start=start)
    goals = list(_remove_context_room_goals(goals, bindings))
    goals = [goal for goal in sequence_goals + goals if goal not in set(avoids)]
    goals_tuple = _dedupe_goals(goals)
    formula = _with_avoids(_sequence_formula(goals_tuple), tuple(dict.fromkeys(avoids)))
    used_aps = atomic_propositions(formula)
    inventory = set(graph.ap_inventory())
    missing = sorted(used_aps - inventory)
    status = "SUCCESS" if goals and not missing else "GROUNDING_FAILED"
    validation = {
        "used_atomic_propositions": sorted(used_aps),
        "missing_atomic_propositions": missing,
        "valid": not missing and bool(goals),
        "backend": "local_formula_progression",
    }
    return TranslationResult(
        status=status,
        formula=formula,
        raw_ltl=str(formula),
        normalized_ltl=str(formula.simplify()),
        goals=goals_tuple,
        avoids=tuple(dict.fromkeys(avoids)),
        entity_bindings=tuple(sequence_bindings + bindings + avoid_bindings),
        ambiguous_bindings=tuple(sequence_ambiguities + ambiguities + avoid_ambiguities),
        unsupported_constraints=_unsupported_constraints(text),
        validation=validation,
    )
