"""SemPathBench-specific grounding adapter around ILN planning calls."""

from __future__ import annotations

import json
import re
from typing import Mapping, Optional, Sequence

from scripts.methods.iln.graph_builder import _layer, _value_at
from scripts.methods.iln.graph_serializer import serialize_graph_for_prompt
from scripts.methods.iln.graph_types import (
    AreaPassageGraph,
    GroundedStage,
    GroundingResult,
    object_key_to_id,
    room_key_to_id,
)
from scripts.methods.iln.llm_client import GeminiClient, extract_json_object
from scripts.methods.iln.prompts import (
    STRICT_DESTINATION_SYSTEM_PROMPT,
    strict_destination_user_prompt,
)
from scripts.methods.util.map_objects import (
    object_cells_by_id as rich_object_cells_by_id,
    object_room_id as rich_object_room_id,
)


def _tokens(text: str) -> list[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [part for part in cleaned.split() if part]


def _phrase(category: str) -> str:
    return category.replace("_", " ").lower()


def _compact_with_positions(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char.isalnum():
            chars.append(char)
            positions.append(index)
    return "".join(chars), positions


def _instances(map_state: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    raw = map_state.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _object_room_map(map_state: Mapping[str, object]) -> dict[int, int]:
    result: dict[int, int] = {}
    for object_id in rich_object_cells_by_id(map_state):
        room_id = rich_object_room_id(map_state, object_id)
        if room_id is not None:
            result[object_id] = room_id
    return result


def _object_category_map(map_state: Mapping[str, object]) -> dict[int, str]:
    categories: dict[int, str] = {}
    for item in _instances(map_state, "object_instances"):
        raw_id = item.get("id")
        if isinstance(raw_id, int) and not isinstance(raw_id, bool):
            categories[raw_id] = str(item.get("category") or "object").lower()
    return categories


def _area_landmarks(map_state: Mapping[str, object], graph: AreaPassageGraph) -> dict[str, list[str]]:
    """Expose only area-level object semantics, never object instance targets."""
    object_rooms = _object_room_map(map_state)
    object_categories = _object_category_map(map_state)
    grouped: dict[str, set[str]] = {area.key: set() for area in graph.areas.values()}
    for object_id, room_id in object_rooms.items():
        area = graph.areas.get(room_id)
        category = object_categories.get(object_id)
        if area is not None and category:
            grouped[area.key].add(category)
    return {
        area_key: sorted(categories) for area_key, categories in sorted(grouped.items())
    }


def _replacement_object_id(
    *,
    object_id: int,
    requested_area_id: int | None,
    object_rooms: Mapping[int, int],
    object_categories: Mapping[int, str],
) -> int | None:
    category = object_categories.get(object_id)
    if not category:
        return None
    candidates = [
        candidate_id
        for candidate_id, candidate_category in object_categories.items()
        if candidate_category == category and candidate_id in object_rooms
    ]
    if requested_area_id is not None:
        same_area = [
            candidate_id
            for candidate_id in candidates
            if object_rooms.get(candidate_id) == requested_area_id
        ]
        if same_area:
            return min(same_area)
    return min(candidates) if candidates else None


# These aliases are evaluated when the module is imported.  Use Optional here
# rather than ``int | None`` so ILN can run in the project's Python 3.9
# Lang2LTL environment as well as newer interpreters.
Candidate = tuple[str, str, int, Optional[int]]
Mention = tuple[int, str, str, int, Optional[int]]


def _find_mentions(instruction_text: str, candidates: list[Candidate]) -> list[Mention]:
    text = instruction_text.lower().replace("_", " ")
    compact_text, compact_positions = _compact_with_positions(text)
    mentions: list[Mention] = []
    for kind, category, entity_id, room_id in candidates:
        phrase = _phrase(category)
        match = re.search(rf"\b{re.escape(phrase)}s?\b", text)
        position: int | None = match.start() if match else None
        if position is None:
            compact_phrase = "".join(char for char in phrase if char.isalnum())
            compact_index = compact_text.find(compact_phrase)
            if compact_index >= 0 and compact_index < len(compact_positions):
                position = compact_positions[compact_index]
        if position is not None:
            mentions.append((position, kind, category, entity_id, room_id))
    return sorted(mentions, key=lambda item: (item[0], item[2]))


def _soft_or_start_context(instruction_text: str, position: int) -> bool:
    text = instruction_text.lower().replace("_", " ")
    before = text[max(0, position - 28) : position]
    after = text[position : min(len(text), position + 28)]
    return any(
        marker in before
        for marker in (
            "start from",
            "starting from",
            "closer to",
            "farther from",
            "far from",
            "towards",
            "toward",
        )
    ) or "beside you" in after


def _room_stage_context(instruction_text: str, position: int) -> bool:
    text = instruction_text.lower().replace("_", " ")
    before = text[max(0, position - 58) : position]
    after = text[position : min(len(text), position + 58)]
    return any(
        marker in before
        for marker in (
            "pass through",
            "go through",
            "must pass",
            "via",
            "visit",
            "go to",
            "reach",
            "navigate to",
        )
    ) or "in sequence" in after


def heuristic_grounding(
    map_state: Mapping[str, object],
    graph: AreaPassageGraph,
    instruction: Mapping[str, object],
) -> GroundingResult:
    instruction_text = str(instruction.get("instruction") or "")
    room_candidates: list[Candidate] = [
        ("area", area.category, area.id, area.id)
        for area in sorted(graph.areas.values(), key=lambda item: item.id)
    ]
    object_candidates: list[Candidate] = []
    object_rooms = _object_room_map(map_state)
    for item in _instances(map_state, "object_instances"):
        raw_id = item.get("id")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            continue
        category = str(item.get("category") or "object").lower()
        object_candidates.append(("object", category, raw_id, object_rooms.get(raw_id)))

    mentions = _find_mentions(instruction_text, object_candidates + room_candidates)
    room_mentions: list[Mention] = []
    object_mentions: list[Mention] = []
    seen_categories: set[tuple[str, str]] = set()
    seen_unsupported: set[str] = set()
    unsupported_preferences: list[dict[str, object]] = []
    for position, kind, category, entity_id, room_id in mentions:
        if room_id is None or room_id not in graph.areas:
            continue
        if kind == "object" and _soft_or_start_context(instruction_text, position):
            if category not in seen_unsupported:
                seen_unsupported.add(category)
                unsupported_preferences.append(
                    {
                        "text": category,
                        "reason": "object_mention_in_start_or_soft_preference_context",
                    }
                )
            continue
        if kind == "area" and not _room_stage_context(instruction_text, position):
            continue
        category_key = (kind, category)
        if category_key in seen_categories:
            continue
        seen_categories.add(category_key)
        if kind == "area":
            room_mentions.append((position, kind, category, entity_id, room_id))
        else:
            object_mentions.append((position, kind, category, entity_id, room_id))

    stages: list[GroundedStage] = []
    for _position, _kind, category, entity_id, room_id in room_mentions:
        stages.append(
            GroundedStage(
                stage_id=f"stage_{len(stages) + 1}",
                target_type="area",
                area_id=room_id,
                object_id=None,
                source=f"heuristic_{category}_mention",
            )
        )
    if object_mentions:
        _position, _kind, category, entity_id, room_id = object_mentions[0]
        stages.append(
            GroundedStage(
                stage_id=f"stage_{len(stages) + 1}",
                target_type="object",
                area_id=room_id,
                object_id=entity_id,
                source=f"heuristic_final_{category}_mention",
            )
        )
    if not stages:
        fallback_area = next((area.id for area in graph.areas.values() if area.id != graph.start_area), graph.start_area)
        if fallback_area is not None:
            stages.append(
                GroundedStage(
                    stage_id="stage_1",
                    target_type="area",
                    area_id=fallback_area,
                    source="heuristic_fallback_area",
                )
            )
    constraints = {
        "forbidden_areas": [],
        "soft_avoid_areas": [],
        "forbidden_passages": [],
        "unsupported_preferences": unsupported_preferences,
    }
    notes = [
        "heuristic_grounding_uses_instruction_text_and_visible_map_instances_only",
        "instruction_objects_annotation_not_used",
    ]
    return GroundingResult("heuristic", stages, constraints, notes)


def heuristic_strict_grounding(
    map_state: Mapping[str, object],
    graph: AreaPassageGraph,
    instruction: Mapping[str, object],
) -> GroundingResult:
    """Choose one final area without object targets, stages, or route constraints."""
    instruction_text = str(instruction.get("instruction") or "")
    object_rooms = _object_room_map(map_state)
    candidates: list[Candidate] = [
        ("area", area.category, area.id, area.id)
        for area in sorted(graph.areas.values(), key=lambda item: item.id)
    ]
    for item in _instances(map_state, "object_instances"):
        raw_id = item.get("id")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            continue
        room_id = object_rooms.get(raw_id)
        if room_id in graph.areas:
            candidates.append(
                ("object", str(item.get("category") or "object").lower(), raw_id, room_id)
            )

    valid_mentions: list[Mention] = []
    for mention in _find_mentions(instruction_text, candidates):
        position, kind, _category, _entity_id, room_id = mention
        if room_id is None or room_id not in graph.areas:
            continue
        if kind == "object" and _soft_or_start_context(instruction_text, position):
            continue
        if kind == "area" and not _room_stage_context(instruction_text, position):
            continue
        valid_mentions.append(mention)

    if valid_mentions:
        position = max(item[0] for item in valid_mentions)
        final_mentions = [item for item in valid_mentions if item[0] == position]
        # Repeated categories are resolved at area level only. This stable
        # choice does not create or repair an object-instance prediction.
        selected = min(final_mentions, key=lambda item: (int(item[4]), item[2]))
        destination_area = int(selected[4])
        source = f"final_{selected[1]}_mention"
    else:
        fallback = next(
            (
                area.id
                for area in sorted(graph.areas.values(), key=lambda item: item.id)
                if area.id != graph.start_area
            ),
            graph.start_area,
        )
        if fallback is None:
            return GroundingResult(
                "heuristic",
                [],
                {},
                ["no_destination_area_available"],
            )
        destination_area = fallback
        source = "heuristic_fallback_area"

    stage = GroundedStage(
        stage_id="destination",
        target_type="area",
        area_id=destination_area,
        object_id=None,
        source=source,
    )
    return GroundingResult(
        "heuristic",
        [stage],
        {},
        [
            "single_area_destination_only",
            "no_object_instance_goal",
            "no_ordered_stage_decomposition",
            "no_instruction_derived_graph_constraints",
        ],
    )


def _parse_strict_destination(
    payload: Mapping[str, object],
    graph: AreaPassageGraph,
) -> GroundingResult:
    area_id = room_key_to_id(payload.get("destination_area"))
    if area_id is None or area_id not in graph.areas:
        raise ValueError("ILN returned an unknown destination area.")
    stage = GroundedStage(
        stage_id="destination",
        target_type="area",
        area_id=area_id,
        object_id=None,
        source="llm_destination_area",
    )
    return GroundingResult(
        "llm",
        [stage],
        {},
        [
            "single_area_destination_only",
            "no_object_instance_goal",
            "no_ordered_stage_decomposition",
            "no_instruction_derived_graph_constraints",
        ],
        raw_response=dict(payload),
    )


def _parse_llm_grounding(
    payload: Mapping[str, object],
    graph: AreaPassageGraph,
    map_state: Mapping[str, object],
) -> GroundingResult:
    object_rooms = _object_room_map(map_state)
    object_categories = _object_category_map(map_state)
    stages_raw = payload.get("stages")
    stages: list[GroundedStage] = []
    notes: list[str] = []
    if isinstance(stages_raw, list):
        for index, item in enumerate(stages_raw, start=1):
            if not isinstance(item, Mapping):
                continue
            area_id = room_key_to_id(item.get("area_id"))
            target_type = str(item.get("target_type") or "area")
            object_id = object_key_to_id(item.get("object_id")) if target_type == "object" else None
            if object_id is not None:
                if object_id not in object_rooms:
                    replacement_id = _replacement_object_id(
                        object_id=object_id,
                        requested_area_id=area_id,
                        object_rooms=object_rooms,
                        object_categories=object_categories,
                    )
                    if replacement_id is not None:
                        notes.append(
                            f"replaced_unlocalizable_object: object_{object_id} -> object_{replacement_id}"
                        )
                        object_id = replacement_id
                object_room = object_rooms.get(object_id)
                if object_room is not None and object_room in graph.areas and object_room != area_id:
                    notes.append(
                        f"corrected_object_area: object_{object_id} area {area_id} -> {object_room}"
                    )
                    area_id = object_room
            if area_id is None or area_id not in graph.areas:
                continue
            stages.append(
                GroundedStage(
                    stage_id=str(item.get("stage_id") or f"stage_{index}"),
                    target_type="object" if object_id is not None else "area",
                    area_id=area_id,
                    object_id=object_id,
                    source=str(item.get("source") or "llm"),
                )
            )
    constraints = payload.get("adapter_graph_constraints")
    if not isinstance(constraints, Mapping):
        constraints = {}
    normalized_constraints = {
        "forbidden_areas": list(constraints.get("forbidden_areas", [])) if isinstance(constraints.get("forbidden_areas"), list) else [],
        "soft_avoid_areas": list(constraints.get("soft_avoid_areas", [])) if isinstance(constraints.get("soft_avoid_areas"), list) else [],
        "forbidden_passages": list(constraints.get("forbidden_passages", [])) if isinstance(constraints.get("forbidden_passages"), list) else [],
        "unsupported_preferences": list(constraints.get("unsupported_preferences", [])) if isinstance(constraints.get("unsupported_preferences"), list) else [],
    }
    raw_notes = payload.get("adapter_notes")
    if isinstance(raw_notes, list):
        notes.extend(str(item) for item in raw_notes)
    return GroundingResult(
        "llm",
        stages,
        normalized_constraints,
        notes,
        raw_response=dict(payload),
    )


def ground_instruction(
    map_state: Mapping[str, object],
    graph: AreaPassageGraph,
    instruction: Mapping[str, object],
    *,
    mode: str,
    llm_client: GeminiClient | None = None,
    max_prompt_objects: int = 120,
) -> GroundingResult:
    """Select the single destination area used by the canonical ILN method."""
    if mode == "heuristic":
        return heuristic_strict_grounding(map_state, graph, instruction)
    if mode not in {"llm", "auto"}:
        raise ValueError(f"Unknown ILN grounding mode: {mode}")

    instruction_text = str(instruction.get("instruction") or "")
    graph_payload = serialize_graph_for_prompt(
        graph,
        map_state=map_state,
        instruction_text=instruction_text,
        max_prompt_objects=max_prompt_objects,
    )
    try:
        if llm_client is None:
            raise RuntimeError("llm_client is required for ILN destination selection.")
        text, cache_info = llm_client.response_text(
            system_prompt=STRICT_DESTINATION_SYSTEM_PROMPT,
            user_prompt=strict_destination_user_prompt(
                instruction_text=instruction_text,
                graph_payload=graph_payload,
                area_landmarks=_area_landmarks(map_state, graph),
            ),
            cache_namespace="destination_area_v1",
        )
        payload = extract_json_object(text)
        result = _parse_strict_destination(payload, graph)
        result.notes.append(json.dumps(cache_info, sort_keys=True))
        return result
    except Exception as exc:
        if mode == "auto":
            result = heuristic_strict_grounding(map_state, graph, instruction)
            result.mode = "auto_heuristic_fallback"
            result.notes.append(f"llm_destination_failed: {exc}")
            return result
        raise
