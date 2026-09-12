"""Instruction parsing for GroundPlan."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from scripts.methods.groundplan.alias_utils import merged_entity_aliases, merged_room_aliases
from scripts.methods.groundplan.config import ENTITY_ALIASES, ROOM_ALIASES
from scripts.methods.groundplan.llm_client import GeminiClient
from scripts.methods.groundplan.parse.api_program import (
    APIProgramParseError,
    llm_parse_api_program_instruction,
)
from scripts.methods.groundplan.parse.intent import (
    IntentParseError,
    llm_parse_intent_instruction,
)
from scripts.methods.groundplan.parse.direct_id import (
    DirectIDParseError,
    llm_parse_direct_id_instruction,
)
from scripts.methods.groundplan.parse.ltl import LTLParseError, llm_parse_ltl_instruction
from scripts.methods.groundplan.ir import GPProgramSpec


class GroundPlanParseError(RuntimeError):
    def __init__(self, message: str, metadata: Mapping[str, object]) -> None:
        super().__init__(message)
        self.metadata = dict(metadata)


def _clean_category(value: object, aliases: Mapping[str, str] | None = None) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        key.replace("-", "_").replace(" ", "_"): target
        for key, target in (aliases or {}).items()
    }
    return alias_map.get(normalized, normalized)


def _instances(map_state: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    raw = map_state.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def supported_categories(map_state: Mapping[str, object]) -> tuple[list[str], list[str]]:
    entity_aliases = merged_entity_aliases(ENTITY_ALIASES)
    room_aliases = merged_room_aliases(ROOM_ALIASES)
    object_categories = sorted(
        {
            category
            for item in _instances(map_state, "object_instances")
            if (category := _clean_category(item.get("category"), entity_aliases))
        }
    )
    room_categories = sorted(
        {
            category
            for item in _instances(map_state, "room_instances")
            if (category := _clean_category(item.get("category"), room_aliases))
        }
    )
    return object_categories, room_categories


def scene_category_summary(map_state: Mapping[str, object]) -> dict[str, object]:
    entity_aliases = merged_entity_aliases(ENTITY_ALIASES)
    room_aliases = merged_room_aliases(ROOM_ALIASES)
    object_counts = Counter(
        category
        for item in _instances(map_state, "object_instances")
        if (category := _clean_category(item.get("category"), entity_aliases))
    )
    room_counts = Counter(
        category
        for item in _instances(map_state, "room_instances")
        if (category := _clean_category(item.get("category"), room_aliases))
    )
    return {
        "room_counts": dict(sorted(room_counts.items())),
        "object_counts": dict(sorted(object_counts.items())),
    }


def parse_instruction(
    instruction: Mapping[str, object],
    map_state: Mapping[str, object],
    *,
    mode: str,
    llm_client: GeminiClient | None = None,
    max_parse_repairs: int = 0,
) -> tuple[GPProgramSpec, dict[str, object]]:
    if mode not in {"intent", "api", "direct_id", "ltl"}:
        raise GroundPlanParseError(
            (
                "GroundPlan parser supports intent, api, direct_id, and ltl modes; "
                f"unsupported parse mode: {mode}"
            ),
            {"mode": mode, "status": "failed", "failure_reason": "UNSUPPORTED_PARSE_MODE"},
        )
    if llm_client is None:
        raise GroundPlanParseError(
            "llm_client is required for GroundPlan parser mode.",
            {"mode": mode, "status": "failed", "failure_reason": "LLM_CLIENT_REQUIRED"},
        )

    instruction_text = str(instruction.get("instruction") or "")
    entity_categories, room_categories = supported_categories(map_state)
    metadata: dict[str, object] = {
        "mode": mode,
        "instruction_text": instruction_text,
        "entity_category_count": len(entity_categories),
        "room_categories": room_categories,
    }
    try:
        if mode == "intent":
            program, llm_meta = llm_parse_intent_instruction(
                instruction_text,
                entity_categories=entity_categories,
                room_categories=room_categories,
                llm_client=llm_client,
                max_parse_repairs=max_parse_repairs,
            )
        elif mode == "api":
            program, llm_meta = llm_parse_api_program_instruction(
                instruction_text,
                entity_categories=entity_categories,
                room_categories=room_categories,
                llm_client=llm_client,
                max_parse_repairs=max_parse_repairs,
            )
        elif mode == "direct_id":
            program, llm_meta = llm_parse_direct_id_instruction(
                instruction_text,
                map_state=map_state,
                instruction=instruction,
                llm_client=llm_client,
                max_parse_repairs=max_parse_repairs,
            )
        else:
            program, llm_meta = llm_parse_ltl_instruction(
                instruction_text,
                map_state=map_state,
                instruction=instruction,
                llm_client=llm_client,
                max_parse_repairs=max_parse_repairs,
            )
    except IntentParseError as exc:
        metadata["status"] = "failed"
        metadata["failure_reason"] = f"INTENT_PARSE_ERROR: {exc}"
        if hasattr(exc, "raw_intent"):
            metadata["raw_intent"] = getattr(exc, "raw_intent")
        if hasattr(exc, "llm"):
            metadata["llm"] = getattr(exc, "llm")
        if hasattr(exc, "parse_attempts"):
            metadata["parse_attempts"] = getattr(exc, "parse_attempts")
        raise GroundPlanParseError(str(exc), metadata) from exc
    except APIProgramParseError as exc:
        metadata["status"] = "failed"
        metadata["failure_reason"] = f"API_PROGRAM_PARSE_ERROR: {exc}"
        if hasattr(exc, "raw_api_program"):
            metadata["raw_api_program"] = getattr(exc, "raw_api_program")
        if hasattr(exc, "llm"):
            metadata["llm"] = getattr(exc, "llm")
        if hasattr(exc, "parse_attempts"):
            metadata["parse_attempts"] = getattr(exc, "parse_attempts")
        raise GroundPlanParseError(str(exc), metadata) from exc
    except DirectIDParseError as exc:
        metadata["status"] = "failed"
        metadata["failure_reason"] = f"DIRECT_ID_PARSE_ERROR: {exc}"
        if hasattr(exc, "raw_direct_id"):
            metadata["raw_direct_id"] = getattr(exc, "raw_direct_id")
        if hasattr(exc, "llm"):
            metadata["llm"] = getattr(exc, "llm")
        if hasattr(exc, "parse_attempts"):
            metadata["parse_attempts"] = getattr(exc, "parse_attempts")
        raise GroundPlanParseError(str(exc), metadata) from exc
    except LTLParseError as exc:
        metadata["status"] = "failed"
        metadata["failure_reason"] = f"LTL_PARSE_ERROR: {exc}"
        if hasattr(exc, "raw_ltl"):
            metadata["raw_ltl"] = getattr(exc, "raw_ltl")
        if hasattr(exc, "llm"):
            metadata["llm"] = getattr(exc, "llm")
        if hasattr(exc, "parse_attempts"):
            metadata["parse_attempts"] = getattr(exc, "parse_attempts")
        raise GroundPlanParseError(str(exc), metadata) from exc
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failure_reason"] = f"PARSE_ERROR: {type(exc).__name__}: {exc}"
        raise GroundPlanParseError(str(exc), metadata) from exc

    metadata.update(llm_meta)
    metadata["status"] = "success"
    return program, metadata
