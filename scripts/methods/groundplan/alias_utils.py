"""Helpers for learned GroundPlan category aliases."""

from __future__ import annotations

import pprint
from pathlib import Path
from typing import Mapping

from scripts.methods.groundplan import alias


ALIAS_PATH = Path(__file__).resolve().parent / "alias.py"


def normalize_category_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def learned_entity_aliases() -> dict[str, str]:
    raw = getattr(alias, "LEARNED_ENTITY_ALIASES", {})
    if not isinstance(raw, Mapping):
        return {}
    return {
        normalize_category_name(key): normalize_category_name(value)
        for key, value in raw.items()
        if normalize_category_name(key) and normalize_category_name(value)
    }


def learned_room_aliases() -> dict[str, str]:
    raw = getattr(alias, "LEARNED_ROOM_ALIASES", {})
    if not isinstance(raw, Mapping):
        return {}
    return {
        normalize_category_name(key): normalize_category_name(value)
        for key, value in raw.items()
        if normalize_category_name(key) and normalize_category_name(value)
    }


def merged_entity_aliases(base_aliases: Mapping[str, str] | None = None) -> dict[str, str]:
    aliases = {
        normalize_category_name(key): normalize_category_name(value)
        for key, value in (base_aliases or {}).items()
        if normalize_category_name(key) and normalize_category_name(value)
    }
    aliases.update(learned_entity_aliases())
    return aliases


def merged_room_aliases(base_aliases: Mapping[str, str] | None = None) -> dict[str, str]:
    aliases = {
        normalize_category_name(key): normalize_category_name(value)
        for key, value in (base_aliases or {}).items()
        if normalize_category_name(key) and normalize_category_name(value)
    }
    aliases.update(learned_room_aliases())
    return aliases


def register_entity_alias(source: str, target: str) -> dict[str, str]:
    source_key = normalize_category_name(source)
    target_key = normalize_category_name(target)
    if not source_key or not target_key:
        raise ValueError("Alias source and target must be non-empty category names.")

    entity_aliases = learned_entity_aliases()
    room_aliases = learned_room_aliases()
    entity_aliases[source_key] = target_key
    setattr(alias, "LEARNED_ENTITY_ALIASES", dict(entity_aliases))
    _write_alias_file(entity_aliases, room_aliases)
    return dict(entity_aliases)


def _write_alias_file(
    entity_aliases: Mapping[str, str],
    room_aliases: Mapping[str, str],
) -> None:
    text = (
        '"""Learned GroundPlan category aliases.\n\n'
        "This file is updated by the GroundPlan alias resolver when Gemini maps an\n"
        "unknown parsed category to a supported scene category.\n"
        '"""\n\n'
        f"LEARNED_ENTITY_ALIASES = {pprint.pformat(dict(sorted(entity_aliases.items())), sort_dicts=True)}\n\n"
        f"LEARNED_ROOM_ALIASES = {pprint.pformat(dict(sorted(room_aliases.items())), sort_dicts=True)}\n"
    )
    ALIAS_PATH.write_text(text, encoding="utf-8")
