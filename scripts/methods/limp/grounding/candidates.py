"""Candidate lookup helpers for LIMP grounding."""

from __future__ import annotations

from collections.abc import Iterable

from scripts.methods.limp.grounding.scene_adapter import CandidateRegistry
from scripts.methods.limp.utils.geometry import normalize_name


ALIASES = {
    "basketball": "basket_ball",
    "basketballs": "basket_ball",
    "countertop": "counter_top",
    "countertops": "counter_top",
    "counter": "counter_top",
    "tv": "television",
    "door": "doorway",
    "doors": "doorway",
    "armchair": "arm_chair",
    "couch": "sofa",
    "plant": "house_plant",
    "houseplant": "house_plant",
    "houseplants": "house_plant",
    "garbagecan": "garbage_can",
    "tvstand": "tv_stand",
    "plants": "house_plant",
    "lamp": "floor_lamp",
    "lamps": "floor_lamp",
    "table": "dining_table",
    "tables": "dining_table",
    "original_living_room": "living_room",
    "next_living_room": "living_room",
    "other_living_room": "living_room",
    "original_room": "room",
    "next_room": "room",
    "livingroom": "living_room",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "kitchen": "kitchen",
    "diningroom": "dining_room",
}
EXTENDED_ONLY_ALIAS_KEYS = {"houseplant", "houseplants", "garbagecan", "tvstand"}


SELECTOR_TOKENS = {
    "nearest",
    "closest",
    "nearby",
    "farthest",
    "furthest",
    "first",
    "second",
    "third",
    "other",
    "another",
    "original",
    "previous",
    "next",
    "largest",
    "biggest",
    "smallest",
    "not",
}


def _lookup_keys(base: str, *, extended: bool = True) -> set[str]:
    normalized = normalize_name(base)
    raw_keys = {normalized}
    if not extended:
        alias = normalized if normalized in EXTENDED_ONLY_ALIAS_KEYS else ALIASES.get(normalized, normalized)
        keys = {normalized, alias}
        if normalized.endswith("s"):
            keys.add(normalized[:-1])
        if normalized.endswith("es"):
            keys.add(normalized[:-2])
        return {key for key in keys if key}

    parts = normalized.split("_")
    stripped = [part for part in parts if part not in SELECTOR_TOKENS and not part.isdigit()]
    if stripped and stripped != parts:
        raw_keys.add("_".join(stripped))

    keys = set(raw_keys)
    keys.update(ALIASES.get(key, key) for key in tuple(raw_keys))
    for key in tuple(keys):
        if key.endswith("s"):
            keys.add(key[:-1])
        if key.endswith("es"):
            keys.add(key[:-2])
    return {key for key in keys if key}


def lookup_candidates(
    registry: CandidateRegistry,
    base: str,
    *,
    extended: bool = True,
) -> tuple[str, ...]:
    keys = _lookup_keys(base, extended=extended)

    result: set[str] = set()
    for key in keys:
        result.update(registry.by_category.get(key, ()))

    if result:
        return tuple(sorted(result))

    # Last resort: substring match over category tokens. This still uses only
    # map-derived categories and does not inspect benchmark annotations.
    for category, candidate_ids in registry.by_category.items():
        for key in keys:
            if key and (key in category or category in key):
                result.update(candidate_ids)
    return tuple(sorted(result))


def candidates_to_records(registry: CandidateRegistry, candidate_ids: Iterable[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate_id in candidate_ids:
        candidate = registry.get(candidate_id)
        if candidate is None:
            continue
        records.append(
            {
                "candidate_id": candidate.candidate_id,
                "kind": candidate.kind,
                "instance_id": candidate.instance_id,
                "category": candidate.category,
                "name": candidate.name,
                "centroid": [candidate.centroid[0], candidate.centroid[1]],
                "room_id": candidate.room_id,
                "room_category": candidate.room_category,
            }
        )
    return records
