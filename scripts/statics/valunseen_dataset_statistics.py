#!/usr/bin/env python3
"""Summarize SemPathBench valunseen scenes and instructions.

The script reads canonical map and instruction JSON files only. It reports
scene-level inventory statistics and episode-level instruction/constraint
statistics for overall, easy, and hard subsets.
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import median
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_ROOT = REPO_ROOT / "resources" / "maps" / "procthor" / "valunseen"
DEFAULT_INSTRUCTION_ROOT = REPO_ROOT / "resources" / "instructions" / "procthor"
DEFAULT_EASY_REFERENCE_PATH = Path(__file__).resolve().parent / "easy_ref.json"
DEFAULT_HARD_REFERENCE_PATH = Path(__file__).resolve().parent / "hard_ref.json"
SECTIONS = ("overall", "easy", "hard")


def load_object(source_file: Path) -> dict[str, object]:
    payload = json.loads(source_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {source_file}")
    return payload


def instances(payload: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    raw_instances = payload.get(key)
    if not isinstance(raw_instances, Sequence) or isinstance(raw_instances, (str, bytes)):
        return []
    return [item for item in raw_instances if isinstance(item, Mapping)]


def category(instance: Mapping[str, object]) -> str:
    raw_category = instance.get("category")
    return str(raw_category).strip() if raw_category is not None else "unknown"


def map_paths(map_root: Path) -> list[Path]:
    maps = sorted(map_root.glob("*_valunseen/*_valunseen.json"))
    if not maps:
        raise ValueError(f"No valunseen map JSON files found under {map_root}")
    return maps


def instruction_paths(instruction_root: Path) -> list[Path]:
    instructions = sorted(
        instruction_root.glob("*_valunseen/instruction_files/instruction_[0-9][0-9][0-9][0-9][0-9][0-9].json")
    )
    if not instructions:
        raise ValueError(f"No valunseen instruction JSON files found under {instruction_root}")
    return instructions


def scene_statistics(maps: Sequence[Path]) -> dict[str, object]:
    object_categories: set[str] = set()
    room_categories: set[str] = set()
    object_counts: list[int] = []
    room_counts: list[int] = []

    for map_file in maps:
        payload = load_object(map_file)
        scene_objects = instances(payload, "object_instances")
        scene_rooms = instances(payload, "room_instances")
        object_counts.append(len(scene_objects))
        room_counts.append(len(scene_rooms))
        object_categories.update(category(item) for item in scene_objects)
        room_categories.update(category(item) for item in scene_rooms)

    scene_count = len(maps)
    return {
        "scene_count": scene_count,
        "object_instances": {
            "total": sum(object_counts),
            "average_per_scene": sum(object_counts) / scene_count,
            "unique_category_count": len(object_categories),
            "categories": sorted(object_categories),
        },
        "room_instances": {
            "total": sum(room_counts),
            "average_per_scene": sum(room_counts) / scene_count,
            "unique_category_count": len(room_categories),
            "categories": sorted(room_categories),
        },
    }


def empty_episode_bucket() -> dict[str, Any]:
    return {
        "episode_count": 0,
        "instruction_word_total": 0,
        "instruction_non_whitespace_character_total": 0,
        "room_transition_total": 0,
        "rooms_visited_total": 0,
        "room_transition_count_per_episode": Counter(),
        "hard_constraints": Counter(),
        "additional_hard_constraint_count_per_episode": Counter(),
        "soft_constraint_count_per_episode": Counter(),
        "soft_constraints_excluding_clearance": Counter(),
        "clearance_constraints_excluded": 0,
    }


def normalized_difficulty(payload: Mapping[str, object]) -> str:
    raw_difficulty = payload.get("difficulty_level")
    return str(raw_difficulty).strip().lower() if raw_difficulty is not None else "unknown"


def room_traversal_counts(payload: Mapping[str, object], room_layer: object) -> tuple[int, int]:
    trajectory = payload.get("human_expert_trajectory")
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        return 0, 0
    if not isinstance(room_layer, Sequence) or isinstance(room_layer, (str, bytes)):
        return 0, 0
    traversed_rooms: list[int] = []
    for point in trajectory:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            continue
        row, column = point[0], point[1]
        if not isinstance(row, int) or isinstance(row, bool) or not isinstance(column, int) or isinstance(column, bool):
            continue
        if row < 0 or row >= len(room_layer):
            continue
        room_row = room_layer[row]
        if not isinstance(room_row, Sequence) or isinstance(room_row, (str, bytes)) or column < 0 or column >= len(room_row):
            continue
        room_id = room_row[column]
        if not isinstance(room_id, int) or isinstance(room_id, bool) or room_id <= 0:
            continue
        if not traversed_rooms or traversed_rooms[-1] != room_id:
            traversed_rooms.append(room_id)
    return max(0, len(traversed_rooms) - 1), len(traversed_rooms)


def add_episode(
    bucket: dict[str, Any],
    payload: Mapping[str, object],
    room_transitions: int,
    rooms_visited: int,
) -> None:
    text = str(payload.get("instruction", ""))
    bucket["episode_count"] += 1
    bucket["instruction_word_total"] += len(text.split())
    bucket["instruction_non_whitespace_character_total"] += sum(
        not character.isspace() for character in text
    )
    bucket["room_transition_total"] += room_transitions
    bucket["rooms_visited_total"] += rooms_visited
    bucket["room_transition_count_per_episode"][room_transitions] += 1

    episode_hard_constraints = instances(payload, "hard_constraints")
    episode_soft_constraints = instances(payload, "soft_constraints")
    bucket["additional_hard_constraint_count_per_episode"][max(0, len(episode_hard_constraints) - 1)] += 1
    bucket["soft_constraint_count_per_episode"][len(episode_soft_constraints)] += 1
    for constraint in episode_hard_constraints:
        kind = str(constraint.get("kind", "unknown")).strip() or "unknown"
        bucket["hard_constraints"][kind] += 1

    for constraint in episode_soft_constraints:
        preference_type = str(constraint.get("preference_type", "unknown")).strip() or "unknown"
        if preference_type == "clearance":
            bucket["clearance_constraints_excluded"] += 1
        else:
            bucket["soft_constraints_excluding_clearance"][preference_type] += 1


def finalize_episode_bucket(bucket: Mapping[str, Any]) -> dict[str, object]:
    episode_count = int(bucket["episode_count"])
    hard_counts: Counter[str] = bucket["hard_constraints"]
    soft_counts: Counter[str] = bucket["soft_constraints_excluding_clearance"]
    hard_total = sum(hard_counts.values())
    return {
        "episode_count": episode_count,
        "instruction_length": {
            "average_words": bucket["instruction_word_total"] / episode_count if episode_count else None,
            "average_non_whitespace_characters": (
                bucket["instruction_non_whitespace_character_total"] / episode_count
                if episode_count
                else None
            ),
        },
        "room_traversal": {
            "average_room_transitions": bucket["room_transition_total"] / episode_count if episode_count else None,
            "average_rooms_visited": bucket["rooms_visited_total"] / episode_count if episode_count else None,
            "transition_count_distribution": {
                str(count): bucket["room_transition_count_per_episode"][count]
                for count in range(max(bucket["room_transition_count_per_episode"], default=0) + 1)
            },
        },
        "hard_constraints": {
            "total": hard_total,
            "must_pass": hard_counts["must_pass"],
            "must_avoid": hard_counts["must_avoid"],
            "other": hard_total - hard_counts["must_pass"] - hard_counts["must_avoid"],

        },
        "constraint_count_distribution": {
            "additional_hard_constraints": {
                str(count): bucket["additional_hard_constraint_count_per_episode"][count]
                for count in range(max(bucket["additional_hard_constraint_count_per_episode"], default=0) + 1)
            },
            "soft_constraints_including_clearance": {
                str(count): bucket["soft_constraint_count_per_episode"][count]
                for count in range(max(bucket["soft_constraint_count_per_episode"], default=0) + 1)
            },
        },
        "soft_constraints_excluding_clearance": {
            "total": sum(soft_counts.values()),
            "by_preference_type": dict(sorted(soft_counts.items())),
            "clearance_constraints_excluded": bucket["clearance_constraints_excluded"],
        },
    }


def episode_statistics(instructions: Sequence[Path], maps: Sequence[Path]) -> dict[str, object]:
    buckets = {section: empty_episode_bucket() for section in SECTIONS}
    map_by_scene = {map_file.parent.name: map_file for map_file in maps}
    instruction_groups: dict[str, list[Path]] = {}
    for instruction_file in instructions:
        instruction_groups.setdefault(instruction_file.parent.parent.name, []).append(instruction_file)

    unknown_difficulty_count = 0
    for scene_name, scene_instructions in instruction_groups.items():
        map_file = map_by_scene.get(scene_name)
        if map_file is None:
            raise ValueError(f"No map JSON found for instruction scene: {scene_name}")
        map_payload = load_object(map_file)
        layers = map_payload.get("layers")
        room_layer = layers.get("room") if isinstance(layers, Mapping) else None
        for instruction_file in scene_instructions:
            payload = load_object(instruction_file)
            room_transitions, rooms_visited = room_traversal_counts(payload, room_layer)
            add_episode(buckets["overall"], payload, room_transitions, rooms_visited)
            difficulty = normalized_difficulty(payload)
            if difficulty in {"easy", "hard"}:
                add_episode(buckets[difficulty], payload, room_transitions, rooms_visited)
            else:
                unknown_difficulty_count += 1

    result: dict[str, object] = {
        section: finalize_episode_bucket(buckets[section]) for section in SECTIONS
    }
    result["unknown_difficulty_episode_count"] = unknown_difficulty_count
    return result

def normalize_reference_category(value: object) -> str:
    normalized = "_".join(str(value).strip().lower().replace("-", " ").split())
    aliases = {
        "armchair": "arm_chair",
        "basketball": "basket_ball",
        "countertop": "counter_top",
        "houseplant": "house_plant",
        "tv": "television",
    }
    return aliases.get(normalized, normalized)


def value_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "maximum": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": median(ordered),
        "p90": ordered[max(0, math.ceil(len(ordered) * 0.9) - 1)],
        "maximum": ordered[-1],
    }


def reference_records(reference_path: Path) -> dict[int, list[Mapping[str, object]]]:
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a reference list: {reference_path}")
    records: dict[int, list[Mapping[str, object]]] = {}
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("instruction_id")
        try:
            instruction_id = int(str(raw_id))
        except (TypeError, ValueError):
            continue
        references = item.get("references")
        records[instruction_id] = [reference for reference in references if isinstance(reference, Mapping)] if isinstance(references, Sequence) else []
    return records


def candidate_catalog(map_payload: Mapping[str, object]) -> tuple[Counter[str], Counter[str], int, int]:
    object_counts = Counter(normalize_reference_category(instance.get("category", "")) for instance in instances(map_payload, "object_instances"))
    room_counts = Counter(normalize_reference_category(instance.get("category", "")) for instance in instances(map_payload, "room_instances"))
    return object_counts, room_counts, sum(object_counts.values()), sum(room_counts.values())


def candidate_for_reference(
    reference: Mapping[str, object],
    catalog: tuple[Counter[str], Counter[str], int, int],
) -> tuple[str, int] | None:
    object_counts, room_counts, total_objects, total_rooms = catalog
    category = normalize_reference_category(reference.get("category", ""))
    if category in {"room", "area", "space"}:
        return "room", total_rooms
    if category == "object":
        return "object", total_objects
    if category in room_counts:
        return "room", room_counts[category]
    if category in object_counts:
        return "object", object_counts[category]
    return None


def grounding_statistics(instructions: Sequence[Path], maps: Sequence[Path]) -> dict[str, object]:
    reference_by_difficulty = {
        "easy": reference_records(DEFAULT_EASY_REFERENCE_PATH),
        "hard": reference_records(DEFAULT_HARD_REFERENCE_PATH),
    }
    instruction_by_difficulty: dict[str, list[Path]] = {"easy": [], "hard": []}
    for instruction_file in instructions:
        payload = load_object(instruction_file)
        difficulty = normalized_difficulty(payload)
        if difficulty in instruction_by_difficulty:
            instruction_by_difficulty[difficulty].append(instruction_file)

    map_by_scene = {map_file.parent.name: map_file for map_file in maps}
    catalogs: dict[str, tuple[Counter[str], Counter[str], int, int]] = {}
    for scene_name, map_file in map_by_scene.items():
        catalogs[scene_name] = candidate_catalog(load_object(map_file))

    result: dict[str, object] = {
        "candidate_definition": "Category-level candidates before relational filtering; generic room/area/space refers to all room instances, and generic object refers to all object instances.",
        "depth_definition": "Human-annotated referential / relational depth from the reference JSON files.",
    }
    for difficulty, instruction_files in instruction_by_difficulty.items():
        records = reference_by_difficulty[difficulty]
        all_candidates: list[float] = []
        object_candidates: list[float] = []
        room_candidates: list[float] = []
        depths: list[float] = []
        episode_max_depths: list[float] = []
        episode_total_depths: list[float] = []
        complexity_scores: list[float] = []
        unknown_categories: Counter[str] = Counter()
        annotated_episode_count = 0
        for ordinal, instruction_file in enumerate(instruction_files, start=1):
            references = records.get(ordinal)
            if references is None:
                continue
            annotated_episode_count += 1
            catalog = catalogs[instruction_file.parent.parent.name]
            episode_depths: list[float] = []
            for reference in references:
                raw_depth = reference.get("depth")
                if isinstance(raw_depth, (int, float)) and not isinstance(raw_depth, bool):
                    depth = float(raw_depth)
                    depths.append(depth)
                    episode_depths.append(depth)
                else:
                    depth = 0.0
                candidate = candidate_for_reference(reference, catalog)
                if candidate is None:
                    unknown_categories[str(reference.get("category", ""))] += 1
                    continue
                kind, count = candidate
                all_candidates.append(float(count))
                (object_candidates if kind == "object" else room_candidates).append(float(count))
                complexity_scores.append(math.log2(1 + count) * depth)
            episode_max_depths.append(max(episode_depths, default=0.0))
            episode_total_depths.append(sum(episode_depths))
        result[difficulty] = {
            "annotated_episode_count": annotated_episode_count,
            "reference_count": len(depths),
            "candidate_instances_before_relations": {
                "all_references": value_summary(all_candidates),
                "object_references": value_summary(object_candidates),
                "room_references": value_summary(room_candidates),
                "ambiguity_rate": sum(value > 1 for value in all_candidates) / len(all_candidates) if all_candidates else None,
                "unresolved_reference_category_count": sum(unknown_categories.values()),
                "unresolved_reference_categories": dict(unknown_categories.most_common()),
            },
            "referential_relational_depth": {
                "per_reference": value_summary(depths),
                "episode_maximum": value_summary(episode_max_depths),
                "episode_total": value_summary(episode_total_depths),
            },
            "grounding_complexity_proxy": {
                "definition": "depth * log2(1 + category-level candidate instances)",
                "per_reference": value_summary(complexity_scores),
            },
        }
    return result


def display_source(source_path: Path) -> str:
    try:
        return str(source_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(source_path)


def build_report(map_root: Path, instruction_root: Path) -> dict[str, object]:
    maps = map_paths(map_root)
    instructions = instruction_paths(instruction_root)
    return {
        "dataset": "SemPathBench Procthor valunseen",
        "source": {
            "map_root": display_source(map_root),
            "instruction_root": display_source(instruction_root),
        },
        "scene_statistics": scene_statistics(maps),
        "episode_statistics": episode_statistics(instructions, maps),
        "grounding_statistics": grounding_statistics(instructions, maps),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the 40 SemPathBench Procthor valunseen scenes and their episodes."
    )
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--instruction-root", type=Path, default=DEFAULT_INSTRUCTION_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path to write in addition to printing the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.map_root.resolve(), args.instruction_root.resolve())
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
