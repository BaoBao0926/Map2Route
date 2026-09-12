#!/usr/bin/env python3
"""Summarize capped instruction annotation time from created_at/updated_at fields."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION_ROOT = REPO_ROOT / "resources" / "instructions"
INSTRUCTION_SET_CHOICES = ("valunseen", "train", "all")
DIFFICULTY_DURATION_CAPS_SECONDS = {
    "easy": 10 * 60,
    "hard": 15 * 60,
}


@dataclass(frozen=True)
class AnnotationRecord:
    path: Path
    scene: str
    instruction_id: str
    difficulty: str
    created_at: datetime
    updated_at: datetime
    raw_duration_seconds: float
    duration_seconds: float
    duration_cap_seconds: int | None
    capped: bool


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_timestamp(value: object, *, path: Path, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{display_path(path)} missing {field}")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{display_path(path)} has invalid {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def instruction_id_for(path: Path, payload: dict[str, Any]) -> str:
    raw_id = payload.get("id")
    if isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id > 0:
        return f"instruction_{raw_id:06d}"
    return path.stem


def scene_key_for_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    if "instruction_files" in parts:
        return "/".join(parts[: parts.index("instruction_files")])
    return "/".join(parts[:-1])


def instruction_set_for_scene(scene: str) -> str:
    leaf = scene.strip("/").rsplit("/", 1)[-1].lower()
    if leaf.endswith("_valunseen") or leaf.startswith("valunseen_"):
        return "valunseen"
    if leaf.endswith("_train") or leaf.startswith("train_"):
        return "train"
    return ""


def normalize_instruction_set(value: str) -> str:
    normalized = value.strip().lower().replace("-", "").replace("_", "")
    if normalized in INSTRUCTION_SET_CHOICES:
        return normalized
    raise ValueError(
        f"unsupported set {value!r}; expected one of {', '.join(INSTRUCTION_SET_CHOICES)}"
    )


def duration_cap_for_difficulty(difficulty: str) -> int | None:
    return DIFFICULTY_DURATION_CAPS_SECONDS.get(difficulty)


def load_record(path: Path, root: Path) -> AnnotationRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{display_path(path)} must contain a JSON object")

    created_at = parse_timestamp(payload.get("created_at"), path=path, field="created_at")
    updated_at = parse_timestamp(payload.get("updated_at"), path=path, field="updated_at")
    raw_duration_seconds = (updated_at - created_at).total_seconds()
    if raw_duration_seconds < 0:
        raise ValueError(
            f"{display_path(path)} has updated_at before created_at: "
            f"{updated_at.isoformat()} < {created_at.isoformat()}"
        )
    difficulty = str(payload.get("difficulty_level") or "unknown").strip().lower() or "unknown"
    duration_cap_seconds = duration_cap_for_difficulty(difficulty)
    if duration_cap_seconds is None:
        duration_seconds = raw_duration_seconds
        capped = False
    else:
        duration_seconds = min(raw_duration_seconds, duration_cap_seconds)
        capped = raw_duration_seconds > duration_cap_seconds

    return AnnotationRecord(
        path=path,
        scene=scene_key_for_path(path, root),
        instruction_id=instruction_id_for(path, payload),
        difficulty=difficulty,
        created_at=created_at,
        updated_at=updated_at,
        raw_duration_seconds=raw_duration_seconds,
        duration_seconds=duration_seconds,
        duration_cap_seconds=duration_cap_seconds,
        capped=capped,
    )


def iter_instruction_files(
    root: Path,
    scene_filters: set[str],
    instruction_set: str,
) -> list[Path]:
    paths = sorted(root.rglob("instruction_files/instruction_*.json"))
    selected: list[Path] = []
    for path in paths:
        scene = scene_key_for_path(path, root)
        if scene_filters and scene not in scene_filters and path.parent.parent.name not in scene_filters:
            continue
        if instruction_set != "all" and instruction_set_for_scene(scene) != instruction_set:
            continue
        selected.append(path)
    return selected


def format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(records: list[AnnotationRecord]) -> dict[str, Any]:
    by_scene: dict[str, list[AnnotationRecord]] = defaultdict(list)
    by_difficulty: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for record in records:
        by_scene[record.scene].append(record)
        by_difficulty[record.difficulty].append(record)

    scene_summaries = []
    for scene, scene_records in sorted(by_scene.items()):
        durations = [record.duration_seconds for record in scene_records]
        raw_durations = [record.raw_duration_seconds for record in scene_records]
        scene_summaries.append(
            {
                "scene": scene,
                "instruction_count": len(scene_records),
                "total_seconds": sum(durations),
                "raw_total_seconds": sum(raw_durations),
                "average_instruction_seconds": statistics.fmean(durations),
                "capped_instruction_count": sum(1 for record in scene_records if record.capped),
                "easy_count": sum(1 for record in scene_records if record.difficulty == "easy"),
                "hard_count": sum(1 for record in scene_records if record.difficulty == "hard"),
                "unknown_count": sum(
                    1
                    for record in scene_records
                    if record.difficulty not in {"easy", "hard"}
                ),
            }
        )

    difficulty_summaries = {}
    for difficulty, difficulty_records in sorted(by_difficulty.items()):
        durations = [record.duration_seconds for record in difficulty_records]
        raw_durations = [record.raw_duration_seconds for record in difficulty_records]
        difficulty_summaries[difficulty] = {
            "instruction_count": len(difficulty_records),
            "total_seconds": sum(durations),
            "raw_total_seconds": sum(raw_durations),
            "average_instruction_seconds": statistics.fmean(durations),
            "duration_cap_seconds": duration_cap_for_difficulty(difficulty),
            "capped_instruction_count": sum(1 for record in difficulty_records if record.capped),
        }

    total_seconds = sum(record.duration_seconds for record in records)
    raw_total_seconds = sum(record.raw_duration_seconds for record in records)
    scene_total_seconds = [scene["total_seconds"] for scene in scene_summaries]

    return {
        "instruction_count": len(records),
        "scene_count": len(scene_summaries),
        "total_seconds": total_seconds,
        "raw_total_seconds": raw_total_seconds,
        "capped_instruction_count": sum(1 for record in records if record.capped),
        "duration_caps_seconds": dict(DIFFICULTY_DURATION_CAPS_SECONDS),
        "average_scene_seconds": mean_or_none(scene_total_seconds),
        "average_instruction_seconds": mean_or_none(
            [record.duration_seconds for record in records]
        ),
        "by_difficulty": difficulty_summaries,
        "by_scene": scene_summaries,
    }


def print_text_summary(summary: dict[str, Any], *, verbose: bool = False) -> None:
    print("Annotation time summary")
    print("  duration caps: easy<=10m 00s, hard<=15m 00s")
    print(f"  instructions: {summary['instruction_count']}")
    print(f"  scenes: {summary['scene_count']}")
    print(f"  total annotation time: {format_duration(summary['total_seconds'])}")
    print(f"  raw total before caps: {format_duration(summary['raw_total_seconds'])}")
    print(f"  capped instructions: {summary['capped_instruction_count']}")
    if summary["average_scene_seconds"] is not None:
        print(
            "  average per scene: "
            f"{format_duration(summary['average_scene_seconds'])}"
        )
    if summary["average_instruction_seconds"] is not None:
        print(
            "  average per instruction: "
            f"{format_duration(summary['average_instruction_seconds'])}"
        )

    print("\nBy difficulty")
    for difficulty in ("easy", "hard", "unknown"):
        block = summary["by_difficulty"].get(difficulty)
        if not block:
            continue
        print(
            f"  {difficulty}: count={block['instruction_count']} "
            f"total={format_duration(block['total_seconds'])} "
            f"avg={format_duration(block['average_instruction_seconds'])} "
            f"capped={block['capped_instruction_count']}"
        )
    for difficulty, block in summary["by_difficulty"].items():
        if difficulty in {"easy", "hard", "unknown"}:
            continue
        print(
            f"  {difficulty}: count={block['instruction_count']} "
            f"total={format_duration(block['total_seconds'])} "
            f"avg={format_duration(block['average_instruction_seconds'])} "
            f"capped={block['capped_instruction_count']}"
        )

    if verbose:
        print("\nBy scene")
        for scene in summary["by_scene"]:
            print(
                f"  {scene['scene']}: count={scene['instruction_count']} "
                f"easy={scene['easy_count']} hard={scene['hard_count']} "
                f"unknown={scene['unknown_count']} "
                f"total={format_duration(scene['total_seconds'])} "
                f"avg/instruction={format_duration(scene['average_instruction_seconds'])} "
                f"capped={scene['capped_instruction_count']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize annotation duration using each instruction's "
            "updated_at - created_at time, capped at 10 minutes for easy "
            "and 15 minutes for hard."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=INSTRUCTION_ROOT,
        help="Instruction root to scan. Defaults to resources/instructions.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help=(
            "Scene key to include, e.g. procthor/009_valunseen. Can be passed "
            "multiple times. Defaults to all scenes."
        ),
    )
    parser.add_argument(
        "--set",
        dest="instruction_set",
        choices=INSTRUCTION_SET_CHOICES,
        default="all",
        help="Instruction split to include. Defaults to all.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-scene text summary details.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    scene_filters = {scene.strip().strip("/") for scene in args.scene if scene.strip()}
    instruction_set = normalize_instruction_set(args.instruction_set)
    paths = iter_instruction_files(root, scene_filters, instruction_set)
    if scene_filters and not paths:
        print(f"no matching instructions under {display_path(root)}", file=sys.stderr)
        return 1

    records = [load_record(path, root) for path in paths]
    summary = summarize(records)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print_text_summary(summary, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
