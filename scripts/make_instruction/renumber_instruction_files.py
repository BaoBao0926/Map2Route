#!/usr/bin/env python3
"""Renumber instruction files within each scene by difficulty and annotation time.

For every ``instruction_files`` directory, this script orders samples by
difficulty first, then by annotation timestamp inside each difficulty group.
By default the order is ``easy`` followed by ``hard``. File names and the JSON
``id`` field are rewritten to match the new 1-based order.

The default mode rewrites files. Pass ``--dry-run`` to preview the plan without
changing file names or JSON ids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION_ROOT = REPO_ROOT / "resources" / "instructions"
DEFAULT_DIFFICULTY_ORDER = ("easy", "hard")


@dataclass(frozen=True)
class InstructionRecord:
    path: Path
    payload: dict[str, Any]
    difficulty: str
    timestamp: datetime
    original_index: int
    old_id: int | None


@dataclass(frozen=True)
class RenumberAction:
    record: InstructionRecord
    new_id: int
    new_path: Path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timestamp_for(path: Path, payload: dict[str, Any]) -> datetime:
    return (
        parse_timestamp(payload.get("created_at"))
        or parse_timestamp(payload.get("updated_at"))
        or datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    )


def old_id_for(path: Path, payload: dict[str, Any]) -> int | None:
    raw_id = payload.get("id")
    if isinstance(raw_id, int) and not isinstance(raw_id, bool):
        return raw_id
    stem = path.stem
    prefix = "instruction_"
    if stem.startswith(prefix):
        suffix = stem[len(prefix) :]
        if suffix.isdigit():
            return int(suffix)
    return None


def load_instruction(path: Path, original_index: int) -> InstructionRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Instruction file must contain a JSON object: {path}")
    difficulty = str(payload.get("difficulty_level") or "").strip().lower()
    return InstructionRecord(
        path=path,
        payload=payload,
        difficulty=difficulty,
        timestamp=timestamp_for(path, payload),
        original_index=original_index,
        old_id=old_id_for(path, payload),
    )


def scene_key_for_directory(instruction_dir: Path, root: Path) -> str:
    relative = instruction_dir.relative_to(root)
    parts = relative.parts
    if parts and parts[-1] == "instruction_files":
        parts = parts[:-1]
    return "/".join(parts)


def iter_instruction_dirs(root: Path, scene_filters: set[str]) -> list[Path]:
    candidates = sorted(path for path in root.rglob("instruction_files") if path.is_dir())
    if not scene_filters:
        return candidates
    selected: list[Path] = []
    for path in candidates:
        scene_key = scene_key_for_directory(path, root)
        if scene_key in scene_filters or path.parent.name in scene_filters:
            selected.append(path)
    return selected


def build_actions(
    instruction_dir: Path,
    *,
    difficulty_order: tuple[str, ...],
    include_unknown: bool,
) -> tuple[list[RenumberAction], list[str]]:
    paths = sorted(instruction_dir.glob("instruction_*.json"))
    warnings: list[str] = []
    if not paths:
        return [], warnings

    records = [
        load_instruction(path, original_index)
        for original_index, path in enumerate(paths)
    ]

    difficulty_rank = {difficulty: index for index, difficulty in enumerate(difficulty_order)}
    unknown_records = [
        record
        for record in records
        if record.difficulty not in difficulty_rank
    ]
    if unknown_records and not include_unknown:
        warnings.append(
            "skipping scene because some files have unknown difficulty_level: "
            + ", ".join(
                f"{record.path.name}={record.difficulty or '<missing>'}"
                for record in unknown_records
            )
        )
        return [], warnings

    def sort_key(record: InstructionRecord) -> tuple[int, datetime, int | float, str]:
        rank = difficulty_rank.get(record.difficulty, len(difficulty_rank))
        old_id = record.old_id if record.old_id is not None else float("inf")
        return (rank, record.timestamp, old_id, record.path.name)

    ordered = sorted(records, key=sort_key)
    actions = [
        RenumberAction(
            record=record,
            new_id=new_id,
            new_path=instruction_dir / f"instruction_{new_id:06d}.json",
        )
        for new_id, record in enumerate(ordered, start=1)
    ]
    return actions, warnings


def action_changes(action: RenumberAction) -> bool:
    return action.record.path != action.new_path or action.record.old_id != action.new_id


def print_plan(scene_key: str, actions: list[RenumberAction], warnings: list[str]) -> None:
    for warning in warnings:
        print(f"[warn] {scene_key}: {warning}")
    if not actions:
        return

    changed = [action for action in actions if action_changes(action)]
    print(
        f"[scene] {scene_key}: files={len(actions)} changed={len(changed)} "
        f"easy={sum(1 for a in actions if a.record.difficulty == 'easy')} "
        f"hard={sum(1 for a in actions if a.record.difficulty == 'hard')}"
    )
    for action in actions:
        if not action_changes(action):
            continue
        timestamp = action.record.timestamp.isoformat().replace("+00:00", "Z")
        print(
            "  "
            f"{action.record.path.name} id={action.record.old_id} "
            f"{action.record.difficulty or '<missing>'} {timestamp}"
            f" -> {action.new_path.name} id={action.new_id}"
        )


def apply_actions(actions: list[RenumberAction]) -> None:
    if not actions:
        return

    temp_suffix = f".renumber-tmp-{uuid.uuid4().hex}"
    temp_paths: list[tuple[Path, Path]] = []
    try:
        for action in actions:
            temp_path = action.record.path.with_name(action.record.path.name + temp_suffix)
            os.replace(action.record.path, temp_path)
            temp_paths.append((temp_path, action.record.path))

        for action, (temp_path, _original_path) in zip(actions, temp_paths, strict=True):
            payload = dict(action.record.payload)
            payload["id"] = action.new_id
            action.new_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temp_path.unlink()
    except Exception:
        for temp_path, original_path in temp_paths:
            if temp_path.exists() and not original_path.exists():
                os.replace(temp_path, original_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Renumber each scene's instruction_*.json files so easy samples come "
            "first, then hard samples, with each group ordered by created_at."
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
            "Scene key to process, e.g. procthor/009_valunseen. Can be passed "
            "multiple times. Defaults to all scenes under --root."
        ),
    )
    parser.add_argument(
        "--difficulty-order",
        default="easy,hard",
        help="Comma-separated difficulty order. Defaults to easy,hard.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Place unknown difficulty levels after the configured order instead of skipping the scene.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned changes; do not rewrite file names or JSON id fields.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Deprecated no-op kept for compatibility. The script applies changes by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    difficulty_order = tuple(
        item.strip().lower()
        for item in args.difficulty_order.split(",")
        if item.strip()
    )
    if not difficulty_order:
        print("difficulty order must not be empty", file=sys.stderr)
        return 2

    scene_filters = {scene.strip().strip("/") for scene in args.scene if scene.strip()}
    instruction_dirs = iter_instruction_dirs(root, scene_filters)
    if scene_filters and not instruction_dirs:
        print(f"no matching scenes under {display_path(root)}: {sorted(scene_filters)}", file=sys.stderr)
        return 1

    all_actions: list[RenumberAction] = []
    skipped = 0
    for instruction_dir in instruction_dirs:
        scene_key = scene_key_for_directory(instruction_dir, root)
        actions, warnings = build_actions(
            instruction_dir,
            difficulty_order=difficulty_order,
            include_unknown=args.include_unknown,
        )
        if warnings and not actions:
            skipped += 1
        print_plan(scene_key, actions, warnings)
        all_actions.extend(actions)

    changed_actions = [action for action in all_actions if action_changes(action)]
    if args.dry_run:
        print(
            f"[dry-run] scenes={len(instruction_dirs)} skipped={skipped} "
            f"files={len(all_actions)} changed={len(changed_actions)}"
        )
        print("Run without --dry-run to rewrite file names and JSON id fields.")
        return 0

    by_dir: dict[Path, list[RenumberAction]] = {}
    for action in all_actions:
        by_dir.setdefault(action.record.path.parent, []).append(action)

    for actions in by_dir.values():
        apply_actions(actions)

    print(
        f"[applied] scenes={len(by_dir)} skipped={skipped} "
        f"files={len(all_actions)} changed={len(changed_actions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
