"""Instruction file helpers shared by method implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from scripts.make_instruction.make_instruction import REPO_ROOT


INSTRUCTION_ROOT = REPO_ROOT / "resources" / "instructions"
DEFAULT_INSTRUCTION_SET = "valunseen"
INSTRUCTION_SET_CHOICES = ("valunseen", "train", "all")


def instruction_id_from_payload(path: Path, payload: Mapping[str, object]) -> str:
    raw_id = payload.get("id")
    if isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id > 0:
        return f"instruction_{raw_id:06d}"
    return path.stem


def load_instruction(path: Path | str) -> dict[str, object]:
    instruction_path = Path(path)
    payload = json.loads(instruction_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Instruction file must contain a JSON object: {instruction_path}")
    return payload


def map_id_from_instruction_path(path: Path, root: Path = INSTRUCTION_ROOT) -> str:
    relative = path.relative_to(root)
    parts = list(relative.parts)
    if "instruction_files" in parts:
        return "/".join(parts[: parts.index("instruction_files")])
    return "/".join(parts[:-1])


def normalize_instruction_set(value: str | None) -> str:
    if value is None or not value.strip():
        return DEFAULT_INSTRUCTION_SET
    normalized = value.strip().lower().replace("-", "").replace("_", "")
    if normalized == "valunseen":
        return "valunseen"
    if normalized == "train":
        return "train"
    if normalized == "all":
        return "all"
    raise ValueError(
        f"Unsupported instruction set {value!r}; expected one of "
        f"{', '.join(INSTRUCTION_SET_CHOICES)}."
    )


def instruction_set_from_map_id(map_id: str) -> str:
    leaf = map_id.replace("\\", "/").strip("/").rsplit("/", 1)[-1].lower()
    if leaf.endswith("_valunseen") or leaf.startswith("valunseen_"):
        return "valunseen"
    if leaf.endswith("_train") or leaf.startswith("train_"):
        return "train"
    return ""


def map_id_matches_instruction_set(map_id: str, instruction_set: str | None) -> bool:
    selected = normalize_instruction_set(instruction_set)
    return selected == "all" or instruction_set_from_map_id(map_id) == selected


def iter_instruction_files(
    input_root: Path = INSTRUCTION_ROOT,
    *,
    instruction_set: str | None = DEFAULT_INSTRUCTION_SET,
) -> list[Path]:
    selected = normalize_instruction_set(instruction_set)
    paths = sorted(input_root.rglob("instruction_*.json"))
    if selected == "all":
        return paths
    return [
        path
        for path in paths
        if map_id_matches_instruction_set(
            map_id_from_instruction_path(path, input_root),
            selected,
        )
    ]
