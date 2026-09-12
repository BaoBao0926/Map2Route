#!/usr/bin/env python3
"""Build reusable map- and instruction-level evaluator caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.annotation.path_shape_schema import PATH_SHAPE_ANNOTATION_FIELD
from scripts.evaluation.evaluation_metrics import build_instruction_metric_cache
from scripts.evaluation.metric_cache import build_map_metric_cache
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state
from scripts.methods.util.instructions import iter_instruction_files, map_id_from_instruction_path


DEFAULT_MAP_ROOT = REPO_ROOT / "resources" / "maps"
DEFAULT_INSTRUCTION_ROOT = REPO_ROOT / "resources" / "instructions"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(REPO_ROOT).as_posix() if resolved.is_relative_to(REPO_ROOT) else str(path)


def map_id_from_map_path(path: Path) -> str:
    """Return the evaluator map id for standard ``resources/maps`` layout."""

    relative = path.resolve().relative_to(DEFAULT_MAP_ROOT.resolve())
    parts = relative.parts
    if len(parts) >= 4 and parts[-3] in {"train", "valunseen"}:
        return "/".join((*parts[:-3], parts[-2]))
    return "/".join(parts[:-1])


def iter_map_files(map_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(map_root.rglob("*.json"))
        if not any(part.endswith("_metric_cache") for part in path.relative_to(map_root).parts)
        and path.name != "template_instruction.json"
        and not path.name.endswith("_metadata.json")
        and not path.name.endswith("_thinggraph.json")
    ]


def has_embedded_path_shape_annotation(path: Path) -> bool:
    """Return whether an instruction contains a canonical path-shape annotation."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and any(
        isinstance(constraint, dict)
        and constraint.get(PATH_SHAPE_ANNOTATION_FIELD) is not None
        for constraint in payload.get("soft_constraints", [])
    )


def build_instruction_caches_for_map(
    map_id: str,
    paths: list[Path],
    *,
    overwrite: bool,
) -> tuple[str, int]:
    """Build one map's instruction caches in an isolated worker process."""

    map_state = load_map_state(map_id)
    build_map_metric_cache(Path(str(map_state["map_path"])), map_state, overwrite=False)
    for instruction_path in paths:
        payload = json.loads(instruction_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Instruction must be a JSON object: {instruction_path}")
        build_instruction_metric_cache(instruction_path, payload, map_state)
    return map_id, len(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-id", action="append", default=[], help="Build a map cache by map id.")
    parser.add_argument("--instruction-file", type=Path, action="append", default=[])
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--instruction-root", type=Path, default=DEFAULT_INSTRUCTION_ROOT)
    parser.add_argument("--all-maps", action="store_true")
    parser.add_argument("--all-instructions", action="store_true")
    parser.add_argument(
        "--path-shape-instructions",
        action="store_true",
        help="Build caches only for instructions with embedded path-shape annotations.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Print one progress line per map.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel map workers for instruction cache generation (default: up to 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (
        args.map_id
        or args.instruction_file
        or args.all_maps
        or args.all_instructions
        or args.path_shape_instructions
    ):
        raise SystemExit(
            "Specify --map-id, --instruction-file, --all-maps, "
            "--all-instructions, or --path-shape-instructions."
        )
    if args.workers <= 0:
        raise SystemExit("--workers must be positive.")

    map_ids = list(dict.fromkeys(args.map_id))
    if args.all_maps:
        for path in iter_map_files(args.map_root.resolve()):
            try:
                map_ids.append(map_id_from_map_path(path))
            except ValueError:
                continue
    map_ids = list(dict.fromkeys(map_ids))
    for map_id in map_ids:
        map_state = load_map_state(map_id)
        map_path = Path(str(map_state["map_path"]))
        result = build_map_metric_cache(map_path, map_state, overwrite=args.overwrite)
        print(f"[map] {map_id}: {result['status']} {display_path(Path(str(result['path'])))}")

    instruction_files = list(args.instruction_file)
    if args.all_instructions:
        instruction_files.extend(iter_instruction_files(args.instruction_root))
    if args.path_shape_instructions:
        instruction_files.extend(
            path
            for path in iter_instruction_files(args.instruction_root)
            if has_embedded_path_shape_annotation(path)
        )
    grouped_instructions: dict[str, list[Path]] = defaultdict(list)
    for instruction_path in sorted(dict.fromkeys(path.resolve() for path in instruction_files)):
        map_id = map_id_from_instruction_path(instruction_path, args.instruction_root)
        grouped_instructions[map_id].append(instruction_path)

    total_instructions = sum(len(paths) for paths in grouped_instructions.values())
    completed = 0
    groups = sorted(grouped_instructions.items())
    if args.workers == 1:
        completed_groups = [
            build_instruction_caches_for_map(map_id, paths, overwrite=args.overwrite)
            for map_id, paths in groups
        ]
    else:
        completed_groups = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    build_instruction_caches_for_map,
                    map_id,
                    paths,
                    overwrite=args.overwrite,
                ): map_id
                for map_id, paths in groups
            }
            for future in as_completed(futures):
                completed_groups.append(future.result())
    for map_id, count in completed_groups:
        completed += count
        if args.quiet:
            print(
                f"[map] {map_id}: instruction_caches={count} "
                f"progress={completed}/{total_instructions}",
                flush=True,
            )


if __name__ == "__main__":
    main()
