#!/usr/bin/env python3
"""Move legacy flat ProcTHOR map directories under train/valunseen folders."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAP_ROOT = REPO_ROOT / "resources" / "maps" / "procthor"


def map_split(directory_name: str) -> str | None:
    if directory_name.endswith("_train"):
        return "train"
    if directory_name.endswith("_valunseen"):
        return "valunseen"
    return None


def planned_moves(map_root: Path) -> list[tuple[Path, Path]]:
    """Return safe whole-directory moves from the former flat layout."""

    moves: list[tuple[Path, Path]] = []
    for source in sorted(map_root.iterdir() if map_root.exists() else []):
        if not source.is_dir() or source.name in {"train", "valunseen"}:
            continue
        split = map_split(source.name)
        if split is None:
            continue
        primary_map = source / f"{source.name}.json"
        if primary_map.exists():
            moves.append((source, map_root / split / source.name))
    return moves


def migrate_map_directories(map_root: Path, *, dry_run: bool = False) -> list[tuple[Path, Path]]:
    moves = planned_moves(map_root)
    conflicts = [target for _source, target in moves if target.exists()]
    if conflicts:
        rendered = "\n".join(str(path) for path in conflicts)
        raise FileExistsError(f"Refusing to merge map directories into existing targets:\n{rendered}")
    if dry_run:
        return moves
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    return moves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    moves = migrate_map_directories(args.map_root.resolve(), dry_run=args.dry_run)
    action = "would move" if args.dry_run else "moved"
    for source, target in moves:
        print(f"{action}: {source} -> {target}")
    print(f"{action}_count={len(moves)}")


if __name__ == "__main__":
    main()
