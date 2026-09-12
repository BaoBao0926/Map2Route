#!/usr/bin/env python3
"""Recompute human-solver metrics and summary with the current evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluate_prediction import evaluate_prediction
from scripts.methods.human_solver.run import (
    DIFFICULTY_LEVELS,
    METHOD_NAME,
    METHOD_ROOT,
    SUMMARY_PATH,
    difficulty_from_solution_payload,
    display_path,
    metric_summary,
    runtime_summary,
    score_block,
)


DEFAULT_SOLUTION_ROOT = METHOD_ROOT / "procthor"


def load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def iter_solution_files(solution_root: Path) -> list[Path]:
    if not solution_root.exists():
        return []
    return sorted(
        path
        for path in solution_root.rglob("*.json")
        if "_archived" not in path.relative_to(solution_root).parts
    )


def evaluate_solution_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    trajectory = payload.get("trajectory")
    map_id = payload.get("map_id")
    instruction_id = payload.get("instruction_id")
    if not isinstance(trajectory, list):
        raise ValueError("solution has no trajectory list.")
    if not isinstance(map_id, str) or not map_id.strip():
        raise ValueError("solution has no map_id.")
    if not isinstance(instruction_id, str) or not instruction_id.strip():
        raise ValueError("solution has no instruction_id.")
    return evaluate_prediction(
        trajectory,
        map_id=map_id,
        instruction_id=instruction_id,
    )


def record_for_summary(
    payload: Mapping[str, object],
    path: Path,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    trajectory = payload.get("trajectory")
    scene_id = payload.get("scene_id", payload.get("map_id"))
    if not isinstance(scene_id, str) or not scene_id.strip():
        scene_id = str(payload.get("map_id") or "")
    return {
        "solution_id": payload.get("solution_id"),
        "map_id": payload.get("map_id"),
        "scene_id": scene_id,
        "instruction_id": payload.get("instruction_id"),
        "difficulty_level": difficulty_from_solution_payload(payload),
        "created_at": payload.get("created_at"),
        "created_by": payload.get("created_by"),
        "path": display_path(path),
        "trajectory_image": payload.get("trajectory_image"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "runtime": payload.get("runtime"),
        "trajectory_length": len(trajectory) if isinstance(trajectory, list) else 0,
        "metrics": dict(metrics),
        "metrics_summary": metric_summary(metrics),
    }


def rebuild_summary(records: list[dict[str, object]]) -> dict[str, object]:
    by_difficulty = {
        difficulty: [
            record
            for record in records
            if str(record.get("difficulty_level") or "unknown").lower() == difficulty
        ]
        for difficulty in DIFFICULTY_LEVELS
    }
    return {
        "version": 1,
        "method": METHOD_NAME,
        "runtime_summary": runtime_summary(records),
        "overall": score_block(records),
        "by_difficulty": {
            difficulty: score_block(difficulty_records)
            for difficulty, difficulty_records in by_difficulty.items()
        },
    }


def update_human_solver_summary(
    *,
    solution_root: Path = DEFAULT_SOLUTION_ROOT,
    summary_path: Path = SUMMARY_PATH,
    dry_run: bool = False,
) -> tuple[dict[str, object], dict[str, int]]:
    records: list[dict[str, object]] = []
    stats = {
        "seen": 0,
        "updated_records": 0,
        "unchanged_records": 0,
        "skipped_records": 0,
        "failed_records": 0,
    }

    for path in iter_solution_files(solution_root):
        stats["seen"] += 1
        try:
            payload = load_json_object(path)
            metrics = evaluate_solution_payload(payload)
        except Exception as exc:
            stats["failed_records"] += 1
            print(f"[failed] {display_path(path)}: {exc}", file=sys.stderr)
            continue

        metrics_summary = metric_summary(metrics)
        if (
            payload.get("metrics") != metrics
            or payload.get("metrics_summary") != metrics_summary
        ):
            stats["updated_records"] += 1
            if not dry_run:
                payload["metrics"] = metrics
                payload["metrics_summary"] = metrics_summary
                path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        else:
            stats["unchanged_records"] += 1

        records.append(record_for_summary(payload, path, metrics))

    summary = rebuild_summary(records)
    if not dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute resources/methods/baselines/human_solver summary from saved "
            "human-solver trajectories using the current evaluator."
        )
    )
    parser.add_argument(
        "--solution-root",
        type=Path,
        default=DEFAULT_SOLUTION_ROOT,
        help="Root containing saved human-solver JSON files.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_PATH,
        help="Summary JSON path to write.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Recompute and print stats without writing records or summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, stats = update_human_solver_summary(
        solution_root=args.solution_root,
        summary_path=args.summary_path,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "summary_path": display_path(args.summary_path),
                "solution_root": display_path(args.solution_root),
                "stats": stats,
                "overall": summary["overall"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
