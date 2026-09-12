#!/usr/bin/env python3
"""Evaluate the human expert trajectories stored in instruction annotations.

This is intentionally separate from ``update_summary.py``.  The latter
recomputes trajectories submitted through the Human Solver web UI, whereas
this script evaluates the ``human_expert_trajectory`` embedded in every
instruction JSON produced by ``make_instruction``.
"""

from __future__ import annotations

import argparse
from threading import Lock
import json
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluate_prediction import evaluate_prediction
from scripts.evaluation.evaluation_metrics import (
    worst_case_metrics_for_evaluation_error,
)
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state
from scripts.methods.tutorial.utils import (
    difficulty_from_instruction,
    display_path,
    print_compact_metric,
    score_block,
)
from scripts.methods.util.instructions import (
    DEFAULT_INSTRUCTION_SET,
    INSTRUCTION_ROOT,
    INSTRUCTION_SET_CHOICES,
    instruction_id_from_payload,
    iter_instruction_files,
    load_instruction,
    map_id_from_instruction_path,
    normalize_instruction_set,
)
from scripts.methods.util.workers import run_instruction_jobs_grouped


METHOD_NAME = "human_solver"
TRAJECTORY_FIELD = "human_expert_trajectory"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "resources" / "methods" / "baselines" / METHOD_NAME / "human_expert_summary.json"
)


def evaluate_instruction(
    instruction_path: Path,
    instruction: Mapping[str, object],
    *,
    worst_case_on_metric_error: bool = False,
) -> dict[str, object]:
    """Evaluate one annotated human reference trajectory."""
    trajectory = instruction.get(TRAJECTORY_FIELD)
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"instruction has no non-empty {TRAJECTORY_FIELD}.")

    map_id = instruction.get("map_id")
    if not isinstance(map_id, str) or not map_id.strip():
        map_id = map_id_from_instruction_path(instruction_path)
    instruction_id = instruction_id_from_payload(instruction_path, instruction)
    fallback_error: str | None = None
    try:
        metrics = evaluate_prediction(
            trajectory,
            map_id=map_id,
            instruction_id=instruction_id,
            instruction_file=instruction_path,
        )
    except Exception as metric_error:
        if not worst_case_on_metric_error:
            raise
        metrics = worst_case_metrics_for_evaluation_error(
            trajectory,
            instruction,
            load_map_state(map_id),
            error=metric_error,
        )
        fallback_error = f"{type(metric_error).__name__}: {metric_error}"
    record: dict[str, object] = {
        "map_id": map_id,
        "instruction_id": instruction_id,
        "difficulty_level": difficulty_from_instruction(instruction),
        "instruction_file": display_path(instruction_path),
        "trajectory_length": len(trajectory),
        "metrics": metrics,
    }
    if fallback_error is not None:
        record["metric_fallback"] = {
            "policy": "worst_case",
            "error": fallback_error,
        }
    return record


def build_summary(
    records: list[dict[str, object]],
    *,
    instruction_set: str,
    elapsed_seconds: float,
    source_count: int,
    skipped_count: int,
    failed_count: int,
    fallback_count: int = 0,
    completed: bool = True,
) -> dict[str, object]:
    """Build the same overall/easy/hard metric layout used by methods."""
    by_difficulty = {
        difficulty: [
            record
            for record in records
            if record.get("difficulty_level") == difficulty
        ]
        for difficulty in ("easy", "hard")
    }
    return {
        "version": 1,
        "method": METHOD_NAME,
        "trajectory_source": f"instruction.{TRAJECTORY_FIELD}",
        "status": "completed" if completed else "in_progress",
        "instruction_set": instruction_set,
        "overall": score_block(records),
        "easy": score_block(by_difficulty["easy"]),
        "hard": score_block(by_difficulty["hard"]),
        "runtime_summary": {
            "total_evaluation_seconds": elapsed_seconds,
            "average_evaluation_seconds": (
                elapsed_seconds / len(records) if records else None
            ),
        },
        "source_instruction_count": source_count,
        "skipped_instruction_count": skipped_count,
        "failed_instruction_count": failed_count,
        "fallback_instruction_count": fallback_count,
        "records": records,
    }


def evaluate_human_experts(
    *,
    input_root: Path = INSTRUCTION_ROOT,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
    limit: int | None = None,
    offset: int = 0,
    fail_fast: bool = False,
    quiet: bool = False,
    workers: int = 1,
    checkpoint_every: int = 10,
    worst_case_on_metric_error: bool = False,
    checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
    cache_loader: Callable[
        [Path, Mapping[str, object]], Mapping[str, object] | None
    ]
    | None = None,
    cache_writer: Callable[
        [Path, Mapping[str, object], Mapping[str, object]], None
    ]
    | None = None,
) -> tuple[dict[str, object], dict[str, int]]:
    """Evaluate annotated trajectories and return their aggregate summary."""

    selected_set = normalize_instruction_set(instruction_set)
    instruction_paths = iter_instruction_files(input_root, instruction_set=selected_set)
    if offset:
        instruction_paths = instruction_paths[offset:]
    if limit is not None:
        instruction_paths = instruction_paths[:limit]

    records: list[dict[str, object]] = []
    stats = {
        "seen_instructions": len(instruction_paths),
        "evaluated_instructions": 0,
        "computed_instructions": 0,
        "resumed_instructions": 0,
        "fallback_instructions": 0,
        "skipped_instructions": 0,
        "failed_instructions": 0,
    }
    # A metric evaluation temporarily materializes map-sized arrays. Serialise
    # work for one map so multiple workers do not load multiple large copies;
    # different maps still run concurrently.
    map_locks: dict[str, Lock] = {}
    map_locks_guard = Lock()

    def lock_for_instruction(instruction_path: Path) -> Lock:
        map_id = map_id_from_instruction_path(instruction_path, input_root)
        with map_locks_guard:
            return map_locks.setdefault(map_id, Lock())

    started_at = time.perf_counter()

    def write_checkpoint_if_due() -> None:
        completed_count = (
            stats["evaluated_instructions"]
            + stats["skipped_instructions"]
            + stats["failed_instructions"]
        )
        if (
            checkpoint_callback is None
            or checkpoint_every <= 0
            or completed_count == 0
            or completed_count % checkpoint_every != 0
        ):
            return
        checkpoint_callback(
            build_summary(
                records,
                instruction_set=selected_set,
                elapsed_seconds=time.perf_counter() - started_at,
                source_count=len(instruction_paths),
                skipped_count=stats["skipped_instructions"],
                failed_count=stats["failed_instructions"],
                fallback_count=stats["fallback_instructions"],
                completed=False,
            )
        )

    def run_one(index: int, instruction_path: Path) -> dict[str, object] | None:
        try:
            instruction = load_instruction(instruction_path)
            if not instruction.get(TRAJECTORY_FIELD):
                return None
            if cache_loader is not None:
                cached = cache_loader(instruction_path, instruction)
                if cached is not None:
                    return {
                        "_metric_record": dict(cached),
                        "_metric_cache_hit": True,
                    }
            with lock_for_instruction(instruction_path):
                record = evaluate_instruction(
                    instruction_path,
                    instruction,
                    worst_case_on_metric_error=(
                        worst_case_on_metric_error and not fail_fast
                    ),
                )
            if cache_writer is not None:
                cache_writer(instruction_path, instruction, record)
            return {
                "_metric_record": record,
                "_metric_cache_hit": False,
            }
        except Exception as exc:
            print(
                f"[{index}/{len(instruction_paths)}] failed "
                f"{display_path(instruction_path)}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if fail_fast:
                raise
            return {
                "_evaluation_error": str(exc),
                "instruction_file": display_path(instruction_path),
            }

    for index, envelope in run_instruction_jobs_grouped(
        instruction_paths,
        workers=workers,
        group_key=lambda path: map_id_from_instruction_path(path, input_root),
        run_one=run_one,
    ):
        if envelope is None:
            stats["skipped_instructions"] += 1
            if not quiet:
                print(
                    f"[{index}/{len(instruction_paths)}] skipped: missing "
                    f"{TRAJECTORY_FIELD}",
                    flush=True,
                )
            write_checkpoint_if_due()
            continue
        if "_evaluation_error" in envelope:
            stats["failed_instructions"] += 1
            write_checkpoint_if_due()
            continue
        raw_record = envelope.get("_metric_record")
        if not isinstance(raw_record, Mapping):
            stats["failed_instructions"] += 1
            write_checkpoint_if_due()
            continue

        record = dict(raw_record)
        cache_hit = envelope.get("_metric_cache_hit") is True
        records.append(record)
        stats["evaluated_instructions"] += 1
        if isinstance(record.get("metric_fallback"), Mapping):
            stats["fallback_instructions"] += 1
        if cache_hit:
            stats["resumed_instructions"] += 1
        else:
            stats["computed_instructions"] += 1
        if not quiet:
            metric = score_block([record])["metric"]
            assert isinstance(metric, Mapping)
            status = (
                " skipped reason=resume_cache_match"
                if cache_hit
                else ""
            )
            print_compact_metric(
                f"[{index}/{len(instruction_paths)}] "
                f"{record['map_id']}/{record['instruction_id']}{status}",
                metric,
            )
        write_checkpoint_if_due()

    elapsed_seconds = time.perf_counter() - started_at
    return (
        build_summary(
            records,
            instruction_set=selected_set,
            elapsed_seconds=elapsed_seconds,
            source_count=len(instruction_paths),
            skipped_count=stats["skipped_instructions"],
            failed_count=stats["failed_instructions"],
            fallback_count=stats["fallback_instructions"],
        ),
        stats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate make_instruction human_expert_trajectory annotations using "
            "the shared SemPathBench metric."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INSTRUCTION_ROOT,
        help="Instruction root. Defaults to resources/instructions.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write an in-progress summary after every N processed instructions (default: 10).",
    )
    parser.add_argument(
        "--set",
        dest="instruction_set",
        choices=INSTRUCTION_SET_CHOICES,
        default=DEFAULT_INSTRUCTION_SET,
        help="Instruction split to evaluate. Defaults to valunseen.",
    )
    parser.add_argument("--limit", type=int, help="Evaluate at most N instructions.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N instructions.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent instruction evaluations to run concurrently.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Summary JSON to write. Defaults to "
            "resources/methods/baselines/human_solver/human_expert_summary.json."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and print results without writing the summary JSON.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first unreadable or invalid instruction.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-instruction metric lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    if args.offset < 0:
        raise ValueError("--offset must be non-negative.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative.")

    def write_checkpoint(payload: dict[str, object]) -> None:
        if args.dry_run:
            return
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary, stats = evaluate_human_experts(
        input_root=args.input_root,
        instruction_set=args.instruction_set,
        limit=args.limit,
        offset=args.offset,
        fail_fast=args.fail_fast,
        quiet=args.quiet,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=None if args.dry_run else write_checkpoint,
    )
    if not args.dry_run:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for scope in ("overall", "easy", "hard"):
        block = summary[scope]
        assert isinstance(block, Mapping)
        metric = block["metric"]
        assert isinstance(metric, Mapping)
        print_compact_metric(
            f"{scope} n={block['record_count']}",
            metric,
        )
    print(
        json.dumps(
            {
                "output_path": None if args.dry_run else display_path(args.output_path),
                "stats": stats,
                "trajectory_source": summary["trajectory_source"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
