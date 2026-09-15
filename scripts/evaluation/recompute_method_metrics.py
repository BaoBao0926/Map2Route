#!/usr/bin/env python3
"""Recompute saved method metrics from existing trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluate_prediction import (
    evaluate_prediction,
    load_instruction,
    load_instruction_by_id,
)
from scripts.evaluation.evaluation_metrics import (
    worst_case_metrics_for_evaluation_error,
)
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state
from scripts.methods.human_solver.evaluate_human_expert import evaluate_human_experts
from scripts.methods.tutorial.utils import (
    DIFFICULTY_LEVELS,
    display_path,
    metric_summary,
    record_with_trajectory_last,
    runtime_summary,
    score_block,
)
from scripts.methods.util.instructions import (
    instruction_id_from_payload,
    map_id_from_instruction_path,
)
from scripts.methods.util.methods import save_trajectory_image_for_record


DEFAULT_METHOD_ROOT = REPO_ROOT / "resources" / "methods"
AGGREGATE_SECTIONS = ("overall", "easy", "hard")
ARCHIVED_RUN_DIRECTORIES = {"first_run", "2nd_run"}
HUMAN_EXPERT_METHOD_NAME = "human_solver"
DEFAULT_WORKERS = 1
DEFAULT_CHECKPOINT_EVERY = 10
CHECKPOINT_VERSION = 1
EVALUATOR_SOURCE_PATHS = (
    REPO_ROOT / "scripts" / "evaluation" / "evaluate_prediction.py",
    REPO_ROOT / "scripts" / "evaluation" / "evaluation_metrics.py",
    REPO_ROOT / "scripts" / "evaluation" / "metric_geometry.py",
    REPO_ROOT / "scripts" / "evaluation" / "metric_cache.py",
    REPO_ROOT / "scripts" / "evaluation" / "hyparameter.py",
    REPO_ROOT / "scripts" / "methods" / "util" / "grid_astar.py",
    REPO_ROOT / "scripts" / "methods" / "tutorial" / "utils.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluator_fingerprint() -> str:
    """Return a stable id for the code that defines metrics and aggregation."""

    digest = hashlib.sha256()
    for path in EVALUATOR_SOURCE_PATHS:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"evaluator_{digest.hexdigest()[:16]}"


def safe_run_label(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value.strip()
    ).strip("._")
    if not cleaned:
        raise ValueError("--run-id must contain at least one letter or number.")
    return cleaned


def json_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_fingerprint(
    trajectory: Sequence[object],
    *,
    instruction_path: Path,
    instruction: Mapping[str, object],
    map_id: str,
    evaluator_id: str,
) -> str:
    """Hash every episode-level input that can change a saved metric."""

    digest = hashlib.sha256()
    digest.update(evaluator_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(map_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            trajectory,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(instruction_path.read_bytes())

    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {display_path(path)}")
    return payload


def write_json_object(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace one JSON object without exposing a partial file."""
    output_payload = record_with_trajectory_last(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def is_skipped_directory(part: str) -> bool:
    return part.startswith("_") or part == "__pycache__"


def iter_prediction_files(method_root: Path) -> list[Path]:
    if not method_root.exists():
        return []
    paths: list[Path] = []
    for path in method_root.rglob("*.json"):
        relative_parts = path.relative_to(method_root).parts
        if path.name == "summary.json":
            continue
        if "_archived" in relative_parts or any(part in ARCHIVED_RUN_DIRECTORIES for part in relative_parts):
            continue
        if any(is_skipped_directory(part) for part in relative_parts):
            continue
        paths.append(path)
    return sorted(paths)


def has_prediction_fields(payload: Mapping[str, object]) -> bool:
    return (
        isinstance(payload.get("trajectory"), list)
        and isinstance(payload.get("map_id"), str)
        and isinstance(payload.get("instruction_id"), str)
    )


def embedded_template_instruction_id(payload: Mapping[str, object]) -> str | None:
    instruction = payload.get("instruction")
    if not isinstance(instruction, Mapping):
        return None
    template_id = instruction.get("template_instruction_id")
    if isinstance(template_id, str) and template_id.strip():
        return template_id.strip()
    return None


def instruction_template_id(path: Path) -> str | None:
    try:
        instruction = load_instruction(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    template_id = instruction.get("template_instruction_id")
    if isinstance(template_id, str) and template_id.strip():
        return template_id.strip()
    return None


def find_instruction_file_by_template_id(map_id: str, template_id: str) -> Path | None:
    instruction_dir = (
        REPO_ROOT
        / "resources"
        / "instructions"
        / map_id.strip().strip("/")
        / "instruction_files"
    )
    matches: list[Path] = []
    for path in sorted(instruction_dir.glob("instruction_*.json")):
        if instruction_template_id(path) == template_id:
            matches.append(path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple instructions in {display_path(instruction_dir)} "
            f"match template_instruction_id={template_id!r}."
        )
    return None


def resolve_instruction_file_for_payload(payload: Mapping[str, object]) -> Path:
    map_id = payload.get("map_id")
    instruction_id = payload.get("instruction_id")
    if not isinstance(map_id, str) or not map_id.strip():
        raise ValueError("record has no map_id.")
    if not isinstance(instruction_id, str) or not instruction_id.strip():
        raise ValueError("record has no instruction_id.")

    template_id = embedded_template_instruction_id(payload)
    id_path: Path | None = None
    id_template: str | None = None
    try:
        _instruction, id_path = load_instruction_by_id(map_id, instruction_id)
        id_template = instruction_template_id(id_path)
    except ValueError:
        id_path = None

    if template_id:
        if id_path is not None and id_template == template_id:
            return id_path
        template_path = find_instruction_file_by_template_id(map_id, template_id)
        if template_path is not None:
            return template_path

    if id_path is not None:
        return id_path
    raise ValueError(f"Unknown instruction_id for {map_id}: {instruction_id!r}")


def evaluate_payload(payload: Mapping[str, object]) -> dict[str, object]:
    trajectory = payload.get("trajectory")
    map_id = payload.get("map_id")
    if not isinstance(trajectory, list):
        raise ValueError("record has no trajectory list.")
    if not isinstance(map_id, str) or not map_id.strip():
        raise ValueError("record has no map_id.")
    instruction_file = resolve_instruction_file_for_payload(payload)
    return evaluate_prediction(
        trajectory,
        map_id=map_id,
        instruction_file=instruction_file,
    )


def difficulty_from_payload(payload: Mapping[str, object]) -> str:
    difficulty = payload.get("difficulty_level")
    if isinstance(difficulty, str) and difficulty.strip():
        return difficulty.strip().lower()

    instruction = payload.get("instruction")
    if isinstance(instruction, Mapping):
        difficulty = instruction.get("difficulty_level")
        if isinstance(difficulty, str) and difficulty.strip():
            return difficulty.strip().lower()
    return "unknown"


def nested_image_record_keys(payload: Mapping[str, object]) -> tuple[str, ...]:
    method = payload.get("method")
    candidates: list[str] = []
    if isinstance(method, str) and method.strip():
        candidates.append(method.strip().lower())
    for key in ("grounding2route", "sayplan", "osgllm", "limp", "iln"):
        if key not in candidates:
            candidates.append(key)
    return tuple(key for key in candidates if isinstance(payload.get(key), Mapping))


def record_for_summary(
    payload: Mapping[str, object],
    path: Path,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    trajectory = payload.get("trajectory")
    scene_id = payload.get("scene_id", payload.get("map_id"))
    if not isinstance(scene_id, str) or not scene_id.strip():
        scene_id = str(payload.get("map_id") or "")

    record: dict[str, object] = {
        "map_id": payload.get("map_id"),
        "scene_id": scene_id,
        "instruction_id": payload.get("instruction_id"),
        "difficulty_level": difficulty_from_payload(payload),
        "path": display_path(path),
        "trajectory_length": len(trajectory) if isinstance(trajectory, list) else 0,
        "metrics": dict(metrics),
        "metrics_summary": metric_summary(metrics),
    }
    for key in (
        "method",
        "prediction_id",
        "solution_id",
        "created_at",
        "created_by",
        "runtime_seconds",
        "runtime",
        "trajectory_image",
        "trajectory_with_gt_image",
    ):
        if key in payload:
            record[key] = payload[key]
    return record


def method_name_from_summary(method_root: Path) -> str:
    summary_path = method_root / "summary.json"
    if summary_path.exists():
        try:
            payload = load_json_object(summary_path)
        except (OSError, json.JSONDecodeError, ValueError):
            payload = {}
        method = payload.get("method")
        if isinstance(method, str) and method.strip():
            return method.strip()
    return method_root.name


def rebuild_method_summary(
    records: list[dict[str, object]],
    *,
    method_name: str,
) -> dict[str, object]:
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
        "method": method_name,
        "runtime_summary": runtime_summary(records),
        "overall": score_block(records),
        **{
            difficulty: score_block(by_difficulty[difficulty])
            for difficulty in ("easy", "hard")
        },
    }


def print_progress(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def recompute_method(
    method_root: Path,
    *,
    method_name: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    fail_fast: bool = False,
    quiet: bool = False,
    progress_every: int = 1,
    rewrite_images: bool = False,
    evaluator_id: str,
    checkpoint_directory: Path,
    resume: bool = True,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> tuple[dict[str, object], dict[str, int]]:
    records: list[dict[str, object]] = []
    stats = {
        "seen_json": 0,
        "prediction_records": 0,
        "resumed_records": 0,
        "fallback_records": 0,
        "resumed_fallback_records": 0,
        "updated_records": 0,
        "unchanged_records": 0,
        "skipped_records": 0,
        "failed_records": 0,
        "rewritten_images": 0,
        "failed_image_records": 0,
    }
    map_cache: dict[str, Mapping[str, object]] = {}

    prediction_files = iter_prediction_files(method_root)
    if limit is not None:
        prediction_files = prediction_files[:limit]
    resolved_method_name = method_name or method_name_from_summary(method_root)
    checkpoint_path = checkpoint_directory / "state.json"
    partial_summary_path = checkpoint_directory / "summary.partial.json"
    expected_root = str(method_root.resolve())
    checkpoint: dict[str, object]
    if resume and not dry_run and checkpoint_path.exists():
        checkpoint = load_json_object(checkpoint_path)
        if (
            checkpoint.get("version") != CHECKPOINT_VERSION
            or checkpoint.get("evaluator_fingerprint") != evaluator_id
            or checkpoint.get("method_root") != expected_root
        ):
            raise ValueError(
                f"Incompatible metric checkpoint: {display_path(checkpoint_path)}"
            )
    else:
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "method": resolved_method_name,
            "method_root": expected_root,
            "evaluator_fingerprint": evaluator_id,
            "status": "in_progress",
            "completed": {},
            "failures": {},
            "fallbacks": {},
        }
    completed = checkpoint.get("completed")
    failures = checkpoint.get("failures")
    fallbacks = checkpoint.setdefault("fallbacks", {})
    if (
        not isinstance(completed, dict)
        or not isinstance(failures, dict)
        or not isinstance(fallbacks, dict)
    ):
        raise ValueError(f"Invalid metric checkpoint: {display_path(checkpoint_path)}")

    def write_checkpoint(status: str) -> None:
        if dry_run:
            return
        checkpoint.update(
            {
                "status": status,
                "updated_at": utc_now(),
                "json_file_count": len(prediction_files),
                "completed_count": len(completed),
                "failed_count": len(failures),
                "fallback_count": len(fallbacks),
                "stats": dict(stats),
            }
        )
        write_json_object(checkpoint_path, checkpoint)

    def write_partial_summary(status: str) -> dict[str, object]:
        partial = rebuild_method_summary(records, method_name=resolved_method_name)
        partial["metric_recompute"] = {
            "status": status,
            "evaluator_fingerprint": evaluator_id,
            "completed_count": len(records),
            "json_file_count": len(prediction_files),
            "failed_count": stats["failed_records"],
            "fallback_count": len(fallbacks),
            "scores_on_completed_records_only": status != "completed",
        }
        if not dry_run:
            write_json_object(partial_summary_path, partial)
        return partial

    print_progress(
        f"[method:start] {resolved_method_name} root={display_path(method_root)} "
        f"json_files={len(prediction_files)} dry_run={dry_run} "
        f"resume={resume and not dry_run} rewrite_images={rewrite_images}",
        quiet=quiet,
    )

    for index, path in enumerate(prediction_files, start=1):
        stats["seen_json"] += 1
        relative_path = display_path(path)
        checkpoint_key = path.relative_to(method_root).as_posix()
        resumed_record = False
        record_used_fallback = False
        metric_fallback_error: str | None = None
        payload_write_allowed = True
        record_json_error: BaseException | None = None
        try:
            try:
                payload = load_json_object(path)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                if fail_fast:
                    raise
                relative = path.relative_to(method_root)
                if relative.parent == Path("."):
                    raise ValueError(
                        "Cannot recover map_id from malformed prediction path: "
                        f"{relative_path}"
                    ) from exc
                map_id = relative.parent.as_posix()
                instruction_id = path.stem
                instruction, instruction_path = load_instruction_by_id(
                    map_id,
                    instruction_id,
                )
                payload = {
                    "method": resolved_method_name,
                    "map_id": map_id,
                    "scene_id": map_id,
                    "instruction_id": instruction_id,
                    "difficulty_level": instruction.get(
                        "difficulty_level",
                        "unknown",
                    ),
                    "trajectory": [],
                }
                payload_write_allowed = False
                record_json_error = exc

            if record_json_error is None and not has_prediction_fields(payload):
                stats["skipped_records"] += 1
                print_progress(
                    f"[record:skip] {resolved_method_name} "
                    f"{index}/{len(prediction_files)} path={relative_path} "
                    "reason=missing_prediction_fields",
                    quiet=quiet,
                )
                continue

            trajectory = payload.get("trajectory")
            map_id = payload.get("map_id")
            if not isinstance(trajectory, list) or not isinstance(map_id, str):
                raise ValueError("record has invalid prediction fields.")
            if record_json_error is None:
                instruction_path = resolve_instruction_file_for_payload(payload)
                instruction = load_instruction(instruction_path)
            source_fingerprint = input_fingerprint(
                trajectory,
                instruction_path=instruction_path,
                instruction=instruction,
                map_id=map_id,
                evaluator_id=evaluator_id,
            )
            if record_json_error is not None:
                digest = hashlib.sha256(source_fingerprint.encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                source_fingerprint = digest.hexdigest()

            entry = completed.get(checkpoint_key)
            saved_metrics = payload.get("metrics")
            if record_json_error is not None:
                if map_id not in map_cache:
                    map_cache[map_id] = load_map_state(map_id)
                metrics = worst_case_metrics_for_evaluation_error(
                    trajectory,
                    instruction,
                    map_cache[map_id],
                    error=record_json_error,
                )
                metric_fallback_error = (
                    f"{type(record_json_error).__name__}: {record_json_error}"
                )
                record_used_fallback = True
                stats["fallback_records"] += 1
                print(
                    f"[record:fallback] {relative_path}: "
                    f"{metric_fallback_error}",
                    file=sys.stderr,
                )
            elif (
                resume
                and not dry_run
                and isinstance(entry, Mapping)
                and entry.get("input_fingerprint") == source_fingerprint
                and isinstance(saved_metrics, Mapping)
                and entry.get("metrics_fingerprint")
                == json_fingerprint(saved_metrics)
            ):
                metrics = dict(saved_metrics)
                resumed_record = True
                record_used_fallback = entry.get("worst_case_fallback") is True
                stats["resumed_records"] += 1
                if record_used_fallback:
                    stats["resumed_fallback_records"] += 1
            else:
                try:
                    metrics = evaluate_prediction(
                        trajectory,
                        map_id=map_id,
                        instruction_file=instruction_path,
                    )
                except Exception as metric_error:
                    if fail_fast:
                        raise
                    if map_id not in map_cache:
                        map_cache[map_id] = load_map_state(map_id)
                    metrics = worst_case_metrics_for_evaluation_error(
                        trajectory,
                        instruction,
                        map_cache[map_id],
                        error=metric_error,
                    )
                    metric_fallback_error = (
                        f"{type(metric_error).__name__}: {metric_error}"
                    )
                    record_used_fallback = True
                    stats["fallback_records"] += 1
                    print(
                        f"[record:fallback] {relative_path}: "
                        f"{metric_fallback_error}",
                        file=sys.stderr,
                    )
        except Exception as exc:
            stats["failed_records"] += 1
            failures[checkpoint_key] = {
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": utc_now(),
            }
            write_checkpoint("in_progress_with_failures")
            print(f"[record:failed] {relative_path}: {exc}", file=sys.stderr)
            if fail_fast:
                raise
            continue

        stats["prediction_records"] += 1
        metrics_summary = metric_summary(metrics)
        record_changed = (
            payload.get("metrics") != metrics
            or payload.get("metrics_summary") != metrics_summary
        )
        if rewrite_images and payload_write_allowed and not dry_run:
            instruction_id = payload.get("instruction_id")
            try:
                if not isinstance(instruction_id, str) or not instruction_id.strip():
                    raise ValueError("record has no instruction_id.")
                if map_id not in map_cache:
                    map_cache[map_id] = load_map_state(map_id)
                previous_image = payload.get("trajectory_image")
                previous_gt_image = payload.get("trajectory_with_gt_image")
                save_trajectory_image_for_record(
                    payload,
                    map_cache[map_id],
                    trajectory,
                    path,
                    nested_record_keys=nested_image_record_keys(payload),
                    instruction=instruction,
                )
                stats["rewritten_images"] += 1
                if (
                    payload.get("trajectory_image") != previous_image
                    or payload.get("trajectory_with_gt_image") != previous_gt_image
                ):
                    record_changed = True
            except Exception as exc:
                stats["failed_image_records"] += 1
                print(f"[image:failed] {relative_path}: {exc}", file=sys.stderr)
                if fail_fast:
                    raise

        payload["metrics"] = metrics
        payload["metrics_summary"] = metrics_summary
        if record_changed and payload_write_allowed:
            stats["updated_records"] += 1
            if not dry_run:
                write_json_object(path, payload)
            if metric_fallback_error is not None:
                action = "would_fallback" if dry_run else "fallback"
            else:
                action = "would_update" if dry_run else "updated"
        elif metric_fallback_error is not None:
            action = "would_fallback" if dry_run else "fallback"
        elif resumed_record:
            action = "skip"
        else:
            stats["unchanged_records"] += 1
            action = "unchanged"

        records.append(record_for_summary(payload, path, metrics))
        completed[checkpoint_key] = {
            "input_fingerprint": source_fingerprint,
            "metrics_fingerprint": json_fingerprint(metrics),
            "worst_case_fallback": record_used_fallback,
            "updated_at": utc_now(),
        }
        if metric_fallback_error is not None:
            fallbacks[checkpoint_key] = {
                "error": metric_fallback_error,
                "policy": "worst_case",
                "updated_at": utc_now(),
            }
        elif not record_used_fallback:
            fallbacks.pop(checkpoint_key, None)
        failures.pop(checkpoint_key, None)
        write_checkpoint("in_progress")

        if checkpoint_every > 0 and stats["prediction_records"] % checkpoint_every == 0:
            write_partial_summary("in_progress")

        if progress_every > 0 and (
            action != "unchanged"
            or stats["prediction_records"] % progress_every == 0
            or index == len(prediction_files)
        ):
            reason = (
                " reason=resume_checkpoint_match"
                if action == "skip"
                else ""
            )
            print_progress(
                f"[record:{action}] {resolved_method_name} "
                f"{index}/{len(prediction_files)} path={relative_path}"
                f"{reason} HCS={metrics_summary.get('HCS')} "
                f"SCS={metrics_summary.get('SCS')} "
                f"PL={metrics_summary.get('PL')} "
                f"H-SPL={metrics_summary.get('segment_wise_SPL')}",
                quiet=quiet,
            )

    final_status = (
        "completed"
        if stats["failed_records"] == 0
        else "incomplete_with_failures"
    )
    summary = write_partial_summary(final_status)
    summary_path = method_root / "summary.json"
    official_summary = dict(summary)
    official_summary.pop("metric_recompute", None)
    if not dry_run and final_status == "completed":
        write_json_object(summary_path, official_summary)
    write_checkpoint(final_status)
    summary_action = (
        "would_write"
        if dry_run
        else "wrote"
        if final_status == "completed"
        else "kept_previous_due_to_failures"
    )
    print_progress(
        f"[method:summary] {resolved_method_name} "
        f"{summary_action}={display_path(summary_path)} "
        f"prediction_records={stats['prediction_records']} "
        f"resumed={stats['resumed_records']} updated={stats['updated_records']} "
        f"fallback={len(fallbacks)} unchanged={stats['unchanged_records']} "
        f"skipped={stats['skipped_records']} failed={stats['failed_records']}",
        quiet=quiet,
    )
    return official_summary if final_status == "completed" else summary, stats


def discover_method_roots(root: Path, methods: Sequence[str] | None) -> list[Path]:
    if methods:
        resolved: list[Path] = []
        for method in methods:
            candidates = [
                root / method,
                root / "baselines" / method,
            ]
            if method.lower() in {"grounding2route", "main"}:
                candidates.append(root / "grounding2route" / "main_result")
            resolved.append(next((path for path in candidates if path.exists()), candidates[0]))
        return resolved
    if not root.exists():
        return []
    if root.resolve() == DEFAULT_METHOD_ROOT.resolve():
        baseline_root = root / "baselines"
        primary = (
            [
                path
                for path in baseline_root.iterdir()
                if path.is_dir() and not is_skipped_directory(path.name)
            ]
            if baseline_root.exists()
            else []
        )
        grounding2route_root = root / "grounding2route" / "main_result"
        if grounding2route_root.exists():
            primary.append(grounding2route_root)
        return sorted(primary)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not is_skipped_directory(path.name)
    )


def build_aggregate_summary(
    summaries: Iterable[tuple[Path, Mapping[str, object]]],
) -> dict[str, object]:
    methods: dict[str, object] = {}
    for summary_path, summary in summaries:
        method_name = summary.get("method")
        if not isinstance(method_name, str) or not method_name.strip():
            method_name = summary_path.parent.name

        entry: dict[str, object] = {
            "summary_path": display_path(summary_path),
            "overall": summary.get("overall"),
        }
        for difficulty in AGGREGATE_SECTIONS[1:]:
            entry[difficulty] = summary.get(difficulty)
        methods[method_name] = entry

    return {
        "version": 1,
        "sections": list(AGGREGATE_SECTIONS),
        "method_count": len(methods),
        "methods": methods,
    }


def recompute_human_expert_summary(
    output_path: Path,
    *,
    dry_run: bool,
    limit: int | None,
    fail_fast: bool,
    quiet: bool,
    workers: int,
    evaluator_id: str,
    checkpoint_directory: Path,
    resume: bool = True,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> tuple[dict[str, object], dict[str, int]]:
    """Recompute instruction-embedded human demonstrations with resumable cache."""

    partial_summary_path = checkpoint_directory / "summary.partial.json"

    def cache_path(
        instruction_path: Path,
        instruction: Mapping[str, object],
    ) -> Path:
        map_id = instruction.get("map_id")
        if not isinstance(map_id, str) or not map_id.strip():
            map_id = map_id_from_instruction_path(instruction_path)
        instruction_id = instruction_id_from_payload(instruction_path, instruction)
        return (
            checkpoint_directory
            / "records"
            / map_id.strip().strip("/")
            / f"{instruction_id}.json"
        )

    def source_fingerprint(
        instruction_path: Path,
        instruction: Mapping[str, object],
    ) -> str:
        trajectory = instruction.get("human_expert_trajectory")
        map_id = instruction.get("map_id")
        if not isinstance(trajectory, list):
            raise ValueError("instruction has no human_expert_trajectory list.")
        if not isinstance(map_id, str) or not map_id.strip():
            map_id = map_id_from_instruction_path(instruction_path)
        return input_fingerprint(
            trajectory,
            instruction_path=instruction_path,
            instruction=instruction,
            map_id=map_id,
            evaluator_id=evaluator_id,
        )

    def load_cached_record(
        instruction_path: Path,
        instruction: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        if not resume or dry_run:
            return None
        path = cache_path(instruction_path, instruction)
        if not path.exists():
            return None
        try:
            payload = load_json_object(path)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        record = payload.get("record")
        if (
            payload.get("version") != CHECKPOINT_VERSION
            or payload.get("evaluator_fingerprint") != evaluator_id
            or payload.get("input_fingerprint")
            != source_fingerprint(instruction_path, instruction)
            or not isinstance(record, Mapping)
            or payload.get("metrics_fingerprint")
            != json_fingerprint(record.get("metrics"))
        ):
            return None
        return record

    def write_cached_record(
        instruction_path: Path,
        instruction: Mapping[str, object],
        record: Mapping[str, object],
    ) -> None:
        if dry_run:
            return
        write_json_object(
            cache_path(instruction_path, instruction),
            {
                "version": CHECKPOINT_VERSION,
                "evaluator_fingerprint": evaluator_id,
                "input_fingerprint": source_fingerprint(
                    instruction_path,
                    instruction,
                ),
                "metrics_fingerprint": json_fingerprint(record.get("metrics")),
                "updated_at": utc_now(),
                "record": dict(record),
            },
        )

    def write_partial(summary: dict[str, object]) -> None:
        summary["metric_recompute"] = {
            "status": "in_progress",
            "evaluator_fingerprint": evaluator_id,
            "scores_on_completed_records_only": True,
        }
        write_json_object(partial_summary_path, summary)

    print_progress(
        f"[human-expert:start] output={display_path(output_path)} "
        f"dry_run={dry_run} resume={resume and not dry_run} workers={workers}",
        quiet=quiet,
    )
    summary, stats = evaluate_human_experts(
        limit=limit,
        fail_fast=fail_fast,
        quiet=quiet,
        workers=workers,
        checkpoint_every=checkpoint_every,
        worst_case_on_metric_error=True,
        checkpoint_callback=None if dry_run else write_partial,
        cache_loader=load_cached_record,
        cache_writer=write_cached_record,
    )
    final_status = (
        "completed"
        if stats["failed_instructions"] == 0
        else "incomplete_with_failures"
    )
    partial = dict(summary)
    partial["metric_recompute"] = {
        "status": final_status,
        "evaluator_fingerprint": evaluator_id,
        "scores_on_completed_records_only": final_status != "completed",
    }
    if not dry_run:
        write_json_object(partial_summary_path, partial)
        if final_status == "completed":
            write_json_object(output_path, summary)
    print_progress(
        f"[human-expert:summary] "
        f"{'would_write' if dry_run else 'wrote' if final_status == 'completed' else 'kept_previous_due_to_failures'}="
        f"{display_path(output_path)} "
        f"evaluated={stats['evaluated_instructions']} "
        f"resumed={stats['resumed_instructions']} "
        f"computed={stats['computed_instructions']} "
        f"skipped={stats['skipped_instructions']} "
        f"failed={stats['failed_instructions']}",
        quiet=quiet,
    )
    return summary, stats

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute metrics and summaries for saved method trajectories using "
            "the current evaluator. This does not rerun the methods."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_METHOD_ROOT,
        help="Directory containing method output directories.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help=(
            "Method directory name under --root. Repeat to process a fixed set; "
            "omit to process every method directory under --root."
        ),
    )
    parser.add_argument(
        "--method-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional method output root to process, useful for methods saved "
            "outside resources/methods."
        ),
    )
    parser.add_argument(
        "--write-aggregate",
        action="store_true",
        help="Rewrite --aggregate-path from the summaries recomputed in this run.",
    )
    parser.add_argument(
        "--aggregate-path",
        type=Path,
        default=DEFAULT_METHOD_ROOT / "summary.json",
        help="Aggregate method summary path used with --write-aggregate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Recompute and print stats without writing records or summaries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Maximum records per method and human-expert instructions; "
            "omit for all records."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed prediction record.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help=(
            "Print progress every N prediction records. Defaults to every record; "
            "set to 0 to only print method-level progress."
        ),
    )
    parser.add_argument(
        "--rewrite-images",
        action="store_true",
        help=(
            "Regenerate trajectory PNGs for prediction records. This also writes "
            "a *_with_gt.png overlay with black human-expert trajectory points "
            "when the instruction contains human_expert_trajectory."
        ),
    )
    parser.add_argument(
        "--human-expert",
        action="store_true",
        help=(
            "Also recompute instruction.human_expert_trajectory and write "
            "human_solver/human_expert_summary.json using the shared evaluator."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Workers for human-expert metric evaluation; defaults to 1.",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Optional label for this metric migration. The evaluator fingerprint "
            "is always appended, so changed metric code starts a separate run."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Rewrite partial summaries every N completed records; defaults to 10.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching checkpoints and recompute every record.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress lines and only print the final JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative.")
    if args.limit is not None and not args.dry_run:
        raise ValueError("--limit is only supported with --dry-run.")

    root = args.root.resolve()
    evaluator_id = evaluator_fingerprint()
    run_label = (
        f"{safe_run_label(args.run_id)}--{evaluator_id}"
        if args.run_id
        else evaluator_id
    )
    checkpoint_run_root = DEFAULT_METHOD_ROOT / "grounding2route" / "previous" / "diagnostics" / "_metric_recompute" / run_label
    resume = not args.no_resume and not args.dry_run

    def checkpoint_directory(method_root: Path) -> Path:
        root_hash = hashlib.sha256(
            str(method_root.resolve()).encode("utf-8")
        ).hexdigest()[:8]
        return checkpoint_run_root / f"{safe_run_label(method_root.name)}-{root_hash}"

    method_roots = discover_method_roots(args.root, args.method)
    method_roots.extend(args.method_root)
    human_method_root = (
        DEFAULT_METHOD_ROOT / "baselines" / HUMAN_EXPERT_METHOD_NAME
    ).resolve()
    if args.human_expert:
        method_roots = [
            method_root
            for method_root in method_roots
            if method_root.resolve() != human_method_root
        ]

    run_summaries: list[tuple[Path, dict[str, object]]] = []
    report: list[dict[str, object]] = []
    print_progress(
        f"[run:start] root={display_path(root)} "
        f"methods={len(method_roots)} dry_run={args.dry_run} "
        f"resume={resume} evaluator={evaluator_id} "
        f"checkpoint={display_path(checkpoint_run_root)} "
        f"rewrite_images={args.rewrite_images}",
        quiet=args.quiet,
    )
    for method_root in method_roots:
        method_root = method_root.resolve()
        summary_path = method_root / "summary.json"
        summary, stats = recompute_method(
            method_root,
            dry_run=args.dry_run,
            limit=args.limit,
            fail_fast=args.fail_fast,
            quiet=args.quiet,
            progress_every=args.progress_every,
            rewrite_images=args.rewrite_images,
            evaluator_id=evaluator_id,
            checkpoint_directory=checkpoint_directory(method_root),
            resume=resume,
            checkpoint_every=args.checkpoint_every,
        )
        run_summaries.append((summary_path, summary))
        report.append(
            {
                "method": summary.get("method", method_root.name),
                "method_root": display_path(method_root),
                "summary_path": display_path(summary_path),
                "stats": stats,
                "overall": summary.get("overall"),
            }
        )

    if args.human_expert:
        human_summary_path = human_method_root / "human_expert_summary.json"
        human_summary, human_stats = recompute_human_expert_summary(
            human_summary_path,
            dry_run=args.dry_run,
            limit=args.limit,
            fail_fast=args.fail_fast,
            quiet=args.quiet,
            workers=args.workers,
            evaluator_id=evaluator_id,
            checkpoint_directory=checkpoint_directory(human_method_root),
            resume=resume,
            checkpoint_every=args.checkpoint_every,
        )
        run_summaries.append((human_summary_path, human_summary))
        report.append(
            {
                "method": HUMAN_EXPERT_METHOD_NAME,
                "trajectory_source": "instruction.human_expert_trajectory",
                "method_root": display_path(human_method_root),
                "summary_path": display_path(human_summary_path),
                "stats": human_stats,
                "overall": human_summary.get("overall"),
            }
        )

    aggregate_path = args.aggregate_path.resolve()
    if args.write_aggregate:
        aggregate_summary = build_aggregate_summary(run_summaries)
        if not args.dry_run:
            write_json_object(aggregate_path, aggregate_summary)
        print_progress(
            f"[aggregate:{'would_write' if args.dry_run else 'wrote'}] "
            f"path={display_path(aggregate_path)} methods={len(run_summaries)}",
            quiet=args.quiet,
        )
    else:
        aggregate_summary = None
        print_progress(
            "[aggregate:skip] use --write-aggregate to rewrite it",
            quiet=args.quiet,
        )

    print(
        json.dumps(
            {
                "root": display_path(root),
                "dry_run": bool(args.dry_run),
                "resume": resume,
                "run_id": run_label,
                "evaluator_fingerprint": evaluator_id,
                "checkpoint_root": display_path(checkpoint_run_root),
                "method_count": len(report),
                "methods": report,
                "aggregate_path": (
                    display_path(aggregate_path) if args.write_aggregate else None
                ),
                "aggregate": aggregate_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

if __name__ == "__main__":
    main()
