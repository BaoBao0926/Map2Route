#!/usr/bin/env python3
"""Run the ILN SemPathBench adapter."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Mapping

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluate_prediction import evaluate_prediction
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state
from scripts.methods.iln.config import (
    DEFAULT_UNKNOWN_PASSAGE_COST,
    EVENT_MODES,
    GROUNDING_MODES,
    HISTORY_MODES,
    METHOD_NAME,
    METHOD_ROOT,
    PASSAGE_EXTRACTION_MODES,
)
from scripts.methods.iln.llm_client import GeminiClient, configured_model
from scripts.methods.iln.pipeline import ILNRunConfig, run_iln_pipeline
from scripts.methods.tutorial.utils import (
    collect_prediction_results,
    difficulty_from_instruction,
    display_path,
    iso_now,
    load_existing_result,
    metric_summary,
    output_path_for_prediction,
    print_episode_result,
    print_summary,
    result_from_prediction_record,
    write_prediction_record,
    write_summary,
)
from scripts.methods.util.instructions import (
    DEFAULT_INSTRUCTION_SET,
    INSTRUCTION_ROOT,
    INSTRUCTION_SET_CHOICES,
    instruction_id_from_payload,
    iter_instruction_files,
    load_instruction,
    map_id_from_instruction_path,
)
from scripts.methods.util.methods import save_trajectory_image_for_record
from scripts.methods.util.workers import run_instruction_jobs


def verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[iln] {message}", flush=True)


def saved_prediction_is_current_iln(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(payload, Mapping) and payload.get("method") == METHOD_NAME


def scene_id_from_instruction(instruction: Mapping[str, object], map_id: str) -> str:
    scene_id = instruction.get("map_id")
    if isinstance(scene_id, str) and scene_id.strip():
        return scene_id.strip()
    return map_id


def build_prediction_record(
    instruction_file: Path,
    *,
    map_id: str,
    scene_id: str,
    instruction_id: str,
    instruction: Mapping[str, object],
    trajectory: list[list[int]],
    iln_metadata: dict[str, object],
    metrics: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "version": 1,
        "method": METHOD_NAME,
        "prediction_id": f"ILN_{uuid.uuid4().hex[:8]}",
        "map_id": map_id,
        "scene_id": scene_id,
        "instruction_id": instruction_id,
        "difficulty_level": difficulty_from_instruction(instruction),
        "instruction_file": display_path(instruction_file),
        "created_at": iso_now(),
        "trajectory": trajectory,
        "metrics": metrics,
        "metrics_summary": metric_summary(metrics),
        "iln": iln_metadata,
        "instruction": {
            "text": instruction.get("instruction", ""),
            "difficulty_level": instruction.get("difficulty_level", ""),
            "template_instruction_id": instruction.get("template_instruction_id"),
        },
    }


def _evaluate(trajectory: list[list[int]], *, map_id: str, instruction_id: str) -> tuple[dict[str, object] | None, str | None]:
    try:
        return evaluate_prediction(trajectory, map_id=map_id, instruction_id=instruction_id), None
    except Exception as exc:
        return None, str(exc)


def build_all(
    input_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
    limit: int | None = None,
    model: str | None = None,
    llm_cache_root: Path | None = None,
    graph_cache_root: Path | None = None,
    grounding_mode: str = "heuristic",
    event_mode: str = "empty",
    history_mode: str = "empty",
    history_file: Path | None = None,
    event_file: Path | None = None,
    max_prompt_objects: int = 120,
    passage_extraction: str = "boundary",
    unknown_passage_cost: float = DEFAULT_UNKNOWN_PASSAGE_COST,
    allow_grid_fallback: bool = False,
    overwrite_llm_cache: bool = False,
    verbose: bool = False,
    workers: int = 1,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    instruction_files = iter_instruction_files(input_root, instruction_set=instruction_set)
    if limit is not None:
        instruction_files = instruction_files[:limit]
    run_config = ILNRunConfig(
        grounding_mode=grounding_mode,
        event_mode=event_mode,
        history_mode=history_mode,
        passage_extraction=passage_extraction,
        max_prompt_objects=max_prompt_objects,
        unknown_passage_cost=unknown_passage_cost,
        allow_grid_fallback=allow_grid_fallback,
        history_file=history_file,
        event_file=event_file,
        graph_cache_root=graph_cache_root,
    )

    total = len(instruction_files)
    def run_one(index: int, instruction_file: Path) -> dict[str, object]:
        episode_start = time.perf_counter()
        # Keep the API/cache client local to one episode; client instances may
        # carry request-local state and must not be shared across worker threads.
        llm_client = GeminiClient(
            model=model or configured_model(),
            cache_root=llm_cache_root,
            overwrite_cache=overwrite_llm_cache,
        )
        instruction = load_instruction(instruction_file)
        map_id = map_id_from_instruction_path(instruction_file, input_root)
        scene_id = scene_id_from_instruction(instruction, map_id)
        instruction_id = instruction_id_from_payload(instruction_file, instruction)
        output_path = output_path_for_prediction(output_root, map_id, instruction_id)

        if (
            output_path.exists()
            and not overwrite
            and saved_prediction_is_current_iln(output_path)
        ):
            result = load_existing_result(output_path, map_id, instruction_id)
            result["scene_id"] = scene_id
        else:
            if output_path.exists() and not overwrite:
                verbose_print(
                    verbose,
                    f"[{index}/{total}] regenerating stale/non-ILN output "
                    f"{display_path(output_path)}",
                )
            method_start = time.perf_counter()
            verbose_print(verbose, f"[{index}/{total}] loading map {map_id}")
            map_state = load_map_state(map_id)
            verbose_print(verbose, f"[{index}/{total}] running ILN pipeline")
            pipeline_result = run_iln_pipeline(
                map_state,
                instruction,
                config=run_config,
                llm_client=llm_client,
                verbose=verbose,
            )
            trajectory = pipeline_result.trajectory
            if not trajectory:
                trajectory = [[0, 0]]
                pipeline_result.metadata.setdefault("method_level_deviations", [])
                deviations = pipeline_result.metadata["method_level_deviations"]
                if isinstance(deviations, list):
                    deviations.append("empty_failure_trajectory_replaced_with_origin")
            method_seconds = time.perf_counter() - method_start
            metrics, evaluation_error = _evaluate(
                trajectory,
                map_id=map_id,
                instruction_id=instruction_id,
            )
            if evaluation_error:
                pipeline_result.metadata["evaluation_error"] = evaluation_error
            record = build_prediction_record(
                instruction_file,
                map_id=map_id,
                scene_id=scene_id,
                instruction_id=instruction_id,
                instruction=instruction,
                trajectory=trajectory,
                iln_metadata=pipeline_result.metadata,
                metrics=metrics,
            )
            save_trajectory_image_for_record(
                record,
                map_state,
                trajectory,
                output_path,
                nested_record_keys=("iln",),
                instruction=instruction,
            )
            runtime_seconds = time.perf_counter() - episode_start
            record["runtime_seconds"] = runtime_seconds
            record["runtime"] = {
                "seconds": runtime_seconds,
                "method_seconds": method_seconds,
                "status": "written",
            }
            iln_record = record.get("iln")
            if isinstance(iln_record, dict):
                iln_record["runtime_seconds"] = runtime_seconds
            write_prediction_record(output_path, record)
            result = result_from_prediction_record(record, output_path)

        if result.get("runtime_seconds") is None and result.get("status") != "skipped_exists":
            runtime_seconds = time.perf_counter() - episode_start
            result["runtime_seconds"] = runtime_seconds
            result["runtime"] = {"seconds": runtime_seconds, "status": result.get("status")}
        return result

    for index, result in run_instruction_jobs(
        instruction_files,
        workers=workers,
        run_one=run_one,
    ):
        results.append(result)
        print_episode_result(index, total, result)

    written = sum(1 for item in results if item["status"] == "written")
    skipped = sum(1 for item in results if item["status"] == "skipped_exists")
    failed = sum(1 for item in results if item["status"] == "failed")
    print(f"total={len(results)} written={written} skipped={skipped} failed={failed}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ILN SemPathBench adapter.")
    parser.add_argument("--input-root", type=Path, default=INSTRUCTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=METHOD_ROOT)
    parser.add_argument("--set", dest="instruction_set", choices=INSTRUCTION_SET_CHOICES, default=DEFAULT_INSTRUCTION_SET)
    parser.add_argument("--model", default=None)
    parser.add_argument("--llm-cache-root", type=Path, default=None)
    parser.add_argument("--graph-cache-root", type=Path, default=None)
    parser.add_argument("--grounding-mode", choices=GROUNDING_MODES, default="llm")
    parser.add_argument("--event-mode", choices=EVENT_MODES, default="empty")
    parser.add_argument("--history-mode", choices=HISTORY_MODES, default="empty")
    parser.add_argument("--history-file", type=Path)
    parser.add_argument("--event-file", type=Path)
    parser.add_argument("--max-prompt-objects", type=int, default=120)
    parser.add_argument("--passage-extraction", choices=PASSAGE_EXTRACTION_MODES, default="boundary")
    parser.add_argument("--unknown-passage-cost", type=float, default=DEFAULT_UNKNOWN_PASSAGE_COST)
    parser.add_argument("--allow-grid-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-llm-cache", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Number of independent instruction episodes to run concurrently.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Deprecated: metrics are always computed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    llm_cache_root = args.llm_cache_root or output_root / "_llm_cache"
    graph_cache_root = args.graph_cache_root or output_root / "_graph_cache"

    results = build_all(
        args.input_root,
        output_root,
        overwrite=args.overwrite,
        instruction_set=args.instruction_set,
        limit=args.limit,
        model=args.model,
        llm_cache_root=llm_cache_root,
        graph_cache_root=graph_cache_root,
        grounding_mode=args.grounding_mode,
        event_mode=args.event_mode,
        history_mode=args.history_mode,
        history_file=args.history_file,
        event_file=args.event_file,
        max_prompt_objects=args.max_prompt_objects,
        passage_extraction=args.passage_extraction,
        unknown_passage_cost=args.unknown_passage_cost,
        allow_grid_fallback=args.allow_grid_fallback,
        overwrite_llm_cache=args.overwrite_llm_cache,
        verbose=args.verbose,
        workers=args.workers,
    )
    # Build one complete summary from disk. This includes predictions from
    # previous interrupted runs without rewriting it for every resumed skip.
    complete_results = collect_prediction_results(
        output_root,
        method_name=METHOD_NAME,
    )
    summary_path = write_summary(
        complete_results,
        args.input_root,
        output_root,
        method_name=METHOD_NAME,
        evaluate=True,
    )
    print_summary(summary_path)


if __name__ == "__main__":
    main()
