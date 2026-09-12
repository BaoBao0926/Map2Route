#!/usr/bin/env python3
"""Run the SayPlan SemPathBench adaptation."""

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
from scripts.make_instruction.make_instruction import REPO_ROOT
from scripts.methods.sayplan.adapters.sempathbench_loader import load_episode
from scripts.methods.sayplan.config import (
    DEFAULT_MAX_JSON_RETRIES,
    DEFAULT_MAX_REPLANS,
    DEFAULT_MAX_SEARCH_STEPS,
    INFERENCE_CONTRACT_VERSION,
    METHOD_NAME,
)
from scripts.methods.sayplan.llm.client import configured_model
from scripts.methods.sayplan.planning.pipeline import SayPlanRunResult, run_sayplan_episode
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


METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / METHOD_NAME


def _existing_prediction_uses_current_contract(output_path: Path) -> bool:
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    method_record = payload.get(METHOD_NAME)
    return (
        isinstance(method_record, Mapping)
        and method_record.get("inference_contract_version")
        == INFERENCE_CONTRACT_VERSION
    )


def _instruction_summary(instruction: Mapping[str, object]) -> dict[str, object]:
    return {
        "text": instruction.get("instruction", ""),
        "difficulty_level": instruction.get("difficulty_level", ""),
        "template_instruction_id": instruction.get("template_instruction_id"),
    }


def build_prediction_record(
    instruction_file: Path,
    *,
    map_id: str,
    scene_id: str,
    instruction_id: str,
    instruction: Mapping[str, object],
    model: str,
    trajectory: list[list[int]],
    metrics: dict[str, object] | None,
    run_result: SayPlanRunResult,
    max_search_steps: int,
    max_replans: int,
    max_json_retries: int,
    verbose: bool,
    evaluation_error: str | None = None,
) -> dict[str, object]:
    difficulty_level = difficulty_from_instruction(instruction)
    sayplan_record: dict[str, object] = {
        "description": "SayPlan adapted to SemPathBench with scene-graph collapse, expand/contract semantic search, high-level planning, A* path completion, and textual-feedback replanning.",
        "input_contract": "Inference uses only the free-form instruction, start_pose, and the map-derived compact scene graph. GT objects, hard_constraints, soft_constraints, and human_expert_trajectory are excluded. Dense grid paths are produced by the classical planner.",
        "inference_contract_version": INFERENCE_CONTRACT_VERSION,
        "annotation_isolation": {
            "allowed_instruction_fields": ["instruction", "start_pose"],
            "excluded_fields": [
                "objects",
                "hard_constraints",
                "soft_constraints",
                "human_expert_trajectory",
            ],
        },
        "model": model,
        "max_search_steps": max_search_steps,
        "max_replans": max_replans,
        "max_json_retries": max_json_retries,
        "verbose": verbose,
        "planning_failed": run_result.planning_failed,
        "failure_reason": run_result.failure_reason,
        "final_high_level_plan": run_result.final_high_level_plan,
        "task_subgraph": run_result.task_subgraph,
        "semantic_search_trace": run_result.semantic_search_trace,
        "planning_attempts": run_result.planning_attempts,
        "runtime": run_result.runtime,
    }
    if evaluation_error:
        sayplan_record["evaluation_error"] = evaluation_error

    return {
        "version": 1,
        "method": METHOD_NAME,
        "prediction_id": f"{METHOD_NAME}_{uuid.uuid4().hex[:8]}",
        "map_id": map_id,
        "scene_id": scene_id,
        "instruction_id": instruction_id,
        "difficulty_level": difficulty_level,
        "instruction_file": display_path(instruction_file),
        "created_at": iso_now(),
        "trajectory": trajectory,
        "metrics": metrics,
        "metrics_summary": metric_summary(metrics),
        METHOD_NAME: sayplan_record,
        "instruction": _instruction_summary(instruction),
    }


def _evaluate_trajectory(
    trajectory: list[list[int]],
    *,
    map_id: str,
    instruction_id: str,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        return (
            evaluate_prediction(
                trajectory,
                map_id=map_id,
                instruction_id=instruction_id,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - keep batch runs writing debuggable records.
        return None, f"{type(exc).__name__}: {exc}"


def _attach_runtime(
    record: dict[str, object],
    runtime_seconds: float,
    *,
    status: str,
    method_seconds: float | None = None,
) -> None:
    record["runtime_seconds"] = runtime_seconds
    record["runtime"] = {
        "seconds": runtime_seconds,
        "status": status,
    }
    if method_seconds is not None:
        runtime = record["runtime"]
        if isinstance(runtime, dict):
            runtime["method_seconds"] = method_seconds
    sayplan_record = record.get(METHOD_NAME)
    if isinstance(sayplan_record, dict):
        sayplan_record["runtime_seconds"] = runtime_seconds


def build_all(
    input_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
    limit: int | None = None,
    model: str | None = None,
    max_search_steps: int = DEFAULT_MAX_SEARCH_STEPS,
    max_replans: int = DEFAULT_MAX_REPLANS,
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
    verbose: bool = False,
    workers: int = 1,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    instruction_files = iter_instruction_files(
        input_root,
        instruction_set=instruction_set,
    )
    if limit is not None:
        instruction_files = instruction_files[:limit]

    resolved_model = model or configured_model()
    total = len(instruction_files)

    def run_one(index: int, instruction_file: Path) -> dict[str, object]:
        episode_start = time.perf_counter()
        instruction = load_instruction(instruction_file)
        map_id = map_id_from_instruction_path(instruction_file, input_root)
        instruction_id = instruction_id_from_payload(instruction_file, instruction)
        output_path = output_path_for_prediction(output_root, map_id, instruction_id)
        if verbose:
            print(
                f"[SayPlan][runner] episode={index}/{total} "
                f"map_id={map_id} instruction_id={instruction_id} "
                f"instruction_file={display_path(instruction_file)}",
                flush=True,
            )

        existing_is_current = (
            output_path.exists()
            and _existing_prediction_uses_current_contract(output_path)
        )
        if existing_is_current and not overwrite:
            result = load_existing_result(output_path, map_id, instruction_id)
            scene_id = instruction.get("map_id")
            result["scene_id"] = scene_id if isinstance(scene_id, str) else map_id
            if verbose:
                print(
                    f"[SayPlan][runner] skipped_existing output={display_path(output_path)}",
                    flush=True,
                )
        else:
            if verbose and output_path.exists() and not overwrite:
                print(
                    f"[SayPlan][runner] recomputing_stale_prediction "
                    f"required_inference_contract_version={INFERENCE_CONTRACT_VERSION} "
                    f"output={display_path(output_path)}",
                    flush=True,
                )
            method_start = time.perf_counter()
            if verbose:
                print(
                    f"[SayPlan][runner] model={resolved_model} output={display_path(output_path)}",
                    flush=True,
                )
            episode = load_episode(
                instruction_file=instruction_file,
                map_id=map_id,
                instruction_id=instruction_id,
                instruction=instruction,
            )
            if verbose:
                print(
                    f"[SayPlan][runner] loaded_episode scene_id={episode.scene_id} "
                    f"start_cell={episode.start_cell} failure_reason={episode.failure_reason}",
                    flush=True,
                )
            run_result = run_sayplan_episode(
                episode,
                model=resolved_model,
                max_search_steps=max_search_steps,
                max_replans=max_replans,
                max_json_retries=max_json_retries,
                verbose=verbose,
            )
            method_seconds = time.perf_counter() - method_start
            metrics, evaluation_error = _evaluate_trajectory(
                run_result.trajectory,
                map_id=map_id,
                instruction_id=instruction_id,
            )
            if verbose:
                print(
                    f"[SayPlan][runner] evaluated metrics_summary={metric_summary(metrics)} "
                    f"evaluation_error={evaluation_error!r}",
                    flush=True,
                )
            record = build_prediction_record(
                instruction_file=instruction_file,
                map_id=map_id,
                scene_id=episode.scene_id,
                instruction_id=instruction_id,
                instruction=instruction,
                model=resolved_model,
                trajectory=run_result.trajectory,
                metrics=metrics,
                run_result=run_result,
                max_search_steps=max_search_steps,
                max_replans=max_replans,
                max_json_retries=max_json_retries,
                verbose=verbose,
                evaluation_error=evaluation_error,
            )
            try:
                save_trajectory_image_for_record(
                    record,
                    episode.map_state,
                    run_result.trajectory,
                    output_path,
                    nested_record_keys=(METHOD_NAME,),
                    instruction=instruction,
                )
                if verbose:
                    print(
                        f"[SayPlan][runner] trajectory_image_written "
                        f"output={display_path(output_path.with_suffix('.png'))}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001 - image output is non-critical.
                sayplan_record = record.get(METHOD_NAME)
                if isinstance(sayplan_record, dict):
                    sayplan_record["trajectory_image_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                if verbose:
                    print(
                        f"[SayPlan][runner] trajectory_image_failed "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )

            runtime_seconds = time.perf_counter() - episode_start
            _attach_runtime(
                record,
                runtime_seconds,
                status="written",
                method_seconds=method_seconds,
            )
            write_prediction_record(output_path, record)
            if verbose:
                print(
                    f"[SayPlan][runner] record_written output={display_path(output_path)} "
                    f"runtime_seconds={runtime_seconds:.3f}",
                    flush=True,
                )
            result = result_from_prediction_record(record, output_path)

        if (
            result.get("runtime_seconds") is None
            and result.get("status") != "skipped_exists"
        ):
            runtime_seconds = time.perf_counter() - episode_start
            result["runtime_seconds"] = runtime_seconds
            result["runtime"] = {
                "seconds": runtime_seconds,
                "status": result.get("status"),
            }
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
    print(
        f"total={len(results)} written={written} skipped={skipped} failed={failed}",
        flush=True,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SayPlan trajectories for SemPathBench instructions."
    )
    parser.add_argument("--input-root", type=Path, default=INSTRUCTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=METHOD_ROOT)
    parser.add_argument(
        "--set",
        dest="instruction_set",
        choices=INSTRUCTION_SET_CHOICES,
        default=DEFAULT_INSTRUCTION_SET,
        help="Instruction set to run. Use 'all' to run every instruction.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Deprecated: metrics are always computed with the shared evaluator.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Number of independent instruction episodes to run concurrently.")
    parser.add_argument(
        "--model",
        help="Gemini model override. Defaults to MODEL in scripts/methods/api_key.py.",
    )
    parser.add_argument(
        "--max-search-steps",
        type=int,
        default=DEFAULT_MAX_SEARCH_STEPS,
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=DEFAULT_MAX_REPLANS,
    )
    parser.add_argument(
        "--max-json-retries",
        type=int,
        default=DEFAULT_MAX_JSON_RETRIES,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed SayPlan runner, semantic-search, planning, path, and verifier logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_all(
        args.input_root,
        args.output_root,
        overwrite=args.overwrite,
        instruction_set=args.instruction_set,
        limit=args.limit,
        model=args.model,
        max_search_steps=args.max_search_steps,
        max_replans=args.max_replans,
        max_json_retries=args.max_json_retries,
        verbose=args.verbose,
        workers=args.workers,
    )
    # Build one complete summary from disk. This includes predictions from
    # previous interrupted runs without rewriting it for every resumed skip.
    complete_results = collect_prediction_results(
        args.output_root,
        method_name=METHOD_NAME,
    )
    summary_path = write_summary(
        complete_results,
        args.input_root,
        args.output_root,
        method_name=METHOD_NAME,
        evaluate=True,
    )
    print_summary(summary_path)


if __name__ == "__main__":
    main()
