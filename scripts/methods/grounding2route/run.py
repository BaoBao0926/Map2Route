#!/usr/bin/env python3
"""Run the Grounding2Route SemPathBench method."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Mapping

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluate_prediction import evaluate_prediction
from scripts.evaluation.evaluation_metrics import (
    worst_case_metrics_for_evaluation_error,
)
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state
from scripts.methods.grounding2route.config import (
    DEFAULT_GROUNDING_VARIANT,
    DEFAULT_PLANNER_MODE,
    DEFAULT_RELATIVE_COST_MODE,
    GROUNDING_VARIANTS,
    MAX_EXPANSIONS,
    MAX_GROUNDING_REPAIRS,
    MAX_TOOL_CALLS,
    MAX_TOOL_LLM_TURNS,
    MAX_TOOL_QUERY_RETRIES,
    METHOD_NAME,
    METHOD_ROOT,
    PARSE_MODES,
    PLANNER_MODES,
    RELATIVE_COST_MODES,
    WEIGHT_RELATIVE,
)
from scripts.methods.grounding2route.artifacts import save_grounding2route_debug_artifacts
from scripts.methods.grounding2route.constraint_evaluation import evaluate_constraint_grounding, write_grounding_report
from scripts.methods.grounding2route.llm_client import GeminiClient, configured_model
from scripts.methods.grounding2route.pipeline import Grounding2RouteRunConfig, run_grounding2route_pipeline
from scripts.methods.grounding2route.tool_efficiency import write_tool_call_efficiency_report
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
    write_json_atomic,
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
from scripts.methods.util.workers import normalize_worker_count


def verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[grounding2route] {message}", flush=True)


def scene_id_from_instruction(instruction: Mapping[str, object], map_id: str) -> str:
    scene_id = instruction.get("map_id")
    if isinstance(scene_id, str) and scene_id.strip():
        return scene_id.strip()
    return map_id


def intermediate_path_for_prediction(output_path: Path) -> Path:
    return output_path.with_suffix(".steps.json")


def replay_grounding_code_path(root: Path, map_id: str, instruction_id: str) -> Path:
    return output_path_for_prediction(root, map_id, instruction_id).with_suffix(".steps.json")


def load_replay_grounding_code(
    root: Path,
    map_id: str,
    instruction_id: str,
    *,
    max_repair_attempt: int | None = None,
) -> tuple[str, dict[str, object]]:
    path = replay_grounding_code_path(root, map_id, instruction_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Replay steps file is not an object: {path}")
    steps = payload.get("steps")
    if not isinstance(steps, Mapping):
        raise ValueError(f"Replay steps file has no steps: {path}")
    attempts = steps.get("grounding_code_attempts")
    if max_repair_attempt is not None:
        candidates: list[Mapping[str, object]] = []
        if isinstance(attempts, list):
            for attempt in attempts:
                if not isinstance(attempt, Mapping) or attempt.get("stage") != "execute_and_verify":
                    continue
                original_attempt = attempt.get("attempt")
                if (
                    isinstance(original_attempt, int)
                    and original_attempt <= max_repair_attempt
                    and isinstance(attempt.get("code"), str)
                ):
                    candidates.append(attempt)
        if not candidates:
            return "", {
                "source_path": display_path(path),
                "selection": "original_generation_failed",
                "max_repair_attempt": max_repair_attempt,
            }
        selected = candidates[-1]
        return str(selected["code"]), {
            "source_path": display_path(path),
            "selection": "repair_budget",
            "max_repair_attempt": max_repair_attempt,
            "original_attempt": selected.get("attempt"),
            "original_status": selected.get("status"),
        }
    grounding_code = steps.get("grounding_code")
    if isinstance(grounding_code, Mapping) and isinstance(grounding_code.get("source"), str):
        return str(grounding_code["source"]), {
            "source_path": display_path(path),
            "selection": "final_source",
        }
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if isinstance(attempt, Mapping) and attempt.get("status") == "success" and isinstance(attempt.get("code"), str):
                return str(attempt["code"]), {
                    "source_path": display_path(path),
                    "selection": "latest_successful_source",
                    "original_attempt": attempt.get("attempt"),
                }
    return "", {
        "source_path": display_path(path),
        "selection": "original_generation_failed",
    }


def write_intermediate_steps(path: Path, payload: Mapping[str, object]) -> None:
    write_json_atomic(path, payload)


def _prediction_artifacts_complete(
    output_path: Path,
    steps_path: Path,
    *,
    resume_execution_errors: bool = False,
) -> bool:
    """Return whether a saved episode is safe to skip during a resumed run."""

    try:
        record = json.loads(output_path.read_text(encoding="utf-8"))
        steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(record, Mapping) or not isinstance(steps_payload, Mapping):
        return False
    grounding2route = record.get("grounding2route")
    if (
        resume_execution_errors
        and isinstance(grounding2route, Mapping)
        and grounding2route.get("status") == "EXECUTION_ERROR_WORST_CASE"
    ):
        return False
    trajectory = record.get("trajectory")
    metrics = record.get("metrics")
    steps = steps_payload.get("steps")
    if not isinstance(trajectory, list) or not isinstance(metrics, Mapping):
        return False
    if not isinstance(steps, Mapping):
        return False
    for name in ("HCS", "PL", "segment_wise_SPL"):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
    return True


def build_prediction_record(
    instruction_file: Path,
    *,
    map_id: str,
    scene_id: str,
    instruction_id: str,
    instruction: Mapping[str, object],
    trajectory: list[list[int]],
    grounding2route_metadata: dict[str, object],
    metrics: dict[str, object] | None,
    intermediate_path: Path,
    grounded_program: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "version": 1,
        "method": METHOD_NAME,
        "prediction_id": f"Grounding2Route_{uuid.uuid4().hex[:8]}",
        "map_id": map_id,
        "scene_id": scene_id,
        "instruction_id": instruction_id,
        "difficulty_level": difficulty_from_instruction(instruction),
        "instruction_file": display_path(instruction_file),
        "created_at": iso_now(),
        "trajectory": trajectory,
        "grounded_program": dict(grounded_program) if grounded_program is not None else None,
        "metrics": metrics,
        "metrics_summary": metric_summary(metrics),
        "grounding2route": {
            **grounding2route_metadata,
            "intermediate_steps_path": display_path(intermediate_path),
        },
        "instruction": {
            "text": instruction.get("instruction", ""),
            "difficulty_level": instruction.get("difficulty_level", ""),
            "template_instruction_id": instruction.get("template_instruction_id"),
        },
    }


def _evaluate(
    trajectory: list[list[int]],
    *,
    map_id: str,
    instruction_id: str,
    instruction: Mapping[str, object],
    map_state: Mapping[str, object],
) -> tuple[dict[str, object], str | None]:
    try:
        return evaluate_prediction(trajectory, map_id=map_id, instruction_id=instruction_id), None
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        return (
            worst_case_metrics_for_evaluation_error(
                trajectory,
                instruction,
                map_state,
                error=error_text,
            ),
            error_text,
        )


def _start_only_trajectory(instruction: Mapping[str, object]) -> list[list[int]]:
    start_pose = instruction.get("start_pose")
    if not isinstance(start_pose, Mapping):
        return []
    row = start_pose.get("row")
    col = start_pose.get("col")
    if (
        isinstance(row, int)
        and not isinstance(row, bool)
        and isinstance(col, int)
        and not isinstance(col, bool)
    ):
        return [[row, col]]
    return []


def build_all(
    input_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
    limit: int | None = None,
    parse_mode: str = "intent",
    model: str | None = None,
    llm_cache_root: Path | None = None,
    overwrite_llm_cache: bool = False,
    max_expansions: int = MAX_EXPANSIONS,
    max_parse_repairs: int = 0,
    max_grounding_repairs: int = MAX_GROUNDING_REPAIRS,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_tool_llm_turns: int = MAX_TOOL_LLM_TURNS,
    max_tool_query_retries: int = MAX_TOOL_QUERY_RETRIES,
    grounding_variant: str = DEFAULT_GROUNDING_VARIANT,
    code_refinement: bool | None = None,
    allow_grounding_helpers: bool | None = None,
    execution_repair: str | None = None,
    scope_ablation: str = "full",
    planner_mode: str = DEFAULT_PLANNER_MODE,
    planner_heuristic_weight: float = 1.0,
    relative_cost_mode: str = DEFAULT_RELATIVE_COST_MODE,
    relative_weight: float = WEIGHT_RELATIVE,
    soft_weight_scale: float = 1.0,
    global_min_path_improvement_m: float | None = None,
    replay_grounding_code_root: Path | None = None,
    replay_max_grounding_repairs: int | None = None,
    workers: int = 1,
    resume_execution_errors: bool = False,
    save_debug_artifacts: bool = False,
    verbose: bool = False,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    existing_results = collect_prediction_results(output_root, method_name=METHOD_NAME)
    summary_results = {
        (str(result.get("map_id")), str(result.get("instruction_id"))): result
        for result in existing_results
    }
    if not overwrite:
        # A resumed run should expose a complete summary immediately, even if
        # every selected episode is skipped or the process is stopped early.
        write_summary(
            list(summary_results.values()),
            input_root,
            output_root,
            method_name=METHOD_NAME,
            evaluate=True,
        )
    instruction_files = iter_instruction_files(input_root, instruction_set=instruction_set)
    if limit is not None:
        instruction_files = instruction_files[:limit]

    config = Grounding2RouteRunConfig(
        parse_mode=parse_mode,
        max_parse_repairs=max_parse_repairs,
        max_expansions=max_expansions,
        max_grounding_repairs=(0 if replay_grounding_code_root is not None else max_grounding_repairs),
        max_tool_calls=max_tool_calls,
        max_tool_llm_turns=max_tool_llm_turns,
        max_tool_query_retries=max_tool_query_retries,
        grounding_variant=grounding_variant,
        code_refinement=code_refinement,
        allow_grounding_helpers=allow_grounding_helpers,
        relative_cost_mode=relative_cost_mode,
        relative_weight=relative_weight,
        soft_weight_scale=soft_weight_scale,
        global_min_path_improvement_m=global_min_path_improvement_m,
        execution_repair=execution_repair,
        scope_ablation=scope_ablation,
        planner_mode=planner_mode,
        planner_heuristic_weight=planner_heuristic_weight,
    )
    total = len(instruction_files)
    worker_count = normalize_worker_count(workers)
    if grounding_variant == "tool_call" and worker_count != 1:
        print(
            "[Grounding2Route][warning] ToolCall forces --workers 1 after a prior "
            "multi-threaded interpreter corruption; ignoring the requested "
            f"worker count {worker_count}.",
            flush=True,
        )
        worker_count = 1

    def update_running_summary(result: dict[str, object]) -> None:
        key = (str(result.get("map_id")), str(result.get("instruction_id")))
        summary_results[key] = result
        write_summary(
            list(summary_results.values()),
            input_root,
            output_root,
            method_name=METHOD_NAME,
            evaluate=True,
        )

    def run_one(index: int, instruction_file: Path) -> dict[str, object]:
        episode_start = time.perf_counter()
        instruction = load_instruction(instruction_file)
        map_id = map_id_from_instruction_path(instruction_file, input_root)
        scene_id = scene_id_from_instruction(instruction, map_id)
        instruction_id = instruction_id_from_payload(instruction_file, instruction)
        output_path = output_path_for_prediction(output_root, map_id, instruction_id)
        steps_path = intermediate_path_for_prediction(output_path)

        if not overwrite and _prediction_artifacts_complete(
            output_path,
            steps_path,
            resume_execution_errors=resume_execution_errors,
        ):
            result = load_existing_result(output_path, map_id, instruction_id)
            result["scene_id"] = scene_id
        else:
            if not overwrite and (output_path.exists() or steps_path.exists()):
                verbose_print(
                    verbose,
                    f"[{index}/{total}] rerunning incomplete artifacts for {map_id}/{instruction_id}",
                )
            method_start = time.perf_counter()
            verbose_print(verbose, f"[{index}/{total}] loading map {map_id}")
            map_state = load_map_state(map_id)
            verbose_print(verbose, f"[{index}/{total}] running Grounding2Route pipeline")
            replay_selection = (
                load_replay_grounding_code(
                    replay_grounding_code_root,
                    map_id,
                    instruction_id,
                    max_repair_attempt=replay_max_grounding_repairs,
                )
                if replay_grounding_code_root is not None
                else None
            )
            replay_code = replay_selection[0] if replay_selection is not None else None
            llm_client = None if replay_code is not None else GeminiClient(
                model=model or configured_model(),
                cache_root=llm_cache_root,
                overwrite_cache=overwrite_llm_cache,
            )
            pipeline_result = run_grounding2route_pipeline(
                map_state,
                instruction,
                config=config,
                llm_client=llm_client,
                grounding_code_override=replay_code,
            )
            # Keep every LLM request, response, and cache provenance adjacent
            # to the episode-level IR/code/verification artifacts. Replays are
            # intentionally offline and therefore carry an empty trace.
            pipeline_result.steps["llm_trace"] = llm_client.trace() if llm_client is not None else []
            if replay_selection is not None:
                replay_metadata = dict(replay_selection[1])
                pipeline_result.steps["offline_replay"] = replay_metadata
                pipeline_result.metadata["offline_replay"] = replay_metadata
            method_seconds = time.perf_counter() - method_start
            grounded_program_json = (
                pipeline_result.debug_data.grounded.to_json()
                if pipeline_result.debug_data is not None and pipeline_result.debug_data.grounded is not None
                else None
            )
            # This is post-hoc grounding evaluation only. It cannot influence
            # generated code, repair, grounded task, or planner execution.
            if grounded_program_json is not None and pipeline_result.debug_data is not None and pipeline_result.debug_data.scene is not None:
                grounding_evaluation = evaluate_constraint_grounding(
                    pipeline_result.debug_data.grounded,
                    instruction,
                    pipeline_result.debug_data.scene,
                )
                pipeline_result.steps["grounding_evaluation"] = grounding_evaluation
                pipeline_result.metadata["grounding_evaluation"] = {
                    key: value for key, value in grounding_evaluation.items() if key != "errors"
                }
            metrics, evaluation_error = _evaluate(
                pipeline_result.trajectory,
                map_id=map_id,
                instruction_id=instruction_id,
                instruction=instruction,
                map_state=map_state,
            )
            if evaluation_error:
                pipeline_result.metadata["evaluation_error"] = evaluation_error

            steps_payload = {
                "version": 1,
                "method": METHOD_NAME,
                "map_id": map_id,
                "scene_id": scene_id,
                "instruction_id": instruction_id,
                "instruction_text": instruction.get("instruction", ""),
                "steps": pipeline_result.steps,
                "final_status": pipeline_result.metadata.get("status"),
                "failure_reason": pipeline_result.failure_reason,
            }
            write_intermediate_steps(steps_path, steps_payload)

            record = build_prediction_record(
                instruction_file,
                map_id=map_id,
                scene_id=scene_id,
                instruction_id=instruction_id,
                instruction=instruction,
                trajectory=pipeline_result.trajectory,
                grounding2route_metadata=pipeline_result.metadata,
                metrics=metrics,
                intermediate_path=steps_path,
                grounded_program=grounded_program_json,
            )
            try:
                save_trajectory_image_for_record(
                    record,
                    map_state,
                    pipeline_result.trajectory,
                    output_path,
                    nested_record_keys=("grounding2route",),
                    instruction=instruction,
                )
            except Exception as exc:  # noqa: BLE001 - PNG output is non-critical.
                image_error = f"{type(exc).__name__}: {exc}"
                record["trajectory_image_error"] = image_error
                grounding2route = record.get("grounding2route")
                if isinstance(grounding2route, dict):
                    grounding2route["trajectory_image_error"] = image_error
                print(
                    f"[Grounding2Route][warning] trajectory image failed for "
                    f"{scene_id}/{instruction_id}: {image_error}; continuing.",
                    flush=True,
                )
            if save_debug_artifacts:
                artifact_paths = save_grounding2route_debug_artifacts(
                    output_path=output_path,
                    map_state=map_state,
                    instruction=instruction,
                    trajectory=pipeline_result.trajectory,
                    steps=pipeline_result.steps,
                    debug_data=pipeline_result.debug_data,
                )
                record["grounding2route_debug_artifacts"] = artifact_paths
                grounding2route = record.get("grounding2route")
                if isinstance(grounding2route, dict):
                    grounding2route["debug_artifacts"] = artifact_paths
            runtime_seconds = time.perf_counter() - episode_start
            record["runtime_seconds"] = runtime_seconds
            record["runtime"] = {
                "seconds": runtime_seconds,
                "method_seconds": method_seconds,
                "status": "written",
            }
            grounding2route = record.get("grounding2route")
            if isinstance(grounding2route, dict):
                grounding2route["runtime_seconds"] = runtime_seconds
            write_prediction_record(output_path, record)
            result = result_from_prediction_record(record, output_path)

        if result.get("runtime_seconds") is None and result.get("status") != "skipped_exists":
            runtime_seconds = time.perf_counter() - episode_start
            result["runtime_seconds"] = runtime_seconds
            result["runtime"] = {"seconds": runtime_seconds, "status": result.get("status")}
        result["_grounding2route_index"] = index
        return result

    def run_one_guarded(index: int, instruction_file: Path) -> dict[str, object]:
        try:
            return run_one(index, instruction_file)
        except Exception as exc:  # noqa: BLE001 - persist one bad episode and continue.
            if isinstance(exc, SystemError) and "unknown opcode" in str(exc).lower():
                # The interpreter is corrupted; continuing would fabricate
                # worst-case results for every remaining episode.
                raise
            episode_start = time.perf_counter()
            exception_traceback = traceback.format_exc()
            error_text = f"{type(exc).__name__}: {exc}"
            instruction = load_instruction(instruction_file)
            map_id = map_id_from_instruction_path(instruction_file, input_root)
            scene_id = scene_id_from_instruction(instruction, map_id)
            instruction_id = instruction_id_from_payload(instruction_file, instruction)
            output_path = output_path_for_prediction(output_root, map_id, instruction_id)
            steps_path = intermediate_path_for_prediction(output_path)
            try:
                map_state = load_map_state(map_id)
                trajectory = _start_only_trajectory(instruction)
                metrics = worst_case_metrics_for_evaluation_error(
                    trajectory,
                    instruction,
                    map_state,
                    error=f"Grounding2Route unexpected episode failure: {error_text}",
                )
                failure = {
                    "policy": "worst_case",
                    "stage": "run_grounding2route_pipeline",
                    "error": error_text,
                    "traceback": exception_traceback,
                }
                write_intermediate_steps(
                    steps_path,
                    {
                        "version": 1,
                        "method": METHOD_NAME,
                        "map_id": map_id,
                        "scene_id": scene_id,
                        "instruction_id": instruction_id,
                        "instruction_text": instruction.get("instruction", ""),
                        "steps": {"execution_fallback": failure},
                        "final_status": "EXECUTION_ERROR_WORST_CASE",
                        "failure_reason": error_text,
                    },
                )
                runtime_seconds = time.perf_counter() - episode_start
                record = build_prediction_record(
                    instruction_file,
                    map_id=map_id,
                    scene_id=scene_id,
                    instruction_id=instruction_id,
                    instruction=instruction,
                    trajectory=trajectory,
                    grounding2route_metadata={
                        "status": "EXECUTION_ERROR_WORST_CASE",
                        "failure": failure,
                        "runtime_seconds": runtime_seconds,
                    },
                    metrics=metrics,
                    intermediate_path=steps_path,
                )
                record["execution_fallback"] = failure
                record["runtime_seconds"] = runtime_seconds
                record["runtime"] = {
                    "seconds": runtime_seconds,
                    "status": "written_worst_case",
                }
                write_prediction_record(output_path, record)
                result = result_from_prediction_record(record, output_path)
                result["status"] = "written_worst_case"
                result["error"] = error_text
                result["_grounding2route_index"] = index
                print(
                    f"[Grounding2Route][warning] {scene_id}/{instruction_id} failed with "
                    f"{error_text}; wrote metric-specific worst case and continued.",
                    flush=True,
                )
                return result
            except Exception as fallback_exc:  # noqa: BLE001 - preserve suite progress.
                return {
                    "instruction_id": instruction_id,
                    "map_id": map_id,
                    "scene_id": scene_id,
                    "status": "failed",
                    "path": display_path(output_path),
                    "difficulty_level": difficulty_from_instruction(instruction),
                    "runtime_seconds": time.perf_counter() - episode_start,
                    "error": (
                        f"{error_text}; worst-case persistence failed: "
                        f"{type(fallback_exc).__name__}: {fallback_exc}"
                    ),
                    "_grounding2route_index": index,
                }

    if worker_count == 1:
        for index, instruction_file in enumerate(instruction_files, start=1):
            result = run_one_guarded(index, instruction_file)
            result.pop("_grounding2route_index", None)
            results.append(result)
            if result.get("status") != "skipped_exists":
                update_running_summary(result)
            print_episode_result(index, total, result)
    else:
        print(f"workers={worker_count}", flush=True)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_one_guarded, index, instruction_file)
                for index, instruction_file in enumerate(instruction_files, start=1)
            ]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                original_index = result.pop("_grounding2route_index", completed)
                results.append(result)
                results.sort(key=lambda item: str(item.get("path") or ""))
                if result.get("status") != "skipped_exists":
                    update_running_summary(result)
                print_episode_result(completed, total, {**result, "episode_index": original_index})

    written = sum(1 for item in results if item["status"] == "written")
    skipped = sum(1 for item in results if item["status"] == "skipped_exists")
    failed = sum(1 for item in results if item["status"] == "failed")
    print(f"total={len(results)} written={written} skipped={skipped} failed={failed}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Grounding2Route SemPathBench method.")
    parser.add_argument("--input-root", type=Path, default=INSTRUCTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=METHOD_ROOT)
    parser.add_argument("--set", dest="instruction_set", choices=INSTRUCTION_SET_CHOICES, default=DEFAULT_INSTRUCTION_SET)
    parser.add_argument("--parse-mode", choices=PARSE_MODES, default="intent")
    parser.add_argument("--model", default=None)
    parser.add_argument("--llm-cache-root", type=Path, default=METHOD_ROOT / "_llm_cache")
    parser.add_argument("--overwrite-llm-cache", action="store_true")
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=MAX_EXPANSIONS,
        help="Planner expansion budget; 0 means unlimited.",
    )
    parser.add_argument(
        "--max-parse-repairs",
        type=int,
        default=0,
        help="LLM rewrites after representation parse failures; default 0 (disabled).",
    )
    parser.add_argument("--max-grounding-repairs", type=int, default=MAX_GROUNDING_REPAIRS)
    parser.add_argument(
        "--grounding-representation",
        choices=("code", "json", "direct_id", "ltl", "tool_call", "function_call"),
        default="code",
        help=(
            "Grounding representation: canonical Direct-CaP code, JSON IR, "
            "compact-catalog Direct-ID, restricted LTL, or native ToolCall."
        ),
    )
    parser.add_argument(
        "--grounding-variant",
        choices=GROUNDING_VARIANTS,
        default=None,
        help="Advanced architecture override; takes precedence over --grounding-representation.",
    )
    parser.add_argument(
        "--code-refinement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether the LLM refines the deterministic grounding-code scaffold.",
    )
    parser.add_argument(
        "--allow-grounding-helpers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether LLM grounding code may define pure helper functions.",
    )
    parser.add_argument(
        "--execution-repair",
        choices=("off", "code", "code_then_ir", "tool"),
        default=None,
        help="Override repair policy after grounding or RouteIR verification fails.",
    )
    parser.add_argument(
        "--scope-ablation",
        choices=("full", "no_spatial_scope", "no_segment_scope", "no_scope", "no_soft_constraints"),
        default="full",
        help="Direct-CaP scope ablation applied after grounding and before the fixed planner.",
    )
    parser.add_argument(
        "--planner-mode",
        choices=PLANNER_MODES,
        default=DEFAULT_PLANNER_MODE,
        help=(
            "Planner backend: sequential greedy A* (default), exact layered "
            "global DP, or product-state global progress A*."
        ),
    )
    parser.add_argument(
        "--planner-heuristic-weight",
        type=float,
        default=1.0,
        help=(
            "Global-progress A* heuristic weight; global_layered_dp requires 1.0 "
            "and the sequential planner ignores it."
        ),
    )
    parser.add_argument(
        "--relative-cost-mode",
        choices=RELATIVE_COST_MODES,
        default=DEFAULT_RELATIVE_COST_MODE,
        help="Relative preference field; ratio is the canonical full-pipeline setting.",
    )
    parser.add_argument(
        "--relative-weight",
        type=float,
        default=WEIGHT_RELATIVE,
        help="Weight of the relative preference field; canonical value is 64.",
    )
    parser.add_argument(
        "--soft-weight-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier for grounded near/far/relative/path-shape preference fields; "
            "clearance is unchanged. Default 1.0."
        ),
    )
    parser.add_argument(
        "--global-min-path-improvement-m",
        type=float,
        default=None,
        help=(
            "For global_layered_dp, keep the sequential incumbent unless the "
            "global candidate shortens the geometric path by at least this many metres."
        ),
    )
    parser.add_argument("--max-tool-calls", type=int, default=MAX_TOOL_CALLS, help="ToolCall semantic API budget per episode.")
    parser.add_argument("--max-tool-llm-turns", type=int, default=MAX_TOOL_LLM_TURNS, help="ToolCall Gemini turn budget per episode.")
    parser.add_argument("--max-tool-query-retries", type=int, default=MAX_TOOL_QUERY_RETRIES, help="Failed or empty executions allowed for one exact ToolCall query; hard-capped at 3.")
    parser.add_argument("--workers", type=int, default=1, help="Number of episodes to run concurrently; ToolCall is currently forced to 1 for safety.")
    parser.add_argument(
        "--resume-execution-errors",
        action="store_true",
        help=(
            "With --overwrite omitted, rerun saved episodes whose Grounding2Route "
            "status is EXECUTION_ERROR_WORST_CASE; skip all other complete episodes."
        ),
    )
    parser.add_argument(
        "--save-debug-artifacts",
        action="store_true",
        help="Write parser, grounding, cost-map, and semantic-map debug artifacts next to each prediction.",
    )
    parser.add_argument(
        "--replay-grounding-code-root",
        type=Path,
        default=None,
        help="Offline Direct-CaP replay source: read saved .steps.json grounding code instead of calling an LLM.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the offline repair-budget and scope ablation suite from saved full-run steps.",
    )
    parser.add_argument(
        "--ablation-source-root",
        type=Path,
        default=None,
        help="Full Direct-CaP result root used for zero-LLM replay; defaults to --output-root.",
    )
    parser.add_argument(
        "--ablation-output-root",
        type=Path,
        default=None,
        help="Suite output root; defaults to the ablation sibling of main_result.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Deprecated: metrics are always computed with the shared method evaluator.",
    )
    return parser.parse_args()


REPRESENTATION_VARIANTS = {
    "code": DEFAULT_GROUNDING_VARIANT,
    "json": "ir_to_code",
    "direct_id": "direct_id",
    "ltl": "ltl",
    "tool_call": "tool_call",
    "function_call": "tool_call",
}

AUTOMATIC_ABLATION_CASES: tuple[dict[str, object], ...] = (
    {"name": "repair_r0", "repair_budget": 0, "scope_ablation": "full"},
    {"name": "repair_r1", "repair_budget": 1, "scope_ablation": "full"},
    {"name": "repair_r2", "repair_budget": 2, "scope_ablation": "full"},
    {"name": "scope_no_temporal", "repair_budget": None, "scope_ablation": "no_segment_scope"},
    {"name": "scope_no_spatial", "repair_budget": None, "scope_ablation": "no_spatial_scope"},
    {"name": "scope_none", "repair_budget": None, "scope_ablation": "no_scope"},
    {"name": "no_soft_constraints", "repair_budget": None, "scope_ablation": "no_soft_constraints"},
    {
        "name": "planner_global_layered",
        "repair_budget": None,
        "scope_ablation": "full",
        "planner_mode": "global_layered_dp",
        "planner_heuristic_weight": 1.0,
        "max_expansions": 0,
    },
)


AUTOMATIC_ABLATION_PATHS = {
    "repair_r0": "03_verification_and_repair/repair_r0",
    "repair_r1": "03_verification_and_repair/repair_r1",
    "repair_r2": "03_verification_and_repair/repair_r2",
    "scope_no_temporal": "04_soft_constraint_modeling/global_t",
    "scope_no_spatial": "04_soft_constraint_modeling/global_s",
    "scope_none": "04_soft_constraint_modeling/global_ts",
    "no_soft_constraints": "04_soft_constraint_modeling/hard_only",
    "planner_global_layered": "05_global_vs_sequential_planning/global_planner",
}


def _resolved_grounding_variant(args: argparse.Namespace) -> str:
    if isinstance(args.grounding_variant, str):
        return args.grounding_variant
    return REPRESENTATION_VARIANTS[args.grounding_representation]


def _resolved_parse_mode(args: argparse.Namespace, grounding_variant: str) -> str:
    if grounding_variant in {"direct_id", "ltl"}:
        return grounding_variant
    return str(args.parse_mode)


def _write_grounding_evaluation_report(output_root: Path) -> None:
    reports: list[Mapping[str, object]] = []
    for prediction_path in sorted(output_root.rglob("instruction_*.json")):
        if prediction_path.name.endswith(".steps.json"):
            continue
        try:
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        grounding2route = prediction.get("grounding2route") if isinstance(prediction, Mapping) else None
        report = grounding2route.get("grounding_evaluation") if isinstance(grounding2route, Mapping) else None
        if isinstance(report, Mapping):
            reports.append(report)
    report_paths = write_grounding_report(reports, output_root)
    metrics_path = report_paths["grounding_metrics"]
    print(
        f"grounding evaluation episodes={len(reports)} {display_path(metrics_path)}",
        flush=True,
    )


def _run_experiment(
    args: argparse.Namespace,
    *,
    output_root: Path,
    grounding_variant: str,
    scope_ablation: str = "full",
    replay_root: Path | None = None,
    replay_repair_budget: int | None = None,
    planner_mode: str | None = None,
    planner_heuristic_weight: float | None = None,
    max_expansions: int | None = None,
) -> Path:
    build_all(
        args.input_root,
        output_root,
        overwrite=args.overwrite,
        instruction_set=args.instruction_set,
        limit=args.limit,
        parse_mode=_resolved_parse_mode(args, grounding_variant),
        model=args.model,
        llm_cache_root=args.llm_cache_root,
        overwrite_llm_cache=(args.overwrite_llm_cache if replay_root is None else False),
        max_expansions=(args.max_expansions if max_expansions is None else max_expansions),
        max_parse_repairs=(args.max_parse_repairs if replay_root is None else 0),
        max_grounding_repairs=args.max_grounding_repairs,
        max_tool_calls=args.max_tool_calls,
        max_tool_llm_turns=args.max_tool_llm_turns,
        max_tool_query_retries=args.max_tool_query_retries,
        grounding_variant=grounding_variant,
        code_refinement=(args.code_refinement if replay_root is None else None),
        allow_grounding_helpers=(args.allow_grounding_helpers if replay_root is None else None),
        execution_repair=(args.execution_repair if replay_root is None else None),
        scope_ablation=scope_ablation,
        planner_mode=(args.planner_mode if planner_mode is None else planner_mode),
        planner_heuristic_weight=(
            args.planner_heuristic_weight
            if planner_heuristic_weight is None
            else planner_heuristic_weight
        ),
        replay_grounding_code_root=replay_root,
        replay_max_grounding_repairs=replay_repair_budget,
        relative_cost_mode=args.relative_cost_mode,
        relative_weight=args.relative_weight,
        soft_weight_scale=args.soft_weight_scale,
        global_min_path_improvement_m=args.global_min_path_improvement_m,
        workers=args.workers,
        resume_execution_errors=args.resume_execution_errors,
        save_debug_artifacts=args.save_debug_artifacts,
        verbose=args.verbose,
    )
    complete_results = collect_prediction_results(output_root, method_name=METHOD_NAME)
    summary_path = write_summary(
        complete_results,
        args.input_root,
        output_root,
        method_name=METHOD_NAME,
        evaluate=True,
    )
    print_summary(summary_path)
    if grounding_variant.startswith("direct_cap") or grounding_variant == "tool_call":
        _write_grounding_evaluation_report(output_root)
    if grounding_variant == "tool_call":
        efficiency_path = write_tool_call_efficiency_report(output_root)
        print(f"tool-call efficiency: {display_path(efficiency_path)}", flush=True)
    return summary_path


def _validate_ablation_source(args: argparse.Namespace, source_root: Path) -> None:
    instruction_files = iter_instruction_files(args.input_root, instruction_set=args.instruction_set)
    if args.limit is not None:
        instruction_files = instruction_files[: args.limit]
    missing: list[Path] = []
    for instruction_file in instruction_files:
        instruction = load_instruction(instruction_file)
        map_id = map_id_from_instruction_path(instruction_file, args.input_root)
        instruction_id = instruction_id_from_payload(instruction_file, instruction)
        steps_path = replay_grounding_code_path(source_root, map_id, instruction_id)
        if not steps_path.exists():
            missing.append(steps_path)
    if missing:
        examples = ", ".join(display_path(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Ablation source is incomplete: missing {len(missing)} .steps.json files; "
            f"examples: {examples}. Run the canonical Grounding2Route method first."
        )


def run_ablation_suite(args: argparse.Namespace) -> Path:
    source_root = args.ablation_source_root or args.output_root
    suite_root = args.ablation_output_root or args.output_root.parent / "ablation"
    if source_root.resolve() == suite_root.resolve():
        raise ValueError("Ablation output root must differ from the full-run source root.")
    _validate_ablation_source(args, source_root)
    suite_root.mkdir(parents=True, exist_ok=True)
    case_records: list[dict[str, object]] = []
    for case in AUTOMATIC_ABLATION_CASES:
        name = str(case["name"])
        case_root = suite_root / AUTOMATIC_ABLATION_PATHS[name]
        repair_budget = case["repair_budget"]
        scope_ablation = str(case["scope_ablation"])
        print(
            f"[grounding2route-ablation] case={name} offline_replay={display_path(source_root)}",
            flush=True,
        )
        summary_path = _run_experiment(
            args,
            output_root=case_root,
            grounding_variant=DEFAULT_GROUNDING_VARIANT,
            scope_ablation=scope_ablation,
            replay_root=source_root,
            replay_repair_budget=(repair_budget if isinstance(repair_budget, int) else None),
            planner_mode=str(case.get("planner_mode", DEFAULT_PLANNER_MODE)),
            planner_heuristic_weight=float(case.get("planner_heuristic_weight", 1.0)),
            max_expansions=(
                int(case["max_expansions"])
                if isinstance(case.get("max_expansions"), int)
                else args.max_expansions
            ),
        )
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        case_records.append(
            {
                **case,
                "output_root": display_path(case_root),
                "summary_path": display_path(summary_path),
                "overall": summary_payload.get("overall"),
            }
        )
    baseline_summary = source_root / "summary.json"
    manifest = {
        "version": 1,
        "method": METHOD_NAME,
        "protocol": "offline replay of one canonical Direct-CaP run; zero LLM calls",
        "source_root": display_path(source_root),
        "baseline_summary": display_path(baseline_summary) if baseline_summary.exists() else None,
        "canonical_grounding_variant": DEFAULT_GROUNDING_VARIANT,
        "cases": case_records,
    }
    manifest_path = suite_root / "table_iii_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ablation summary: {display_path(manifest_path)}", flush=True)
    return manifest_path


def main() -> None:
    args = parse_args()
    grounding_variant = _resolved_grounding_variant(args)
    if args.ablation:
        if grounding_variant != DEFAULT_GROUNDING_VARIANT:
            raise ValueError(
                "--ablation only runs non-representation ablations from the canonical "
                "code baseline. Run representation settings separately without --ablation."
            )
        if args.replay_grounding_code_root is not None:
            raise ValueError("Use --ablation-source-root, not --replay-grounding-code-root, with --ablation.")
        run_ablation_suite(args)
        return
    _run_experiment(
        args,
        output_root=args.output_root,
        grounding_variant=grounding_variant,
        scope_ablation=args.scope_ablation,
        replay_root=args.replay_grounding_code_root,
    )


if __name__ == "__main__":
    main()
