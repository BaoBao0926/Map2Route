#!/usr/bin/env python3
"""Run the SemPathBench LTLCodeGen pipeline."""

from __future__ import annotations

import argparse
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
from scripts.methods.ltlcodegen.llm import configured_model, translate_instruction_to_ltl
from scripts.methods.ltlcodegen.map_features import build_map_symbols, prompt_inventory
from scripts.methods.ltlcodegen.planner import (
    OBJECT_MODES,
    PLANNING_MODES,
    build_ltlcodegen_trajectory,
    start_cell_from_instruction,
)
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
    write_summary,
    write_prediction_record,
)
from scripts.methods.util.grid_astar import build_traversable_grid
from scripts.methods.util.instructions import (
    DEFAULT_INSTRUCTION_SET,
    INSTRUCTION_SET_CHOICES,
    INSTRUCTION_ROOT,
    instruction_id_from_payload,
    iter_instruction_files,
    load_instruction,
    map_id_from_instruction_path,
)
from scripts.methods.util.methods import save_trajectory_image_for_record
from scripts.methods.util.workers import run_instruction_jobs


METHOD_NAME = "LTLCodeGen"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / METHOD_NAME


def normalize_object_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "vinalla":
        return "vanilla"
    if normalized not in OBJECT_MODES:
        choices = ", ".join(OBJECT_MODES)
        raise argparse.ArgumentTypeError(f"--object-mode must be one of: {choices}")
    return normalized


def verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[ltlcodegen] {message}", flush=True)


def scene_id_from_instruction(
    instruction: Mapping[str, object],
    map_id: str,
) -> str:
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
    planner_details: list[dict[str, object]],
    translation: dict[str, object],
    automaton_summary: dict[str, object],
    map_inventory: dict[str, object],
    metrics: dict[str, object] | None,
    planning_mode: str,
    object_mode: str,
) -> dict[str, object]:
    planner_description = (
        "a product-state planner searches the grid x automaton graph."
        if planning_mode == "vanilla"
        else "a fast plain A* planner visits eventual AP targets in formula order."
    )
    return {
        "version": 1,
        "method": METHOD_NAME,
        "prediction_id": f"{METHOD_NAME}_{uuid.uuid4().hex[:8]}",
        "map_id": map_id,
        "scene_id": scene_id,
        "instruction_id": instruction_id,
        "difficulty_level": difficulty_from_instruction(instruction),
        "instruction_file": display_path(instruction_file),
        "created_at": iso_now(),
        "trajectory": trajectory,
        "metrics": metrics,
        "metrics_summary": metric_summary(metrics),
        "ltlcodegen": {
            "description": (
                "LTLCodeGen SemPathBench adapter: Gemini generates Python code, "
                "the generated question() is executed to produce LTL, and "
                f"{planner_description}"
            ),
            "input_contract": (
                "Planner input is restricted to map layers, map-derived semantic "
                "entities, instruction text, and start_pose. Annotation fields "
                "such as hard_constraints, soft_constraints, objects, and "
                "human_expert_trajectory are not used for planning."
            ),
            "generated_code": translation.get("generated_code"),
            "model": translation.get("model"),
            "planning_mode": planning_mode,
            "object_mode": object_mode,
            "ltl_formula": translation.get("ltl_formula"),
            "formula_ast": translation.get("formula_ast"),
            "removed_initial_aps": translation.get("removed_initial_aps"),
            "translation_attempts": translation.get("attempts"),
            "automaton_summary": automaton_summary,
            "map_inventory": {
                "room_category_counts": map_inventory.get("room_category_counts"),
                "object_category_counts": map_inventory.get("object_category_counts"),
                "object_prompt_count": map_inventory.get("object_prompt_count"),
                "object_total_count": map_inventory.get("object_total_count"),
            },
            "planner_details": planner_details,
        },
        "instruction": {
            "text": instruction.get("instruction", ""),
            "difficulty_level": instruction.get("difficulty_level", ""),
            "template_instruction_id": instruction.get("template_instruction_id"),
        },
    }


def build_all(
    input_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
    model: str,
    llm_cache_root: Path,
    overwrite_llm_cache: bool,
    max_prompt_objects: int,
    object_label_radius: int,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
    planning_mode: str = "vanilla",
    object_mode: str = "vanilla",
    limit: int | None = None,
    verbose: bool = False,
    workers: int = 1,
    max_episode_retries: int = 1,
) -> list[dict[str, object]]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    llm_cache_root = llm_cache_root.resolve()
    verbose_print(verbose, f"input_root={display_path(input_root)}")
    verbose_print(verbose, f"output_root={display_path(output_root)}")
    verbose_print(verbose, f"llm_cache_root={display_path(llm_cache_root)}")
    verbose_print(verbose, f"instruction_set={instruction_set}")
    verbose_print(verbose, f"model={model}")
    verbose_print(verbose, f"planning_mode={planning_mode}")
    verbose_print(verbose, f"object_mode={object_mode}")
    verbose_print(verbose, f"overwrite={overwrite}")
    verbose_print(verbose, f"overwrite_llm_cache={overwrite_llm_cache}")
    verbose_print(verbose, f"max_prompt_objects={max_prompt_objects}")
    verbose_print(verbose, f"object_label_radius={object_label_radius}")
    if max_episode_retries < 0:
        raise ValueError("--max-episode-retries must be at least 0.")
    verbose_print(verbose, f"max_episode_retries={max_episode_retries}")
    results: list[dict[str, object]] = []
    instruction_files = iter_instruction_files(
        input_root,
        instruction_set=instruction_set,
    )
    if limit is not None:
        instruction_files = instruction_files[:limit]
    verbose_print(verbose, f"matched_instruction_count={len(instruction_files)}")

    total = len(instruction_files)
    def run_one(index: int, instruction_file: Path) -> dict[str, object]:
        episode_start = time.perf_counter()
        verbose_print(verbose, "-" * 72)
        verbose_print(
            verbose,
            f"[{index}/{total}] instruction_file={display_path(instruction_file)}",
        )
        instruction = load_instruction(instruction_file)
        map_id = map_id_from_instruction_path(instruction_file)
        scene_id = scene_id_from_instruction(instruction, map_id)
        instruction_id = instruction_id_from_payload(instruction_file, instruction)
        output_path = output_path_for_prediction(output_root, map_id, instruction_id)
        verbose_print(
            verbose,
            f"map_id={map_id} scene_id={scene_id} instruction_id={instruction_id}",
        )
        verbose_print(verbose, f"output_path={display_path(output_path)}")
        verbose_print(verbose, f"instruction={instruction.get('instruction', '')}")

        if output_path.exists() and not overwrite:
            verbose_print(verbose, "existing prediction found; skipping")
            result = load_existing_result(output_path, map_id, instruction_id)
            result["scene_id"] = scene_id
        else:
            try:
                method_start = time.perf_counter()
                verbose_print(verbose, "loading map_state")
                map_state = load_map_state(map_id)
                verbose_print(
                    verbose,
                    f"map_key={map_state.get('map_key', map_id)} grid_size={map_state.get('grid_size')}",
                )
                verbose_print(verbose, "building traversable grid")
                traversable = build_traversable_grid(map_state)
                traversable_count = sum(1 for row in traversable for cell in row if cell)
                total_cells = sum(len(row) for row in traversable)
                verbose_print(
                    verbose,
                    f"traversable_cells={traversable_count}/{total_cells}",
                )
                start = start_cell_from_instruction(map_state, traversable, instruction)
                if start is None:
                    raise ValueError(f"No traversable start for {map_id}/{instruction_id}")
                verbose_print(verbose, f"start_cell={start}")
                verbose_print(verbose, "building map symbols")
                symbols = build_map_symbols(map_state)
                verbose_print(
                    verbose,
                    f"room_entities={len(symbols.room_entities)} object_entities={len(symbols.object_entities)}",
                )
                inventory = prompt_inventory(
                    symbols,
                    max_objects=max_prompt_objects,
                )
                verbose_print(
                    verbose,
                    "inventory "
                    f"rooms={len(inventory.get('rooms', []))} "
                    f"objects_prompt={inventory.get('object_prompt_count')} "
                    f"objects_total={inventory.get('object_total_count')}",
                )
                llm_cache_path = output_path_for_prediction(
                    llm_cache_root,
                    map_id,
                    instruction_id,
                )
                verbose_print(verbose, f"translating instruction cache={display_path(llm_cache_path)}")
                translation = translate_instruction_to_ltl(
                    instruction=str(instruction.get("instruction", "")),
                    inventory=inventory,
                    model=model,
                    cache_path=llm_cache_path,
                    overwrite_cache=overwrite_llm_cache,
                    verbose=verbose,
                )
                formula = translation.get("formula")
                if formula is None:
                    raise ValueError("LTLCodeGen translation did not return a formula.")
                verbose_print(verbose, f"ltl_formula={translation.get('ltl_formula')}")
                verbose_print(
                    verbose,
                    "planning product graph"
                    if planning_mode == "vanilla"
                    else "planning fast direct A*",
                )
                trajectory, planner_details, automaton_summary = build_ltlcodegen_trajectory(
                    map_state,
                    traversable,
                    symbols,
                    formula,  # type: ignore[arg-type]
                    start,
                    object_radius=object_label_radius,
                    planning_mode=planning_mode,
                    object_mode=object_mode,
                    verbose=verbose,
                )
                verbose_print(
                    verbose,
                    f"trajectory_length={len(trajectory)} automaton_summary={automaton_summary}",
                )
                verbose_print(verbose, f"planner_details={planner_details}")
                verbose_print(verbose, "evaluating trajectory")
                method_seconds = time.perf_counter() - method_start
                metrics = evaluate_prediction(
                    trajectory,
                    map_id=map_id,
                    instruction_id=instruction_id,
                )
                verbose_print(verbose, f"metrics_summary={metric_summary(metrics)}")
                record = build_prediction_record(
                    instruction_file=instruction_file,
                    map_id=map_id,
                    scene_id=scene_id,
                    instruction_id=instruction_id,
                    instruction=instruction,
                    trajectory=trajectory,
                    planner_details=planner_details,
                    translation=translation,
                    automaton_summary=automaton_summary,
                    map_inventory=inventory,
                    metrics=metrics,
                    planning_mode=planning_mode,
                    object_mode=object_mode,
                )
                verbose_print(verbose, "writing trajectory image")
                save_trajectory_image_for_record(
                    record,
                    map_state,
                    trajectory,
                    output_path,
                    nested_record_keys=("ltlcodegen",),
                    instruction=instruction,
                )
                verbose_print(verbose, f"trajectory_image={record.get('trajectory_image')}")
                runtime_seconds = time.perf_counter() - episode_start
                record["runtime_seconds"] = runtime_seconds
                record["runtime"] = {
                    "seconds": runtime_seconds,
                    "method_seconds": method_seconds,
                    "status": "written",
                }
                ltlcodegen_metadata = record.get("ltlcodegen")
                if isinstance(ltlcodegen_metadata, dict):
                    ltlcodegen_metadata["runtime_seconds"] = runtime_seconds
                verbose_print(verbose, "writing prediction JSON")
                write_prediction_record(output_path, record)
                verbose_print(verbose, f"wrote {display_path(output_path)}")
                result = result_from_prediction_record(record, output_path)
            except Exception as exc:  # pragma: no cover - defensive runner guard
                runtime_seconds = time.perf_counter() - episode_start
                error = f"{type(exc).__name__}: {exc}"
                verbose_print(verbose, f"episode failed: {error}")
                result = {
                    "instruction_id": instruction_id,
                    "map_id": map_id,
                    "scene_id": scene_id,
                    "status": "failed",
                    "path": display_path(output_path),
                    "difficulty_level": difficulty_from_instruction(instruction),
                    "runtime_seconds": runtime_seconds,
                    "runtime": {"seconds": runtime_seconds, "status": "failed"},
                    "error": error,
                }

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
        if result.get("status") == "skipped_exists":
            verbose_print(verbose, "existing prediction reused")
        else:
            verbose_print(
                verbose,
                f"episode_runtime_seconds={float(result.get('runtime_seconds', 0.0)):.3f}",
            )
        return result

    for index, result in run_instruction_jobs(
        instruction_files,
        workers=workers,
        run_one=run_one,
    ):
        retry_count = 0
        while result.get("status") == "failed" and retry_count < max_episode_retries:
            print_episode_result(index, total, result)
            retry_count += 1
            print(
                f"[{index}/{total}] scene_id={result.get('scene_id', result.get('map_id', 'unknown'))} "
                f"instruction_id={result.get('instruction_id', 'unknown')} "
                f"retrying after error ({retry_count}/{max_episode_retries})",
                flush=True,
            )
            result = run_one(index, instruction_files[index - 1])
        if retry_count:
            result["episode_retry_count"] = retry_count
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
        description="Generate Gemini/LTLCodeGen trajectories for SemPathBench instructions."
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
    parser.add_argument(
        "--llm-cache-root",
        type=Path,
        default=METHOD_ROOT / "_llm_cache",
        help="Cache Gemini translations so repeated runs do not re-call the API.",
    )
    parser.add_argument(
        "--model",
        default=configured_model(),
        help=(
            "Model used for LTLCodeGen translation. Defaults to MODEL in "
            "scripts/methods/api_key.py, then GEMINI_MODEL."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-llm-cache",
        action="store_true",
        help="Force fresh Gemini calls even when cached translations exist.",
    )
    parser.add_argument(
        "--max-prompt-objects",
        type=int,
        default=120,
        help="Maximum simple object AP entries included in each Gemini prompt.",
    )
    parser.add_argument(
        "--object-label-radius",
        type=int,
        default=20,
        help="Grid-cell radius where an object AP is true in the label map.",
    )
    parser.add_argument(
        "--planning-mode",
        "--planning_mode",
        choices=PLANNING_MODES,
        default="vanilla",
        help=(
            "Planning mode. 'vanilla' uses product graph A*; "
            "'fast' visits eventual AP targets with direct grid A*."
        ),
    )
    parser.add_argument(
        "--mode",
        dest="planning_mode",
        choices=PLANNING_MODES,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--object-mode",
        "--object_mode",
        type=normalize_object_mode,
        default="vanilla",
        metavar="{vanilla,all}",
        help=(
            "Object target mode. 'vanilla' uses the selected object AP only; "
            "'all' expands an object AP to every same-category object instance. "
            "The typo 'vinalla' is accepted as 'vanilla'."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Deprecated: metrics are always computed with the shared method evaluator.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Number of independent instruction episodes to run concurrently.")
    parser.add_argument(
        "--max-episode-retries",
        type=int,
        default=1,
        help=(
            "Retry an episode this many times after an unexpected exception "
            "(default: 1; set 0 to disable). Normal no-accepting-path results are not retried."
        ),
    )
    parser.add_argument(
        "--verbose",
        "--vebose",
        dest="verbose",
        action="store_true",
        help="Print detailed progress logs. '--vebose' is accepted as an alias.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_all(
        args.input_root,
        args.output_root,
        overwrite=args.overwrite,
        model=args.model,
        llm_cache_root=args.llm_cache_root,
        overwrite_llm_cache=args.overwrite_llm_cache,
        max_prompt_objects=args.max_prompt_objects,
        object_label_radius=args.object_label_radius,
        instruction_set=args.instruction_set,
        planning_mode=args.planning_mode,
        object_mode=args.object_mode,
        limit=args.limit,
        verbose=args.verbose,
        workers=args.workers,
        max_episode_retries=args.max_episode_retries,
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
