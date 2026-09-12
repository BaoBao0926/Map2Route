#!/usr/bin/env python3
"""Generate a simple A* tutorial trajectory from hard constraints."""

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
from scripts.methods.tutorial.utils import (
    collect_prediction_results,
    connect_waypoints_with_astar,
    difficulty_from_instruction,
    display_path,
    iso_now,
    load_existing_result,
    metric_summary,
    output_path_for_prediction,
    print_episode_result,
    print_summary,
    result_from_prediction_record,
    tutorial_waypoints,
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


METHOD_NAME = "tutorial"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / METHOD_NAME


def scene_id_from_instruction(
    instruction: Mapping[str, object],
    map_id: str,
) -> str:
    scene_id = instruction.get("map_id")
    if isinstance(scene_id, str) and scene_id.strip():
        return scene_id.strip()
    return map_id


def build_tutorial_trajectory(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
) -> tuple[list[list[int]], list[dict[str, object]], dict[str, float]]:
    """
    A tutorial method
    Pick hard-constraint waypoints and connect them with A*.
    """
    trajectory_start = time.perf_counter()
    traversable = build_traversable_grid(map_state)
    waypoints, waypoint_details, segment_transitions = tutorial_waypoints(
        map_state,
        traversable,
        instruction,
    )
    waypoint_selection_seconds = time.perf_counter() - trajectory_start
    astar_start = time.perf_counter()
    trajectory, segment_details = connect_waypoints_with_astar(
        traversable,
        waypoints,
        [
            constraint
            for constraint in instruction.get("hard_constraints", [])
            if isinstance(constraint, Mapping)
        ],
        segment_transitions,
    )
    astar_seconds = time.perf_counter() - astar_start
    return trajectory, waypoint_details + segment_details, {
        "waypoint_selection_seconds": waypoint_selection_seconds,
        "astar_seconds": astar_seconds,
        "trajectory_construction_seconds": waypoint_selection_seconds + astar_seconds,
    }


def build_prediction_record(
    instruction_file: Path,
    *,
    map_id: str,
    scene_id: str,
    instruction_id: str,
    instruction: Mapping[str, object],
    trajectory: list[list[int]],
    waypoint_details: list[dict[str, object]],
    metrics: dict[str, object] | None,
) -> dict[str, object]:
    difficulty_level = difficulty_from_instruction(instruction)
    metrics_summary = metric_summary(metrics)
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
        "metrics_summary": metrics_summary,
        "tutorial": {
            "description": "A simple teaching baseline that uses segment-aware A* to reach the first legal cell of each ordered hard target region while avoiding the must-avoid regions active for that transition.",
            "metrics_note": "Metrics were computed with the shared metric evaluator.",
            "waypoints": waypoint_details,
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
    resume: bool = False,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
    limit: int | None = None,
    offset: int = 0,
    workers: int = 1,
    save_visualizations: bool = True,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    instruction_files = iter_instruction_files(
        input_root,
        instruction_set=instruction_set,
    )
    if offset:
        instruction_files = instruction_files[offset:]
    if limit is not None:
        instruction_files = instruction_files[:limit]

    # A real resume first reconstructs the result set from disk, then submits
    # only missing episodes.  This avoids both needless worker jobs and a
    # partial-run summary that forgets predictions saved by earlier runs.
    pending_instruction_files: list[Path] = []
    if resume:
        for instruction_file in instruction_files:
            instruction = load_instruction(instruction_file)
            map_id = map_id_from_instruction_path(instruction_file)
            scene_id = scene_id_from_instruction(instruction, map_id)
            instruction_id = instruction_id_from_payload(instruction_file, instruction)
            output_path = output_path_for_prediction(output_root, map_id, instruction_id)
            if output_path.exists():
                result = load_existing_result(output_path, map_id, instruction_id)
                result["scene_id"] = scene_id
                results.append(result)
            else:
                pending_instruction_files.append(instruction_file)
        print(
            f"resume: reused={len(results)} pending={len(pending_instruction_files)}",
            flush=True,
        )
        # Immediately repair the aggregate summary, including prior outputs.
    else:
        pending_instruction_files = instruction_files

    total = len(pending_instruction_files)
    def run_one(index: int, instruction_file: Path) -> dict[str, object]:
        episode_start = time.perf_counter()
        instruction = load_instruction(instruction_file)
        map_id = map_id_from_instruction_path(instruction_file)
        scene_id = scene_id_from_instruction(instruction, map_id)
        instruction_id = instruction_id_from_payload(instruction_file, instruction)
        output_path = output_path_for_prediction(output_root, map_id, instruction_id)

        if output_path.exists() and resume:
            # Resume is opt-in: without it, existing predictions are regenerated.
            result = load_existing_result(output_path, map_id, instruction_id)
            result["scene_id"] = scene_id
        else:
            # main logic
            # get instruction info
            map_loading_start = time.perf_counter()
            map_state = load_map_state(map_id)
            map_loading_seconds = time.perf_counter() - map_loading_start
            # method logic
            trajectory, waypoint_details, trajectory_timing = build_tutorial_trajectory(
                map_state,
                instruction,
            )
            # get metrics
            metric_start = time.perf_counter()
            metrics = evaluate_prediction(
                trajectory,
                map_id=map_id,
                instruction_id=instruction_id,
                instruction_file=instruction_file,
            )
            metric_seconds = time.perf_counter() - metric_start
            record = build_prediction_record(
                instruction_file=instruction_file,
                map_id=map_id,
                instruction_id=instruction_id,
                scene_id=scene_id,
                instruction=instruction,
                trajectory=trajectory,
                waypoint_details=waypoint_details,
                metrics=metrics,
            )
            visualization_seconds = 0.0
            if save_visualizations:
                visualization_start = time.perf_counter()
                save_trajectory_image_for_record(
                    record,
                    map_state,
                    trajectory,
                    output_path,
                    nested_record_keys=("tutorial",),
                    instruction=instruction,
                )
                visualization_seconds = time.perf_counter() - visualization_start
            runtime_seconds = time.perf_counter() - episode_start
            breakdown_seconds = {
                "map_loading_seconds": map_loading_seconds,
                **trajectory_timing,
                "metric_seconds": metric_seconds,
                "visualization_seconds": visualization_seconds,
            }
            record["runtime_seconds"] = runtime_seconds
            record["runtime"] = {
                "seconds": runtime_seconds,
                "status": "written",
                "method_seconds": trajectory_timing["trajectory_construction_seconds"],
                "breakdown_seconds": breakdown_seconds,
            }
            tutorial_metadata = record.get("tutorial")
            if isinstance(tutorial_metadata, dict):
                tutorial_metadata["runtime_seconds"] = runtime_seconds
                tutorial_metadata["timing_seconds"] = breakdown_seconds
            write_prediction_record(output_path, record)
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
        pending_instruction_files,
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
        description="Generate tutorial A* trajectories for SemPathBench instructions."
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
        "--resume",
        action="store_true",
        help="Reuse existing prediction files. Without this flag, regenerate all selected episodes.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Deprecated: metrics are always computed with the shared method evaluator.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent instruction episodes to run concurrently.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many selected instruction files before applying --limit.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Do not render trajectory images; trajectory and metrics are unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_all(
        args.input_root,
        args.output_root,
        resume=args.resume,
        instruction_set=args.instruction_set,
        limit=args.limit,
        offset=args.offset,
        workers=args.workers,
        save_visualizations=not args.skip_visualization,
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
