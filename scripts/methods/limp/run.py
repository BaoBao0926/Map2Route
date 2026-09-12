#!/usr/bin/env python3
"""Run the SemPathBench LIMP pipeline."""

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
from scripts.methods.limp.config import (
    DEFAULT_AVOID_RADIUS,
    DEFAULT_MAX_GOAL_CANDIDATES,
    DEFAULT_MAX_PROGRESS_STEPS,
    DEFAULT_NEAR_RADIUS,
    METHOD_NAME,
    LimpConfig,
)
from scripts.methods.limp.language.client import configured_model
from scripts.methods.limp.pipeline import run_episode
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


LIMP_OUTPUT_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / "LIMP"
TRANSLATION_MODES = ("llm", "heuristic")
GROUNDING_MODES = ("core", "extended", "oracle")
TRANSLATION_CONTEXTS = ("none", "categories")
AUTOMATON_BACKENDS = ("ltlf", "spot", "residual-debug")
METHOD_VARIANTS = ("faithful", "extended")


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
    limp_metadata: dict[str, object],
    metrics: dict[str, object] | None,
) -> dict[str, object]:
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
        "limp": limp_metadata,
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
    method_variant: str,
    translation_mode: str,
    grounding_mode: str,
    translation_context: str,
    automaton_backend: str,
    allow_residual_fallback: bool,
    near_radius: int,
    avoid_radius: int,
    max_progress_steps: int,
    max_goal_candidates: int,
    strict_soft: bool,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
    limit: int | None = None,
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

    total = len(instruction_files)
    config = LimpConfig(
        model=model,
        llm_cache_root=llm_cache_root,
        overwrite_llm_cache=overwrite_llm_cache,
        method_variant=method_variant,
        translation_mode=translation_mode,
        grounding_mode=grounding_mode,
        translation_context=translation_context,
        automaton_backend=automaton_backend,
        allow_residual_fallback=allow_residual_fallback,
        near_radius=near_radius,
        avoid_radius=avoid_radius,
        max_progress_steps=max_progress_steps,
        max_goal_candidates=max_goal_candidates,
        strict_soft=strict_soft,
        verbose=verbose,
    )

    def run_one(index: int, instruction_file: Path) -> dict[str, object]:
        episode_start = time.perf_counter()
        instruction = load_instruction(instruction_file)
        map_id = map_id_from_instruction_path(instruction_file)
        scene_id = scene_id_from_instruction(instruction, map_id)
        instruction_id = instruction_id_from_payload(instruction_file, instruction)
        output_path = output_path_for_prediction(output_root, map_id, instruction_id)

        if output_path.exists() and not overwrite:
            result = load_existing_result(output_path, map_id, instruction_id)
            result["scene_id"] = scene_id
        else:
            method_start = time.perf_counter()
            map_state = load_map_state(map_id)
            episode = run_episode(instruction, map_state, config)
            method_seconds = time.perf_counter() - method_start
            metrics = evaluate_prediction(
                episode.trajectory,
                map_id=map_id,
                instruction_id=instruction_id,
            )
            record = build_prediction_record(
                instruction_file=instruction_file,
                map_id=map_id,
                scene_id=scene_id,
                instruction_id=instruction_id,
                instruction=instruction,
                trajectory=episode.trajectory,
                limp_metadata=episode.metadata,
                metrics=metrics,
            )
            save_trajectory_image_for_record(
                record,
                map_state,
                episode.trajectory,
                output_path,
                nested_record_keys=("limp",),
                instruction=instruction,
            )
            runtime_seconds = time.perf_counter() - episode_start
            record["runtime_seconds"] = runtime_seconds
            record["runtime"] = {
                "seconds": runtime_seconds,
                "method_seconds": method_seconds,
                "status": "written",
            }
            limp_meta = record.get("limp")
            if isinstance(limp_meta, dict):
                limp_meta["runtime_seconds"] = runtime_seconds
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
        description="Run LIMP on SemPathBench instructions."
    )
    parser.add_argument("--input-root", type=Path, default=INSTRUCTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=LIMP_OUTPUT_ROOT)
    parser.add_argument(
        "--set",
        dest="instruction_set",
        choices=INSTRUCTION_SET_CHOICES,
        default=DEFAULT_INSTRUCTION_SET,
    )
    parser.add_argument("--model", default=configured_model())
    parser.add_argument(
        "--llm-cache-root",
        type=Path,
        default=LIMP_OUTPUT_ROOT / "_llm_cache",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-llm-cache", action="store_true")
    parser.add_argument(
        "--method-variant",
        choices=METHOD_VARIANTS,
        default="extended",
        help="Use official-aligned grounding/prompt behavior or extended heuristics.",
    )
    parser.add_argument(
        "--translation-mode",
        choices=TRANSLATION_MODES,
        default="llm",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=GROUNDING_MODES,
        default="extended",
    )
    parser.add_argument(
        "--translation-context",
        choices=TRANSLATION_CONTEXTS,
        default="none",
    )
    parser.add_argument(
        "--automaton-backend",
        choices=AUTOMATON_BACKENDS,
        default="ltlf",
        help="Use local LTLf residual-formula progression by default; 'residual-debug' is smoke-test only.",
    )
    parser.add_argument(
        "--allow-residual-fallback",
        action="store_true",
        help="Debug only: allow residual_fallback when Spot/formal backend is unavailable.",
    )
    parser.add_argument("--near-radius", type=int, default=DEFAULT_NEAR_RADIUS)
    parser.add_argument("--avoid-radius", type=int, default=DEFAULT_AVOID_RADIUS)
    parser.add_argument(
        "--max-progress-steps",
        type=int,
        default=DEFAULT_MAX_PROGRESS_STEPS,
    )
    parser.add_argument(
        "--max-goal-candidates",
        type=int,
        default=DEFAULT_MAX_GOAL_CANDIDATES,
    )
    parser.add_argument("--strict-soft", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of independent instruction episodes to run concurrently.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Deprecated: metrics are always computed with the shared method evaluator.",
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
        method_variant=args.method_variant,
        translation_mode=args.translation_mode,
        grounding_mode=args.grounding_mode,
        translation_context=args.translation_context,
        automaton_backend=args.automaton_backend,
        allow_residual_fallback=args.allow_residual_fallback,
        near_radius=args.near_radius,
        avoid_radius=args.avoid_radius,
        max_progress_steps=args.max_progress_steps,
        max_goal_candidates=args.max_goal_candidates,
        strict_soft=args.strict_soft,
        instruction_set=args.instruction_set,
        limit=args.limit,
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
