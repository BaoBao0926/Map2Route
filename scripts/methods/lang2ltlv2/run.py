#!/usr/bin/env python3
"""Run the SemPathBench Lang2LTL-2 pipeline."""

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
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state, normalize_map_key
from scripts.methods.lang2ltl.map_features import prompt_inventory
from scripts.methods.lang2ltl.planner import start_cell_from_instruction
from scripts.methods.lang2ltlv2.lt_checkpoint import (
    DEFAULT_LT_CHECKPOINT_URL,
    ensure_lt_checkpoint,
)
from scripts.methods.lang2ltlv2.planner import PLANNING_MODES, build_lang2ltl2_trajectory
from scripts.methods.lang2ltlv2.sempath_bridge import write_lang2ltl2_map_cache
from scripts.methods.lang2ltlv2.translator import (
    configured_embedding_model,
    configured_model,
    run_original_lang2ltl2_pipeline,
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
    write_prediction_record,
    write_summary,
)
from scripts.methods.util.grid_astar import build_traversable_grid
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


METHOD_NAME = "Lang2LTL2"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / METHOD_NAME


def verbose_print(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[lang2ltlv2] {message}", flush=True)


def progress_print(message: str) -> None:
    print(f"[lang2ltlv2] {message}", flush=True)


def scene_id_from_instruction(
    instruction: Mapping[str, object],
    map_id: str,
) -> str:
    scene_id = instruction.get("map_id")
    if isinstance(scene_id, str) and scene_id.strip():
        return scene_id.strip()
    return map_id


def _planner_description(planning_mode: str) -> str:
    if planning_mode == "vanilla":
        return (
            "an AP-MDP-style planner searches over abstract AP states with LTL "
            "progression, then realizes AP actions with grid A*."
        )
    return (
        "a fast debug planner extracts positive eventual APs from grounded LTL "
        "and connects them with sequential grid A*."
    )


def build_prediction_record(
    instruction_file: Path,
    *,
    map_id: str,
    scene_id: str,
    instruction_id: str,
    instruction: Mapping[str, object],
    trajectory: list[list[int]],
    planner_details: list[dict[str, object]],
    planner_summary: dict[str, object],
    translation_metadata: dict[str, object],
    bridge_metadata: dict[str, object],
    map_inventory: dict[str, object],
    metrics: dict[str, object] | None,
    planning_mode: str,
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
        "lang2ltlv2": {
            "description": (
                "Lang2LTL-2 SemPathBench adapter: original SRER/REG/SPG/LT are "
                "called through a pseudo SemPathBench environment bridge, and "
                f"{_planner_description(planning_mode)}"
            ),
            "input_contract": (
                "Planner input is restricted to instruction text, start_pose, map "
                "layers, and map-derived room/object entities. Annotation fields "
                "such as hard_constraints, soft_constraints, objects, and "
                "human_expert_trajectory are not used for planning."
            ),
            "visual_branch_enabled": False,
            "text_only": True,
            "planning_mode": planning_mode,
            "planner": planner_summary,
            "planner_details": planner_details,
            "translation": translation_metadata,
            "bridge": bridge_metadata,
            "map_inventory": {
                "room_category_counts": map_inventory.get("room_category_counts"),
                "object_category_counts": map_inventory.get("object_category_counts"),
                "object_prompt_count": map_inventory.get("object_prompt_count"),
                "object_total_count": map_inventory.get("object_total_count"),
            },
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
    cache_root: Path,
    overwrite_cache: bool,
    topk_groundings: int,
    object_label_radius: int,
    lt_model_path: Path,
    download_lt_checkpoint: bool,
    lt_checkpoint_url: str,
    model: str,
    embedding_model: str,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
    planning_mode: str = "vanilla",
    limit: int | None = None,
    verbose: bool = False,
    workers: int = 1,
) -> list[dict[str, object]]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    cache_root = cache_root.resolve()
    verbose_print(verbose, f"input_root={display_path(input_root)}")
    verbose_print(verbose, f"output_root={display_path(output_root)}")
    verbose_print(verbose, f"cache_root={display_path(cache_root)}")
    verbose_print(verbose, f"instruction_set={instruction_set}")
    verbose_print(verbose, f"planning_mode={planning_mode}")
    verbose_print(verbose, f"overwrite={overwrite}")
    verbose_print(verbose, f"overwrite_cache={overwrite_cache}")
    verbose_print(verbose, f"topk_groundings={topk_groundings}")
    verbose_print(verbose, f"object_label_radius={object_label_radius}")
    verbose_print(verbose, f"lt_model_path={display_path(lt_model_path)}")
    verbose_print(verbose, f"download_lt_checkpoint={download_lt_checkpoint}")
    verbose_print(verbose, f"model={model}")
    verbose_print(verbose, f"embedding_model={embedding_model}")

    lt_checkpoint_info: dict[str, object] = {}
    try:
        lt_model_path, lt_checkpoint_info = ensure_lt_checkpoint(
            lt_model_path,
            download=download_lt_checkpoint,
            url=lt_checkpoint_url,
            verbose=verbose,
        )
        verbose_print(verbose, f"lt_checkpoint_status={lt_checkpoint_info.get('status')}")
    except FileNotFoundError as exc:
        lt_checkpoint_info = {
            "status": "missing",
            "reason": str(exc),
            "checkpoint_path": str(lt_model_path),
            "download_requested": download_lt_checkpoint,
            "download_url": lt_checkpoint_url,
        }
        verbose_print(verbose, f"lt_checkpoint_missing={exc}")

    results: list[dict[str, object]] = []
    instruction_files = iter_instruction_files(
        input_root,
        instruction_set=instruction_set,
    )
    if limit is not None:
        instruction_files = instruction_files[:limit]
    verbose_print(verbose, f"matched_instruction_count={len(instruction_files)}")
    progress_print(
        "start "
        f"set={instruction_set} total={len(instruction_files)} "
        f"planning_mode={planning_mode} output={display_path(output_root)}"
    )

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
        instruction_text = str(instruction.get("instruction", ""))
        verbose_print(verbose, f"instruction={instruction_text}")

        if output_path.exists() and not overwrite:
            verbose_print(verbose, "existing prediction found; skipping")
            result = load_existing_result(output_path, map_id, instruction_id)
            result["scene_id"] = scene_id
        else:
            method_start = time.perf_counter()
            verbose_print(verbose, f"[{index}/{total}] loading map")
            map_state = load_map_state(map_id)
            traversable = build_traversable_grid(map_state)
            start = start_cell_from_instruction(map_state, traversable, instruction)
            if start is None:
                raise ValueError(f"No traversable start for {map_id}/{instruction_id}")
            verbose_print(verbose, f"start_cell={start}")

            map_cache_key = normalize_map_key(map_id)
            map_cache_root = cache_root / "maps"
            verbose_print(verbose, f"[{index}/{total}] writing bridge cache")
            bridge = write_lang2ltl2_map_cache(
                map_id=map_cache_key,
                map_state=map_state,
                start_cell=start,
                cache_root=map_cache_root,
                overwrite=overwrite_cache,
            )
            inventory = prompt_inventory(bridge.symbols)
            valid_aps = set(bridge.ap_metadata)

            translation_cache_path = (
                cache_root
                / "translations"
                / map_cache_key
                / f"{instruction_id}.json"
            )
            verbose_print(
                verbose,
                f"[{index}/{total}] running original Lang2LTL-2 modules cache={display_path(translation_cache_path)}",
            )
            translation = run_original_lang2ltl2_pipeline(
                instruction_text=instruction_text,
                graph_dpath=bridge.graph_dpath,
                osm_fpath=bridge.osm_fpath,
                valid_aps=valid_aps,
                cache_path=translation_cache_path,
                overwrite_cache=overwrite_cache,
                topk_groundings=topk_groundings,
                text_only=True,
                model=model,
                embedding_model=embedding_model,
                lt_model_path=lt_model_path,
                rel_embeds_path=cache_root / "known_rel_embeds.json",
                reg_query_cache_path=cache_root / "reg_query_cache.pkl",
                symbols=bridge.symbols,
                verbose=verbose,
            )
            translation_metadata = translation.to_metadata()
            translation_metadata["lt_checkpoint"] = lt_checkpoint_info
            verbose_print(
                verbose,
                f"grounded_ltl={translation.grounded_ltl} failures={translation.failures}",
            )

            verbose_print(verbose, f"[{index}/{total}] planning trajectory")
            trajectory, planner_details, planner_summary = build_lang2ltl2_trajectory(
                map_state=map_state,
                traversable=traversable,
                symbols=bridge.symbols,
                grounded_ltl=translation.grounded_ltl,
                start=start,
                target_sequence=translation.target_sequence,
                object_radius=object_label_radius,
                planning_mode=planning_mode,
                verbose=verbose,
            )
            verbose_print(
                verbose,
                f"trajectory_length={len(trajectory)} planner_summary={planner_summary}",
            )

            verbose_print(verbose, f"[{index}/{total}] evaluating trajectory")
            method_seconds = time.perf_counter() - method_start
            metrics = evaluate_prediction(
                trajectory,
                map_id=map_id,
                instruction_id=instruction_id,
            )
            record = build_prediction_record(
                instruction_file=instruction_file,
                map_id=map_id,
                scene_id=scene_id,
                instruction_id=instruction_id,
                instruction=instruction,
                trajectory=trajectory,
                planner_details=planner_details,
                planner_summary=planner_summary,
                translation_metadata=translation_metadata,
                bridge_metadata=bridge.bridge_metadata,
                map_inventory=inventory,
                metrics=metrics,
                planning_mode=planning_mode,
            )
            save_trajectory_image_for_record(
                record,
                map_state,
                trajectory,
                output_path,
                nested_record_keys=("lang2ltlv2",),
                instruction=instruction,
            )
            runtime_seconds = time.perf_counter() - episode_start
            record["runtime_seconds"] = runtime_seconds
            record["runtime"] = {
                "seconds": runtime_seconds,
                "method_seconds": method_seconds,
                "status": "written",
            }
            metadata = record.get("lang2ltlv2")
            if isinstance(metadata, dict):
                metadata["runtime_seconds"] = runtime_seconds
            write_prediction_record(output_path, record)
            verbose_print(verbose, f"wrote {display_path(output_path)}")
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
        description="Generate Lang2LTL-2 trajectories for SemPathBench instructions."
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
        "--cache-root",
        type=Path,
        default=METHOD_ROOT / "_sembench_cache",
        help="Cache pseudo Lang2LTL-2 map files and module outputs.",
    )
    parser.add_argument(
        "--lt-model-path",
        type=Path,
        default=Path.home() / "ground" / "models" / "checkpoint-best",
        help="Path to the original Lang2LTL-2 fine-tuned LT checkpoint.",
    )
    parser.add_argument(
        "--download-lt-checkpoint",
        action="store_true",
        help=(
            "Download the original Lang2LTL-2 fine-tuned T5 checkpoint from "
            "Google Drive if --lt-model-path is missing. This may install gdown."
        ),
    )
    parser.add_argument(
        "--lt-checkpoint-url",
        default=DEFAULT_LT_CHECKPOINT_URL,
        help="Google Drive folder URL for the Lang2LTL-2 LT checkpoint.",
    )
    parser.add_argument(
        "--model",
        default=configured_model(),
        help=(
            "Gemini model used to emulate original Lang2LTL-2 OpenAI chat calls. "
            "Defaults to MODEL in scripts/methods/api_key.py, then GEMINI_MODEL."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        "--embedding_model",
        default=configured_embedding_model(),
        help=(
            "Gemini embedding model used to emulate original Lang2LTL-2 OpenAI "
            "embedding calls. Defaults to EMBEDDING_MODEL in api_key.py, then "
            "GEMINI_EMBEDDING_MODEL."
        ),
    )
    parser.add_argument(
        "--planner",
        "--planning-mode",
        "--planning_mode",
        dest="planning_mode",
        choices=PLANNING_MODES,
        default="fast",
        help=(
            "'vanilla' uses the AP-MDP-style planner; "
            "'fast' uses sequential A* as a debug approximation."
        ),
    )
    parser.add_argument(
        "--topk-groundings",
        "--topk_groundings",
        type=int,
        default=10,
        help="Top-k grounding candidates retained by the original REG/SPG modules.",
    )
    parser.add_argument(
        "--object-label-radius",
        type=int,
        default=20,
        help="Grid-cell radius where an object AP is true for fast debug planning.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Regenerate bridge/module caches.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Deprecated: metrics are always computed with the shared method evaluator.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Number of independent instruction episodes to run concurrently.")
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
    # Refresh the on-disk aggregate before starting new work. This makes a
    # resumed run immediately expose results from episodes completed during
    # earlier, interrupted runs.
    existing_results = collect_prediction_results(
        args.output_root,
        method_name=METHOD_NAME,
    )
    initial_summary_path = write_summary(
        existing_results,
        args.input_root,
        args.output_root,
        method_name=METHOD_NAME,
        evaluate=True,
    )
    print(f"refreshed existing summary: {initial_summary_path}", flush=True)
    results = build_all(
        args.input_root,
        args.output_root,
        overwrite=args.overwrite,
        cache_root=args.cache_root,
        overwrite_cache=args.overwrite_cache,
        topk_groundings=args.topk_groundings,
        object_label_radius=args.object_label_radius,
        lt_model_path=args.lt_model_path,
        download_lt_checkpoint=args.download_lt_checkpoint,
        lt_checkpoint_url=args.lt_checkpoint_url,
        model=args.model,
        embedding_model=args.embedding_model,
        instruction_set=args.instruction_set,
        planning_mode=args.planning_mode,
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
