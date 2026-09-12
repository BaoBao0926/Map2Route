#!/usr/bin/env python3
"""Run the SemPathBench Lang2LTL pipeline."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Mapping

# Some older Lang2LTL environments ship ``requests`` without either optional
# character-detection backend.  Requests emits this non-actionable warning on
# import; keep batch output reserved for episode failures and final metrics.
warnings.filterwarnings(
    "ignore",
    message="Unable to find acceptable character detection dependency.*",
    category=Warning,
    module=r"requests",
)

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluate_prediction import evaluate_prediction
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state, normalize_map_key
from scripts.methods.lang2ltl.map_features import build_map_symbols, prompt_inventory
from scripts.methods.lang2ltl.model_checkpoint import (
    DEFAULT_LANG2LTL_T5_CHECKPOINT_PATH,
    DEFAULT_LANG2LTL_T5_CHECKPOINT_URL,
    ensure_lang2ltl_t5_checkpoint,
)
from scripts.methods.lang2ltl.planner import (
    PLANNING_MODES,
    build_lang2ltl_trajectory,
    start_cell_from_instruction,
)
from scripts.methods.lang2ltl.translator import (
    GROUNDING_MODES,
    TRANSLATION_MODES,
    configured_embedding_model,
    configured_model,
    translate_instruction_to_ltl,
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
from scripts.methods.util.workers import run_instruction_jobs_grouped


METHOD_NAME = "Lang2LTL"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / METHOD_NAME
LLM_BACKEND = "gemini"
SYMBOLIC_TRANSLATOR = "t5"


def verbose_print(verbose: bool, message: str) -> None:
    """Keep batch-run output compact; detailed data is written to JSON records."""
    del verbose, message


def progress_print(message: str) -> None:
    del message


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
    translation_mode: str,
) -> dict[str, object]:
    if planning_mode == "vanilla":
        planner_description = (
            "an AP-MDP-style planner searches over abstract AP states with LTL "
            "progression, then realizes AP actions with grid A*."
        )
    elif planning_mode == "product":
        planner_description = (
            "a product-state planner searches the grid x LTL-progression graph."
        )
    else:
        planner_description = (
            "a fast plain A* planner visits eventual AP targets in formula order."
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
        "lang2ltl": {
            "description": (
                "Lang2LTL SemPathBench adapter: the method extracts referring "
                "expressions, grounds them to map-derived instance APs, translates "
                "the symbolic utterance into LTL, and "
                f"{planner_description}"
            ),
            "input_contract": (
                "Planner input is restricted to instruction text, start_pose, map "
                "layers, and map-derived room/object entities. Annotation fields "
                "such as hard_constraints, soft_constraints, objects, and "
                "human_expert_trajectory are not used for planning."
            ),
            "translation_mode": translation_mode,
            "actual_translation_mode": translation.get("mode"),
            "model": translation.get("model"),
            "llm_backend": translation.get("llm_backend", LLM_BACKEND),
            "symbolic_translator": translation.get("symbolic_translator", SYMBOLIC_TRANSLATOR),
            "symbolic_model_path": translation.get("symbolic_model_path"),
            "symbolic_checkpoint": translation.get("symbolic_checkpoint"),
            "planning_mode": planning_mode,
            "object_mode": "vanilla",
            "referring_expressions": translation.get("referring_expressions"),
            "rer_raw_response": translation.get("rer_raw_response"),
            "grounding": translation.get("grounding"),
            "grounding_selected": translation.get("grounding_selected"),
            "grounding_mode": translation.get("grounding_mode"),
            "grounding_backend": translation.get("grounding_backend"),
            "ap_semantic_descriptions": translation.get("ap_semantic_descriptions"),
            "grounded_utterance": translation.get("grounded_utterance"),
            "placeholder_map": translation.get("placeholder_map"),
            "symbolic_utterance": translation.get("symbolic_utterance"),
            "symbolic_ltl": translation.get("symbolic_ltl"),
            "symbolic_ltl_repair_attempts": translation.get("symbolic_ltl_repair_attempts"),
            "grounded_ltl": translation.get("grounded_ltl"),
            "formula_ast": translation.get("formula_ast"),
            "parse_warnings": translation.get("parse_warnings"),
            "symbolic_parse_fallback": translation.get("symbolic_parse_fallback"),
            "fallback_reason": translation.get("fallback_reason"),
            "fallback_error": translation.get("fallback_error"),
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


def build_failed_prediction_record(
    instruction_file: Path,
    *,
    map_id: str,
    scene_id: str,
    instruction_id: str,
    instruction: Mapping[str, object],
    planning_mode: str,
    translation_mode: str,
    error: str,
) -> dict[str, object]:
    """Persist an unexecutable episode as an explicit zero-score prediction.

    A method failure is part of the benchmark outcome, rather than an omitted
    sample. No trajectory is emitted and scores are zeroed instead of
    evaluating a synthetic route from the start location.
    """
    metrics: dict[str, object] = {
        "HCS": 0.0,
        "SCS": None,
        "PL": 0.0,
        "segment_wise_SPL": 0.0,
        "details": {
            "status": "method_failed_zero_score",
            "failure_reason": error,
            "trajectory_status": "empty_no_action",
            "hcs": {"score": 0.0, "status": "not_evaluated_method_failed"},
            "segments": [],
            "soft_constraints": [],
            "scs_status": "not_evaluated_method_failed",
        },
    }
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
        "status": "method_failed",
        "failure_reason": error,
        "metrics": metrics,
        "metrics_summary": metric_summary(metrics),
        "lang2ltl": {
            "translation_mode": translation_mode,
            "planning_mode": planning_mode,
            "outcome": "method_failed_zero_score",
            "failure_reason": error,
        },
        "instruction": {
            "text": instruction.get("instruction", ""),
            "difficulty_level": instruction.get("difficulty_level", ""),
            "template_instruction_id": instruction.get("template_instruction_id"),
        },
        "trajectory": [],
    }


def build_all(
    input_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
    model: str,
    llm_cache_root: Path,
    embedding_cache_root: Path,
    overwrite_llm_cache: bool,
    symbolic_model_path: Path,
    symbolic_checkpoint_url: str,
    max_prompt_objects: int,
    object_label_radius: int,
    topk_groundings: int,
    planner_progress_interval: int,
    grounding_mode: str,
    embedding_backend: str,
    embedding_model: str,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
    planning_mode: str = "vanilla",
    translation_mode: str = "auto",
    limit: int | None = None,
    verbose: bool = False,
    workers: int = 1,
) -> list[dict[str, object]]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    llm_cache_root = llm_cache_root.resolve()
    embedding_cache_root = embedding_cache_root.resolve()
    verbose_print(verbose, f"input_root={display_path(input_root)}")
    verbose_print(verbose, f"output_root={display_path(output_root)}")
    verbose_print(verbose, f"llm_cache_root={display_path(llm_cache_root)}")
    verbose_print(verbose, f"embedding_cache_root={display_path(embedding_cache_root)}")
    verbose_print(verbose, f"instruction_set={instruction_set}")
    verbose_print(verbose, f"llm_backend={LLM_BACKEND}")
    verbose_print(verbose, f"model={model}")
    verbose_print(verbose, f"symbolic_translator={SYMBOLIC_TRANSLATOR}")
    verbose_print(verbose, f"symbolic_model_path={display_path(symbolic_model_path)}")
    verbose_print(verbose, "auto_download_symbolic_checkpoint=True")
    verbose_print(verbose, f"translation_mode={translation_mode}")
    verbose_print(verbose, f"grounding_mode={grounding_mode}")
    verbose_print(verbose, f"embedding_backend={embedding_backend}")
    verbose_print(verbose, f"embedding_model={embedding_model}")
    verbose_print(verbose, f"planning_mode={planning_mode}")
    verbose_print(verbose, "object_mode=vanilla")
    verbose_print(verbose, f"overwrite={overwrite}")
    verbose_print(verbose, f"overwrite_llm_cache={overwrite_llm_cache}")
    verbose_print(verbose, f"max_prompt_objects={max_prompt_objects}")
    verbose_print(verbose, f"object_label_radius={object_label_radius}")
    verbose_print(verbose, f"topk_groundings={topk_groundings}")
    verbose_print(verbose, f"planner_progress_interval={planner_progress_interval}")

    symbolic_checkpoint_info: dict[str, object] | None = None
    if translation_mode != "heuristic":
        symbolic_model_path, symbolic_checkpoint_info = ensure_lang2ltl_t5_checkpoint(
            symbolic_model_path,
            download=True,
            url=symbolic_checkpoint_url,
            verbose=False,
        )
        verbose_print(
            verbose,
            "symbolic_checkpoint_status="
            f"{symbolic_checkpoint_info.get('status')} "
            f"path={display_path(symbolic_model_path)}",
        )

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
        f"translation_mode={translation_mode} llm_backend={LLM_BACKEND} "
        f"symbolic_translator={SYMBOLIC_TRANSLATOR} grounding_mode={grounding_mode} "
        f"planning_mode={planning_mode} "
        f"output={display_path(output_root)}"
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
        verbose_print(verbose, f"instruction={instruction.get('instruction', '')}")
        instruction_text = str(instruction.get("instruction", "")).strip()
        if len(instruction_text) > 180:
            instruction_text = f"{instruction_text[:177]}..."
        verbose_print(
            verbose,
            f"[{index}/{total}] {map_id}/{instruction_id} "
            f"instruction={instruction_text!r}"
        )

        if output_path.exists() and not overwrite:
            verbose_print(verbose, "existing prediction found; skipping")
            verbose_print(verbose, f"[{index}/{total}] skip existing {display_path(output_path)}")
            result = load_existing_result(output_path, map_id, instruction_id)
            result["scene_id"] = scene_id
        else:
            method_start = time.perf_counter()
            verbose_print(verbose, "loading map_state")
            verbose_print(verbose, f"[{index}/{total}] loading map")
            map_state = load_map_state(map_id)
            verbose_print(
                verbose,
                f"map_key={map_state.get('map_key', map_id)} grid_size={map_state.get('grid_size')}",
            )
            verbose_print(verbose, "building traversable grid")
            verbose_print(verbose, f"[{index}/{total}] building traversable grid")
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
            verbose_print(verbose, f"[{index}/{total}] building room/object AP inventory")
            symbols = build_map_symbols(map_state)
            verbose_print(
                verbose,
                f"room_entities={len(symbols.room_entities)} object_entities={len(symbols.object_entities)}",
            )
            inventory = prompt_inventory(symbols, max_objects=max_prompt_objects)
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
            embedding_cache_path = (
                embedding_cache_root / normalize_map_key(map_id) / "embeddings.json"
            )
            verbose_print(verbose, f"translating instruction cache={display_path(llm_cache_path)}")
            verbose_print(
                verbose,
                f"[{index}/{total}] translating "
                f"mode={translation_mode} grounding={grounding_mode} "
                f"llm_backend={LLM_BACKEND} model={model} "
                f"symbolic_translator={SYMBOLIC_TRANSLATOR} "
                f"cache={display_path(llm_cache_path)} "
                f"embedding_cache={display_path(embedding_cache_path)}"
            )
            translation = translate_instruction_to_ltl(
                instruction=str(instruction.get("instruction", "")),
                symbols=symbols,
                model=model,
                llm_backend=LLM_BACKEND,
                symbolic_translator=SYMBOLIC_TRANSLATOR,
                symbolic_model_path=symbolic_model_path,
                cache_path=llm_cache_path,
                embedding_cache_path=embedding_cache_path,
                overwrite_cache=overwrite_llm_cache,
                translation_mode=translation_mode,
                grounding_mode=grounding_mode,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                topk_groundings=topk_groundings,
                verbose=False,
            )
            if symbolic_checkpoint_info is not None:
                translation["symbolic_checkpoint"] = symbolic_checkpoint_info
            formula = translation.get("formula")
            if formula is None:
                raise ValueError("Lang2LTL translation did not return a formula.")
            verbose_print(verbose, f"grounded_ltl={translation.get('grounded_ltl')}")
            verbose_print(
                verbose,
                f"[{index}/{total}] translated "
                f"actual_mode={translation.get('mode')} "
                f"grounded_ltl={translation.get('grounded_ltl')}"
            )

            verbose_print(
                verbose,
                "planning AP-MDP"
                if planning_mode == "vanilla"
                else "planning product graph"
                if planning_mode == "product"
                else "planning fast direct A*",
            )
            if planning_mode == "vanilla":
                verbose_print(
                    verbose,
                    f"[{index}/{total}] vanilla planner searches AP-level "
                    "MDP states with LTL progression, then realizes each AP "
                    "action with grid A*."
                )
            verbose_print(verbose, f"[{index}/{total}] planning trajectory")
            trajectory, planner_details, automaton_summary = build_lang2ltl_trajectory(
                map_state,
                traversable,
                symbols,
                formula,  # type: ignore[arg-type]
                start,
                object_radius=object_label_radius,
                planning_mode=planning_mode,
                planner_progress_interval=planner_progress_interval,
                verbose=False,
            )
            verbose_print(
                verbose,
                f"trajectory_length={len(trajectory)} automaton_summary={automaton_summary}",
            )
            verbose_print(verbose, f"planner_details={planner_details}")
            verbose_print(verbose, f"[{index}/{total}] planned trajectory_length={len(trajectory)}")
            verbose_print(verbose, "evaluating trajectory")
            verbose_print(verbose, f"[{index}/{total}] evaluating trajectory")
            method_seconds = time.perf_counter() - method_start
            metrics = evaluate_prediction(
                trajectory,
                map_id=map_id,
                instruction_id=instruction_id,
            )
            verbose_print(verbose, f"metrics_summary={metric_summary(metrics)}")
            verbose_print(verbose, f"[{index}/{total}] metrics={metric_summary(metrics)}")
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
                translation_mode=translation_mode,
            )
            verbose_print(verbose, "writing trajectory image")
            verbose_print(verbose, f"[{index}/{total}] writing prediction")
            save_trajectory_image_for_record(
                record,
                map_state,
                trajectory,
                output_path,
                nested_record_keys=("lang2ltl",),
                instruction=instruction,
            )
            runtime_seconds = time.perf_counter() - episode_start
            record["runtime_seconds"] = runtime_seconds
            record["runtime"] = {
                "seconds": runtime_seconds,
                "method_seconds": method_seconds,
                "status": "written",
            }
            lang2ltl_metadata = record.get("lang2ltl")
            if isinstance(lang2ltl_metadata, dict):
                lang2ltl_metadata["runtime_seconds"] = runtime_seconds
            verbose_print(verbose, "writing prediction JSON")
            write_prediction_record(output_path, record)
            verbose_print(verbose, f"wrote {display_path(output_path)}")
            verbose_print(verbose, f"[{index}/{total}] wrote {display_path(output_path)}")
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
        verbose_print(
            verbose,
            f"episode_runtime_seconds={float(result.get('runtime_seconds', 0.0)):.3f}",
        )
        return result

    def run_one_safe(index: int, instruction_file: Path) -> dict[str, object]:
        """Convert one episode failure into a concise result and continue."""
        try:
            return run_one(index, instruction_file)
        except Exception as exc:
            try:
                instruction = load_instruction(instruction_file)
                map_id = map_id_from_instruction_path(instruction_file)
                instruction_id = instruction_id_from_payload(instruction_file, instruction)
                scene_id = scene_id_from_instruction(instruction, map_id)
                output_path = output_path_for_prediction(output_root, map_id, instruction_id)
                difficulty_level = difficulty_from_instruction(instruction)
            except Exception:
                map_id = "unknown"
                instruction_id = instruction_file.stem
                scene_id = map_id
                output_path = instruction_file
                difficulty_level = "unknown"
            error = f"{type(exc).__name__}: {exc}"
            print(
                f"[failed {index}/{total}] {map_id}/{instruction_id}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if map_id != "unknown":
                record = build_failed_prediction_record(
                    instruction_file,
                    map_id=map_id,
                    scene_id=scene_id,
                    instruction_id=instruction_id,
                    instruction=instruction,
                    planning_mode=planning_mode,
                    translation_mode=translation_mode,
                    error=error,
                )
                record["runtime_seconds"] = 0.0
                record["runtime"] = {"seconds": 0.0, "status": "method_failed"}
                write_prediction_record(output_path, record)
                result = result_from_prediction_record(record, output_path)
                result["status"] = "written"
                result["method_status"] = "method_failed"
                return result
            return {
                "instruction_id": instruction_id,
                "map_id": map_id,
                "scene_id": scene_id,
                "status": "failed",
                "path": display_path(output_path),
                "difficulty_level": difficulty_level,
                "runtime_seconds": 0.0,
                "runtime": {"seconds": 0.0, "status": "failed"},
                "error": error,
            }

    # One episode combines T5 inference, map-sized NumPy/SciPy metric arrays,
    # map-scoped embedding-cache writes, and PNG output.  These native-backed
    # operations are not safe when several episodes for the *same* map run in
    # threads.  Maps remain independent, so --workers still parallelizes map
    # groups while each individual map is processed in order.
    for index, result in run_instruction_jobs_grouped(
        instruction_files,
        workers=workers,
        group_key=lambda path: map_id_from_instruction_path(path, input_root),
        run_one=run_one_safe,
    ):
        results.append(result)
        # Match every other batch method: report each completed (or reused)
        # episode immediately in the compact two-line metric format.
        print_episode_result(index, total, result)

    written = sum(1 for item in results if item["status"] == "written")
    skipped = sum(1 for item in results if item["status"] == "skipped_exists")
    failed = sum(1 for item in results if item["status"] == "failed")
    if failed:
        print(f"failed={failed}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Lang2LTL trajectories for SemPathBench instructions."
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
        help="Cache Lang2LTL translations so repeated runs do not re-call the API.",
    )
    parser.add_argument(
        "--embedding-cache-root",
        type=Path,
        default=METHOD_ROOT / "_embedding_cache",
        help="Cache AP and referring-expression embeddings for embedding grounding.",
    )
    parser.add_argument(
        "--model",
        default=configured_model(),
        help=(
            "Gemini model used for LLM-based RER and LTL repair. Defaults to "
            "MODEL in scripts/methods/api_key.py, then GEMINI_MODEL."
        ),
    )
    parser.add_argument(
        "--symbolic-model-path",
        "--symbolic_model_path",
        type=Path,
        default=DEFAULT_LANG2LTL_T5_CHECKPOINT_PATH,
        help="Path to the original Lang2LTL T5 checkpoint-best directory.",
    )
    parser.add_argument(
        "--symbolic-checkpoint-url",
        default=DEFAULT_LANG2LTL_T5_CHECKPOINT_URL,
        help=(
            "Google Drive folder URL for the downloadable Lang2LTL T5 checkpoint. "
            "The checkpoint is downloaded automatically when needed and reused "
            "when already present."
        ),
    )
    parser.add_argument(
        "--translation-mode",
        "--translation_mode",
        choices=TRANSLATION_MODES,
        default="auto",
        help=(
            "'llm' uses LLM RER + grounding + symbolic translation; "
            "'heuristic' uses local category matching; 'auto' tries LLM and "
            "falls back to heuristic."
        ),
    )
    parser.add_argument(
        "--grounding-mode",
        "--grounding_mode",
        choices=GROUNDING_MODES,
        default="embedding",
        help=(
            "Grounding mode for Lang2LTL REG. 'embedding' ranks AP semantic "
            "descriptions by cosine similarity to referring-expression embeddings; "
            "'local' uses the older structured category/attribute matcher for "
            "debugging or ablation."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        "--embedding_model",
        default=configured_embedding_model(),
        help=(
            "Gemini embedding model used for semantic retrieval. Defaults to "
            "gemini-embedding-001."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-llm-cache",
        action="store_true",
        help="Force fresh LLM calls even when cached translations exist.",
    )
    parser.add_argument(
        "--max-prompt-objects",
        type=int,
        default=120,
        help="Maximum simple object AP entries included in translation prompts.",
    )
    parser.add_argument(
        "--object-label-radius",
        type=int,
        default=20,
        help="Grid-cell radius where an object AP is true in the label map.",
    )
    parser.add_argument(
        "--topk-groundings",
        "--topk_groundings",
        type=int,
        default=3,
        help="Number of grounding candidates retained per referring expression.",
    )
    parser.add_argument(
        "--planner-progress-interval",
        "--planner_progress_interval",
        type=int,
        default=50000,
        help=(
            "Print product-search progress every N expanded states. "
            "Use 0 to disable progress logs."
        ),
    )
    parser.add_argument(
        "--planning-mode",
        "--planning_mode",
        choices=PLANNING_MODES,
        default="vanilla",
        help=(
            "Planning mode. 'vanilla' uses AP-MDP-style planning; "
            "'product' uses product graph A*; 'fast' visits eventual AP "
            "targets with direct grid A*."
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
        help=(
            "Accepted for command compatibility. Batch console output stays "
            "compact; details are saved in per-prediction JSON files."
        ),
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
        embedding_cache_root=args.embedding_cache_root,
        overwrite_llm_cache=args.overwrite_llm_cache,
        symbolic_model_path=args.symbolic_model_path,
        symbolic_checkpoint_url=args.symbolic_checkpoint_url,
        max_prompt_objects=args.max_prompt_objects,
        object_label_radius=args.object_label_radius,
        topk_groundings=args.topk_groundings,
        planner_progress_interval=args.planner_progress_interval,
        grounding_mode=args.grounding_mode,
        embedding_backend="gemini",
        embedding_model=args.embedding_model,
        instruction_set=args.instruction_set,
        planning_mode=args.planning_mode,
        translation_mode=args.translation_mode,
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
