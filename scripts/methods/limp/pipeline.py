"""End-to-end LIMP pipeline for a single SemPathBench episode."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from scripts.methods.limp.config import LimpConfig
from scripts.methods.limp.grounding.crd_parser import parse_predicate, parsed_predicate_to_dict
from scripts.methods.limp.grounding.referent_map import ground_predicates
from scripts.methods.limp.grounding.scene_adapter import (
    add_start_context_candidates,
    build_candidate_registry,
    registry_summary,
)
from scripts.methods.limp.language.translator import translate_instruction
from scripts.methods.limp.logic.automaton import AutomatonBackendUnavailable, build_progression
from scripts.methods.limp.logic.ltlf_progression import (
    LTLFParseError,
    negative_atomic_propositions,
    parse_encoded_ltl,
)
from scripts.methods.limp.planning.goal_regions import near_region
from scripts.methods.limp.planning.grid_planner import astar_path_to_any
from scripts.methods.limp.planning.progressive_planner import (
    build_symbol_regions,
    plan_progressively,
    start_cell_from_instruction,
)
from scripts.methods.limp.utils.geometry import distance
from scripts.methods.util.grid_astar import TraversableGrid, build_traversable_grid


@dataclass
class LimpEpisodeResult:
    trajectory: list[list[int]]
    metadata: dict[str, object]


def _start_only_trajectory(instruction: Mapping[str, object], map_state: Mapping[str, object]) -> list[list[int]]:
    traversable = build_traversable_grid(map_state)
    start = start_cell_from_instruction(instruction, traversable)
    if start is None:
        return []
    return [[start[0], start[1]]]


def _failure(
    instruction: Mapping[str, object],
    map_state: Mapping[str, object],
    metadata: dict[str, object],
    status: str,
    reason: str,
) -> LimpEpisodeResult:
    metadata["status"] = status
    metadata["failure_reason"] = reason
    return LimpEpisodeResult(
        trajectory=_start_only_trajectory(instruction, map_state),
        metadata=metadata,
    )


def _apply_extended_grounding_tiebreaks(
    groundings: dict[str, object],
    registry,
    traversable: TraversableGrid,
    start: tuple[int, int] | None,
    config: LimpConfig,
) -> None:
    if start is None:
        return
    for result in groundings.values():
        if getattr(result, "status", None) != "AMBIGUOUS_GROUNDING":
            continue
        candidates = list(getattr(result, "candidate_ids_after_filtering", ()))
        scored: list[tuple[int, float, int, str, int]] = []
        diagnostics: list[dict[str, object]] = []
        for candidate_id in candidates:
            candidate = registry.get(candidate_id)
            if candidate is None:
                continue
            region = near_region(
                candidate,
                traversable,
                radius=config.near_radius,
                grid_size=len(traversable),
            )
            path = astar_path_to_any(traversable, start, region, max_goal_candidates=0)
            if path:
                path_distance = len(path) - 1
                straight_distance = distance(candidate.centroid, start)
                candidate_size = len(candidate.cells)
                scored.append((path_distance, straight_distance, candidate_size, candidate_id, len(region)))
                diagnostics.append(
                    {
                        "candidate_id": candidate_id,
                        "reachable": True,
                        "path_length_cells": path_distance,
                        "straight_line_distance_cells": straight_distance,
                        "candidate_size_cells": candidate_size,
                        "goal_region_size": len(region),
                    }
                )
            else:
                diagnostics.append(
                    {
                        "candidate_id": candidate_id,
                        "reachable": False,
                        "path_length_cells": None,
                        "goal_region_size": len(region),
                    }
                )
        if not scored:
            continue
        referent = str(getattr(result, "referent", ""))
        selector_tokens = _selector_tokens(referent)
        ascending = sorted(scored, key=lambda item: (item[0], item[1], item[3]))
        selected_item = ascending[0]
        policy = "nearest_reachable_candidate_from_start"
        if selector_tokens & {"farthest", "furthest"}:
            selected_item = max(scored, key=lambda item: (item[1], item[0], item[3]))
            policy = "farthest_straight_line_candidate_from_start"
        elif selector_tokens & {"largest", "biggest"}:
            selected_item = max(scored, key=lambda item: (item[2], -item[0], item[3]))
            policy = "largest_candidate_footprint"
        elif selector_tokens & {"smallest"}:
            selected_item = min(scored, key=lambda item: (item[2], item[0], item[3]))
            policy = "smallest_candidate_footprint"
        elif selector_tokens & {"third"} and len(ascending) >= 3:
            selected_item = ascending[2]
            policy = "third_reachable_candidate_from_start"
        elif selector_tokens & {"second", "other", "another", "next"} and len(ascending) >= 2:
            selected_item = ascending[1]
            policy = "second_reachable_candidate_from_start"

        selected = selected_item[3]
        result.status = "GROUNDING_SUCCESS"
        result.selected_candidate_id = selected
        result.selected_candidate_ids = (selected,)
        result.disambiguation = {
            "mode": "extended",
            "policy": policy,
            "uses_hidden_annotation": False,
            "uses_evaluator_feedback": False,
            "selector_tokens": sorted(selector_tokens),
            "selected_candidate_id": selected,
            "candidate_scores": diagnostics,
        }
        result.failure_reason = None


def _selector_tokens(referent: str) -> set[str]:
    normalized = referent.lower().replace("::", "_").replace("(", "_").replace(")", "_")
    parts = set(normalized.replace(",", "_").split("_"))
    return parts & {
        "nearest", "closest", "nearby", "farthest", "furthest",
        "first", "second", "third", "next", "original", "previous",
        "other", "another", "largest", "biggest", "smallest",
    }


def run_episode(
    instruction: Mapping[str, object],
    map_state: Mapping[str, object],
    config: LimpConfig,
) -> LimpEpisodeResult:
    started = time.perf_counter()
    runtime_by_module: dict[str, float] = {}
    extended_adapter = config.method_variant == "extended"
    metadata: dict[str, object] = {
        "description": (
            "LIMP SemPathBench adapter with oracle semantic-instance perception, "
            "CRD grounding, task progression, TPSM construction, and grid planning."
        ),
        "input_contract": (
            "Inference uses instruction text, start_pose, traversability, room "
            "layers, object-instance layers, and map metadata. It does not use "
            "hard_constraints, soft_constraints, instruction objects, evaluator "
            "feedback, or human_expert_trajectory."
        ),
        "status": "RUNNING",
        "failure_reason": None,
        "model": config.model,
        "method_variant": config.method_variant,
        "translation_mode": config.translation_mode,
        "is_debug_or_smoke_test_mode": config.translation_mode == "heuristic"
        or config.allow_residual_fallback
        or config.automaton_backend == "residual-debug",
        "grounding_mode": config.grounding_mode if extended_adapter else "core",
        "automaton_backend": config.automaton_backend,
        "allow_residual_fallback": config.allow_residual_fallback,
        "spatial_frame": {
            "policy": "global_map_frame",
            "point_format": "[row, col]",
            "row_direction": "increases downward",
            "col_direction": "increases rightward",
            "above": "smaller row",
            "below": "larger row",
            "left": "smaller col",
            "right": "larger col",
            "frame_source": "SemPathBench top-down grid convention; front/behind remain unsupported without orientation metadata.",
        },
        "official_component_reuse": {
            "prompts": "adapted_from_official_limp_prompt_structure",
            "predicate_encoding": "local_reimplementation_of_official_parse_spatial_lifted_ltl_behavior",
            "ltl_progression": config.automaton_backend,
        },
        "automaton_summary": None,
        "logic_trace": [],
        "task_state_sequence": [],
        "tpsm_summaries": [],
        "planner_segments": [],
        "groundings": [],
        "predicate_groundings": [],
        "proposition_regions": [],
        "method_level_deviations": [
            "Navigation-only skill library: near[...] is supported; pick/release are unsupported.",
            "VLM/RGB-D perception is replaced by SemPathBench oracle semantic-instance candidates.",
            "Continuous FMT* planning is replaced by 8-connected grid A*.",
            "Formal automaton backend must be available for main runs; residual sequential progression is debug-only.",
        ],
    }

    mark = time.perf_counter()
    registry = build_candidate_registry(map_state)
    traversable = build_traversable_grid(map_state)
    start = start_cell_from_instruction(instruction, traversable)
    if extended_adapter:
        add_start_context_candidates(registry, start)
    runtime_by_module["scene_adapter"] = time.perf_counter() - mark
    metadata["map_inventory"] = registry_summary(registry)

    mark = time.perf_counter()
    translation = translate_instruction(
        instruction,
        registry,
        model=config.model,
        mode=config.translation_mode,
        cache_root=config.llm_cache_root,
        overwrite_cache=config.overwrite_llm_cache,
    )
    runtime_by_module["translation"] = time.perf_counter() - mark
    metadata.update(
        {
            "actual_translation_mode": translation.get("mode"),
            "requested_translation_mode": translation.get("requested_mode"),
            "stage1_prompt_version": translation.get("stage1_prompt_version"),
            "stage2_prompt_version": translation.get("stage2_prompt_version"),
            "raw_stage1_response": translation.get("raw_stage1_response"),
            "raw_stage2_response": translation.get("raw_stage2_response"),
            "parsed_stage1_ltl": translation.get("parsed_stage1_ltl"),
            "parsed_stage2_ltl": translation.get("parsed_stage2_ltl"),
            "translation_failure_reason": translation.get("translation_failure_reason"),
            "stage1_ltl": translation.get("stage1_ltl"),
            "stage2_ltl": translation.get("stage2_ltl"),
            "encoded_ltl": translation.get("encoded_ltl"),
            "encoding_map": translation.get("encoding_map"),
            "translation": translation,
        }
    )
    if translation.get("status") != "ok":
        metadata["runtime_by_module"] = runtime_by_module
        return _failure(
            instruction,
            map_state,
            metadata,
            "TRANSLATION_FAILED",
            "Translation did not produce lifted near[...] predicates.",
        )

    encoding_map = translation.get("encoding_map")
    if not isinstance(encoding_map, dict):
        metadata["runtime_by_module"] = runtime_by_module
        return _failure(instruction, map_state, metadata, "TRANSLATION_FAILED", "Invalid encoding map.")

    unsupported = translation.get("unsupported_predicates", [])
    metadata["unsupported_predicates"] = unsupported
    if unsupported:
        metadata["runtime_by_module"] = runtime_by_module
        return _failure(
            instruction,
            map_state,
            metadata,
            "UNSUPPORTED_PREDICATE",
            f"Unsupported predicates generated: {unsupported}",
        )

    predicates = [
        parse_predicate(str(symbol), str(raw))
        for symbol, raw in sorted(encoding_map.items())
    ]
    metadata["parsed_predicates"] = [parsed_predicate_to_dict(predicate) for predicate in predicates]
    metadata["parsed_crds"] = [
        referent
        for predicate in metadata["parsed_predicates"]  # type: ignore[index]
        for referent in predicate.get("referents", [])  # type: ignore[union-attr]
    ]

    mark = time.perf_counter()
    groundings, predicate_grounding_records = ground_predicates(
        predicates,
        registry,
        extended=extended_adapter,
    )
    if extended_adapter and config.grounding_mode == "extended":
        _apply_extended_grounding_tiebreaks(groundings, registry, traversable, start, config)
        for record in predicate_grounding_records:
            referent = record.get("referent")
            result = groundings.get(str(referent))
            if result is not None:
                record["grounding_status"] = result.status
                record["selected_candidate_id"] = result.selected_candidate_id
    runtime_by_module["grounding"] = time.perf_counter() - mark
    metadata["predicate_groundings"] = predicate_grounding_records
    metadata["groundings"] = [
        result.to_dict(registry)
        for result in groundings.values()
    ]
    failed_grounding = next(
        (result for result in groundings.values() if result.status != "GROUNDING_SUCCESS"),
        None,
    )
    if failed_grounding is not None:
        metadata["runtime_by_module"] = runtime_by_module
        return _failure(
            instruction,
            map_state,
            metadata,
            failed_grounding.status,
            failed_grounding.failure_reason or failed_grounding.status,
        )

    negative_symbols: set[str] = set()
    try:
        negative_symbols = negative_atomic_propositions(
            parse_encoded_ltl(str(translation.get("encoded_ltl") or ""))
        )
    except LTLFParseError:
        # build_progression below remains the single source of parse failures.
        pass

    mark = time.perf_counter()
    symbol_regions, symbol_to_referent, symbol_visit_regions, region_records = build_symbol_regions(
        predicates,
        groundings,
        registry,
        traversable,
        config,
        negative_symbols,
    )
    runtime_by_module["region_construction"] = time.perf_counter() - mark
    metadata["proposition_regions"] = region_records

    ordered_symbols_raw = translation.get("ordered_symbols", [])
    ordered_symbols = [str(symbol) for symbol in ordered_symbols_raw if str(symbol) in symbol_regions]
    if not ordered_symbols:
        metadata["runtime_by_module"] = runtime_by_module
        return _failure(
            instruction,
            map_state,
            metadata,
            "NO_GOAL_REGION",
            "No grounded proposition region is available for planning.",
        )

    try:
        progression, automaton_summary = build_progression(
            str(translation.get("encoded_ltl") or ""),
            ordered_symbols,
            backend=config.automaton_backend,
            allow_residual_fallback=config.allow_residual_fallback,
        )
    except AutomatonBackendUnavailable as exc:
        metadata["runtime_by_module"] = runtime_by_module
        metadata["automaton_summary"] = {
            "backend": config.automaton_backend,
            "available": False,
            "allow_residual_fallback": config.allow_residual_fallback,
            "failure_reason": str(exc),
        }
        return _failure(
            instruction,
            map_state,
            metadata,
            "AUTOMATON_BACKEND_UNAVAILABLE",
            str(exc),
        )
    except LTLFParseError as exc:
        metadata["runtime_by_module"] = runtime_by_module
        metadata["automaton_summary"] = {
            "backend": config.automaton_backend,
            "available": False,
            "failure_reason": str(exc),
            "encoded_ltl": str(translation.get("encoded_ltl") or ""),
        }
        return _failure(
            instruction,
            map_state,
            metadata,
            "LTL_PARSE_FAILED",
            str(exc),
        )
    metadata["automaton_summary"] = automaton_summary

    mark = time.perf_counter()
    plan = plan_progressively(
        instruction,
        traversable,
        progression,
        symbol_regions,
        symbol_to_referent,
        symbol_visit_regions,
        registry,
        predicates,
        config,
    )
    runtime_by_module["planning"] = time.perf_counter() - mark
    metadata.update(
        {
            "status": plan.status,
            "failure_reason": plan.failure_reason,
            "logic_trace": plan.logic_trace,
            "task_state_sequence": plan.task_state_sequence,
            "tpsm_summaries": plan.tpsm_summaries,
            "planner_segments": plan.planner_segments,
        }
    )
    runtime_by_module["total"] = time.perf_counter() - started
    metadata["runtime_by_module"] = runtime_by_module
    return LimpEpisodeResult(trajectory=plan.trajectory, metadata=metadata)
