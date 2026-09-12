"""End-to-end OSG-LLM adapter pipeline for one SemPathBench episode."""

from __future__ import annotations

from collections.abc import Mapping

from scripts.methods.lang2ltl.ltl import atomic_propositions
from scripts.methods.osgllm.config import OSGLLMConfig
from scripts.methods.osgllm.language.llm_adapter import translate_instruction_llm
from scripts.methods.osgllm.language.llm_heuristic import llm_preferred_nodes
from scripts.methods.osgllm.language.translator import translate_instruction_heuristic
from scripts.methods.osgllm.planning.hierarchical_planner import hierarchy_guided_plan
from scripts.methods.osgllm.planning.product_planner import product_astar
from scripts.methods.osgllm.planning.sequential_fallback import sequential_ap_astar
from scripts.methods.osgllm.scene_graph.builder import build_scene_graph
from scripts.methods.osgllm.scene_graph.propositions import build_proposition_model
from scripts.methods.util.grid_astar import build_traversable_grid, is_traversable, nearest_traversable_cell


def start_cell_from_instruction(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    traversable: list[list[bool]],
) -> tuple[int, int] | None:
    start_pose = instruction.get("start_pose")
    if isinstance(start_pose, Mapping):
        row = start_pose.get("row")
        col = start_pose.get("col")
        if isinstance(row, int) and isinstance(col, int):
            if is_traversable(traversable, row, col):
                return row, col
            return nearest_traversable_cell(traversable, (row, col), max_radius=80)
    grid_size = int(map_state["grid_size"])
    return nearest_traversable_cell(
        traversable,
        (grid_size / 2, grid_size / 2),
        max_radius=grid_size,
    )


def build_osgllm_trajectory(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    *,
    config: OSGLLMConfig,
) -> tuple[list[list[int]], dict[str, object]]:
    traversable = build_traversable_grid(map_state)
    start = start_cell_from_instruction(map_state, instruction, traversable)
    if start is None:
        return [], {
            "status": "GROUNDING_FAILED",
            "failure_reason": "No traversable start cell could be found.",
            "method_level_deviations": [],
        }

    graph = build_scene_graph(
        map_state,
        traversable,
        object_reach_radius=config.object_reach_radius,
    )
    deviations: list[str] = [
        "Official OSG-LLM repository code is not used as a runtime dependency.",
        "Spot automata are represented by local LTLf formula progression.",
    ]
    if config.planner in {"amra", "hierarchical"}:
        deviations.append(
            "AMRA* multi-queue abstraction is not fully active in this first implementation; planning uses the occupancy-resolution product anchor queue."
        )
    if config.translation_mode == "llm":
        deviations.append(
            "LLM grounding/translation is experimental and uses deterministic validation/cache."
        )

    translation_details: dict[str, object] = {
        "requested_grounding_mode": config.grounding_mode,
        "requested_translation_mode": config.translation_mode,
    }
    if config.grounding_mode in {"llm", "auto"} or config.translation_mode in {"llm", "auto"}:
        try:
            translation = translate_instruction_llm(
                graph,
                instruction,
                start=start,
                model=config.model,
                cache_root=config.llm_cache_root,
                overwrite_cache=config.overwrite_llm_cache,
                cache_only=config.llm_cache_only,
            )
            translation_details["actual_mode"] = "llm"
        except Exception as exc:
            if config.grounding_mode == "llm" or config.translation_mode == "llm":
                translation = translate_instruction_heuristic(graph, instruction, start=start)
                translation = type(translation)(
                    status="LTL_TRANSLATION_FAILED",
                    formula=translation.formula,
                    raw_ltl=translation.raw_ltl,
                    normalized_ltl=translation.normalized_ltl,
                    goals=translation.goals,
                    avoids=translation.avoids,
                    entity_bindings=translation.entity_bindings,
                    ambiguous_bindings=translation.ambiguous_bindings,
                    unsupported_constraints=translation.unsupported_constraints,
                    validation={
                        **translation.validation,
                        "llm_error": f"{type(exc).__name__}: {exc}",
                        "valid": False,
                    },
                )
                translation_details["actual_mode"] = "llm_failed"
            else:
                translation = translate_instruction_heuristic(graph, instruction, start=start)
                translation_details["actual_mode"] = "heuristic_after_llm_failure"
                translation_details["llm_error"] = f"{type(exc).__name__}: {exc}"
    else:
        translation = translate_instruction_heuristic(graph, instruction, start=start)
        translation_details["actual_mode"] = "heuristic"
    used_aps = atomic_propositions(translation.formula)
    if config.planner in {"hierarchical", "amra"}:
        used_aps = set(used_aps)
        used_aps.update(region.ap for region in graph.by_kind("room"))
        used_aps.add("enter(floor_0)")
    proposition_model = build_proposition_model(
        graph,
        used_aps=set(used_aps),
    )
    metadata: dict[str, object] = {
        "status": translation.status,
        "scene_graph_summary": graph.summary(),
        "grounding": {
            "grounded_instruction": str(instruction.get("instruction", "")),
            "entity_bindings": list(translation.entity_bindings),
            "virtual_regions": [],
            "ambiguous_bindings": list(translation.ambiguous_bindings),
            "unsupported_constraints": list(translation.unsupported_constraints),
        },
        "ltl": {
            "translation_mode": config.translation_mode,
            "grounding_mode": config.grounding_mode,
            "effective_translation_mode": translation_details["actual_mode"],
            "raw_ltl": translation.raw_ltl,
            "normalized_ltl": translation.normalized_ltl,
            "atomic_propositions": sorted(atomic_propositions(translation.formula)),
            "validation": {
                **translation.validation,
                **translation_details,
            },
        },
        "planner": {
            "planner_type": "not_started",
            "use_ltl_heuristic": True,
            "use_llm_heuristic": config.use_llm_heuristic,
            "expanded_states": None,
            "path_cost": None,
            "timeout_seconds": config.max_planning_seconds,
            "max_expansions": config.max_expansions,
            "fallback_disabled": config.disable_fallback,
            "unlimited_planning": (
                config.max_planning_seconds is None
                and config.max_expansions is None
            ),
        },
        "method_level_deviations": deviations,
    }

    if translation.status != "SUCCESS":
        metadata["status"] = translation.status
        metadata["failure_reason"] = "No hard goal could be grounded from the instruction text."
        return [[start[0], start[1]]], metadata

    if config.planner == "astar":
        planning = sequential_ap_astar(
            traversable,
            proposition_model,
            start,
            translation.goals,
            forbidden_aps=translation.avoids,
            max_seconds=config.max_planning_seconds,
        )
        deviations.append(
            "Direct sequential grid A* was used as a planner substitution; "
            "the LTL product state and hierarchy were not searched."
        )
        metadata["status"] = planning.status
        metadata["method_level_deviations"] = deviations
        metadata["planner"] = {
            **metadata["planner"],  # type: ignore[arg-type]
            **planning.details,
            "planner_type": "sequential_grid_astar",
            "use_llm_heuristic": False,
            "fallback_disabled": True,
        }
        return planning.trajectory, metadata

    llm_heuristic: dict[str, object] = {
        "enabled": config.use_llm_heuristic,
        "status": "disabled",
        "preferred_nodes": [],
    }
    preferred_nodes: tuple[str, ...] = ()
    if config.use_llm_heuristic:
        try:
            llm_heuristic = llm_preferred_nodes(
                graph,
                instruction,
                translation.goals,
                model=config.model,
                cache_root=config.llm_cache_root,
                overwrite_cache=config.overwrite_llm_cache,
                cache_only=config.llm_cache_only,
            )
            llm_heuristic["enabled"] = True
            llm_heuristic["status"] = "available"
            raw_preferred = llm_heuristic.get("preferred_nodes", [])
            if isinstance(raw_preferred, list):
                preferred_nodes = tuple(str(item) for item in raw_preferred)
        except Exception as exc:
            llm_heuristic = {
                "enabled": True,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "preferred_nodes": [],
            }
    metadata["planner"]["llm_heuristic"] = llm_heuristic  # type: ignore[index]

    planning = product_astar(
        traversable,
        proposition_model,
        start,
        translation.formula,
        max_expansions=config.max_expansions,
        max_seconds=config.max_planning_seconds,
        verbose=config.verbose,
    )
    if planning.status != "SUCCESS" and config.disable_fallback:
        deviations.append(
            "Hierarchy-guided and sequential fallbacks were disabled; the failed product-state A* attempt is scored with a start-only trajectory."
        )
        metadata["status"] = planning.status
        metadata["method_level_deviations"] = deviations
        metadata["planner"] = {
            **metadata["planner"],  # type: ignore[arg-type]
            **planning.details,
            "planner_type": "occupancy_product_anchor_astar_strict",
            "use_llm_heuristic": config.use_llm_heuristic,
            "fallback_disabled": True,
        }
        return [[start[0], start[1]]], metadata

    if (
        planning.status != "SUCCESS"
        and translation.goals
        and not translation.avoids
        and config.planner in {"hierarchical", "amra"}
    ):
        hierarchy_result = hierarchy_guided_plan(
            graph,
            traversable,
            proposition_model,
            start,
            translation.goals,
            preferred_nodes=preferred_nodes,
        )
        if hierarchy_result.status == "SUCCESS":
            deviations.append(
                "Experimental object/room/floor hierarchy-guided refinement was used after the product anchor planner did not find an accepting path within the configured budget."
            )
            if config.planner == "amra":
                deviations.append(
                    "AMRA* is still an experimental local skeleton: hierarchy queues/refinement are active, but this is not the official C++ AMRA* implementation."
                )
            metadata["status"] = "SUCCESS"
            metadata["method_level_deviations"] = deviations
            metadata["planner"] = {
                **metadata["planner"],  # type: ignore[arg-type]
                "planner_type": "product_anchor_with_hierarchy_guided_refinement",
                "use_llm_heuristic": config.use_llm_heuristic,
                "product_anchor": planning.details,
                "hierarchy": hierarchy_result.details,
                "path_cost": None,
                "timeout_seconds": config.max_planning_seconds,
            }
            return hierarchy_result.trajectory, metadata

    if planning.status != "SUCCESS" and translation.goals and not translation.avoids:
        product_elapsed = float(planning.details.get("elapsed_seconds", 0.0))
        fallback_budget = (
            None
            if config.max_planning_seconds is None
            else max(0.0, config.max_planning_seconds - product_elapsed)
        )
        fallback = sequential_ap_astar(
            traversable,
            proposition_model,
            start,
            translation.goals,
            max_seconds=fallback_budget,
        )
        if fallback.status == "SUCCESS":
            deviations.append(
                "Sequential AP A* fallback was used after the product anchor planner did not find an accepting path within the configured budget."
            )
            metadata["status"] = "SUCCESS"
            metadata["method_level_deviations"] = deviations
            metadata["planner"] = {
                **metadata["planner"],  # type: ignore[arg-type]
                "planner_type": "occupancy_product_anchor_astar_with_sequential_fallback",
                "use_llm_heuristic": config.use_llm_heuristic,
                "product_anchor": planning.details,
                "fallback": fallback.details,
                "path_cost": None,
                "timeout_seconds": config.max_planning_seconds,
            }
            return fallback.trajectory, metadata

    metadata["status"] = planning.status
    metadata["planner"] = {
        **metadata["planner"],  # type: ignore[arg-type]
        **planning.details,
        "use_llm_heuristic": config.use_llm_heuristic,
    }
    return planning.trajectory, metadata
