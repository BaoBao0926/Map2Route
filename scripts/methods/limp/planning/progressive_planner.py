"""Progressive TPSM grid planner for LIMP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from scripts.methods.limp.config import LimpConfig
from scripts.methods.lang2ltl.ltl import Formula
from scripts.methods.limp.grounding.crd_parser import ParsedPredicate
from scripts.methods.limp.grounding.referent_map import GroundingResult, crd_to_text
from scripts.methods.limp.grounding.scene_adapter import CandidateRegistry
from scripts.methods.limp.logic.residual_fallback import FallbackProgression
from scripts.methods.limp.logic.ltlf_progression import LTLFProgression
from scripts.methods.limp.planning.goal_regions import near_region
from scripts.methods.limp.planning.grid_planner import astar_path_to_any, traversable_with_forbidden
from scripts.methods.limp.planning.tpsm import TPSM
from scripts.methods.limp.utils.geometry import Cell
from scripts.methods.util.grid_astar import TraversableGrid, is_traversable, nearest_traversable_cell


@dataclass
class PlanResult:
    status: str
    trajectory: list[list[int]]
    failure_reason: str | None = None
    tpsm_summaries: list[dict[str, object]] = field(default_factory=list)
    planner_segments: list[dict[str, object]] = field(default_factory=list)
    task_state_sequence: list[dict[str, object]] = field(default_factory=list)
    logic_trace: list[dict[str, object]] = field(default_factory=list)


def start_cell_from_instruction(
    instruction: Mapping[str, object],
    traversable: TraversableGrid,
) -> Cell | None:
    start_pose = instruction.get("start_pose")
    if isinstance(start_pose, Mapping):
        row = start_pose.get("row")
        col = start_pose.get("col")
        if isinstance(row, int) and isinstance(col, int):
            if is_traversable(traversable, row, col):
                return row, col
            return nearest_traversable_cell(traversable, (row, col))
    return None


def _predicate_referent(predicate: ParsedPredicate) -> str | None:
    if not predicate.referents:
        return None
    return crd_to_text(predicate.referents[0])


def build_symbol_regions(
    predicates: list[ParsedPredicate],
    groundings: dict[str, GroundingResult],
    registry: CandidateRegistry,
    traversable: TraversableGrid,
    config: LimpConfig,
    negative_symbols: set[str] | None = None,
) -> tuple[dict[str, set[Cell]], dict[str, str], dict[str, list[set[Cell]]], list[dict[str, object]]]:
    grid_size = len(traversable)
    negative_symbols = negative_symbols or set()
    regions: dict[str, set[Cell]] = {}
    symbol_to_referent: dict[str, str] = {}
    symbol_visit_regions: dict[str, list[set[Cell]]] = {}
    records: list[dict[str, object]] = []
    for predicate in predicates:
        if predicate.predicate != "near":
            continue
        referent = _predicate_referent(predicate)
        if referent is None:
            continue
        grounding = groundings.get(referent)
        if grounding is None or grounding.selected_candidate_id is None:
            continue
        candidate_ids = list(grounding.selected_candidate_ids or ())
        if not candidate_ids:
            candidate_ids = [grounding.selected_candidate_id]
        visit_regions: list[set[Cell]] = []
        selected_records: list[dict[str, object]] = []
        for candidate_id in candidate_ids:
            candidate_for_region = registry.get(candidate_id)
            if candidate_for_region is None:
                continue
            region_for_candidate = near_region(
                candidate_for_region,
                traversable,
                radius=config.avoid_radius
                if predicate.encoded in negative_symbols
                else config.near_radius,
                grid_size=grid_size,
            )
            if not region_for_candidate:
                continue
            visit_regions.append(region_for_candidate)
            selected_records.append(
                {
                    "candidate_id": candidate_for_region.candidate_id,
                    "grounded_kind": candidate_for_region.kind,
                    "operational_semantics": "inside_room"
                    if candidate_for_region.kind == "room"
                    else "traversable_ring_around_object_footprint",
                    "region_size": len(region_for_candidate),
                }
            )
        candidate = registry.get(grounding.selected_candidate_id)
        if candidate is None:
            continue
        if not visit_regions:
            continue
        region = set().union(*visit_regions)
        regions[predicate.encoded] = region
        if len(visit_regions) > 1:
            symbol_visit_regions[predicate.encoded] = visit_regions
        symbol_to_referent[predicate.encoded] = referent
        records.append(
            {
                "symbol": predicate.encoded,
                "referent": referent,
                "candidate_id": candidate.candidate_id,
                "candidate_ids": candidate_ids,
                "grounded_kind": candidate.kind,
                "region_role": "avoidance" if predicate.encoded in negative_symbols else "goal",
                "region_radius": config.avoid_radius if predicate.encoded in negative_symbols else config.near_radius,
                "operational_semantics": "inside_room"
                if candidate.kind == "room"
                else "traversable_ring_around_object_footprint",
                "region_size": len(region),
                "visit_all_candidates": len(visit_regions) > 1,
                "visit_regions": selected_records,
            }
        )
    return regions, symbol_to_referent, symbol_visit_regions, records


def evaluate_true_symbols(
    cell: Cell,
    symbol_regions: Mapping[str, set[Cell]],
    *,
    multi_visit_symbols: set[str] | None = None,
    completed_multi_visit_symbols: set[str] | None = None,
) -> set[str]:
    multi_visit_symbols = multi_visit_symbols or set()
    completed_multi_visit_symbols = completed_multi_visit_symbols or set()
    result: set[str] = set()
    for symbol, region in symbol_regions.items():
        if symbol in multi_visit_symbols:
            if symbol in completed_multi_visit_symbols:
                result.add(symbol)
            continue
        if cell in region:
            result.add(symbol)
    return result


def _valuation_index(
    symbol_regions: Mapping[str, set[Cell]],
) -> dict[Cell, set[str]]:
    """Index the propositions true at cells covered by semantic regions."""

    values: dict[Cell, set[str]] = {}
    for symbol, region in symbol_regions.items():
        for cell in region:
            values.setdefault(cell, set()).add(symbol)
    return values


def _ltlf_transition_regions(
    traversable: TraversableGrid,
    progression: LTLFProgression,
    symbol_regions: Mapping[str, set[Cell]],
) -> tuple[set[Cell], set[Cell], dict[str, int], dict[Cell, set[str]]]:
    """Build a grid TPSM from the complete residual-LTL transition."""

    current = (progression.formula or progression.original_formula).simplify()
    values_by_cell = _valuation_index(symbol_regions)
    preview_cache: dict[frozenset[str], Formula] = {}
    goals: set[Cell] = set()
    forbidden: set[Cell] = set()
    residual_counts: dict[str, int] = {}

    for row, row_values in enumerate(traversable):
        for col, traversable_value in enumerate(row_values):
            if not traversable_value:
                continue
            cell = (row, col)
            labels = frozenset(values_by_cell.get(cell, ()))
            after = preview_cache.get(labels)
            if after is None:
                after = progression.preview(set(labels))
                preview_cache[labels] = after
            if after.op == "false":
                forbidden.add(cell)
            elif after != current:
                goals.add(cell)
                key = str(after)
                residual_counts[key] = residual_counts.get(key, 0) + 1

    return goals, forbidden, residual_counts, values_by_cell


def _plan_ltlf_transitions(
    instruction: Mapping[str, object],
    traversable: TraversableGrid,
    progression: LTLFProgression,
    symbol_regions: dict[str, set[Cell]],
    symbol_to_referent: dict[str, str],
    config: LimpConfig,
) -> PlanResult:
    """Plan through complete residual-formula transitions on the grid."""

    start = start_cell_from_instruction(instruction, traversable)
    if start is None:
        return PlanResult(
            status="INVALID_OUTPUT_PATH",
            trajectory=[],
            failure_reason="No traversable start_pose is available.",
        )

    trajectory: list[list[int]] = [[start[0], start[1]]]
    current = start
    true_at_start = evaluate_true_symbols(current, symbol_regions)
    progression.progress(true_at_start, stage="initial")
    task_states = [
        {
            "stage": "initial",
            "cell": [current[0], current[1]],
            "true_symbols": sorted(true_at_start),
            "current_formula": progression.current_formula(),
            "accepting": progression.accepting,
            "failed": progression.failed,
        }
    ]
    if progression.failed:
        return PlanResult(
            status="LOGIC_VIOLATION_AT_START",
            trajectory=trajectory,
            failure_reason="The start-cell valuation violates the translated LTL formula.",
            task_state_sequence=task_states,
            logic_trace=progression.trace,
        )

    tpsm_summaries: list[dict[str, object]] = []
    planner_segments: list[dict[str, object]] = []
    for step in range(1, config.max_progress_steps + 1):
        if progression.accepting:
            return PlanResult(
                status="SUCCESS_ACCEPTING_STATE",
                trajectory=trajectory,
                tpsm_summaries=tpsm_summaries,
                planner_segments=planner_segments,
                task_state_sequence=task_states,
                logic_trace=progression.trace,
            )

        formula_before = progression.current_formula()
        goals, forbidden, residual_counts, values_by_cell = _ltlf_transition_regions(
            traversable,
            progression,
            symbol_regions,
        )
        if not goals:
            return PlanResult(
                status="NO_DFA_TRANSITION_REGION",
                trajectory=trajectory,
                failure_reason=(
                    "No traversable cell enables a non-rejecting transition "
                    f"from residual formula {formula_before}."
                ),
                tpsm_summaries=tpsm_summaries,
                planner_segments=planner_segments,
                task_state_sequence=task_states,
                logic_trace=progression.trace,
            )

        tpsm = TPSM(
            stage_index=step,
            current_formula=formula_before,
            goal_regions={"enabled_dfa_transition": goals},
            forbidden_regions={"rejecting_dfa_transition": forbidden},
            enabled_transition="complete_proposition_valuation",
        ).summary()
        tpsm["candidate_residual_formula_counts"] = residual_counts
        tpsm_summaries.append(tpsm)

        stage_grid = traversable_with_forbidden(
            traversable,
            forbidden,
            start=current,
            goals=goals,
        )
        segment = astar_path_to_any(
            stage_grid,
            current,
            goals,
            max_goal_candidates=0,
        )
        if not segment:
            return PlanResult(
                status="NO_PATH",
                trajectory=trajectory,
                failure_reason="No path reaches an enabled non-rejecting DFA transition.",
                tpsm_summaries=tpsm_summaries,
                planner_segments=planner_segments,
                task_state_sequence=task_states,
                logic_trace=progression.trace,
            )

        stage_start = current
        if len(segment) > 1:
            trajectory.extend(segment[1:])
        current = (segment[-1][0], segment[-1][1])
        true_symbols = set(values_by_cell.get(current, ()))
        progression.progress(true_symbols, stage=f"segment_{step}")
        formula_after = progression.current_formula()
        planner_segments.append(
            {
                "stage_index": step,
                "symbol": " & ".join(sorted(true_symbols)),
                "referent": [symbol_to_referent.get(symbol) for symbol in sorted(true_symbols)],
                "start": [stage_start[0], stage_start[1]],
                "end": [current[0], current[1]],
                "path_length_cells": len(segment),
                "original_goal_region_size": len(goals),
                "planned_goal_region_size": len(goals),
                "goal_region_truncated": False,
                "selected_goal_cell": [current[0], current[1]],
                "distance_to_selected_goal": abs(current[0] - stage_start[0])
                + abs(current[1] - stage_start[1]),
                "true_symbols_at_end": sorted(true_symbols),
                "formula_before": formula_before,
                "formula_after": formula_after,
                "forbidden_cell_count": len(forbidden),
            }
        )
        task_states.append(
            {
                "stage": f"segment_{step}",
                "cell": [current[0], current[1]],
                "true_symbols": sorted(true_symbols),
                "current_formula": formula_after,
                "accepting": progression.accepting,
                "failed": progression.failed,
            }
        )
        if progression.failed:
            return PlanResult(
                status="LOGIC_VIOLATION",
                trajectory=trajectory,
                failure_reason="A planned segment entered the rejecting LTL state.",
                tpsm_summaries=tpsm_summaries,
                planner_segments=planner_segments,
                task_state_sequence=task_states,
                logic_trace=progression.trace,
            )

    return PlanResult(
        status="MAX_PROGRESS_STEPS",
        trajectory=trajectory,
        failure_reason="Maximum progressive planning steps exceeded.",
        tpsm_summaries=tpsm_summaries,
        planner_segments=planner_segments,
        task_state_sequence=task_states,
        logic_trace=progression.trace,
    )


def plan_progressively(
    instruction: Mapping[str, object],
    traversable: TraversableGrid,
    progression: FallbackProgression | LTLFProgression,
    symbol_regions: dict[str, set[Cell]],
    symbol_to_referent: dict[str, str],
    symbol_visit_regions: dict[str, list[set[Cell]]],
    registry: CandidateRegistry,
    predicates: list[ParsedPredicate],
    config: LimpConfig,
) -> PlanResult:
    if isinstance(progression, LTLFProgression):
        return _plan_ltlf_transitions(
            instruction,
            traversable,
            progression,
            symbol_regions,
            symbol_to_referent,
            config,
        )

    start = start_cell_from_instruction(instruction, traversable)
    if start is None:
        return PlanResult(
            status="INVALID_OUTPUT_PATH",
            trajectory=[],
            failure_reason="No traversable start_pose is available.",
        )

    trajectory: list[list[int]] = [[start[0], start[1]]]
    current = start
    multi_visit_symbols = set(symbol_visit_regions)
    completed_multi_visit_symbols: set[str] = set()
    true_at_start = evaluate_true_symbols(
        current,
        symbol_regions,
        multi_visit_symbols=multi_visit_symbols,
        completed_multi_visit_symbols=completed_multi_visit_symbols,
    )
    progression.progress(true_at_start, stage="initial")
    task_states = [
        {
            "stage": "initial",
            "cell": [current[0], current[1]],
            "true_symbols": sorted(true_at_start),
            "current_formula": progression.current_formula(),
            "accepting": progression.accepting,
        }
    ]

    tpsm_summaries: list[dict[str, object]] = []
    planner_segments: list[dict[str, object]] = []

    for step in range(1, config.max_progress_steps + 1):
        if progression.accepting:
            return PlanResult(
                status="SUCCESS_ACCEPTING_STATE",
                trajectory=trajectory,
                tpsm_summaries=tpsm_summaries,
                planner_segments=planner_segments,
                task_state_sequence=task_states,
                logic_trace=progression.trace,
            )

        current_symbol = progression.current_symbol
        if current_symbol is None:
            break
        goal_region = symbol_regions.get(current_symbol, set())
        visit_regions = symbol_visit_regions.get(current_symbol) or [goal_region]
        if not goal_region:
            return PlanResult(
                status="NO_GOAL_REGION",
                trajectory=trajectory,
                failure_reason=f"No goal region for symbol {current_symbol}.",
                tpsm_summaries=tpsm_summaries,
                planner_segments=planner_segments,
                task_state_sequence=task_states,
                logic_trace=progression.trace,
            )

        forbidden: set[Cell] = set()
        forbidden_regions: dict[str, set[Cell]] = {}
        for symbol in progression.forbidden_symbols:
            if symbol == current_symbol:
                continue
            region = symbol_regions.get(symbol, set())
            forbidden.update(region)
            forbidden_regions[symbol] = region

        tpsm = TPSM(
            stage_index=step,
            current_formula=progression.current_formula(),
            goal_regions={current_symbol: goal_region},
            forbidden_regions=forbidden_regions,
            enabled_transition=current_symbol,
        )
        tpsm_summaries.append(tpsm.summary())
        stage_start = current
        stage_path_length = 0
        selected_goal_cell: list[int] | None = None
        subgoals: list[dict[str, object]] = []
        for visit_index, visit_region in enumerate(visit_regions, start=1):
            stage_grid = traversable_with_forbidden(
                traversable,
                forbidden,
                start=current,
                goals=visit_region,
            )
            segment = astar_path_to_any(
                stage_grid,
                current,
                visit_region,
                max_goal_candidates=config.max_goal_candidates,
            )
            if not segment:
                return PlanResult(
                    status="NO_PATH",
                    trajectory=trajectory,
                    failure_reason=f"No path to enabled symbol {current_symbol} visit {visit_index}.",
                    tpsm_summaries=tpsm_summaries,
                    planner_segments=planner_segments,
                    task_state_sequence=task_states,
                    logic_trace=progression.trace,
                )
            if len(segment) > 1:
                trajectory.extend(segment[1:])
            current = (trajectory[-1][0], trajectory[-1][1])
            stage_path_length += len(segment)
            selected_goal_cell = [segment[-1][0], segment[-1][1]]
            subgoals.append(
                {
                    "visit_index": visit_index,
                    "region_size": len(visit_region),
                    "end": selected_goal_cell,
                    "path_length_cells": len(segment),
                }
            )

        if current_symbol in multi_visit_symbols:
            completed_multi_visit_symbols.add(current_symbol)
        true_symbols = evaluate_true_symbols(
            current,
            symbol_regions,
            multi_visit_symbols=multi_visit_symbols,
            completed_multi_visit_symbols=completed_multi_visit_symbols,
        )
        progression.progress(true_symbols, stage=f"segment_{step}")
        planner_segments.append(
            {
                "stage_index": step,
                "symbol": current_symbol,
                "referent": symbol_to_referent.get(current_symbol),
                "start": [stage_start[0], stage_start[1]],
                "end": [current[0], current[1]],
                "path_length_cells": stage_path_length,
                "original_goal_region_size": len(goal_region),
                "planned_goal_region_size": len(goal_region)
                if not config.max_goal_candidates or config.max_goal_candidates <= 0
                else min(len(goal_region), config.max_goal_candidates),
                "goal_region_truncated": bool(
                    config.max_goal_candidates
                    and config.max_goal_candidates > 0
                    and len(goal_region) > config.max_goal_candidates
                ),
                "selected_goal_cell": selected_goal_cell,
                "distance_to_selected_goal": abs(current[0] - stage_start[0])
                + abs(current[1] - stage_start[1]),
                "visit_all_candidates": len(visit_regions) > 1,
                "visit_subgoals": subgoals,
                "true_symbols_at_end": sorted(true_symbols),
            }
        )
        task_states.append(
            {
                "stage": f"segment_{step}",
                "cell": [current[0], current[1]],
                "true_symbols": sorted(true_symbols),
                "current_formula": progression.current_formula(),
                "accepting": progression.accepting,
            }
        )

    return PlanResult(
        status="MAX_PROGRESS_STEPS",
        trajectory=trajectory,
        failure_reason="Maximum progressive planning steps exceeded.",
        tpsm_summaries=tpsm_summaries,
        planner_segments=planner_segments,
        task_state_sequence=task_states,
        logic_trace=progression.trace,
    )
