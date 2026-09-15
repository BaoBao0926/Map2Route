"""End-to-end Grounding2Route pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Mapping

from scripts.methods.grounding2route.config import (
    DEFAULT_PLANNER_MODE,
    DEFAULT_RELATIVE_COST_MODE,
    MAX_EXPANSIONS,
    WEIGHT_RELATIVE,
)
from scripts.methods.grounding2route.alias_resolver import (
    resolve_unknown_entity_alias,
    unknown_entity_category_from_failure,
)
from scripts.methods.grounding2route.grounding import SceneMap, ground_program
from scripts.methods.grounding2route.grounding.code import (
    GroundingCodeError,
    compile_program_to_grounding_code,
    execute_grounding_code,
    llm_generate_direct_grounding_code,
    llm_refine_grounding_code,
    llm_repair_grounding_code,
)
from scripts.methods.grounding2route.llm_client import GeminiClient
from scripts.methods.grounding2route.parse import (
    parse_instruction,
    scene_category_summary,
    supported_categories,
)
from scripts.methods.grounding2route.parse.api_program import llm_repair_api_after_grounding_failure
from scripts.methods.grounding2route.parse.intent import llm_repair_intent_after_grounding_failure
from scripts.methods.grounding2route.planner import plan_grounded_program
from scripts.methods.grounding2route.constraint_evaluation import apply_scope_ablation
from scripts.methods.grounding2route.ir import GPProgramSpec, GroundedProgram, PlannedProgram


@dataclass(frozen=True)
class Grounding2RouteRunConfig:
    parse_mode: str = "intent"
    max_parse_repairs: int = 0
    max_expansions: int = MAX_EXPANSIONS
    max_grounding_repairs: int = 1
    grounding_variant: str = "full"
    code_refinement: bool | None = None
    allow_grounding_helpers: bool | None = None
    execution_repair: str | None = None
    scope_ablation: str = "full"
    planner_mode: str = DEFAULT_PLANNER_MODE
    planner_heuristic_weight: float = 1.0
    relative_cost_mode: str = DEFAULT_RELATIVE_COST_MODE
    relative_weight: float = WEIGHT_RELATIVE
    soft_weight_scale: float = 1.0
    global_min_path_improvement_m: float | None = None
    max_tool_calls: int = 32
    max_tool_llm_turns: int = 40
    max_tool_query_retries: int = 3


@dataclass(frozen=True)
class GroundingArchitecture:
    """Resolved ablation settings for the grounding module only."""

    variant: str
    uses_ir: bool
    uses_code: bool
    uses_tool_calls: bool
    code_refinement: bool
    allow_helpers: bool
    repair_policy: str

    def to_json(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "structured_ir": self.uses_ir,
            "deterministic_ir_to_code": self.uses_code and self.uses_ir,
            "native_function_calling": self.uses_tool_calls,
            "code_refinement": self.code_refinement,
            "new_helper_functions": self.allow_helpers,
            "execution_repair": self.repair_policy,
            "planner_visible_to_grounding_code": False,
        }


def resolve_grounding_architecture(config: Grounding2RouteRunConfig) -> GroundingArchitecture:
    if config.grounding_variant == "tool_call":
        uses_ir, uses_code, uses_tool_calls, refinement, helpers, repair = (False, False, True, False, False, "tool")
    else:
        uses_tool_calls = False
        presets = {
        "direct_dsl": (True, False, False, False, "off"),
        "direct_cap": (False, True, True, True, "off"),
        "direct_cap_repair": (False, True, True, True, "code"),
        "ir_to_code": (True, True, False, False, "off"),
        "ir_code_refine": (True, True, True, True, "off"),
        "ir_code_refine_code_repair": (True, True, True, True, "code"),
        "full": (True, True, True, True, "code_then_ir"),
        "direct_id": (True, False, False, False, "off"),
        "ltl": (True, False, False, False, "off"),
        }
    if config.grounding_variant != "tool_call" and config.grounding_variant not in presets:
        raise ValueError(f"Unsupported Grounding2Route grounding variant: {config.grounding_variant!r}")
    if config.grounding_variant != "tool_call":
        uses_ir, uses_code, refinement, helpers, repair = presets[config.grounding_variant]
    if config.code_refinement is not None:
        refinement = config.code_refinement
    if config.allow_grounding_helpers is not None:
        helpers = config.allow_grounding_helpers
    if config.execution_repair is not None:
        repair = config.execution_repair
    if not uses_code:
        refinement = False
        helpers = False
        if not uses_tool_calls:
            repair = "off"
    if uses_tool_calls and repair not in {"off", "tool"}:
        raise ValueError(f"ToolCall repair policy must be off or tool, got {repair!r}")
    if repair not in {"off", "code", "code_then_ir", "tool"}:
        raise ValueError(f"Unsupported execution repair policy: {repair!r}")
    if config.scope_ablation not in {
        "full",
        "no_spatial_scope",
        "no_segment_scope",
        "no_scope",
        "no_soft_constraints",
    }:
        raise ValueError(f"Unsupported scope ablation: {config.scope_ablation!r}")
    return GroundingArchitecture(config.grounding_variant, uses_ir, uses_code, uses_tool_calls, refinement, helpers, repair)


@dataclass(frozen=True)
class Grounding2RoutePipelineResult:
    trajectory: list[list[int]]
    metadata: dict[str, object]
    steps: dict[str, object]
    success: bool
    failure_reason: str | None = None
    debug_data: Grounding2RouteDebugData | None = None


@dataclass(frozen=True)
class Grounding2RouteDebugData:
    program: GPProgramSpec | None = None
    scene: SceneMap | None = None
    grounded: GroundedProgram | None = None
    planned: PlannedProgram | None = None


def run_grounding2route_pipeline(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    *,
    config: Grounding2RouteRunConfig,
    llm_client: GeminiClient | None = None,
    grounding_code_override: str | None = None,
) -> Grounding2RoutePipelineResult:
    architecture = resolve_grounding_architecture(config)
    max_grounding_repairs = (
        max(0, config.max_grounding_repairs)
        if architecture.repair_policy != "off"
        else 0
    )
    steps: dict[str, object] = {}
    metadata: dict[str, object] = {
        "status": "running",
        "grounder_type": ("ToolCallGrounder" if architecture.uses_tool_calls else "CodeOrIRGrounder"),
        "model": (llm_client.model if llm_client is not None else None),
        "model_parameters": {"temperature": 0},
        "random_seed": None,
        "input_contract": (
            "Uses instruction text, start_pose, occupancy/room/object layers, "
            "room_instances, object_instances, and map-derived geometry. Does not use "
            "hard_constraints, soft_constraints, instruction.objects, human_expert_trajectory, "
            "or evaluator feedback during inference."
        ),
        "llm_map_view": (
            "compact_scene_catalog"
            if config.parse_mode in {"direct_id", "ltl"}
            else "category_inventory_and_restricted_grounding_interface"
        ),
        "llm_prompt_excludes": [
            "dense_map_layers",
            "object_footprint_cells",
            "hard_constraints",
            "soft_constraints",
            "human_expert_trajectory",
        ],
        "parser": {},
        "parse_repair": {
            "enabled": config.max_parse_repairs > 0,
            "max_attempts": max(0, config.max_parse_repairs),
        },
        "planner_config": {
            "mode": config.planner_mode,
            "heuristic_weight": config.planner_heuristic_weight,
            "relative_cost_mode": config.relative_cost_mode,
            "relative_weight": config.relative_weight,
            "soft_weight_scale": (
                0.0
                if config.scope_ablation == "no_soft_constraints"
                else config.soft_weight_scale
            ),
            "clearance_enabled": config.scope_ablation != "no_soft_constraints",
            "global_min_path_improvement_m": config.global_min_path_improvement_m,
        },
        "grounding": {},
        "grounding_architecture": architecture.to_json(),
        "planner": {},
        "grounding_repair": {
            "enabled": max_grounding_repairs > 0,
            "max_attempts": max_grounding_repairs,
        },
        "alias_resolution": {
            "enabled": max_grounding_repairs > 0,
            "attempts": [],
        },
        "method_level_deviations": [],
    }
    if architecture.uses_tool_calls:
        from scripts.methods.grounding2route.grounding.tool_pipeline import run_tool_call_pipeline

        tool_result = run_tool_call_pipeline(
            map_state, instruction, config=config, llm_client=llm_client,
            metadata=metadata, steps=steps,
        )
        tool_scene = SceneMap(map_state, instruction)
        return Grounding2RoutePipelineResult(
            tool_result.trajectory, metadata, steps, tool_result.success,
            tool_result.failure_reason,
            Grounding2RouteDebugData(
                program=None, scene=tool_scene, grounded=tool_result.grounded, planned=tool_result.planned,
            ),
        )

    # Direct CaP is the one ablation which intentionally skips JSON IR.  It is
    # still executed in exactly the same sandbox and handed to the identical
    # deterministic planner as every other variant.
    if not architecture.uses_ir:
        return _run_code_grounding_pipeline(
            map_state,
            instruction,
            config=config,
            architecture=architecture,
            metadata=metadata,
            steps=steps,
            program=None,
            llm_client=llm_client,
            max_grounding_repairs=max_grounding_repairs,
            grounding_code_override=grounding_code_override,
        )

    try:
        program, parser_meta = parse_instruction(
            instruction,
            map_state,
            mode=config.parse_mode,
            llm_client=llm_client,
            max_parse_repairs=max(0, config.max_parse_repairs),
        )
        steps["parse"] = {
            "status": "success",
            "metadata": parser_meta,
            "program": program.to_json(),
        }
        parser_attempts = parser_meta.get("parse_attempts")
        if isinstance(parser_attempts, list):
            steps["parse_attempts"] = parser_attempts
        else:
            steps["parse_attempts"] = [
                {
                    "attempt": 0,
                    "kind": "initial",
                    "status": "success",
                    "metadata": parser_meta,
                    "program": program.to_json(),
                }
            ]
        metadata["parser"] = {
            "mode": config.parse_mode,
            "status": "success",
            "diagnostics": list(program.diagnostics),
        }
        latest_api_program = parser_meta.get("raw_api_program")
        if not isinstance(latest_api_program, str):
            latest_api_program = program.source
        latest_intent_source = parser_meta.get("raw_intent")
        if not isinstance(latest_intent_source, str):
            latest_intent_source = None
    except Exception as exc:
        metadata["status"] = "PARSE_FAILED"
        parser_metadata = getattr(exc, "metadata", None)
        if isinstance(parser_metadata, dict):
            parser_payload = dict(parser_metadata)
            parser_payload.setdefault("error", str(exc))
        else:
            parser_payload = {"mode": config.parse_mode, "status": "failed", "error": str(exc)}
        metadata["parser"] = parser_payload
        steps["parse"] = dict(metadata["parser"])  # type: ignore[arg-type]
        parser_attempts = parser_payload.get("parse_attempts")
        if isinstance(parser_attempts, list):
            steps["parse_attempts"] = parser_attempts
        else:
            steps["parse_attempts"] = [
                {
                    "attempt": 0,
                    "kind": "initial",
                    **dict(metadata["parser"]),  # type: ignore[arg-type]
                }
            ]
        trajectory = _start_only(map_state, instruction)
        return Grounding2RoutePipelineResult(trajectory, metadata, steps, False, "PARSE_FAILED")

    scene = SceneMap(map_state, instruction)
    entity_categories, room_categories = supported_categories(map_state)
    category_summary = scene_category_summary(map_state)
    grounded: GroundedProgram | None = None
    planned: PlannedProgram | None = None

    if architecture.uses_code:
        return _run_code_grounding_pipeline(
            map_state,
            instruction,
            config=config,
            architecture=architecture,
            metadata=metadata,
            steps=steps,
            program=program,
            llm_client=llm_client,
            max_grounding_repairs=max_grounding_repairs,
            scene=scene,
            entity_categories=entity_categories,
            room_categories=room_categories,
            category_summary=category_summary,
        )
    try:
        grounding_attempts: list[dict[str, object]] = []
        repair_attempts: list[dict[str, object]] = []
        alias_resolution_attempts: list[dict[str, object]] = []
        for attempt in range(max_grounding_repairs + 1):
            grounded = ground_program(program, scene)
            grounded_json = grounded.to_json()
            grounding_attempts.append(
                {
                    "attempt": attempt,
                    "status": grounded.status,
                    **grounded_json,
                }
            )
            steps["grounding"] = grounded_json
            steps["grounding_attempts"] = grounding_attempts
            metadata["grounding"] = {
                "status": grounded.status,
                "failure_reason": grounded.failure_reason,
                "segment_count": len(grounded.segments),
                "attempt": attempt,
                "repair_attempt_count": len(repair_attempts),
            }
            if grounded.status == "success":
                if repair_attempts:
                    steps["repair_attempts"] = repair_attempts
                    metadata["parser"]["repair_attempt_count"] = len(repair_attempts)  # type: ignore[index]
                if alias_resolution_attempts:
                    steps["alias_resolution_attempts"] = alias_resolution_attempts
                    metadata["alias_resolution"] = {
                        "enabled": True,
                        "attempts": alias_resolution_attempts,
                    }
                break
            alias_retry = (
                _maybe_resolve_unknown_entity_alias(
                    grounded.failure_reason,
                    instruction,
                    program_source=program.source,
                    entity_categories=entity_categories,
                    llm_client=llm_client,
                )
                if max_grounding_repairs > 0
                else None
            )
            if alias_retry is not None:
                alias_resolution_attempts.append(alias_retry)
                steps["alias_resolution_attempts"] = alias_resolution_attempts
                metadata["alias_resolution"] = {
                    "enabled": True,
                    "attempts": alias_resolution_attempts,
                }
                if alias_retry.get("status") == "resolved":
                    grounded = ground_program(program, scene)
                    grounded_json = grounded.to_json()
                    grounding_attempts.append(
                        {
                            "attempt": attempt,
                            "kind": "alias_retry",
                            "status": grounded.status,
                            **grounded_json,
                        }
                    )
                    steps["grounding"] = grounded_json
                    steps["grounding_attempts"] = grounding_attempts
                    metadata["grounding"] = {
                        "status": grounded.status,
                        "failure_reason": grounded.failure_reason,
                        "segment_count": len(grounded.segments),
                        "attempt": attempt,
                        "repair_attempt_count": len(repair_attempts),
                        "alias_retry": True,
                    }
                    if grounded.status == "success":
                        break
            if (
                llm_client is None
                or attempt >= max_grounding_repairs
            ):
                metadata["status"] = "GROUNDING_FAILED"
                if repair_attempts:
                    steps["repair_attempts"] = repair_attempts
                    metadata["parser"]["repair_attempt_count"] = len(repair_attempts)  # type: ignore[index]
                if alias_resolution_attempts:
                    steps["alias_resolution_attempts"] = alias_resolution_attempts
                    metadata["alias_resolution"] = {
                        "enabled": True,
                        "attempts": alias_resolution_attempts,
                    }
                trajectory = _start_only(map_state, instruction)
                debug_data = Grounding2RouteDebugData(program=program, scene=scene, grounded=grounded)
                return Grounding2RoutePipelineResult(
                    trajectory,
                    metadata,
                    steps,
                    False,
                    "GROUNDING_FAILED",
                    debug_data,
                )

            repair_index = attempt + 1
            try:
                if config.parse_mode == "intent":
                    repaired_program, repair_meta = llm_repair_intent_after_grounding_failure(
                        str(instruction.get("instruction") or ""),
                        previous_intent=latest_intent_source or "{}",
                        previous_compiled_dsl=program.source,
                        grounding_error=str(grounded.failure_reason or grounded.status),
                        entity_categories=entity_categories,
                        room_categories=room_categories,
                        scene_summary=category_summary,
                        llm_client=llm_client,
                        attempt=repair_index,
                    )
                else:
                    repaired_program, repair_meta = llm_repair_api_after_grounding_failure(
                        str(instruction.get("instruction") or ""),
                        previous_api_program=latest_api_program,
                        grounding_error=str(grounded.failure_reason or grounded.status),
                        entity_categories=entity_categories,
                        room_categories=room_categories,
                        scene_summary=category_summary,
                        llm_client=llm_client,
                        attempt=repair_index,
                    )
            except Exception as exc:
                repair_payload = {
                    "attempt": repair_index,
                    "status": "failed",
                    "error": str(exc),
                    "previous_grounding_error": grounded.failure_reason,
                }
                parser_metadata = getattr(exc, "metadata", None)
                if isinstance(parser_metadata, dict):
                    repair_payload.update(parser_metadata)
                repair_attempts.append(repair_payload)
                steps["repair_attempts"] = repair_attempts
                metadata["status"] = "GROUNDING_REPAIR_FAILED"
                trajectory = _start_only(map_state, instruction)
                debug_data = Grounding2RouteDebugData(program=program, scene=scene, grounded=grounded)
                return Grounding2RoutePipelineResult(
                    trajectory,
                    metadata,
                    steps,
                    False,
                    "GROUNDING_REPAIR_FAILED",
                    debug_data,
                )

            repair_payload = {
                "attempt": repair_index,
                "status": "success",
                "metadata": repair_meta,
                "program": repaired_program.to_json(),
            }
            repair_attempts.append(repair_payload)
            steps["repair_attempts"] = repair_attempts
            parse_attempts = steps.setdefault("parse_attempts", [])
            if isinstance(parse_attempts, list):
                parse_attempts.append(
                    {
                        "attempt": repair_index,
                        "kind": str(repair_meta.get("mode") or "grounding_repair"),
                        "status": "success",
                        "metadata": repair_meta,
                        "program": repaired_program.to_json(),
                    }
                )
            steps["parse"] = {
                "status": "success",
                "metadata": repair_meta,
                "program": repaired_program.to_json(),
            }
            metadata["parser"] = {
                "mode": str(repair_meta.get("mode") or f"{config.parse_mode}_grounding_repair"),
                "status": "success",
                "diagnostics": list(repaired_program.diagnostics),
                "repair_attempt_count": len(repair_attempts),
            }
            program = repaired_program
            next_api_program = repair_meta.get("raw_api_program")
            if isinstance(next_api_program, str):
                latest_api_program = next_api_program
            next_intent_source = repair_meta.get("raw_intent")
            if isinstance(next_intent_source, str):
                latest_intent_source = next_intent_source
    except Exception as exc:
        metadata["status"] = "GROUNDING_FAILED"
        metadata["grounding"] = {"status": "failed", "error": str(exc)}
        steps["grounding"] = dict(metadata["grounding"])  # type: ignore[arg-type]
        trajectory = _start_only(map_state, instruction)
        debug_data = Grounding2RouteDebugData(program=program, scene=scene, grounded=grounded)
        return Grounding2RoutePipelineResult(
            trajectory,
            metadata,
            steps,
            False,
            "GROUNDING_FAILED",
            debug_data,
        )

    try:
        planner_started = time.perf_counter()
        planned = plan_grounded_program(
            scene, grounded, max_expansions=config.max_expansions,
            planner_mode=config.planner_mode, heuristic_weight=config.planner_heuristic_weight,
            relative_cost_mode=config.relative_cost_mode,
            relative_weight=config.relative_weight,
            soft_weight_scale=(
                0.0
                if config.scope_ablation == "no_soft_constraints"
                else config.soft_weight_scale
            ),
            include_clearance=config.scope_ablation != "no_soft_constraints",
            global_min_path_improvement_m=config.global_min_path_improvement_m,
        )
        planning_wall_seconds = time.perf_counter() - planner_started
        planner_payload = planned.to_json()
        planner_payload["planning_wall_seconds"] = planning_wall_seconds
        steps["planner"] = planner_payload
        metadata["planner"] = {
            "status": planned.status,
            "failure_reason": planned.failure_reason,
            "mode": config.planner_mode,
            "heuristic_weight": config.planner_heuristic_weight,
            "details": dict(planned.details),
            "segment_count": len(planned.segments),
            "trajectory_length": len(planned.trajectory),
            "planning_wall_seconds": planning_wall_seconds,
        }
    except Exception as exc:
        metadata["status"] = "PLANNING_FAILED"
        metadata["planner"] = {"status": "failed", "error": str(exc)}
        steps["planner"] = dict(metadata["planner"])  # type: ignore[arg-type]
        trajectory = _start_only(map_state, instruction)
        debug_data = Grounding2RouteDebugData(program=program, scene=scene, grounded=grounded, planned=planned)
        return Grounding2RoutePipelineResult(
            trajectory,
            metadata,
            steps,
            False,
            "PLANNING_FAILED",
            debug_data,
        )

    trajectory = planned.trajectory or _start_only(map_state, instruction)
    metadata["status"] = "success" if planned.status == "success" else str(planned.failure_reason or planned.status)
    return Grounding2RoutePipelineResult(
        trajectory,
        metadata,
        steps,
        planned.status == "success",
        None if planned.status == "success" else planned.failure_reason,
        Grounding2RouteDebugData(program=program, scene=scene, grounded=grounded, planned=planned),
    )


def _run_code_grounding_pipeline(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    *,
    config: Grounding2RouteRunConfig,
    architecture: GroundingArchitecture,
    metadata: dict[str, object],
    steps: dict[str, object],
    program: GPProgramSpec | None,
    llm_client: GeminiClient | None,
    max_grounding_repairs: int,
    scene: SceneMap | None = None,
    entity_categories: list[str] | None = None,
    room_categories: list[str] | None = None,
    category_summary: Mapping[str, object] | None = None,
    grounding_code_override: str | None = None,
) -> Grounding2RoutePipelineResult:
    """Run the IR→code (or Direct CaP) grounding half, then the fixed planner."""

    scene = scene or SceneMap(map_state, instruction)
    entity_categories = entity_categories or supported_categories(map_state)[0]
    room_categories = room_categories or supported_categories(map_state)[1]
    category_summary = category_summary or scene_category_summary(map_state)
    instruction_text = str(instruction.get("instruction") or "")
    code_attempts: list[dict[str, object]] = []
    repair_attempts: list[dict[str, object]] = []
    grounded: GroundedProgram | None = None
    source = ""

    if program is None:
        steps["parse"] = {
            "status": "skipped",
            "mode": "direct_cap",
            "reason": "Direct CaP ablation intentionally does not create JSON IR.",
        }
        steps["parse_attempts"] = [dict(steps["parse"])]
        metadata["parser"] = dict(steps["parse"])
        if grounding_code_override is not None:
            source = grounding_code_override
            code_attempts.append({"attempt": 0, "stage": "direct_cap_replay", "status": "success", "code": source})
        else:
            if llm_client is None:
                return _code_failure_result(
                    map_state, instruction, metadata, steps, None, scene,
                    "LLM_UNAVAILABLE", "Direct CaP requires an LLM client to generate grounding code.",
                )
            try:
                source, initial_meta = llm_generate_direct_grounding_code(
                    instruction_text,
                    entity_categories=entity_categories,
                    room_categories=room_categories,
                    scene_summary=category_summary,
                    llm_client=llm_client,
                    allow_helpers=architecture.allow_helpers,
                )
                code_attempts.append({"attempt": 0, "stage": "direct_cap_generation", "status": "success", "metadata": initial_meta, "code": source})
            except Exception as exc:
                return _code_failure_result(map_state, instruction, metadata, steps, None, scene, "CODE_GENERATION_FAILED", str(exc))
    else:
        source = compile_program_to_grounding_code(program)
        code_attempts.append({"attempt": 0, "stage": "deterministic_ir_to_code", "status": "success", "code": source})
        if architecture.code_refinement:
            if llm_client is None:
                code_attempts.append({"attempt": 0, "stage": "code_refinement", "status": "skipped", "reason": "LLM unavailable"})
            else:
                try:
                    source, refinement_meta = llm_refine_grounding_code(
                        instruction_text,
                        program=program,
                        scaffold=source,
                        entity_categories=entity_categories,
                        room_categories=room_categories,
                        scene_summary=category_summary,
                        llm_client=llm_client,
                        allow_helpers=architecture.allow_helpers,
                    )
                    code_attempts.append({"attempt": 0, "stage": "code_refinement", "status": "success", "metadata": refinement_meta, "code": source})
                except Exception as exc:
                    return _code_failure_result(map_state, instruction, metadata, steps, program, scene, "CODE_REFINEMENT_FAILED", str(exc))

    for attempt in range(max_grounding_repairs + 1):
        try:
            execution = execute_grounding_code(
                source,
                scene,
                expected_program=program,
                allow_helpers=architecture.allow_helpers,
            )
            grounded = execution.grounded
            code_attempts.append(
                {
                    "attempt": attempt,
                    "stage": "execute_and_verify",
                    "status": "success",
                    "code": source,
                    "verification": dict(execution.verification),
                    "grounded": grounded.to_json(),
                }
            )
            steps["grounding_code"] = {"source": source, "verification": dict(execution.verification)}
            steps["grounding_code_attempts"] = code_attempts
            steps["grounding"] = grounded.to_json()
            metadata["grounding"] = {
                "status": "success",
                "mode": "executable_grounding_code",
                "attempt": attempt,
                "repair_attempt_count": len(repair_attempts),
                "segment_count": len(grounded.segments),
                "verification": dict(execution.verification),
            }
            break
        except GroundingCodeError as exc:
            feedback = exc.feedback()
            code_attempts.append(
                {"attempt": attempt, "stage": "execute_and_verify", "status": "failed", "code": source, "feedback": feedback}
            )
            steps["grounding_code_attempts"] = code_attempts
            steps["grounding_code"] = {"source": source, "verification": {"status": "failed", "feedback": feedback}}
            if llm_client is None or architecture.repair_policy == "off" or attempt >= max_grounding_repairs:
                return _code_failure_result(
                    map_state, instruction, metadata, steps, program, scene, exc.code, exc.message,
                    details=feedback, grounded=grounded,
                )
            repair_index = attempt + 1
            try:
                if architecture.repair_policy == "code_then_ir" and attempt > 0 and program is not None:
                    # A first local code repair already failed.  Now repair the
                    # semantic scaffold itself, then deterministically rebuild
                    # its executable code before any optional refinement.
                    if config.parse_mode == "intent":
                        parse_step = steps.get("parse")
                        parse_metadata = parse_step.get("metadata") if isinstance(parse_step, Mapping) else None
                        previous_intent = parse_metadata.get("raw_intent") if isinstance(parse_metadata, Mapping) else None
                        program, repair_meta = llm_repair_intent_after_grounding_failure(
                            instruction_text,
                            previous_intent=previous_intent if isinstance(previous_intent, str) else "{}",
                            previous_compiled_dsl=program.source,
                            grounding_error=json.dumps(feedback, ensure_ascii=False),
                            entity_categories=entity_categories,
                            room_categories=room_categories,
                            scene_summary=category_summary,
                            llm_client=llm_client,
                            attempt=repair_index,
                        )
                    else:
                        program, repair_meta = llm_repair_api_after_grounding_failure(
                            instruction_text,
                            previous_api_program=program.source,
                            grounding_error=json.dumps(feedback, ensure_ascii=False),
                            entity_categories=entity_categories,
                            room_categories=room_categories,
                            scene_summary=category_summary,
                            llm_client=llm_client,
                            attempt=repair_index,
                        )
                    source = compile_program_to_grounding_code(program)
                    repair_attempts.append({"attempt": repair_index, "level": "ir", "status": "success", "metadata": repair_meta, "program": program.to_json()})
                    parse_attempts = steps.setdefault("parse_attempts", [])
                    if isinstance(parse_attempts, list):
                        parse_attempts.append({"attempt": repair_index, "kind": "ir_repair", "status": "success", "program": program.to_json(), "metadata": repair_meta})
                    if architecture.code_refinement:
                        source, refine_meta = llm_refine_grounding_code(
                            instruction_text,
                            program=program,
                            scaffold=source,
                            entity_categories=entity_categories,
                            room_categories=room_categories,
                            scene_summary=category_summary,
                            llm_client=llm_client,
                            allow_helpers=architecture.allow_helpers,
                            attempt=repair_index,
                        )
                        repair_attempts[-1]["code_refinement"] = refine_meta
                else:
                    source, repair_meta = llm_repair_grounding_code(
                        instruction_text,
                        program=program,
                        previous_code=source,
                        feedback=feedback,
                        entity_categories=entity_categories,
                        room_categories=room_categories,
                        scene_summary=category_summary,
                        llm_client=llm_client,
                        allow_helpers=architecture.allow_helpers,
                        attempt=repair_index,
                    )
                    repair_attempts.append({"attempt": repair_index, "level": "code", "status": "success", "metadata": repair_meta})
            except Exception as repair_exc:
                repair_attempts.append({"attempt": repair_index, "status": "failed", "error": f"{type(repair_exc).__name__}: {repair_exc}", "feedback": feedback})
                steps["grounding_repair_attempts"] = repair_attempts
                return _code_failure_result(
                    map_state, instruction, metadata, steps, program, scene,
                    "GROUNDING_REPAIR_FAILED", str(repair_exc), details=feedback, grounded=grounded,
                )
            steps["grounding_repair_attempts"] = repair_attempts

    if grounded is None:
        return _code_failure_result(map_state, instruction, metadata, steps, program, scene, "GROUNDING_FAILED", "No grounded task was produced.")
    if config.scope_ablation != "full":
        grounded = apply_scope_ablation(grounded, config.scope_ablation)
        steps["grounded_program_ablation"] = {
            "mode": config.scope_ablation,
            "grounded": grounded.to_json(),
        }
        steps["grounding"] = grounded.to_json()
        metadata["grounded_program_ablation"] = {"mode": config.scope_ablation}
    try:
        planner_started = time.perf_counter()
        planned = plan_grounded_program(
            scene, grounded, max_expansions=config.max_expansions,
            planner_mode=config.planner_mode, heuristic_weight=config.planner_heuristic_weight,
            relative_cost_mode=config.relative_cost_mode,
            relative_weight=config.relative_weight,
            soft_weight_scale=(
                0.0
                if config.scope_ablation == "no_soft_constraints"
                else config.soft_weight_scale
            ),
            include_clearance=config.scope_ablation != "no_soft_constraints",
            global_min_path_improvement_m=config.global_min_path_improvement_m,
        )
        planning_wall_seconds = time.perf_counter() - planner_started
        planner_payload = planned.to_json()
        planner_payload["planning_wall_seconds"] = planning_wall_seconds
        steps["planner"] = planner_payload
        metadata["planner"] = {
            "status": planned.status,
            "failure_reason": planned.failure_reason,
            "mode": config.planner_mode,
            "heuristic_weight": config.planner_heuristic_weight,
            "details": dict(planned.details),
            "segment_count": len(planned.segments),
            "trajectory_length": len(planned.trajectory),
            "planning_wall_seconds": planning_wall_seconds,
            "input": "verified_grounded_program",
        }
    except Exception as exc:
        metadata["status"] = "PLANNING_FAILED"
        metadata["planner"] = {"status": "failed", "error": str(exc)}
        steps["planner"] = dict(metadata["planner"])  # type: ignore[arg-type]
        return Grounding2RoutePipelineResult(
            _start_only(map_state, instruction), metadata, steps, False, "PLANNING_FAILED",
            Grounding2RouteDebugData(program=program, scene=scene, grounded=grounded),
        )
    trajectory = planned.trajectory or _start_only(map_state, instruction)
    metadata["status"] = "success" if planned.status == "success" else str(planned.failure_reason or planned.status)
    return Grounding2RoutePipelineResult(
        trajectory,
        metadata,
        steps,
        planned.status == "success",
        None if planned.status == "success" else planned.failure_reason,
        Grounding2RouteDebugData(program=program, scene=scene, grounded=grounded, planned=planned),
    )


def _code_failure_result(
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    metadata: dict[str, object],
    steps: dict[str, object],
    program: GPProgramSpec | None,
    scene: SceneMap,
    code: str,
    message: str,
    *,
    details: Mapping[str, object] | None = None,
    grounded: GroundedProgram | None = None,
) -> Grounding2RoutePipelineResult:
    metadata["status"] = "GROUNDING_FAILED"
    metadata["grounding"] = {
        "status": "failed",
        "mode": "executable_grounding_code",
        "failure_reason": f"{code}: {message}",
        "feedback": dict(details or {}),
    }
    steps["grounding"] = dict(metadata["grounding"])
    return Grounding2RoutePipelineResult(
        _start_only(map_state, instruction), metadata, steps, False, "GROUNDING_FAILED",
        Grounding2RouteDebugData(program=program, scene=scene, grounded=grounded),
    )


def _maybe_resolve_unknown_entity_alias(
    failure_reason: object,
    instruction: Mapping[str, object],
    *,
    program_source: str,
    entity_categories: list[str],
    llm_client: GeminiClient | None,
) -> dict[str, object] | None:
    unknown_category = unknown_entity_category_from_failure(failure_reason)
    if unknown_category is None:
        return None
    if llm_client is None:
        return {
            "status": "skipped_llm_unavailable",
            "unknown_category": unknown_category,
            "failure_reason": str(failure_reason),
        }
    try:
        return resolve_unknown_entity_alias(
            unknown_category=unknown_category,
            object_categories=entity_categories,
            instruction_text=str(instruction.get("instruction") or ""),
            program_source=program_source,
            failure_reason=str(failure_reason),
            llm_client=llm_client,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "unknown_category": unknown_category,
            "failure_reason": str(failure_reason),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _start_only(map_state: Mapping[str, object], instruction: Mapping[str, object]) -> list[list[int]]:
    start_pose = instruction.get("start_pose")
    if isinstance(start_pose, Mapping):
        row = start_pose.get("row")
        col = start_pose.get("col")
        if isinstance(row, int) and isinstance(col, int):
            return [[row, col]]
    grid_size = int(map_state.get("grid_size", 1))
    center = max(0, grid_size // 2)
    return [[center, center]]
