"""End-to-end SayPlan pipeline for one SemPathBench episode."""

from __future__ import annotations

import time
from dataclasses import dataclass

from scripts.methods.sayplan.adapters.sempathbench_loader import SayPlanEpisode
from scripts.methods.sayplan.config import (
    DEFAULT_MAX_JSON_RETRIES,
    DEFAULT_MAX_REPLANS,
    DEFAULT_MAX_SEARCH_STEPS,
)
from scripts.methods.sayplan.planning.classical_planner import complete_path
from scripts.methods.sayplan.planning.high_level_planner import generate_high_level_plan
from scripts.methods.sayplan.planning.semantic_search import run_semantic_search
from scripts.methods.sayplan.planning.verifier import verify_plan


@dataclass
class SayPlanRunResult:
    trajectory: list[list[int]]
    final_high_level_plan: list[str]
    task_subgraph: dict[str, object]
    semantic_search_trace: list[dict[str, object]]
    planning_attempts: list[dict[str, object]]
    planning_failed: bool
    failure_reason: str | None
    runtime: dict[str, object]


def run_sayplan_episode(
    episode: SayPlanEpisode,
    *,
    model: str,
    max_search_steps: int = DEFAULT_MAX_SEARCH_STEPS,
    max_replans: int = DEFAULT_MAX_REPLANS,
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
    verbose: bool = False,
) -> SayPlanRunResult:
    start_time = time.perf_counter()
    if verbose:
        print(
            f"[SayPlan][episode] map_id={episode.map_id} "
            f"scene_id={episode.scene_id} instruction_id={episode.instruction_id} "
            f"start_cell={episode.start_cell} failure_reason={episode.failure_reason}",
            flush=True,
        )
        print(
            f"[SayPlan][episode] graph_nodes={len(episode.full_graph.nodes)} "
            f"graph_edges={len(episode.full_graph.edges)}",
            flush=True,
        )
    if episode.failure_reason:
        trajectory = [[episode.start_cell[0], episode.start_cell[1]]] if episode.start_cell else []
        if verbose:
            print(
                f"[SayPlan][episode] failed_before_search failure_reason={episode.failure_reason}",
                flush=True,
            )
        return SayPlanRunResult(
            trajectory=trajectory,
            final_high_level_plan=[],
            task_subgraph={},
            semantic_search_trace=[],
            planning_attempts=[],
            planning_failed=True,
            failure_reason=episode.failure_reason,
            runtime={"seconds": time.perf_counter() - start_time, "status": "failed"},
        )

    search = run_semantic_search(
        graph=episode.full_graph,
        instruction=episode.instruction,
        model=model,
        max_search_steps=max_search_steps,
        max_json_retries=max_json_retries,
        verbose=verbose,
    )
    if verbose:
        print(
            f"[SayPlan][episode] semantic_search_done "
            f"failure_reason={search.failure_reason} "
            f"visible_nodes={len(search.view.visible_nodes)}",
            flush=True,
        )

    attempts: list[dict[str, object]] = []
    feedback = ""
    previous_plan: list[str] | None = None
    best_trajectory: list[list[int]] = (
        [[episode.start_cell[0], episode.start_cell[1]]] if episode.start_cell else []
    )
    final_plan: list[str] = []
    failure_reason = search.failure_reason

    for iteration in range(1, max_replans + 1):
        if verbose:
            print(
                f"[SayPlan][replan] iteration={iteration} feedback={feedback!r}",
                flush=True,
            )
        plan_result = generate_high_level_plan(
            view=search.view,
            instruction=episode.instruction,
            model=model,
            feedback=feedback,
            previous_plan=previous_plan,
            max_json_retries=max_json_retries,
            verbose=verbose,
        )
        if plan_result.failure_reason:
            attempts.append(
                {
                    "iteration": iteration,
                    "status": "failed_llm_plan",
                    "failure_reason": plan_result.failure_reason,
                }
            )
            failure_reason = plan_result.failure_reason
            if verbose:
                print(
                    f"[SayPlan][replan] failed_llm_plan "
                    f"failure_reason={plan_result.failure_reason}",
                    flush=True,
                )
            break

        path_result = complete_path(
            graph=episode.full_graph,
            traversable=episode.traversable,
            start_cell=episode.start_cell,
            high_level_plan=plan_result.actions,
            verbose=verbose,
        )
        verification = verify_plan(
            task_graph_view=search.view,
            high_level_plan=plan_result.actions,
            path_result=path_result,
        )
        if verbose:
            print(
                f"[SayPlan][verify] success={verification.success} "
                f"feedback={verification.feedback!r} failures={verification.failures}",
                flush=True,
            )
        attempt = {
            "iteration": iteration,
            "raw_llm_response": plan_result.response.raw_json if plan_result.response else None,
            "high_level_plan": plan_result.plan_strings,
            "path_result": path_result.to_json(),
            "verifier_feedback": verification.feedback,
            "failures": verification.failures,
        }
        attempts.append(attempt)
        previous_plan = plan_result.plan_strings
        final_plan = plan_result.plan_strings
        if len(path_result.trajectory) > len(best_trajectory):
            best_trajectory = path_result.trajectory
        if verification.success:
            if verbose:
                print(
                    f"[SayPlan][episode] success trajectory_length={len(path_result.trajectory)} "
                    f"runtime_seconds={time.perf_counter() - start_time:.3f}",
                    flush=True,
                )
            return SayPlanRunResult(
                trajectory=path_result.trajectory,
                final_high_level_plan=final_plan,
                task_subgraph=search.view.task_subgraph_json(),
                semantic_search_trace=search.trace,
                planning_attempts=attempts,
                planning_failed=False,
                failure_reason=None,
                runtime={"seconds": time.perf_counter() - start_time, "status": "success"},
            )
        feedback = verification.feedback
        failure_reason = path_result.failure_reason or "max_replans_exceeded"

    if verbose:
        print(
            f"[SayPlan][episode] failed failure_reason={failure_reason or 'max_replans_exceeded'} "
            f"best_trajectory_length={len(best_trajectory)} "
            f"runtime_seconds={time.perf_counter() - start_time:.3f}",
            flush=True,
        )
    return SayPlanRunResult(
        trajectory=best_trajectory,
        final_high_level_plan=final_plan,
        task_subgraph=search.view.task_subgraph_json(),
        semantic_search_trace=search.trace,
        planning_attempts=attempts,
        planning_failed=True,
        failure_reason=failure_reason or "max_replans_exceeded",
        runtime={"seconds": time.perf_counter() - start_time, "status": "failed"},
    )
