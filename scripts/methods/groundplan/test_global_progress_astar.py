"""Regression tests for GroundPlan global position-progress A*."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from scripts.methods.groundplan.config import (
    DEFAULT_PLANNER_MODE,
    DEFAULT_RELATIVE_COST_MODE,
    MAX_EXPANSIONS,
    WEIGHT_FAR,
    WEIGHT_RELATIVE,
)
from scripts.methods.groundplan.ir import GPRef, GroundedConstraint, GroundedProgram, GroundedSegment
from scripts.methods.groundplan.pipeline import GroundPlanRunConfig
from scripts.methods.groundplan.planner.semantic_astar import _compile_soft, plan_grounded_program


class TinyScene:
    def __init__(self, size: int, start: tuple[int, int], *, blocked: set[tuple[int, int]] = set()) -> None:
        self.grid_size = size
        self.resolution = 1.0
        self.traversable = np.ones((size, size), dtype=bool)
        for row, col in blocked:
            self.traversable[row, col] = False
        self.map_state: dict[str, object] = {"layers": {}}
        self.start = GPRef("position", "start", cells=frozenset({start}))
        self.rooms: dict[str, GPRef] = {}


def region(identifier: str, *cells: tuple[int, int]) -> GPRef:
    return GPRef("region", identifier, cells=frozenset(cells))


def segment(identifier: str, target: GPRef, constraints: tuple[GroundedConstraint, ...] = ()) -> GroundedSegment:
    return GroundedSegment(identifier, GPRef("position", "unused", cells=frozenset({(0, 0)})), target, constraints)


def zero_soft(scene: TinyScene, constraints: object, **_kwargs: object) -> tuple[np.ndarray, list[dict[str, object]]]:
    return np.zeros((scene.grid_size, scene.grid_size), dtype=float), []


class PlannerDefaultTests(unittest.TestCase):
    def test_full_pipeline_uses_canonical_relative_ratio_defaults(self) -> None:
        config = GroundPlanRunConfig()
        self.assertEqual(DEFAULT_PLANNER_MODE, "sequential_greedy_astar")
        self.assertEqual(config.planner_mode, DEFAULT_PLANNER_MODE)
        self.assertEqual(config.relative_cost_mode, DEFAULT_RELATIVE_COST_MODE)
        self.assertEqual(config.relative_cost_mode, "ratio")
        self.assertEqual(config.relative_weight, WEIGHT_RELATIVE)
        self.assertEqual(config.relative_weight, 64.0)
        self.assertEqual(config.max_expansions, MAX_EXPANSIONS)
        self.assertEqual(config.soft_weight_scale, 1.0)
        self.assertIsNone(config.global_min_path_improvement_m)

    def test_soft_weight_scale_excludes_clearance(self) -> None:
        scene = TinyScene(5, (0, 0))
        constraint = GroundedConstraint(
            "prefer_far",
            (region("reference", (2, 2)),),
            ("s1",),
            constraint_id="far",
            hardness="soft",
        )
        clearance, _ = _compile_soft(scene, (), soft_weight_scale=0.0)
        baseline, _ = _compile_soft(scene, (constraint,), soft_weight_scale=1.0)
        doubled, details = _compile_soft(
            scene, (constraint,), soft_weight_scale=2.0
        )
        np.testing.assert_allclose(
            doubled - clearance,
            2.0 * (baseline - clearance),
        )
        self.assertEqual(details[0]["weight"], 2.0 * WEIGHT_FAR)

    def test_clearance_can_be_disabled_independently(self) -> None:
        scene = TinyScene(5, (0, 0))

        field, details = _compile_soft(scene, (), include_clearance=False)

        np.testing.assert_array_equal(field, np.zeros((5, 5), dtype=float))
        self.assertEqual(details, [])

    def test_planner_api_defaults_to_sequential(self) -> None:
        scene = TinyScene(12, (0, 0))
        first = region("first", (1, 0), (0, 5))
        final = region("final", (0, 10))
        program = GroundedProgram((segment("s1", first), segment("s2", final)), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            planned = plan_grounded_program(scene, program)
        self.assertEqual(planned.status, "success")
        self.assertEqual(planned.segments[0].endpoint, (1, 0))


class GlobalProgressAStarTests(unittest.TestCase):
    def test_single_target_matches_sequential_optimal_cost(self) -> None:
        scene = TinyScene(6, (0, 0))
        program = GroundedProgram((segment("s1", region("goal", (4, 3))),), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            sequential = plan_grounded_program(scene, program, planner_mode="sequential_greedy_astar")
            global_plan = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(sequential.status, "success")
        self.assertEqual(global_plan.status, "success")
        self.assertAlmostEqual(sequential.segments[0].path_length, global_plan.segments[0].path_length)
        self.assertAlmostEqual(global_plan.details["total_cost"], global_plan.segments[0].path_length)

    def test_global_search_selects_non_greedy_intermediate_arrival(self) -> None:
        scene = TinyScene(12, (0, 0))
        first = region("first", (1, 0), (0, 5))
        final = region("final", (0, 10))
        program = GroundedProgram((segment("s1", first), segment("s2", final)), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            sequential = plan_grounded_program(scene, program, planner_mode="sequential_greedy_astar")
            global_plan = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(sequential.segments[0].endpoint, (1, 0))
        self.assertEqual(global_plan.segments[0].endpoint, (0, 5))
        self.assertLess(global_plan.details["total_cost"], sum(item.path_length for item in sequential.segments))

    def test_start_inside_first_target_initializes_ordered_prefix(self) -> None:
        scene = TinyScene(5, (0, 0))
        program = GroundedProgram((segment("s1", region("first", (0, 0))), segment("s2", region("final", (0, 3)))), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            planned = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(planned.status, "success")
        self.assertEqual(planned.details["initial_progress"], 1)
        self.assertEqual(planned.segments[0].path, [[0, 0]])
        self.assertEqual(planned.segments[1].endpoint, (0, 3))

    def test_overlapping_targets_advance_only_one_stage_per_move(self) -> None:
        scene = TinyScene(4, (0, 0))
        overlap = region("overlap", (0, 1))
        program = GroundedProgram((segment("s1", overlap), segment("s2", overlap)), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            planned = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(planned.status, "success")
        self.assertEqual(planned.segments[0].path, [[0, 0], [0, 1]])
        self.assertGreaterEqual(len(planned.trajectory), 4)
        self.assertEqual(planned.segments[1].endpoint, (0, 1))

    def test_segment_and_spatial_scopes_gate_soft_cost(self) -> None:
        scene = TinyScene(4, (0, 0))
        reference = region("ref", (0, 0))
        scope = region("scope", (1, 1))
        constraint = GroundedConstraint("prefer_near", (reference,), ("s2",), spatial_scope=scope, constraint_id="scoped", hardness="soft")
        field, _details = _compile_soft(scene, (constraint,))
        self.assertGreater(float(field[1, 1]), 0.0)
        self.assertEqual(float(field[1, 2]), 0.0)
        program = GroundedProgram((segment("s1", region("a", (0, 1))), segment("s2", region("b", (0, 2)), (constraint,))), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", wraps=_compile_soft) as compile_soft:
            plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        seen = [tuple(item.args[1]) for item in compile_soft.call_args_list]
        self.assertIn((), seen)
        self.assertIn((constraint,), seen)

    def test_scoped_hard_avoidance_is_stage_dependent(self) -> None:
        scene = TinyScene(5, (0, 0))
        forbidden = region("forbidden", (0, 1))
        constraint = GroundedConstraint("forbid", (forbidden,), ("s1",), constraint_id="avoid_s1", hardness="hard")
        program = GroundedProgram((segment("s1", region("a", (0, 2)), (constraint,)), segment("s2", region("b", (0, 1))),), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            planned = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(planned.status, "success")
        self.assertNotIn([0, 1], planned.segments[0].path)
        self.assertEqual(planned.segments[1].endpoint, (0, 1))

    def test_astar_matches_zero_heuristic_dijkstra(self) -> None:
        scene = TinyScene(7, (0, 0), blocked={(2, 2), (2, 3), (3, 2)})
        program = GroundedProgram((segment("s1", region("a", (1, 5), (5, 1))), segment("s2", region("b", (6, 6))),), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            astar = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
            with patch("scripts.methods.groundplan.planner.semantic_astar._reverse_distance_field", side_effect=lambda scene, base, mask: np.zeros((scene.grid_size, scene.grid_size))):
                dijkstra = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(astar.status, "success")
        self.assertEqual(dijkstra.status, "success")
        self.assertAlmostEqual(astar.details["total_cost"], dijkstra.details["total_cost"], places=7)


    def test_spatial_scope_gates_hard_forbidden_cells(self) -> None:
        scene = TinyScene(4, (0, 0))
        forbidden = region("forbidden", (0, 1), (0, 2))
        spatial_scope = region("scope", (0, 1))
        constraint = GroundedConstraint(
            "forbid", (forbidden,), ("s1",), spatial_scope=spatial_scope,
            constraint_id="scoped_forbid", hardness="hard",
        )
        program = GroundedProgram((segment("s1", region("goal", (0, 2)), (constraint,)),), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            planned = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(planned.status, "success")
        self.assertNotIn([0, 1], planned.trajectory)
        self.assertEqual(planned.segments[0].endpoint, (0, 2))

    def test_relative_ratio_cost_rewards_stronger_satisfying_margin(self) -> None:
        scene = TinyScene(5, (0, 0))
        preferred = region("preferred", (0, 0))
        farther = region("farther", (0, 4))
        constraint = GroundedConstraint(
            "prefer_relative",
            (preferred, farther),
            ("s1",),
            relation="closer_to",
            constraint_id="relative",
            hardness="soft",
        )
        with patch(
            "scripts.methods.groundplan.planner.semantic_astar._clearance_field",
            return_value=np.zeros((5, 5), dtype=float),
        ):
            difference, _ = _compile_soft(
                scene, (constraint,), relative_cost_mode="difference", relative_weight=1.0
            )
            ratio, _ = _compile_soft(
                scene, (constraint,), relative_cost_mode="ratio", relative_weight=1.0
            )

        # The hinge cost becomes zero as soon as the ordering is satisfied.
        self.assertEqual(float(difference[0, 1]), 0.0)
        # Ratio cost is the bounded monotonic transform 1 / (1 + D_far/D_close).
        self.assertAlmostEqual(float(ratio[0, 1]), 1.0 / 4.0)
        self.assertEqual(float(ratio[0, 0]), 0.0)
        # Unlike a clipped inverse ratio, it retains a gradient in violations.
        self.assertAlmostEqual(float(ratio[0, 3]), 3.0 / 4.0)
        self.assertLess(float(ratio[0, 2]), float(ratio[0, 3]))

    def test_unreachable_target_returns_structured_failure(self) -> None:
        scene = TinyScene(4, (0, 0), blocked={(0, 1), (1, 0), (1, 1)})
        program = GroundedProgram((segment("s1", region("blocked_goal", (3, 3))),), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            planned = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(planned.status, "failed")
        self.assertEqual(planned.failure_reason, "NO_FEASIBLE_PATH")
        self.assertEqual(planned.trajectory, [[0, 0]])


class GlobalLayeredDPTests(unittest.TestCase):
    def test_selects_globally_better_intermediate_arrival(self) -> None:
        scene = TinyScene(12, (0, 0))
        first = region("first", (1, 0), (0, 5))
        final = region("final", (0, 10))
        program = GroundedProgram((segment("s1", first), segment("s2", final)), {})
        with patch(
            "scripts.methods.groundplan.planner.semantic_astar._compile_soft",
            side_effect=zero_soft,
        ):
            sequential = plan_grounded_program(
                scene, program, planner_mode="sequential_greedy_astar"
            )
            layered = plan_grounded_program(
                scene,
                program,
                planner_mode="global_layered_dp",
                max_expansions=0,
            )
        self.assertEqual(sequential.segments[0].endpoint, (1, 0))
        self.assertEqual(layered.segments[0].endpoint, (0, 5))
        self.assertEqual(layered.status, "success")
        self.assertTrue(layered.details["search_complete"])
        self.assertTrue(layered.details["optimality_proven"])
        self.assertFalse(layered.details["fallback_used"])
        self.assertLess(
            layered.details["total_cost"],
            sum(item.path_length for item in sequential.segments),
        )

    def test_minimum_path_improvement_can_keep_sequential_incumbent(self) -> None:
        scene = TinyScene(12, (0, 0))
        first = region("first", (1, 0), (0, 5))
        final = region("final", (0, 10))
        program = GroundedProgram(
            (segment("s1", first), segment("s2", final)), {}
        )
        with patch(
            "scripts.methods.groundplan.planner.semantic_astar._compile_soft",
            side_effect=zero_soft,
        ):
            layered = plan_grounded_program(
                scene,
                program,
                planner_mode="global_layered_dp",
                max_expansions=0,
                global_min_path_improvement_m=2.0,
            )
        self.assertEqual(layered.status, "success")
        self.assertEqual(layered.segments[0].endpoint, (1, 0))
        self.assertTrue(layered.details["fallback_used"])
        self.assertEqual(
            layered.details["fallback_reason"],
            "MINIMUM_PATH_IMPROVEMENT_NOT_MET",
        )
        self.assertEqual(layered.details["selection"], "sequential_incumbent")

    def test_zero_expansion_limit_means_unlimited(self) -> None:
        scene = TinyScene(8, (0, 0))
        program = GroundedProgram((segment("s1", region("goal", (7, 7))),), {})
        with patch(
            "scripts.methods.groundplan.planner.semantic_astar._compile_soft",
            side_effect=zero_soft,
        ):
            layered = plan_grounded_program(
                scene,
                program,
                planner_mode="global_layered_dp",
                max_expansions=0,
            )
        self.assertEqual(layered.status, "success")
        self.assertEqual(layered.segments[0].endpoint, (7, 7))
        self.assertTrue(layered.details["optimality_proven"])

    def test_positive_limit_returns_sequential_incumbent(self) -> None:
        scene = TinyScene(12, (0, 0))
        program = GroundedProgram(
            (
                segment("s1", region("first", (1, 0), (0, 5))),
                segment("s2", region("final", (0, 10))),
            ),
            {},
        )
        with patch(
            "scripts.methods.groundplan.planner.semantic_astar._compile_soft",
            side_effect=zero_soft,
        ):
            layered = plan_grounded_program(
                scene,
                program,
                planner_mode="global_layered_dp",
                max_expansions=2,
            )
        self.assertEqual(layered.status, "success")
        self.assertEqual(layered.segments[0].endpoint, (1, 0))
        self.assertFalse(layered.details["search_complete"])
        self.assertFalse(layered.details["optimality_proven"])
        self.assertTrue(layered.details["fallback_used"])
        self.assertEqual(layered.details["fallback_reason"], "MAX_EXPANSIONS")

    def test_filters_arrivals_for_next_segment_hard_scope(self) -> None:
        scene = TinyScene(5, (0, 0))
        forbidden = region("forbidden", (0, 1))
        next_segment_constraint = GroundedConstraint(
            "forbid",
            (forbidden,),
            ("s2",),
            constraint_id="avoid_s2_start",
            hardness="hard",
        )
        program = GroundedProgram(
            (
                segment("s1", region("first", (0, 1), (1, 1))),
                segment("s2", region("final", (0, 2)), (next_segment_constraint,)),
            ),
            {},
        )
        with patch(
            "scripts.methods.groundplan.planner.semantic_astar._compile_soft",
            side_effect=zero_soft,
        ):
            layered = plan_grounded_program(
                scene,
                program,
                planner_mode="global_layered_dp",
                max_expansions=0,
            )
        self.assertEqual(layered.status, "success")
        self.assertEqual(layered.segments[0].endpoint, (1, 1))
        self.assertFalse(layered.details["fallback_used"])

    def test_overlapping_targets_require_one_move_per_stage(self) -> None:
        scene = TinyScene(4, (0, 0))
        overlap = region("overlap", (0, 1))
        program = GroundedProgram(
            (segment("s1", overlap), segment("s2", overlap)), {}
        )
        with patch(
            "scripts.methods.groundplan.planner.semantic_astar._compile_soft",
            side_effect=zero_soft,
        ):
            layered = plan_grounded_program(
                scene,
                program,
                planner_mode="global_layered_dp",
                max_expansions=0,
            )
        self.assertEqual(layered.status, "success")
        self.assertEqual(layered.segments[0].endpoint, (0, 1))
        self.assertEqual(layered.segments[1].endpoint, (0, 1))
        self.assertGreaterEqual(len(layered.trajectory), 4)

    def test_path_shape_stages_track_heading(self) -> None:
        scene = TinyScene(7, (0, 0))
        first_waypoint = GPRef(
            "region",
            "shape_1",
            cells=frozenset({(1, 2)}),
            construction={"type": "circle_waypoint", "index": 1, "count": 2},
        )
        second_waypoint = GPRef(
            "region",
            "shape_2",
            cells=frozenset({(2, 3)}),
            construction={"type": "circle_waypoint", "index": 2, "count": 2},
        )
        ordered = GroundedConstraint(
            "require_visit_in_order",
            (first_waypoint, second_waypoint),
            ("s1",),
            constraint_id="shape",
            hardness="hard",
        )
        program = GroundedProgram(
            (segment("s1", region("goal", (5, 5)), (ordered,)),), {}
        )
        with patch(
            "scripts.methods.groundplan.planner.semantic_astar._compile_soft",
            side_effect=zero_soft,
        ):
            layered = plan_grounded_program(
                scene,
                program,
                planner_mode="global_layered_dp",
                max_expansions=0,
            )
        self.assertEqual(layered.status, "success")
        self.assertIn([1, 2], layered.trajectory)
        self.assertIn([2, 3], layered.trajectory)
        self.assertTrue(layered.details["layer_details"][0]["track_heading"])
        self.assertTrue(layered.details["optimality_proven"])


class GlobalProgressAStarLimitTests(unittest.TestCase):
    def test_expansion_limit_retains_best_executable_prefix(self) -> None:
        scene = TinyScene(10, (0, 0))
        ordered = GroundedConstraint(
            "require_visit_in_order",
            (region("shape_1", (0, 2)), region("shape_2", (0, 5))),
            ("s1",),
            constraint_id="shape",
            hardness="hard",
        )
        program = GroundedProgram((segment("s1", region("goal", (9, 9)), (ordered,)),), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            planned = plan_grounded_program(scene, program, max_expansions=2, planner_mode="global_progress_astar")
        self.assertEqual(planned.status, "failed")
        self.assertEqual(planned.failure_reason, "MAX_EXPANSIONS")
        self.assertGreater(len(planned.trajectory), 1)
        self.assertGreaterEqual(planned.details["completed_stage_count"], 1)
        self.assertTrue(planned.details["partial_trajectory"])

    def test_late_empty_stage_retains_completed_segment_prefix(self) -> None:
        scene = TinyScene(7, (0, 0))
        program = GroundedProgram(
            (
                segment("s1", region("first", (0, 2))),
                segment("s2", region("empty")),
            ),
            {},
        )
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            planned = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(planned.status, "failed")
        self.assertEqual(planned.failure_reason, "EMPTY_TARGET_REGION")
        self.assertEqual(planned.trajectory[-1], [0, 2])
        self.assertEqual(planned.details["completed_stage_count"], 1)
        self.assertTrue(planned.details["partial_trajectory"])

    def test_first_empty_stage_keeps_structured_failure(self) -> None:
        scene = TinyScene(4, (0, 0))
        program = GroundedProgram((segment("s1", region("empty")),), {})
        with patch("scripts.methods.groundplan.planner.semantic_astar._compile_soft", side_effect=zero_soft):
            planned = plan_grounded_program(scene, program, planner_mode="global_progress_astar")
        self.assertEqual(planned.status, "failed")
        self.assertEqual(planned.failure_reason, "EMPTY_TARGET_REGION")
        self.assertEqual(planned.trajectory, [[0, 0]])

if __name__ == "__main__":
    unittest.main()
