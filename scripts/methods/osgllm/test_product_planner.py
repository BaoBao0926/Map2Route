"""Tests for the OSG-LLM product planner."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.methods.lang2ltl.ltl import Formula, ltl_eventually
from scripts.methods.osgllm.config import OSGLLMConfig
from scripts.methods.osgllm.language.llm_adapter import _parse_llm_prefix, translate_instruction_llm
from scripts.methods.osgllm.pipeline import build_osgllm_trajectory
from scripts.methods.osgllm.planning.hierarchical_planner import hierarchy_guided_plan
from scripts.methods.osgllm.planning.product_planner import product_astar
from scripts.methods.osgllm.planning.sequential_fallback import sequential_ap_astar
from scripts.methods.osgllm.scene_graph.builder import build_scene_graph
from scripts.methods.osgllm.scene_graph.propositions import build_proposition_model
from scripts.methods.osgllm.test_scene_graph import tiny_map_state
from scripts.methods.util.grid_astar import build_traversable_grid


class ProductPlannerTest(unittest.TestCase):
    def test_formula_hash_compatibility_alias_is_structural(self) -> None:
        formula = ltl_eventually(Formula("ap", value="reach(object_4)"))

        self.assertEqual(formula.hash, hash(formula))
        self.assertEqual(
            formula.hash,
            ltl_eventually(Formula("ap", value="reach(object_4)")).hash,
        )

    def test_reaches_room_goal(self) -> None:
        map_state = tiny_map_state()
        traversable = build_traversable_grid(map_state)
        graph = build_scene_graph(map_state, traversable, object_reach_radius=1)
        formula = ltl_eventually(Formula("ap", value="enter(room_2)"))
        model = build_proposition_model(graph, used_aps={"enter(room_2)"})

        result = product_astar(
            traversable,
            model,
            (0, 0),
            formula,
            max_expansions=1000,
            max_seconds=2.0,
        )

        self.assertEqual(result.status, "SUCCESS")
        self.assertGreater(len(result.trajectory), 1)
        self.assertGreaterEqual(result.trajectory[-1][1], 3)

    def test_reaches_object_goal(self) -> None:
        map_state = tiny_map_state()
        traversable = build_traversable_grid(map_state)
        graph = build_scene_graph(map_state, traversable, object_reach_radius=1)
        formula = ltl_eventually(Formula("ap", value="reach(object_4)"))
        model = build_proposition_model(graph, used_aps={"reach(object_4)"})

        result = product_astar(
            traversable,
            model,
            (0, 0),
            formula,
            max_expansions=1000,
            max_seconds=2.0,
        )

        self.assertEqual(result.status, "SUCCESS")
        final_cell = tuple(result.trajectory[-1])
        self.assertIn("reach(object_4)", model.labels_for_cell(final_cell))

    def test_product_astar_accepts_unlimited_budgets(self) -> None:
        map_state = tiny_map_state()
        traversable = build_traversable_grid(map_state)
        graph = build_scene_graph(map_state, traversable, object_reach_radius=1)
        formula = ltl_eventually(Formula("ap", value="reach(object_4)"))
        model = build_proposition_model(graph, used_aps={"reach(object_4)"})

        result = product_astar(
            traversable,
            model,
            (0, 0),
            formula,
            max_expansions=None,
            max_seconds=None,
        )

        self.assertEqual(result.status, "SUCCESS")
        self.assertIsNone(result.details["max_expansions"])
        self.assertIsNone(result.details["timeout_seconds"])

    def test_strict_pipeline_does_not_fallback(self) -> None:
        trajectory, metadata = build_osgllm_trajectory(
            tiny_map_state(),
            {
                "instruction": "Go to the lamp.",
                "start_pose": {"row": 0, "col": 0},
            },
            config=OSGLLMConfig(
                object_reach_radius=0,
                max_expansions=1,
                max_planning_seconds=2.0,
                disable_fallback=True,
            ),
        )

        self.assertEqual(trajectory, [[0, 0]])
        self.assertEqual(metadata["status"], "PLANNER_TIMEOUT")
        self.assertEqual(
            metadata["planner"]["planner_type"],
            "occupancy_product_anchor_astar_strict",
        )
        self.assertTrue(metadata["planner"]["fallback_disabled"])

    def test_direct_astar_pipeline_bypasses_product_state(self) -> None:
        trajectory, metadata = build_osgllm_trajectory(
            tiny_map_state(),
            {
                "instruction": "Go to the lamp.",
                "start_pose": {"row": 0, "col": 0},
            },
            config=OSGLLMConfig(
                planner="astar",
                object_reach_radius=0,
                max_expansions=1,
                max_planning_seconds=2.0,
            ),
        )

        self.assertEqual(metadata["status"], "SUCCESS")
        self.assertGreater(len(trajectory), 1)
        self.assertEqual(
            metadata["planner"]["planner_type"],
            "sequential_grid_astar",
        )

    def test_llm_cache_only_rejects_missing_entry(self) -> None:
        map_state = tiny_map_state()
        traversable = build_traversable_grid(map_state)
        graph = build_scene_graph(map_state, traversable, object_reach_radius=1)

        with TemporaryDirectory() as cache_dir:
            with self.assertRaisesRegex(RuntimeError, "Required cached LLM"):
                translate_instruction_llm(
                    graph,
                    {"instruction": "Go to the lamp."},
                    start=(0, 0),
                    model=None,
                    cache_root=Path(cache_dir),
                    overwrite_cache=False,
                    cache_only=True,
                )

    def test_sequential_fallback_connects_ap_goals(self) -> None:
        map_state = tiny_map_state()
        traversable = build_traversable_grid(map_state)
        graph = build_scene_graph(map_state, traversable, object_reach_radius=1)
        model = build_proposition_model(
            graph,
            used_aps={"enter(room_2)", "reach(object_4)"},
        )

        result = sequential_ap_astar(
            traversable,
            model,
            (0, 0),
            ("enter(room_2)", "reach(object_4)"),
        )

        self.assertEqual(result.status, "SUCCESS")
        final_cell = tuple(result.trajectory[-1])
        self.assertIn("reach(object_4)", model.labels_for_cell(final_cell))

    def test_sequential_fallback_honours_zero_time_budget(self) -> None:
        map_state = tiny_map_state()
        traversable = build_traversable_grid(map_state)
        graph = build_scene_graph(map_state, traversable, object_reach_radius=1)
        model = build_proposition_model(graph, used_aps={"enter(room_2)"})

        result = sequential_ap_astar(
            traversable,
            model,
            (0, 0),
            ("enter(room_2)",),
            max_seconds=0.0,
        )

        self.assertEqual(result.status, "PLANNER_TIMEOUT")
        self.assertEqual(result.details["status"], "timeout")

    def test_llm_prefix_parser_preserves_function_style_aps(self) -> None:
        map_state = tiny_map_state()
        traversable = build_traversable_grid(map_state)
        graph = build_scene_graph(map_state, traversable, object_reach_radius=1)

        formula = _parse_llm_prefix(
            "F & enter(room_2) F reach(object_4)",
            graph,
        )

        self.assertIn("enter(room_2)", str(formula))
        self.assertIn("reach(object_4)", str(formula))

    def test_hierarchy_guided_plan_records_hierarchy_expansion(self) -> None:
        map_state = tiny_map_state()
        traversable = build_traversable_grid(map_state)
        graph = build_scene_graph(map_state, traversable, object_reach_radius=1)
        model = build_proposition_model(
            graph,
            used_aps={"enter(room_2)", "reach(object_4)"},
        )

        result = hierarchy_guided_plan(
            graph,
            traversable,
            model,
            (0, 0),
            ("enter(room_2)", "reach(object_4)"),
        )

        self.assertEqual(result.status, "SUCCESS")
        self.assertGreater(result.details["expanded_states"]["room"], 0)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
