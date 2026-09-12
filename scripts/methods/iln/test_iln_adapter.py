from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.methods.iln.config import METHOD_NAME, METHOD_ROOT
from scripts.methods.iln.graph_builder import build_area_passage_graph
from scripts.methods.iln.graph_planner import plan_route
from scripts.methods.iln.grounding_adapter import (
    _parse_strict_destination,
    heuristic_strict_grounding,
)
from scripts.methods.iln.passage_cost_evaluator import _navigation_task
from scripts.methods.iln.pipeline import ILNRunConfig, run_iln_pipeline
from scripts.methods.iln.run import saved_prediction_is_current_iln


def toy_map_state() -> dict[str, object]:
    occupancy = [[0 for _ in range(6)] for _ in range(6)]
    room = [
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 4, 4, 4],
    ]
    object_instance = [[0 for _ in range(6)] for _ in range(6)]
    object_instance[4][4] = 7
    return {
        "grid_size": 6,
        "layer_legends": {"occupancy": {"free": {"value": 0}}},
        "layers": {
            "occupancy": occupancy,
            "room": room,
            "object_instance": object_instance,
        },
        "room_instances": [
            {"id": 1, "category": "kitchen", "name": "kitchen_1"},
            {"id": 2, "category": "living_room", "name": "living_room_1"},
            {"id": 3, "category": "bathroom", "name": "bathroom_1"},
            {"id": 4, "category": "bedroom", "name": "bedroom_1"},
        ],
        "object_instances": [
            {"id": 7, "category": "basket_ball", "name": "basket_ball_1"},
        ],
    }


class ILNAdapterTests(unittest.TestCase):
    def test_single_iln_identity_and_output_root(self) -> None:
        self.assertEqual(METHOD_NAME, "ILN")
        self.assertEqual(METHOD_ROOT.name, "ILN")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.json"
            path.write_text(json.dumps({"method": "ILN"}), encoding="utf-8")
            self.assertTrue(saved_prediction_is_current_iln(path))
            path.write_text(
                json.dumps({"method": "ILN-Adapted"}),
                encoding="utf-8",
            )
            self.assertFalse(saved_prediction_is_current_iln(path))

    def test_passage_extraction_finds_room_boundaries(self) -> None:
        graph = build_area_passage_graph(
            toy_map_state(),
            {"start_pose": {"row": 0, "col": 0}},
        )
        self.assertEqual(graph.start_area, 1)
        self.assertGreaterEqual(len(graph.passages), 4)
        self.assertTrue(any(passage.rooms == (1, 2) for passage in graph.passages.values()))

    def test_graph_planner_same_area_bypasses_passages(self) -> None:
        graph = build_area_passage_graph(
            toy_map_state(),
            {"start_pose": {"row": 0, "col": 0}},
        )
        route = plan_route(
            graph,
            start_area=1,
            goal_area=1,
            door_costs={},
        )
        self.assertEqual(route.status, "same_area_direct_realization")
        self.assertEqual(route.passage_sequence, [])

    def test_graph_planner_uses_passage_costs(self) -> None:
        graph = build_area_passage_graph(
            toy_map_state(),
            {"start_pose": {"row": 0, "col": 0}},
        )
        route = plan_route(
            graph,
            start_area=1,
            goal_area=4,
            door_costs={},
            unknown_passage_cost=10.0,
        )
        self.assertEqual(route.status, "success")
        self.assertGreaterEqual(len(route.passage_sequence), 2)

    def test_navigation_task_mismatch_is_repaired(self) -> None:
        task, diagnostics = _navigation_task(
            {"navigation_task": ["room_9", "room_8"]},
            "room_1",
            "room_2",
        )
        self.assertEqual(task, ["room_1", "room_2"])
        self.assertIn("PCE_NAVIGATION_TASK_MISMATCH", diagnostics)

    def test_destination_is_one_area_only(self) -> None:
        map_state = toy_map_state()
        graph = build_area_passage_graph(map_state, {"start_pose": {"row": 0, "col": 0}})
        result = _parse_strict_destination(
            {"destination_area": "room_4", "reason": "final destination"},
            graph,
        )
        self.assertEqual(len(result.stages), 1)
        self.assertEqual(result.stages[0].target_type, "area")
        self.assertEqual(result.stages[0].area_id, 4)
        self.assertIsNone(result.stages[0].object_id)
        self.assertEqual(result.adapter_graph_constraints, {})

    def test_heuristic_keeps_only_final_destination_area(self) -> None:
        map_state = toy_map_state()
        graph = build_area_passage_graph(map_state, {"start_pose": {"row": 0, "col": 0}})
        result = heuristic_strict_grounding(
            map_state,
            graph,
            {
                "instruction": "Pass through the living room, then reach the basket ball.",
                "start_pose": {"row": 0, "col": 0},
            },
        )
        self.assertEqual(len(result.stages), 1)
        self.assertEqual(result.stages[0].target_type, "area")
        self.assertEqual(result.stages[0].area_id, 4)
        self.assertIsNone(result.stages[0].object_id)
        self.assertEqual(result.adapter_graph_constraints, {})

    def test_pipeline_has_no_object_goal_or_adapter_constraints(self) -> None:
        map_state = toy_map_state()
        result = run_iln_pipeline(
            map_state,
            {
                "instruction": "Reach the basket ball.",
                "start_pose": {"row": 0, "col": 0},
            },
            config=ILNRunConfig(grounding_mode="heuristic"),
        )
        self.assertTrue(result.success)
        self.assertNotIn("variant", result.metadata)
        grounding = result.metadata["grounding_adapter"]
        self.assertEqual(len(grounding["stages"]), 1)
        self.assertEqual(grounding["stages"][0]["target_type"], "area")
        self.assertNotIn("object_id", grounding["stages"][0])
        self.assertEqual(grounding["adapter_graph_constraints"], {})

if __name__ == "__main__":
    unittest.main()
