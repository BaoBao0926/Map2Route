"""Tests for OSG-LLM scene graph construction."""

from __future__ import annotations

import unittest

from scripts.methods.osgllm.scene_graph.builder import build_scene_graph
from scripts.methods.osgllm.scene_graph.propositions import build_proposition_model
from scripts.methods.util.grid_astar import build_traversable_grid


def tiny_map_state() -> dict[str, object]:
    return {
        "grid_size": 5,
        "layer_legends": {"occupancy": {"free": {"value": 0}}},
        "layers": {
            "occupancy": [[0 for _ in range(5)] for _ in range(5)],
            "room": [
                [1, 1, 1, 2, 2],
                [1, 1, 1, 2, 2],
                [1, 1, 1, 2, 2],
                [1, 1, 1, 2, 2],
                [1, 1, 1, 2, 2],
            ],
            "object_instance": [
                [0, 0, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0],
                [0, 0, 0, 0, 0],
            ],
        },
        "room_instances": [
            {"id": 1, "category": "kitchen", "name": "kitchen_1"},
            {"id": 2, "category": "bedroom", "name": "bedroom_1"},
        ],
        "object_instances": [
            {"id": 3, "category": "chair", "name": "chair_1"},
            {"id": 4, "category": "lamp", "name": "lamp_1"},
        ],
    }


class SceneGraphTest(unittest.TestCase):
    def test_builds_attribute_regions_and_labels(self) -> None:
        map_state = tiny_map_state()
        traversable = build_traversable_grid(map_state)
        graph = build_scene_graph(map_state, traversable, object_reach_radius=1)

        self.assertIn("room_1", graph.regions)
        self.assertIn("room_2", graph.regions)
        self.assertIn("object_3", graph.regions)
        self.assertEqual(graph.object_room["object_4"], "room_2")
        self.assertIn("room_2", graph.room_adjacency["room_1"])

        model = build_proposition_model(graph)
        self.assertIn("enter(room_1)", model.labels_for_cell((0, 0)))
        self.assertIn("enter(room_2)", model.labels_for_cell((0, 4)))
        self.assertIn("reach(object_3)", model.labels_for_cell((1, 1)))


if __name__ == "__main__":
    unittest.main()

