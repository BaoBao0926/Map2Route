"""Regression tests for Tutorial route construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.methods.tutorial.utils import (
    collect_prediction_results,
    connect_waypoints_with_astar,
    tutorial_waypoints,
)
from scripts.methods.util.grid_astar import build_traversable_grid


class TutorialAvoidPlanningTest(unittest.TestCase):
    def test_traversable_grid_accepts_mapping_backed_rows(self) -> None:
        map_state = {
            "grid_size": 3,
            "layer_legends": {"occupancy": {"free": {"value": 0}}},
            "layers": {
                "occupancy": [
                    [0, 1, 0],
                    {"0": 0, "1": 0, "2": 1},
                    {0: 1, 1: 0},
                ]
            },
        }

        traversable = build_traversable_grid(map_state)

        self.assertEqual(
            traversable,
            [
                [True, False, True],
                [True, True, False],
                [False, True, False],
            ],
        )

    def test_astar_detours_around_active_global_must_avoid(self) -> None:
        map_state = {
            "grid_size": 5,
            "layer_legends": {"occupancy": {"free": {"value": 0}}},
            "layers": {"occupancy": [[0] * 5 for _ in range(5)]},
        }
        instruction = {
            "start_pose": {"row": 2, "col": 0},
            "hard_constraints": [
                {
                    "constraint_id": "record",
                    "kind": "must_pass",
                    "order": 1,
                    "shape": "freeform",
                    "cells": [[2, 0]],
                    "center": [2, 0],
                },
                {
                    "constraint_id": "goal",
                    "kind": "must_pass",
                    "order": 2,
                    "shape": "freeform",
                    "cells": [[2, 4]],
                    "center": [2, 4],
                },
                {
                    "constraint_id": "avoid",
                    "kind": "must_avoid",
                    "shape": "freeform",
                    "cells": [[2, 2]],
                    "center": [2, 2],
                    "scope": {"type": "global"},
                },
            ],
        }
        traversable = build_traversable_grid(map_state)
        waypoints, _details, transitions = tutorial_waypoints(
            map_state,
            traversable,
            instruction,
        )
        trajectory, segments = connect_waypoints_with_astar(
            traversable,
            waypoints,
            instruction["hard_constraints"],
            transitions,
        )

        self.assertNotIn([2, 2], trajectory)
        self.assertNotEqual(trajectory, [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]])
        self.assertEqual(segments[0]["active_must_avoid_ids"], ["avoid"])

    def test_stops_at_first_legal_target_region_cell(self) -> None:
        map_state = {
            "grid_size": 7,
            "layer_legends": {"occupancy": {"free": {"value": 0}}},
            "layers": {"occupancy": [[0] * 7 for _ in range(7)]},
        }
        instruction = {
            "start_pose": {"row": 3, "col": 0},
            "hard_constraints": [
                {
                    "constraint_id": "record",
                    "kind": "must_pass",
                    "order": 1,
                    "shape": "freeform",
                    "cells": [[3, 0]],
                    "center": [3, 0],
                },
                {
                    "constraint_id": "wide_goal",
                    "kind": "must_pass",
                    "order": 2,
                    "shape": "freeform",
                    "cells": [[3, 4], [3, 5]],
                    "center": [3, 5],
                },
            ],
        }
        traversable = build_traversable_grid(map_state)
        waypoints, _details, transitions = tutorial_waypoints(
            map_state,
            traversable,
            instruction,
        )
        trajectory, segments = connect_waypoints_with_astar(
            traversable,
            waypoints,
            instruction["hard_constraints"],
            transitions,
        )

        self.assertEqual(trajectory[-1], [3, 4])
        self.assertEqual(segments[0]["status"], "connected")


class PredictionCollectionTest(unittest.TestCase):
    def test_ignores_intermediate_steps_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            map_id = "procthor/001_valunseen"
            prediction_dir = output_root / map_id
            prediction_dir.mkdir(parents=True)
            prediction = {
                "method": "GroundPlan",
                "map_id": map_id,
                "scene_id": map_id,
                "instruction_id": "instruction_000001",
                "difficulty_level": "easy",
                "trajectory": [[0, 0]],
                "metrics": {
                    "HCS": 1.0,
                    "SCS": 1.0,
                    "PL": 0.0,
                    "segment_wise_SPL": 1.0,
                },
            }
            (prediction_dir / "instruction_000001.json").write_text(
                json.dumps(prediction), encoding="utf-8"
            )
            (prediction_dir / "instruction_000001.steps.json").write_text(
                json.dumps(
                    {
                        "method": "GroundPlan",
                        "map_id": map_id,
                        "scene_id": map_id,
                        "instruction_id": "instruction_000001",
                        "steps": {},
                    }
                ),
                encoding="utf-8",
            )

            results = collect_prediction_results(
                output_root, method_name="GroundPlan"
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["instruction_id"], "instruction_000001")
            self.assertIn("metrics_summary", results[0])


if __name__ == "__main__":
    unittest.main()
