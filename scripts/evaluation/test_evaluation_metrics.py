"""Focused regression tests for SemPathBench metric helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation.evaluate_prediction import apply_embedded_path_shape_annotations
from scripts.evaluation.evaluation_metrics import (
    build_instruction_metric_cache,
    compute_hcs,
    compute_segment_wise_spl,
    compute_soft_details,
)
from scripts.evaluation.metric_cache import build_map_metric_cache


def region(
    constraint_id: str,
    row: int,
    col: int,
    *,
    kind: str = "must_pass",
    order: int | None = None,
    scope: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "constraint_id": constraint_id,
        "kind": kind,
        "shape": "freeform",
        "cells": [[row, col]],
        "center": [row, col],
        "order": order,
    }
    if scope is not None:
        result["scope"] = scope
    return result


def map_state() -> dict[str, object]:
    return {
        "grid_size": 5,
        "layer_legends": {"occupancy": {"free": {"value": 0}}},
        "layers": {
            "occupancy": [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "object_instance": [[0] * 5 for _ in range(5)],
        },
        "object_instances": [],
    }


class EvaluationMetricsTest(unittest.TestCase):
    def test_segment_cache_avoids_on_demand_shortest_path_search(self) -> None:
        state = map_state()
        instruction = {
            "hard_constraints": [
                region("record", 1, 1, order=1),
                region("goal", 3, 3, order=2),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_path = root / "map.json"
            instruction_path = root / "instruction_000001.json"
            map_path.write_text(json.dumps(state), encoding="utf-8")
            instruction_path.write_text(json.dumps(instruction), encoding="utf-8")
            state["map_path"] = str(map_path)
            build_map_metric_cache(map_path, state)
            build_instruction_metric_cache(instruction_path, instruction, state)
            instruction["_metric_instruction_path"] = str(instruction_path)

            with patch(
                "scripts.evaluation.evaluation_metrics._shortest_legal_path_length",
                side_effect=AssertionError("cached distance field should be used"),
            ):
                score, details = compute_segment_wise_spl(
                    [[1, 1], [2, 2], [3, 3]], instruction, state
                )

        self.assertAlmostEqual(score or 0.0, 1.0, places=6)
        self.assertEqual(details[0]["status"], "completed")
        self.assertAlmostEqual(float(details[0]["L_star"]), float(details[0]["L"]), places=6)

    def test_global_must_avoid_zeroes_hcs_after_goal_completion(self) -> None:
        hard = [
            region("record", 0, 0, order=1),
            region("goal", 0, 1, order=2),
            region("global_avoid", 0, 2, kind="must_avoid"),
        ]
        score, details = compute_hcs([[0, 0], [0, 1], [0, 2]], hard)

        self.assertEqual(score, 0.0)
        self.assertEqual(details["failure_type"], "global_must_avoid_violation")
        self.assertEqual(details["global_must_avoid_violations"], ["global_avoid"])

    def test_near_empty_region_uses_farthest_region_cell(self) -> None:
        constraint = {
            "constraint_id": "near",
            "preference_type": "near_preference",
            "shape": "freeform",
            "cells": [[0, 1], [0, 2]],
            "reference_region": {"mode": "object", "object_id": 1, "cells": [[0, 0]]},
        }
        details = compute_soft_details([[4, 4]], [], [constraint], map_state())

        self.assertEqual(details[0]["status"], "empty_region_max_distance")
        self.assertEqual(details[0]["raw_value"], 2.0)

    def test_distance_measurements_use_map_metres(self) -> None:
        constraint = {
            "constraint_id": "near",
            "preference_type": "near_preference",
            "shape": "freeform",
            "cells": [[0, 2]],
            "reference_region": {"mode": "object", "object_id": 1, "cells": [[0, 0]]},
        }
        state = map_state()
        state["metadata"] = {
            "grid_coordinate_frame": {"resolution": 0.5},
        }
        details = compute_soft_details([[0, 2]], [], [constraint], state)

        self.assertEqual(details[0]["raw_value"], 1.0)
        self.assertEqual(details[0]["unit"], "m")

    def test_far_empty_region_is_zero(self) -> None:
        constraint = {
            "constraint_id": "far",
            "preference_type": "far_preference",
            "shape": "freeform",
            "cells": [[0, 1]],
            "reference_region": {"mode": "object", "object_id": 1, "cells": [[0, 0]]},
        }
        details = compute_soft_details([[4, 4]], [], [constraint], map_state())

        self.assertEqual(details[0]["status"], "empty_region_zero_distance")
        self.assertEqual(details[0]["raw_value"], 0.0)

    def test_relative_reports_ratio_of_mean_footprint_distances(self) -> None:
        constraint = {
            "constraint_id": "relative",
            "preference_type": "relative_preference",
            "shape": "freeform",
            "cells": [[1, 1], [1, 2]],
            "reference_regions": [
                {"object_id": 1, "cells": [[1, 0]]},
                {"object_id": 2, "cells": [[1, 5]]},
            ],
        }
        details = compute_soft_details([[1, 1], [1, 2]], [], [constraint], map_state())

        self.assertAlmostEqual(float(details[0]["D_A"]), 1.5)
        self.assertAlmostEqual(float(details[0]["D_B"]), 3.5)
        self.assertAlmostEqual(float(details[0]["raw_value"]), 3.5 / 1.5)

    def test_relative_empty_region_is_zero(self) -> None:
        constraint = {
            "constraint_id": "relative",
            "preference_type": "relative_preference",
            "shape": "freeform",
            "cells": [[1, 1], [1, 2]],
            "reference_regions": [
                {"object_id": 1, "cells": [[1, 0]]},
                {"object_id": 2, "cells": [[1, 5]]},
            ],
        }
        details = compute_soft_details([[4, 4]], [], [constraint], map_state())

        self.assertEqual(details[0]["status"], "empty_region_zero_score")
        self.assertEqual(details[0]["raw_value"], 0.0)
        self.assertEqual(details[0]["region_waypoint_count"], 0)

    def test_scoped_relative_requires_completed_segment(self) -> None:
        hard = [
            region("record", 0, 0, order=1),
            region("goal", 4, 4, order=2),
        ]
        constraint = {
            "constraint_id": "relative",
            "preference_type": "relative_preference",
            "shape": "freeform",
            "cells": [[1, 1]],
            "reference_regions": [
                {"object_id": 1, "cells": [[1, 0]]},
                {"object_id": 2, "cells": [[1, 4]]},
            ],
            "scope": {
                "type": "between_hard_constraints",
                "from_order": 1,
                "to_order": 2,
            },
        }

        details = compute_soft_details(
            [[0, 0], [1, 1]],
            hard,
            [constraint],
            map_state(),
        )

        self.assertEqual(details[0]["status"], "segment_not_completed_worst_case")
        self.assertEqual(details[0]["raw_value"], 0.0)
        self.assertEqual(details[0]["region_waypoint_count"], 0)
        self.assertEqual(details[0]["active_waypoint_count"], 0)
        self.assertEqual(details[0]["ungated_active_waypoint_count"], 2)
        self.assertFalse(details[0]["segment_succeeded"])

    def test_scoped_relative_scores_completed_segment(self) -> None:
        hard = [
            region("record", 0, 0, order=1),
            region("goal", 1, 2, order=2),
        ]
        constraint = {
            "constraint_id": "relative",
            "preference_type": "relative_preference",
            "shape": "freeform",
            "cells": [[1, 1]],
            "reference_regions": [
                {"object_id": 1, "cells": [[1, 0]]},
                {"object_id": 2, "cells": [[1, 4]]},
            ],
            "scope": {
                "type": "between_hard_constraints",
                "from_order": 1,
                "to_order": 2,
            },
        }

        details = compute_soft_details(
            [[0, 0], [1, 1], [1, 2]],
            hard,
            [constraint],
            map_state(),
        )

        self.assertEqual(details[0]["status"], "computed")
        self.assertAlmostEqual(float(details[0]["raw_value"]), 3.0)
        self.assertEqual(details[0]["region_waypoint_count"], 1)
        self.assertTrue(details[0]["segment_succeeded"])

    def test_scoped_relative_rejects_target_reached_after_active_avoid(self) -> None:
        hard = [
            region("record", 0, 0, order=1),
            region("goal", 0, 2, order=2),
            region(
                "avoid",
                0,
                1,
                kind="must_avoid",
                scope={
                    "type": "between_hard_constraints",
                    "from_order": 1,
                    "to_order": 2,
                },
            ),
        ]
        constraint = {
            "constraint_id": "relative",
            "preference_type": "relative_preference",
            "shape": "freeform",
            "cells": [[0, 1]],
            "reference_regions": [
                {"object_id": 1, "cells": [[1, 1]]},
                {"object_id": 2, "cells": [[4, 4]]},
            ],
            "scope": {
                "type": "between_hard_constraints",
                "from_order": 1,
                "to_order": 2,
            },
        }

        details = compute_soft_details(
            [[0, 0], [0, 1], [0, 2]],
            hard,
            [constraint],
            map_state(),
        )

        self.assertEqual(details[0]["status"], "segment_not_completed_worst_case")
        self.assertEqual(details[0]["raw_value"], 0.0)
        self.assertFalse(details[0]["segment_succeeded"])

    def test_geometric_path_empty_region_is_zero(self) -> None:
        constraint = {
            "constraint_id": "shape",
            "preference_type": "geometric_path_preference",
            "shape": "freeform",
            "cells": [[1, 1], [1, 2]],
            "reference_trajectory": [[1, 1], [1, 2]],
        }
        details = compute_soft_details([[4, 4]], [], [constraint], map_state())

        self.assertEqual(details[0]["status"], "empty_region_zero_score")
        self.assertEqual(details[0]["raw_value"], 0.0)
        self.assertEqual(details[0]["nDTW"], 0.0)
        self.assertEqual(details[0]["region_waypoint_count"], 0)

    def test_clearance_is_mean_distance_not_inverse_square_cost(self) -> None:
        constraint = {"constraint_id": "clearance", "preference_type": "clearance"}
        details = compute_soft_details([[0, 1], [0, 2]], [], [constraint], map_state())

        self.assertEqual(details[0]["preference_type"], "clearance")
        self.assertAlmostEqual(float(details[0]["raw_value"]), 1.5)

    def test_unfinished_final_segment_is_zero_in_segment_wise_spl(self) -> None:
        instruction = {
            "hard_constraints": [
                region("record", 1, 1, order=1),
                region("a", 1, 2, order=2),
                region("b", 1, 3, order=3),
            ]
        }
        score, details = compute_segment_wise_spl(
            [[1, 1], [1, 2]],
            instruction,
            map_state(),
        )

        self.assertEqual(details[0]["S"], 1)
        self.assertEqual(details[0]["SPL"], 1.0)
        self.assertEqual(details[1]["S"], 0)
        self.assertEqual(details[1]["SPL"], 0.0)
        self.assertEqual(score, 0.5)


    def test_embedded_path_shape_annotation_overrides_legacy_reference(self) -> None:
        constraint = {
            "constraint_id": "shape",
            "preference_type": "path_shape_preference",
            "shape": "freeform",
            "cells": [[1, 1], [1, 2], [1, 3]],
            "reference_trajectory": [[3, 1], [3, 2]],
            "path_shape_annotation": {
                "version": 1,
                "annotation_type": "canonical_path_shape_reference",
                "shape_spec": {
                    "type": "expert_polyline",
                    "direction": "forward",
                    "ordered_control_points": [[1, 1], [1, 3]],
                },
                "shape_reference_trajectory": [[1, 1], [1, 2], [1, 3]],
                "shape_reference_source": "human_expert_canonical_v1",
            },
        }
        instruction: dict[str, object] = {"soft_constraints": [constraint]}
        instruction_path = Path("/tmp/instruction_000001.json")

        loaded = apply_embedded_path_shape_annotations(
            instruction,
            grid_size=5,
            instruction_path=instruction_path,
        )
        details = compute_soft_details(
            [[1, 1], [1, 2], [1, 3]],
            [],
            [constraint],
            map_state(),
        )

        self.assertEqual(len(loaded), 1)
        self.assertEqual(constraint["reference_trajectory"], [[1, 1], [1, 2], [1, 3]])
        self.assertAlmostEqual(float(details[0]["nDTW"]), 1.0)
        self.assertEqual(
            details[0]["reference_trajectory_source"],
            "human_expert_canonical_v1",
        )
        self.assertEqual(details[0]["reference_waypoint_count"], 3)
        self.assertEqual(
            details[0]["reference_trajectory_file"],
            str(instruction_path),
        )


if __name__ == "__main__":
    unittest.main()
