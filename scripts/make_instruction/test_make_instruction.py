#!/usr/bin/env python3
"""Tests for instruction annotation schema migrations."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import scripts.make_instruction.make_instruction as make_instruction
from scripts.make_instruction.make_instruction import (
    annotate_soft_reference_distances,
    collect_template_object_centers,
    global_constraint_usage_summary,
    global_difficulty_totals,
    mean_squared_turning_angle,
    mean_unique_waypoint_distance,
    object_cells_for_map,
    route_collision_violations_for_map,
    sync_labeled_instructions_with_templates,
    validate_map_payload,
    validate_sample,
    validate_soft_constraints,
)


def base_constraint() -> dict[str, object]:
    return {
        "constraint_id": "soft_1",
        "preference_type": "near_preference",
        "shape": "circle",
        "center": [10, 20],
        "radius": 4,
        "width": 8,
        "height": 8,
        "object_ids": [],
        "cells": [],
    }


class SoftReferenceRegionTest(unittest.TestCase):
    def test_map_list_orders_valunseen_before_train(self) -> None:
        original_map_root = make_instruction.MAP_ROOT
        original_map_dir = make_instruction.MAP_DIR
        original_registry_path = make_instruction.OBJECT_COLOR_REGISTRY_PATH
        with TemporaryDirectory(dir=make_instruction.REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            make_instruction.MAP_ROOT = root
            make_instruction.MAP_DIR = root / "simple_demo"
            make_instruction.OBJECT_COLOR_REGISTRY_PATH = (
                root / "object_color_registry.json"
            )
            for map_key in (
                "procthor/001_train",
                "procthor/006_valunseen",
                "procthor/003_valunseen",
                "simple_demo/simple_demo_map",
            ):
                parts = map_key.split("/")
                map_dir = (
                    root / parts[0] / ("valunseen" if parts[-1].endswith("_valunseen") else "train") / parts[-1]
                    if parts[0] == "procthor"
                    else root / map_key
                )
                map_dir.mkdir(parents=True)
                (map_dir / f"{map_dir.name}.json").write_text(
                    "{}",
                    encoding="utf-8",
                )

            try:
                map_ids = make_instruction.list_map_ids()
                summary_ids = [
                    summary["map_id"]
                    for summary in make_instruction.list_map_summaries()
                ]
            finally:
                make_instruction.MAP_ROOT = original_map_root
                make_instruction.MAP_DIR = original_map_dir
                make_instruction.OBJECT_COLOR_REGISTRY_PATH = original_registry_path

        self.assertEqual(
            map_ids,
            [
                "procthor/003_valunseen",
                "procthor/006_valunseen",
                "procthor/001_train",
                "simple_demo/simple_demo_map",
            ],
        )
        self.assertEqual(summary_ids, map_ids)

    def test_global_difficulty_totals_counts_saved_instruction_files(self) -> None:
        original_root = make_instruction.INSTRUCTION_ROOT
        with TemporaryDirectory() as tmp_dir:
            make_instruction.INSTRUCTION_ROOT = Path(tmp_dir)
            instruction_dir = (
                Path(tmp_dir) / "procthor" / "001_train" / "instruction_files"
            )
            valunseen_instruction_dir = (
                Path(tmp_dir) / "procthor" / "006_valunseen" / "instruction_files"
            )
            instruction_dir.mkdir(parents=True)
            valunseen_instruction_dir.mkdir(parents=True)
            (instruction_dir / "instruction_000001.json").write_text(
                '{"difficulty_level": "easy"}',
                encoding="utf-8",
            )
            (instruction_dir / "instruction_000002.json").write_text(
                '{"difficulty_level": "Hard"}',
                encoding="utf-8",
            )
            (Path(tmp_dir) / "procthor" / "001_train" / "instruction_000002.json").write_text(
                '{"difficulty_level": "extreme"}',
                encoding="utf-8",
            )
            (Path(tmp_dir) / "procthor" / "001_train" / "instruction_000004.json").write_text(
                '{"difficulty_level": "extreme"}',
                encoding="utf-8",
            )
            (valunseen_instruction_dir / "instruction_000001.json").write_text(
                '{"difficulty_level": "extreme"}',
                encoding="utf-8",
            )
            unknown_instruction_dir = (
                Path(tmp_dir) / "simple_demo" / "map_1" / "instruction_files"
            )
            unknown_instruction_dir.mkdir(parents=True)
            (unknown_instruction_dir / "instruction_000001.json").write_text(
                '{"difficulty_level": "easy"}',
                encoding="utf-8",
            )
            (instruction_dir / "instruction_000003.json").write_text(
                '{"difficulty_level": "medium"}',
                encoding="utf-8",
            )
            (instruction_dir / "bad.json").write_text(
                '{"difficulty_level": "extreme"}',
                encoding="utf-8",
            )

            try:
                totals = global_difficulty_totals()
            finally:
                make_instruction.INSTRUCTION_ROOT = original_root

        self.assertEqual(
            totals["total"],
            {"easy": 1, "extreme": 2, "hard": 1},
        )
        self.assertEqual(
            totals["by_split"]["train"],
            {"easy": 1, "extreme": 1, "hard": 1},
        )
        self.assertEqual(
            totals["by_split"]["valunseen"],
            {"easy": 0, "extreme": 1, "hard": 0},
        )

    def test_global_constraint_usage_counts_soft_types_and_must_avoid(self) -> None:
        original_root = make_instruction.INSTRUCTION_ROOT
        with TemporaryDirectory() as tmp_dir:
            make_instruction.INSTRUCTION_ROOT = Path(tmp_dir)
            instruction_dir = (
                Path(tmp_dir) / "procthor" / "001_train" / "instruction_files"
            )
            valunseen_instruction_dir = (
                Path(tmp_dir) / "procthor" / "006_valunseen" / "instruction_files"
            )
            instruction_dir.mkdir(parents=True)
            valunseen_instruction_dir.mkdir(parents=True)
            (instruction_dir / "instruction_000001.json").write_text(
                """
                {
                  "soft_constraints": [
                    {"preference_type": "near_preference"},
                    {"preference_type": "far_preference"},
                    {"preference_type": "relative_preference"},
                    {"preference_type": "move_smoothness"},
                    {"preference_type": "clearance"},
                    {"preference_type": "path_shape_preference"}
                  ],
                  "hard_constraints": [
                    {"kind": "must_avoid"},
                    {"kind": "must_pass"}
                  ]
                }
                """,
                encoding="utf-8",
            )
            (instruction_dir / "instruction_000002.json").write_text(
                """
                {
                  "soft_constraints": [
                    {"preference_type": "near_preference"}
                  ],
                  "hard_constraints": [
                    {"kind": "must_avoid"}
                  ]
                }
                """,
                encoding="utf-8",
            )
            (valunseen_instruction_dir / "instruction_000001.json").write_text(
                """
                {
                  "soft_constraints": [
                    {"preference_type": "path_shape_preference"}
                  ],
                  "hard_constraints": [
                    {"kind": "must_avoid"}
                  ]
                }
                """,
                encoding="utf-8",
            )
            unknown_instruction_dir = (
                Path(tmp_dir) / "simple_demo" / "map_1" / "instruction_files"
            )
            unknown_instruction_dir.mkdir(parents=True)
            (unknown_instruction_dir / "instruction_000001.json").write_text(
                """
                {
                  "soft_constraints": [
                    {"preference_type": "near_preference"}
                  ],
                  "hard_constraints": [
                    {"kind": "must_avoid"}
                  ]
                }
                """,
                encoding="utf-8",
            )

            try:
                summary = global_constraint_usage_summary()
            finally:
                make_instruction.INSTRUCTION_ROOT = original_root

        self.assertEqual(summary["total"], 9)
        self.assertEqual(
            [item["key"] for item in summary["items"]],
            [
                "must_avoid",
                "near_preference",
                "far_preference",
                "relative_preference",
                "path_shape_preference",
            ],
        )
        by_key = {item["key"]: item for item in summary["items"]}
        self.assertEqual(by_key["near_preference"]["count"], 2)
        self.assertEqual(by_key["must_avoid"]["count"], 3)
        self.assertEqual(by_key["path_shape_preference"]["count"], 2)
        self.assertIsNone(by_key["must_avoid"]["percentage"])
        self.assertAlmostEqual(
            by_key["near_preference"]["percentage"],
            2 * 100 / 6,
        )
        train_by_key = {
            item["key"]: item
            for item in summary["by_split"]["train"]["items"]
        }
        valunseen_by_key = {
            item["key"]: item
            for item in summary["by_split"]["valunseen"]["items"]
        }
        self.assertEqual(summary["by_split"]["train"]["total"], 7)
        self.assertEqual(summary["by_split"]["valunseen"]["total"], 2)
        self.assertEqual(train_by_key["near_preference"]["count"], 2)
        self.assertEqual(train_by_key["path_shape_preference"]["count"], 1)
        self.assertEqual(valunseen_by_key["must_avoid"]["count"], 1)
        self.assertIsNone(valunseen_by_key["must_avoid"]["percentage"])
        self.assertEqual(valunseen_by_key["path_shape_preference"]["count"], 1)
        self.assertAlmostEqual(
            valunseen_by_key["path_shape_preference"]["percentage"],
            100.0,
        )

    def test_sync_labeled_instructions_updates_templates_and_canonical_copy(self) -> None:
        original_root = make_instruction.INSTRUCTION_ROOT
        original_template_path = make_instruction.template_instruction_path
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            make_instruction.INSTRUCTION_ROOT = root / "instructions"
            legacy_dir = make_instruction.INSTRUCTION_ROOT / "procthor" / "002_train"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "instruction_000001.json").write_text(
                """
                {
                  "id": 1,
                  "template_instruction_id": "template_3_000191",
                  "updated_at": "2026-01-01T00:00:00+00:00"
                }
                """,
                encoding="utf-8",
            )
            template_path = root / "template_instruction.json"
            template_bundle = {
                "version": 1,
                "map_id": "procthor/002_train",
                "distance_rule": {},
                "object_counts": [2, 3, 4, 5, 10],
                "randomized_order": True,
                "template_limits": {},
                "templates": [
                    {
                        "template_instruction_id": "template_3_000191",
                        "map_id": "procthor/002_train",
                        "object_count": 2,
                        "objects": [
                            {
                                "order": 1,
                                "object_id": 1,
                                "name": "chair_1",
                                "category": "chair",
                                "center": [0, 0],
                            },
                            {
                                "order": 2,
                                "object_id": 2,
                                "name": "table_1",
                                "category": "table",
                                "center": [100, 0],
                            },
                        ],
                        "segment_distances": [100],
                        "status": "unlabeled",
                        "labeled_instruction_id": None,
                        "created_at": "",
                        "updated_at": "",
                    }
                ],
                "created_at": "",
                "updated_at": "",
            }

            def fake_template_path(_map_id: str) -> Path:
                return template_path

            make_instruction.template_instruction_path = fake_template_path
            try:
                synced = sync_labeled_instructions_with_templates(
                    "procthor/002_train",
                    template_bundle,
                )
                canonical_exists = (
                    root
                    / "instructions"
                    / "procthor"
                    / "002_train"
                    / "instruction_files"
                    / "instruction_000001.json"
                ).exists()
            finally:
                make_instruction.INSTRUCTION_ROOT = original_root
                make_instruction.template_instruction_path = original_template_path

        template = synced["templates"][0]
        self.assertEqual(template["status"], "labeled")
        self.assertEqual(template["labeled_instruction_id"], 1)
        self.assertTrue(canonical_exists)

    def test_route_collision_uses_occupancy_as_traversability_source(self) -> None:
        map_state = {
            "layer_legends": {
                "occupancy": {
                    "free": {"value": 0},
                    "obstacle": {"value": 1},
                }
            },
            "layers": {
                "occupancy": [
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                "object_instance": [
                    [0, 1, 0],
                    [2, 0, 0],
                    [3, 0, 0],
                ],
            },
            "object_instances": [
                {"id": 1, "category": "doorway"},
                {"id": 2, "category": "table"},
                {"id": 3, "category": "doorframe"},
            ],
        }
        sample = {
            "expert_route": [
                [0, 0],
                [0, 1],
                [2, 0],
                [1, 0],
                [0, 2],
            ]
        }

        self.assertEqual(
            route_collision_violations_for_map(sample, map_state),
            [[0, 2]],
        )

    def test_sample_difficulty_level_is_normalized(self) -> None:
        sample = validate_sample(
            {
                "sample_id": "sample_1",
                "name": "Sample 1",
                "difficulty_level": "Extreme",
            },
            "simple_demo/simple_demo_map",
            1,
        )

        self.assertEqual(sample["difficulty_level"], "extreme")

    def test_invalid_sample_difficulty_level_is_cleared(self) -> None:
        sample = validate_sample(
            {
                "sample_id": "sample_1",
                "name": "Sample 1",
                "difficulty_level": "medium",
            },
            "simple_demo/simple_demo_map",
            1,
        )

        self.assertEqual(sample["difficulty_level"], "")

    def test_sample_validation_adds_global_clearance(self) -> None:
        sample = validate_sample(
            {
                "sample_id": "sample_1",
                "name": "Sample 1",
                "soft_constraints": [],
            },
            "simple_demo/simple_demo_map",
            1,
        )

        clearance = [
            constraint
            for constraint in sample["soft_constraints"]
            if constraint["preference_type"] == "clearance"
        ]
        self.assertEqual(len(clearance), 1)
        self.assertEqual(clearance[0]["scope"]["type"], "global")
        self.assertEqual(clearance[0]["shape"], "freeform")
        self.assertEqual(clearance[0]["cells"], [])

    def test_clearance_soft_constraint_does_not_require_region_fields(self) -> None:
        validated = validate_soft_constraints(
            [
                {
                    "constraint_id": "clearance",
                    "preference_type": "clearance",
                }
            ]
        )

        self.assertEqual(validated[0]["preference_type"], "clearance")
        self.assertEqual(validated[0]["scope"]["type"], "global")
        self.assertEqual(validated[0]["center"], [0.0, 0.0])
        self.assertEqual(validated[0]["cells"], [])

    def test_legacy_reference_point_migrates_to_brush_region(self) -> None:
        constraint = base_constraint()
        constraint["reference_point"] = [11, 21]

        validated = validate_soft_constraints([constraint])[0]

        self.assertEqual(
            validated["reference_region"],
            {
                "mode": "brush",
                "object_id": None,
                "cells": [[11, 21]],
            },
        )

    def test_object_reference_region_keeps_object_and_cells(self) -> None:
        constraint = base_constraint()
        constraint["reference_region"] = {
            "mode": "object",
            "object_id": 3,
            "cells": [[4, 5], [4, 6]],
        }

        validated = validate_soft_constraints([constraint])[0]

        self.assertEqual(validated["reference_region"]["mode"], "object")
        self.assertEqual(validated["reference_region"]["object_id"], 3)
        self.assertEqual(validated["reference_region"]["cells"], [[4, 5], [4, 6]])

    def test_object_cells_falls_back_to_center_hint(self) -> None:
        map_state = {
            "grid_size": 8,
            "layers": {
                "object_instance": [[0 for _col in range(8)] for _row in range(8)],
            },
            "object_instances": [
                {
                    "id": 41,
                    "category": "cell_phone",
                    "center_grid": [2.2, 5.7],
                }
            ],
        }

        self.assertEqual(object_cells_for_map(map_state, 41), [[2, 6]])

    def test_object_cells_prefers_rich_object_footprint(self) -> None:
        object_grid = [[0 for _col in range(6)] for _row in range(6)]
        object_grid[5][5] = 7
        map_state = {
            "grid_size": 6,
            "layers": {
                "object_instance": object_grid,
            },
            "object_instances": [
                {
                    "id": 7,
                    "category": "remote_control",
                    "center_grid": [1, 2],
                }
            ],
            "object_footprints": {
                "7": {
                    "object_id": 7,
                    "cells": [[1, 2], [1, 3]],
                    "center_grid": [1, 2.5],
                }
            },
        }

        self.assertEqual(object_cells_for_map(map_state, 7), [[1, 2], [1, 3]])

    def test_validate_map_payload_preserves_rich_object_fields(self) -> None:
        grid_size = 3
        payload = {
            "grid_size": grid_size,
            "metadata": {
                "map_id": "toy",
                "grid_coordinate_frame": {
                    "index_order": "[row, col]",
                    "row_axis": "world_z",
                    "col_axis": "world_x",
                    "resolution": 0.25,
                    "x_min": -1.0,
                    "z_min": -2.0,
                },
            },
            "layers": {
                "occupancy": [[0 for _col in range(grid_size)] for _row in range(grid_size)],
                "room": [[0 for _col in range(grid_size)] for _row in range(grid_size)],
                "object_instance": [[0 for _col in range(grid_size)] for _row in range(grid_size)],
            },
            "room_instances": [],
            "object_instances": [
                {
                    "id": 7,
                    "category": "chair",
                    "name": "Chair|1",
                    "attributes": [],
                    "center_grid": [1, 1],
                    "metadata_grid_cell": [1, 1],
                    "room_id": 2,
                    "assignment_method": "metadata_grid_cell",
                    "num_grid_cells": 1,
                    "footprint_source": "center_fallback",
                }
            ],
            "object_footprints": {
                "7": {
                    "object_id": 7,
                    "cells": [[1, 1]],
                    "center_grid": [1, 1],
                }
            },
            "cell_object_ids": {"1,1": [7]},
        }

        validated = validate_map_payload(payload, "toy")

        self.assertEqual(validated["object_instances"][0]["center_grid"], [1.0, 1.0])
        self.assertEqual(validated["object_instances"][0]["room_id"], 2)
        self.assertEqual(validated["object_footprints"]["7"]["cells"], [[1, 1]])
        self.assertEqual(validated["cell_object_ids"]["1,1"], [7])
        self.assertEqual(
            validated["metadata"]["grid_coordinate_frame"]["resolution"],
            0.25,
        )

    def test_template_object_centers_use_rich_object_footprints(self) -> None:
        map_state = {
            "grid_size": 6,
            "layers": {
                "object_instance": [[0 for _col in range(6)] for _row in range(6)],
            },
            "object_instances": [
                {
                    "id": 7,
                    "category": "remote_control",
                    "name": "RemoteControl|surface",
                    "attributes": [],
                }
            ],
            "object_footprints": {
                "7": {
                    "object_id": 7,
                    "cells": [[1, 2], [1, 3]],
                    "center_grid": [1, 2.5],
                }
            },
        }

        objects = collect_template_object_centers(map_state)

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["object_id"], 7)
        self.assertEqual(objects[0]["center"], [1.0, 2.5])

    def test_brush_reference_region_deduplicates_cells(self) -> None:
        constraint = base_constraint()
        constraint["reference_region"] = {
            "mode": "brush",
            "object_id": None,
            "cells": [[4, 5], [4, 5], [4, 6]],
        }

        validated = validate_soft_constraints([constraint])[0]

        self.assertEqual(validated["reference_region"]["object_id"], None)
        self.assertEqual(validated["reference_region"]["cells"], [[4, 5], [4, 6]])

    def test_reference_distance_deduplicates_waypoints_and_uses_nearest_cell(
        self,
    ) -> None:
        constraint = base_constraint()
        constraint.update(
            {
                "shape": "rectangle",
                "center": [0, 1],
                "width": 4,
                "height": 2,
            }
        )

        distance, count = mean_unique_waypoint_distance(
            [[0, 0], [0, 2], [0, 0], [0, 2], [9, 9]],
            constraint,
            [[0, 4], [3, 0]],
        )

        self.assertAlmostEqual(distance, 2.5)
        self.assertEqual(count, 2)

    def test_annotation_records_d_ref_and_empty_region_as_null(self) -> None:
        near = base_constraint()
        near["reference_region"] = {
            "mode": "object",
            "object_id": 3,
            "cells": [[10, 24], [11, 24]],
        }
        empty = base_constraint()
        empty["constraint_id"] = "soft_2"
        empty["center"] = [100, 100]
        empty["reference_region"] = {
            "mode": "brush",
            "object_id": None,
            "cells": [[100, 105]],
        }
        sample = {
            "expert_route": [[10, 20], [10, 20], [11, 20]],
            "soft_constraints": validate_soft_constraints([near, empty]),
        }

        annotate_soft_reference_distances(sample)

        self.assertEqual(sample["soft_constraints"][0]["D_ref"], 4.0)
        self.assertEqual(
            sample["soft_constraints"][0]["reference_waypoint_count"],
            2,
        )
        self.assertIsNone(sample["soft_constraints"][1]["D_ref"])
        self.assertEqual(
            sample["soft_constraints"][1]["reference_waypoint_count"],
            0,
        )

    def test_mean_squared_turning_angle_penalizes_large_turns(self) -> None:
        cost, count = mean_squared_turning_angle(
            [[0, 0], [0, 1], [1, 1], [1, 0]]
        )

        self.assertEqual(count, 2)
        self.assertAlmostEqual(cost, (3.141592653589793 / 2) ** 2)

    def test_move_smoothness_records_reference_turning_cost(self) -> None:
        smoothness = base_constraint()
        smoothness.update(
            {
                "preference_type": "move_smoothness",
                "shape": "freeform",
                "center": [0, 0],
                "cells": [],
                "reference_region": None,
            }
        )
        sample = {
            "expert_route": [[0, 0], [0, 1], [1, 1]],
            "soft_constraints": validate_soft_constraints([smoothness]),
        }

        annotate_soft_reference_distances(sample)

        constraint = sample["soft_constraints"][0]
        self.assertIsNone(constraint["D_ref"])
        self.assertAlmostEqual(
            constraint["C_smooth_ref"],
            (3.141592653589793 / 2) ** 2,
        )
        self.assertEqual(constraint["reference_waypoint_count"], 1)

    def test_clearance_records_reference_cost_without_region(self) -> None:
        clearance = base_constraint()
        clearance.update(
            {
                "preference_type": "clearance",
                "shape": "freeform",
                "center": [0, 0],
                "cells": [],
                "reference_region": None,
            }
        )
        map_state = {
            "layer_legends": {"occupancy": {"free": {"value": 0}}},
            "layers": {
                "occupancy": [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                "object_instance": [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            },
            "object_instances": [],
        }
        sample = {
            "expert_route": [[0, 1], [2, 2]],
            "soft_constraints": validate_soft_constraints([clearance]),
        }

        annotate_soft_reference_distances(sample, map_state)

        constraint = sample["soft_constraints"][0]
        self.assertIsNone(constraint["D_ref"])
        self.assertIsNone(constraint["C_smooth_ref"])
        self.assertAlmostEqual(
            constraint["C_clear_ref"],
            (1 / ((1 + 1e-6) ** 2) + 1 / ((8 ** 0.5 + 1e-6) ** 2)) / 2,
        )
        self.assertEqual(constraint["reference_waypoint_count"], 2)

    def test_relative_preference_records_reference_margin(self) -> None:
        relative = base_constraint()
        relative.update(
            {
                "preference_type": "relative_preference",
                "shape": "rectangle",
                "center": [0, 3],
                "width": 8,
                "height": 2,
                "object_ids": [1, 2],
                "reference_regions": [
                    {"object_id": 1, "cells": [[0, 0]]},
                    {"object_id": 2, "cells": [[0, 6]]},
                ],
            }
        )
        sample = {
            "expert_route": [[0, 1], [0, 3], [0, 5]],
            "soft_constraints": validate_soft_constraints([relative]),
        }

        annotate_soft_reference_distances(sample)

        constraint = sample["soft_constraints"][0]
        self.assertIsNone(constraint["D_ref"])
        self.assertAlmostEqual(constraint["Q_relative_ref"], 0.0)
        self.assertEqual(constraint["reference_waypoint_count"], 2)
        self.assertAlmostEqual(constraint["reference_valid_path_length"], 4.0)

    def test_path_shape_records_scoped_reference_trajectory(self) -> None:
        path_shape = base_constraint()
        path_shape.update(
            {
                "preference_type": "path_shape_preference",
                "shape": "freeform",
                "center": [0.5, 1.333],
                "cells": [[0, 1], [1, 1], [0, 2]],
                "reference_region": None,
                "scope": {
                    "type": "between_hard_constraints",
                    "from_order": 1,
                    "to_order": 2,
                },
            }
        )
        sample = {
            "expert_route": [[0, 0], [0, 1], [1, 1], [0, 2], [9, 9]],
            "hard_constraints": [
                {
                    "constraint_id": "A",
                    "kind": "must_pass",
                    "shape": "freeform",
                    "center": [0, 1],
                    "order": 1,
                    "cells": [[0, 1]],
                },
                {
                    "constraint_id": "B",
                    "kind": "must_pass",
                    "shape": "freeform",
                    "center": [0, 2],
                    "order": 2,
                    "cells": [[0, 2]],
                },
            ],
            "soft_constraints": validate_soft_constraints([path_shape]),
        }

        annotate_soft_reference_distances(sample)

        constraint = sample["soft_constraints"][0]
        self.assertEqual(
            constraint["reference_trajectory"],
            [[0, 1], [1, 1], [0, 2]],
        )
        self.assertEqual(constraint["reference_waypoint_count"], 3)
        self.assertIsNone(constraint["D_ref"])
        self.assertIsNone(constraint["C_smooth_ref"])
        self.assertIsNone(constraint["C_clear_ref"])

    def test_path_shape_reference_falls_back_when_scope_end_is_missing(self) -> None:
        path_shape = base_constraint()
        path_shape.update(
            {
                "preference_type": "path_shape_preference",
                "shape": "freeform",
                "center": [1, 1],
                "cells": [[1, 1]],
                "reference_region": None,
                "scope": {
                    "type": "between_hard_constraints",
                    "from_order": 1,
                    "to_order": 2,
                },
            }
        )
        sample = {
            "expert_route": [[0, 0], [0, 1], [1, 1], [9, 9]],
            "hard_constraints": [
                {
                    "constraint_id": "A",
                    "kind": "must_pass",
                    "shape": "freeform",
                    "center": [0, 1],
                    "order": 1,
                    "cells": [[0, 1]],
                },
                {
                    "constraint_id": "B",
                    "kind": "must_pass",
                    "shape": "freeform",
                    "center": [0, 2],
                    "order": 2,
                    "cells": [[0, 2]],
                },
            ],
            "soft_constraints": validate_soft_constraints([path_shape]),
        }

        annotate_soft_reference_distances(sample)

        constraint = sample["soft_constraints"][0]
        self.assertEqual(constraint["reference_trajectory"], [[1, 1]])
        self.assertEqual(constraint["reference_waypoint_count"], 1)

    def test_embedded_path_shape_annotation_is_preserved_during_recompute(self) -> None:
        path_shape = base_constraint()
        path_shape.update(
            {
                "preference_type": "path_shape_preference",
                "reference_trajectory": [[9, 9], [9, 10]],
                "path_shape_annotation": {
                    "version": 1,
                    "annotation_type": "canonical_path_shape_reference",
                    "shape_spec": {
                        "type": "expert_polyline",
                        "direction": "clockwise",
                        "ordered_control_points": [[1, 1], [1, 3]],
                    },
                    "shape_reference_trajectory": [[1, 1], [1, 2], [1, 3]],
                    "shape_reference_source": "human_expert_canonical_v1",
                    "history": [],
                },
            }
        )
        sample = {
            "expert_route": [[9, 9], [9, 10]],
            "hard_constraints": [],
            "soft_constraints": validate_soft_constraints([path_shape]),
        }

        annotate_soft_reference_distances(sample)

        constraint = sample["soft_constraints"][0]
        self.assertEqual(
            constraint["reference_trajectory"],
            [[1, 1], [1, 2], [1, 3]],
        )
        self.assertEqual(constraint["reference_waypoint_count"], 3)
        self.assertIn("path_shape_annotation", constraint)

    def test_path_shape_requires_a_region(self) -> None:
        path_shape = base_constraint()
        path_shape.update(
            {
                "preference_type": "path_shape_preference",
                "shape": "freeform",
                "cells": [],
                "reference_region": None,
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "freeform soft constraints must contain at least one cell",
        ):
            validate_soft_constraints([path_shape])


if __name__ == "__main__":
    unittest.main()
