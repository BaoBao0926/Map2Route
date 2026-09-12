"""Regression tests for path-shape annotations embedded in instructions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.annotation.annotate_path_shapes import (
    add_human_path_shape_references,
    save_embedded_annotation,
)
from scripts.annotation.path_shape_schema import normalize_path_shape_annotation


def path_shape_constraint() -> dict[str, object]:
    return {
        "constraint_id": "shape_1",
        "preference_type": "path_shape_preference",
        "shape": "freeform",
        "center": [1, 2],
        "radius": 1,
        "width": 1,
        "height": 1,
        "cells": [[1, 1], [1, 2], [1, 3]],
        "scope": {"type": "global", "from_order": None, "to_order": None},
        "reference_trajectory": [[1, 1], [1, 2]],
    }


class EmbeddedPathShapeAnnotationTest(unittest.TestCase):
    def test_schema_normalizes_embedded_history(self) -> None:
        record = {
            "version": 1,
            "annotation_type": "canonical_path_shape_reference",
            "shape_spec": {
                "type": "expert_polyline",
                "direction": "clockwise",
                "ordered_control_points": [[1, 1], [1, 3]],
            },
            "shape_reference_trajectory": [[1, 1], [1, 2], [1, 3]],
            "history": [
                {
                    "version": 1,
                    "annotation_type": "canonical_path_shape_reference",
                    "shape_reference_trajectory": [[2, 1], [2, 2]],
                }
            ],
        }

        normalized = normalize_path_shape_annotation(record, grid_size=5)

        self.assertEqual(len(normalized["history"]), 1)
        self.assertEqual(
            normalized["shape_reference_trajectory"],
            [[1, 1], [1, 2], [1, 3]],
        )

    def test_save_writes_instruction_and_appends_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            instruction_path = Path(temporary) / "instruction_000001.json"
            instruction_path.write_text(
                json.dumps(
                    {
                        "map_id": "procthor/001_valunseen",
                        "soft_constraints": [path_shape_constraint()],
                    }
                ),
                encoding="utf-8",
            )

            def load_instruction(_map_id: str, _instruction_id: str):
                return json.loads(instruction_path.read_text()), instruction_path

            first_payload = {
                "map_id": "procthor/001_valunseen",
                "instruction_id": "instruction_000001",
                "constraint_id": "shape_1",
                "shape_reference_trajectory": [[1, 1], [1, 2], [1, 3]],
                "ordered_control_points": [[1, 1], [1, 3]],
                "direction": "clockwise",
            }
            second_payload = {
                **first_payload,
                "shape_reference_trajectory": [[1, 3], [1, 2], [1, 1]],
                "direction": "counterclockwise",
            }
            map_state = {"map_key": "procthor/001_valunseen", "grid_size": 5}
            with (
                patch(
                    "scripts.annotation.annotate_path_shapes.load_map_state",
                    return_value=map_state,
                ),
                patch(
                    "scripts.annotation.annotate_path_shapes.load_instruction_by_id",
                    side_effect=load_instruction,
                ),
                patch(
                    "scripts.annotation.annotate_path_shapes.build_instruction_metric_cache"
                ) as cache_builder,
            ):
                first = save_embedded_annotation(first_payload)
                second = save_embedded_annotation(second_payload)

            saved = json.loads(instruction_path.read_text())
            constraint = saved["soft_constraints"][0]
            annotation = constraint["path_shape_annotation"]
            self.assertTrue(first["newly_annotated"])
            self.assertFalse(second["newly_annotated"])
            self.assertEqual(len(annotation["history"]), 1)
            self.assertEqual(
                constraint["reference_trajectory"],
                second_payload["shape_reference_trajectory"],
            )
            self.assertEqual(cache_builder.call_count, 2)

    def test_human_qa_reference_is_computed_separately(self) -> None:
        constraint = path_shape_constraint()
        constraint["reference_trajectory"] = [[4, 4], [4, 3]]
        instruction = {
            "human_expert_trajectory": [[0, 0], [1, 1], [1, 2], [2, 2]],
            "hard_constraints": [],
            "soft_constraints": [constraint],
        }

        add_human_path_shape_references(instruction)

        self.assertEqual(
            constraint["_human_reference_trajectory"],
            [[1, 1], [1, 2]],
        )
        self.assertEqual(constraint["reference_trajectory"], [[4, 4], [4, 3]])


if __name__ == "__main__":
    unittest.main()
