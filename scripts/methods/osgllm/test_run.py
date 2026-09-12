"""Regression tests for the resumable OSGLLM runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.methods.osgllm.run import build_all


class OSGLLMRunTest(unittest.TestCase):
    def test_zero_byte_existing_record_is_recomputed(self) -> None:
        instruction = {
            "instruction": "Go to the target.",
            "difficulty_level": "easy",
            "start_pose": {"row": 4, "col": 5},
        }
        metrics = {
            "HCS": 1.0,
            "SCS": None,
            "PL": 0.0,
            "segment_wise_SPL": 1.0,
            "details": {"soft_constraints": []},
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "instructions"
            output_root = root / "predictions"
            instruction_file = input_root / "instruction_000013.json"
            output_path = (
                output_root
                / "procthor"
                / "030_valunseen"
                / "instruction_000013.json"
            )
            output_path.parent.mkdir(parents=True)
            output_path.write_text("", encoding="utf-8")

            with (
                patch(
                    "scripts.methods.osgllm.run.iter_instruction_files",
                    return_value=[instruction_file],
                ),
                patch(
                    "scripts.methods.osgllm.run.load_instruction",
                    return_value=instruction,
                ),
                patch(
                    "scripts.methods.osgllm.run.map_id_from_instruction_path",
                    return_value="procthor/030_valunseen",
                ),
                patch(
                    "scripts.methods.osgllm.run.instruction_id_from_payload",
                    return_value="instruction_000013",
                ),
                patch(
                    "scripts.methods.osgllm.run.load_map_state",
                    return_value={"grid_size": 10},
                ),
                patch(
                    "scripts.methods.osgllm.run.build_osgllm_trajectory",
                    return_value=(
                        [[4, 5]],
                        {"status": "SUCCESS", "planner": {}},
                    ),
                ) as build_trajectory,
                patch(
                    "scripts.methods.osgllm.run.evaluate_prediction",
                    return_value=metrics,
                ),
                patch(
                    "scripts.methods.osgllm.run.save_trajectory_image_for_record",
                    return_value=None,
                ),
                redirect_stdout(StringIO()),
            ):
                result = build_all(input_root, output_root, overwrite=False)

            self.assertEqual(result[0]["status"], "written")
            self.assertEqual(build_trajectory.call_count, 1)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["metrics"]["HCS"], 1.0)

    def test_unexpected_error_is_persisted_once_as_worst_case(self) -> None:
        instruction = {
            "instruction": "Go to the target.",
            "difficulty_level": "hard",
            "start_pose": {"row": 4, "col": 5},
            "hard_constraints": [
                {"constraint_id": "start", "kind": "must_pass", "order": 1},
                {"constraint_id": "goal", "kind": "must_pass", "order": 2},
            ],
            "soft_constraints": [
                {"constraint_id": "far", "preference_type": "far_preference"},
                {
                    "constraint_id": "shape",
                    "preference_type": "path_shape_preference",
                },
                {
                    "constraint_id": "clearance",
                    "preference_type": "clearance",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "instructions"
            output_root = root / "predictions"
            instruction_file = input_root / "instruction_000018.json"

            with (
                patch(
                    "scripts.methods.osgllm.run.iter_instruction_files",
                    return_value=[instruction_file],
                ),
                patch(
                    "scripts.methods.osgllm.run.load_instruction",
                    return_value=instruction,
                ),
                patch(
                    "scripts.methods.osgllm.run.map_id_from_instruction_path",
                    return_value="procthor/003_valunseen",
                ),
                patch(
                    "scripts.methods.osgllm.run.instruction_id_from_payload",
                    return_value="instruction_000018",
                ),
                patch(
                    "scripts.methods.osgllm.run.load_map_state",
                    return_value={"grid_size": 10},
                ),
                patch(
                    "scripts.methods.osgllm.run.build_osgllm_trajectory",
                    side_effect=TypeError("'str' object is not callable"),
                ) as build_trajectory,
                redirect_stdout(StringIO()),
            ):
                first = build_all(input_root, output_root, overwrite=False)
                second = build_all(input_root, output_root, overwrite=False)

            self.assertEqual(first[0]["status"], "written_worst_case")
            self.assertEqual(second[0]["status"], "skipped_exists")
            self.assertEqual(build_trajectory.call_count, 1)

            output_path = (
                output_root
                / "procthor"
                / "003_valunseen"
                / "instruction_000018.json"
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["trajectory"], [[4, 5]])
            self.assertEqual(saved["metrics"]["HCS"], 0.0)
            self.assertEqual(saved["metrics"]["segment_wise_SPL"], 0.0)
            self.assertEqual(saved["runtime"]["status"], "written_worst_case")
            failure = saved["execution_fallback"]
            self.assertEqual(failure["policy"], "worst_case")
            self.assertEqual(failure["stage"], "build_trajectory")
            self.assertIn("TypeError", failure["error"])
            self.assertIn("Traceback", failure["traceback"])


if __name__ == "__main__":
    unittest.main()
