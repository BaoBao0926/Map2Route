"""Regression tests for SayPlan inference-data isolation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.methods.sayplan.adapters.sempathbench_loader import (
    inference_instruction_view,
    start_cell_for_instruction,
)
from scripts.methods.sayplan.config import INFERENCE_CONTRACT_VERSION, METHOD_NAME
from scripts.methods.sayplan.llm.prompts import (
    instruction_context,
    planning_prompt,
    semantic_search_prompt,
)
from scripts.methods.sayplan.run import _existing_prediction_uses_current_contract


class SayPlanDataIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.annotated_instruction = {
            "instruction": "Go to the chair after visiting the kitchen.",
            "start_pose": {"row": 1, "col": 2},
            "objects": [
                {
                    "object_id": 987654,
                    "order": 1,
                    "name": "GT_OBJECT_SENTINEL",
                    "center": [77, 88],
                }
            ],
            "hard_constraints": [
                {"object_id": 876543, "kind": "GT_HARD_SENTINEL"}
            ],
            "soft_constraints": [
                {"object_id": 765432, "kind": "GT_SOFT_SENTINEL"}
            ],
            "human_expert_trajectory": [[66, 55], [54, 53]],
        }

    def test_instruction_projection_excludes_all_annotations(self) -> None:
        projected = inference_instruction_view(self.annotated_instruction)
        self.assertEqual(
            projected,
            {
                "instruction": self.annotated_instruction["instruction"],
                "start_pose": {"row": 1, "col": 2},
            },
        )
        self.assertEqual(
            instruction_context(self.annotated_instruction),
            {"instruction": self.annotated_instruction["instruction"]},
        )

    def test_prompts_do_not_contain_annotation_sentinels(self) -> None:
        graph = {"nodes": [], "edges": []}
        memory = {"expanded_nodes": [], "contracted_nodes": [], "commands": []}
        prompts = (
            semantic_search_prompt(
                instruction=self.annotated_instruction,
                visible_graph=graph,
                memory=memory,
            )[1],
            planning_prompt(
                instruction=self.annotated_instruction,
                task_graph=graph,
                memory=memory,
            )[1],
        )
        forbidden_values = (
            "GT_OBJECT_SENTINEL",
            "GT_HARD_SENTINEL",
            "GT_SOFT_SENTINEL",
            "987654",
            "876543",
            "765432",
            "66, 55",
        )
        for prompt in prompts:
            for value in forbidden_values:
                self.assertNotIn(value, prompt)

    def test_start_requires_explicit_start_pose(self) -> None:
        instruction_without_start = dict(self.annotated_instruction)
        instruction_without_start.pop("start_pose")
        traversable = ((True, True, True),) * 3
        self.assertIsNone(
            start_cell_for_instruction(
                {"grid_size": 3},
                traversable,
                instruction_without_start,
            )
        )

    def test_stale_predictions_are_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "prediction.json"
            output_path.write_text(
                json.dumps({METHOD_NAME: {"inference_contract_version": 1}}),
                encoding="utf-8",
            )
            self.assertFalse(_existing_prediction_uses_current_contract(output_path))
            output_path.write_text(
                json.dumps(
                    {
                        METHOD_NAME: {
                            "inference_contract_version": INFERENCE_CONTRACT_VERSION
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_existing_prediction_uses_current_contract(output_path))


if __name__ == "__main__":
    unittest.main()
