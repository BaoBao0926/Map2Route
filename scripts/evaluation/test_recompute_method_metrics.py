"""Regression tests for resumable metric recomputation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation.recompute_method_metrics import recompute_method
from scripts.methods.human_solver.evaluate_human_expert import (
    evaluate_human_experts,
    evaluate_instruction,
)


METRICS = {
    "HCS": 1.0,
    "SCS": None,
    "PL": 1.0,
    "segment_wise_SPL": 1.0,
    "details": {},
}


class RecomputeResumeTests(unittest.TestCase):
    def test_method_resume_skips_matching_record_and_invalidates_changed_trajectory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method_root = root / "ILN"
            episode_path = (
                method_root
                / "procthor"
                / "003_valunseen"
                / "instruction_000001.json"
            )
            episode_path.parent.mkdir(parents=True)
            episode_path.write_text(
                json.dumps(
                    {
                        "method": "ILN",
                        "map_id": "procthor/003_valunseen",
                        "instruction_id": "instruction_000001",
                        "difficulty_level": "easy",
                        "trajectory": [[0, 0]],
                    }
                ),
                encoding="utf-8",
            )
            instruction_path = root / "instruction_000001.json"
            instruction = {
                "map_id": "procthor/003_valunseen",
                "instruction_id": "instruction_000001",
                "hard_constraints": [],
                "soft_constraints": [],
            }
            instruction_path.write_text(json.dumps(instruction), encoding="utf-8")
            checkpoint_directory = root / "_metric_recompute" / "ILN"

            with (
                patch(
                    "scripts.evaluation.recompute_method_metrics."
                    "resolve_instruction_file_for_payload",
                    return_value=instruction_path,
                ),
                patch(
                    "scripts.evaluation.recompute_method_metrics.load_instruction",
                    return_value=instruction,
                ),
                patch(
                    "scripts.evaluation.recompute_method_metrics.evaluate_prediction",
                    return_value=dict(METRICS),
                ) as evaluate,
            ):
                _summary, first = recompute_method(
                    method_root,
                    method_name="ILN",
                    evaluator_id="evaluator_test",
                    checkpoint_directory=checkpoint_directory,
                    quiet=True,
                    checkpoint_every=1,
                )
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(first["resumed_records"], 0)

                evaluate.reset_mock()
                with patch(
                    "scripts.evaluation.recompute_method_metrics.print_progress"
                ) as progress:
                    _summary, second = recompute_method(
                        method_root,
                        method_name="ILN",
                        evaluator_id="evaluator_test",
                        checkpoint_directory=checkpoint_directory,
                        quiet=False,
                        checkpoint_every=1,
                    )
                self.assertEqual(evaluate.call_count, 0)
                self.assertEqual(second["resumed_records"], 1)
                messages = [call.args[0] for call in progress.call_args_list]
                self.assertTrue(
                    any(
                        "[record:skip]" in message
                        and "reason=resume_checkpoint_match" in message
                        for message in messages
                    )
                )

                payload = json.loads(episode_path.read_text(encoding="utf-8"))
                payload["trajectory"] = [[0, 0], [0, 1]]
                episode_path.write_text(json.dumps(payload), encoding="utf-8")

                evaluate.reset_mock()
                _summary, third = recompute_method(
                    method_root,
                    method_name="ILN",
                    evaluator_id="evaluator_test",
                    checkpoint_directory=checkpoint_directory,
                    quiet=True,
                    checkpoint_every=1,
                )
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(third["resumed_records"], 0)

    def test_metric_error_is_saved_as_worst_case_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method_root = root / "LIMP"
            episode_path = (
                method_root
                / "procthor"
                / "003_valunseen"
                / "instruction_000001.json"
            )
            episode_path.parent.mkdir(parents=True)
            episode_path.write_text(
                json.dumps(
                    {
                        "method": "LIMP",
                        "map_id": "procthor/003_valunseen",
                        "instruction_id": "instruction_000001",
                        "difficulty_level": "easy",
                        "trajectory": [[0, 0], [0, 1]],
                    }
                ),
                encoding="utf-8",
            )
            soft_constraints = [
                {
                    "constraint_id": "near",
                    "preference_type": "near_preference",
                    "shape": "freeform",
                    "cells": [[0, 0], [2, 0]],
                    "reference_region": {"cells": [[0, 0]]},
                },
                {
                    "constraint_id": "far",
                    "preference_type": "far_preference",
                },
                {
                    "constraint_id": "relative",
                    "preference_type": "relative_preference",
                },
                {
                    "constraint_id": "shape",
                    "preference_type": "path_shape_preference",
                },
                {
                    "constraint_id": "clearance",
                    "preference_type": "clearance",
                },
            ]
            instruction = {
                "map_id": "procthor/003_valunseen",
                "instruction_id": "instruction_000001",
                "hard_constraints": [],
                "soft_constraints": soft_constraints,
            }
            instruction_path = root / "instruction_000001.json"
            instruction_path.write_text(json.dumps(instruction), encoding="utf-8")
            checkpoint_directory = root / "_metric_recompute" / "LIMP"

            with (
                patch(
                    "scripts.evaluation.recompute_method_metrics."
                    "resolve_instruction_file_for_payload",
                    return_value=instruction_path,
                ),
                patch(
                    "scripts.evaluation.recompute_method_metrics.load_instruction",
                    return_value=instruction,
                ),
                patch(
                    "scripts.evaluation.recompute_method_metrics.load_map_state",
                    return_value={"grid_size": 5},
                ),
                patch(
                    "scripts.evaluation.recompute_method_metrics.evaluate_prediction",
                    side_effect=TypeError("synthetic metric error"),
                ),
            ):
                summary, stats = recompute_method(
                    method_root,
                    method_name="LIMP",
                    evaluator_id="evaluator_test",
                    checkpoint_directory=checkpoint_directory,
                    quiet=True,
                    checkpoint_every=1,
                )

            self.assertEqual(stats["fallback_records"], 1)
            self.assertEqual(stats["failed_records"], 0)
            self.assertEqual(summary["overall"]["record_count"], 1)
            saved = json.loads(episode_path.read_text(encoding="utf-8"))
            metrics = saved["metrics"]
            self.assertEqual(metrics["HCS"], 0.0)
            self.assertEqual(metrics["segment_wise_SPL"], 0.0)
            details = metrics["details"]["soft_constraints"]
            raw_by_id = {
                detail["constraint_id"]: detail["raw_value"]
                for detail in details
            }
            self.assertEqual(raw_by_id["near"], 2.0)
            self.assertEqual(raw_by_id["far"], 0.0)
            self.assertEqual(raw_by_id["relative"], 0.0)
            self.assertEqual(raw_by_id["shape"], 0.0)
            self.assertEqual(raw_by_id["clearance"], 0.0)
            self.assertEqual(
                metrics["details"]["evaluation_fallback"]["policy"],
                "worst_case",
            )
            checkpoint = json.loads(
                (checkpoint_directory / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["status"], "completed")
            self.assertEqual(checkpoint["fallback_count"], 1)
            self.assertEqual(checkpoint["failed_count"], 0)

    def test_malformed_prediction_json_uses_worst_case_without_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method_root = root / "Lang2LTL"
            episode_path = (
                method_root
                / "procthor"
                / "003_valunseen"
                / "instruction_000001.json"
            )
            episode_path.parent.mkdir(parents=True)
            malformed_payload = '{"method": "Lang2LTL", broken'
            episode_path.write_text(malformed_payload, encoding="utf-8")

            instruction = {
                "map_id": "procthor/003_valunseen",
                "instruction_id": "instruction_000001",
                "difficulty_level": "easy",
                "hard_constraints": [],
                "soft_constraints": [
                    {
                        "constraint_id": "relative",
                        "preference_type": "relative_preference",
                    }
                ],
            }
            instruction_path = root / "instruction_000001.json"
            instruction_path.write_text(json.dumps(instruction), encoding="utf-8")
            checkpoint_directory = root / "_metric_recompute" / "Lang2LTL"

            with (
                patch(
                    "scripts.evaluation.recompute_method_metrics."
                    "load_instruction_by_id",
                    return_value=(instruction, instruction_path),
                ),
                patch(
                    "scripts.evaluation.recompute_method_metrics.load_map_state",
                    return_value={"grid_size": 5},
                ),
                patch(
                    "scripts.evaluation.recompute_method_metrics.evaluate_prediction"
                ) as evaluate,
            ):
                summary, stats = recompute_method(
                    method_root,
                    method_name="Lang2LTL",
                    evaluator_id="evaluator_test",
                    checkpoint_directory=checkpoint_directory,
                    quiet=True,
                    checkpoint_every=1,
                )

            evaluate.assert_not_called()
            self.assertEqual(stats["fallback_records"], 1)
            self.assertEqual(stats["failed_records"], 0)
            self.assertEqual(stats["updated_records"], 0)
            self.assertEqual(summary["overall"]["record_count"], 1)
            self.assertEqual(summary["overall"]["metric"]["HCS"], 0.0)
            self.assertEqual(
                summary["overall"]["metric"]["SCS"]["relative preference"],
                0.0,
            )
            self.assertEqual(
                episode_path.read_text(encoding="utf-8"),
                malformed_payload,
            )
            checkpoint = json.loads(
                (checkpoint_directory / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["status"], "completed")
            self.assertEqual(checkpoint["fallback_count"], 1)
            self.assertEqual(checkpoint["failed_count"], 0)
            fallback = checkpoint["fallbacks"][
                "procthor/003_valunseen/instruction_000001.json"
            ]
            self.assertEqual(fallback["policy"], "worst_case")
            self.assertIn("JSONDecodeError", fallback["error"])

    def test_human_expert_uses_cached_episode_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            instruction_path = (
                input_root
                / "procthor"
                / "003_valunseen"
                / "instruction_files"
                / "instruction_000001.json"
            )
            instruction_path.parent.mkdir(parents=True)
            instruction = {
                "map_id": "procthor/003_valunseen",
                "instruction_id": "instruction_000001",
                "difficulty_level": "easy",
                "human_expert_trajectory": [[0, 0]],
            }
            instruction_path.write_text(json.dumps(instruction), encoding="utf-8")
            cached: dict[str, object] = {}

            def load_cache(_path: Path, _instruction: object) -> object:
                return cached.get("record")

            def write_cache(
                _path: Path,
                _instruction: object,
                record: object,
            ) -> None:
                cached["record"] = record

            record = {
                "map_id": "procthor/003_valunseen",
                "instruction_id": "instruction_000001",
                "difficulty_level": "easy",
                "instruction_file": str(instruction_path),
                "trajectory_length": 1,
                "metrics": dict(METRICS),
            }
            with (
                patch(
                    "scripts.methods.human_solver.evaluate_human_expert."
                    "iter_instruction_files",
                    return_value=[instruction_path],
                ),
                patch(
                    "scripts.methods.human_solver.evaluate_human_expert."
                    "evaluate_instruction",
                    return_value=record,
                ) as evaluate,
            ):
                _summary, first = evaluate_human_experts(
                    input_root=input_root,
                    quiet=True,
                    cache_loader=load_cache,
                    cache_writer=write_cache,
                )
                with patch(
                    "scripts.methods.human_solver.evaluate_human_expert."
                    "print_compact_metric"
                ) as compact:
                    _summary, second = evaluate_human_experts(
                        input_root=input_root,
                        quiet=False,
                        cache_loader=load_cache,
                        cache_writer=write_cache,
                    )

            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(first["computed_instructions"], 1)
            self.assertEqual(first["resumed_instructions"], 0)
            self.assertEqual(second["computed_instructions"], 0)
            self.assertEqual(second["resumed_instructions"], 1)
            self.assertIn(
                "skipped reason=resume_cache_match",
                compact.call_args.args[0],
            )


    def test_human_expert_metric_error_uses_worst_case_record(self) -> None:
        instruction = {
            "map_id": "procthor/003_valunseen",
            "instruction_id": "instruction_000001",
            "difficulty_level": "easy",
            "human_expert_trajectory": [[0, 0], [0, 1]],
            "hard_constraints": [],
            "soft_constraints": [
                {
                    "constraint_id": "near",
                    "preference_type": "near_preference",
                    "shape": "freeform",
                    "cells": [[0, 0], [3, 0]],
                    "reference_region": {"cells": [[0, 0]]},
                }
            ],
        }
        with (
            patch(
                "scripts.methods.human_solver.evaluate_human_expert."
                "evaluate_prediction",
                side_effect=RuntimeError("synthetic human metric error"),
            ),
            patch(
                "scripts.methods.human_solver.evaluate_human_expert."
                "load_map_state",
                return_value={"grid_size": 5},
            ),
        ):
            record = evaluate_instruction(
                Path("instruction_000001.json"),
                instruction,
                worst_case_on_metric_error=True,
            )

        self.assertEqual(record["metrics"]["HCS"], 0.0)
        self.assertEqual(
            record["metrics"]["details"]["soft_constraints"][0]["raw_value"],
            3.0,
        )
        self.assertEqual(record["metric_fallback"]["policy"], "worst_case")

if __name__ == "__main__":
    unittest.main()
