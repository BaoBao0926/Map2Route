from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from scripts.methods.groundplan.run import (
    AUTOMATIC_ABLATION_CASES,
    _prediction_artifacts_complete,
    build_all,
    load_replay_grounding_code,
    replay_grounding_code_path,
)
from scripts.methods.tutorial.utils import write_summary


class AblationRunnerTests(unittest.TestCase):
    def test_resume_only_skips_complete_prediction_and_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "instruction_000001.json"
            steps_path = root / "instruction_000001.steps.json"
            prediction = {
                "trajectory": [[0, 0]],
                "metrics": {
                    "HCS": 0.0,
                    "PL": 0.0,
                    "segment_wise_SPL": 0.0,
                },
            }
            prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
            steps_path.write_text(json.dumps({"steps": {"failure": {}}}), encoding="utf-8")
            self.assertTrue(
                _prediction_artifacts_complete(prediction_path, steps_path)
            )

            prediction["groundplan"] = {"status": "EXECUTION_ERROR_WORST_CASE"}
            prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
            self.assertTrue(
                _prediction_artifacts_complete(prediction_path, steps_path)
            )
            self.assertFalse(
                _prediction_artifacts_complete(
                    prediction_path, steps_path, resume_execution_errors=True
                )
            )

            prediction_path.write_text("", encoding="utf-8")
            self.assertFalse(
                _prediction_artifacts_complete(prediction_path, steps_path)
            )

            prediction_path.write_text(json.dumps({"trajectory": [[0, 0]]}), encoding="utf-8")
            self.assertFalse(
                _prediction_artifacts_complete(prediction_path, steps_path)
            )

            prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
            steps_path.write_text("", encoding="utf-8")
            self.assertFalse(
                _prediction_artifacts_complete(prediction_path, steps_path)
            )

    def test_unexpected_episode_error_writes_worst_case_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_root = root / "output"
            instruction_files = [
                input_root / "instruction_000001.json",
                input_root / "instruction_000002.json",
            ]
            instruction = {
                "instruction": "Go to the target.",
                "difficulty_level": "easy",
                "start_pose": {"row": 1, "col": 2},
                "hard_constraints": [
                    {"constraint_id": "goal", "kind": "must_pass", "order": 1}
                ],
                "soft_constraints": [],
            }
            metrics = {
                "HCS": 1.0,
                "SCS": None,
                "PL": 1.0,
                "segment_wise_SPL": 1.0,
                "details": {"soft_constraints": []},
            }
            worst_metrics = {
                "HCS": 0.0,
                "SCS": None,
                "PL": 0.0,
                "segment_wise_SPL": 0.0,
                "details": {"soft_constraints": []},
            }
            success = SimpleNamespace(
                trajectory=[[1, 2], [1, 3]],
                metadata={"status": "success"},
                steps={},
                failure_reason=None,
                debug_data=None,
            )

            with (
                patch(
                    "scripts.methods.groundplan.run.iter_instruction_files",
                    return_value=instruction_files,
                ),
                patch(
                    "scripts.methods.groundplan.run.load_instruction",
                    return_value=instruction,
                ),
                patch(
                    "scripts.methods.groundplan.run.map_id_from_instruction_path",
                    return_value="procthor/001_valunseen",
                ),
                patch(
                    "scripts.methods.groundplan.run.instruction_id_from_payload",
                    side_effect=lambda source, _payload: source.stem,
                ),
                patch(
                    "scripts.methods.groundplan.run.load_map_state",
                    return_value={"grid_size": 4, "layers": {"occupancy": [[0] * 4 for _ in range(4)]}},
                ),
                patch(
                    "scripts.methods.groundplan.run.run_groundplan_pipeline",
                    side_effect=[
                        AttributeError("synthetic episode failure"),
                        success,
                    ],
                ),
                patch(
                    "scripts.methods.groundplan.run.worst_case_metrics_for_evaluation_error",
                    return_value=worst_metrics,
                ),
                patch(
                    "scripts.methods.groundplan.run._evaluate",
                    return_value=(metrics, None),
                ),
                patch(
                    "scripts.methods.groundplan.run.save_trajectory_image_for_record",
                    return_value=None,
                ),
            ):
                results = build_all(input_root, output_root, overwrite=False)

            self.assertEqual(results[0]["status"], "written_worst_case")
            self.assertEqual(results[1]["status"], "written")
            first_path = (
                output_root
                / "procthor"
                / "001_valunseen"
                / "instruction_000001.json"
            )
            first_record = json.loads(first_path.read_text(encoding="utf-8"))
            self.assertEqual(first_record["metrics"]["HCS"], 0.0)
            self.assertEqual(
                first_record["groundplan"]["status"],
                "EXECUTION_ERROR_WORST_CASE",
            )

    def test_repair_budget_selects_latest_attempt_within_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = replay_grounding_code_path(
                root,
                "procthor/003_valunseen",
                "instruction_000001",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "steps": {
                            "grounding_code": {"source": "code_r2"},
                            "grounding_code_attempts": [
                                {
                                    "attempt": 0,
                                    "stage": "execute_and_verify",
                                    "status": "failed",
                                    "code": "code_r0",
                                },
                                {
                                    "attempt": 1,
                                    "stage": "execute_and_verify",
                                    "status": "success",
                                    "code": "code_r1",
                                },
                                {
                                    "attempt": 2,
                                    "stage": "execute_and_verify",
                                    "status": "failed",
                                    "code": "code_r2",
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_replay_grounding_code(root, "procthor/003_valunseen", "instruction_000001", max_repair_attempt=0)[0], "code_r0")
            self.assertEqual(load_replay_grounding_code(root, "procthor/003_valunseen", "instruction_000001", max_repair_attempt=1)[0], "code_r1")
            self.assertEqual(load_replay_grounding_code(root, "procthor/003_valunseen", "instruction_000001", max_repair_attempt=2)[0], "code_r2")
            self.assertEqual(load_replay_grounding_code(root, "procthor/003_valunseen", "instruction_000001")[0], "code_r2")

    def test_suite_contains_requested_non_representation_ablations(self) -> None:
        names = {str(case["name"]) for case in AUTOMATIC_ABLATION_CASES}
        self.assertTrue({f"repair_r{budget}" for budget in range(3)} <= names)
        self.assertNotIn("repair_r3", names)
        self.assertTrue({"scope_no_temporal", "scope_no_spatial", "scope_none", "no_soft_constraints"} <= names)
        self.assertIn("planner_global_layered", names)
        planner_case = next(
            case for case in AUTOMATIC_ABLATION_CASES
            if case["name"] == "planner_global_layered"
        )
        self.assertEqual(planner_case["planner_mode"], "global_layered_dp")
        self.assertEqual(planner_case["planner_heuristic_weight"], 1.0)
        self.assertEqual(planner_case["max_expansions"], 0)
        self.assertEqual(planner_case["scope_ablation"], "full")

    def test_build_all_updates_summary_after_each_completed_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_id = "procthor/001_valunseen"
            for worker_count in (1, 2):
                with self.subTest(workers=worker_count):
                    input_root = root / f"input_{worker_count}"
                    output_root = root / f"output_{worker_count}"
                    instruction_dir = input_root / map_id / "instruction_files"
                    instruction_dir.mkdir(parents=True)
                    metrics = {"HCS": 1.0, "SCS": 1.0, "PL": 1.0, "segment_wise_SPL": 1.0, "details": {"soft_constraints": []}}
                    for index in (1, 2):
                        instruction_id = f"instruction_{index:06d}"
                        (instruction_dir / f"{instruction_id}.json").write_text(
                            json.dumps({"id": index, "instruction": f"go {index}"}),
                            encoding="utf-8",
                        )

                    def fake_pipeline_result(*_args: object, **_kwargs: object) -> SimpleNamespace:
                        return SimpleNamespace(
                            trajectory=[[0, 0]],
                            metadata={"status": "success"},
                            steps={},
                            failure_reason=None,
                            debug_data=None,
                        )
                    with (
                        patch("scripts.methods.groundplan.run.load_map_state", return_value={}),
                        patch("scripts.methods.groundplan.run.GeminiClient") as client_factory,
                        patch("scripts.methods.groundplan.run.run_groundplan_pipeline", side_effect=fake_pipeline_result),
                        patch("scripts.methods.groundplan.run._evaluate", return_value=(metrics, None)),
                        patch("scripts.methods.groundplan.run.save_trajectory_image_for_record", return_value=None),
                        patch(
                            "scripts.methods.groundplan.run.write_summary",
                            wraps=write_summary,
                        ) as summary_writer,
                    ):
                        client_factory.return_value.trace.return_value = []
                        results = build_all(
                            input_root,
                            output_root,
                            overwrite=True,
                            workers=worker_count,
                        )
                    self.assertEqual(len(results), 2)
                    self.assertEqual(summary_writer.call_count, 2)
                    self.assertEqual(
                        [len(call.args[0]) for call in summary_writer.call_args_list],
                        [1, 2],
                    )
                    summary = json.loads(
                        (output_root / "summary.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(summary["overall"]["record_count"], 2)

    def test_resume_writes_complete_existing_summary_before_skips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_root = root / "output"
            map_id = "procthor/001_valunseen"
            instruction_dir = input_root / map_id / "instruction_files"
            prediction_dir = output_root / map_id
            instruction_dir.mkdir(parents=True)
            prediction_dir.mkdir(parents=True)
            metrics = {
                "HCS": 1.0,
                "SCS": 1.0,
                "PL": 1.0,
                "segment_wise_SPL": 1.0,
                "details": {"soft_constraints": []},
            }
            for index in (1, 2):
                instruction_id = f"instruction_{index:06d}"
                (instruction_dir / f"{instruction_id}.json").write_text(
                    json.dumps({"id": index, "instruction": f"go {index}"}),
                    encoding="utf-8",
                )
                (prediction_dir / f"{instruction_id}.json").write_text(
                    json.dumps(
                        {
                            "method": "GroundPlan",
                            "map_id": map_id,
                            "scene_id": map_id,
                            "instruction_id": instruction_id,
                            "difficulty_level": "easy",
                            "trajectory": [[0, 0]],
                            "metrics": metrics,
                        }
                    ),
                    encoding="utf-8",
                )
                (prediction_dir / f"{instruction_id}.steps.json").write_text(
                    json.dumps(
                        {
                            "method": "GroundPlan",
                            "map_id": map_id,
                            "scene_id": map_id,
                            "instruction_id": instruction_id,
                            "steps": {},
                        }
                    ),
                    encoding="utf-8",
                )

            with patch(
                "scripts.methods.groundplan.run.write_summary",
                wraps=write_summary,
            ) as summary_writer:
                results = build_all(input_root, output_root, overwrite=False)

            self.assertEqual(len(results), 2)
            self.assertTrue(
                all(result["status"] == "skipped_exists" for result in results)
            )
            self.assertEqual(summary_writer.call_count, 1)
            self.assertEqual(len(summary_writer.call_args.args[0]), 2)
            summary = json.loads(
                (output_root / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["overall"]["record_count"], 2)
