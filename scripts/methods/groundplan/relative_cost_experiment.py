#!/usr/bin/env python3
"""Compare GroundPlan relative-cost formulations with oracle annotations.

The experiment performs zero LLM calls. It selects the first N val-unseen
instructions containing a relative preference, converts annotations directly
to GroundedProgram, and sweeps independently tuned weights for each cost mode.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluate_prediction import evaluate_prediction
from scripts.make_instruction.make_instruction import REPO_ROOT, load_map_state
from scripts.methods.groundplan.config import MAX_EXPANSIONS
from scripts.methods.groundplan.grounding.scene import SceneMap
from scripts.methods.groundplan.planner import plan_grounded_program
from scripts.methods.groundplan.planner.debug_annotation_planner import (
    _grounded_program_from_annotations,
)
from scripts.methods.tutorial.utils import metric_block, raw_values_for_preference
from scripts.methods.util.instructions import (
    DEFAULT_INSTRUCTION_SET,
    INSTRUCTION_ROOT,
    instruction_id_from_payload,
    iter_instruction_files,
    load_instruction,
    map_id_from_instruction_path,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "resources"
    / "methods"
    / "groundplan"
    / "ablation"
    / "06_planner_over_oracle_grounding"
    / "relative_cost_metric_aligned_oracle20_max1m.json"
)
DEFAULT_DIFFERENCE_WEIGHTS = (64.0, 160.0, 320.0, 640.0, 1280.0)
DEFAULT_RATIO_WEIGHTS = (1.0, 4.0, 16.0, 64.0, 256.0)


def _relative_constraints(instruction: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = instruction.get("soft_constraints")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [
        item
        for item in raw
        if isinstance(item, Mapping)
        and item.get("preference_type") == "relative_preference"
    ]


def _select_paths(
    input_root: Path,
    instruction_set: str,
    limit: int,
) -> list[Path]:
    selected: list[Path] = []
    for path in iter_instruction_files(input_root, instruction_set=instruction_set):
        if _relative_constraints(load_instruction(path)):
            selected.append(path)
            if len(selected) >= limit:
                break
    return selected


def _setting_key(mode: str, weight: float) -> str:
    return f"{mode}:w={weight:g}"


def _settings(
    difference_weights: Sequence[float],
    ratio_weights: Sequence[float],
) -> list[tuple[str, float]]:
    return [
        *(("difference", float(weight)) for weight in difference_weights),
        *(("ratio", float(weight)) for weight in ratio_weights),
    ]


def _relative_episode_value(metrics: Mapping[str, object]) -> float | None:
    values = raw_values_for_preference(metrics, ("relative_preference",))
    return sum(values) / len(values) if values else None


def _run_one(
    path: Path,
    *,
    input_root: Path,
    mode: str,
    weight: float,
    max_expansions: int,
) -> dict[str, object]:
    started = time.perf_counter()
    instruction = load_instruction(path)
    map_id = map_id_from_instruction_path(path, input_root)
    instruction_id = instruction_id_from_payload(path, instruction)
    scene = SceneMap(load_map_state(map_id), instruction)
    grounded, oracle_dump = _grounded_program_from_annotations(scene, instruction)
    planned = plan_grounded_program(
        scene,
        grounded,
        max_expansions=max_expansions,
        relative_cost_mode=mode,
        relative_weight=weight,
    )
    trajectory = planned.trajectory
    if not trajectory:
        trajectory = [[row, col] for row, col in sorted(scene.start.cells)]
    metrics = evaluate_prediction(
        trajectory,
        map_id=map_id,
        instruction_id=instruction_id,
        instruction_file=path,
    )
    relative_value = _relative_episode_value(metrics)
    return {
        "setting": _setting_key(mode, weight),
        "cost_mode": mode,
        "relative_weight": weight,
        "instruction_file": str(path.relative_to(REPO_ROOT)),
        "map_id": map_id,
        "instruction_id": instruction_id,
        "relative_constraint_count": len(_relative_constraints(instruction)),
        "oracle_grounded_relative_count": sum(
            1
            for item in oracle_dump.get("soft_constraints", [])
            if isinstance(item, Mapping)
            and item.get("preference_type") == "relative_preference"
        ),
        "planner_status": planned.status,
        "planner_failure_reason": planned.failure_reason,
        "trajectory_waypoint_count": len(trajectory),
        "relative_metric": relative_value,
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if isinstance(row.get("metrics"), Mapping)]
    relative_values = [
        float(value)
        for row in valid
        for value in [row.get("relative_metric")]
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    runtimes = [
        float(value)
        for row in rows
        for value in [row.get("runtime_seconds")]
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    successes = [row.get("planner_status") == "success" for row in rows]
    return {
        "record_count": len(rows),
        "valid_metric_count": len(valid),
        "error_count": sum(1 for row in rows if "error" in row),
        "planner_success_rate": (
            sum(1 for success in successes if success) / len(successes)
            if successes
            else None
        ),
        "metric": metric_block(valid),
        "official_relative_ratio": {
            "definition": "episode mean of D_far / D_close; larger is better",
            "finite_episode_count": len(relative_values),
            "mean": _mean(relative_values),
            "median": statistics.median(relative_values) if relative_values else None,
            "satisfaction_rate_gt_1": (
                sum(value > 1.0 for value in relative_values) / len(relative_values)
                if relative_values
                else None
            ),
        },
        "mean_runtime_seconds": _mean(runtimes),
    }


def _write_output(
    output_path: Path,
    *,
    protocol: Mapping[str, object],
    selected_paths: Sequence[Path],
    records: Sequence[Mapping[str, object]],
) -> None:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in records:
        setting = row.get("setting")
        if isinstance(setting, str):
            grouped.setdefault(setting, []).append(row)
    payload = {
        "protocol": dict(protocol),
        "selected_instruction_files": [
            str(path.relative_to(REPO_ROOT)) for path in selected_paths
        ],
        "completed_run_count": len(records),
        "settings": {
            key: {
                "summary": _aggregate(sorted(rows, key=lambda row: str(row.get("instruction_file")))),
                "records": sorted(rows, key=lambda row: str(row.get("instruction_file"))),
            }
            for key, rows in sorted(grouped.items())
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-LLM oracle sweep for GroundPlan relative cost."
    )
    parser.add_argument("--input-root", type=Path, default=INSTRUCTION_ROOT)
    parser.add_argument("--set", dest="instruction_set", default=DEFAULT_INSTRUCTION_SET)
    parser.add_argument("--limit-relative", type=int, default=20)
    parser.add_argument(
        "--difference-weights",
        nargs="+",
        type=float,
        default=DEFAULT_DIFFERENCE_WEIGHTS,
    )
    parser.add_argument(
        "--ratio-weights",
        nargs="+",
        type=float,
        default=DEFAULT_RATIO_WEIGHTS,
    )
    parser.add_argument("--max-expansions", type=int, default=MAX_EXPANSIONS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore resumable records already present in --output.",
    )
    args = parser.parse_args()

    if args.limit_relative <= 0:
        parser.error("--limit-relative must be positive")
    all_weights = [*args.difference_weights, *args.ratio_weights]
    if any(not math.isfinite(weight) or weight < 0 for weight in all_weights):
        parser.error("all weights must be finite and non-negative")

    paths = _select_paths(args.input_root, args.instruction_set, args.limit_relative)
    if len(paths) < args.limit_relative:
        parser.error(
            f"found only {len(paths)} relative instructions; requested {args.limit_relative}"
        )
    settings = _settings(args.difference_weights, args.ratio_weights)
    protocol = {
        "llm_calls": 0,
        "grounding": "oracle annotations via _grounded_program_from_annotations",
        "instruction_set": args.instruction_set,
        "first_n_relative_instructions": args.limit_relative,
        "planner": "global_progress_astar",
        "max_expansions": args.max_expansions,
        "difference_formula": "clip(max(0, d_close-d_far)/relative_radius, 0, 1)",
        "ratio_formula": "d_close/max(d_close+d_far, 1e-6) = 1/(1+D_far/D_close)",
        "official_metric": "D_far/D_close, larger_is_better",
        "difference_weights": list(args.difference_weights),
        "ratio_weights": list(args.ratio_weights),
        "all_non_relative_costs": "unchanged",
    }

    records: list[dict[str, object]] = []
    if args.output.exists() and not args.fresh:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        existing_protocol = existing.get("protocol") if isinstance(existing, Mapping) else None
        compatibility_fields = (
            "grounding",
            "instruction_set",
            "first_n_relative_instructions",
            "planner",
            "max_expansions",
            "difference_formula",
            "ratio_formula",
            "official_metric",
            "all_non_relative_costs",
        )
        compatible = isinstance(existing_protocol, Mapping) and all(
            existing_protocol.get(field) == protocol.get(field)
            for field in compatibility_fields
        )
        if compatible:
            for setting_payload in existing.get("settings", {}).values():
                if isinstance(setting_payload, Mapping):
                    raw_records = setting_payload.get("records", [])
                    if isinstance(raw_records, list):
                        records.extend(
                            row for row in raw_records if isinstance(row, dict)
                        )
        else:
            print(
                f"Ignoring incompatible resumable records in {args.output}.",
                flush=True,
            )

    expected = {
        (
            str(path.relative_to(REPO_ROOT)),
            mode,
            float(weight),
        )
        for path in paths
        for mode, weight in settings
    }
    records = [
        row
        for row in records
        if (
            str(row.get("instruction_file")),
            str(row.get("cost_mode")),
            float(row.get("relative_weight", -1.0)),
        )
        in expected
        and "error" not in row
    ]
    completed_keys = {
        (
            str(row.get("instruction_file")),
            str(row.get("cost_mode")),
            float(row.get("relative_weight", -1.0)),
        )
        for row in records
    }
    pending = [
        (path, mode, weight)
        for path in paths
        for mode, weight in settings
        if (str(path.relative_to(REPO_ROOT)), mode, weight) not in completed_keys
    ]

    print(
        f"Selected {len(paths)} relative instructions; "
        f"{len(settings)} settings; {len(pending)} pending runs; zero LLM calls.",
        flush=True,
    )
    _write_output(
        args.output,
        protocol=protocol,
        selected_paths=paths,
        records=records,
    )

    # Some evaluator/map helpers retain process-global state. Submit exactly
    # one task per worker, then replace the whole pool for the next batch.
    worker_count = max(1, args.workers)
    completed = 0
    for batch_start in range(0, len(pending), worker_count):
        batch = pending[batch_start : batch_start + worker_count]
        with ProcessPoolExecutor(max_workers=len(batch)) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    path,
                    input_root=args.input_root,
                    mode=mode,
                    weight=weight,
                    max_expansions=args.max_expansions,
                ): (path, mode, weight)
                for path, mode, weight in batch
            }
            for future in as_completed(futures):
                completed += 1
                path, mode, weight = futures[future]
                try:
                    row = future.result()
                    state = "done"
                except Exception as exc:
                    row = {
                        "setting": _setting_key(mode, weight),
                        "cost_mode": mode,
                        "relative_weight": weight,
                        "instruction_file": str(path.relative_to(REPO_ROOT)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    state = "error"
                records.append(row)
                _write_output(
                    args.output,
                    protocol=protocol,
                    selected_paths=paths,
                    records=records,
                )
                print(
                    f"[{completed}/{len(pending)}] {mode} w={weight:g} "
                    f"{path.parent.parent.name}/{path.name} {state}",
                    flush=True,
                )

    print(args.output, flush=True)


if __name__ == "__main__":
    main()
