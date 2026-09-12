#!/usr/bin/env python3
"""Build episode-level data for the benchmark difficulty analysis.

This script deliberately reuses the reference parsing and candidate-counting
logic in ``valunseen_dataset_statistics.py``.  It joins those complexity
features to the saved per-episode HCS values for GroundPlan and OSG-LLM, then
writes both a flat CSV and a plot-ready JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from valunseen_dataset_statistics import (
    DEFAULT_EASY_REFERENCE_PATH,
    DEFAULT_HARD_REFERENCE_PATH,
    DEFAULT_INSTRUCTION_ROOT,
    DEFAULT_MAP_ROOT,
    REPO_ROOT,
    candidate_catalog,
    candidate_for_reference,
    instances,
    instruction_paths,
    load_object,
    map_paths,
    normalized_difficulty,
    reference_records,
)


STATICS_ROOT = Path(__file__).resolve().parent
DEFAULT_GROUNDPLAN_ROOT = REPO_ROOT / "resources" / "methods" / "groundplan" / "main_result" / "procthor"
DEFAULT_BASELINE_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / "OSGLLM" / "procthor"
DEFAULT_CSV_OUTPUT = STATICS_ROOT / "difficulty_analysis_episodes.csv"
DEFAULT_JSON_OUTPUT = STATICS_ROOT / "difficulty_analysis_summary.json"
METHOD_COLUMNS = {
    "GroundPlan": "groundplan_hcs",
    "OSG-LLM": "osgllm_hcs",
}

# The values are discrete, so explicit, interpretable bins are preferable to
# quantiles whose labels can change after a benchmark update.  These boundaries
# are intentionally centralized for easy paper-figure iteration.  Each tuple is
# (inclusive upper bound, display label); None means no upper bound.
BIN_SPECS: dict[str, list[tuple[int | None, str]]] = {
    "reference_depth": [(2, "1–2"), (3, "3"), (4, "4"), (None, "≥5")],
    "candidate_instances": [(4, "1–4"), (8, "5–8"), (16, "9–16"), (None, "≥17")],
    "hard_constraints": [(1, "1"), (2, "2"), (3, "3"), (None, "≥4")],
}
FACTOR_TITLES = {
    "reference_depth": "Reference Depth",
    "candidate_instances": "Candidate Instances",
    "hard_constraints": "Number of Hard Constraints",
}


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def instruction_key(instruction_path: Path, payload: Mapping[str, object]) -> str:
    map_id = payload.get("map_id")
    if not isinstance(map_id, str) or not map_id.strip():
        map_id = f"procthor/{instruction_path.parent.parent.name}"
    raw_id = payload.get("id")
    try:
        instruction_id = f"instruction_{int(str(raw_id)):06d}"
    except (TypeError, ValueError):
        instruction_id = instruction_path.stem
    return f"{map_id.strip().strip('/')}/{instruction_id}"


def saved_hcs(result_path: Path) -> float:
    payload = load_object(result_path)
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Missing metrics object: {display_path(result_path)}")
    raw_hcs = metrics.get("HCS")
    if not isinstance(raw_hcs, (int, float)) or isinstance(raw_hcs, bool):
        raise ValueError(f"Missing numeric metrics.HCS: {display_path(result_path)}")
    hcs = float(raw_hcs)
    if not math.isfinite(hcs) or not 0.0 <= hcs <= 1.0:
        raise ValueError(f"HCS must be finite and in [0, 1]: {display_path(result_path)}")
    return hcs


def result_path(method_root: Path, instruction_path: Path) -> Path:
    path = method_root / instruction_path.parent.parent.name / instruction_path.name
    if not path.is_file():
        raise ValueError(f"Missing method result: {display_path(path)}")
    return path


def episode_complexity(
    instruction: Mapping[str, object],
    references: Sequence[Mapping[str, object]],
    catalog: tuple[Counter[str], Counter[str], int, int],
) -> tuple[int, int, int, int]:
    """Return max depth, max candidates, scored hard count, unresolved count."""

    depths: list[int] = []
    candidate_counts: list[int] = []
    unresolved_count = 0
    for reference in references:
        raw_depth = reference.get("depth")
        if isinstance(raw_depth, (int, float)) and not isinstance(raw_depth, bool):
            depths.append(int(raw_depth))
        candidate = candidate_for_reference(reference, catalog)
        if candidate is None:
            unresolved_count += 1
        else:
            candidate_counts.append(int(candidate[1]))

    if not depths:
        raise ValueError("Episode has no annotated reference depths.")
    if not candidate_counts:
        raise ValueError("Episode has no resolvable candidate counts.")

    # The first must-pass entry records the episode's fixed starting anchor.
    # Existing benchmark statistics call the remainder "additional hard
    # constraints"; these are the constraints whose number varies by task.
    hard_constraint_count = max(0, len(instances(instruction, "hard_constraints")) - 1)
    return max(depths), max(candidate_counts), hard_constraint_count, unresolved_count


def collect_rows(
    *,
    map_root: Path,
    instruction_root: Path,
    easy_reference_path: Path,
    hard_reference_path: Path,
    groundplan_root: Path,
    baseline_root: Path,
) -> list[dict[str, Any]]:
    maps = map_paths(map_root)
    instructions = instruction_paths(instruction_root)
    catalogs = {
        path.parent.name: candidate_catalog(load_object(path)) for path in maps
    }
    references_by_difficulty = {
        "easy": reference_records(easy_reference_path),
        "hard": reference_records(hard_reference_path),
    }
    instructions_by_difficulty: dict[str, list[Path]] = {"easy": [], "hard": []}
    for path in instructions:
        difficulty = normalized_difficulty(load_object(path))
        if difficulty not in instructions_by_difficulty:
            raise ValueError(f"Unknown difficulty {difficulty!r}: {display_path(path)}")
        instructions_by_difficulty[difficulty].append(path)

    rows: list[dict[str, Any]] = []
    for difficulty in ("easy", "hard"):
        reference_records_for_split = references_by_difficulty[difficulty]
        split_instructions = instructions_by_difficulty[difficulty]
        if len(reference_records_for_split) != len(split_instructions):
            raise ValueError(
                f"{difficulty} reference/instruction mismatch: "
                f"{len(reference_records_for_split)} vs {len(split_instructions)}"
            )
        for ordinal, path in enumerate(split_instructions, start=1):
            instruction = load_object(path)
            references = reference_records_for_split.get(ordinal)
            if references is None:
                raise ValueError(f"Missing {difficulty} reference record {ordinal}")
            scene_name = path.parent.parent.name
            complexity = episode_complexity(instruction, references, catalogs[scene_name])
            max_depth, max_candidates, hard_count, unresolved_count = complexity
            rows.append(
                {
                    "episode": instruction_key(path, instruction),
                    "difficulty": difficulty,
                    "reference_depth": max_depth,
                    "candidate_instances": max_candidates,
                    "hard_constraints": hard_count,
                    "unresolved_references": unresolved_count,
                    "groundplan_hcs": saved_hcs(result_path(groundplan_root, path)),
                    "osgllm_hcs": saved_hcs(result_path(baseline_root, path)),
                }
            )

    if len(rows) != len(instructions):
        raise ValueError(f"Collected {len(rows)} rows for {len(instructions)} instructions")
    if len({row["episode"] for row in rows}) != len(rows):
        raise ValueError("Episode keys are not unique.")
    return rows


def bin_index(value: int, bins: Sequence[tuple[int | None, str]]) -> int:
    for index, (upper, _label) in enumerate(bins):
        if upper is None or value <= upper:
            return index
    raise AssertionError("Final bin must have no upper bound.")


def summarize_factor(rows: Sequence[Mapping[str, Any]], factor: str) -> dict[str, Any]:
    bins = BIN_SPECS[factor]
    grouped: list[list[Mapping[str, Any]]] = [[] for _ in bins]
    for row in rows:
        grouped[bin_index(int(row[factor]), bins)].append(row)

    summaries: list[dict[str, Any]] = []
    lower = 0
    for (upper, label), bin_rows in zip(bins, grouped, strict=True):
        if not bin_rows:
            raise ValueError(f"Empty {factor} bin: {label}")
        summaries.append(
            {
                "label": label,
                "lower_inclusive": lower,
                "upper_inclusive": upper,
                "count": len(bin_rows),
                "value_min": min(int(row[factor]) for row in bin_rows),
                "value_max": max(int(row[factor]) for row in bin_rows),
                "mean_hcs": {
                    method: mean(float(row[column]) for row in bin_rows)
                    for method, column in METHOD_COLUMNS.items()
                },
            }
        )
        lower = (upper + 1) if upper is not None else lower
    return {
        "title": FACTOR_TITLES[factor],
        "episode_aggregation": "maximum" if factor != "hard_constraints" else "count excluding start anchor",
        "bins": summaries,
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    map_root: Path,
    instruction_root: Path,
    easy_reference_path: Path,
    hard_reference_path: Path,
    groundplan_root: Path,
    baseline_root: Path,
) -> dict[str, Any]:
    return {
        "version": 1,
        "benchmark": "RouteSemBench ProcTHOR valunseen",
        "episode_count": len(rows),
        "methods": list(METHOD_COLUMNS),
        "strongest_baseline": "OSG-LLM",
        "sources": {
            "map_root": display_path(map_root),
            "instruction_root": display_path(instruction_root),
            "easy_reference_annotations": display_path(easy_reference_path),
            "hard_reference_annotations": display_path(hard_reference_path),
            "GroundPlan_results": display_path(groundplan_root),
            "OSG-LLM_results": display_path(baseline_root),
            "saved_metric_field": "metrics.HCS",
        },
        "definitions": {
            "reference_depth": "Maximum human-annotated referential/relational depth in the episode.",
            "candidate_instances": "Maximum category-level candidate count before relational filtering among resolved references in the episode.",
            "hard_constraints": "Number of variable task hard constraints after excluding the fixed start-anchor must-pass record.",
        },
        "unresolved_reference_count": sum(int(row["unresolved_references"]) for row in rows),
        "factors": {
            factor: summarize_factor(rows, factor) for factor in BIN_SPECS
        },
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: Mapping[str, Any], csv_output: Path, json_output: Path) -> None:
    print("Difficulty-analysis data")
    print(f"  episodes: {summary['episode_count']}")
    print(f"  strongest baseline: {summary['strongest_baseline']} (fixed by paper setup)")
    print("  source files/directories:")
    for name, path in summary["sources"].items():
        print(f"    {name}: {path}")
    print(f"  episode table: {display_path(csv_output)}")
    print(f"  binned summary: {display_path(json_output)}")
    print(f"  unresolved references ignored for candidate maxima: {summary['unresolved_reference_count']}")
    for factor, factor_summary in summary["factors"].items():
        print(f"\n{factor_summary['title']} ({factor})")
        print(f"  episode aggregation: {factor_summary['episode_aggregation']}")
        for item in factor_summary["bins"]:
            means = item["mean_hcs"]
            print(
                f"  {item['label']:>4}  n={item['count']:>3}  "
                f"GroundPlan={means['GroundPlan']:.3f}  OSG-LLM={means['OSG-LLM']:.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--instruction-root", type=Path, default=DEFAULT_INSTRUCTION_ROOT)
    parser.add_argument("--easy-reference", type=Path, default=DEFAULT_EASY_REFERENCE_PATH)
    parser.add_argument("--hard-reference", type=Path, default=DEFAULT_HARD_REFERENCE_PATH)
    parser.add_argument("--groundplan-root", type=Path, default=DEFAULT_GROUNDPLAN_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_args = {
        "map_root": args.map_root.resolve(),
        "instruction_root": args.instruction_root.resolve(),
        "easy_reference_path": args.easy_reference.resolve(),
        "hard_reference_path": args.hard_reference.resolve(),
        "groundplan_root": args.groundplan_root.resolve(),
        "baseline_root": args.baseline_root.resolve(),
    }
    rows = collect_rows(**path_args)
    summary = build_summary(rows, **path_args)
    csv_output = args.csv_output.resolve()
    json_output = args.json_output.resolve()
    write_csv(csv_output, rows)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_summary(summary, csv_output, json_output)


if __name__ == "__main__":
    main()
