#!/usr/bin/env python3
"""Browser-based human-solver method for SemPathBench instructions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.evaluation.evaluate_prediction import (
    evaluate_prediction,
    load_instruction_by_id,
)
from scripts.make_instruction.make_instruction import (
    REPO_ROOT,
    instruction_files_directory,
    list_map_summaries,
    load_map_state,
    normalize_map_key,
    validate_route,
)
from scripts.methods.util.instructions import (
    DEFAULT_INSTRUCTION_SET,
    INSTRUCTION_SET_CHOICES,
    instruction_id_from_payload,
    map_id_matches_instruction_set,
    normalize_instruction_set,
)
from scripts.methods.util.methods import save_trajectory_image_for_record
from scripts.methods.tutorial.utils import score_block as shared_score_block


METHOD_NAME = "human_solver"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / METHOD_NAME
HUMAN_SOLVER_RESOURCE_ROOT = REPO_ROOT / "resources" / METHOD_NAME
SUMMARY_PATH = METHOD_ROOT / "summary.json"
DIFFICULTY_LEVELS = ("easy", "hard", "extreme", "unknown")


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compact_json_for_html(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")


def display_path(path: Path) -> str:
    if path.is_relative_to(REPO_ROOT):
        return path.relative_to(REPO_ROOT).as_posix()
    return str(path)


def metric_value(metrics: Mapping[str, object] | None, key: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def metric_summary(metrics: Mapping[str, object] | None) -> dict[str, float | None]:
    return {
        "HCS": metric_value(metrics, "HCS"),
        "SCS": metric_value(metrics, "SCS"),
        "PL": metric_value(metrics, "PL"),
        "segment_wise_SPL": metric_value(metrics, "segment_wise_SPL"),
    }


def runtime_seconds(record: Mapping[str, object]) -> float | None:
    value = record.get("runtime_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    runtime = record.get("runtime")
    if isinstance(runtime, Mapping):
        nested_value = runtime.get("seconds")
        if isinstance(nested_value, (int, float)) and not isinstance(
            nested_value,
            bool,
        ):
            return float(nested_value)
    return None


def runtime_summary(records: list[dict[str, object]]) -> dict[str, object]:
    values = [value for record in records if (value := runtime_seconds(record)) is not None]
    return {
        "total_seconds": sum(values),
        "average_seconds": mean(values),
    }


def solution_files_for_instruction(map_id: str, instruction_id: str) -> list[Path]:
    return sorted((METHOD_ROOT / normalize_map_key(map_id) / instruction_id).glob("*.json"))


def active_solution_files() -> list[Path]:
    if not METHOD_ROOT.exists():
        return []
    return sorted(
        path
        for path in METHOD_ROOT.rglob("*.json")
        if "_archived" not in path.relative_to(METHOD_ROOT).parts
    )


def latest_solution_file_for_instruction(map_id: str, instruction_id: str) -> Path | None:
    solution_files = solution_files_for_instruction(map_id, instruction_id)
    return solution_files[-1] if solution_files else None


def load_latest_solution(map_id: str, instruction_id: str) -> tuple[dict[str, object], Path]:
    path = latest_solution_file_for_instruction(map_id, instruction_id)
    if path is None:
        raise ValueError(f"No saved human-solver solution for {map_id}/{instruction_id}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Solution file must contain a JSON object: {path}")
    return payload, path


def difficulty_from_solution_payload(payload: Mapping[str, object]) -> str:
    instruction = payload.get("instruction")
    if isinstance(instruction, Mapping):
        difficulty = instruction.get("difficulty_level")
        if isinstance(difficulty, str) and difficulty.strip():
            return difficulty.strip().lower()
    difficulty = payload.get("difficulty_level")
    if isinstance(difficulty, str) and difficulty.strip():
        return difficulty.strip().lower()
    return "unknown"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def metric_values(records: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        metrics = record.get("metrics")
        value = metric_value(metrics, key) if isinstance(metrics, Mapping) else None
        if value is None:
            summary = record.get("metrics_summary")
            if isinstance(summary, Mapping):
                raw_value = summary.get(key)
                if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                    value = float(raw_value)
        if value is not None:
            values.append(value)
    return values


def soft_scores_for_preference(
    metrics: Mapping[str, object],
    preference_type: str,
) -> list[float]:
    details = metrics.get("details")
    if not isinstance(details, Mapping):
        return []

    entries: object = None
    by_type = details.get("soft_preferences_by_type")
    if isinstance(by_type, Mapping):
        entries = by_type.get(preference_type)
    if entries is None:
        soft_constraints = details.get("soft_constraints")
        if isinstance(soft_constraints, list):
            entries = [
                detail
                for detail in soft_constraints
                if isinstance(detail, Mapping)
                and detail.get("preference_type") == preference_type
            ]

    if not isinstance(entries, list):
        return []

    values: list[float] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        score = entry.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            values.append(float(score))
    return values


def soft_scs_values(
    records: list[dict[str, object]],
    preference_type: str,
) -> list[float]:
    values: list[float] = []
    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        scores = soft_scores_for_preference(metrics, preference_type)
        if scores:
            values.append(compute_scs(scores))
    return values


def score_block(records: list[dict[str, object]]) -> dict[str, object]:
    return shared_score_block(records)


def update_summary_file() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for path in active_solution_files():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics")
        trajectory = payload.get("trajectory")
        map_id = payload.get("map_id")
        instruction_id = payload.get("instruction_id")
        if (
            isinstance(trajectory, list)
            and isinstance(map_id, str)
            and isinstance(instruction_id, str)
        ):
            try:
                metrics = evaluate_prediction(
                    trajectory,
                    map_id=map_id,
                    instruction_id=instruction_id,
                )
                metrics_summary = metric_summary(metrics)
                if (
                    metrics != payload.get("metrics")
                    or metrics_summary != payload.get("metrics_summary")
                ):
                    payload["metrics"] = metrics
                    payload["metrics_summary"] = metrics_summary
                    path.write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
            except Exception:
                metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        difficulty_level = difficulty_from_solution_payload(payload)
        scene_id = payload.get("scene_id", payload.get("map_id"))
        if not isinstance(scene_id, str) or not scene_id.strip():
            scene_id = str(payload.get("map_id") or "")
        records.append(
            {
                "solution_id": payload.get("solution_id"),
                "map_id": payload.get("map_id"),
                "scene_id": scene_id,
                "instruction_id": payload.get("instruction_id"),
                "difficulty_level": difficulty_level,
                "created_at": payload.get("created_at"),
                "created_by": payload.get("created_by"),
                "path": display_path(path),
                "trajectory_image": payload.get("trajectory_image"),
                "runtime_seconds": payload.get("runtime_seconds"),
                "runtime": payload.get("runtime"),
                "trajectory_length": len(trajectory) if isinstance(trajectory, list) else 0,
                "metrics": metrics,
                "metrics_summary": metric_summary(metrics),
            }
        )

    by_difficulty = {
        difficulty: [
            record
            for record in records
            if str(record.get("difficulty_level") or "unknown").lower() == difficulty
        ]
        for difficulty in DIFFICULTY_LEVELS
    }
    summary = {
        "version": 1,
        "method": METHOD_NAME,
        "runtime_summary": runtime_summary(records),
        "overall": score_block(records),
        **{
            difficulty: score_block(by_difficulty[difficulty])
            for difficulty in ("easy", "hard")
        },
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def list_instruction_records(map_id: str) -> list[dict[str, object]]:
    map_key = normalize_map_key(map_id)
    directory = instruction_files_directory(map_key)
    records: list[dict[str, object]] = []
    for path in sorted(directory.glob("instruction_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        instruction_id = instruction_id_from_payload(path, payload)
        solution_files = solution_files_for_instruction(map_key, instruction_id)
        latest_solution_file = solution_files[-1] if solution_files else None
        records.append(
            {
                "instruction_id": instruction_id,
                "path": path,
                "path_display": path.relative_to(REPO_ROOT).as_posix(),
                "name": payload.get("name") or instruction_id,
                "instruction": payload.get("instruction") or "",
                "difficulty_level": payload.get("difficulty_level") or "",
                "object_count": payload.get("object_count"),
                "finished": bool(solution_files),
                "solution_count": len(solution_files),
                "latest_solution_path": (
                    latest_solution_file.relative_to(REPO_ROOT).as_posix()
                    if latest_solution_file is not None
                    else None
                ),
            }
        )
    return records


def map_summaries_with_instruction_counts(
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
) -> list[dict[str, object]]:
    selected_set = normalize_instruction_set(instruction_set)
    summaries: list[dict[str, object]] = []
    for summary in list_map_summaries():
        map_id = str(summary["map_id"])
        if not map_id_matches_instruction_set(map_id, selected_set):
            continue
        records = list_instruction_records(map_id)
        item: dict[str, object] = dict(summary)
        item["instruction_count"] = len(records)
        item["finished_count"] = sum(1 for record in records if record["finished"])
        summaries.append(item)
    return summaries


def resolve_initial_map_id(
    requested_map_id: str | None,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
) -> str:
    if requested_map_id:
        return str(load_map_state(requested_map_id)["map_key"])
    summaries = map_summaries_with_instruction_counts(instruction_set)
    with_instructions = next(
        (item for item in summaries if int(item.get("instruction_count", 0)) > 0),
        None,
    )
    if with_instructions is not None:
        return str(with_instructions["map_id"])
    if summaries:
        return str(summaries[0]["map_id"])
    return str(load_map_state(None)["map_key"])

def output_path_for_solution(
    map_id: str,
    instruction_id: str,
    solution_id: str,
) -> Path:
    return (
        METHOD_ROOT
        / normalize_map_key(map_id)
        / instruction_id
        / f"{solution_id}.json"
    )


def save_solution(payload: Mapping[str, object]) -> dict[str, object]:
    episode_start = time.perf_counter()
    map_id = payload.get("map_id")
    instruction_id = payload.get("instruction_id")
    trajectory = payload.get("trajectory")
    solver_id = payload.get("solver_id", "")
    if not isinstance(map_id, str) or not map_id.strip():
        raise ValueError("map_id must be a non-empty string.")
    if not isinstance(instruction_id, str) or not instruction_id.strip():
        raise ValueError("instruction_id must be a non-empty string.")

    map_state = load_map_state(map_id)
    resolved_map_id = str(map_state["map_key"])
    route = validate_route(
        trajectory,
        int(map_state["grid_size"]),
        label="trajectory",
    )
    if not route:
        raise ValueError("trajectory cannot be empty.")

    instruction, instruction_path = load_instruction_by_id(
        resolved_map_id,
        instruction_id,
    )
    metrics = evaluate_prediction(
        route,
        map_id=resolved_map_id,
        instruction_id=instruction_id,
    )
    raw_scene_id = instruction.get("map_id")
    scene_id = raw_scene_id.strip() if isinstance(raw_scene_id, str) and raw_scene_id.strip() else resolved_map_id
    difficulty_level = difficulty_from_solution_payload({"instruction": instruction})
    created_at = iso_now()
    stamp = created_at.replace(":", "").replace("-", "")
    solution_id = f"{METHOD_NAME}_{stamp}_{uuid.uuid4().hex[:8]}"
    record = {
        "version": 1,
        "method": METHOD_NAME,
        "solution_id": solution_id,
        "map_id": resolved_map_id,
        "scene_id": scene_id,
        "instruction_id": instruction_id,
        "difficulty_level": difficulty_level,
        "instruction_file": display_path(instruction_path),
        "created_at": created_at,
        "created_by": solver_id if isinstance(solver_id, str) else "",
        "trajectory": route,
        "metrics": metrics,
        "metrics_summary": metric_summary(metrics),
        "instruction": {
            "text": instruction.get("instruction", ""),
            "name": instruction.get("name", ""),
            "difficulty_level": instruction.get("difficulty_level", ""),
            "template_instruction_id": instruction.get("template_instruction_id"),
        },
    }
    output_path = output_path_for_solution(resolved_map_id, instruction_id, solution_id)
    save_trajectory_image_for_record(
        record,
        map_state,
        route,
        output_path,
        instruction=instruction,
    )
    runtime_seconds_value = time.perf_counter() - episode_start
    record["runtime_seconds"] = runtime_seconds_value
    record["runtime"] = {
        "seconds": runtime_seconds_value,
        "status": "written",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = update_summary_file()
    return {
        "message": f"Saved {solution_id}.",
        "solution_id": solution_id,
        "path": display_path(output_path),
        "record": record,
        "summary_path": display_path(SUMMARY_PATH),
        "summary": summary,
    }


def unfinish_instruction(payload: Mapping[str, object]) -> dict[str, object]:
    map_id = payload.get("map_id")
    instruction_id = payload.get("instruction_id")
    if not isinstance(map_id, str) or not map_id.strip():
        raise ValueError("map_id must be a non-empty string.")
    if not isinstance(instruction_id, str) or not instruction_id.strip():
        raise ValueError("instruction_id must be a non-empty string.")

    map_state = load_map_state(map_id)
    resolved_map_id = str(map_state["map_key"])
    solution_files = solution_files_for_instruction(resolved_map_id, instruction_id)
    if not solution_files:
        summary = update_summary_file()
        return {
            "message": f"{instruction_id} is already unfinished.",
            "archived_count": 0,
            "summary_path": display_path(SUMMARY_PATH),
            "summary": summary,
        }

    archive_stamp = iso_now().replace(":", "").replace("-", "")
    archive_dir = (
        METHOD_ROOT
        / "_archived"
        / normalize_map_key(resolved_map_id)
        / instruction_id
        / archive_stamp
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_paths: list[str] = []
    for path in solution_files:
        target = archive_dir / path.name
        shutil.move(str(path), str(target))
        archived_paths.append(target.relative_to(REPO_ROOT).as_posix())

    for directory in [
        METHOD_ROOT / normalize_map_key(resolved_map_id) / instruction_id,
        METHOD_ROOT / normalize_map_key(resolved_map_id),
    ]:
        try:
            directory.rmdir()
        except OSError:
            pass

    summary = update_summary_file()
    return {
        "message": f"Marked {instruction_id} unfinished and archived {len(archived_paths)} solution file(s).",
        "archived_count": len(archived_paths),
        "archived_paths": archived_paths,
        "summary_path": display_path(SUMMARY_PATH),
        "summary": summary,
    }


def build_client_state(
    map_id: str | None = None,
    instruction_set: str = DEFAULT_INSTRUCTION_SET,
) -> dict[str, object]:
    selected_set = normalize_instruction_set(instruction_set)
    resolved_map_id = resolve_initial_map_id(map_id, selected_set)
    map_state = load_map_state(resolved_map_id)
    resolved_map_id = str(map_state["map_key"])
    instructions = list_instruction_records(resolved_map_id)
    return {
        "method": METHOD_NAME,
        "instruction_set": selected_set,
        "current_map_id": resolved_map_id,
        "map_summaries": map_summaries_with_instruction_counts(selected_set),
        "map_state": map_state,
        "instructions": [
            {
                key: value
                for key, value in record.items()
                if key != "path"
            }
            for record in instructions
        ],
        "paths": {
            "method_root": display_path(METHOD_ROOT),
            "summary_json": display_path(SUMMARY_PATH),
            "instruction_directory": instruction_files_directory(
                resolved_map_id
            ).relative_to(REPO_ROOT).as_posix(),
        },
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SemPathBench Human Solver</title>
  <style>
    :root {
      --bg: #f7f8f6;
      --panel: #ffffff;
      --line: #d6dbd2;
      --ink: #20231f;
      --muted: #667061;
      --accent: #126b5d;
      --accent-2: #b44d3e;
      --soft: #e7f2ef;
      --warn: #a43d32;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    button, input, select, textarea { font: inherit; }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 360px;
      gap: 12px;
      padding: 12px;
    }
    .panel, .canvas-wrap {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .panel-scroll {
      max-height: calc(100vh - 48px);
      overflow-y: auto;
      padding-right: 4px;
    }
    h1, h2, p { margin: 0; }
    h1 { font-size: 22px; margin-bottom: 6px; }
    h2 { font-size: 18px; margin-bottom: 8px; }
    .hint, .small { color: var(--muted); font-size: 13px; line-height: 1.45; }
    .section { margin-top: 14px; display: grid; gap: 8px; }
    .section-title {
      font-size: 11px;
      font-weight: 800;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .grow { flex: 1 1 auto; min-width: 0; }
    input[type="text"], select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fbfcfa;
      color: var(--ink);
    }
    textarea { min-height: 220px; resize: vertical; line-height: 1.45; }
    .instruction-text {
      width: 100%;
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfa;
      color: var(--ink);
      line-height: 1.55;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 9px 11px;
      background: #222722;
      color: #fff;
      cursor: pointer;
    }
    button.alt { background: #e6e9e3; color: var(--ink); }
    button.warn { background: var(--warn); }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfa;
    }
    .stat-label { color: var(--muted); font-size: 12px; }
    .stat-value { margin-top: 3px; font-size: 18px; font-weight: 800; }
    .instruction-list { display: grid; gap: 7px; max-height: 250px; overflow-y: auto; }
    .instruction-item {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfa;
      cursor: pointer;
      text-align: left;
      color: var(--ink);
    }
    .instruction-item.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(18, 107, 93, 0.15);
    }
    .instruction-item.finished {
      background: #eef4f1;
    }
    .instruction-item.unfinished {
      background: #fffdfa;
    }
    .filter-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfa;
    }
    .mono {
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      color: #4e584b;
      line-height: 1.45;
      word-break: break-word;
    }
    .canvas-wrap { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
    .toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .status {
      border: 1px solid #cbd9d4;
      border-radius: 6px;
      background: var(--soft);
      padding: 9px 10px;
      font-size: 13px;
      line-height: 1.4;
    }
    .inspector {
      display: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfa;
      padding: 9px 10px;
      font-size: 13px;
      line-height: 1.45;
    }
    .inspector.visible {
      display: block;
    }
    .canvas-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: auto;
      max-height: calc(100vh - 168px);
    }
    canvas { display: block; image-rendering: pixelated; cursor: crosshair; }
    .soft-detail {
      display: grid;
      gap: 6px;
      max-height: 260px;
      overflow-y: auto;
    }
    .soft-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      border-bottom: 1px solid #edf0eb;
      padding-bottom: 6px;
      font-size: 12px;
    }
    @media (max-width: 1100px) {
      .shell { grid-template-columns: 1fr; }
      .panel-scroll { max-height: none; }
      .canvas-box { max-height: none; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel">
      <div class="panel-scroll">
        <h1>Human Solver</h1>
        <p class="hint">Read one saved instruction, draw the trajectory, and save the human-method result.</p>

        <div class="section">
          <div class="section-title">Map</div>
          <div class="row">
            <select id="mapSelect" class="grow"></select>
            <button id="openMapBtn" type="button">Open</button>
          </div>
          <div id="mapInfo" class="item small"></div>
        </div>

        <div class="section">
          <div class="section-title">Solver</div>
          <input id="solverIdInput" type="text" placeholder="Name or participant ID">
        </div>

        <div class="section">
          <div class="section-title">Actions</div>
          <div class="row">
            <button id="saveBtn" type="button">Save Solution</button>
            <button id="unfinishBtn" type="button" class="warn">Mark Unfinished</button>
            <button id="clearBtn" type="button" class="alt">Clear</button>
            <button id="undoBtn" type="button" class="alt">Undo</button>
          </div>
        </div>

        <div class="section">
          <div class="section-title">Trajectory</div>
          <div id="startInfo" class="item small"></div>
          <div class="stat-grid">
            <div class="stat"><div class="stat-label">Waypoints</div><div id="waypointValue" class="stat-value">-</div></div>
            <div class="stat"><div class="stat-label">Path Length</div><div id="plValue" class="stat-value">-</div></div>
          </div>
        </div>

        <div class="section">
          <div class="section-title">Files</div>
          <div id="paths" class="mono"></div>
        </div>
      </div>
    </aside>

    <main class="canvas-wrap">
      <div class="toolbar">
        <label class="row"><span>Zoom</span><input id="zoomSlider" type="range" min="1" max="12" step="1" value="2"><span id="zoomValue">2x</span></label>
        <label class="row"><input id="gridToggle" type="checkbox" checked><span>Grid</span></label>
        <label class="row"><input id="roomToggle" type="checkbox" checked><span>Rooms</span></label>
        <label class="row"><input id="objectToggle" type="checkbox" checked><span>Objects</span></label>
        <label class="row"><input id="eraseToggle" type="checkbox"><span>Erase</span></label>
      </div>
      <div id="status" class="status">Loading...</div>
      <div id="mapInspector" class="inspector"></div>
      <div class="canvas-box"><canvas id="gridCanvas"></canvas></div>
    </main>

    <aside class="panel">
      <div class="panel-scroll">
        <h2>Instruction</h2>
        <div class="section">
          <div class="section-title">Filters</div>
          <div class="filter-grid">
            <select id="difficultyFilter">
              <option value="">All difficulty</option>
              <option value="easy">Easy</option>
              <option value="hard">Hard</option>
              <option value="extreme">Extreme</option>
            </select>
            <select id="completionFilter">
              <option value="unfinished">Unfinished</option>
              <option value="finished">Finished</option>
              <option value="">All status</option>
            </select>
          </div>
        </div>
        <div class="section">
          <div class="section-title">Saved Instructions</div>
          <div id="instructionList" class="instruction-list"></div>
        </div>
        <div class="section">
          <div class="section-title">Text</div>
          <div id="instructionText" class="instruction-text"></div>
        </div>
      </div>
    </aside>
  </div>

  <script>
    let state = __INITIAL_STATE__;
    let mapState = state.map_state;
    let currentMapId = state.current_map_id;
    let instructions = state.instructions || [];
    let activeInstructionId = instructions[0]?.instruction_id || null;
    let zoom = 2;
    let showGrid = true;
    let showRooms = true;
    let showObjects = true;
    let eraseMode = false;
    let isPainting = false;
    let lastCell = null;
    let routeGestureStartCell = null;
    let routeGestureDragged = false;
    let trajectory = [];
    let routeSegments = [];
    let activeRouteSegment = null;
    let lastSavedMetrics = null;
    let inspectedMapEntity = null;

    const canvas = document.getElementById("gridCanvas");
    const ctx = canvas.getContext("2d");
    const mapSelect = document.getElementById("mapSelect");
    const instructionList = document.getElementById("instructionList");
    const instructionText = document.getElementById("instructionText");
    const difficultyFilter = document.getElementById("difficultyFilter");
    const completionFilter = document.getElementById("completionFilter");
    const mapInspector = document.getElementById("mapInspector");
    const unfinishBtn = document.getElementById("unfinishBtn");
    const statusEl = document.getElementById("status");
    const zoomSlider = document.getElementById("zoomSlider");
    const zoomValue = document.getElementById("zoomValue");
    const solverIdInput = document.getElementById("solverIdInput");

    function setStatus(message) {
      statusEl.textContent = message;
    }

    function activeInstruction() {
      return instructions.find(item => item.instruction_id === activeInstructionId) || null;
    }

    function activeInstructionPayload() {
      return activeInstruction()?._payload || activeInstruction();
    }

    function instructionStartCell(instruction = activeInstructionPayload()) {
      const startPose = instruction?.start_pose;
      if (
        startPose &&
        Number.isInteger(startPose.row) &&
        Number.isInteger(startPose.col)
      ) {
        return [startPose.row, startPose.col];
      }
      return null;
    }

    function resetTrajectoryToStart() {
      const start = instructionStartCell();
      trajectory = start ? [[start[0], start[1]]] : [];
      routeSegments = [];
      activeRouteSegment = null;
      routeGestureStartCell = null;
      routeGestureDragged = false;
      lastSavedMetrics = null;
    }

    function loadSolverId() {
      solverIdInput.value = localStorage.getItem("sempathbench_human_solver_id") || "";
    }

    function saveSolverId() {
      localStorage.setItem("sempathbench_human_solver_id", solverIdInput.value.trim());
    }

    function hexToRgba(hex, alpha) {
      const value = hex.replace("#", "");
      const r = parseInt(value.slice(0, 2), 16);
      const g = parseInt(value.slice(2, 4), 16);
      const b = parseInt(value.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function cellKey(row, col) {
      return `${row},${col}`;
    }

    function sameCell(a, b) {
      return a && b && a[0] === b[0] && a[1] === b[1];
    }

    function interpolateLine(start, end) {
      const [r0, c0] = start;
      const [r1, c1] = end;
      const steps = Math.max(Math.abs(r1 - r0), Math.abs(c1 - c0));
      const cells = [];
      for (let index = 0; index <= steps; index += 1) {
        const t = steps === 0 ? 0 : index / steps;
        const row = Math.round(r0 + (r1 - r0) * t);
        const col = Math.round(c0 + (c1 - c0) * t);
        const key = cellKey(row, col);
        if (!cells.some(cell => cellKey(cell[0], cell[1]) === key)) {
          cells.push([row, col]);
        }
      }
      return cells;
    }

    function eventToCell(event) {
      const rect = canvas.getBoundingClientRect();
      const col = Math.floor((event.clientX - rect.left) / zoom);
      const row = Math.floor((event.clientY - rect.top) / zoom);
      if (row < 0 || col < 0 || row >= mapState.grid_size || col >= mapState.grid_size) {
        return null;
      }
      return [row, col];
    }

    function objectMap() {
      return new Map((mapState.object_instances || []).map(item => [item.id, item]));
    }

    function roomMap() {
      return new Map((mapState.room_instances || []).map(item => [item.id, item]));
    }

    function occupancyName(value) {
      const occupancy = mapState.layer_legends.occupancy || {};
      for (const [name, item] of Object.entries(occupancy)) {
        if (item.value === value) {
          return name;
        }
      }
      return value === 0 ? "free" : "obstacle";
    }

    function categoryLabel(group, category) {
      return mapState.layer_legends?.[group]?.[category]?.label || category || "unknown";
    }

    function describeCell(row, col) {
      const occupancy = occupancyName(mapState.layers.occupancy[row][col]);
      const parts = [`Cell (${row}, ${col})`, categoryLabel("occupancy", occupancy)];
      const roomId = mapState.layers.room[row][col];
      const objectId = mapState.layers.object_instance[row][col];
      const room = roomMap().get(roomId);
      const object = objectMap().get(objectId);
      if (room) {
        parts.push(`room=${room.name} (${categoryLabel("room_categories", room.category)})`);
      }
      if (object) {
        parts.push(`object=${object.name} (${categoryLabel("object_categories", object.category)})`);
      }
      return parts.join(" | ");
    }

    function inspectCell(row, col) {
      const objectId = mapState.layers.object_instance[row][col];
      const object = objectMap().get(objectId);
      if (object) {
        if (
          inspectedMapEntity?.type === "object" &&
          inspectedMapEntity.category === object.category
        ) {
          inspectedMapEntity = null;
          mapInspector.classList.remove("visible");
          mapInspector.innerHTML = "";
          drawGrid();
          setStatus(`Closed object highlight for ${categoryLabel("object_categories", object.category)}.`);
          return;
        }
        inspectedMapEntity = { type: "object", id: object.id, category: object.category };
        mapInspector.classList.add("visible");
        mapInspector.innerHTML = `<strong>Object type</strong><br>${categoryLabel("object_categories", object.category)}<br>clicked=${object.name} · id=${object.id}`;
        drawGrid();
        setStatus(describeCell(row, col));
        return;
      }
      const roomId = mapState.layers.room[row][col];
      const room = roomMap().get(roomId);
      if (room) {
        if (inspectedMapEntity?.type === "room" && inspectedMapEntity.id === room.id) {
          inspectedMapEntity = null;
          mapInspector.classList.remove("visible");
          mapInspector.innerHTML = "";
          drawGrid();
          setStatus(`Closed room highlight for ${room.name}.`);
          return;
        }
        inspectedMapEntity = { type: "room", id: room.id, category: room.category };
        mapInspector.classList.add("visible");
        mapInspector.innerHTML = `<strong>Room</strong><br>${room.name}<br>${categoryLabel("room_categories", room.category)} · id=${room.id}`;
        drawGrid();
        setStatus(describeCell(row, col));
        return;
      }
      inspectedMapEntity = null;
      mapInspector.classList.remove("visible");
      mapInspector.innerHTML = "";
      drawGrid();
      setStatus(describeCell(row, col));
    }

    function drawGrid() {
      const gridSize = mapState.grid_size;
      const occupancy = mapState.layers.occupancy;
      const rooms = mapState.layers.room;
      const objects = mapState.layers.object_instance;
      const roomById = roomMap();
      const objectById = objectMap();
      const legends = mapState.layer_legends;
      canvas.width = gridSize * zoom;
      canvas.height = gridSize * zoom;
      zoomValue.textContent = `${zoom}x`;

      for (let row = 0; row < gridSize; row += 1) {
        for (let col = 0; col < gridSize; col += 1) {
          const occName = occupancyName(occupancy[row][col]);
          ctx.fillStyle = legends.occupancy?.[occName]?.color || (occName === "free" ? "#ffffff" : "#111111");
          ctx.fillRect(col * zoom, row * zoom, zoom, zoom);

          const room = roomById.get(rooms[row][col]);
          if (showRooms && room && occName === "free") {
            const color = legends.room_categories?.[room.category]?.color;
            if (color) {
              ctx.fillStyle = hexToRgba(color, 0.45);
              ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
            }
          }

          const object = objectById.get(objects[row][col]);
          if (showObjects && object) {
            const color = legends.object_categories?.[object.category]?.color;
            if (color) {
              ctx.fillStyle = hexToRgba(color, 0.82);
              ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
            }
          }
        }
      }

      trajectory.forEach(([row, col], index) => {
        const ratio = trajectory.length <= 1 ? 0 : index / (trajectory.length - 1);
        ctx.fillStyle = `rgb(${Math.round(18 + 170 * ratio)}, ${Math.round(107 - 35 * ratio)}, ${Math.round(93 - 35 * ratio)})`;
        ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
      });
      if (trajectory.length) {
        const [row, col] = trajectory[0];
        ctx.fillStyle = "#2d6cdf";
        ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
      }

      const start = instructionStartCell();
      if (start) {
        const [row, col] = start;
        const centerX = (col + 0.5) * zoom;
        const centerY = (row + 0.5) * zoom;
        ctx.fillStyle = "rgba(45, 108, 223, 0.16)";
        ctx.strokeStyle = "#2d6cdf";
        ctx.lineWidth = Math.max(2, Math.floor(zoom / 2));
        ctx.beginPath();
        ctx.arc(centerX, centerY, 5 * zoom, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
        ctx.lineWidth = Math.max(1, Math.floor(zoom / 4));
        ctx.beginPath();
        ctx.arc(centerX, centerY, Math.max(1, 5 * zoom - 2), 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = "#2d6cdf";
        ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = Math.max(1, Math.floor(zoom / 3));
        ctx.strokeRect(col * zoom + 1, row * zoom + 1, Math.max(1, zoom - 2), Math.max(1, zoom - 2));
      }

      if (inspectedMapEntity) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = Math.max(1, Math.floor(zoom / 4));
        for (let row = 0; row < gridSize; row += 1) {
          for (let col = 0; col < gridSize; col += 1) {
            if (
              inspectedMapEntity.type === "room" &&
              rooms[row][col] !== inspectedMapEntity.id
            ) {
              continue;
            }
            if (
              inspectedMapEntity.type === "object" &&
              objectById.get(objects[row][col])?.category !== inspectedMapEntity.category
            ) {
              continue;
            }
            ctx.fillRect(col * zoom, row * zoom, zoom, zoom);
            if (zoom >= 3) {
              ctx.strokeRect(col * zoom + 0.5, row * zoom + 0.5, zoom - 1, zoom - 1);
            }
          }
        }
      }

      if (showGrid) {
        ctx.lineWidth = 1;
        for (let index = 0; index <= gridSize; index += 1) {
          ctx.strokeStyle = index % 16 === 0 ? "#aeb8aa" : "#e1e5de";
          ctx.beginPath();
          ctx.moveTo(0, index * zoom + 0.5);
          ctx.lineTo(canvas.width, index * zoom + 0.5);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(index * zoom + 0.5, 0);
          ctx.lineTo(index * zoom + 0.5, canvas.height);
          ctx.stroke();
        }
      }
    }

    function computePathLength() {
      let length = 0;
      for (let index = 1; index < trajectory.length; index += 1) {
        length += Math.hypot(
          trajectory[index][0] - trajectory[index - 1][0],
          trajectory[index][1] - trajectory[index - 1][1]
        );
      }
      return length;
    }

    function renderRouteStats() {
      document.getElementById("waypointValue").textContent = trajectory.length
        ? String(trajectory.length)
        : "-";
      document.getElementById("plValue").textContent = trajectory.length
        ? computePathLength().toFixed(3)
        : "-";
    }

    function isTraversableCell(row, col) {
      if (
        row < 0 ||
        row >= mapState.grid_size ||
        col < 0 ||
        col >= mapState.grid_size
      ) {
        return false;
      }
      const objectId = mapState.layers.object_instance[row][col];
      const objectItem = objectId === 0 ? null : objectMap().get(objectId);
      const traversableObjects = new Set(["doorframe", "doorway"]);
      const objectAllowsTraversal = (
        objectId === 0 ||
        traversableObjects.has(objectItem?.category)
      );
      const freeValue = mapState.layer_legends.occupancy?.free?.value ?? 0;
      return mapState.layers.occupancy[row][col] === freeValue && objectAllowsTraversal;
    }

    function canTraverseBetween(fromRow, fromCol, toRow, toCol) {
      if (!isTraversableCell(toRow, toCol)) {
        return false;
      }
      const dRow = toRow - fromRow;
      const dCol = toCol - fromCol;
      if (Math.abs(dRow) !== 1 || Math.abs(dCol) !== 1) {
        return true;
      }
      return (
        isTraversableCell(fromRow + dRow, fromCol) &&
        isTraversableCell(fromRow, fromCol + dCol)
      );
    }

    function heuristic(row, col, goalRow, goalCol) {
      const rowDelta = Math.abs(row - goalRow);
      const colDelta = Math.abs(col - goalCol);
      const diagonalSteps = Math.min(rowDelta, colDelta);
      const straightSteps = Math.max(rowDelta, colDelta) - diagonalSteps;
      return diagonalSteps * Math.SQRT2 + straightSteps;
    }

    function parseKey(key) {
      return key.split(",").map(Number);
    }

    function reconstructPath(cameFrom, endKey) {
      const path = [];
      let currentKey = endKey;
      while (currentKey) {
        path.push(parseKey(currentKey));
        currentKey = cameFrom.get(currentKey) || null;
      }
      path.reverse();
      return path;
    }

    function findAStarPath(startCell, goalCell) {
      const [startRow, startCol] = startCell;
      const [goalRow, goalCol] = goalCell;
      const startKey = cellKey(startRow, startCol);
      const goalKey = cellKey(goalRow, goalCol);
      if (startKey === goalKey) {
        return [startCell];
      }

      const openList = [startKey];
      const openKeys = new Set([startKey]);
      const cameFrom = new Map();
      const gScore = new Map([[startKey, 0]]);
      const fScore = new Map([[startKey, heuristic(startRow, startCol, goalRow, goalCol)]]);
      const directions = [
        [-1, 0, 1],
        [1, 0, 1],
        [0, -1, 1],
        [0, 1, 1],
        [-1, -1, Math.SQRT2],
        [-1, 1, Math.SQRT2],
        [1, -1, Math.SQRT2],
        [1, 1, Math.SQRT2],
      ];

      while (openList.length > 0) {
        let currentIndex = 0;
        let currentKey = openList[0];
        let currentScore = fScore.get(currentKey) ?? Number.POSITIVE_INFINITY;
        for (let index = 1; index < openList.length; index += 1) {
          const candidateKey = openList[index];
          const candidateScore = fScore.get(candidateKey) ?? Number.POSITIVE_INFINITY;
          if (candidateScore < currentScore) {
            currentIndex = index;
            currentKey = candidateKey;
            currentScore = candidateScore;
          }
        }
        openList.splice(currentIndex, 1);
        openKeys.delete(currentKey);

        if (currentKey === goalKey) {
          return reconstructPath(cameFrom, goalKey);
        }

        const [row, col] = parseKey(currentKey);
        const currentG = gScore.get(currentKey) ?? Number.POSITIVE_INFINITY;
        for (const [dRow, dCol, stepCost] of directions) {
          const nextRow = row + dRow;
          const nextCol = col + dCol;
          if (
            nextRow < 0 ||
            nextRow >= mapState.grid_size ||
            nextCol < 0 ||
            nextCol >= mapState.grid_size ||
            !canTraverseBetween(row, col, nextRow, nextCol)
          ) {
            continue;
          }
          const nextKey = cellKey(nextRow, nextCol);
          const tentativeG = currentG + stepCost;
          if (tentativeG >= (gScore.get(nextKey) ?? Number.POSITIVE_INFINITY)) {
            continue;
          }
          cameFrom.set(nextKey, currentKey);
          gScore.set(nextKey, tentativeG);
          fScore.set(nextKey, tentativeG + heuristic(nextRow, nextCol, goalRow, goalCol));
          if (!openKeys.has(nextKey)) {
            openList.push(nextKey);
            openKeys.add(nextKey);
          }
        }
      }

      return null;
    }

    function cloneRouteCells(cells) {
      return cells.map(point => [point[0], point[1]]);
    }

    function beginRouteSegment(kind) {
      activeRouteSegment = { kind, cells: [], usedAStar: false };
    }

    function recordRouteSegmentCells(kind, cells) {
      if (!Array.isArray(cells) || cells.length === 0) {
        return;
      }
      if (activeRouteSegment) {
        activeRouteSegment.cells.push(...cloneRouteCells(cells));
        if (kind === "astar") {
          activeRouteSegment.usedAStar = true;
        }
        return;
      }
      routeSegments.push({ kind, cells: cloneRouteCells(cells) });
    }

    function finishRouteSegment() {
      if (!activeRouteSegment) {
        return;
      }
      if (activeRouteSegment.cells.length > 0) {
        routeSegments.push({
          kind: activeRouteSegment.usedAStar ? "freehand_with_astar" : activeRouteSegment.kind,
          cells: cloneRouteCells(activeRouteSegment.cells),
        });
      }
      activeRouteSegment = null;
    }

    function cancelRouteSegment() {
      activeRouteSegment = null;
    }

    function appendRouteCells(cells, segmentKind = "manual") {
      const appended = [];
      for (const [row, col] of cells) {
        const last = trajectory[trajectory.length - 1];
        if (last && last[0] === row && last[1] === col) {
          continue;
        }
        const point = [row, col];
        trajectory.push(point);
        appended.push(point);
      }
      recordRouteSegmentCells(segmentKind, appended);
      lastSavedMetrics = null;
      renderRouteStats();
      drawGrid();
      return appended;
    }

    function eraseRouteCells(cells) {
      const remove = new Set(cells.map(cell => cellKey(cell[0], cell[1])));
      trajectory = trajectory.filter(cell => !remove.has(cellKey(cell[0], cell[1])));
      routeSegments = [];
      activeRouteSegment = null;
      lastSavedMetrics = null;
      renderRouteStats();
      drawGrid();
    }

    function undoLastRouteSegment() {
      if (routeSegments.length === 0) {
        setStatus("No route segment is available to undo.");
        return;
      }
      const segment = routeSegments.pop();
      if (!segment || segment.cells.length === 0) {
        setStatus("The latest route segment was empty.");
        return;
      }
      const tailStart = trajectory.length - segment.cells.length;
      const matchesTail = tailStart >= 0 && segment.cells.every((point, index) => {
        const routePoint = trajectory[tailStart + index];
        return routePoint && routePoint[0] === point[0] && routePoint[1] === point[1];
      });
      if (matchesTail) {
        trajectory = trajectory.slice(0, tailStart);
      } else {
        trajectory = trajectory.filter(([row, col]) => {
          return !segment.cells.some(point => point[0] === row && point[1] === col);
        });
      }
      lastSavedMetrics = null;
      renderRouteStats();
      drawGrid();
      setStatus(`Removed the previous ${segment.kind} route segment with ${segment.cells.length} cell(s).`);
    }

    function routeTailCell() {
      return trajectory.length > 0 ? trajectory[trajectory.length - 1] : null;
    }

    function appendAStarRouteSegment(startCell, goalCell) {
      const [startRow, startCol] = startCell;
      const [goalRow, goalCol] = goalCell;
      if (sameCell(startCell, goalCell)) {
        return true;
      }
      if (!isTraversableCell(startRow, startCol)) {
        setStatus(`Cannot connect from non-traversable cell (${startRow}, ${startCol}).`);
        return false;
      }
      if (!isTraversableCell(goalRow, goalCol)) {
        setStatus(`Target must be traversable: (${goalRow}, ${goalCol}).`);
        return false;
      }
      const path = findAStarPath(startCell, goalCell);
      if (!path) {
        setStatus(`No path found from (${startRow}, ${startCol}) to (${goalRow}, ${goalCol}).`);
        return false;
      }
      appendRouteCells(path, "astar");
      setStatus(`Connected (${startRow}, ${startCol}) to (${goalRow}, ${goalCol}) with ${path.length} cells.`);
      return true;
    }

    function connectRouteTailToCell(cell) {
      const tail = routeTailCell();
      if (!tail) {
        if (!isTraversableCell(cell[0], cell[1])) {
          setStatus(`Route start must be traversable: (${cell[0]}, ${cell[1]}).`);
          return false;
        }
        appendRouteCells([cell], "manual");
        setStatus(`Route start set at (${cell[0]}, ${cell[1]}).`);
        return true;
      }
      return appendAStarRouteSegment(tail, cell);
    }

    function handleRouteClick(cell) {
      connectRouteTailToCell(cell);
    }

    function beginRouteFreehand(cell) {
      const tail = routeTailCell();
      if (tail && !sameCell(tail, cell)) {
        if (!appendAStarRouteSegment(tail, cell)) {
          return false;
        }
      } else {
        if (!tail && !isTraversableCell(cell[0], cell[1])) {
          setStatus(`Route start must be traversable: (${cell[0]}, ${cell[1]}).`);
          return false;
        }
        appendRouteCells([cell], "manual");
      }
      return true;
    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Request failed.");
      }
      return await response.json();
    }

    async function openMap() {
      const payload = await postJson("/open", { map_id: mapSelect.value });
      applyState(payload.state);
      setStatus(payload.message);
    }

    async function fetchInstructionPayload(instructionId) {
      if (!instructionId) {
        return null;
      }
      const payload = await postJson("/instruction", {
        map_id: currentMapId,
        instruction_id: instructionId,
      });
      const target = instructions.find(item => item.instruction_id === instructionId);
      if (target) {
        target._payload = payload.instruction;
      }
      return payload.instruction;
    }

    async function fetchLatestSolution(instructionId) {
      if (!instructionId) {
        return null;
      }
      const payload = await postJson("/latest_solution", {
        map_id: currentMapId,
        instruction_id: instructionId,
      });
      return payload.solution;
    }

    function loadTrajectoryFromSolution(solution) {
      const savedTrajectory = Array.isArray(solution?.trajectory)
        ? solution.trajectory
        : [];
      trajectory = savedTrajectory.map(point => [point[0], point[1]]);
      routeSegments = [];
      activeRouteSegment = null;
      routeGestureStartCell = null;
      routeGestureDragged = false;
      lastSavedMetrics = solution?.metrics || null;
    }

    async function selectInstruction(instructionId) {
      activeInstructionId = instructionId;
      await fetchInstructionPayload(instructionId);
      const current = activeInstruction();
      if (current?.finished) {
        const solution = await fetchLatestSolution(instructionId);
        loadTrajectoryFromSolution(solution);
        setStatus(`Loaded saved trajectory for ${instructionId}.`);
      } else {
        resetTrajectoryToStart();
      }
      renderInstructionList();
      renderInstructionText();
      renderStartInfo();
      renderUnfinishButton();
      renderRouteStats();
      drawGrid();
    }

    async function saveSolution() {
      saveSolverId();
      if (!activeInstructionId) {
        throw new Error("Select an instruction first.");
      }
      if (!trajectory.length) {
        throw new Error("Draw a trajectory before saving.");
      }
      const payload = await postJson("/save_solution", {
        map_id: currentMapId,
        instruction_id: activeInstructionId,
        solver_id: solverIdInput.value.trim(),
        trajectory,
      });
      lastSavedMetrics = payload.record.metrics;
      const current = activeInstruction();
      const wasFinished = Boolean(current?.finished);
      if (current) {
        current.finished = true;
        current.solution_count = Number(current.solution_count || 0) + 1;
        current.latest_solution_path = payload.path;
      }
      const currentSummary = state.map_summaries.find(item => item.map_id === currentMapId);
      if (currentSummary && !wasFinished) {
        currentSummary.finished_count = Math.min(
          Number(currentSummary.instruction_count || instructions.length),
          Number(currentSummary.finished_count || 0) + 1
        );
      }
      renderMapSelect();
      renderMapInfo();
      const next = filteredInstructions().find(item => item.instruction_id !== activeInstructionId);
      if (completionFilter.value === "unfinished" && next) {
        await selectInstruction(next.instruction_id);
        setStatus(`${payload.message} ${payload.path} Opened next unfinished instruction.`);
        return;
      }
      renderInstructionList();
      renderUnfinishButton();
      renderRouteStats();
      setStatus(`${payload.message} ${payload.path}`);
    }

    async function markActiveInstructionUnfinished() {
      const current = activeInstruction();
      if (!current) {
        throw new Error("Select an instruction first.");
      }
      if (!current.finished) {
        setStatus("This instruction is already unfinished.");
        return;
      }
      const payload = await postJson("/unfinish_instruction", {
        map_id: currentMapId,
        instruction_id: current.instruction_id,
      });
      current.finished = false;
      current.solution_count = 0;
      current.latest_solution_path = null;
      const currentSummary = state.map_summaries.find(item => item.map_id === currentMapId);
      if (currentSummary) {
        currentSummary.finished_count = Math.max(
          0,
          Number(currentSummary.finished_count || 0) - 1
        );
      }
      completionFilter.value = "unfinished";
      resetTrajectoryToStart();
      renderMapSelect();
      renderMapInfo();
      renderInstructionList();
      renderUnfinishButton();
      renderRouteStats();
      drawGrid();
      setStatus(`${payload.message} Summary updated at ${payload.summary_path}.`);
    }

    function renderMapSelect() {
      mapSelect.innerHTML = "";
      for (const summary of state.map_summaries) {
        const option = document.createElement("option");
        option.value = summary.map_id;
        option.textContent = `${summary.map_id} (${summary.finished_count || 0}/${summary.instruction_count || 0})`;
        mapSelect.appendChild(option);
      }
      mapSelect.value = currentMapId;
    }

    function renderMapInfo() {
      const finished = instructions.filter(item => item.finished).length;
      document.getElementById("mapInfo").textContent = `${currentMapId} | grid=${mapState.grid_size} | finished=${finished}/${instructions.length}`;
      document.getElementById("paths").textContent = `${state.paths.instruction_directory}\\n${state.paths.method_root}\\n${state.paths.summary_json}`;
    }

    function filteredInstructions() {
      const difficulty = difficultyFilter.value;
      const completion = completionFilter.value;
      return instructions.filter(item => {
        const difficultyOk = !difficulty || item.difficulty_level === difficulty;
        const completionOk = (
          !completion ||
          (completion === "finished" && item.finished) ||
          (completion === "unfinished" && !item.finished)
        );
        return difficultyOk && completionOk;
      });
    }

    function selectFirstVisibleInstruction() {
      const visible = filteredInstructions();
      if (visible.some(item => item.instruction_id === activeInstructionId)) {
        return false;
      }
      activeInstructionId = visible[0]?.instruction_id || null;
      return true;
    }

    function renderInstructionList() {
      instructionList.innerHTML = "";
      const visibleInstructions = filteredInstructions();
      if (!visibleInstructions.length) {
        const empty = document.createElement("div");
        empty.className = "item small";
        empty.textContent = "No instructions match the current filters.";
        instructionList.appendChild(empty);
        return;
      }
      for (const item of visibleInstructions) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = [
          "instruction-item",
          item.finished ? "finished" : "unfinished",
          item.instruction_id === activeInstructionId ? "active" : "",
        ].filter(Boolean).join(" ");
        const status = item.finished ? `finished · ${item.solution_count || 1}` : "unfinished";
        button.innerHTML = `<strong>${item.instruction_id}</strong><br><span class="small">${item.difficulty_level || "unknown"} · ${item.object_count || "?"} objects · ${status}</span>`;
        button.addEventListener("click", () => {
          selectInstruction(item.instruction_id).catch(error => setStatus(error.message));
        });
        instructionList.appendChild(button);
      }
    }

    function renderUnfinishButton() {
      const current = activeInstruction();
      unfinishBtn.disabled = !current?.finished;
      unfinishBtn.style.opacity = current?.finished ? "1" : "0.45";
    }

    function renderInstructionText() {
      const instruction = activeInstructionPayload();
      instructionText.textContent = instruction?.instruction || "";
    }

    function renderStartInfo() {
      const start = instructionStartCell();
      const startInfo = document.getElementById("startInfo");
      if (!start) {
        startInfo.textContent = "Start: not available for this instruction.";
        return;
      }
      startInfo.textContent = `Start: row ${start[0]}, col ${start[1]}. The trajectory begins from this cell.`;
    }

    function applyState(nextState) {
      state = nextState;
      mapState = state.map_state;
      currentMapId = state.current_map_id;
      instructions = state.instructions || [];
      activeInstructionId = instructions[0]?.instruction_id || null;
      selectFirstVisibleInstruction();
      resetTrajectoryToStart();
      inspectedMapEntity = null;
      mapInspector.classList.remove("visible");
      mapInspector.innerHTML = "";
      renderMapSelect();
      renderMapInfo();
      renderInstructionList();
      renderUnfinishButton();
      if (activeInstructionId) {
        selectInstruction(activeInstructionId).catch(error => setStatus(error.message));
      } else {
        renderInstructionText();
        renderStartInfo();
        renderUnfinishButton();
        renderRouteStats();
        drawGrid();
      }
    }

    canvas.addEventListener("mousedown", event => {
      if (event.button !== 0) {
        return;
      }
      const cell = eventToCell(event);
      if (!cell) {
        return;
      }
      isPainting = true;
      routeGestureStartCell = cell;
      routeGestureDragged = false;
      cancelRouteSegment();
      lastCell = cell;
      if (eraseMode) {
        eraseRouteCells([cell]);
      }
    });
    canvas.addEventListener("mousemove", event => {
      const cell = eventToCell(event);
      if (!cell) {
        return;
      }
      if (!isPainting) {
        setStatus(describeCell(cell[0], cell[1]));
        return;
      }
      if (sameCell(lastCell, cell)) {
        return;
      }
      if (eraseMode) {
        eraseRouteCells(interpolateLine(lastCell, cell).slice(1));
        lastCell = cell;
        return;
      }
      if (!routeGestureDragged) {
        routeGestureDragged = true;
        beginRouteSegment("freehand");
        if (!beginRouteFreehand(routeGestureStartCell)) {
          cancelRouteSegment();
          isPainting = false;
          routeGestureStartCell = null;
          lastCell = null;
          return;
        }
      }
      appendRouteCells(interpolateLine(lastCell, cell).slice(1), "manual");
      lastCell = cell;
    });
    window.addEventListener("mouseup", () => {
      if (
        isPainting &&
        !eraseMode &&
        routeGestureStartCell &&
        !routeGestureDragged
      ) {
        handleRouteClick(routeGestureStartCell);
      }
      if (isPainting && !eraseMode && routeGestureDragged) {
        finishRouteSegment();
      } else {
        cancelRouteSegment();
      }
      isPainting = false;
      lastCell = null;
      routeGestureStartCell = null;
      routeGestureDragged = false;
    });
    canvas.addEventListener("contextmenu", event => {
      event.preventDefault();
      const cell = eventToCell(event);
      if (!cell) {
        return;
      }
      inspectCell(cell[0], cell[1]);
    });

    document.getElementById("openMapBtn").addEventListener("click", () => {
      openMap().catch(error => setStatus(error.message));
    });
    document.getElementById("saveBtn").addEventListener("click", () => {
      saveSolution().catch(error => setStatus(error.message));
    });
    unfinishBtn.addEventListener("click", () => {
      markActiveInstructionUnfinished().catch(error => setStatus(error.message));
    });
    document.getElementById("clearBtn").addEventListener("click", () => {
      resetTrajectoryToStart();
      lastSavedMetrics = null;
      renderRouteStats();
      drawGrid();
      setStatus("Trajectory reset to the start cell.");
    });
    document.getElementById("undoBtn").addEventListener("click", () => {
      undoLastRouteSegment();
    });
    zoomSlider.addEventListener("input", () => {
      zoom = Number(zoomSlider.value);
      drawGrid();
    });
    document.getElementById("gridToggle").addEventListener("change", event => {
      showGrid = event.target.checked;
      drawGrid();
    });
    document.getElementById("roomToggle").addEventListener("change", event => {
      showRooms = event.target.checked;
      drawGrid();
    });
    document.getElementById("objectToggle").addEventListener("change", event => {
      showObjects = event.target.checked;
      drawGrid();
    });
    document.getElementById("eraseToggle").addEventListener("change", event => {
      eraseMode = event.target.checked;
      setStatus(eraseMode ? "Erase mode on." : "Draw mode on.");
    });
    solverIdInput.addEventListener("change", saveSolverId);
    difficultyFilter.addEventListener("change", () => {
      const changed = selectFirstVisibleInstruction();
      renderInstructionList();
      if (changed && activeInstructionId) {
        selectInstruction(activeInstructionId).catch(error => setStatus(error.message));
        return;
      }
      renderInstructionText();
      renderStartInfo();
      renderRouteStats();
      drawGrid();
    });
    completionFilter.addEventListener("change", () => {
      const changed = selectFirstVisibleInstruction();
      renderInstructionList();
      if (changed && activeInstructionId) {
        selectInstruction(activeInstructionId).catch(error => setStatus(error.message));
        return;
      }
      renderInstructionText();
      renderStartInfo();
      renderRouteStats();
      drawGrid();
    });

    loadSolverId();
    renderMapSelect();
    renderMapInfo();
    renderInstructionList();
    if (activeInstructionId) {
      selectInstruction(activeInstructionId).catch(error => setStatus(error.message));
    } else {
      drawGrid();
      renderRouteStats();
    }
  </script>
</body>
</html>
"""


class HumanSolverHandler(BaseHTTPRequestHandler):
    state: dict[str, object] = {}

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        html = HTML_PAGE.replace(
            "__INITIAL_STATE__",
            compact_json_for_html(self.state),
        )
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/open":
                map_id = payload.get("map_id")
                if not isinstance(map_id, str) or not map_id.strip():
                    raise ValueError("map_id must be a non-empty string.")
                instruction_set = str(
                    self.state.get("instruction_set", DEFAULT_INSTRUCTION_SET)
                )
                self.state = build_client_state(map_id, instruction_set)
                self._send_json(
                    {
                        "message": f"Opened map '{self.state['current_map_id']}'.",
                        "state": self.state,
                    }
                )
                return

            if self.path == "/instruction":
                map_id = payload.get("map_id")
                instruction_id = payload.get("instruction_id")
                if not isinstance(map_id, str) or not map_id.strip():
                    raise ValueError("map_id must be a non-empty string.")
                if not isinstance(instruction_id, str) or not instruction_id.strip():
                    raise ValueError("instruction_id must be a non-empty string.")
                instruction, _path = load_instruction_by_id(map_id, instruction_id)
                self._send_json({"instruction": instruction})
                return

            if self.path == "/latest_solution":
                map_id = payload.get("map_id")
                instruction_id = payload.get("instruction_id")
                if not isinstance(map_id, str) or not map_id.strip():
                    raise ValueError("map_id must be a non-empty string.")
                if not isinstance(instruction_id, str) or not instruction_id.strip():
                    raise ValueError("instruction_id must be a non-empty string.")
                solution, path = load_latest_solution(map_id, instruction_id)
                self._send_json(
                    {
                        "solution": solution,
                        "path": path.relative_to(REPO_ROOT).as_posix(),
                    }
                )
                return

            if self.path == "/save_solution":
                result = save_solution(payload)
                self._send_json(result)
                return

            if self.path == "/unfinish_instruction":
                result = unfinish_instruction(payload)
                self._send_json(result)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found.")
        except Exception as exc:  # pragma: no cover - runtime validation.
            self._send_text(str(exc), HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SemPathBench human-solver method interface."
    )
    parser.add_argument("--map-id", default=None, help="Map id such as procthor/002_train.")
    parser.add_argument(
        "--set",
        dest="instruction_set",
        choices=INSTRUCTION_SET_CHOICES,
        default=DEFAULT_INSTRUCTION_SET,
        help="Instruction set to show. Use 'all' to show every map.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    HumanSolverHandler.state = build_client_state(args.map_id, args.instruction_set)
    server = ThreadingHTTPServer((args.host, args.port), HumanSolverHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving SemPathBench human solver at {url}")
    print(f"Saving results under {METHOD_ROOT.relative_to(REPO_ROOT)}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
