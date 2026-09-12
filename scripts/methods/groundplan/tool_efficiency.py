"""Aggregate ToolCall efficiency statistics from saved prediction records."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Mapping


FIELDS = (
    "llm_invocations",
    "llm_input_tokens",
    "llm_output_tokens",
    "tool_call_requests",
    "semantic_api_calls",
    "query_retry_limit_blocks",
    "verifier_repair_rounds",
    "grounding_wall_seconds",
)


def write_tool_call_efficiency_report(output_root: Path) -> Path:
    rows: list[dict[str, object]] = []
    for path in sorted(output_root.rglob("instruction_*.json")):
        if path.name.endswith(".steps.json"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        groundplan = record.get("groundplan") if isinstance(record, Mapping) else None
        stats = groundplan.get("grounding_efficiency") if isinstance(groundplan, Mapping) else None
        if not isinstance(stats, Mapping):
            continue
        metrics = record.get("metrics") if isinstance(record, Mapping) else None
        rows.append({
            "difficulty": str(record.get("difficulty_level") or "unknown").lower(),
            "HCS": metrics.get("HCS") if isinstance(metrics, Mapping) else None,
            **{field: stats.get(field) for field in FIELDS},
            "verification_success": stats.get("verification_success"),
            "tool_budget_exhausted": stats.get("tool_budget_exhausted"),
            "planning_success": stats.get("planning_success"),
        })

    groups = {"overall": rows}
    for difficulty in ("easy", "hard"):
        groups[difficulty] = [row for row in rows if row["difficulty"] == difficulty]
    payload = {
        "version": 1,
        "grounder": "ToolCallGrounder",
        "groups": {name: _aggregate(items) for name, items in groups.items()},
    }
    path = output_root / "tool_call_efficiency.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"episodes": len(rows)}
    for field in ("HCS", *FIELDS):
        values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)]
        summary[field] = {
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
        }
    for field in ("verification_success", "tool_budget_exhausted", "planning_success"):
        values = [bool(row[field]) for row in rows if isinstance(row.get(field), bool)]
        summary[field + "_rate"] = mean(values) if values else None
    return summary
