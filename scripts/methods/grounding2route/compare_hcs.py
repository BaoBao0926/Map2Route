"""Compare paired HCS values for Grounding2Route full and ToolCall outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FULL_ROOT = REPO_ROOT / "resources" / "methods" / "grounding2route" / "main_result"
DEFAULT_TOOL_ROOT = REPO_ROOT / "resources" / "methods" / "grounding2route" / "ablation" / "01_grounding_strategy" / "tool_call"


def _records(root: Path) -> dict[tuple[str, str], dict[str, object]]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(root.rglob("instruction_*.json")):
        if path.name.endswith(".steps.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A live run may be observed while one record is being written.
            continue
        if not isinstance(payload, Mapping):
            continue
        metrics = payload.get("metrics")
        hcs = metrics.get("HCS") if isinstance(metrics, Mapping) else None
        map_id = payload.get("map_id")
        instruction_id = payload.get("instruction_id")
        if (
            not isinstance(map_id, str)
            or not isinstance(instruction_id, str)
            or not isinstance(hcs, (int, float))
            or isinstance(hcs, bool)
        ):
            continue
        records[(map_id, instruction_id)] = {
            "hcs": float(hcs),
            "difficulty": str(payload.get("difficulty_level") or "unknown").lower(),
        }
    return records


def compare_hcs(full_root: Path, tool_root: Path) -> dict[str, object]:
    full = _records(full_root)
    tool = _records(tool_root)
    keys = sorted(full.keys() & tool.keys())

    rows = [
        {
            "key": key,
            "difficulty": tool[key]["difficulty"],
            "full": float(full[key]["hcs"]),
            "tool": float(tool[key]["hcs"]),
        }
        for key in keys
    ]

    def aggregate(items: list[dict[str, object]]) -> dict[str, object]:
        if not items:
            return {
                "paired_episodes": 0,
                "grounding2route_full_hcs": None,
                "tool_call_hcs": None,
                "tool_minus_full": None,
                "tool_win_tie_loss": {"win": 0, "tie": 0, "loss": 0},
            }
        full_values = [float(row["full"]) for row in items]
        tool_values = [float(row["tool"]) for row in items]
        deltas = [tool_value - full_value for tool_value, full_value in zip(tool_values, full_values)]
        tolerance = 1e-12
        return {
            "paired_episodes": len(items),
            "grounding2route_full_hcs": mean(full_values),
            "tool_call_hcs": mean(tool_values),
            "tool_minus_full": mean(deltas),
            "tool_win_tie_loss": {
                "win": sum(delta > tolerance for delta in deltas),
                "tie": sum(abs(delta) <= tolerance for delta in deltas),
                "loss": sum(delta < -tolerance for delta in deltas),
            },
        }

    return {
        "grounding2route_full_root": str(full_root),
        "tool_call_root": str(tool_root),
        "available_records": {"grounding2route_full": len(full), "tool_call": len(tool)},
        "overall": aggregate(rows),
        "easy": aggregate([row for row in rows if row["difficulty"] == "easy"]),
        "hard": aggregate([row for row in rows if row["difficulty"] == "hard"]),
    }


def _print_table(report: Mapping[str, object]) -> None:
    available = report["available_records"]
    assert isinstance(available, Mapping)
    print(
        "available: "
        f"Grounding2Route-full={available['grounding2route_full']} "
        f"ToolCall={available['tool_call']}"
    )
    print("split\tpaired\tGrounding2Route-full\tToolCall\tToolCall-full\tToolCall W/T/L")
    for split in ("overall", "easy", "hard"):
        row = report[split]
        assert isinstance(row, Mapping)
        outcomes = row["tool_win_tie_loss"]
        assert isinstance(outcomes, Mapping)
        full_hcs = row["grounding2route_full_hcs"]
        tool_hcs = row["tool_call_hcs"]
        delta = row["tool_minus_full"]
        values = (
            "-\t-\t-"
            if full_hcs is None or tool_hcs is None or delta is None
            else f"{float(full_hcs):.6f}\t{float(tool_hcs):.6f}\t{float(delta):+.6f}"
        )
        print(
            f"{split}\t{row['paired_episodes']}\t{values}\t"
            f"{outcomes['win']}/{outcomes['tie']}/{outcomes['loss']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Grounding2Route full and ToolCall HCS on their shared episodes."
    )
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--tool-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_hcs(args.full_root, args.tool_root)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_table(report)


if __name__ == "__main__":
    main()
