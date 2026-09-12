#!/usr/bin/env python3
"""Build a compact cross-method summary from method-level summary.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS_ROOT = REPO_ROOT / "resources" / "methods"
DEFAULT_OUTPUT_PATH = METHODS_ROOT / "summary.json"
SECTION_ORDER = ("overall", "easy", "hard")


def summary_sources() -> list[Path]:
	baseline_summaries = (METHODS_ROOT / "baselines").glob("*/summary.json")
	groundplan_summary = METHODS_ROOT / "groundplan" / "main_result" / "summary.json"
	return sorted([*baseline_summaries, *([groundplan_summary] if groundplan_summary.exists() else [])])


def load_json_object(path: Path) -> dict[str, Any]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		raise ValueError(f"Expected JSON object in {path}")
	return payload


def extract_block(summary: dict[str, Any], section: str) -> dict[str, Any]:
	source = summary.get(section)

	if not isinstance(source, dict):
		raise ValueError(f"Missing section '{section}' in summary for {summary.get('method')}")

	return {
		"record_count": source.get("record_count"),
		"metric": source.get("metric"),
	}


def build_total_summary() -> dict[str, Any]:
	methods: dict[str, Any] = {}
	for path in summary_sources():
		summary = load_json_object(path)
		method_name = summary.get("method")
		if not isinstance(method_name, str) or not method_name.strip():
			raise ValueError(f"Missing method name in {path}")

		methods[method_name] = {
			"summary_path": path.relative_to(REPO_ROOT).as_posix(),
			**{section: extract_block(summary, section) for section in SECTION_ORDER},
		}

	return {
		"version": 1,
		"sections": list(SECTION_ORDER),
		"method_count": len(methods),
		"methods": methods,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Aggregate record_count and metric from all method summary.json files "
			"into resources/methods/summary.json."
		)
	)
	parser.add_argument(
		"--output-path",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="Path where the aggregated summary JSON will be written.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	payload = build_total_summary()
	args.output_path.parent.mkdir(parents=True, exist_ok=True)
	args.output_path.write_text(
		json.dumps(payload, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
	print(args.output_path.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
	main()
