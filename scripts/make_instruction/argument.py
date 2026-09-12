#!/usr/bin/env python3
"""Persisted UI arguments for the instruction annotator."""

from __future__ import annotations

from pathlib import Path


ARGUMENTS = {
    "zoom": 2,
}

_PATH = Path(__file__)


def load_arguments() -> dict[str, int]:
    zoom = ARGUMENTS.get("zoom", 6)
    if not isinstance(zoom, int):
        zoom = 6
    return {"zoom": min(max(zoom, 1), 12)}


def save_arguments(arguments: dict[str, object]) -> dict[str, int]:
    current = load_arguments()
    zoom = arguments.get("zoom", current["zoom"])
    if not isinstance(zoom, int) or zoom < 1 or zoom > 12:
        raise ValueError("zoom must be an integer between 1 and 12.")
    current["zoom"] = zoom

    source = _PATH.read_text(encoding="utf-8")
    start = source.index("ARGUMENTS = {")
    end = source.index("\n}\n", start) + 3
    replacement = f'ARGUMENTS = {{\n    "zoom": {zoom},\n}}\n'
    _PATH.write_text(source[:start] + replacement + source[end:], encoding="utf-8")
    ARGUMENTS.clear()
    ARGUMENTS.update(current)
    return current
