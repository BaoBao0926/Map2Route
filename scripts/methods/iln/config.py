"""Configuration constants for the ILN SemPathBench adapter."""

from __future__ import annotations

from scripts.make_instruction.make_instruction import REPO_ROOT


METHOD_NAME = "ILN"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / "ILN"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_UNKNOWN_PASSAGE_COST = 10.0
GROUNDING_MODES = ("llm", "heuristic", "auto")
EVENT_MODES = ("empty", "llm")
HISTORY_MODES = ("empty", "file")
PASSAGE_EXTRACTION_MODES = ("boundary", "doorway", "auto")
