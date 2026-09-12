"""Configuration defaults for the SemPathBench LIMP adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.make_instruction.make_instruction import REPO_ROOT


METHOD_NAME = "LIMP-Nav-Prototype"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "baselines" / "LIMP"
DEFAULT_NEAR_RADIUS = 24
DEFAULT_AVOID_RADIUS = 18
DEFAULT_MAX_PROGRESS_STEPS = 16
DEFAULT_MAX_GOAL_CANDIDATES = 0


@dataclass(frozen=True)
class LimpConfig:
    model: str
    llm_cache_root: Path
    method_variant: str = "extended"
    overwrite_llm_cache: bool = False
    translation_mode: str = "llm"
    grounding_mode: str = "extended"
    translation_context: str = "none"
    automaton_backend: str = "ltlf"
    allow_residual_fallback: bool = False
    near_radius: int = DEFAULT_NEAR_RADIUS
    avoid_radius: int = DEFAULT_AVOID_RADIUS
    max_progress_steps: int = DEFAULT_MAX_PROGRESS_STEPS
    max_goal_candidates: int = DEFAULT_MAX_GOAL_CANDIDATES
    strict_soft: bool = False
    verbose: bool = False
