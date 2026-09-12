"""Configuration for the local OSG-LLM SemPathBench adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OSGLLMConfig:
    translation_mode: str = "heuristic"
    grounding_mode: str = "heuristic"
    planner: str = "product"
    object_reach_radius: int = 20
    max_expansions: int | None = 250_000
    max_planning_seconds: float | None = 30.0
    disable_fallback: bool = False
    llm_cache_only: bool = False
    use_llm_heuristic: bool = False
    model: str | None = None
    llm_cache_root: Path | None = None
    overwrite_llm_cache: bool = False
    verbose: bool = False
