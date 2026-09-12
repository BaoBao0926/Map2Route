"""Task Progression Semantic Map construction."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.methods.limp.utils.geometry import Cell


@dataclass(frozen=True)
class TPSM:
    stage_index: int
    current_formula: str
    goal_regions: dict[str, set[Cell]]
    forbidden_regions: dict[str, set[Cell]]
    enabled_transition: str

    def summary(self) -> dict[str, object]:
        return {
            "stage_index": self.stage_index,
            "current_formula": self.current_formula,
            "goal_region_sizes": {
                key: len(value) for key, value in sorted(self.goal_regions.items())
            },
            "forbidden_region_sizes": {
                key: len(value) for key, value in sorted(self.forbidden_regions.items())
            },
            "enabled_transition": self.enabled_transition,
        }

