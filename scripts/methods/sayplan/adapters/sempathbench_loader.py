"""Load SemPathBench episodes for the SayPlan pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from scripts.make_instruction.make_instruction import load_map_state, normalize_map_key
from scripts.methods.util.grid_astar import (
    TraversableGrid,
    build_traversable_grid,
    is_traversable,
    nearest_traversable_cell,
)
from scripts.methods.sayplan.adapters.map_to_scene_graph import build_scene_graph
from scripts.methods.sayplan.graph.scene_graph import SceneGraph


@dataclass
class SayPlanEpisode:
    map_id: str
    scene_id: str
    instruction_id: str
    instruction_file: Path
    instruction: dict[str, object]
    instruction_text: str
    map_state: dict[str, object]
    traversable: TraversableGrid
    start_cell: tuple[int, int] | None
    full_graph: SceneGraph
    failure_reason: str | None = None


def scene_id_from_instruction(
    instruction: Mapping[str, object],
    map_id: str,
) -> str:
    scene_id = instruction.get("map_id")
    if isinstance(scene_id, str) and scene_id.strip():
        return scene_id.strip()
    return map_id


def load_episode(
    *,
    instruction_file: Path,
    map_id: str,
    instruction_id: str,
    instruction: dict[str, object],
) -> SayPlanEpisode:
    map_state = load_map_state(map_id)
    inference_instruction = inference_instruction_view(instruction)
    normalized_map_id = normalize_map_key(map_id)
    if map_state.get("map_key") != normalized_map_id:
        traversable = build_traversable_grid(map_state)
        return SayPlanEpisode(
            map_id=map_id,
            scene_id=scene_id_from_instruction(instruction, map_id),
            instruction_id=instruction_id,
            instruction_file=instruction_file,
            instruction=inference_instruction,
            instruction_text=str(instruction.get("instruction", "")),
            map_state=map_state,
            traversable=traversable,
            start_cell=None,
            full_graph=SceneGraph(),
            failure_reason="map_load_mismatch",
        )
    traversable = build_traversable_grid(map_state)
    start_cell = start_cell_for_instruction(
        map_state, traversable, inference_instruction
    )
    graph = build_scene_graph(map_state, traversable, start_cell)
    return SayPlanEpisode(
        map_id=map_id,
        scene_id=scene_id_from_instruction(instruction, map_id),
        instruction_id=instruction_id,
        instruction_file=instruction_file,
        instruction=inference_instruction,
        instruction_text=str(instruction.get("instruction", "")),
        map_state=map_state,
        traversable=traversable,
        start_cell=start_cell,
        full_graph=graph,
        failure_reason=None if start_cell is not None else "no_start_cell",
    )


def start_cell_for_instruction(
    map_state: Mapping[str, object],
    traversable: TraversableGrid,
    instruction: Mapping[str, object],
) -> tuple[int, int] | None:
    start_pose = instruction.get("start_pose")
    if isinstance(start_pose, Mapping):
        row = start_pose.get("row")
        col = start_pose.get("col")
        if isinstance(row, int) and isinstance(col, int):
            return _nearest_if_needed(traversable, (row, col))
    return None


def inference_instruction_view(
    instruction: Mapping[str, object],
) -> dict[str, object]:
    """Project a benchmark record onto permitted SayPlan inference fields."""
    result: dict[str, object] = {
        "instruction": str(instruction.get("instruction", "")),
    }
    start_pose = instruction.get("start_pose")
    if isinstance(start_pose, Mapping):
        row = start_pose.get("row")
        col = start_pose.get("col")
        if (
            isinstance(row, int)
            and not isinstance(row, bool)
            and isinstance(col, int)
            and not isinstance(col, bool)
        ):
            result["start_pose"] = {"row": row, "col": col}
    return result


def _nearest_if_needed(
    traversable: TraversableGrid,
    cell: tuple[int, int],
) -> tuple[int, int] | None:
    if is_traversable(traversable, *cell):
        return cell
    return nearest_traversable_cell(traversable, cell)
