"""Typed records used by the ILN adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


Cell = tuple[int, int]


@dataclass(frozen=True)
class Area:
    id: int
    key: str
    category: str
    name: str
    centroid: tuple[float, float]
    medoid: Cell
    cell_count: int

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "key": self.key,
            "category": self.category,
            "name": self.name,
            "centroid": [self.centroid[0], self.centroid[1]],
            "medoid": [self.medoid[0], self.medoid[1]],
            "cell_count": self.cell_count,
        }


@dataclass(frozen=True)
class Passage:
    id: str
    rooms: tuple[int, int]
    anchor: Cell
    width: int
    component_size: int

    def connects(self, room_id: int) -> bool:
        return room_id in self.rooms

    def other_room(self, room_id: int) -> int | None:
        if self.rooms[0] == room_id:
            return self.rooms[1]
        if self.rooms[1] == room_id:
            return self.rooms[0]
        return None

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "rooms": [self.rooms[0], self.rooms[1]],
            "anchor": [self.anchor[0], self.anchor[1]],
            "width": self.width,
            "component_size": self.component_size,
        }


@dataclass
class AreaPassageGraph:
    areas: dict[int, Area]
    passages: dict[str, Passage]
    room_passages: dict[int, list[str]]
    start_cell: Cell | None
    start_area: int | None
    metadata: dict[str, object] = field(default_factory=dict)

    def area_key(self, room_id: int | None) -> str | None:
        if room_id is None:
            return None
        area = self.areas.get(room_id)
        return area.key if area else f"room_{room_id}"

    def to_json(self) -> dict[str, object]:
        return {
            "area_count": len(self.areas),
            "passage_count": len(self.passages),
            "start_cell": [self.start_cell[0], self.start_cell[1]]
            if self.start_cell
            else None,
            "start_area": self.area_key(self.start_area),
            "areas": [area.to_json() for area in sorted(self.areas.values(), key=lambda item: item.id)],
            "passages": [
                passage.to_json()
                for passage in sorted(self.passages.values(), key=lambda item: item.id)
            ],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GroundedStage:
    stage_id: str
    target_type: str
    area_id: int
    object_id: int | None = None
    source: str = "heuristic"

    @property
    def area_key(self) -> str:
        return f"room_{self.area_id}"

    @property
    def object_key(self) -> str | None:
        return f"object_{self.object_id}" if self.object_id is not None else None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "stage_id": self.stage_id,
            "target_type": self.target_type,
            "area_id": self.area_key,
            "source": self.source,
        }
        if self.object_id is not None:
            payload["object_id"] = self.object_key
        return payload


@dataclass
class GroundingResult:
    mode: str
    stages: list[GroundedStage]
    adapter_graph_constraints: dict[str, object]
    notes: list[str] = field(default_factory=list)
    raw_response: object | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "stages": [stage.to_json() for stage in self.stages],
            "adapter_graph_constraints": self.adapter_graph_constraints,
            "notes": self.notes,
            "raw_response": self.raw_response,
        }


@dataclass
class StageGoal:
    stage: GroundedStage
    endpoint: Cell | None
    candidate_count: int
    status: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "stage": self.stage.to_json(),
            "endpoint": [self.endpoint[0], self.endpoint[1]] if self.endpoint else None,
            "candidate_count": self.candidate_count,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class PassageCostResult:
    mode: str
    navigation_task: list[str]
    door_costs: dict[str, float]
    raw_response: object | None
    diagnostics: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "navigation_task": self.navigation_task,
            "door_costs": {key: {"cost": value} for key, value in sorted(self.door_costs.items())},
            "raw_response": self.raw_response,
            "diagnostics": self.diagnostics,
        }


@dataclass
class RouteResult:
    status: str
    area_sequence: list[int]
    passage_sequence: list[str]
    cost: float
    diagnostics: list[str] = field(default_factory=list)

    def to_json(self, graph: AreaPassageGraph) -> dict[str, object]:
        return {
            "status": self.status,
            "area_sequence": [graph.area_key(area_id) for area_id in self.area_sequence],
            "passage_sequence": self.passage_sequence,
            "cost": self.cost,
            "diagnostics": self.diagnostics,
        }


def room_key_to_id(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text.startswith("room_"):
        text = text[5:]
    try:
        return int(text)
    except ValueError:
        return None


def object_key_to_id(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text.startswith("object_"):
        text = text[7:]
    try:
        return int(text)
    except ValueError:
        return None


def list_from_mapping(value: Mapping[str, object], key: str) -> list[object]:
    raw = value.get(key)
    return raw if isinstance(raw, list) else []

