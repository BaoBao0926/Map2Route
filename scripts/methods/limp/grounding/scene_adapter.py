"""Convert SemPathBench map_state objects into LIMP grounding candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from scripts.methods.limp.utils.geometry import Cell, bbox, centroid, normalize_name
from scripts.methods.util.map_objects import (
    object_cells_by_id as rich_object_cells_by_id,
    object_room_id as rich_object_room_id,
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str
    instance_id: int
    category: str
    name: str
    cells: tuple[Cell, ...]
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    room_id: int | None = None
    room_category: str | None = None
    attributes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CandidateRegistry:
    candidates: dict[str, Candidate]
    by_kind: dict[str, tuple[str, ...]]
    by_category: dict[str, tuple[str, ...]]
    object_room: dict[int, int]

    def get(self, candidate_id: str) -> Candidate | None:
        return self.candidates.get(candidate_id)


def _metadata_by_id(items: object) -> dict[int, Mapping[str, object]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return {}
    result: dict[int, Mapping[str, object]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("id")
        if isinstance(raw_id, int) and not isinstance(raw_id, bool):
            result[raw_id] = item
    return result


def _layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    layer = layers.get(name)
    if not isinstance(layer, Sequence) or isinstance(layer, (str, bytes)):
        return []
    return layer  # type: ignore[return-value]


def _scan_cells(layer: Sequence[Sequence[object]]) -> dict[int, list[Cell]]:
    cells_by_id: dict[int, list[Cell]] = defaultdict(list)
    for row, values in enumerate(layer):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for col, raw_value in enumerate(values):
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                cells_by_id[value].append((row, col))
    return cells_by_id


def _attributes(item: Mapping[str, object]) -> tuple[str, ...]:
    attrs = item.get("attributes")
    if not isinstance(attrs, Sequence) or isinstance(attrs, (str, bytes)):
        return ()
    return tuple(str(attr) for attr in attrs)


def build_candidate_registry(map_state: Mapping[str, object]) -> CandidateRegistry:
    room_layer = _layer(map_state, "room")
    room_cells = _scan_cells(room_layer)
    object_cells = rich_object_cells_by_id(map_state)
    room_meta = _metadata_by_id(map_state.get("room_instances"))
    object_meta = _metadata_by_id(map_state.get("object_instances"))

    room_category_by_id: dict[int, str] = {}
    candidates: dict[str, Candidate] = {}
    object_room: dict[int, int] = {}

    for room_id, cells in room_cells.items():
        meta = room_meta.get(room_id, {})
        category = normalize_name(meta.get("category", f"room_{room_id}"))
        name = str(meta.get("name") or f"{category}_{room_id}")
        room_category_by_id[room_id] = category
        candidate_id = f"room_{room_id}"
        candidates[candidate_id] = Candidate(
            candidate_id=candidate_id,
            kind="room",
            instance_id=room_id,
            category=category,
            name=name,
            cells=tuple(cells),
            centroid=centroid(cells),
            bbox=bbox(cells),
            attributes=_attributes(meta),
        )

    for object_id, cells in object_cells.items():
        meta = object_meta.get(object_id, {})
        category = normalize_name(meta.get("category", f"object_{object_id}"))
        name = str(meta.get("name") or f"{category}_{object_id}")
        room_id = rich_object_room_id(map_state, object_id)
        if room_id is not None:
            object_room[object_id] = room_id
        candidate_id = f"object_{object_id}"
        candidates[candidate_id] = Candidate(
            candidate_id=candidate_id,
            kind="object",
            instance_id=object_id,
            category=category,
            name=name,
            cells=tuple(cells),
            centroid=centroid(cells),
            bbox=bbox(cells),
            room_id=room_id,
            room_category=room_category_by_id.get(room_id) if room_id is not None else None,
            attributes=_attributes(meta),
        )

    by_kind_temp: dict[str, list[str]] = defaultdict(list)
    by_category_temp: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates.values():
        by_kind_temp[candidate.kind].append(candidate.candidate_id)
        by_category_temp[candidate.category].append(candidate.candidate_id)

    return CandidateRegistry(
        candidates=candidates,
        by_kind={key: tuple(sorted(value)) for key, value in by_kind_temp.items()},
        by_category={key: tuple(sorted(value)) for key, value in by_category_temp.items()},
        object_room=object_room,
    )


def add_start_context_candidates(
    registry: CandidateRegistry,
    start: Cell | None,
) -> None:
    """Add instruction-relative aliases derived only from the start pose.

    LIMP instructions frequently refer to ``you``, ``the robot``, or the
    current/starting room.  These are not semantic-map categories, but their
    referents are fully determined by the allowed start pose and room layer.
    """

    if start is None:
        return

    start_id = "context_start"
    if start_id not in registry.candidates:
        registry.candidates[start_id] = Candidate(
            candidate_id=start_id,
            kind="virtual",
            instance_id=-1,
            category="starting_point",
            name="robot starting point",
            cells=(start,),
            centroid=(float(start[0]), float(start[1])),
            bbox=(start[0], start[1], start[0], start[1]),
            attributes=("start_pose", "virtual_grounding"),
        )
        registry.by_kind["virtual"] = tuple(
            sorted((*registry.by_kind.get("virtual", ()), start_id))
        )

    for alias in (
        "robot",
        "you",
        "start",
        "starting_point",
        "start_point",
        "current_position",
        "starting_position",
    ):
        registry.by_category[alias] = (start_id,)

    containing_room = next(
        (
            candidate.candidate_id
            for candidate in registry.candidates.values()
            if candidate.kind == "room" and start in candidate.cells
        ),
        None,
    )
    if containing_room is not None:
        for alias in ("current_room", "starting_room", "start_room", "initial_room"):
            registry.by_category[alias] = (containing_room,)


def registry_summary(registry: CandidateRegistry) -> dict[str, object]:
    object_counts: Counter[str] = Counter()
    room_counts: Counter[str] = Counter()
    for candidate in registry.candidates.values():
        if candidate.kind == "object":
            object_counts[candidate.category] += 1
        elif candidate.kind == "room":
            room_counts[candidate.category] += 1
    return {
        "candidate_count": len(registry.candidates),
        "object_count": len(registry.by_kind.get("object", ())),
        "room_count": len(registry.by_kind.get("room", ())),
        "object_category_counts": dict(sorted(object_counts.items())),
        "room_category_counts": dict(sorted(room_counts.items())),
    }
