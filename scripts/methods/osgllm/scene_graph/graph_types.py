"""Typed scene-graph objects used by the OSG-LLM adapter."""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.methods.util.grid_astar import TraversableGrid


Cell = tuple[int, int]


@dataclass(frozen=True)
class AttributeRegion:
    node_id: str
    kind: str
    instance_id: int | None
    category: str
    name: str
    cells: tuple[Cell, ...]
    center: tuple[float, float]
    parent_id: str | None = None
    source_cells: tuple[Cell, ...] = ()
    attributes: tuple[str, ...] = ()

    @property
    def ap(self) -> str:
        if self.kind == "object":
            return f"reach({self.node_id})"
        return f"enter({self.node_id})"


@dataclass(frozen=True)
class SceneGraph:
    grid_size: int
    traversable: TraversableGrid
    base_cells: tuple[Cell, ...]
    regions: dict[str, AttributeRegion]
    room_adjacency: dict[str, tuple[str, ...]]
    object_room: dict[str, str]
    warnings: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def by_kind(self, kind: str) -> tuple[AttributeRegion, ...]:
        return tuple(
            sorted(
                (region for region in self.regions.values() if region.kind == kind),
                key=lambda region: (region.instance_id is None, region.instance_id or 0, region.node_id),
            )
        )

    def ap_inventory(self) -> dict[str, str]:
        return {region.ap: region.node_id for region in self.regions.values()}

    def summary(self) -> dict[str, object]:
        return {
            "floor_count": len(self.by_kind("floor")),
            "room_count": len(self.by_kind("room")),
            "object_count": len(self.by_kind("object")),
            "traversable_cell_count": len(self.base_cells),
            "attribute_region_count": len(self.regions),
            "room_adjacency_count": sum(len(value) for value in self.room_adjacency.values()) // 2,
            "warnings": list(self.warnings),
            **self.metadata,
        }

