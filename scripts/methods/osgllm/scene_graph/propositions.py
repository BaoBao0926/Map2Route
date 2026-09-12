"""Atomic proposition inventory and cell labelling."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.methods.osgllm.scene_graph.graph_types import Cell, SceneGraph


LabelMap = dict[Cell, frozenset[str]]


@dataclass(frozen=True)
class PropositionModel:
    label_map: LabelMap
    ap_to_cells: dict[str, tuple[Cell, ...]]
    inventory: dict[str, str]

    def labels_for_cell(self, cell: Cell) -> frozenset[str]:
        return self.label_map.get(cell, frozenset())


def build_proposition_model(
    graph: SceneGraph,
    *,
    used_aps: set[str] | None = None,
) -> PropositionModel:
    inventory = graph.ap_inventory()
    ap_to_cells: dict[str, tuple[Cell, ...]] = {}
    label_sets: dict[Cell, set[str]] = {}
    for ap_name, node_id in inventory.items():
        if used_aps is not None and ap_name not in used_aps:
            continue
        region = graph.regions.get(node_id)
        if region is None:
            continue
        ap_to_cells[ap_name] = region.cells
        for cell in region.cells:
            label_sets.setdefault(cell, set()).add(ap_name)
    label_map = {cell: frozenset(labels) for cell, labels in label_sets.items()}
    return PropositionModel(
        label_map=label_map,
        ap_to_cells=ap_to_cells,
        inventory=inventory,
    )

