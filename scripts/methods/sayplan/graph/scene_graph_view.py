"""Collapsed/expanded LLM-visible views over a full scene graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.methods.sayplan.graph.scene_graph import SceneGraph


@dataclass
class SceneGraphMemory:
    expanded_nodes: list[str] = field(default_factory=list)
    contracted_nodes: list[str] = field(default_factory=list)
    commands: list[dict[str, object]] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {
            "expanded_nodes": self.expanded_nodes,
            "contracted_nodes": self.contracted_nodes,
            "commands": self.commands,
        }


class SceneGraphView:
    def __init__(self, full_graph: SceneGraph, visible_nodes: set[str]) -> None:
        self.full_graph = full_graph
        self.visible_nodes = set(visible_nodes)
        self.memory = SceneGraphMemory()

    @classmethod
    def collapse(cls, full_graph: SceneGraph) -> "SceneGraphView":
        visible = {"scene_0", "agent_0"}
        visible.update(
            node_id
            for node_id in full_graph.children("scene_0")
            if full_graph.get_node(node_id).type == "room"
        )
        return cls(full_graph, {node_id for node_id in visible if full_graph.has_node(node_id)})

    def expand(self, node_id: str) -> None:
        if not self.full_graph.has_node(node_id):
            raise KeyError(node_id)
        self.visible_nodes.add(node_id)
        self.visible_nodes.update(self.full_graph.children(node_id))
        if node_id not in self.memory.expanded_nodes:
            self.memory.expanded_nodes.append(node_id)
        self.memory.commands.append(
            {"command_name": "expand_node", "node_name": node_id}
        )

    def contract(self, node_id: str) -> None:
        if not self.full_graph.has_node(node_id):
            raise KeyError(node_id)
        for child in self.full_graph.children(node_id):
            self._hide_subtree(child)
        self.visible_nodes.add(node_id)
        if node_id not in self.memory.contracted_nodes:
            self.memory.contracted_nodes.append(node_id)
        self.memory.commands.append(
            {"command_name": "contract_node", "node_name": node_id}
        )

    def _hide_subtree(self, node_id: str) -> None:
        self.visible_nodes.discard(node_id)
        for child in self.full_graph.children(node_id):
            self._hide_subtree(child)

    def visible_json(self) -> dict[str, object]:
        return self._json_for_nodes(self.visible_nodes)

    def task_subgraph_json(self) -> dict[str, object]:
        return self.visible_json()

    def _json_for_nodes(self, node_ids: set[str]) -> dict[str, object]:
        nodes = []
        for node_id in sorted(node_ids):
            node = self.full_graph.get_node(node_id)
            visible_children = [
                child for child in self.full_graph.children(node_id) if child in node_ids
            ]
            nodes.append(node.prompt_json(visible_children=visible_children))
        edges = [
            edge.prompt_json()
            for edge in self.full_graph.edges
            if edge.source in node_ids and edge.target in node_ids
        ]
        return {"nodes": nodes, "edges": edges}

    def memory_json(self) -> dict[str, object]:
        return self.memory.to_json()
