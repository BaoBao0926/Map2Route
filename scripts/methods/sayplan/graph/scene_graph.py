"""Scene graph data structures for the SayPlan adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from scripts.methods.sayplan.config import (
    MAX_PROMPT_ATTRIBUTE_CHARS,
    MAX_PROMPT_ATTRIBUTES,
)


@dataclass
class GraphNode:
    id: str
    type: str
    category: str | None = None
    name: str = ""
    position: tuple[int, int] | None = None
    room_id: str | None = None
    attributes: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def prompt_json(self, *, visible_children: list[str] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "type": self.type,
            "name": self.name or self.id,
        }
        if self.category:
            payload["category"] = self.category
        if self.position is not None:
            payload["position"] = [self.position[0], self.position[1]]
        if self.room_id:
            payload["room_id"] = self.room_id
        child_count = self.metadata.get("child_count")
        if isinstance(child_count, int):
            payload["child_count"] = child_count
        if visible_children is not None:
            payload["visible_children"] = visible_children
        attrs = [
            item[:MAX_PROMPT_ATTRIBUTE_CHARS]
            for item in self.attributes[:MAX_PROMPT_ATTRIBUTES]
            if item
        ]
        if attrs:
            payload["attributes"] = attrs
        return payload


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    metadata: dict[str, object] = field(default_factory=dict)

    def prompt_json(self) -> list[str]:
        return [self.source, self.target, self.relation]


class SceneGraph:
    """Full immutable-ish graph; views decide what is visible to the LLM."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._children: dict[str, list[str]] = {}
        self._parents: dict[str, list[str]] = {}

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node
        self._children.setdefault(node.id, [])
        self._parents.setdefault(node.id, [])

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.edges.append(GraphEdge(source, target, relation, dict(metadata or {})))
        if relation in {"contains", "located_at", "approach"}:
            if target not in self._children.setdefault(source, []):
                self._children[source].append(target)
            if source not in self._parents.setdefault(target, []):
                self._parents[target].append(source)

    def get_node(self, node_id: str) -> GraphNode:
        return self.nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def children(self, node_id: str) -> list[str]:
        return list(self._children.get(node_id, []))

    def parents(self, node_id: str) -> list[str]:
        return list(self._parents.get(node_id, []))

    def ancestors(self, node_id: str) -> list[str]:
        result: list[str] = []
        stack = self.parents(node_id)
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.append(current)
            stack.extend(self.parents(current))
        return result

    def containment_edges(self) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.relation == "contains"]

