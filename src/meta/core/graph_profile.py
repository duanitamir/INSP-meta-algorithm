"""Stable structural descriptions of graphs used in offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from src.graph.graph_manager import GraphManager


@dataclass(frozen=True)
class GraphProfile:
    """A reproducible structural envelope for one evaluation graph."""

    vertex_count: int
    edge_count: int
    density: float
    component_count: int
    max_degree: int
    mean_degree: float

    @classmethod
    def from_graph(cls, graph: GraphManager) -> "GraphProfile":
        """Create a profile using only stable graph structure measurements."""
        vertex_count = graph.num_vertices()
        edge_count = graph.num_edges()
        degrees = [graph.degree(node_id) for node_id in graph.vertices()]
        possible_edges = vertex_count * (vertex_count - 1) / 2

        return cls(
            vertex_count=vertex_count,
            edge_count=edge_count,
            density=edge_count / possible_edges if possible_edges else 0.0,
            component_count=len(graph.get_connected_components()),
            max_degree=max(degrees, default=0),
            mean_degree=sum(degrees) / vertex_count if vertex_count else 0.0,
        )
