from typing import Dict, Optional, Set
from src.graph.graph_manager import GraphManager
from src.state.state_store import StateStore


class GraphVisualizer:
    """Visualizes graph structure and state."""

    def __init__(self, graph: GraphManager):
        self.graph = graph

    def render_to_ascii(
        self,
        state_store: Optional[StateStore] = None,
        matching: Optional[Dict[int, int]] = None,
    ) -> str:
        """
        Render graph to ASCII format.

        Args:
            state_store: Optional state to show node info
            matching: Optional matching to highlight edges
        """
        lines = []
        lines.append(f"Graph: {self.graph.num_vertices()} vertices, {self.graph.num_edges()} edges")
        lines.append("")

        matched_edges: Set[tuple[int, int]] = set()
        if matching:
            for u, v in matching.items():
                if u < v:
                    matched_edges.add((u, v))
                else:
                    matched_edges.add((v, u))

        lines.append("Vertex Summary:")
        for vertex_id in sorted(self.graph.vertices()):
            degree = self.graph.degree(vertex_id)
            matched = ""
            if state_store:
                state = state_store.get_node_state(vertex_id)
                if state.is_matched():
                    matched_to = state.get_matched_to()
                    matched = f" → matched to {matched_to}"
            lines.append(f"  {vertex_id}: degree={degree}{matched}")

        lines.append("")
        lines.append("Edge Summary:")
        for u, v in sorted(self.graph._graph.edges()):
            weight = self.graph.get_edge_weight(u, v)
            status = ""
            if (u, v) in matched_edges or (v, u) in matched_edges:
                status = " [MATCHED]"
            lines.append(f"  {u} -- {v}: weight={weight:.2f}{status}")

        return "\n".join(lines)

    def render_matching_to_ascii(self, matching: Dict[int, int]) -> str:
        """Render matching results."""
        lines = []
        lines.append("MATCHING ANALYSIS")
        lines.append("=" * 50)

        total_vertices = self.graph.num_vertices()
        matched_vertices = len(matching)
        matching_size = matched_vertices // 2 if matched_vertices > 0 else 0

        lines.append(f"Total vertices: {total_vertices}")
        lines.append(f"Matched vertices: {matched_vertices}")
        lines.append(f"Unmatched vertices: {total_vertices - matched_vertices}")
        lines.append(f"Matching size (edges): {matching_size}")

        if matching_size > 0:
            total_weight = sum(
                self.graph.get_edge_weight(u, v)
                for u, v in matching.items()
                if u < v
            )
            lines.append(f"Total weight: {total_weight:.2f}")
            lines.append(f"Average edge weight: {total_weight / matching_size:.2f}")

        lines.append("")
        lines.append("Matched edges:")
        for u, v in sorted(matching.items()):
            if u < v:
                weight = self.graph.get_edge_weight(u, v)
                lines.append(f"  {u} -- {v}: weight={weight:.2f}")

        unmatched = set(self.graph.vertices()) - set(matching.keys())
        if unmatched:
            lines.append("")
            lines.append(f"Unmatched vertices: {sorted(unmatched)}")

        return "\n".join(lines)

    def summary(self) -> str:
        """Get graph summary."""
        lines = []
        lines.append("=" * 50)
        lines.append("GRAPH SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Vertices: {self.graph.num_vertices()}")
        lines.append(f"Edges: {self.graph.num_edges()}")
        if self.graph.num_vertices() > 0:
            lines.append(f"Max degree: {self.graph.max_degree()}")
            avg_degree = 2 * self.graph.num_edges() / self.graph.num_vertices()
            lines.append(f"Average degree: {avg_degree:.2f}")
        lines.append(f"Connected: {self.graph.is_connected()}")
        return "\n".join(lines)
