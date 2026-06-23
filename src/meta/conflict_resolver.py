"""Resolver for merging algorithm matchings and handling conflicts."""

from typing import Dict
from src.graph.graph_manager import GraphManager


class ConflictResolver:
    """Merges multiple algorithm matchings into a single symmetric matching.

    Handles conflicts by keeping highest-weight edges and enforcing symmetry.
    """

    def resolve(
        self,
        greedy_matching: Dict[int, int],
        itai_matching: Dict[int, int],
        luby_matching: Dict[int, int],
        graph: GraphManager,
    ) -> Dict[int, int]:
        """Merge three matchings into single symmetric matching.

        Args:
            greedy_matching: Dict mapping node_id -> matched_partner from Greedy
            itai_matching: Dict mapping node_id -> matched_partner from Itai-Israeli
            luby_matching: Dict mapping node_id -> matched_partner from Luby
            graph: GraphManager with vertices and edges

        Returns:
            Dict[int, int]: Symmetric matching (u -> v and v -> u both present)

        Raises:
            ValueError: If graph is None or has no vertices
        """
        if graph is None or len(graph.vertices()) == 0:
            return {}

        # Collect all edges from all matchings with their weights
        edge_weights: Dict[tuple, float] = {}

        for matching in [greedy_matching, itai_matching, luby_matching]:
            for node_u, node_v in matching.items():
                edge = GraphManager.normalize_edge(node_u, node_v)

                if edge not in edge_weights:
                    weight = graph.get_edge_weight(node_u, node_v)
                    if weight is not None:
                        edge_weights[edge] = weight

        # Build result with conflict resolution (highest weight wins)
        result: Dict[int, int] = {}
        matched_nodes: set = set()

        # Sort edges by weight descending (greedy selection by weight)
        sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)

        for (u, v), weight in sorted_edges:
            # Conflict resolution: skip edge if either endpoint already matched
            # This ensures no node appears in multiple edges (valid matching)
            if u in matched_nodes or v in matched_nodes:
                continue

            # Add symmetric pair to result
            result[u] = v
            result[v] = u
            matched_nodes.add(u)
            matched_nodes.add(v)

        return result

    def name(self) -> str:
        """Return resolver name."""
        return "ConflictResolver"
