"""LocalGraph: Safe graph wrapper restricting nodes to local neighborhood only.

This wrapper provides distributed nodes with a restricted view of the graph,
ensuring they can only access information about their immediate neighbors,
not the entire graph structure. This is essential for true distributed systems
where nodes must not have global knowledge.
"""

from typing import List, Optional
from src.graph.graph_manager import GraphManager


class DistributionViolation(Exception):
    """Raised when a node attempts to access global graph state."""

    def __init__(self, message: str):
        """Initialize with error message."""
        self.message = message
        super().__init__(self.message)


class LocalGraph:
    """
    Graph wrapper restricting node to local neighborhood only.

    Provides safe accessors for distributed nodes:
    - neighbors(node_id): Get neighbors of any node
    - get_edge_weight(u, v): Get edge weight between nodes
    - degree(node_id): Get degree of a node

    Blocks global inspection (raises DistributionViolation):
    - vertices(): Cannot see all vertices
    - all_edges(): Cannot see full edge set
    - any other global graph inspection

    This ensures nodes cannot violate the distributed algorithm contract
    which requires local knowledge only.
    """

    def __init__(self, full_graph: GraphManager, node_id: int):
        """
        Initialize LocalGraph wrapper.

        Args:
            full_graph: The underlying GraphManager (read-only)
            node_id: The ID of the node this wrapper belongs to
        """
        self._graph = full_graph
        self._node_id = node_id

    def neighbors(self, node_id: Optional[int] = None) -> List[int]:
        """
        Get neighbors of a node.

        Args:
            node_id: The node ID to get neighbors for (defaults to self)

        Returns:
            List of node IDs that are neighbors
        """
        if node_id is None:
            node_id = self._node_id
        return list(self._graph.neighbors(node_id))

    def get_edge_weight(self, u: int, v: int) -> float:
        """
        Get weight of edge between two nodes.

        Args:
            u: Source node ID
            v: Target node ID

        Returns:
            Weight of the edge
        """
        return self._graph.get_edge_weight(u, v)

    def degree(self, node_id: Optional[int] = None) -> int:
        """
        Get degree of a node.

        Args:
            node_id: The node ID to get degree for (defaults to self)

        Returns:
            Number of neighbors
        """
        if node_id is None:
            node_id = self._node_id
        return len(self.neighbors(node_id))

    def vertices(self) -> List[int]:
        """
        BLOCKED: Node cannot access all vertices.

        This violates the distributed algorithm contract. Nodes must not
        have knowledge of the entire graph.

        Raises:
            DistributionViolation: Always raised - this operation not allowed
        """
        raise DistributionViolation(
            "Node cannot access all vertices. "
            "Use neighbors() for local neighborhood access only."
        )

    def all_edges(self) -> List[tuple]:
        """
        BLOCKED: Node cannot access full edge set.

        This violates the distributed algorithm contract. Nodes must not
        have knowledge of the entire edge set.

        Raises:
            DistributionViolation: Always raised - this operation not allowed
        """
        raise DistributionViolation(
            "Node cannot inspect full edge set. "
            "This violates local knowledge constraint."
        )

    def calculate_matching_weight(self, matching: dict) -> float:
        """
        Calculate weight of a matching (safe - uses full graph but for read-only calculation).

        Args:
            matching: Dict mapping node -> matched_node

        Returns:
            Total weight of matching
        """
        return self._graph.calculate_matching_weight(matching)
