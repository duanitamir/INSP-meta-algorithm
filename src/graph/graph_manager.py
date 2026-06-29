from typing import Dict, List, FrozenSet, Any
import networkx as nx


class GraphManager:
    """Manages graph structure and queries using NetworkX."""

    def __init__(self):
        self._graph: nx.Graph = nx.Graph()

    @classmethod
    def create_empty_graph(cls) -> "GraphManager":
        """Create an empty graph."""
        return cls()

    @classmethod
    def create_from_edges(
        cls,
        vertices: List[int],
        edges: List[tuple[int, int, float]],
    ) -> "GraphManager":
        """
        Create graph from vertex and edge lists.

        Args:
            vertices: List of vertex IDs
            edges: List of (u, v, weight) tuples

        Returns:
            GraphManager instance
        """
        manager = cls()
        for vertex in vertices:
            manager._graph.add_node(vertex)
        for u, v, weight in edges:
            manager._graph.add_edge(u, v, weight=weight)
        return manager

    def add_vertex(self, vertex_id: int, properties: Dict[str, Any] | None = None) -> None:
        """Add a vertex to the graph."""
        self._graph.add_node(vertex_id, **(properties or {}))

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """Add an edge between two vertices."""
        if u not in self._graph or v not in self._graph:
            raise ValueError("Vertices must exist before adding edge")
        self._graph.add_edge(u, v, weight=weight)

    def num_vertices(self) -> int:
        """Get number of vertices."""
        return self._graph.number_of_nodes()

    def num_edges(self) -> int:
        """Get number of edges."""
        return self._graph.number_of_edges()

    def vertices(self) -> FrozenSet[int]:
        """Get all vertices as frozenset."""
        return frozenset(self._graph.nodes())

    def neighbors(
        self, vertex_id: int, state_store=None, filter_active: bool = False
    ) -> FrozenSet[int]:
        """Get neighbors of a vertex.

        Args:
            vertex_id: The vertex to get neighbors for
            state_store: Optional StateStore to filter by node state
            filter_active: If True and state_store provided, return only unmatched neighbors

        Returns:
            FrozenSet of neighbor node IDs. If state_store and filter_active are provided,
            returns only neighbors where is_matched() == False (i.e., active/unmatched nodes).
        """
        if vertex_id not in self._graph:
            raise ValueError(f"Vertex {vertex_id} not in graph")

        all_neighbors = frozenset(self._graph.neighbors(vertex_id))

        # If no filtering requested, return all neighbors
        if state_store is None or not filter_active:
            return all_neighbors

        # Filter to only unmatched (active) neighbors
        active_neighbors = frozenset(
            neighbor
            for neighbor in all_neighbors
            if not state_store.get_node_state(neighbor).is_matched()
        )
        return active_neighbors

    def degree(self, vertex_id: int) -> int:
        """Get degree of a vertex."""
        if vertex_id not in self._graph:
            raise ValueError(f"Vertex {vertex_id} not in graph")
        return self._graph.degree(vertex_id)

    def max_degree(self) -> int:
        """Get maximum degree in graph."""
        if self.num_vertices() == 0:
            return 0
        return max(degree for _, degree in self._graph.degree())

    def has_edge(self, u: int, v: int) -> bool:
        """Check if edge exists between u and v."""
        return self._graph.has_edge(u, v)

    def get_edge_weight(self, u: int, v: int) -> float:
        """Get weight of edge between u and v."""
        if not self.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) does not exist")
        return self._graph[u][v].get("weight", 1.0)

    def is_connected(self) -> bool:
        """Check if graph is connected."""
        return nx.is_connected(self._graph)

    def get_connected_components(self) -> List[FrozenSet[int]]:
        """Get connected components."""
        return [frozenset(comp) for comp in nx.connected_components(self._graph)]

    def get_subgraph(self, nodes: FrozenSet[int] | set) -> "GraphManager":
        """Create a subgraph containing only specified nodes.

        Args:
            nodes: Set or FrozenSet of node IDs to include in subgraph

        Returns:
            New GraphManager containing only specified nodes and edges between them

        Raises:
            ValueError: If nodes set is empty or contains invalid node IDs
        """
        if not nodes:
            raise ValueError("Cannot create subgraph with empty node set")

        invalid_nodes = set(nodes) - set(self._graph.nodes())
        if invalid_nodes:
            raise ValueError(f"Invalid node IDs in subgraph: {invalid_nodes}")

        nx_subgraph = self._graph.subgraph(nodes)
        result = GraphManager.create_empty_graph()
        result._graph = nx_subgraph.copy()
        return result

    def calculate_matching_weight(self, matching: Dict[int, int]) -> float:
        """Calculate total weight of a matching.

        Args:
            matching: Dict mapping node_id -> matched_partner

        Returns:
            float: Sum of edge weights (counting each edge once)
        """
        total = 0.0
        visited = set()

        for u, v in matching.items():
            if u not in visited:
                weight = self.get_edge_weight(u, v)
                if weight is not None:
                    total += weight
                visited.add(u)
                visited.add(v)

        return total

    @staticmethod
    def normalize_edge(u: int, v: int) -> tuple:
        """Normalize edge representation to canonical form (smaller, larger).

        Args:
            u: First node ID
            v: Second node ID

        Returns:
            tuple: (min(u, v), max(u, v)) for consistent edge representation
        """
        return (u, v) if u < v else (v, u)
