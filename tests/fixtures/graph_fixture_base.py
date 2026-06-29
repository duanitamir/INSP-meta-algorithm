"""Abstract base class for graph fixtures with optimal matching computation.

All graph fixtures should inherit from this to provide consistent interface
for computing optimal matching weights and quality metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import networkx as nx
from src.graph.graph_manager import GraphManager


class GraphFixtureBase(ABC):
    """Abstract base class for test graph fixtures.

    Provides:
    - Automatic optimal matching computation via NetworkX
    - Quality ratio calculation
    - Consistent interface across all fixtures
    - Helper methods for graph analysis
    """

    def __init__(self):
        """Initialize fixture (subclass calls after building graph)."""
        self.graph: GraphManager = GraphManager.create_empty_graph()
        self._optimal_matching: Optional[Dict[int, int]] = None
        self._optimal_weight: Optional[float] = None
        self._computed_optimal = False

    @abstractmethod
    def build(self, **kwargs) -> "GraphFixtureBase":
        """Build the graph structure.

        Subclasses must implement this to create vertices and edges.
        Should return self for chaining.

        Args:
            **kwargs: Parameters specific to graph type

        Returns:
            self for method chaining
        """
        pass

    def _compute_optimal(self) -> None:
        """Compute optimal matching using NetworkX Blossom algorithm."""
        if self._computed_optimal:
            return

        try:
            optimal_nx = nx.max_weight_matching(self.graph._graph, weight='weight')
            # Convert from set of tuples to dict format
            self._optimal_matching = {}
            for u, v in optimal_nx:
                self._optimal_matching[u] = v
                self._optimal_matching[v] = u
            self._optimal_weight = self.graph.calculate_matching_weight(self._optimal_matching)
        except Exception as e:
            # If graph is empty or invalid, set empty
            self._optimal_matching = {}
            self._optimal_weight = 0.0

        self._computed_optimal = True

    def get_optimal_matching(self) -> Dict[int, int]:
        """Get optimal matching (computed on demand).

        Returns:
            Dict mapping node -> matched_node for optimal matching
        """
        if not self._computed_optimal:
            self._compute_optimal()
        return self._optimal_matching or {}

    def get_optimal_weight(self) -> float:
        """Get optimal matching weight.

        Returns:
            float: Sum of edge weights in optimal matching
        """
        if not self._computed_optimal:
            self._compute_optimal()
        return self._optimal_weight or 0.0

    def quality_ratio(self, matching: Dict[int, int]) -> float:
        """Calculate how close a matching is to optimal.

        Args:
            matching: Matching to evaluate

        Returns:
            float: Ratio of our_weight / optimal_weight (0-1, higher is better)

        Raises:
            ValueError: If optimal_weight is 0 (can't divide)
        """
        optimal = self.get_optimal_weight()
        if optimal == 0:
            raise ValueError("Cannot compute quality ratio: optimal weight is 0")

        our_weight = self.graph.calculate_matching_weight(matching)
        return our_weight / optimal

    def get_gap_percent(self, matching: Dict[int, int]) -> float:
        """Calculate percentage gap from optimal.

        Args:
            matching: Matching to evaluate

        Returns:
            float: Gap as percentage (0-100, lower is better)
        """
        ratio = self.quality_ratio(matching)
        return (1 - ratio) * 100

    def num_vertices(self) -> int:
        """Get number of vertices."""
        return self.graph.num_vertices()

    def num_edges(self) -> int:
        """Get number of edges."""
        return self.graph.num_edges()

    def summary(self) -> str:
        """Get human-readable summary of graph.

        Returns:
            str: Summary with vertices, edges, optimal weight
        """
        return (
            f"{self.__class__.__name__}:\n"
            f"  Vertices: {self.num_vertices()}\n"
            f"  Edges: {self.num_edges()}\n"
            f"  Optimal Weight: {self.get_optimal_weight():.1f}\n"
            f"  Optimal Matching: {len(self.get_optimal_matching()) // 2} pairs"
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(vertices={self.num_vertices()}, "
            f"edges={self.num_edges()}, optimal={self.get_optimal_weight():.1f})"
        )


class PathGraphFixture(GraphFixtureBase):
    """Path graph: 1-2-3-...-N."""

    def build(self, num_vertices: int = 4, edge_weight: float = 1.0) -> "PathGraphFixture":
        """Build path graph.

        Args:
            num_vertices: Number of vertices in path
            edge_weight: Weight of each edge

        Returns:
            self
        """
        for i in range(1, num_vertices + 1):
            self.graph.add_vertex(i)

        for i in range(1, num_vertices):
            self.graph.add_edge(i, i + 1, edge_weight)

        return self


class GridGraphFixture(GraphFixtureBase):
    """Grid graph: rectangular lattice."""

    def build(
        self, rows: int = 3, cols: int = 3, edge_weight: float = 1.0
    ) -> "GridGraphFixture":
        """Build grid graph.

        Args:
            rows: Number of rows
            cols: Number of columns
            edge_weight: Weight of each edge

        Returns:
            self
        """
        # Add vertices
        for i in range(rows):
            for j in range(cols):
                vertex_id = i * cols + j + 1
                self.graph.add_vertex(vertex_id)

        # Add horizontal edges
        for i in range(rows):
            for j in range(cols - 1):
                u = i * cols + j + 1
                v = i * cols + j + 2
                self.graph.add_edge(u, v, edge_weight)

        # Add vertical edges
        for i in range(rows - 1):
            for j in range(cols):
                u = i * cols + j + 1
                v = (i + 1) * cols + j + 1
                self.graph.add_edge(u, v, edge_weight)

        return self


class CompleteGraphFixture(GraphFixtureBase):
    """Complete graph K_N: every vertex connected to every other."""

    def build(self, num_vertices: int = 5, edge_weight: float = 1.0) -> "CompleteGraphFixture":
        """Build complete graph.

        Args:
            num_vertices: Number of vertices
            edge_weight: Weight of each edge

        Returns:
            self
        """
        for i in range(1, num_vertices + 1):
            self.graph.add_vertex(i)

        for i in range(1, num_vertices + 1):
            for j in range(i + 1, num_vertices + 1):
                self.graph.add_edge(i, j, edge_weight)

        return self


class StarGraphFixture(GraphFixtureBase):
    """Star graph: central hub connected to N leaves."""

    def build(
        self, num_leaves: int = 5, hub_weight: float = 1.0, leaf_weight: float = None
    ) -> "StarGraphFixture":
        """Build star graph.

        Args:
            num_leaves: Number of leaf nodes
            hub_weight: Weight of edges from hub to leaves
            leaf_weight: Weight of edges between leaves (default None = no edges)

        Returns:
            self
        """
        # Hub is vertex 0
        self.graph.add_vertex(0)

        # Add leaves
        for i in range(1, num_leaves + 1):
            self.graph.add_vertex(i)

        # Connect hub to all leaves
        for i in range(1, num_leaves + 1):
            self.graph.add_edge(0, i, hub_weight)

        # Optionally add leaf-to-leaf edges
        if leaf_weight is not None:
            for i in range(1, num_leaves + 1):
                for j in range(i + 1, num_leaves + 1):
                    self.graph.add_edge(i, j, leaf_weight)

        return self


class BipartiteGraphFixture(GraphFixtureBase):
    """Bipartite graph: two groups with edges only between groups."""

    def build(
        self, left_size: int = 3, right_size: int = 3, edge_weight: float = 1.0
    ) -> "BipartiteGraphFixture":
        """Build bipartite graph.

        Args:
            left_size: Number of vertices in left partition
            right_size: Number of vertices in right partition
            edge_weight: Weight of each edge

        Returns:
            self
        """
        # Left vertices: 1 to left_size
        for i in range(1, left_size + 1):
            self.graph.add_vertex(i)

        # Right vertices: left_size+1 to left_size+right_size
        for i in range(left_size + 1, left_size + right_size + 1):
            self.graph.add_vertex(i)

        # Connect all left to all right (complete bipartite)
        for i in range(1, left_size + 1):
            for j in range(left_size + 1, left_size + right_size + 1):
                self.graph.add_edge(i, j, edge_weight)

        return self


class CustomGraphFixture(GraphFixtureBase):
    """Custom graph: build from vertices and edges lists."""

    def build(
        self, vertices: list, edges: list
    ) -> "CustomGraphFixture":
        """Build custom graph.

        Args:
            vertices: List of vertex IDs
            edges: List of (u, v, weight) tuples

        Returns:
            self
        """
        for v in vertices:
            self.graph.add_vertex(v)

        for u, v, w in edges:
            self.graph.add_edge(u, v, w)

        return self


class ClusteredGraphFixture(GraphFixtureBase):
    """Clustered graph: multiple dense clusters with sparse inter-cluster edges."""

    def build(
        self,
        num_clusters: int = 3,
        cluster_size: int = 5,
        intra_weight: float = 10.0,
        inter_weight: float = 1.0,
    ) -> "ClusteredGraphFixture":
        """Build clustered graph.

        Args:
            num_clusters: Number of clusters
            cluster_size: Vertices per cluster
            intra_weight: Weight of edges within cluster
            inter_weight: Weight of edges between clusters

        Returns:
            self
        """
        vertex_id = 1
        cluster_starts = []

        # Create clusters
        for cluster_idx in range(num_clusters):
            cluster_start = vertex_id
            cluster_starts.append(cluster_start)

            # Add vertices in cluster
            for i in range(cluster_size):
                self.graph.add_vertex(vertex_id)
                vertex_id += 1

            # Add intra-cluster edges (fully connected)
            for i in range(cluster_start, cluster_start + cluster_size):
                for j in range(i + 1, cluster_start + cluster_size):
                    self.graph.add_edge(i, j, intra_weight)

        # Add inter-cluster edges (connect cluster heads)
        for i in range(len(cluster_starts) - 1):
            u = cluster_starts[i]
            v = cluster_starts[i + 1]
            self.graph.add_edge(u, v, inter_weight)

        return self


class ScaleFreeGraphFixture(GraphFixtureBase):
    """Scale-free graph: power-law degree distribution (Barabási-Albert model)."""

    def build(self, num_vertices: int = 20, attachment_edges: int = 2) -> "ScaleFreeGraphFixture":
        """Build scale-free graph using Barabási-Albert model.

        Args:
            num_vertices: Total vertices
            attachment_edges: Edges each new vertex attaches to

        Returns:
            self
        """
        # Use NetworkX to generate, then copy to our graph
        ba_graph = nx.barabasi_albert_graph(num_vertices, attachment_edges)

        # Add vertices
        for v in ba_graph.nodes():
            self.graph.add_vertex(v)

        # Add edges with equal weight
        for u, v in ba_graph.edges():
            self.graph.add_edge(u, v, 1.0)

        return self


class RandomGraphFixture(GraphFixtureBase):
    """Random graph: Erdős-Rényi model."""

    def build(
        self, num_vertices: int = 20, edge_probability: float = 0.3
    ) -> "RandomGraphFixture":
        """Build random graph using Erdős-Rényi model.

        Args:
            num_vertices: Total vertices
            edge_probability: Probability of edge between any two vertices

        Returns:
            self
        """
        # Use NetworkX to generate, then copy to our graph
        er_graph = nx.erdos_renyi_graph(num_vertices, edge_probability)

        # Add vertices
        for v in er_graph.nodes():
            self.graph.add_vertex(v)

        # Add edges with random weights
        import random
        for u, v in er_graph.edges():
            weight = random.uniform(0.5, 2.0)
            self.graph.add_edge(u, v, weight)

        return self
