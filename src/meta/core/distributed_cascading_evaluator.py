"""Cascading fitness evaluator using fully distributed orchestration.

Implements cascading rounds on shrinking graphs using DistributedOrchestrator.
Each cascade runs each algorithm independently on autonomous nodes, then merges results.
Matched nodes are removed between passes, accumulating matches across cascades.
"""

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.distributed.orchestrator import DistributedOrchestrator


class DistributedCascadingEvaluator:
    """Evaluates fitness using cascading rounds with autonomous distributed nodes.

    For each cascade round:
    1. Create filtered graph with only unmatched nodes
    2. Run DistributedOrchestrator with autonomous nodes on filtered graph
    3. Nodes execute all algorithms (Greedy, Itai, Luby) independently
    4. Nodes resolve conflicts via endpoint voting
    5. Collect matching from autonomous nodes
    6. Mark matched nodes as inactive for next cascade
    7. Check convergence (improvement < threshold)
    8. Continue if improvement sufficient, else stop

    Each cascade runs fully distributed - nodes make autonomous decisions,
    exchange proposals via messages, resolve conflicts via voting.
    """

    def __init__(self, max_workers: int = 4, min_rounds: int = 10) -> None:
        """Initialize cascading evaluator.

        Args:
            max_workers: Number of worker threads for parallel node execution
            min_rounds: Minimum iterations before allowing early termination
        """
        self.max_workers = max_workers
        self.min_rounds = min_rounds  # Ensure enough rounds for synchronization
        self.last_num_cascades = 0
        self.last_weights_per_cascade = []

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate fitness using cascading rounds with standard algorithms.

        Uses the same algorithm execution as standard evaluator, but runs repeatedly
        on a shrinking graph where matched nodes are removed between passes.

        Compatible with FitnessEvaluator interface - returns just the fitness weight.
        Cascading details stored in self.last_num_cascades and self.last_weights_per_cascade.

        Args:
            graph: GraphManager instance
            vector: CanonicalVector with parameters including max_iterations and convergence_threshold

        Returns:
            Final weight (fitness score). Cascade details in self.last_* attributes for analysis.
        """
        is_valid, error = vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid vector: {error}")

        # Get parameters from vector using generic .get() method (100% agnostic)
        max_cascades = int(vector.get("max_iterations") or 100)
        convergence_threshold = vector.get("convergence_threshold") or 0.05

        # Initialize matched nodes tracking
        already_matched_nodes = set()

        prev_weight = 0.0
        weight_per_round = []
        cascade_round = 0
        total_weight = 0.0  # Accumulate weight across ALL cascades

        # Cascading rounds using distributed orchestrator (each algo independent, then merge)
        for cascade_round in range(max_cascades):
            # If no unmatched nodes remain, stop cascading
            unmatched_nodes = set(graph.vertices()) - already_matched_nodes
            if not unmatched_nodes:
                break

            # Create filtered graph with only unmatched nodes
            cascade_graph = self._create_filtered_graph(graph, already_matched_nodes)

            # Run distributed orchestrator on filtered graph
            orchestrator = DistributedOrchestrator(
                max_workers=self.max_workers,
                use_convergence_detection=False,
                min_iterations=self.min_rounds
            )

            matching, _ = orchestrator.execute(cascade_graph, vector)

            # Calculate weight for THIS CASCADE (new matches found on filtered graph)
            curr_weight = 0.0
            if matching:
                for u, v in matching.items():
                    if u < v:  # Count each edge once
                        curr_weight += graph.get_edge_weight(u, v)

            weight_per_round.append(curr_weight)
            total_weight += curr_weight  # Accumulate across cascades

            # Check convergence
            if cascade_round > 0:
                improvement = (curr_weight - prev_weight) / (prev_weight + 1e-10)
                if improvement < convergence_threshold:
                    # Convergence reached, stop cascading
                    break

            # Update matched nodes for next cascade
            for u, v in matching.items():
                already_matched_nodes.add(u)
                already_matched_nodes.add(v)

            prev_weight = curr_weight

        # Store details for analysis
        self.last_num_cascades = cascade_round + 1
        self.last_weights_per_cascade = weight_per_round

        # Return total accumulated matched weight across all cascades
        return total_weight

    def _create_filtered_graph(
        self, graph: GraphManager, already_matched_nodes: set
    ) -> GraphManager:
        """Create a filtered graph containing only unmatched nodes.

        Args:
            graph: Original full graph
            already_matched_nodes: Set of node IDs already matched in previous cascades

        Returns:
            New GraphManager containing only unmatched nodes and their edges
        """
        filtered_graph = GraphManager.create_empty_graph()

        # Add only unmatched nodes
        for node_id in graph.vertices():
            if node_id not in already_matched_nodes:
                filtered_graph.add_vertex(node_id)

        # Add edges between unmatched nodes only
        for u in filtered_graph.vertices():
            for v in graph.neighbors(u):
                if v not in already_matched_nodes and u < v:  # Avoid duplicate edges
                    weight = graph.get_edge_weight(u, v)
                    filtered_graph.add_edge(u, v, weight)

        return filtered_graph

    def name(self) -> str:
        """Return evaluator name."""
        return "DistributedCascadingEvaluator"
