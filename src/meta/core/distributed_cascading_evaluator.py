"""Distributed cascading evaluator for GA fitness evaluation.

Implements cascading rounds using DistributedOrchestrator where autonomous nodes
run repeatedly on shrinking graphs. Matched nodes become inactive between cascades,
creating a logically smaller graph each round.
"""

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector


class DistributedCascadingEvaluator:
    """Evaluates fitness using distributed cascading rounds with autonomous nodes.

    For each cascade round:
    1. Create filtered graph with only unmatched nodes
    2. Run DistributedOrchestrator (autonomous nodes with message passing)
    3. Collect matched edges from autonomous node coordination
    4. Mark matched nodes as inactive for next cascade
    5. Check convergence (improvement < threshold)
    6. Continue if improvement sufficient, else stop

    This implements TRUE distributed cascading where:
    - Autonomous nodes execute with message passing coordination
    - Conflict resolution via endpoint voting
    - Convergence detection via quorum voting
    - Multi-pass refinement on shrinking graphs
    """

    def __init__(self) -> None:
        """Initialize distributed cascading evaluator."""
        self.last_num_cascades = 0
        self.last_weights_per_cascade = []

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate fitness using distributed cascading rounds with autonomous nodes.

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

        from src.meta.distributed.orchestrator import DistributedOrchestrator

        # Get parameters from vector
        max_cascades = int(vector.max_iterations)
        convergence_threshold = vector.convergence_threshold

        # Initialize matched nodes tracking
        already_matched_nodes = set()

        prev_weight = 0.0
        weight_per_round = []
        cascade_round = 0
        total_weight = 0.0  # Accumulate weight across ALL cascades

        # Cascading rounds with autonomous node execution
        for cascade_round in range(max_cascades):
            # If no unmatched nodes remain, stop cascading
            unmatched_nodes = set(graph.vertices()) - already_matched_nodes
            if not unmatched_nodes:
                break

            # Create filtered graph with only unmatched nodes
            # This ensures DistributedOrchestrator only creates nodes for unmatched vertices
            cascade_graph = self._create_filtered_graph(graph, already_matched_nodes)

            # Run DistributedOrchestrator (autonomous nodes with message passing)
            # Pass pre_matched_nodes so matched nodes are marked as finished
            orchestrator = DistributedOrchestrator(max_workers=4)
            matching, metrics = orchestrator.execute(
                cascade_graph,
                vector,
                pre_matched_nodes=already_matched_nodes
            )

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
        # This represents the cumulative value of all matches found by autonomous nodes
        # across multiple distributed execution passes on the shrinking graph
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
