"""Cascading fitness evaluator for GA optimization.

Implements cascading rounds on shrinking graphs using centralized approach.
Each cascade runs 3 algorithms independently on progressively smaller graphs
where matched nodes are removed between passes.

Matched nodes become inactive between cascades, creating progressively smaller graphs.
Multiple cascades accumulate matches from each pass on the shrinking graph.
"""

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector


class CascadingEvaluator:
    """Evaluates fitness using centralized cascading rounds on shrinking graphs.

    For each cascade round:
    1. Create filtered graph with only unmatched nodes
    2. Run all 3 algorithms independently (fresh StateStore each)
    3. Merge matchings via conflict resolution
    4. Mark matched nodes as inactive for next cascade
    5. Check convergence (improvement < threshold)
    6. Continue if improvement sufficient, else stop

    Each algorithm gets independent access to nodes in the current cascade,
    producing different matchings that merge together for better results.
    """

    def __init__(self) -> None:
        """Initialize cascading evaluator."""
        self.last_num_cascades = 0
        self.last_weights_per_cascade = []

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate fitness using cascading rounds on shrinking graphs.

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

        from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer
        from src.meta.core.matching_merger import merge_matchings
        from src.state.store import StateStore

        # Get parameters from vector
        max_cascades = int(vector.max_iterations)
        convergence_threshold = vector.convergence_threshold

        # Initialize matched nodes tracking
        already_matched_nodes = set()

        prev_weight = 0.0
        weight_per_round = []
        cascade_round = 0
        total_weight = 0.0  # Accumulate weight across ALL cascades

        # Cascading rounds
        for cascade_round in range(max_cascades):
            # If no unmatched nodes remain, stop cascading
            unmatched_nodes = set(graph.vertices()) - already_matched_nodes
            if not unmatched_nodes:
                break

            # Create filtered graph with only unmatched nodes
            # This ensures algorithms only work on the shrinking graph
            cascade_graph = self._create_filtered_graph(graph, already_matched_nodes)

            # Run all 3 algorithms and merge their results
            # CRITICAL: Create fresh StateStore for EACH algorithm
            # If we reuse one StateStore, Greedy's matches mark all nodes as matched,
            # and then Itai/Luby see an empty graph and return the same Greedy matching!
            parameterizers = [
                UnifiedAlgorithmParameterizer("greedy"),
                UnifiedAlgorithmParameterizer("itai"),
                UnifiedAlgorithmParameterizer("luby"),
            ]

            matchings = []
            for parameterizer in parameterizers:
                try:
                    # Fresh StateStore for each algorithm (crucial!)
                    state_store = StateStore(cascade_graph)
                    matching = parameterizer.execute(cascade_graph, vector, state_store=state_store)
                    matchings.append(matching)
                except Exception:
                    matchings.append({})

            # Merge matchings via conflict resolution
            matching = merge_matchings(matchings, cascade_graph)

            # Calculate weight for THIS CASCADE (all matches found on filtered graph)
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
