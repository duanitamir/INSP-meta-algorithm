"""Fitness evaluator for scoring CanonicalVectors in genetic algorithm.

Uses CentralizedOrchestrator for fast GA fitness evaluation.
"""

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.simulation.centralized_orchestrator import CentralizedOrchestrator


class FitnessEvaluator:
    """Evaluates fitness of a CanonicalVector using centralized orchestration.

    Uses CentralizedOrchestrator for fast fitness evaluation in GA.
    Fitness is measured as the total weight of the matching produced.
    """

    def __init__(self) -> None:
        """Initialize fitness evaluator with centralized orchestrator."""
        self.orchestrator = CentralizedOrchestrator()

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate fitness of a vector on given graph.

        Uses parameterizers to run algorithms with full parameter support.
        Each parameterizer extracts and uses relevant parameters from CanonicalVector.

        Args:
            graph: GraphManager instance
            vector: CanonicalVector to evaluate

        Returns:
            float: Fitness score (matching weight, higher is better)

        Raises:
            ValueError: If vector is invalid
        """
        is_valid, error = vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid vector: {error}")

        # Import parameterizers here to avoid circular imports
        from src.meta.parameterizers.greedy import GreedyParameterizer
        from src.meta.parameterizers.itai import ItaiParameterizer
        from src.meta.parameterizers.luby import LubyParameterizer

        # Create parameterizers
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        # Run all 3 parameterizers to get their matchings
        matchings = []
        for parameterizer in parameterizers:
            try:
                matching = parameterizer.execute(graph, vector)
                matchings.append(matching)
            except Exception:
                matchings.append({})

        # Merge matchings via conflict resolution (keep highest-weight edges)
        final_matching = self._merge_matchings(matchings, graph)

        # Calculate weight as fitness score
        if final_matching:
            weight = 0.0
            for u, v in final_matching.items():
                if u < v:  # Count each edge once
                    weight += graph.get_edge_weight(u, v)
            return weight

        return 0.0

    def _merge_matchings(self, matchings: list, graph: GraphManager) -> dict:
        """Merge multiple matchings via conflict resolution.

        Keeps highest-weight non-conflicting edges from all matchings.

        Args:
            matchings: List of matching dicts
            graph: GraphManager for edge weights

        Returns:
            Merged matching dict
        """
        edge_proposals = {}

        # Collect all proposed edges with their weights
        for matching in matchings:
            for u, v in matching.items():
                if u < v:  # Normalize edge
                    weight = graph.get_edge_weight(u, v)
                    if (u, v) not in edge_proposals:
                        edge_proposals[(u, v)] = weight
                    else:
                        # Keep highest weight
                        edge_proposals[(u, v)] = max(edge_proposals[(u, v)], weight)

        # Sort by weight descending
        sorted_edges = sorted(edge_proposals.items(), key=lambda x: x[1], reverse=True)

        # Greedily select non-conflicting edges
        final_matching = {}
        used_nodes = set()

        for (u, v), weight in sorted_edges:
            if u not in used_nodes and v not in used_nodes:
                final_matching[u] = v
                final_matching[v] = u
                used_nodes.add(u)
                used_nodes.add(v)

        return final_matching

    def name(self) -> str:
        """Return evaluator name."""
        return "FitnessEvaluator"
