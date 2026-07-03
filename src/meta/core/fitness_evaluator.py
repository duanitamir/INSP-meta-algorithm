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

        # Import parameterizer here to avoid circular imports
        from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer

        # Create parameterizers
        parameterizers = [
            UnifiedAlgorithmParameterizer("greedy"),
            UnifiedAlgorithmParameterizer("itai"),
            UnifiedAlgorithmParameterizer("luby"),
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

        Prioritizes edges agreed upon by multiple algorithms, breaks ties by weight.
        Ensures symmetric matching. Uses same merge logic as CentralizedOrchestrator.

        Args:
            matchings: List of matching dicts
            graph: GraphManager for edge weights

        Returns:
            Merged matching dict
        """
        if not matchings:
            return {}

        edge_proposals = {}
        for matching in matchings:
            for u, v in matching.items():
                if u is None or v is None:
                    continue
                edge = tuple(sorted([u, v]))
                if edge not in edge_proposals:
                    edge_proposals[edge] = {"weights": [], "count": 0}
                weight = graph.get_edge_weight(u, v)
                edge_proposals[edge]["weights"].append(weight)
                edge_proposals[edge]["count"] += 1

        # Sort by: (proposal_count DESC, max_weight DESC)
        # Prefer edges found by multiple algorithms, break ties by weight
        final_matching = {}
        used_nodes = set()
        for edge in sorted(
            edge_proposals.keys(),
            key=lambda e: (
                edge_proposals[e]["count"],  # Algorithms agreeing (desc)
                max(edge_proposals[e]["weights"])  # Highest weight (desc)
            ),
            reverse=True,
        ):
            u, v = edge
            if u not in used_nodes and v not in used_nodes:
                final_matching[u] = v
                final_matching[v] = u
                used_nodes.add(u)
                used_nodes.add(v)

        return final_matching

    def name(self) -> str:
        """Return evaluator name."""
        return "FitnessEvaluator"
