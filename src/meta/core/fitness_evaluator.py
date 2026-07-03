"""Fitness evaluator for scoring CanonicalVectors in genetic algorithm.

Uses CentralizedOrchestrator for fast GA fitness evaluation.
"""

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.core.matching_merger import merge_matchings
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
        final_matching = merge_matchings(matchings, graph)

        # Calculate weight as fitness score
        if final_matching:
            weight = 0.0
            for u, v in final_matching.items():
                if u < v:  # Count each edge once
                    weight += graph.get_edge_weight(u, v)
            return weight

        return 0.0

    def name(self) -> str:
        """Return evaluator name."""
        return "FitnessEvaluator"
