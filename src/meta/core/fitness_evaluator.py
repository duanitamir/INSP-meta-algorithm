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

        # Setup orchestrator with graph
        self.orchestrator.setup(graph)

        # Run until convergence with max iterations from vector
        matching = self.orchestrator.run_until_convergence(
            max_rounds=int(vector.max_iterations),
            vector=vector
        )

        # Calculate weight as fitness score
        if matching:
            weight = 0.0
            for u, v in matching.items():
                if u < v:  # Count each edge once
                    weight += graph.get_edge_weight(u, v)
            return weight

        return 0.0

    def name(self) -> str:
        """Return evaluator name."""
        return "FitnessEvaluator"
