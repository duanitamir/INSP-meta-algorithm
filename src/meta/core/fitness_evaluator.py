"""Fitness evaluator for scoring CanonicalVectors in genetic algorithm.

Uses DistributedOrchestrator (distributed algorithm) instead of centralized CascadingLoop.
"""

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.distributed.orchestrator import DistributedOrchestrator
from src.meta.parameterizers.factory import ParameterizerFactory


class FitnessEvaluator:
    """Evaluates fitness of a CanonicalVector using distributed orchestration.

    Uses DistributedOrchestrator (distributed conflict resolution + convergence detection)
    instead of centralized CascadingLoop. Fitness is measured as the total weight of the
    matching produced. Higher weights indicate better fitness.
    """

    def __init__(self) -> None:
        """Initialize fitness evaluator with distributed orchestrator."""
        self.orchestrator = DistributedOrchestrator()

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate fitness of a vector on given graph using distributed algorithm.

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

        # Run distributed orchestrator with the vector
        parameterizers = ParameterizerFactory.create_default()
        matching, metrics = self.orchestrator.execute(graph, vector, parameterizers)

        # Return final weight as fitness score
        return metrics["final_weight"]

    def name(self) -> str:
        """Return evaluator name."""
        return "FitnessEvaluator"
