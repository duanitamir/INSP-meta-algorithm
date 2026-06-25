"""Fitness evaluator for scoring CanonicalVectors in genetic algorithm."""

from src.graph.graph_manager import GraphManager
from src.meta.canonical_vector import CanonicalVector
from src.meta.cascading_loop import CascadingLoop
from src.meta.greedy_parameterizer import GreedyParameterizer
from src.meta.itai_parameterizer import ItaiParameterizer
from src.meta.luby_parameterizer import LubyParameterizer


class FitnessEvaluator:
    """Evaluates fitness of a CanonicalVector by running CascadingLoop.

    Fitness is measured as the total weight of the matching produced.
    Higher weights indicate better fitness.
    """

    def __init__(self, cascading_loop: CascadingLoop) -> None:
        """Initialize fitness evaluator.

        Args:
            cascading_loop: CascadingLoop instance for running algorithm
        """
        self.cascading_loop = cascading_loop

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

        # Run cascading loop with the vector
        parameterizers = [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

        matching, metrics = self.cascading_loop.execute(graph, vector, parameterizers)

        # Return final weight as fitness score
        return metrics["final_weight"]

    def name(self) -> str:
        """Return evaluator name."""
        return "FitnessEvaluator"
