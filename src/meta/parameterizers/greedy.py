"""Parameterizer wrapper for Greedy matching algorithm."""

from typing import Any, Dict

from src.meta.parameterizers.base import AlgorithmParameterizer
from src.meta.core.canonical_vector import CanonicalVector
from src.algorithms.implementations.greedy_matching import GreedyMatching
from src.simulation.scheduler import Scheduler
from src.simulation.config import SimulationConfig


class GreedyParameterizer(AlgorithmParameterizer):
    """Wraps Greedy algorithm with canonical vector parameters.

    Greedy has no algorithm-specific tunable parameters in the canonical vector.
    Uses Template Method pattern from AlgorithmParameterizer base class.
    """

    def _extract_parameters(self, canonical_vector: CanonicalVector) -> Dict[str, Any]:
        """Extract Greedy parameters from canonical vector.

        Greedy uses only max_iterations from the vector.

        Args:
            canonical_vector: 10-parameter chromosome

        Returns:
            Dict with 'max_rounds' key
        """
        return {"max_rounds": int(canonical_vector.max_iterations)}

    def _run_algorithm(self, graph: Any, parameters: Dict[str, Any]) -> Dict[int, int]:
        """Run Greedy algorithm with extracted parameters.

        Args:
            graph: GraphManager instance
            parameters: Dict with 'max_rounds' key

        Returns:
            Matching dict {node_id -> matched_partner}
        """
        # Create Greedy algorithm
        greedy = GreedyMatching()

        # Create scheduler with extracted parameters
        config = SimulationConfig(max_rounds=parameters["max_rounds"])
        scheduler = Scheduler(graph, greedy, config)

        # Run scheduler until termination
        scheduler.run_until_termination()

        # Extract and return matching
        matching = greedy.extract_matching(scheduler.state_store, graph)

        return matching

    def name(self) -> str:
        """Return algorithm name."""
        return "Greedy"
