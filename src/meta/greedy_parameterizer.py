"""Parameterizer wrapper for Greedy matching algorithm."""

from typing import Dict
from src.meta.algorithm_parameterizer import AlgorithmParameterizer
from src.meta.canonical_vector import CanonicalVector
from src.algorithms.implementations.greedy_matching import GreedyMatching
from src.simulation.scheduler import Scheduler
from src.simulation.config import SimulationConfig


class GreedyParameterizer(AlgorithmParameterizer):
    """Wraps Greedy algorithm with canonical vector parameters.

    Greedy has no algorithm-specific tunable parameters in the canonical vector.
    This wrapper simply executes the base Greedy algorithm on the full graph.
    """

    def execute(
        self,
        graph,
        canonical_vector: CanonicalVector,
    ) -> Dict[int, int]:
        """Execute Greedy algorithm.

        Args:
            graph: GraphManager instance with vertices and edges
            canonical_vector: 10-parameter chromosome (ignored for Greedy)

        Returns:
            Dict mapping node_id -> matched_partner (maximal matching).
        """
        # Validate vector
        is_valid, error = canonical_vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid canonical vector: {error}")

        # Create Greedy algorithm
        greedy = GreedyMatching()

        # Create scheduler with algorithm
        config = SimulationConfig(max_rounds=100)
        scheduler = Scheduler(graph, greedy, config)

        # Run scheduler until termination
        scheduler.run_until_termination()

        # Extract matching
        matching = greedy.extract_matching(scheduler.state_store, graph)

        return matching

    def name(self) -> str:
        """Return algorithm name."""
        return "Greedy"
