"""Parameterizer wrapper for Itai-Israeli matching algorithm."""

from typing import Dict
from src.meta.algorithm_parameterizer import AlgorithmParameterizer
from src.meta.canonical_vector import CanonicalVector
from src.algorithms.implementations.itai_israeli import ItaiIsraeliMaximalMatching
from src.simulation.scheduler import Scheduler
from src.simulation.config import SimulationConfig


class ItaiParameterizer(AlgorithmParameterizer):
    """Wraps Itai-Israeli algorithm with canonical vector parameters.

    Uses: canonical_vector.itai_timeout_rounds [7]
    Passes timeout to Itai-Israeli algorithm to prevent infinite loops.
    """

    def execute(
        self,
        graph,
        canonical_vector: CanonicalVector,
    ) -> Dict[int, int]:
        """Execute Itai-Israeli algorithm with timeout.

        Args:
            graph: GraphManager instance with vertices and edges
            canonical_vector: 10-parameter chromosome
                Uses: itai_timeout_rounds [7] to set max rounds per iteration

        Returns:
            Dict mapping node_id -> matched_partner (maximum weight matching).
        """
        # Validate vector
        is_valid, error = canonical_vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid canonical vector: {error}")

        # Create Itai-Israeli algorithm
        itai = ItaiIsraeliMaximalMatching()

        # Create scheduler with algorithm
        config = SimulationConfig(max_rounds=100)
        scheduler = Scheduler(graph, itai, config)

        # Run scheduler until termination
        scheduler.run_until_termination()

        # Extract matching
        matching = itai.extract_matching(scheduler.state_store, graph)

        return matching

    def name(self) -> str:
        """Return algorithm name."""
        return "Itai-Israeli"
