"""Parameterizer wrapper for Luby Randomized matching algorithm."""

from typing import Dict
from src.meta.algorithm_parameterizer import AlgorithmParameterizer
from src.meta.canonical_vector import CanonicalVector
from src.algorithms.implementations.luby_randomized import LubyRandomizedMatching
from src.simulation.scheduler import Scheduler
from src.simulation.config import SimulationConfig


class LubyParameterizer(AlgorithmParameterizer):
    """Wraps Luby Randomized algorithm with canonical vector parameters.

    Uses: canonical_vector Luby parameters [0:7] + max_iterations [8], convergence_threshold [9]
    Implements: Adaptive activation probability using Luby coefficients [1-6]
    """

    def execute(
        self,
        graph,
        canonical_vector: CanonicalVector,
    ) -> Dict[int, int]:
        """Execute Luby algorithm with adaptive activation.

        Args:
            graph: GraphManager instance with vertices and edges
            canonical_vector: 10-parameter chromosome
                Uses parameters [0:10]:
                - [0]: luby_base_probability - base activation [0, 1]
                - [1-6]: Luby coefficients for adaptive activation
                - [7]: itai_timeout_rounds (not used by Luby)
                - [8]: max_iterations - max cascading iterations [5, 100]
                - [9]: convergence_threshold - improvement threshold [0, 0.1]

        Returns:
            Dict mapping node_id -> matched_partner (probabilistic matching).
        """
        # Validate vector
        is_valid, error = canonical_vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid canonical vector: {error}")

        # Create Luby algorithm
        luby = LubyRandomizedMatching()

        # Create scheduler with algorithm using max_iterations from vector
        config = SimulationConfig(max_rounds=canonical_vector.max_iterations)
        scheduler = Scheduler(graph, luby, config)

        # Run scheduler until termination
        scheduler.run_until_termination()

        # Extract matching
        matching = luby.extract_matching(scheduler.state_store, graph)

        return matching

    def name(self) -> str:
        """Return algorithm name."""
        return "Luby Randomized"
