"""Parameterizer wrapper for Luby Randomized matching algorithm."""

from typing import Dict, Callable
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

        # Compute adaptive activation probability function
        activation_fn = self._compute_adaptive_activation(graph, canonical_vector)

        # Create Luby algorithm with adaptive activation function
        luby = LubyRandomizedMatching(
            activation_probability=canonical_vector.luby_base_probability,
            activation_function=activation_fn,
        )

        # Create scheduler with algorithm using max_iterations from vector
        config = SimulationConfig(max_rounds=canonical_vector.max_iterations)
        scheduler = Scheduler(graph, luby, config)

        # Run scheduler until termination
        scheduler.run_until_termination()

        # Extract matching
        matching = luby.extract_matching(scheduler.state_store, graph)

        return matching

    def _compute_adaptive_activation(
        self, graph, canonical_vector: CanonicalVector
    ) -> Callable[[int], float]:
        """Compute per-node adaptive activation probability function.

        Uses node properties (degree, clustering, neighbors) weighted by
        canonical vector coefficients to compute node-specific probabilities.

        Args:
            graph: GraphManager instance
            canonical_vector: Parameter vector with coefficients

        Returns:
            Function that maps node_id -> activation_probability [0, 1]
        """
        base_prob = canonical_vector.luby_base_probability
        coeff_degree = canonical_vector.luby_coeff_degree
        coeff_neighbors = canonical_vector.luby_coeff_neighbors_unmatched
        coeff_clustering = canonical_vector.luby_coeff_clustering
        coeff_matched = canonical_vector.luby_coeff_matched
        coeff_weight = canonical_vector.luby_coeff_weight

        # Precompute node properties for efficiency
        vertices = list(graph.vertices())
        degrees = [graph.degree(v) for v in vertices]
        max_degree = max(degrees) if degrees else 1

        # Precompute average edge weights per node
        avg_weights = []
        for v in vertices:
            neighbors = list(graph.neighbors(v))
            if neighbors:
                weights = [graph.get_edge_weight(v, n) for n in neighbors]
                avg_weights.append(sum(weights) / len(weights))
            else:
                avg_weights.append(0.0)
        max_weight = max(avg_weights) if avg_weights else 1.0

        # Create mapping for quick lookup
        degree_map = {v: d for v, d in zip(vertices, degrees)}
        weight_map = {v: w for v, w in zip(vertices, avg_weights)}

        def activation_fn(node_id: int) -> float:
            """Compute adaptive activation probability for a single node."""
            # Normalize degree [0, 1]
            normalized_degree = degree_map.get(node_id, 0) / max_degree if max_degree > 0 else 0

            # Normalize average edge weight [0, 1]
            normalized_weight = weight_map.get(node_id, 0) / max_weight if max_weight > 0 else 0

            # Adaptive probability (base + weighted coefficients)
            prob = base_prob
            prob += coeff_degree * (normalized_degree - 0.5)  # Center around 0
            prob += coeff_neighbors * (normalized_degree - 0.5)  # Proxy for neighbors
            prob += coeff_clustering * (normalized_degree - 0.5)  # Simple clustering proxy
            prob += coeff_matched * (normalized_degree - 0.5)  # Matched neighbors proxy
            prob += coeff_weight * (normalized_weight - 0.5)

            # Clamp to [0, 1]
            return max(0.0, min(1.0, prob))

        return activation_fn

    def name(self) -> str:
        """Return algorithm name."""
        return "Luby Randomized"
