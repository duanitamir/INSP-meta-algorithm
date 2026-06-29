"""Parameterizer wrapper for Luby Randomized matching algorithm."""

from typing import Any, Callable, Dict

from src.meta.parameterizers.base import AlgorithmParameterizer
from src.meta.core.canonical_vector import CanonicalVector
from src.algorithms.implementations.luby_randomized import LubyRandomizedMatching
from src.simulation.scheduler import Scheduler
from src.simulation.config import SimulationConfig


class LubyParameterizer(AlgorithmParameterizer):
    """Wraps Luby Randomized algorithm with canonical vector parameters.

    Uses: canonical_vector Luby parameters [0:7] + max_iterations [8]
    Implements: Adaptive activation probability using Luby coefficients [1-6]
    Uses Template Method pattern from AlgorithmParameterizer base class.
    """

    def _extract_parameters(self, canonical_vector: CanonicalVector) -> Dict[str, Any]:
        """Extract Luby parameters from canonical vector.

        Uses: luby_base_probability [0], all coefficients [1-6], max_iterations [8].

        Args:
            canonical_vector: 10-parameter chromosome

        Returns:
            Dict with Luby-specific parameters
        """
        return {
            "base_probability": canonical_vector.luby_base_probability,
            "coeff_degree": canonical_vector.luby_coeff_degree,
            "coeff_neighbors_unmatched": canonical_vector.luby_coeff_neighbors_unmatched,
            "coeff_clustering": canonical_vector.luby_coeff_clustering,
            "coeff_matched": canonical_vector.luby_coeff_matched,
            "coeff_round": canonical_vector.luby_coeff_round,
            "coeff_weight": canonical_vector.luby_coeff_weight,
            "max_rounds": int(canonical_vector.max_iterations),
        }

    def _run_algorithm(self, graph: Any, parameters: Dict[str, Any]) -> Dict[int, int]:
        """Run Luby algorithm with adaptive activation parameters.

        Args:
            graph: GraphManager instance
            parameters: Dict with Luby parameters

        Returns:
            Matching dict {node_id -> matched_partner}
        """
        # Create the graph parameter dict for activation function
        # (avoids circular reference in _compute_adaptive_activation)
        graph_param = {
            "base_prob": parameters["base_probability"],
            "coeff_degree": parameters["coeff_degree"],
            "coeff_neighbors_unmatched": parameters["coeff_neighbors_unmatched"],
            "coeff_clustering": parameters["coeff_clustering"],
            "coeff_matched": parameters["coeff_matched"],
            "coeff_round": parameters["coeff_round"],
            "coeff_weight": parameters["coeff_weight"],
        }

        # Compute adaptive activation probability function
        activation_fn = self._compute_adaptive_activation_fn(graph, graph_param)

        # Create Luby algorithm with adaptive activation function
        luby = LubyRandomizedMatching(
            activation_probability=parameters["base_probability"],
            activation_function=activation_fn,
        )

        # Create scheduler with extracted parameters
        config = SimulationConfig(max_rounds=parameters["max_rounds"])
        scheduler = Scheduler(graph, luby, config)

        # Run scheduler until termination
        scheduler.run_until_termination()

        # Extract and return matching
        matching = luby.extract_matching(scheduler.state_store, graph)

        return matching

    def _compute_adaptive_activation_fn(
        self, graph: Any, params: Dict[str, float]
    ) -> Callable[[int], float]:
        """Compute per-node adaptive activation probability function.

        Uses node properties (degree, clustering, neighbors) weighted by
        parameters to compute node-specific probabilities.

        Args:
            graph: GraphManager instance
            params: Dict with coefficient values

        Returns:
            Function that maps node_id -> activation_probability [0, 1]
        """
        base_prob = params["base_prob"]
        coeff_degree = params["coeff_degree"]
        coeff_neighbors = params["coeff_neighbors_unmatched"]
        coeff_clustering = params["coeff_clustering"]
        coeff_matched = params["coeff_matched"]
        coeff_weight = params["coeff_weight"]

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
