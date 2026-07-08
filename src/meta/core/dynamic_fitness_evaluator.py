"""Fitness evaluator for selected algorithms.

Uses DynamicCanonicalVector to evaluate only the algorithms specified.
Evaluates matching quality for algorithm combinations.
"""

from typing import List, Optional, Any
from src.graph.graph_manager import GraphManager
from src.meta.core.dynamic_canonical_vector import DynamicCanonicalVector
from src.meta.core.algorithm_registry import AlgorithmRegistry
from src.meta.core.matching_merger import merge_matchings


class DynamicFitnessEvaluator:
    """Fitness evaluator for selected algorithms.

    Evaluates DynamicCanonicalVector for a specific set of algorithms.
    Only runs the algorithms selected, ignoring others in registry.

    Example:
        evaluator = DynamicFitnessEvaluator(["greedy", "luby"])
        vector = DynamicCanonicalVector(["greedy", "luby"])
        fitness = evaluator.evaluate(graph, vector)
    """

    def __init__(
        self,
        selected_algorithms: List[str],
        registry: Optional[AlgorithmRegistry] = None,
    ):
        """Initialize evaluator for selected algorithms.

        Args:
            selected_algorithms: List of algorithm names to evaluate (e.g., ["greedy", "luby"])
            registry: AlgorithmRegistry (default: singleton)

        Raises:
            ValueError: If no algorithms selected or invalid algorithm names
        """
        if not selected_algorithms:
            raise ValueError("At least one algorithm must be selected")

        self.registry = registry or AlgorithmRegistry.instance()
        self.selected_algorithms = selected_algorithms

        # Create parameterizers only for selected algorithms
        from src.meta.parameterizers.algorithm_parameterizer import (
            UnifiedAlgorithmParameterizer,
        )

        self.parameterizers = []
        for algo_name in selected_algorithms:
            # Verify algorithm exists in registry
            if not self.registry.is_algorithm_registered(algo_name):
                raise ValueError(f"Algorithm '{algo_name}' not found in registry")

            # Create parameterizer for this algorithm
            parameterizer = UnifiedAlgorithmParameterizer(algo_name)
            self.parameterizers.append(parameterizer)

    def evaluate(
        self,
        graph: GraphManager,
        vector: DynamicCanonicalVector,
    ) -> float:
        """Evaluate fitness of a vector for selected algorithms.

        Args:
            graph: GraphManager instance to evaluate on
            vector: DynamicCanonicalVector with parameters for selected algorithms

        Returns:
            float: Matching weight (fitness score). Higher is better.

        Raises:
            ValueError: If vector algorithms don't match evaluator algorithms
        """
        # Verify vector has same algorithms as evaluator
        if set(vector.get_algorithms()) != set(self.selected_algorithms):
            raise ValueError(
                f"Vector algorithms {vector.get_algorithms()} don't match "
                f"evaluator algorithms {self.selected_algorithms}"
            )

        # Run each algorithm and collect matchings
        matchings = []
        for parameterizer in self.parameterizers:
            try:
                # Execute algorithm with vector parameters
                matching = parameterizer.execute(graph, vector)
                matchings.append(matching)
            except Exception as e:
                # If algorithm fails, add empty matching and continue
                print(f"Warning: {parameterizer.name()} failed: {e}")
                matchings.append({})

        # Merge matchings via conflict resolution
        final_matching = merge_matchings(matchings, graph)

        # Calculate total weight as fitness
        if final_matching:
            weight = 0.0
            for u, v in final_matching.items():
                if u < v:  # Count each edge once
                    weight += graph.get_edge_weight(u, v)
            return weight

        return 0.0

    def get_algorithms(self) -> List[str]:
        """Get list of algorithms evaluated by this evaluator.

        Returns:
            List of algorithm names
        """
        return self.selected_algorithms.copy()

    def __repr__(self) -> str:
        """String representation for debugging."""
        algos_str = ", ".join(self.selected_algorithms)
        return f"DynamicFitnessEvaluator(algorithms=[{algos_str}])"
