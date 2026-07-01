"""Parameterizer wrapper for Greedy matching algorithm."""

from typing import Any, Dict

from src.meta.parameterizers.base import AlgorithmParameterizer
from src.meta.core.canonical_vector import CanonicalVector
from src.algorithms.implementations.greedy_matching import GreedyMatching
from src.simulation.distributed_node import DistributedNode
from src.communication.drivers.in_memory_driver import InMemoryDriver


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

        # Create distributed node with transport driver
        transport = InMemoryDriver(graph)
        node = DistributedNode(node_id=0, graph=graph)

        # Run algorithm until convergence (max_rounds is a safety limit)
        max_rounds = parameters.get("max_rounds", 100)
        for _ in range(max_rounds):
            should_continue, _ = node.execute(greedy)
            if not should_continue:
                break

        # Extract and return matching
        matching = node.get_matching()

        return matching

    def name(self) -> str:
        """Return algorithm name."""
        return "Greedy"
