"""Parameterizer wrapper for Itai-Israeli matching algorithm."""

from typing import Any, Dict

from src.meta.parameterizers.base import AlgorithmParameterizer
from src.meta.core.canonical_vector import CanonicalVector
from src.algorithms.implementations.itai_israeli import ItaiIsraeliMaximalMatching
from src.simulation.distributed_node import DistributedNode
from src.communication.drivers.in_memory_driver import InMemoryDriver


class ItaiParameterizer(AlgorithmParameterizer):
    """Wraps Itai-Israeli algorithm with canonical vector parameters.

    Uses: canonical_vector.itai_timeout_rounds [7]
    Uses Template Method pattern from AlgorithmParameterizer base class.
    """

    def _extract_parameters(self, canonical_vector: CanonicalVector) -> Dict[str, Any]:
        """Extract Itai-Israeli parameters from canonical vector.

        Uses: itai_timeout_rounds [7] and max_iterations [8].

        Args:
            canonical_vector: 10-parameter chromosome

        Returns:
            Dict with 'timeout_rounds' and 'max_rounds' keys
        """
        return {
            "timeout_rounds": canonical_vector.itai_timeout_rounds,
            "max_rounds": int(canonical_vector.max_iterations),
        }

    def _run_algorithm(self, graph: Any, parameters: Dict[str, Any]) -> Dict[int, int]:
        """Run Itai-Israeli algorithm with extracted parameters.

        Args:
            graph: GraphManager instance
            parameters: Dict with 'timeout_rounds' and 'max_rounds' keys

        Returns:
            Matching dict {node_id -> matched_partner}
        """
        # Create Itai-Israeli algorithm with timeout parameter
        itai = ItaiIsraeliMaximalMatching(timeout_rounds=parameters["timeout_rounds"])

        # Create distributed node with transport driver
        transport = InMemoryDriver(graph)
        node = DistributedNode(node_id=0, graph=graph)

        # Run algorithm until convergence (max_rounds is a safety limit)
        max_rounds = parameters.get("max_rounds", 100)
        for _ in range(max_rounds):
            should_continue, _ = node.execute(itai)
            if not should_continue:
                break

        # Extract and return matching
        matching = node.get_matching()

        return matching

    def name(self) -> str:
        """Return algorithm name."""
        return "Itai-Israeli"
