"""Adapter to provide per-node interface for existing algorithms.

Wraps a MatchingAlgorithm to provide execute_local_step() method
that computes behavior for a single node, not the full graph.

This allows existing algorithms (GreedyMatching, ItaiIsraeliMaximalMatching, etc.)
to be used without modification while providing per-node interface.
"""

from typing import Tuple, Dict, Any, List
from src.simulation.node_context import NodeContext
from src.algorithms.base import MatchingAlgorithm
from src.state.node_state import NodeState


class LocalAlgorithmAdapter:
    """Adapts any MatchingAlgorithm to provide per-node execution interface.

    Provides: execute_local_step(node_context) -> (new_state, outgoing_messages)

    Instead of: algorithm.execute(full_graph) -> full_matching

    This allows algorithms to be used at node level without wrapping them in
    StateStore + MessageQueue + ThreadPoolExecutor.
    """

    def __init__(self, algorithm: MatchingAlgorithm):
        """Initialize adapter with an algorithm.

        Args:
            algorithm: Any MatchingAlgorithm (GreedyMatching, ItaiIsraeliMaximalMatching, etc.)
        """
        self.algorithm = algorithm

    def execute_local_step(
        self, node_context: NodeContext
    ) -> Tuple[NodeState, List[Dict[str, Any]]]:
        """Execute algorithm for one node in one round.

        Args:
            node_context: Execution context containing node_id, state, messages, etc.

        Returns:
            Tuple of (new_state, outgoing_messages)
        """
        # Get current state and messages for this node
        node_id = node_context.node_id
        current_state = node_context.state
        incoming_messages = node_context.incoming_messages

        # Call the algorithm's node_behavior method
        # This is the existing algorithm interface that already computes
        # new state for a single node
        from src.simulation.algorithm_context import AlgorithmContext

        context = AlgorithmContext(
            graph=node_context.graph,
            state_store=node_context.state_store,
            round_num=node_context.round_number,
        )

        new_state, outgoing_messages = self.algorithm.node_behavior(
            node_id=node_id,
            node_state=current_state,
            messages=incoming_messages,
            context=context,
        )

        return new_state, outgoing_messages

    def name(self) -> str:
        """Get adapter name for logging."""
        return f"LocalAdapter({self.algorithm.name()})"
