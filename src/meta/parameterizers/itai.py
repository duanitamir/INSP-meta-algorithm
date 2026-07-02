"""Parameterizer wrapper for Itai-Israeli matching algorithm.

Uses ThreadPoolExecutor for parallel node execution within each round (Option 2 optimization).
Each node in the round executes concurrently with fine-grained per-node locking in StateStore.
"""

from typing import Any, Dict, Tuple, List

from src.meta.parameterizers.base import AlgorithmParameterizer
from src.meta.core.canonical_vector import CanonicalVector
from src.state.node_state import NodeState
from src.simulation.node_context import NodeContext


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
        """Run Itai-Israeli algorithm via per-node execution (Phase 3 refactored).

        PHASE 3 REFACTOR: Delegates to execute_local_step() for per-node execution.

        Creates minimal resources needed for backward compatibility with old interface.
        In real distributed deployment, orchestrator would call execute_local_step() directly.

        Args:
            graph: GraphManager instance
            parameters: Dict with parameters (timeout_rounds, max_rounds)

        Returns:
            Matching dict {node_id -> matched_partner}
        """
        # Import here to avoid circular imports
        from src.state.state_store import StateStore
        from src.communication.message_queue import MessageQueue
        from src.simulation.node_context import NodeContext

        # Create minimal resources for backward compatibility
        state_store = StateStore(graph)
        message_queue = MessageQueue(graph)

        # For backward compatibility, create a temporary vector with params
        if not hasattr(self, "_temp_vector"):
            from src.meta.core.canonical_vector import CanonicalVector

            timeout = parameters.get("timeout_rounds", 5)
            self._temp_vector = CanonicalVector(itai_timeout_rounds=timeout)

        # Execute each node using the new per-node interface
        matching = {}
        for node_id in graph.vertices():
            node_state = state_store.get_node_state(node_id)
            messages = message_queue.get_messages(node_id)

            context = NodeContext(
                node_id=node_id,
                state=node_state,
                incoming_messages=messages,
                graph=graph,
                vector=self._temp_vector,
                round_number=0,
                state_store=state_store,
            )

            new_state, _ = self.execute_local_step(context)
            state_store.update_node_state(node_id, new_state)

            if new_state.is_matched():
                matching[node_id] = new_state.get_matched_to()

        return matching

    def name(self) -> str:
        """Return algorithm name."""
        return "Itai-Israeli"

    def execute_local_step(
        self, node_context: NodeContext
    ) -> Tuple[NodeState, List[Dict[str, Any]]]:
        """Execute Itai-Israeli algorithm for one node in one round (Phase 1).

        Implements per-node execution without creating centralized StateStore
        or running full graph iterations.

        Args:
            node_context: NodeContext containing node_id, state, messages, etc.

        Returns:
            Tuple of (new_node_state, outgoing_messages)
        """
        from src.algorithms.implementations.itai_israeli import ItaiIsraeliMaximalMatching
        from src.simulation.algorithm_context import AlgorithmContext

        # Extract timeout from vector
        timeout_rounds = node_context.vector.itai_timeout_rounds
        itai = ItaiIsraeliMaximalMatching(timeout_rounds=timeout_rounds)
        node_id = node_context.node_id
        current_state = node_context.state
        incoming_messages = node_context.incoming_messages

        # Create context for algorithm execution
        context = AlgorithmContext(
            graph=node_context.graph,
            state_store=node_context.state_store,
            round_num=node_context.round_number,
        )

        # Execute Itai-Israeli behavior for THIS NODE ONLY
        new_state, outgoing_messages = itai.node_behavior(
            node_id=node_id,
            node_state=current_state,
            messages=incoming_messages,
            context=context,
        )

        return new_state, outgoing_messages
