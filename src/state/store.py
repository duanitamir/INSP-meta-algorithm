from typing import Dict, Any
from threading import RLock
from src.graph.graph_manager import GraphManager
from src.state.node import NodeState
from src.utils.types import RoundNumber


class StateStore:
    """Central repository for all node states (thread-safe with per-node locks)."""

    def __init__(self, graph: GraphManager):
        self.graph = graph
        self._node_states: Dict[int, NodeState] = {}

        # Global state fields (formerly MetaState)
        self.round_num: RoundNumber = RoundNumber(0)
        self.converged: bool = False
        self.termination_reason: str | None = None
        self.final_matching: Dict[int, int] | None = None
        self.metadata: Dict[str, Any] | None = {}

        # Fine-grained per-node locks for thread-safe parallel execution (Option 2)
        self._node_locks: Dict[int, RLock] = {}

        for vertex_id in graph.vertices():
            node_state = NodeState(vertex_id)
            # Initialize neighbors dict with local knowledge
            neighbors = list(graph.neighbors(vertex_id))
            edge_weights = {n: graph.get_edge_weight(vertex_id, n) for n in neighbors}
            node_state.initialize_neighbors(neighbors, edge_weights)
            self._node_states[vertex_id] = node_state
            self._node_locks[vertex_id] = RLock()  # Each node gets its own lock

    def get_node_state(self, node_id: int) -> NodeState:
        """Get state for a node (defensive copy, thread-safe via per-node lock)."""
        if node_id not in self._node_states:
            raise ValueError(f"Node {node_id} not in state store")
        with self._node_locks[node_id]:
            return self._node_states[node_id].clone()

    def get_node_state_ref(self, node_id: int) -> NodeState:
        """Get state for a node (borrowed reference).

        Args:
            node_id: Node ID

        Returns:
            NodeState reference (NOT a clone) - READ-ONLY use only

        Raises:
            ValueError: If node_id not in state store
        """
        if node_id not in self._node_states:
            raise ValueError(f"Node {node_id} not in state store")
        # No lock needed for read-only access in this pattern
        # Caller must NOT modify the returned state
        return self._node_states[node_id]

    def get_all_states(self) -> Dict[int, NodeState]:
        """Get all node states (defensive copy)."""
        return {vid: state.clone() for vid, state in self._node_states.items()}

    def set_round(self, round_num: RoundNumber) -> None:
        """Set current round number."""
        self.round_num = round_num

    def mark_converged(self, reason: str) -> None:
        """Mark simulation as converged with termination reason."""
        self.converged = True
        self.termination_reason = reason

    def set_final_matching(self, matching: Dict[int, int]) -> None:
        """Set final matching result."""
        self.final_matching = matching

    def update_node_state(self, node_id: int, state: NodeState) -> None:
        """Update state for a node (thread-safe via per-node lock)."""
        if node_id not in self._node_states:
            raise ValueError(f"Node {node_id} not in state store")
        with self._node_locks[node_id]:
            self._node_states[node_id] = state.clone()

    def batch_update_node_states(self, states: Dict[int, NodeState]) -> None:
        """Batch update multiple node states with minimal lock contention.

        Args:
            states: Dict mapping node_id to updated NodeState

        Raises:
            ValueError: If any node_id is not in the state store
        """
        # Validate all nodes exist first (fail fast)
        for node_id in states:
            if node_id not in self._node_states:
                raise ValueError(f"Node {node_id} not in state store")

        # Apply all updates (acquire locks in consistent order to prevent deadlocks)
        for node_id in sorted(states.keys()):
            with self._node_locks[node_id]:
                self._node_states[node_id] = states[node_id].clone()

    def update_all_states(self, states: Dict[int, NodeState]) -> None:
        """Atomically update all node states."""
        for node_id, state in states.items():
            if node_id not in self._node_states:
                raise ValueError(f"Node {node_id} not in state store")
        for node_id, state in states.items():
            self._node_states[node_id] = state.clone()


