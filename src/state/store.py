from typing import Dict, List, Any
from dataclasses import dataclass
from threading import RLock
from src.graph.graph_manager import GraphManager
from src.state.node import NodeState
from src.utils.types import RoundNumber


@dataclass
class MetaState:
    """Global simulation state."""

    round_num: RoundNumber
    converged: bool = False
    termination_reason: str | None = None
    final_matching: Dict[int, int] | None = None
    metadata: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def with_round(self, round_num: RoundNumber) -> "MetaState":
        """Return new MetaState with updated round number."""
        return MetaState(
            round_num=round_num,
            converged=self.converged,
            termination_reason=self.termination_reason,
            final_matching=self.final_matching,
            metadata=self.metadata.copy() if self.metadata else {},
        )

    def with_convergence(self, reason: str) -> "MetaState":
        """Return new MetaState marked as converged."""
        return MetaState(
            round_num=self.round_num,
            converged=True,
            termination_reason=reason,
            final_matching=self.final_matching,
            metadata=self.metadata.copy() if self.metadata else {},
        )


@dataclass
class StateSnapshot:
    """Immutable snapshot of all state at a specific round."""

    round_num: RoundNumber
    node_states: Dict[int, NodeState]
    meta_state: MetaState

    def get_node_state(self, node_id: int) -> NodeState:
        """Get state for a node."""
        return self.node_states[node_id].clone()

    def get_all_node_states(self) -> Dict[int, NodeState]:
        """Get all node states (defensive copy)."""
        return {vid: state.clone() for vid, state in self.node_states.items()}

    def get_meta_state(self) -> MetaState:
        """Get global state."""
        return self.meta_state


class StateStore:
    """Central repository for all node states (thread-safe with per-node locks)."""

    def __init__(self, graph: GraphManager):
        self.graph = graph
        self._node_states: Dict[int, NodeState] = {}
        self._meta_state = MetaState(round_num=RoundNumber(0))
        self._snapshots: List[StateSnapshot] = []

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

    def get_meta_state(self) -> MetaState:
        """Get global simulation state."""
        return self._meta_state

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

    def update_meta_state(self, meta_state: MetaState) -> None:
        """Update global simulation state."""
        self._meta_state = meta_state

    def create_snapshot(self, round_num: RoundNumber) -> StateSnapshot:
        """Create immutable snapshot of current state."""
        snapshot = StateSnapshot(
            round_num=round_num,
            node_states={vid: state.clone() for vid, state in self._node_states.items()},
            meta_state=self._meta_state,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def restore_snapshot(self, snapshot: StateSnapshot) -> None:
        """Restore state store to a previous snapshot."""
        self._node_states = {vid: state.clone() for vid, state in snapshot.node_states.items()}
        self._meta_state = snapshot.meta_state

    def get_snapshots(self) -> List[StateSnapshot]:
        """Get all saved snapshots."""
        return self._snapshots.copy()
