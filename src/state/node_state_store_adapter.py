"""Adapter to make NodeState look like StateStore for algorithm compatibility."""

from src.state.node_state import NodeState
from src.state.state_store import MetaState
from src.utils.types import RoundNumber


class NodeStateStoreAdapter:
    """
    Adapter that wraps a single NodeState to look like a StateStore.

    This allows algorithms to work seamlessly in distributed mode
    where each node has its own state, without requiring algorithm rewrites.

    Algorithms call methods like state_store.get_node_state(node_id),
    but in distributed mode, we only care about THIS node's state.
    """

    def __init__(self, node_state: NodeState, node_id: int):
        """Initialize adapter.

        Args:
            node_state: The NodeState for this node
            node_id: The node ID
        """
        self.node_state = node_state
        self.node_id = node_id
        self._meta_state = MetaState(round_num=RoundNumber(0))

    def get_node_state(self, node_id: int) -> NodeState:
        """Get state for a node.

        In distributed mode, only THIS node's state is available.
        For other nodes, return empty state (algorithms won't see neighbors' state).

        Args:
            node_id: Node identifier

        Returns:
            NodeState clone for this node, or empty NodeState for others
        """
        if node_id == self.node_id:
            return self.node_state.clone()
        else:
            # Return empty state for other nodes (algorithms handle this gracefully)
            return NodeState(node_id)

    def get_all_states(self):
        """Get all node states.

        In distributed mode, only this node's state exists.

        Returns:
            Dict with only this node's state
        """
        return {self.node_id: self.node_state.clone()}

    def get_meta_state(self) -> MetaState:
        """Get global simulation state.

        Returns:
            MetaState
        """
        return self._meta_state

    def update_node_state(self, node_id: int, state: NodeState) -> None:
        """Update state for a node.

        In distributed mode, only THIS node can be updated.
        Updates for other nodes are no-op (ignored).

        Args:
            node_id: Node identifier
            state: New NodeState
        """
        if node_id == self.node_id:
            self.node_state = state.clone()
        # Silently ignore updates for other nodes

    def update_all_states(self, states):
        """Update all node states.

        In distributed mode, only this node's state can be updated.

        Args:
            states: Dict of node_id -> NodeState

        Raises:
            ValueError if trying to update other nodes
        """
        if self.node_id not in states:
            return
        if len(states) > 1:
            raise ValueError(
                f"Distributed mode: can only update this node's state, got {len(states)} nodes"
            )
        self.node_state = states[self.node_id].clone()

    def update_meta_state(self, meta_state: MetaState) -> None:
        """Update global simulation state.

        Args:
            meta_state: New MetaState
        """
        self._meta_state = meta_state

    def create_snapshot(self, round_num):
        """Create snapshot of current state.

        Not really supported in distributed mode, but we provide a minimal implementation.

        Args:
            round_num: Round number

        Returns:
            Snapshot-like dict
        """
        return {
            "round_num": round_num,
            "node_state": self.node_state.clone(),
            "meta_state": self._meta_state
        }

    def restore_snapshot(self, snapshot):
        """Restore from snapshot.

        Args:
            snapshot: Snapshot dict
        """
        self.node_state = snapshot["node_state"].clone()
        self._meta_state = snapshot["meta_state"]

    def get_snapshots(self):
        """Get all snapshots.

        Not supported in distributed mode.

        Returns:
            Empty list
        """
        return []
