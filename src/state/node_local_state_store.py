"""Node-local state store for Phase 2 of architecture refactor.

Each DistributedNode owns a local StateStore that only manages that node's state.
This replaces the centralized StateStore while maintaining the same interface
for backward compatibility.

This allows:
1. Node-local state management (Phase 2 goal)
2. Replacement of centralized StateStore
3. Easier distributed deployment
"""

from typing import Dict
from threading import RLock
from src.state.node_state import NodeState


class NodeLocalStateStore:
    """State store for a single node.

    Replaces centralized StateStore for this node.
    Has same interface but only manages one node's state.

    Thread-safe for concurrent access via locks.
    """

    def __init__(self, node_id: int):
        """Initialize node-local state store.

        Args:
            node_id: ID of the node this store manages
        """
        self.node_id = node_id
        self._state: Dict[int, NodeState] = {node_id: NodeState(node_id)}
        self._locks: Dict[int, RLock] = {node_id: RLock()}

    def get_node_state(self, node_id: int) -> NodeState:
        """Get state for a node.

        For this node-local store, only node_id matching this store's node_id
        returns valid state. Other node IDs return dummy state for compatibility.

        Args:
            node_id: ID of node to get state for

        Returns:
            NodeState for the node (or dummy state for other nodes)
        """
        if node_id == self.node_id:
            with self._locks[self.node_id]:
                return self._state[self.node_id]
        else:
            # Return dummy state for other nodes (for compatibility with old interfaces)
            return NodeState(node_id)

    def update_node_state(self, node_id: int, new_state: NodeState) -> None:
        """Update state for a node.

        For this node-local store, only updates are allowed for this node.
        Updates for other nodes are silently ignored.

        Args:
            node_id: ID of node to update
            new_state: New NodeState for the node

        Raises:
            ValueError: If attempting to update a different node (strict mode)
        """
        if node_id == self.node_id:
            with self._locks[self.node_id]:
                self._state[self.node_id] = new_state
        else:
            # Silently ignore updates for other nodes
            # (they have their own local state stores)
            pass

    def get_all_node_states(self) -> Dict[int, NodeState]:
        """Get all node states.

        For this node-local store, only returns this node's state.

        Returns:
            Dict with only this node's state
        """
        with self._locks[self.node_id]:
            return {self.node_id: self._state[self.node_id]}

    def reset(self) -> None:
        """Reset all state to initial values.

        For this node-local store, resets only this node.
        """
        with self._locks[self.node_id]:
            self._state[self.node_id] = NodeState(self.node_id)
