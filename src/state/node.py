"""Unified node state for all use cases (distributed and centralized)."""

from typing import Any, Dict, Callable, FrozenSet, Optional
import copy


class NodeState:
    """Mutable state for a single node in distributed protocols.

    Provides generic state storage and specialized matching state.
    Works for both centralized and distributed use cases.
    """

    def __init__(self, node_id: int):
        """Initialize node state.

        Args:
            node_id: Unique identifier for this node
        """
        self.node_id = node_id
        self._state: Dict[str, Any] = {}

    # Generic state operations

    def set(self, key: str, value: Any) -> None:
        """Set a state variable.

        Args:
            key: State key
            value: State value
        """
        self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state variable or default if not found.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value or default
        """
        return self._state.get(key, default)

    def update(self, key: str, fn: Callable[[Any], Any]) -> None:
        """Apply a function to update a state variable.

        Args:
            key: State key
            fn: Function to apply (takes current value, returns new value)
        """
        if key in self._state:
            self._state[key] = fn(self._state[key])

    def exists(self, key: str) -> bool:
        """Check if a state variable exists.

        Args:
            key: State key

        Returns:
            True if key exists, False otherwise
        """
        return key in self._state

    def delete(self, key: str) -> None:
        """Delete a state variable.

        Args:
            key: State key to delete
        """
        if key in self._state:
            del self._state[key]

    def keys(self) -> FrozenSet[str]:
        """Get all state variable keys.

        Returns:
            Frozen set of all state keys
        """
        return frozenset(self._state.keys())

    # Matching-specific operations

    def set_matched_to(self, vertex_id: Optional[int]) -> None:
        """Set which vertex this node is matched to.

        Args:
            vertex_id: ID of vertex this node is matched to (None if unmatched)
        """
        self._state["matched_to"] = vertex_id

    def get_matched_to(self) -> Optional[int]:
        """Get which vertex this node is matched to.

        Returns:
            Vertex ID this node is matched to, or None if unmatched
        """
        return self._state.get("matched_to")

    def is_matched(self) -> bool:
        """Check if node is currently matched.

        Returns:
            True if node is matched, False otherwise
        """
        return self._state.get("matched_to") is not None

    # Serialization & Utilities

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary.

        Returns:
            Deep copy of internal state as dictionary
        """
        return copy.deepcopy(self._state)

    def clone(self) -> "NodeState":
        """Create a deep copy of this node state.

        Returns:
            New NodeState with copied state
        """
        cloned = NodeState(self.node_id)
        cloned._state = copy.deepcopy(self._state)
        return cloned

    def __repr__(self) -> str:
        return f"NodeState(node_id={self.node_id}, state={self._state})"
