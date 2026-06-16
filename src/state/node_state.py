from typing import Any, Dict, Callable, FrozenSet, Optional
import copy


class NodeState:
    """Mutable state for a single node."""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self._state: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Set a state variable."""
        self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state variable or default if not found."""
        return self._state.get(key, default)

    def update(self, key: str, fn: Callable[[Any], Any]) -> None:
        """Apply a function to update a state variable."""
        if key in self._state:
            self._state[key] = fn(self._state[key])

    def exists(self, key: str) -> bool:
        """Check if a state variable exists."""
        return key in self._state

    def delete(self, key: str) -> None:
        """Delete a state variable."""
        if key in self._state:
            del self._state[key]

    def keys(self) -> FrozenSet[str]:
        """Get all state variable keys."""
        return frozenset(self._state.keys())

    def clone(self) -> "NodeState":
        """Create a deep copy of this node state."""
        cloned = NodeState(self.node_id)
        cloned._state = copy.deepcopy(self._state)
        return cloned

    def set_matched_to(self, vertex_id: Optional[int]) -> None:
        """Set which vertex this node is matched to."""
        self._state["matched_to"] = vertex_id

    def get_matched_to(self) -> Optional[int]:
        """Get which vertex this node is matched to."""
        return self._state.get("matched_to")

    def is_matched(self) -> bool:
        """Check if node is currently matched."""
        return self._state.get("matched_to") is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return copy.deepcopy(self._state)

    def __repr__(self) -> str:
        return f"NodeState(node_id={self.node_id}, state={self._state})"
