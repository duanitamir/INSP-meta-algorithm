"""Unified node state for all use cases (distributed and centralized)."""

from typing import Any, Dict, Callable, FrozenSet, Optional
import copy
from src.state.node_state_schema import NodeStateSchema


class NodeState:
    """Mutable state for a single node in distributed protocols.

    Provides generic state storage and specialized matching state.
    Works for both centralized and distributed use cases.
    """

    PROPOSAL_PENDING = "proposal_pending"
    TENTATIVE_PARTNER = "tentative_partner"
    PROPOSAL_DEADLINE = "proposal_deadline"
    PROPOSAL_ATTEMPT = "proposal_attempt"
    NEGOTIATION_EPOCH = "negotiation_epoch"
    TERMINAL_UNMATCHED = "terminal_unmatched"
    UNAVAILABLE_NEIGHBORS = "unavailable_neighbors"

    def __init__(self, node_id: int):
        """Initialize node state.

        Args:
            node_id: Unique identifier for this node
        """
        self.node_id = node_id
        self._state: NodeStateSchema = {}  # type: ignore
        # Local neighbors dict: {neighbor_id: {matched: bool, matched_to: int, ...}}
        # Each node maintains this independently and updates via messages
        self.neighbors: Dict[int, Dict[str, Any]] = {}

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

    # Neighbor management (distributed node state)

    def initialize_neighbors(self, neighbor_ids: list, edge_weights: Dict[int, float]) -> None:
        """Initialize neighbors dict with local knowledge.

        Args:
            neighbor_ids: List of neighbor node IDs
            edge_weights: Dict mapping neighbor_id -> edge weight
        """
        self.neighbors = {
            nid: {
                "matched": False,
                "matched_to": None,
                "weight": edge_weights.get(nid, 0.0),
            }
            for nid in neighbor_ids
        }

    def update_neighbor_status(
        self, neighbor_id: int, matched: bool, matched_to: Optional[int]
    ) -> None:
        """Update neighbor status when receiving a message.

        Called when receiving STATUS_UPDATE message from neighbor.

        Args:
            neighbor_id: ID of neighbor
            matched: Whether neighbor is matched
            matched_to: Who the neighbor is matched to (if matched)
        """
        if neighbor_id in self.neighbors:
            self.neighbors[neighbor_id]["matched"] = matched
            self.neighbors[neighbor_id]["matched_to"] = matched_to

    def get_unmatched_neighbors(self) -> list:
        """Get list of unmatched neighbor IDs.

        Returns:
            List of neighbor IDs that are not matched
        """
        return [nid for nid, info in self.neighbors.items() if not info.get("matched", False)]

    def get_matched_neighbors(self) -> list:
        """Get list of matched neighbor IDs.

        Returns:
            List of neighbor IDs that are matched
        """
        return [nid for nid, info in self.neighbors.items() if info.get("matched", False)]

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

    def mark_neighbor_unavailable(self, neighbor_id: int) -> None:
        """Remember that a directly connected neighbor has a final match."""
        unavailable = set(self.get(self.UNAVAILABLE_NEIGHBORS, set()))
        unavailable.add(neighbor_id)
        self.set(self.UNAVAILABLE_NEIGHBORS, frozenset(unavailable))

    def is_neighbor_eligible(self, neighbor_id: int) -> bool:
        """Return whether this endpoint can still start a match with a neighbor."""
        return neighbor_id not in self.get(self.UNAVAILABLE_NEIGHBORS, frozenset())

    def has_eligible_neighbor(self, neighbor_ids: list[int]) -> bool:
        """Return whether any locally known neighbor can still be proposed to."""
        return any(self.is_neighbor_eligible(neighbor_id) for neighbor_id in neighbor_ids)

    def mark_terminal_unmatched(self) -> None:
        """Record that this endpoint has no remaining local matching work."""
        if not self.is_matched():
            self.set(self.TERMINAL_UNMATCHED, True)

    def is_terminal_unmatched(self) -> bool:
        """Return whether this endpoint correctly finished without a partner."""
        return bool(self.get(self.TERMINAL_UNMATCHED, False))

    def is_terminal(self) -> bool:
        """Return whether this endpoint is matched or permanently unmatched."""
        return self.is_matched() or self.is_terminal_unmatched()

    def begin_tentative_match(
        self,
        partner_id: int,
        deadline: int,
        *,
        proposing: bool,
        attempt: int | None = None,
    ) -> int:
        """Record a local negotiation and return its monotonic attempt identity."""
        previous_epoch = int(self.get(self.NEGOTIATION_EPOCH, 0))
        attempt = previous_epoch + 1 if attempt is None else int(attempt)
        if attempt < 0:
            raise ValueError("Proposal attempt must be non-negative")
        self.set(self.NEGOTIATION_EPOCH, max(previous_epoch, attempt))
        self.set(self.TENTATIVE_PARTNER, partner_id)
        self.set(self.PROPOSAL_DEADLINE, deadline)
        self.set(self.PROPOSAL_PENDING, proposing)
        self.set(self.PROPOSAL_ATTEMPT, attempt)
        return attempt

    def clear_tentative_match(self) -> None:
        """Clear local negotiation state without changing a final match."""
        self.delete(self.TENTATIVE_PARTNER)
        self.delete(self.PROPOSAL_DEADLINE)
        self.delete(self.PROPOSAL_PENDING)
        self.delete(self.PROPOSAL_ATTEMPT)

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
        cloned.neighbors = copy.deepcopy(self.neighbors)
        return cloned

    def __repr__(self) -> str:
        return f"NodeState(node_id={self.node_id}, state={self._state})"
