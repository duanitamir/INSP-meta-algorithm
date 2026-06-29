"""Base class for per-node state in distributed protocols."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class DistributedNodeState(ABC):
    """Base class for node-specific state in distributed algorithms.

    Each node maintains local state for its role in the distributed protocol.
    Common functionality includes initialization with node_id and reset logic.
    """

    def __init__(self, node_id: int) -> None:
        """Initialize node state.

        Args:
            node_id: Unique identifier for this node
        """
        self.node_id = node_id

    @abstractmethod
    def reset(self) -> None:
        """Reset state for next iteration/round.

        Called between protocol rounds to clear transient state while
        preserving historical information needed across rounds.
        """
        pass

    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """Return state snapshot for serialization.

        Returns:
            Dict with all relevant state (for checkpointing, debugging, etc.)
        """
        pass
