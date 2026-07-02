"""Execution context passed to algorithms for per-node execution.

Contains all information an algorithm needs to execute for a single node
in one round, without requiring access to centralized StateStore or MessageQueue.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.state.state_store import StateStore


@dataclass
class NodeContext:
    """Immutable execution context for single node in one round.

    Replaces passing entire StateStore/MessageQueue to algorithms.
    Provides exactly what a single node needs to execute.
    """

    node_id: int
    """ID of node executing this round."""

    state: Any
    """Current NodeState for this node (from StateStore or local)."""

    incoming_messages: List[Dict[str, Any]]
    """Messages received from neighbors in previous round."""

    graph: GraphManager
    """Read-only reference to shared graph (immutable)."""

    vector: Optional[CanonicalVector] = None
    """Read-only reference to parameter vector (immutable).

    Optional for Phase 3 backward compatibility - can be None during refactoring.
    """

    round_number: int = 0
    """Current round number for this execution."""

    state_store: Optional[StateStore] = None
    """Optional reference to centralized StateStore (for backward compatibility).

    Can be replaced by node-local StateStore in Phase 2.
    """

    def __post_init__(self) -> None:
        """Validate context on creation."""
        if self.node_id < 0:
            raise ValueError(f"node_id must be >= 0, got {self.node_id}")
        if self.round_number < 0:
            raise ValueError(f"round_number must be >= 0, got {self.round_number}")
        if not self.graph:
            raise ValueError("graph cannot be None")
