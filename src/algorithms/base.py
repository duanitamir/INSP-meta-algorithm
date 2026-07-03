from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from src.state.store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber


@dataclass
class AlgorithmMetadata:
    """Metadata about an algorithm."""

    name: str
    description: str
    version: str
    authors: List[str]
    references: List[str]
    properties: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class MatchingAlgorithm(ABC):
    """
    Abstract base class for distributed matching algorithms.

    Provides common functionality for:
    - Matching extraction and validation
    - Termination checking
    - Helper methods for message filtering and neighbor selection
    """

    # ===== ABSTRACT METHODS (Must implement) =====

    @abstractmethod
    def initialize_state(
        self,
        state_store: StateStore,
        graph,
    ) -> None:
        """
        Initialize node states for this algorithm.

        Args:
            state_store: State repository to initialize
            graph: Graph structure
        """
        ...

    @abstractmethod
    def node_behavior(
        self,
        node_id: int,
        node_state,
        messages: List[Message],
        context,
    ) -> Tuple:
        """
        Compute behavior for a single node in one round.

        Args:
            node_id: The node being executed
            node_state: Current state of this node
            messages: Messages received by this node
            context: Execution context (graph, round info)

        Returns:
            Tuple of (updated_node_state, messages_to_send)
        """
        ...

    @abstractmethod
    def check_termination(
        self,
        state_store: StateStore,
        round_num: RoundNumber,
        messages_sent: int,
    ) -> Tuple[bool, str | None]:
        """
        Check if algorithm should terminate.

        Args:
            state_store: Current state
            round_num: Current round number
            messages_sent: Messages sent this round

        Returns:
            Tuple of (should_terminate, termination_reason)
        """
        ...

    @property
    @abstractmethod
    def metadata(self) -> AlgorithmMetadata:
        """Get algorithm metadata."""
        ...

    def extract_matching(
        self,
        state_store: StateStore,
        graph,
    ) -> Dict[int, int]:
        """
        Extract final matching from state store.

        Returns:
            Dict mapping matched vertices (v1 → v2, v2 → v1)
        """
        matching = {}
        all_states = state_store.get_all_states()
        for node_id, state in all_states.items():
            matched_to = state.get_matched_to()
            if matched_to is not None:
                matching[node_id] = matched_to
        return matching

    def validate_matching(
        self,
        matching: Dict[int, int],
        graph,
    ) -> Tuple[bool, str | None]:
        """
        Validate that matching is valid.

        Returns:
            Tuple of (is_valid, error_message)
        """
        seen = set()

        for u, v in matching.items():
            if u in seen:
                return False, f"Vertex {u} matched multiple times"
            seen.add(u)

            if v not in matching or matching[v] != u:
                return False, f"Matching not symmetric: {u}-{v}"

            if not graph.has_edge(u, v):
                return False, f"Edge {u}-{v} not in graph"

        return True, None

    def is_maximal_matching(
        self,
        matching: Dict[int, int],
        graph,
    ) -> bool:
        """Check if matching is maximal (no unmatched edge has both endpoints unmatched)."""
        matched_vertices = set(matching.keys())

        for u in graph.vertices():
            if u in matched_vertices:
                continue
            for v in graph.neighbors(u):
                if v not in matched_vertices:
                    return False

        return True

    # ===== COMMON HELPER METHODS =====

    def get_active_neighbors(self, node_id: int, context) -> List[int]:
        """
        Get unmatched neighbors of a node (common to all algorithms).

        Returns:
            List of neighbor IDs that are currently unmatched
        """
        return list(
            context.graph.neighbors(node_id, state_store=context.state_store, filter_active=True)
        )

    def check_no_neighbors(self, neighbors: List[int]) -> bool:
        """Check if node has no active neighbors."""
        return len(neighbors) == 0

    def check_default_termination(
        self,
        state_store: StateStore,
        round_num: RoundNumber,
        messages_sent: int,
        max_rounds: int,
    ) -> Tuple[bool, str | None]:
        """
        Check termination using standard criteria (common to all algorithms).

        Terminates if:
        1. All nodes are inactive (matched or no neighbors)
        2. No progress (no messages sent after round 0)
        3. Max rounds exceeded

        Args:
            state_store: Current node states
            round_num: Current round number
            messages_sent: Messages sent this round
            max_rounds: Maximum allowed rounds

        Returns:
            Tuple of (should_terminate, reason)
        """
        all_states = state_store.get_all_states()

        # Check if all nodes are inactive or matched
        active_nodes = sum(1 for state in all_states.values() if state.get("active"))
        if active_nodes == 0:
            return True, "all_nodes_inactive"

        # Check for convergence: no messages sent
        if messages_sent == 0 and round_num > RoundNumber(0):
            return True, "no_progress"

        # Check max rounds exceeded
        if round_num > RoundNumber(max_rounds):
            return True, "max_rounds_exceeded"

        return False, None
