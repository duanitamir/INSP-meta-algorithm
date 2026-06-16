from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from src.state.state_store import StateStore
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
    """Abstract base class for matching algorithms."""

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
