"""Immutable message format for distributed edge voting."""

from dataclasses import dataclass

from src.meta.messages.base import DistributedMessage


@dataclass(frozen=True)
class EdgeVotingMessage(DistributedMessage):
    """Message for voting on contested edges in distributed conflict resolution.

    Each endpoint (u, v) of an edge votes independently on whether to include
    the edge in the final matching. Validates common distributed message fields.

    Attributes:
        sender_node_id: Node ID casting the vote
        edge: Tuple (u, v) being voted on (normalized: u <= v)
        vote: True to keep edge, False to reject
        round_num: Algorithm round number
        weight: Edge weight (for tie-breaking if needed)
    """

    sender_node_id: int
    edge: tuple[int, int]
    vote: bool
    round_num: int
    weight: float

    def __post_init__(self) -> None:
        """Validate message on creation."""
        self.validate_base_fields(self.round_num, self.weight)

        if len(self.edge) != 2:
            raise ValueError(f"edge must be 2-tuple, got {self.edge}")
