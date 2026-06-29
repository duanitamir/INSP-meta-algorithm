"""Immutable message format for distributed convergence detection."""

from dataclasses import dataclass

from src.meta.messages.base import DistributedMessage


@dataclass(frozen=True)
class ConvergenceGossipMessage(DistributedMessage):
    """Message for gossiping convergence votes in distributed termination.

    Each node votes independently on whether the network has converged.
    Validates common distributed message fields.

    Attributes:
        sender_node_id: Node ID casting the convergence vote
        should_stop: True if node votes to stop, False to continue
        round_num: Algorithm round number
        weight: Current matching weight (for diagnostics)
    """

    sender_node_id: int
    should_stop: bool
    round_num: int
    weight: float

    def __post_init__(self) -> None:
        """Validate message on creation."""
        self.validate_base_fields(self.round_num, self.weight)
