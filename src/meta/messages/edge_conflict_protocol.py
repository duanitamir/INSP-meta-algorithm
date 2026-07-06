"""Phase 4: Edge conflict resolution protocol via endpoint voting.

Messages for nodes to vote on edge acceptance/rejection.
Implements distributed consensus for conflict resolution.
"""

from dataclasses import dataclass
from typing import Tuple
from src.communication.message import Message


@dataclass(frozen=True)
class EdgeProposalMessage(Message):
    """Proposal to match an edge (u, v).

    Sent by node u to node v: "I propose we match (u, v) with weight W"
    Node v then decides to accept or reject based on its local state.
    """

    @property
    def edge(self) -> Tuple[int, int]:
        """Get the proposed edge as (u, v) tuple."""
        if isinstance(self.payload, dict) and "edge" in self.payload:
            edge = self.payload["edge"]
            return tuple(edge) if isinstance(edge, (list, tuple)) else edge
        return (self.sender, self.recipient)

    @property
    def weight(self) -> float:
        """Get the edge weight."""
        if isinstance(self.payload, dict) and "weight" in self.payload:
            return float(self.payload["weight"])
        return 0.0


@dataclass(frozen=True)
class EdgeAcceptanceMessage(Message):
    """Vote to accept or reject a proposed edge.

    Sent by node v to node u in response to EdgeProposalMessage.
    vote=True: I accept matching this edge
    vote=False: I reject matching this edge
    """

    @property
    def edge(self) -> Tuple[int, int]:
        """Get the edge being voted on."""
        if isinstance(self.payload, dict) and "edge" in self.payload:
            edge = self.payload["edge"]
            return tuple(edge) if isinstance(edge, (list, tuple)) else edge
        return (self.sender, self.recipient)

    @property
    def vote(self) -> bool:
        """Get the vote (True = accept, False = reject)."""
        if isinstance(self.payload, dict) and "vote" in self.payload:
            return bool(self.payload["vote"])
        return False

    @property
    def reason(self) -> str:
        """Get optional reason for vote."""
        if isinstance(self.payload, dict) and "reason" in self.payload:
            return str(self.payload["reason"])
        return "No reason provided"


@dataclass(frozen=True)
class MatchConfirmationMessage(Message):
    """Both endpoints confirm they're matched to each other."""

    @property
    def edge(self) -> Tuple[int, int]:
        """Get the confirmed edge."""
        if isinstance(self.payload, dict) and "edge" in self.payload:
            edge = self.payload["edge"]
            return tuple(edge) if isinstance(edge, (list, tuple)) else edge
        return (self.sender, self.recipient)

    @property
    def status(self) -> str:
        """Get status: CONFIRMED or CANCELLED."""
        if isinstance(self.payload, dict) and "status" in self.payload:
            return str(self.payload["status"])
        return "CONFIRMED"


@dataclass(frozen=True)
class MatchCancellationMessage(Message):
    """Node was previously accepted but found better match."""

    @property
    def edge(self) -> Tuple[int, int]:
        """Get the cancelled edge."""
        if isinstance(self.payload, dict) and "edge" in self.payload:
            edge = self.payload["edge"]
            return tuple(edge) if isinstance(edge, (list, tuple)) else edge
        return (self.sender, self.recipient)

    @property
    def new_partner(self) -> int:
        """Get the node we're switching to."""
        if isinstance(self.payload, dict) and "new_partner" in self.payload:
            return int(self.payload["new_partner"])
        return -1
