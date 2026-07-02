"""Semantic message wrapper for high-level communication.

Wraps base Message with semantic type information.
Enables NodeCommunicator to use named communication patterns
(bid, accept, reject, match, convergence_vote, etc.)
"""

from dataclasses import dataclass
from typing import Any
from src.utils.types import RoundNumber


@dataclass(frozen=True)
class SemanticMessage:
    """Message with semantic type information.

    Provides high-level semantic types for different communication patterns:
    - bid: Propose an edge
    - accept: Vote yes for an edge
    - reject: Vote no for an edge
    - match: Announce a confirmed match
    - convergence_vote: Vote on algorithm termination
    - state_update: Share local state
    - gossip_parameter: Share parameter information
    """

    sender: int
    recipient: int
    message_type: str
    payload: Any
    round_num: RoundNumber = 0
    message_id: int | None = None

    def __hash__(self) -> int:
        return hash(
            (self.sender, self.recipient, self.round_num, self.message_id, self.message_type)
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "payload": self.payload,
            "round_num": self.round_num,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticMessage":
        """Create from dictionary (deserialization).

        Args:
            data: Dictionary with message data

        Returns:
            SemanticMessage instance
        """
        return cls(
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=data["message_type"],
            payload=data["payload"],
            round_num=data.get("round_num", 0),
            message_id=data.get("message_id"),
        )
