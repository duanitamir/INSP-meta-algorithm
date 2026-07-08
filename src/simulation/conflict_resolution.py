"""Phase 1: Distributed Conflict Resolution via Endpoint Voting.

Implements resolve_conflicts() and conflict_solution() methods.
Nodes resolve conflicts locally without central merger.
"""

from typing import Dict, Tuple
from enum import Enum


class MatchState(Enum):
    """Node's current matching state."""
    UNMATCHED = 0
    TENTATIVELY_MATCHED = 1
    CONFIRMED_MATCHED = 2


class Phase1ConflictResolution:
    """Mixin for DistributedNode providing Phase 1 conflict resolution."""

    def __init__(self):
        """Initialize conflict resolution state."""
        self.pending_proposals: Dict[int, float] = {}  # neighbor_id -> weight
        self.current_match: int | None = None
        self.match_state = MatchState.UNMATCHED
        self.best_weight = -1.0

    def receive_edge_proposal(self, proposer_id: int, weight: float) -> None:
        """Receive proposal from another node.

        Args:
            proposer_id: Node ID making the proposal
            weight: Weight of the proposed edge
        """
        self.pending_proposals[proposer_id] = weight

    def resolve_conflicts(self) -> int | None:
        """Find best proposal locally.

        Returns:
            ID of best proposer, or None if no proposals
        """
        if not self.pending_proposals:
            return None

        # Sort by: (weight DESC, then node_id ASC for determinism)
        best_proposer = max(
            self.pending_proposals.items(),
            key=lambda item: (item[1], -item[0])
        )
        return best_proposer[0]

    def conflict_solution(self) -> None:
        """Resolve all pending proposals and send responses.

        Sends EdgeAcceptanceMessage to best proposer (accept)
        and to all others (reject).
        """
        best_proposer = self.resolve_conflicts()

        if best_proposer is None:
            # No proposals, nothing to do
            self.pending_proposals.clear()
            return

        # Send acceptance to best proposer
        self.send_edge_acceptance_message(
            recipient=best_proposer,
            status="ACCEPTED",
            reason="Best proposal"
        )

        # Send rejection to all others
        for proposer_id in self.pending_proposals.keys():
            if proposer_id != best_proposer:
                self.send_edge_acceptance_message(
                    recipient=proposer_id,
                    status="REJECTED",
                    reason="Better proposal chosen"
                )

        # Clear pending proposals for next round
        self.pending_proposals.clear()

    def send_edge_acceptance_message(
        self,
        recipient: int,
        status: str,
        reason: str
    ) -> None:
        """Send acceptance/rejection to proposer.

        Args:
            recipient: Node ID of proposer
            status: "ACCEPTED" or "REJECTED"
            reason: Human-readable reason
        """
        from src.meta.messages.edge_conflict_protocol import EdgeAcceptanceMessage

        msg = EdgeAcceptanceMessage(
            sender=self.id,
            recipient=recipient,
            payload={
                "edge": (self.id, recipient),
                "vote": (status == "ACCEPTED"),
                "reason": reason
            }
        )

        self.communicator.send_message(msg)

    def handle_edge_acceptance(self, msg) -> None:
        """Handle response to our proposal.

        Args:
            msg: EdgeAcceptanceMessage
        """
        if msg.vote:  # msg.vote == True means ACCEPTED
            # Proposer accepted us
            self.current_match = msg.sender
            self.match_state = MatchState.TENTATIVELY_MATCHED
            self.best_weight = self.graph.get_edge_weight(self.id, msg.sender)
        else:
            # Proposer rejected us - we can try next neighbor next round
            pass
