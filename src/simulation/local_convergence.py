"""Node-local convergence voting and neighbor quorum tracking."""

import random
from collections.abc import Callable, Sequence

from src.communication.message import Message
from src.graph.local_graph import LocalGraph
from src.state.node import NodeState


class LocalConvergence:
    """Own a node's local stop vote, observed neighbor votes, and quorum rule."""

    def __init__(
        self,
        node_id: int,
        graph: LocalGraph,
        state: NodeState,
        send_message: Callable[[Message], None],
        threshold: float,
        quorum_threshold: float,
    ) -> None:
        self.node_id = node_id
        self.graph = graph
        self.state = state
        self._send_message = send_message
        self.threshold = threshold
        self.quorum_threshold = quorum_threshold
        self.vote: bool | None = None
        self.known_votes: dict[int, bool] = {}
        self.last_matching_weight = 0.0

    def reset(self, state: NodeState) -> None:
        """Reset state owned by a node execution without changing its configuration."""
        self.state = state
        self.vote = None
        self.known_votes.clear()
        self.last_matching_weight = 0.0

    def process_messages(self, messages: Sequence[Message]) -> None:
        """Record only addressed convergence votes from direct neighbors."""
        for message in messages:
            if message.sender not in self.graph.neighbors() or not isinstance(message.payload, dict):
                continue
            if message.payload.get("type") == "CONVERGENCE_VOTE":
                self.known_votes[message.sender] = bool(message.payload.get("vote", False))

    def decide(self, round_number: int) -> None:
        """Calculate this node's stop/continue vote from its local matching state."""
        if round_number == 0:
            self.vote = False
            return

        matched_edges = self.state.get("matched_edges", [])
        current_weight = sum(edge.weight for edge in matched_edges) if matched_edges else 0.0
        if self.last_matching_weight > 0:
            improvement = (current_weight - self.last_matching_weight) / self.last_matching_weight
        else:
            improvement = 1.0 if current_weight > 0 else 0.0

        self.last_matching_weight = current_weight
        self.vote = improvement < self.threshold

    def gossip(self, round_number: int) -> None:
        """Share this node's vote with a bounded random sample of neighbors."""
        neighbors = self.graph.neighbors()
        if not neighbors:
            return

        weight = sum(edge.weight for edge in self.state.get("matched_edges", []))
        payload = {
            "type": "CONVERGENCE_VOTE",
            "vote": self.vote if self.vote is not None else False,
            "should_stop": self.vote if self.vote is not None else False,
            "round": round_number,
            "weight": weight,
            "active": self.state.get("active", True),
            "matched": self.state.is_matched(),
        }
        for neighbor_id in random.sample(neighbors, min(3, len(neighbors))):
            self._send_message(
                Message(
                    sender=self.node_id,
                    recipient=neighbor_id,
                    payload=payload.copy(),
                    round_num=round_number,
                )
            )

    def should_stop(self) -> bool:
        """Return whether direct-neighbor votes satisfy this node's local quorum."""
        if self.state.get(NodeState.TENTATIVE_PARTNER) is not None or not self.known_votes:
            return False
        return (
            sum(1 for vote in self.known_votes.values() if vote) / len(self.known_votes)
            > self.quorum_threshold
        )
