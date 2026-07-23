"""Distributed convergence detection via autonomous node voting."""

from typing import Dict, List

from src.meta.messages.gossip_message import GossipMessage
from src.state.convergence_state import ConvergenceState


class DistributedConvergenceDetector:
    """Detects network-wide convergence via quorum-based voting.

    Each node independently:
    1. Computes improvement from previous iteration
    2. Votes: "Should we stop?" based on convergence threshold
    3. Gossips vote to random neighbors every N rounds
    4. Tallies votes from network
    5. Signals termination if >50% vote to stop

    Result: Autonomous convergence detection without central orchestrator.
    """

    def __init__(
        self,
        convergence_threshold: float = 0.05,
        quorum_threshold: float = 0.5,
        gossip_frequency: int = 5,
        max_iterations: int = 100,
    ) -> None:
        """Initialize distributed convergence detector.

        Args:
            convergence_threshold: Min improvement to continue (default 5%)
            quorum_threshold: Network consensus threshold for stopping (default >50%)
            gossip_frequency: Gossip votes every N algorithm rounds (default 5)
            max_iterations: Timeout safety: max iterations allowed (default 100)
        """
        self.convergence_threshold = convergence_threshold
        self.quorum_threshold = quorum_threshold
        self.gossip_frequency = gossip_frequency
        self.max_iterations = max_iterations
        self.node_states: Dict[int, ConvergenceState] = {}

    def initialize(self, node_ids: List[int]) -> None:
        """Initialize convergence state for all nodes.

        Args:
            node_ids: List of node IDs in the network
        """
        total_nodes = len(node_ids)
        for node_id in node_ids:
            self.node_states[node_id] = ConvergenceState(node_id, total_nodes)

    def should_gossip(self, node_id: int, round_num: int) -> bool:
        """Check if this round is a gossip round.

        Args:
            node_id: Node checking gossip status
            round_num: Current algorithm round number

        Returns:
            True if should gossip this round
        """
        return round_num > 0 and round_num % self.gossip_frequency == 0

    def decide_convergence(self, node_id: int, prev_weight: float, curr_weight: float) -> None:
        """Decide if node should vote to stop based on improvement.

        Args:
            node_id: Node making decision
            prev_weight: Matching weight from previous iteration
            curr_weight: Matching weight from current iteration
        """
        state = self.node_states[node_id]
        state.update_weights(prev_weight, curr_weight)
        state.decide_convergence(self.convergence_threshold)

    def create_convergence_message(self, node_id: int, round_num: int) -> GossipMessage:
        """Create a convergence gossip message from node's decision.

        Args:
            node_id: Node creating message
            round_num: Current algorithm round

        Returns:
            GossipMessage with subtype="convergence" and this node's vote
        """
        state = self.node_states[node_id]
        return GossipMessage.convergence_gossip(
            sender_node_id=node_id,
            should_stop=state.should_stop,
            round_num=round_num,
            weight=state.curr_weight,
        )

    def broadcast_convergence_votes(
        self, node_id: int, round_num: int
    ) -> List[GossipMessage]:
        """Generate convergence messages for broadcasting.

        Args:
            node_id: Node broadcasting
            round_num: Current round number

        Returns:
            List of GossipMessage with subtype="convergence" to broadcast to neighbors
        """
        msg = self.create_convergence_message(node_id, round_num)
        return [msg]

    def receive_convergence_votes(
        self, node_id: int, messages: List[GossipMessage]
    ) -> None:
        """Process received convergence messages.

        Args:
            node_id: Node receiving votes
            messages: List of GossipMessage with subtype="convergence" from other nodes
        """
        state = self.node_states[node_id]

        for msg in messages:
            state.add_convergence_vote(msg.sender_node_id, msg.should_stop)

    def check_network_convergence(self, node_id: int) -> bool:
        """Check if network has reached convergence quorum.

        Args:
            node_id: Node checking convergence

        Returns:
            True if >quorum_threshold of nodes voted to stop
        """
        state = self.node_states[node_id]
        converged = state.has_quorum_consensus(self.quorum_threshold)
        return converged

    def check_timeout(self, node_id: int) -> bool:
        """Check if node has exceeded maximum iterations (safety timeout).

        Args:
            node_id: Node checking timeout

        Returns:
            True if node reached max_iterations
        """
        state = self.node_states[node_id]
        return state.iteration_count >= self.max_iterations

    def should_terminate(self, node_id: int) -> bool:
        """Check if node should terminate (convergence OR timeout).

        Args:
            node_id: Node checking termination condition

        Returns:
            True if network converged OR max iterations reached
        """
        return self.check_network_convergence(node_id) or self.check_timeout(node_id)

    def reset_convergence_votes(self, node_id: int) -> None:
        """Reset convergence votes for new gossip round.

        Args:
            node_id: Node resetting votes
        """
        self.node_states[node_id].reset_votes()
