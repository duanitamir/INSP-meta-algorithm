"""Per-node state tracking for distributed convergence detection."""

from typing import Any, Dict

from src.state.distributed_node_state import DistributedNodeState


class ConvergenceState(DistributedNodeState):
    """Tracks convergence state for each node in distributed termination.

    Maintains:
    - iteration_count: Number of iterations completed
    - prev_weight: Matching weight from previous iteration
    - curr_weight: Matching weight from current iteration
    - improvement: Relative improvement (curr - prev) / prev
    - convergence_votes: Votes received from other nodes about stopping
    - should_stop: This node's vote on convergence
    - total_nodes: Total number of nodes in network
    """

    def __init__(self, node_id: int, total_nodes: int = 0) -> None:
        """Initialize convergence state for a node.

        Args:
            node_id: Unique identifier for this node
            total_nodes: Total number of nodes in network (0 = unknown)
        """
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.iteration_count = 0
        self.prev_weight = 0.0
        self.curr_weight = 0.0
        self.improvement = 0.0
        self.convergence_votes: Dict[int, bool] = {}  # node_id -> should_stop
        self.should_stop = False

    def update_weights(self, prev_weight: float, curr_weight: float) -> None:
        """Update iteration weights and compute improvement.

        Args:
            prev_weight: Matching weight from previous iteration
            curr_weight: Matching weight from current iteration
        """
        self.prev_weight = prev_weight
        self.curr_weight = curr_weight

        # Compute improvement: (curr - prev) / prev
        if prev_weight > 0:
            self.improvement = (curr_weight - prev_weight) / prev_weight
        elif curr_weight > 0:
            # If prev was 0 but curr > 0, count as full improvement
            self.improvement = 1.0
        else:
            # Both are 0, no improvement
            self.improvement = 0.0

    def decide_convergence(self, convergence_threshold: float = 0.05) -> bool:
        """Decide if this node should vote to stop.

        Args:
            convergence_threshold: Minimum improvement to continue (default 5%)

        Returns:
            True if should stop, False if should continue
        """
        self.should_stop = self.improvement < convergence_threshold
        return self.should_stop

    def add_convergence_vote(self, node_id: int, vote: bool) -> None:
        """Record a convergence vote from another node.

        Args:
            node_id: Node ID casting the vote
            vote: True to stop, False to continue
        """
        self.convergence_votes[node_id] = vote

    def has_quorum_consensus(self, consensus_threshold: float = 0.5) -> bool:
        """Check if >threshold of nodes voted to stop.

        Args:
            consensus_threshold: Quorum threshold (default >50%)

        Returns:
            True if >threshold voted to stop
        """
        if not self.convergence_votes:
            return False

        stop_votes = sum(1 for vote in self.convergence_votes.values() if vote)
        total_votes = len(self.convergence_votes)

        return stop_votes / total_votes > consensus_threshold

    def reset_votes(self) -> None:
        """Clear convergence votes for next gossip round."""
        self.convergence_votes = {}

    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self.iteration_count += 1

    def reset(self) -> None:
        """Reset state for next round (clear votes but keep weights)."""
        self.reset_votes()

    def get_state_dict(self) -> Dict[str, Any]:
        """Return state snapshot for serialization.

        Returns:
            Dict containing all convergence detection state
        """
        return {
            "node_id": self.node_id,
            "iteration_count": self.iteration_count,
            "prev_weight": self.prev_weight,
            "curr_weight": self.curr_weight,
            "improvement": self.improvement,
            "should_stop": self.should_stop,
            "convergence_votes_count": len(self.convergence_votes),
        }
