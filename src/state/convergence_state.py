"""Convergence voting state for distributed nodes."""

from typing import Dict


class ConvergenceState:
    """Tracks convergence voting state for a single node."""

    def __init__(self, node_id: int, total_nodes: int):
        """Initialize convergence state for a node."""
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.prev_weight = 0.0
        self.curr_weight = 0.0
        self.should_stop = False
        self.iteration = 0
        self.iteration_count = 0  # Alias for compatibility
        self.convergence_votes: Dict[int, bool] = {}  # node_id -> should_stop vote

    def update_weights(self, prev_weight: float, curr_weight: float) -> None:
        """Update matching weights for improvement calculation."""
        self.prev_weight = prev_weight
        self.curr_weight = curr_weight

    def decide_convergence(self, convergence_threshold: float) -> None:
        """Decide if should stop based on improvement threshold."""
        if self.prev_weight > 0:
            improvement = (self.curr_weight - self.prev_weight) / self.prev_weight
        else:
            improvement = 1.0 if self.curr_weight > 0 else 0.0

        self.should_stop = improvement < convergence_threshold
        self.iteration += 1
        self.iteration_count += 1  # Keep both in sync

    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self.iteration += 1
        self.iteration_count += 1

    def add_convergence_vote(self, node_id: int, should_stop: bool) -> None:
        """Record convergence vote from another node.

        Args:
            node_id: Node that cast the vote
            should_stop: Vote result (True = stop, False = continue)
        """
        self.convergence_votes[node_id] = should_stop

    def has_quorum_consensus(self, quorum_threshold: float) -> bool:
        """Check if quorum of nodes voted to stop.

        Args:
            quorum_threshold: Fraction of nodes needed (e.g., 0.5 for >50%)

        Returns:
            True if >= quorum_threshold fraction votes to stop
        """
        if not self.convergence_votes:
            return False

        stop_votes = sum(1 for v in self.convergence_votes.values() if v)
        total_votes = len(self.convergence_votes)

        if total_votes == 0:
            return False

        fraction_voting_stop = stop_votes / total_votes
        return fraction_voting_stop > quorum_threshold

    def reset_votes(self) -> None:
        """Clear convergence votes for next round."""
        self.convergence_votes.clear()
