"""Shared termination checking logic for convergence detection."""


class TerminationChecker:
    """Utility for checking algorithm termination conditions."""

    def __init__(self, convergence_threshold: float = 0.05, quorum_threshold: float = 0.5):
        """Initialize termination checker.

        Args:
            convergence_threshold: Min improvement ratio to continue
            quorum_threshold: Min fraction of nodes voting to stop
        """
        self.convergence_threshold = convergence_threshold
        self.quorum_threshold = quorum_threshold

    def compute_improvement(self, current_weight: float, last_weight: float) -> float:
        """Compute improvement ratio between consecutive weights."""
        if last_weight > 0:
            return (current_weight - last_weight) / last_weight
        return 1.0 if current_weight > 0 else 0.0

    def should_continue(self, improvement: float) -> bool:
        """Check if improvement is sufficient to continue."""
        return improvement >= self.convergence_threshold

    def has_quorum_to_stop(self, convergence_votes: dict, total_nodes: int) -> bool:
        """Check if quorum of nodes voted to stop.

        Args:
            convergence_votes: Dict mapping node_id -> vote (True=stop, False=continue)
            total_nodes: Total number of nodes in system

        Returns:
            True if >= quorum_threshold fraction voted to stop
        """
        if not convergence_votes or total_nodes == 0:
            return False
        stop_votes = sum(1 for v in convergence_votes.values() if v)
        return stop_votes >= total_nodes * self.quorum_threshold
