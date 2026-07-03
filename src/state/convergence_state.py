"""Convergence voting state for distributed nodes."""


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

    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self.iteration += 1
