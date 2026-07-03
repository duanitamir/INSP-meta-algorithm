"""Distributed voting for edge consensus and convergence detection."""

from typing import Any, Dict, Set, Tuple


class EdgeVoting:
    """Tracks voting state for distributed edge conflict resolution.

    Maintains:
    - proposed_edges: All edges proposed locally by algorithms
    - received_votes: Votes received from endpoints on contested edges
    - final_matching: Final matching after consensus
    - vote_tally: Count of votes per edge
    """

    def __init__(self, node_id: int) -> None:
        """Initialize voting state for a node.

        Args:
            node_id: Unique identifier for this node
        """
        self.node_id = node_id
        self.proposed_edges: Dict[str, Dict[int, int]] = {}  # algo_name -> {u: v}
        self.received_votes: Dict[Tuple[int, int], list[bool]] = {}  # edge -> [votes]
        self.final_matching: Dict[int, int] = {}
        self.vote_tally: Dict[Tuple[int, int], int] = {}  # edge -> count of votes

    def propose_edges(self, algorithm: str, matching: Dict[int, int]) -> None:
        """Record edges proposed by an algorithm.

        Args:
            algorithm: Algorithm name (e.g., "greedy", "itai", "luby")
            matching: Matching dict {u: v}
        """
        self.proposed_edges[algorithm] = matching.copy()

    def add_vote(self, edge: Tuple[int, int], vote: bool) -> None:
        """Record a vote on an edge.

        Args:
            edge: Normalized edge tuple (min, max)
            vote: True to keep edge, False to reject
        """
        if edge not in self.received_votes:
            self.received_votes[edge] = []
            self.vote_tally[edge] = 0

        self.received_votes[edge].append(vote)
        self.vote_tally[edge] += 1

    def has_consensus(self, edge: Tuple[int, int], threshold: float = 0.5) -> bool:
        """Check if edge has >threshold consensus.

        Args:
            edge: Normalized edge tuple
            threshold: Consensus threshold (default 50%)

        Returns:
            True if edge has consensus, False otherwise
        """
        if edge not in self.received_votes:
            return False

        votes = self.received_votes[edge]
        if not votes:
            return False

        yes_count = sum(1 for v in votes if v)
        return yes_count / len(votes) > threshold

    def get_consensus_edges(self, threshold: float = 0.5) -> Set[Tuple[int, int]]:
        """Get all edges with consensus.

        Args:
            threshold: Consensus threshold

        Returns:
            Set of edges that have consensus
        """
        consensus = set()
        for edge in list(self.received_votes.keys()):
            votes = self.received_votes[edge]
            if votes and (sum(1 for v in votes if v) / len(votes)) > threshold:
                consensus.add(edge)
        return consensus

    def finalize_matching(self, threshold: float = 0.5) -> None:
        """Build final matching from consensus edges.

        Args:
            threshold: Consensus threshold
        """
        self.final_matching = {}
        consensus_edges = self.get_consensus_edges(threshold)

        for u, v in consensus_edges:
            self.final_matching[u] = v
            self.final_matching[v] = u

    def reset_votes(self) -> None:
        """Clear received votes for next iteration."""
        self.received_votes = {}

    def reset(self) -> None:
        """Reset state for next round (clear transient voting data)."""
        self.reset_votes()

    def get_state_dict(self) -> Dict[str, Any]:
        """Return state snapshot for serialization.

        Returns:
            Dict containing all voting state
        """
        return {
            "node_id": self.node_id,
            "proposed_edges": self.proposed_edges.copy(),
            "received_votes": {str(k): v for k, v in self.received_votes.items()},
            "final_matching": self.final_matching.copy(),
            "vote_tally": {str(k): v for k, v in self.vote_tally.items()},
        }


class ConvergenceDetection:
    """Tracks convergence state for distributed termination detection.

    Maintains:
    - iteration_count: Number of iterations completed
    - prev_weight: Matching weight from previous iteration
    - curr_weight: Matching weight from current iteration
    - improvement: Relative improvement (curr - prev) / prev
    - convergence_votes: Votes received from other nodes about stopping
    - should_stop: This node's vote on convergence
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

        if prev_weight > 0:
            self.improvement = (curr_weight - prev_weight) / prev_weight
        elif curr_weight > 0:
            self.improvement = 1.0
        else:
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
