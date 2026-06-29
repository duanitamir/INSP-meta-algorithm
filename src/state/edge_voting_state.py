"""Per-node state tracking for distributed edge voting."""

from typing import Any, Dict, Set, Tuple

from src.state.distributed_node_state import DistributedNodeState


class EdgeVotingState(DistributedNodeState):
    """Tracks voting state for each node in distributed conflict resolution.

    Maintains:
    - proposed_edges: All edges proposed locally by algorithms
    - received_votes: Votes received from endpoints on contested edges
    - final_matching: Final matching after consensus
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
        for edge in self.received_votes.keys():
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
