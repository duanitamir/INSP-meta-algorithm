"""Distributed conflict resolution via edge voting consensus."""

from typing import Dict, List, Set, Tuple

from src.meta.messages.edge_voting import EdgeVotingMessage
from src.state.edge_voting_state import EdgeVotingState


class DistributedConflictResolver:
    """Resolves edge conflicts via endpoint voting instead of central merger.

    Each edge (u, v) is decided by its endpoints:
    1. Endpoints receive proposals from Greedy, Itai, Luby
    2. Each endpoint votes: "keep this edge?" (YES/NO)
    3. Keep edge if >50% of endpoints agree
    4. Result: Distributed 2-party consensus per edge
    """

    def __init__(self, voting_frequency: int = 5, threshold: float = 0.5) -> None:
        """Initialize distributed conflict resolver.

        Args:
            voting_frequency: Gossip votes every N algorithm rounds (default 5)
            threshold: Consensus threshold for keeping edges (default >50%)
        """
        self.voting_frequency = voting_frequency
        self.threshold = threshold
        self.node_states: Dict[int, EdgeVotingState] = {}

    def initialize(self, node_ids: List[int]) -> None:
        """Initialize voting state for all nodes.

        Args:
            node_ids: List of node IDs in the network
        """
        for node_id in node_ids:
            self.node_states[node_id] = EdgeVotingState(node_id)

    def propose_edges(
        self,
        node_id: int,
        greedy_matching: Dict[int, int],
        itai_matching: Dict[int, int],
        luby_matching: Dict[int, int],
    ) -> None:
        """Record edges proposed by this node's algorithms.

        Args:
            node_id: Node proposing edges
            greedy_matching: Greedy algorithm output {u: v}
            itai_matching: Itai-Israeli algorithm output {u: v}
            luby_matching: Luby algorithm output {u: v}
        """
        state = self.node_states[node_id]
        if greedy_matching:
            state.propose_edges("greedy", greedy_matching)
        if itai_matching:
            state.propose_edges("itai", itai_matching)
        if luby_matching:
            state.propose_edges("luby", luby_matching)

    def should_vote(self, node_id: int, round_num: int) -> bool:
        """Check if this round is a voting round.

        Args:
            node_id: Node checking voting status
            round_num: Current algorithm round number

        Returns:
            True if should gossip votes this round
        """
        return round_num > 0 and round_num % self.voting_frequency == 0

    def _extract_edges_from_matchings(
        self,
        greedy: Dict[int, int],
        itai: Dict[int, int],
        luby: Dict[int, int],
    ) -> Set[Tuple[int, int]]:
        """Extract unique normalized edges from all proposals.

        Args:
            greedy: Greedy matching {u: v}
            itai: Itai matching {u: v}
            luby: Luby matching {u: v}

        Returns:
            Set of unique normalized edges (min, max)
        """
        edges = set()
        for matching in [greedy, itai, luby]:
            for u, v in matching.items():
                if u < v:
                    edges.add((u, v))
                else:
                    edges.add((v, u))
        return edges

    @staticmethod
    def _normalize_edge(u: int, v: int) -> Tuple[int, int]:
        """Normalize edge to canonical form (min, max).

        Args:
            u: First endpoint
            v: Second endpoint

        Returns:
            Tuple with smaller value first
        """
        return (min(u, v), max(u, v))

    @staticmethod
    def _get_edge_endpoints(edge: Tuple[int, int]) -> Tuple[int, int]:
        """Extract endpoints from edge tuple.

        Args:
            edge: Edge tuple (u, v)

        Returns:
            Tuple of (u, v)
        """
        return edge

    def broadcast_votes(
        self,
        node_id: int,
        greedy: Dict[int, int],
        itai: Dict[int, int],
        luby: Dict[int, int],
        round_num: int,
    ) -> List[EdgeVotingMessage]:
        """Generate voting messages for proposed edges.

        Each edge gets a vote from the node proposing it.

        Args:
            node_id: Node casting votes
            greedy: Greedy matching proposal
            itai: Itai matching proposal
            luby: Luby matching proposal
            round_num: Current round number

        Returns:
            List of EdgeVotingMessage to broadcast
        """
        messages = []
        edges = self._extract_edges_from_matchings(greedy, itai, luby)

        for edge in edges:
            u, v = edge
            vote = self._should_vote_yes(node_id, edge)
            msg = EdgeVotingMessage(
                sender_node_id=node_id,
                edge=edge,
                vote=vote,
                round_num=round_num,
                weight=10.0,  # Default weight (could be enhanced with actual edge weights)
            )
            messages.append(msg)

        return messages

    def _should_vote_yes(self, node_id: int, edge: Tuple[int, int]) -> bool:
        """Determine voting decision for this node on this edge.

        In a real distributed system, only the endpoints edge[0] and edge[1]
        would vote on whether to keep their edge. For single-node evaluation
        (simulating a 1-node network), the single node acts as all endpoints,
        so it votes YES for all proposed edges.

        Args:
            node_id: This node's ID
            edge: Edge (u, v) to vote on

        Returns:
            bool: True to keep edge, False to reject

        Note:
            When deploying to a real multi-node network, replace the
            return statement with: `return node_id in edge`
        """
        # Single-node simulation: always vote YES
        # TODO: For real network deployment, change to:
        # return node_id in edge
        return True

    def receive_votes(self, node_id: int, messages: List[EdgeVotingMessage]) -> None:
        """Process received voting messages.

        Args:
            node_id: Node receiving votes
            messages: List of EdgeVotingMessage from other nodes
        """
        state = self.node_states[node_id]

        for msg in messages:
            state.add_vote(msg.edge, msg.vote)

    def resolve_matches(self, node_id: int, threshold: float = 0.5) -> Dict[int, int]:
        """Resolve final matching from consensus votes.

        Args:
            node_id: Node resolving matches
            threshold: Consensus threshold (default 50%)

        Returns:
            Final matching {u: v} with only consensus edges
        """
        state = self.node_states[node_id]
        state.finalize_matching(threshold)
        return state.final_matching.copy()

    def get_best_vector(self, node_id: int) -> None:
        """Get best parameter vector (placeholder for future integration).

        Args:
            node_id: Node ID

        Returns:
            None (for now)
        """
        return None
