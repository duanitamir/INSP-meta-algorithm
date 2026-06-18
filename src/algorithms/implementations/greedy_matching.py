"""
Simplified Greedy Distributed Matching Algorithm

Protocol:
1. Each round, unmatched nodes bid to their best neighbor (by weight, then edge for tie break)
2. If two nodes bid to each other (mutual bid), they match immediately
3. Both become inactive
4. Continue until no unmatched nodes or no progress

Tie-breaking:
- Primary: Edge weight (higher is better)
- Secondary: Canonical edge (u, v) where u < v (lexicographic)
This prevents circular bidding chains with equal weights.

Edge Representation:
- Canonical: Edge(u, v) where u <= v
- Matched edges tracked as MatchedEdge(edge, weight)
"""
from asyncio import log
from typing import List, Tuple, Dict
import random
from src.algorithms.base import MatchingAlgorithm, AlgorithmMetadata
from src.state.state_store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber, Edge, MatchedEdge


class GreedyMatching(MatchingAlgorithm):
    """Simplified Greedy Distributed Matching with Mutual Bidding."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self._metadata = AlgorithmMetadata(
            name="Simplified Greedy Distributed Matching",
            description="Each round: bid to best neighbor. Mutual bids match immediately.",
            version="2.0.0",
            authors=["Distributed Systems"],
            references=["Greedy matching"],
            properties={
                "produces_maximal": True,
                "produces_maximum": False,
                "deterministic": True,
                "message_complexity": "O(m)",
                "max_rounds": 100,
            },
        )

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Get algorithm metadata."""
        return self._metadata

    def initialize_state(self, state_store: StateStore, graph) -> None:
        """Initialize state for all nodes."""
        for node_id in graph.vertices():
            state = state_store.get_node_state(node_id)
            state.set("matched_to", None)
            state.set("last_bid_to", None)  # Who we bid to this round
            state.set("last_bid_weight", None)  # Weight of our bid
            state.set("active", graph.degree(node_id) > 0)
            state.set("matched_edges", [])  # Track matched edges
            state_store.update_node_state(node_id, state)

    def node_behavior(
        self, node_id: int, node_state, messages: List[Message], context
    ) -> Tuple:
        """
        Greedy matching with mutual bidding.
        Each round: send BID to best neighbor, check for mutual bids to match.
        """
        new_state = node_state.clone()

        # If already matched, stay inactive
        if new_state.is_matched():
            new_state.set("active", False)
            return new_state, []

        out_messages: List[Message] = []
        # Get only active (unmatched) neighbors for bidding
        neighbors = list(context.graph.neighbors(node_id, state_store=context.state_store, filter_active=True))

        # No active neighbors -> become inactive
        if not neighbors:
            new_state.set("active", False)
            return new_state, out_messages

        best_neighbor = None
        best_weight = -1
        best_edge = None

        for neighbor in neighbors:
            weight = context.graph.get_edge_weight(node_id, neighbor)
            edge = Edge.from_nodes(node_id, neighbor)


            # Compare: (weight DESC, edge canonical)
            if best_neighbor is None or (weight > best_weight or
                (weight == best_weight and edge > best_edge)):
                best_weight = weight
                best_neighbor = neighbor
                best_edge = edge

        if best_neighbor is None:
            new_state.set("active", False)
            return (new_state, out_messages)

        # Send BID to best neighbor
        new_state.set("last_bid_to", best_neighbor)
        new_state.set("last_bid_weight", best_weight)
        new_state.set("active", True)

        out_messages.append(
            Message(
                sender=node_id,
                recipient=best_neighbor,
                payload={
                    "type": "BID",
                    "weight": best_weight,
                    "bidder_id": node_id,
                },
                round_num=context.round_num,
            )
        )

        # Check for mutual bids: did best_neighbor bid to us?
        received_bids = [msg for msg in messages if msg.payload.get("type") == "BID"]


        for bid in received_bids:
            bidder = bid.sender

            # Mutual bid: bidder bid to us AND we bid to bidder
            if bidder == best_neighbor:
                # MATCH: both nodes match each other
                new_state.set_matched_to(best_neighbor)
                new_state.set("active", False)
                new_state.set("last_bid_to", None)
                new_state.set("last_bid_weight", None)

                # Record matched edge
                matched_edge = MatchedEdge(
                    edge=Edge.from_nodes(node_id, best_neighbor),
                    weight=best_weight,
                )
                matched_edges = new_state.get("matched_edges", [])
                matched_edges.append(matched_edge)
                new_state.set("matched_edges", matched_edges)

                # Send confirmation that we matched
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=best_neighbor,
                        payload={"type": "MATCH_CONFIRMED"},
                        round_num=context.round_num,
                    )
                )
                return (new_state, out_messages)

        return (new_state, out_messages)

    def check_termination(
        self, state_store: StateStore, round_num: RoundNumber, messages_sent: int
    ) -> Tuple[bool, str | None]:
        """Check if algorithm has converged."""
        all_states = state_store.get_all_states()
        active_nodes = sum(1 for state in all_states.values() if state.get("active"))

        if active_nodes == 0:
            return True, "all_nodes_inactive"

        if messages_sent == 0 and round_num > RoundNumber(0):
            return True, "no_progress"

        max_rounds = (
            self.metadata.properties.get("max_rounds", 100)
            if self.metadata.properties
            else 100
        )
        if round_num > RoundNumber(max_rounds):
            return True, "max_rounds_exceeded"

        return False, None

    def extract_matching(self, state_store: StateStore, graph) -> Dict[int, int]:
        """Extract final matching from state store."""
        matching: Dict[int, int] = {}
        all_states = state_store.get_all_states()

        for node_id, state in all_states.items():
            matched_to = state.get_matched_to()
            if matched_to is not None:
                matching[node_id] = matched_to

        return matching

    def validate_matching(
        self, matching: Dict[int, int], graph
    ) -> Tuple[bool, str | None]:
        """Validate that matching is symmetric and valid."""
        # Check symmetry
        for u, v in matching.items():
            if v not in matching or matching[v] != u:
                return False, f"Asymmetric: {u}->{v} but {v}->? (not {u})"

            # Check edge exists
            if not graph.has_edge(u, v):
                return False, f"Edge {u}-{v} doesn't exist in graph"

        # Check no node matched twice
        matched_nodes = set()
        for u, v in matching.items():
            if u in matched_nodes:
                return False, f"Node {u} matched multiple times"
            matched_nodes.add(u)

        return True, None

    def is_maximal_matching(self, matching: Dict[int, int], graph) -> bool:
        """Check if matching is maximal (can't add more edges)."""
        matched_nodes = set(matching.keys())

        # For each unmatched edge, at least one endpoint must be matched
        for u in graph.vertices():
            if u not in matched_nodes:
                # u is unmatched, check all neighbors
                for v in graph.neighbors(u):
                    if v not in matched_nodes:
                        # Both u and v unmatched, edge (u,v) could be added
                        return False

        return True
