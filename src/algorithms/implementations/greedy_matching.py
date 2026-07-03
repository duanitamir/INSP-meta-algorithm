"""
Greedy Distributed Matching Algorithm

Execution Model: SYNCHRONOUS ROUNDS
All nodes execute in lock-step synchronized rounds:
- Round 1: All nodes execute node_behavior() simultaneously with messages from Round 0
- Update barrier: All state changes applied atomically at end of round
- Message delivery: All messages queued during round delivered for next round
- Each node sees consistent global state for its round (messages from previous round)

Protocol (3-Phase to Ensure Symmetric Matching):

**Phase 1: PROPOSE**
1. Each round, unmatched nodes propose to their best neighbor (by weight, then edge for tie break)
2. Send PROPOSE message to best neighbor

**Phase 2: ACCEPT (Acknowledge Mutual Proposal)**
3. Node receives PROPOSEs from other nodes
4. If best neighbor also proposed to me: send ACCEPT to confirm mutual proposal (but don't match yet!)
5. Record the ACCEPT as "confirmed proposal pending"

**Phase 3: CONFIRM (Finalize Mutual Agreement)**
6. Receive ACCEPTs from other nodes
7. If partner sent ACCEPT: both nodes have confirmed mutual agreement
8. Now finalize match by setting matched_to

Why 3 phases?
- Prevents "half-matches" where one node thinks it matched but the other doesn't
- Ensures both nodes see the mutual PROPOSE before either changes state
- In synchronous rounds: Phase 1 sends, Phase 2 acks, Phase 3 finalizes
- Guarantees symmetric matching

Tie-breaking:
- Primary: Edge weight (higher is better)
- Secondary: Canonical edge (u, v) where u < v (lexicographic)
This prevents circular bidding chains with equal weights.

Edge Representation:
- Canonical: Edge(u, v) where u <= v
- Matched edges tracked as MatchedEdge(edge, weight)
"""
from typing import List, Tuple, Dict
from src.algorithms.base import MatchingAlgorithm, AlgorithmMetadata
from src.state.store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber, Edge, MatchedEdge


class GreedyMatching(MatchingAlgorithm):
    """Simplified Greedy Distributed Matching with Mutual Bidding."""

    def __init__(self):

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
            state.set("proposal_to", None)  # Phase 1: Who we proposed to
            state.set("proposal_weight", None)  # Phase 1: Weight of our proposal
            state.set("accept_from", None)  # Phase 2: Sent ACCEPT to this node (pending confirmation)
            state.set("accept_confirmed", False)  # Phase 3: Received ACCEPT back?
            state.set("active", graph.degree(node_id) > 0)
            state.set("matched_edges", [])  # Track matched edges
            state_store.update_node_state(node_id, state)

    def node_behavior(
        self, node_id: int, node_state, messages: List[Message], context
    ) -> Tuple:
        """
        Greedy distributed matching algorithm (3-phase):
        Phase 1: Each unmatched node sends PROPOSE to best neighbor
        Phase 2: If best neighbor also bid to us, send PROPOSE_ACK
        Phase 3: If we receive PROPOSE_ACK from partner, finalize match
        """
        new_state = node_state.clone()

        # If already matched, stay inactive (common to all algorithms)
        if new_state.is_matched():
            new_state.set("active", False)
            return new_state, []

        # Get active neighbors (common to all algorithms)
        neighbors = self.get_active_neighbors(node_id, context)

        # No active neighbors -> become inactive (common to all algorithms)
        if self.check_no_neighbors(neighbors):
            new_state.set("active", False)
            return new_state, []

        out_messages: List[Message] = []

        # PHASE 3: Check if partner sent PROPOSE_ACK (finalize match)
        bid_acks = [msg for msg in messages if msg.payload.get("type") == "PROPOSE_ACK"]
        bid_ack_from = new_state.get("accept_from")  # Who we sent PROPOSE_ACK to
        if bid_ack_from and any(msg.sender == bid_ack_from for msg in bid_acks):
            # Partner confirmed! Finalize match
            new_state = self.stage_finalize_match(new_state, node_id, bid_ack_from)
            return (new_state, [])

        # PHASE 2: Check for mutual PROPOSEs and send PROPOSE_ACK if mutual
        received_bids = [msg for msg in messages if msg.payload.get("type") == "PROPOSE"]

        # PHASE 1: Find best neighbor to bid to
        best_neighbor, best_weight, best_edge = self._find_best_neighbor(neighbors, node_id, context)

        if best_neighbor is None:
            new_state.set("active", False)
            return (new_state, [])

        # Send PROPOSE to best neighbor
        new_state.set("proposal_to", best_neighbor)
        new_state.set("proposal_weight", best_weight)
        new_state.set("active", True)
        out_messages.append(self.stage_send_bid(node_id, best_neighbor, best_weight, context))

        # Check if best neighbor also bid to us (mutual bid)
        if received_bids and any(bid.sender == best_neighbor for bid in received_bids):
            # Mutual PROPOSE detected! Send PROPOSE_ACK (don't match yet)
            new_state.set("accept_from", best_neighbor)
            new_state.set("accept_confirmed", False)
            out_messages.append(
                Message(
                    sender=node_id,
                    recipient=best_neighbor,
                    payload={"type": "PROPOSE_ACK"},
                    round_num=context.round_num,
                )
            )

        return (new_state, out_messages)

    # ===== Helper Methods for Stage Operations =====

    def stage_send_bid(self, node_id: int, best_neighbor: int, best_weight: float, context) -> Message:
        """Send PROPOSE to best neighbor."""
        return Message(
            sender=node_id,
            recipient=best_neighbor,
            payload={
                "type": "PROPOSE",
                "weight": best_weight,
                "bidder_id": node_id,
            },
            round_num=context.round_num,
        )

    def stage_finalize_match(self, new_state, node_id: int, partner: int):
        """Finalize match after both nodes sent PROPOSE_ACK."""
        weight = new_state.get("proposal_weight")
        new_state.set_matched_to(partner)
        new_state.set("active", False)
        new_state.set("proposal_to", None)
        new_state.set("proposal_weight", None)
        new_state.set("accept_from", None)
        new_state.set("accept_confirmed", True)

        # Record matched edge
        matched_edge = MatchedEdge(
            edge=Edge.from_nodes(node_id, partner),
            weight=weight,
        )
        matched_edges = new_state.get("matched_edges", [])
        matched_edges.append(matched_edge)
        new_state.set("matched_edges", matched_edges)

        return new_state

    # ===== Private Helper Methods =====

    def _find_best_neighbor(self, neighbors, node_id: int, context) -> Tuple[int | None, float, Edge | None]:
        """Find best neighbor by weight, then by canonical edge."""
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

        return best_neighbor, best_weight, best_edge

    def check_termination(
        self, state_store: StateStore, round_num: RoundNumber, messages_sent: int
    ) -> Tuple[bool, str | None]:
        """Check if algorithm has converged (uses common termination logic)."""
        max_rounds = (
            self.metadata.properties.get("max_rounds", 100)
            if self.metadata.properties
            else 100
        )
        return self.check_default_termination(state_store, round_num, messages_sent, max_rounds)

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
