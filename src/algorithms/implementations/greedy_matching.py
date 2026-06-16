"""
Greedy Distributed Matching Algorithm

Distributed greedy algorithm for weighted matching:
- Each node greedily matches with its highest-weight available neighbor
- Nodes compare bids and the winner is selected (higher ID wins ties)
- Once matched, node becomes inactive
- Converges in O(1) rounds with distributed decision-making
"""

from typing import List, Tuple
import random
from src.algorithms.base import MatchingAlgorithm, AlgorithmMetadata
from src.state.state_store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber


class GreedyMatching(MatchingAlgorithm):
    """Greedy Distributed Weighted Matching Algorithm."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self._metadata = AlgorithmMetadata(
            name="Greedy Distributed Matching",
            description="Distributed greedy algorithm that matches nodes with highest-weight neighbors",
            version="1.0.0",
            authors=["Distributed Systems"],
            references=["Standard greedy matching"],
            properties={
                "produces_maximal": True,
                "produces_maximum": False,
                "deterministic": False,
                "round_complexity": "O(log n)",
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
            state.set("current_bid", None)
            state.set("current_bid_partner", None)
            neighbors = list(graph.neighbors(node_id))
            state.set("neighbors", neighbors)
            state.set("active", len(neighbors) > 0)
            state_store.update_node_state(node_id, state)

    def node_behavior(
        self, node_id: int, node_state, messages: List[Message], context
    ) -> Tuple:
        """
        Greedy matching with symmetric confirmation.
        Protocol: BID -> ACCEPT -> CONFIRM to ensure both nodes match.
        """
        new_state = node_state.clone()

        # If already matched, stay inactive
        if new_state.is_matched():
            new_state.set("active", False)
            return (new_state, [])

        out_messages: List[Message] = []
        neighbors = new_state.get("neighbors", [])

        # No neighbors -> become inactive
        if not neighbors:
            new_state.set("active", False)
            return (new_state, out_messages)

        current_bid_partner = new_state.get("current_bid_partner")

        # Step 1: Process CONFIRM messages - if we receive CONFIRM, we're matched!
        confirms = [msg for msg in messages if msg.payload.get("type") == "CONFIRM"]
        for msg in confirms:
            if msg.sender == current_bid_partner:
                new_state.set_matched_to(current_bid_partner)
                new_state.set("active", False)
                new_state.set("current_bid", None)
                new_state.set("current_bid_partner", None)
                return (new_state, out_messages)

        # Step 2: Process ACCEPT messages - if partner accepted our bid, send CONFIRM
        acceptances = [
            msg for msg in messages if msg.payload.get("type") == "ACCEPT"
        ]
        for msg in acceptances:
            if msg.sender == current_bid_partner:
                # Match and send CONFIRM
                new_state.set_matched_to(current_bid_partner)
                new_state.set("active", False)
                new_state.set("current_bid", None)
                new_state.set("current_bid_partner", None)
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=msg.sender,
                        payload={"type": "CONFIRM"},
                        round_num=context.round_num,
                    )
                )
                return (new_state, out_messages)

        # Step 3: Process REJECT messages - clear bid if rejected
        rejections = [
            msg for msg in messages if msg.payload.get("type") == "REJECT"
        ]
        for msg in rejections:
            if msg.sender == current_bid_partner:
                new_state.set("current_bid", None)
                new_state.set("current_bid_partner", None)

        # Step 4: If we don't have an active bid, send one to best neighbor
        if not new_state.get("current_bid_partner"):
            best_neighbor = None
            best_weight = -1

            for neighbor in neighbors:
                weight = context.graph.get_edge_weight(node_id, neighbor)
                if weight > best_weight:
                    best_weight = weight
                    best_neighbor = neighbor

            if best_neighbor is not None:
                new_state.set("current_bid", best_weight)
                new_state.set("current_bid_partner", best_neighbor)
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
            else:
                new_state.set("active", False)
                return (new_state, out_messages)

        # Step 5: Process incoming BIDs - respond with ACCEPT/REJECT
        bids = [msg for msg in messages if msg.payload.get("type") == "BID"]
        if bids:
            # Sort by (weight DESC, bidder_id DESC) for deterministic tie-breaking
            bids_sorted = sorted(
                bids,
                key=lambda m: (m.payload["weight"], m.payload["bidder_id"]),
                reverse=True,
            )

            best_bid = bids_sorted[0]
            best_bidder = best_bid.sender
            best_weight_received = best_bid.payload["weight"]

            current_bid_weight = new_state.get("current_bid", -1)
            current_partner = new_state.get("current_bid_partner")

            # Decide if we should accept this best bid
            # Tie-breaking: on equal weights, higher node ID accepts
            # This prevents deadlock when nodes with equal-weight edges bid to each other
            should_accept = (
                current_partner is None
                or best_weight_received > current_bid_weight
                or (
                    best_weight_received >= current_bid_weight
                    and best_bidder > node_id
                )
            )

            if should_accept:
                # Reject previous partner if we had one
                if current_partner is not None:
                    out_messages.append(
                        Message(
                            sender=node_id,
                            recipient=current_partner,
                            payload={"type": "REJECT"},
                            round_num=context.round_num,
                        )
                    )

                # Accept best bid
                new_state.set("current_bid_partner", best_bidder)
                new_state.set("current_bid", best_weight_received)
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=best_bidder,
                        payload={"type": "ACCEPT"},
                        round_num=context.round_num,
                    )
                )

            # Reject all other bids
            for other_bid in bids_sorted[1:]:
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=other_bid.sender,
                        payload={"type": "REJECT"},
                        round_num=context.round_num,
                    )
                )

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

        max_rounds = self.metadata.properties.get("max_rounds", 100) if self.metadata.properties else 100
        if round_num > RoundNumber(max_rounds):
            return True, "max_rounds_exceeded"

        return False, None
