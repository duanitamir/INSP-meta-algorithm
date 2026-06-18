"""
Luby-style Randomized Distributed Maximal Matching Algorithm

Distributed algorithm for computing a maximal matching using randomization:
- Each node independently decides to be "active" with probability 1/2
- Active nodes propose matches to random neighbors
- Neighbors accept highest-weight proposals
- Matched nodes become inactive for future rounds
- Converges to maximal matching in O(log n) rounds with high probability

NOTE: Known Limitation - Asymmetry in Complex Graphs
The 3-message PROPOSE/ACCEPT/CONFIRM protocol can produce asymmetric matchings
on complex graphs due to race conditions in the asynchronous distributed model.
This is a fundamental architectural issue with how proposals and confirmations
are processed at different logical times. Simple path graphs work correctly.
"""

from typing import List, Tuple
import random
from src.algorithms.base import MatchingAlgorithm, AlgorithmMetadata
from src.state.state_store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber


class LubyRandomizedMatching(MatchingAlgorithm):
    """Luby-style Randomized Distributed Maximal Matching Algorithm."""

    def __init__(self, seed: int | None = None, activation_probability: float = 0.5):
        self.seed = seed
        self.activation_probability = activation_probability
        if seed is not None:
            random.seed(seed)

        self._metadata = AlgorithmMetadata(
            name="Luby-style Randomized Distributed Maximal Matching",
            description="Distributed randomized algorithm for computing a maximal matching",
            version="1.0.0",
            authors=["Michael Luby"],
            references=["Luby (1986): A simple parallel algorithm for the maximal independent set problem"],
            properties={
                "produces_maximal": True,
                "produces_maximum": False,
                "deterministic": False,
                "round_complexity": "O(log n)",
                "message_complexity": "O(m log n)",
                "max_rounds": 200,
                "activation_probability": activation_probability,
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
            state.set("is_active", True)
            state.set("proposal_from", None)
            state.set("proposal_weight", None)
            state.set("has_neighbors", graph.degree(node_id) > 0)
            state_store.update_node_state(node_id, state)

    def node_behavior(
        self, node_id: int, node_state, messages: List[Message], context
    ) -> Tuple:
        """
        Luby randomized algorithm behavior:
        1. Inactive nodes stay inactive
        2. Active nodes with probability p propose to random neighbor
        3. Nodes accept best proposal, reject others
        4. Exchange PROPOSE/ACCEPT/CONFIRM messages
        """
        new_state = node_state.clone()

        # If already matched, stay inactive
        if new_state.is_matched():
            new_state.set("is_active", False)
            return (new_state, [])

        out_messages: List[Message] = []
        neighbors = list(context.graph.neighbors(node_id))

        # No active neighbors -> become inactive
        if not neighbors:
            new_state.set("is_active", False)
            return (new_state, out_messages)

        # Step 1: Process CONFIRM messages - if we receive CONFIRM, match
        confirms = [msg for msg in messages if msg.payload.get("type") == "CONFIRM"]
        for msg in confirms:
            if msg.sender == new_state.get("proposal_to"):
                new_state.set_matched_to(new_state.get("proposal_to"))
                new_state.set("is_active", False)
                new_state.set("proposal_weight", None)
                new_state.set("proposal_to", None)
                return (new_state, out_messages)

        # Step 2: Process ACCEPT messages - if partner accepted our proposal, send CONFIRM
        acceptances = [
            msg for msg in messages if msg.payload.get("type") == "ACCEPT"
        ]
        for msg in acceptances:
            if msg.sender == new_state.get("proposal_to"):
                # Match and send CONFIRM
                new_state.set_matched_to(new_state.get("proposal_to"))
                new_state.set("is_active", False)
                new_state.set("proposal_weight", None)
                new_state.set("proposal_to", None)
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=msg.sender,
                        payload={"type": "CONFIRM"},
                        round_num=context.round_num,
                    )
                )
                return (new_state, out_messages)

        # Step 3: Process REJECT messages - clear proposal if rejected
        rejections = [
            msg for msg in messages if msg.payload.get("type") == "REJECT"
        ]
        for msg in rejections:
            if msg.sender == new_state.get("proposal_to"):
                new_state.set("proposal_weight", None)
                new_state.set("proposal_to", None)

        # Step 4: If we don't have an active proposal, send one to best neighbor
        if not new_state.get("proposal_to"):
            best_neighbor = None
            best_weight = -1

            for neighbor in neighbors:
                weight = context.graph.get_edge_weight(node_id, neighbor)
                if weight > best_weight:
                    best_weight = weight
                    best_neighbor = neighbor

            if best_neighbor is not None:
                new_state.set("proposal_weight", best_weight)
                new_state.set("proposal_to", best_neighbor)
                new_state.set("is_active", True)

                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=best_neighbor,
                        payload={
                            "type": "PROPOSE",
                            "weight": best_weight,
                            "proposer_id": node_id,
                        },
                        round_num=context.round_num,
                    )
                )
            else:
                new_state.set("is_active", False)
                return (new_state, out_messages)

        # Step 5: Process incoming PROPOSE messages - respond with ACCEPT/REJECT
        proposes = [msg for msg in messages if msg.payload.get("type") == "PROPOSE"]
        if proposes:
            # Sort by (weight DESC, proposer_id DESC) for deterministic tie-breaking
            proposes_sorted = sorted(
                proposes,
                key=lambda m: (m.payload["weight"], m.payload["proposer_id"]),
                reverse=True,
            )

            best_propose = proposes_sorted[0]
            best_proposer = best_propose.sender
            best_weight_received = best_propose.payload["weight"]

            current_proposal_weight = new_state.get("proposal_weight", -1)
            current_proposer = new_state.get("proposal_from")

            # Decide if we should accept this best proposal
            # Tie-breaking: on equal weights, higher node ID accepts
            should_accept = (
                current_proposer is None
                or best_weight_received > current_proposal_weight
                or (
                    best_weight_received >= current_proposal_weight
                    and best_proposer > node_id
                )
            )

            if should_accept:
                # Reject previous proposer if we had one
                if current_proposer is not None:
                    out_messages.append(
                        Message(
                            sender=node_id,
                            recipient=current_proposer,
                            payload={"type": "REJECT"},
                            round_num=context.round_num,
                        )
                    )

                # Accept best proposal
                new_state.set("proposal_from", best_proposer)
                new_state.set("proposal_weight", best_weight_received)
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=best_proposer,
                        payload={"type": "ACCEPT"},
                        round_num=context.round_num,
                    )
                )

            # Reject all other proposals
            for other_propose in proposes_sorted[1:]:
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=other_propose.sender,
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
        active_nodes = sum(1 for state in all_states.values() if state.get("is_active"))

        # Check if all nodes are either matched or inactive
        if active_nodes == 0:
            return True, "all_nodes_inactive"

        # Check for convergence: no messages sent
        if messages_sent == 0 and round_num > RoundNumber(0):
            return True, "no_progress"

        # Check max rounds exceeded
        max_rounds = self.metadata.properties.get("max_rounds", 200) if self.metadata.properties else 200
        if round_num > RoundNumber(max_rounds):
            return True, "max_rounds_exceeded"

        return False, None
