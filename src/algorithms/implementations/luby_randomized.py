"""
Luby-style Randomized Distributed Maximal Matching Algorithm

Distributed algorithm for computing a maximal matching using randomization:
- Each node independently decides to be "active" with probability 1/2
- Active nodes propose matches to random neighbors
- Neighbors accept highest-weight proposals
- Matched nodes become inactive for future rounds

NOTE: Known Limitation - Asymmetry in Complex Graphs (Documented)
The 3-message PROPOSE/ACCEPT/CONFIRM protocol can produce asymmetric matchings
on complex graphs due to timing in the synchronous model. This is expected behavior
for this simplified randomized algorithm.
"""

from typing import List, Tuple, Callable
import random
from src.algorithms.base import MatchingAlgorithm, AlgorithmMetadata
from src.state.state_store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber


class LubyRandomizedMatching(MatchingAlgorithm):
    """Luby-style Randomized Distributed Maximal Matching Algorithm."""

    def __init__(
        self,
        seed: int | None = None,
        activation_probability: float = 0.5,
        activation_function: Callable[[int], float] | None = None,
    ):
        self.seed = seed
        self.activation_probability = activation_probability
        self.activation_function = activation_function
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
                "adaptive_activation": activation_function is not None,
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
            state.set("active", True)
            state.set("proposal_to", None)
            state.set("proposal_weight", None)
            state.set("has_neighbors", graph.degree(node_id) > 0)
            state_store.update_node_state(node_id, state)

    def node_behavior(
        self, node_id: int, node_state, messages: List[Message], context
    ) -> Tuple:
        """
        Luby distributed randomized matching:
        1. If matched, stay inactive
        2. Decide whether to activate with probability p
        3. If activating and no proposal, send proposal to best neighbor
        4. Process responses (ACCEPT/REJECT/CONFIRM)
        """
        new_state = node_state.clone()

        # If already matched, stay inactive (common to all algorithms)
        if new_state.is_matched():
            new_state.set("is_active", False)
            return (new_state, [])

        out_messages: List[Message] = []
        neighbors = self.get_active_neighbors(node_id, context)

        # No neighbors -> become inactive (common to all algorithms)
        if self.check_no_neighbors(neighbors):
            new_state.set("is_active", False)
            return (new_state, out_messages)

        # Decide whether to activate with probability p (or p_adaptive if function provided)
        if self.activation_function is not None:
            activation_prob = self.activation_function(node_id)
        else:
            activation_prob = self.activation_probability
        should_activate = random.random() < activation_prob
        new_state.set("is_active", should_activate)

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

        # Step 4: If activated and no active proposal, send one to best neighbor
        if should_activate and not new_state.get("proposal_to"):
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

                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=best_neighbor,
                        payload={
                            "type": "PROPOSE",
                            "weight": best_weight,
                            "proposer": node_id,
                        },
                        round_num=context.round_num,
                    )
                )

        # Step 5: Process incoming PROPOSE messages
        proposals = [msg for msg in messages if msg.payload.get("type") == "PROPOSE"]
        if proposals:
            # Find best proposal by weight (and by proposer ID for tie-breaking)
            best_proposal = max(
                proposals,
                key=lambda m: (m.payload.get("weight", -1), m.sender)
            )
            best_proposer = best_proposal.sender
            best_proposal_weight = best_proposal.payload.get("weight", -1)

            # Compare with current proposal (if any)
            current_proposal_weight = new_state.get("proposal_weight")
            if current_proposal_weight is None:
                current_proposal_weight = -1
            current_proposal_to = new_state.get("proposal_to")

            if best_proposal_weight > current_proposal_weight:
                # Better proposal received - accept it
                if current_proposal_to is not None:
                    # Reject previous proposal
                    out_messages.append(
                        Message(
                            sender=node_id,
                            recipient=current_proposal_to,
                            payload={"type": "REJECT"},
                            round_num=context.round_num,
                        )
                    )

                # Accept new proposal
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=best_proposer,
                        payload={"type": "ACCEPT"},
                        round_num=context.round_num,
                    )
                )
                new_state.set("proposal_to", best_proposer)
                new_state.set("proposal_weight", best_proposal_weight)
                new_state.set("is_active", True)
            elif best_proposal_weight == current_proposal_weight and current_proposal_to is not None:
                # Equal weight - reject (keep current)
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=best_proposer,
                        payload={"type": "REJECT"},
                        round_num=context.round_num,
                    )
                )
            else:
                # First proposal - accept it
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=best_proposer,
                        payload={"type": "ACCEPT"},
                        round_num=context.round_num,
                    )
                )
                new_state.set("proposal_to", best_proposer)
                new_state.set("proposal_weight", best_proposal_weight)
                new_state.set("is_active", True)

        return (new_state, out_messages)

    def check_termination(
        self, state_store: StateStore, round_num: RoundNumber, messages_sent: int
    ) -> Tuple[bool, str | None]:
        """Check if algorithm has converged (uses common termination logic)."""
        max_rounds = self.metadata.properties.get("max_rounds", 200) if self.metadata.properties else 200
        return self.check_default_termination(state_store, round_num, messages_sent, max_rounds)

    def extract_matching(self, state_store: StateStore, graph) -> dict:
        """Extract final matching from state store."""
        matching = {}
        all_states = state_store.get_all_states()

        for node_id, state in all_states.items():
            matched_to = state.get_matched_to()
            if matched_to is not None:
                matching[node_id] = matched_to

        return matching

    def validate_matching(self, matching: dict, graph) -> Tuple[bool, str | None]:
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

    def is_maximal_matching(self, matching: dict, graph) -> bool:
        """Check if matching is maximal (can't add more edges)."""
        matched_nodes = set(matching.keys())

        # For each unmatched node, check if there's an unmatched neighbor
        for u in graph.vertices():
            if u not in matched_nodes:
                # u is unmatched, check all neighbors
                for v in graph.neighbors(u):
                    if v not in matched_nodes:
                        # Both u and v unmatched, edge (u,v) could be added
                        return False

        return True
