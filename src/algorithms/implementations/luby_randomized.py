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

from typing import List, Tuple, Dict
import random
from src.algorithms.base import MatchingAlgorithm, AlgorithmMetadata
from src.state.store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber


class LubyRandomizedMatching(MatchingAlgorithm):
    """Luby-style Randomized Distributed Maximal Matching Algorithm."""

    # Unified parameter definition: {param_name -> {min, max, default, type, description}}
    PARAMETERS = {
        "base_probability": {
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "type": "number",
            "description": "Base activation probability",
        },
        "coeff_degree": {
            "min": -1.0,
            "max": 1.0,
            "default": 0.1,
            "type": "number",
            "description": "Coefficient for degree influence on activation",
        },
        "coeff_neighbors_unmatched": {
            "min": -1.0,
            "max": 1.0,
            "default": 0.1,
            "type": "number",
            "description": "Coefficient for unmatched neighbors influence",
        },
        "coeff_clustering": {
            "min": -1.0,
            "max": 1.0,
            "default": 0.1,
            "type": "number",
            "description": "Coefficient for clustering influence",
        },
        "coeff_matched": {
            "min": -1.0,
            "max": 1.0,
            "default": 0.1,
            "type": "number",
            "description": "Coefficient for matched nodes influence",
        },
        "coeff_round": {
            "min": -1.0,
            "max": 1.0,
            "default": 0.1,
            "type": "number",
            "description": "Coefficient for round number influence",
        },
        "coeff_weight": {
            "min": -1.0,
            "max": 1.0,
            "default": 0.1,
            "type": "number",
            "description": "Coefficient for edge weight influence",
        },
        "max_rounds": {
            "min": 5,
            "max": 100,
            "default": 100,
            "type": "integer",
            "description": "Maximum execution rounds",
        }
    }

    # PARAMETER_DEFINITION for registry compatibility (auto-generated from PARAMETERS)
    PARAMETER_DEFINITION = {
        "name": "luby",
        "display_name": "Luby Randomized Matching",
        "parameters": {
            param: (p["min"], p["max"], (lambda pm=param, pp=p: random.uniform(pp["min"], pp["max"]) if pp["type"] == "number" else random.randint(int(pp["min"]), int(pp["max"]))))
            for param, p in PARAMETERS.items()
        }
    }

    # PARAMETER_DEFAULTS for initialization (auto-generated from PARAMETERS)
    PARAMETER_DEFAULTS = {param: p["default"] for param, p in PARAMETERS.items()}

    # PARAMETER_SCHEMA for validation (auto-generated from PARAMETERS)
    PARAMETER_SCHEMA = {
        "type": "object",
        "properties": {
            param: {
                "type": p["type"],
                "minimum": p["min"],
                "maximum": p["max"],
                "description": p["description"],
            }
            for param, p in PARAMETERS.items()
        },
        "required": list(PARAMETERS.keys()),
    }

    def __init__(self, parameters: Dict = None):
        """Initialize Luby Randomized Matching algorithm.

        Args:
            parameters: Optional parameter dict. Missing parameters use defaults from PARAMETER_DEFAULTS.
        """
        # Merge provided parameters with defaults
        self.parameters = {**self.PARAMETER_DEFAULTS}
        if parameters:
            self.parameters.update(parameters)

        # Extract specific parameters
        self.seed = self.parameters.get("seed")
        self.activation_probability = self.parameters.get("base_probability", 0.5)
        self.activation_function = None

        if self.seed is not None:
            random.seed(self.seed)

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
                "max_rounds": self.parameters.get("max_rounds", 100),
                "activation_probability": self.activation_probability,
                "adaptive_activation": False,
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

    def propose_to_neighbors(self, node_id: int, neighbors: List[int], context) -> Dict[int, float]:
        """
        Luby-style algorithm proposal: which neighbors to propose to?

        Returns proposals only to neighbors (local scope), not entire graph.

        Args:
            node_id: This node's ID
            neighbors: List of direct neighbors only
            context: Algorithm context with graph

        Returns:
            Dict[neighbor_id, weight] - proposals to send (can be empty or single)
        """
        if not neighbors or len(neighbors) == 0:
            return {}

        # Luby-style: probabilistically decide whether to activate
        if self.activation_function is not None:
            activation_prob = self.activation_function(node_id)
        else:
            activation_prob = self.activation_probability

        if random.random() >= activation_prob:
            # Not activating this round
            return {}

        # Find best neighbor
        best_neighbor = None
        best_weight = -1

        for neighbor in neighbors:
            weight = context.graph.get_edge_weight(node_id, neighbor)
            if weight > best_weight:
                best_weight = weight
                best_neighbor = neighbor

        if best_neighbor is None:
            return {}

        # Luby-style: propose to best neighbor only (with randomization)
        return {best_neighbor: best_weight}

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
        """Check if algorithm has converged."""
        max_rounds = self.metadata.properties.get("max_rounds", 200) if self.metadata.properties else 200
        return self.check_default_termination(state_store, round_num, messages_sent, max_rounds)
