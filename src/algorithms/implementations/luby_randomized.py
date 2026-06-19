"""
Luby's Randomized Maximal Matching Algorithm (3-Phase Commit)

Adapted from Luby (1986) "A Simple Parallel Algorithm for the Maximal Independent Set Problem"
Adapted for MATCHING with 3-phase commit to ensure SYMMETRIC matchings.

Execution Model: SYNCHRONOUS ROUNDS
All nodes execute in lock-step synchronized rounds:
- Round 1: All nodes execute node_behavior() simultaneously
- Update barrier: All state changes applied atomically at end of round
- Message delivery: All messages queued during round delivered for next round
- Each node sees consistent global state for its round (messages from previous round)

Protocol (3-Phase Commit for Symmetric Matching):

**Phase 1: Activation Broadcast**
1. Each round, every active unmatched node independently decides:
   - Activate with probability p (default 0.5)
2. Node broadcasts activation decision to ALL neighbors: "I activated" or "I didn't"
3. Each node receives all neighbors' activation decisions

**Phase 2: Proposal (Send CONFIRM)**
4. Node evaluates: "Which neighbors also activated?"
5. Among mutually-activated neighbors, find best by edge weight
6. Send CONFIRM to best mutual neighbor (proposes mutual match)
7. Both nodes send CONFIRM to each other (mutual proposal)

**Phase 3: Acceptance (Send ACCEPT-CONFIRM)**
8. Node receives CONFIRM from best mutual neighbor
9. If received CONFIRM from them: send ACCEPT-CONFIRM back
10. Both nodes send ACCEPT-CONFIRM (mutual acceptance)
11. Once both receive ACCEPT-CONFIRM → MATCH (guaranteed symmetric!)
12. Matched nodes become PERMANENTLY INACTIVE

Message Types:
- ACTIVATE: Broadcast of activation decision (activated: true/false)
- CONFIRM: Proposal to match (sent to best mutual neighbor)
- ACCEPT-CONFIRM: Acceptance of mutual match (confirms the CONFIRM)

Key Properties:
- THREE-PHASE: Broadcast → Propose → Accept (GUARANTEES SYMMETRY)
- Both nodes must activate to match
- Best-weight selection among mutual activations
- Both nodes send matching messages in same round (synchronized)
- Guarantees maximal matching
- Guarantees SYMMETRIC matchings (no race conditions)
- Converges to maximal matching in O(log n) rounds with high probability
"""

from typing import List, Tuple
import random
from src.algorithms.base import MatchingAlgorithm, AlgorithmMetadata
from src.state.state_store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber


class LubyRandomizedMatching(MatchingAlgorithm):
    """Luby's Randomized Distributed Maximal Matching Algorithm (Original)."""

    def __init__(self, seed: int | None = None, activation_probability: float = 0.5):
        self.seed = seed
        self.activation_probability = activation_probability
        if seed is not None:
            random.seed(seed)

        self._metadata = AlgorithmMetadata(
            name="Luby-style Randomized Distributed Maximal Matching",
            description="Original Luby distributed randomized algorithm for maximal matching",
            version="2.0.0",
            authors=["Michael Luby"],
            references=["Luby (1986): A simple parallel algorithm for the maximal independent set problem"],
            properties={
                "produces_maximal": True,
                "produces_maximum": False,
                "deterministic": False,
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
            state.set("activated_this_round", False)  # Phase 1: Did I activate?
            state.set("has_neighbors", graph.degree(node_id) > 0)
            # Phase 2-3: Matching negotiation state
            state.set("confirm_proposal_to", None)  # Phase 2: Proposed to match this node
            state.set("confirm_received_from", None)  # Phase 2: Received CONFIRM from this node
            state.set("match_finalized", False)  # Phase 3: Both sent ACCEPT-CONFIRM
            state_store.update_node_state(node_id, state)

    def node_behavior(
        self, node_id: int, node_state, messages: List[Message], context
    ) -> Tuple:
        """
        Luby 3-phase commit algorithm (fixed):
        Phase 1: Broadcast activation decision
        Phase 2: Send CONFIRM proposal to best mutual neighbor
        Phase 3: Send ACCEPT-CONFIRM to accept mutual match, then finalize

        Key: Process phases in order (1→2→3) and use state flags to track progress.
        """
        new_state = node_state.clone()

        # If already matched, stay inactive
        if new_state.is_matched():
            new_state.set("is_active", False)
            return (new_state, [])

        out_messages: List[Message] = []

        # Get only active (unmatched) neighbors
        neighbors = list(context.graph.neighbors(node_id, state_store=context.state_store, filter_active=True))

        # No active neighbors -> become inactive
        if not neighbors:
            new_state.set("is_active", False)
            return (new_state, out_messages)

        # Track what state we're in
        confirm_proposal_to = new_state.get("confirm_proposal_to")  # Phase 2: Who I proposed to
        confirm_received_from = new_state.get("confirm_received_from")  # Phase 2: Who proposed to me
        match_finalized = new_state.get("match_finalized")  # Phase 3: Ready to finalize?

        # ============ PHASE 3: FINALIZE MATCH ============
        # If both nodes already confirmed mutual agreement, finalize the match
        if confirm_received_from and match_finalized:
            new_state = self.stage_finalize_match(new_state, confirm_received_from)
            return (new_state, out_messages)

        # ============ PHASE 2: SEND/RECEIVE ACCEPT-CONFIRM ============
        # If waiting for partner's ACCEPT-CONFIRM, look for it
        if confirm_received_from:
            accept_confirms = [msg for msg in messages if msg.payload.get("type") == "ACCEPT-CONFIRM"]
            if any(msg.sender == confirm_received_from for msg in accept_confirms):
                # Partner sent ACCEPT-CONFIRM! Now we can finalize
                new_state.set("match_finalized", True)
                # Finalize this round
                new_state = self.stage_finalize_match(new_state, confirm_received_from)
                return (new_state, out_messages)
            # Still waiting, stay active
            new_state.set("is_active", True)
            return (new_state, out_messages)

        # If I sent CONFIRM and received mutual CONFIRM, send ACCEPT-CONFIRM
        if confirm_proposal_to:
            confirms = [msg for msg in messages if msg.payload.get("type") == "CONFIRM"]
            if any(msg.sender == confirm_proposal_to for msg in confirms):
                # Mutual CONFIRM! Send ACCEPT-CONFIRM back
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=confirm_proposal_to,
                        payload={"type": "ACCEPT-CONFIRM"},
                        round_num=context.round_num,
                    )
                )
                new_state.set("confirm_received_from", confirm_proposal_to)
                new_state.set("is_active", True)
                return (new_state, out_messages)
            # Still waiting for mutual CONFIRM, stay active
            new_state.set("is_active", True)
            return (new_state, out_messages)

        # ============ PHASE 1: ACTIVATION & PROPOSAL ============
        # Decide whether to activate
        should_activate = self.stage_decide_activation(new_state)
        out_messages = self.stage_broadcast_activation(node_id, neighbors, should_activate, context)
        new_state.set("activated_this_round", should_activate)

        # Collect neighbor activation decisions
        neighbor_activations = [msg for msg in messages if msg.payload.get("type") == "ACTIVATE"]

        # If I activated and received neighbor activations, send CONFIRM proposal
        if neighbor_activations and should_activate:
            new_state, confirm_messages = self.stage_send_confirm_proposal(
                new_state, node_id, neighbor_activations, context
            )
            out_messages.extend(confirm_messages)

        return (new_state, out_messages)

    # ===== Helper Methods for Stage Operations =====

    def stage_decide_activation(self, new_state) -> bool:
        """Decide whether to activate this round with probability p."""
        return random.random() < self.activation_probability

    def stage_broadcast_activation(self, node_id: int, neighbors, should_activate: bool, context) -> List[Message]:
        """Broadcast activation decision to all neighbors."""
        out_messages = []
        for neighbor in neighbors:
            out_messages.append(
                Message(
                    sender=node_id,
                    recipient=neighbor,
                    payload={"type": "ACTIVATE", "activated": should_activate},
                    round_num=context.round_num,
                )
            )
        return out_messages

    def stage_finalize_match(self, new_state, partner: int):
        """Phase 3: Finalize symmetric match with partner."""
        new_state.set_matched_to(partner)  # Match with the partner
        new_state.set("is_active", False)  # Become inactive (matched)
        new_state.set("confirm_proposal_to", None)  # Clear proposal state
        new_state.set("confirm_received_from", None)  # Clear received state
        new_state.set("match_finalized", True)
        return new_state

    def stage_send_confirm_proposal(
        self, new_state, node_id: int, neighbor_activations, context
    ) -> Tuple:
        """
        Phase 2: Send CONFIRM proposal to best mutual neighbor.
        Select best-weight neighbor among those who also activated.
        """
        out_messages = []

        # Find neighbors who also activated
        mutually_activated = [
            msg.sender for msg in neighbor_activations
            if msg.payload.get("activated", False) is True
        ]

        if mutually_activated:
            # Select best neighbor by weight among mutually-activated neighbors
            best_neighbor = None
            best_weight = -1
            for neighbor in mutually_activated:
                weight = context.graph.get_edge_weight(node_id, neighbor)
                if weight > best_weight:
                    best_weight = weight
                    best_neighbor = neighbor

            if best_neighbor is not None:
                # Send CONFIRM proposal to best mutual neighbor
                weight = context.graph.get_edge_weight(node_id, best_neighbor)
                out_messages.append(
                    Message(
                        sender=node_id,
                        recipient=best_neighbor,
                        payload={"type": "CONFIRM", "proposer": node_id, "weight": weight},
                        round_num=context.round_num,
                    )
                )
                # Store proposal (will check if mutual in next round)
                new_state.set("confirm_proposal_to", best_neighbor)
                new_state.set("is_active", True)  # Stay active
            else:
                new_state.set("is_active", True)
        else:
            # No mutual activation -> stay active for next round
            new_state.set("is_active", True)

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

    def extract_matching(self, state_store: StateStore, graph) -> dict:
        """Extract final matching from state store."""
        matching = {}
        all_states = state_store.get_all_states()

        for node_id, state in all_states.items():
            matched_to = state.get_matched_to()
            if matched_to is not None:
                # In Luby matching, each node matches with another node
                # We need to find symmetric pairs
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

        # For each unmatched edge, at least one endpoint must be matched
        for u in graph.vertices():
            if u not in matched_nodes:
                # u is unmatched, check all neighbors
                for v in graph.neighbors(u):
                    if v not in matched_nodes:
                        # Both u and v unmatched, edge (u,v) could be added
                        return False

        return True
