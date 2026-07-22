"""
Itai-Israeli Distributed Maximal Matching Algorithm

Execution Model: SYNCHRONOUS ROUNDS
All nodes execute in lock-step synchronized rounds:
- Round 1: All nodes execute node_behavior() simultaneously with messages from Round 0
- Update barrier: All state changes applied atomically at end of round
- Message delivery: All messages queued during round delivered for next round
- Each node sees consistent global state for its round (messages from previous round)

Protocol (Three-Message Exchange):
1. Each unmatched node sends PROPOSE to the best (highest weight) unmatched neighbor
2. Receiver accepts the BEST incoming proposal (highest weight, ties broken by sender ID)
   - If already negotiating with someone, compares new proposal to current one
   - If new proposal is better: silently drops current partner and accepts the new one (no rejection sent)
   - If new proposal is worse: IGNORES it and continues with current negotiation
3. Receiver sends ACCEPT back to proposer
4. Proposer receives ACCEPT, sends CONFIRM to finalize match - both now matched and inactive
5. If no response after timeout_rounds, reset negotiation and retry with best unmatched neighbor
   - Note: May be same or different neighbor depending on current graph state (other nodes may have matched)
6. Continue until no active nodes or no progress

Proposal Switching Logic:
- A receiver may switch partners mid-negotiation if it receives a better proposal
- Switch is SILENT: no explicit rejection sent to abandoned partner (timeout will reset them)
- Comparison: weight-based (higher is better), then by sender ID for ties
- This allows nodes to adapt to higher-quality opportunities dynamically
- Abandoned partners eventually timeout and seek new neighbors

Message Types:
- PROPOSE: unmatched node proposes to neighbor (weight included)
- ACCEPT: receiver agrees to match
- CONFIRM: proposer confirms match established
"""

from typing import List, Tuple, Dict
from src.algorithms.base import MatchingAlgorithm, AlgorithmMetadata
from src.state.store import StateStore
from src.communication.message import Message
from src.utils.types import RoundNumber


class ItaiIsraeliMaximalMatching(MatchingAlgorithm):
    """Itai-Israeli Distributed Maximal Matching Algorithm."""

    # Unified parameter definition: {param_name -> {min, max, default, type, description}}
    PARAMETERS = {
        "timeout_rounds": {
            "min": 1,
            "max": 20,
            "default": 5,
            "type": "integer",
            "description": "Maximum rounds to wait before resetting negotiation",
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
        "name": "itai",
        "display_name": "Itai-Israeli Maximal Matching",
        "parameters": {
            param: (p["min"], p["max"], (lambda pm=param, pp=p: __import__("random").randint(pp["min"], pp["max"]) if pp["type"] == "integer" else __import__("random").uniform(pp["min"], pp["max"])))
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
        """Initialize Itai-Israeli Matching algorithm.

        Args:
            parameters: Optional parameter dict. Missing parameters use defaults from PARAMETER_DEFAULTS.
        """
        # Merge provided parameters with defaults
        self.parameters = {**self.PARAMETER_DEFAULTS}
        if parameters:
            self.parameters.update(parameters)

        self.timeout_rounds = self.parameters.get("timeout_rounds", 5)

        self._metadata = AlgorithmMetadata(
            name="Itai-Israeli Distributed Maximal Matching",
            description="Distributed algorithm for computing a maximal matching",
            version="1.0.0",
            authors=["Adi Itai", "Michael Rodeh"],
            references=["Itai & Rodeh (1978)"],
            properties={
                "produces_maximal": True,
                "produces_maximum": False,
                "deterministic": True,
                "max_rounds": self.parameters.get("max_rounds", 100),
                "timeout_rounds": self.timeout_rounds,
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
            state.set("status", "free")  # free, proposing, or matched
            state.set("negotiation_partner", None)
            state.set("negotiation_stage", None)
            state.set("stage_round", 0)
            state.set("active", graph.degree(node_id) > 0)
            state_store.update_node_state(node_id, state)

    def node_behavior(self, node_id: int, node_state, messages: List[Message], context) -> Tuple:
        new_state = node_state.clone()

        # If already matched, stay inactive (common to all algorithms)
        if new_state.is_matched():
            new_state.set("active", False)
            return (new_state, [])

        # Get active neighbors (common to all algorithms)
        neighbors = self.get_active_neighbors(node_id, context)
        if self.check_no_neighbors(neighbors):
            new_state.set("active", False)
            return (new_state, [])

        # Parse incoming messages by type
        confirms = [msg.sender for msg in messages if msg.payload.get("type") == "CONFIRM"]
        accepts = [msg.sender for msg in messages if msg.payload.get("type") == "ACCEPT"]

        # Get current negotiation state
        partner = new_state.get("negotiation_partner")
        stage = new_state.get("negotiation_stage")

        # Stages 1 & 2: Process CONFIRM and ACCEPT messages (if we have a partner)
        if partner:
            out_messages = []
            # Process CONFIRM messages (final matching stage)
            if partner in confirms and stage == "accepted":
                new_state = self.stage_confirm(new_state, partner)
            # Process ACCEPT messages (send CONFIRM) - only if no CONFIRM received
            elif partner in accepts and stage == "proposed":
                out_messages = self.stage_accept(partner, node_id, context)
                # Move to "accepted" stage (proposer waiting for nothing - can match now)
                new_state = self._match_nodes(new_state, partner)
                return (new_state, out_messages)
            # If in "accepting" stage, check for CONFIRM from proposer
            elif partner in confirms and stage == "accepting":
                new_state = self.stage_confirm(new_state, partner)

            # If matched, return early (don't process other stages)
            if new_state.is_matched():
                return (new_state, out_messages)

        # Check for timeout and reset if needed
        new_state, partner = self.stage_timeout(new_state, partner)

        # Process incoming PROPOSE from other nodes
        proposes = [msg for msg in messages if msg.payload.get("type") == "PROPOSE"]
        if proposes:
            best = self._find_best_propose(proposes)
            best_weight = best.payload.get("weight", 0)
            current_weight = new_state.get("last_bid_weight", -1) if partner else -1

            if not partner or best_weight > current_weight or (best_weight == current_weight and best.sender > partner):
                out_messages = self.stage_accept_propose(best.sender, node_id, context)
                # Start negotiation (NOT immediate matching) - will match only after CONFIRM
                new_state = self._start_negotiation(new_state, best.sender, "accepting")
                new_state.set("last_bid_weight", best_weight)  # Track weight of current negotiation
                return (new_state, out_messages)

        # Generate new PROPOSE if not currently negotiating
        # After timeout, retry with best neighbor from CURRENT graph state
        # (other nodes may have matched since we last tried)
        if not partner:
            out_messages = self.stage_generate_propose(neighbors, node_id, context)
            recipient = out_messages[0].recipient if out_messages else None

            if recipient is not None:
                # Valid proposal generated
                new_state = self._start_negotiation(new_state, recipient, "proposed")
                return (new_state, out_messages)
            else:
                # No valid neighbors (all have higher IDs, will accept their proposals)
                new_state.set("active", True)
                return (new_state, [])
        else:
            # Already negotiating, increment timeout counter
            stage_round = new_state.get("stage_round", 0)
            new_state.set("stage_round", stage_round + 1)
            new_state.set("active", True)
            return (new_state, [])

    # ===== Helper Methods for Stage Operations =====

    def stage_confirm(self, new_state, partner):
        """Match confirmed by partner - finalize the match."""
        return self._match_nodes(new_state, partner)

    def stage_accept(self, partner, node_id, context):
        """Send CONFIRM to partner after receiving ACCEPT."""
        return [Message(
            sender=node_id,
            recipient=partner,
            payload={"type": "CONFIRM"},
            round_num=context.round_num,
        )]

    def stage_timeout(self, new_state, partner):
        """Check if negotiation has timed out and reset if needed."""
        stage_round = new_state.get("stage_round", 0)

        if partner and stage_round >= self.timeout_rounds:
            new_state.set("negotiation_partner", None)
            new_state.set("negotiation_stage", None)
            new_state.set("stage_round", 0)
            partner = None

        return (new_state, partner)

    def stage_accept_propose(self, proposer_id, node_id, context):
        """Accept incoming PROPOSE and send ACCEPT."""
        return [Message(
            sender=node_id,
            recipient=proposer_id,
            payload={"type": "ACCEPT"},
            round_num=context.round_num,
        )]

    def propose_to_neighbors(self, node_id: int, neighbors: List[int], context) -> Dict[int, float]:
        """
        Itai-Israeli algorithm proposal: which neighbors to propose to?

        Returns proposals only to neighbors (local scope), not entire graph.
        CRITICAL: Only propose to neighbors with ID < node_id to prevent mutual proposals.

        Args:
            node_id: This node's ID
            neighbors: List of direct neighbors only
            context: Algorithm context with graph

        Returns:
            Dict[neighbor_id, weight] - proposals to send (can be empty or single)
        """
        if not neighbors or len(neighbors) == 0:
            return {}

        best_neighbor = None
        best_weight = -1

        # Only consider neighbors with smaller IDs to prevent mutual proposals
        for neighbor in neighbors:
            if neighbor < node_id:  # KEY FIX: directional proposal
                weight = context.graph.get_edge_weight(node_id, neighbor)
                if weight > best_weight:
                    best_weight = weight
                    best_neighbor = neighbor

        if best_neighbor is None:
            return {}

        # Itai-Israeli: propose to best neighbor only (with lower ID)
        return {best_neighbor: best_weight}

    def stage_generate_propose(self, neighbors, node_id, context):
        """Generate PROPOSE to best (highest weight) neighbor.

        CRITICAL FIX: Only propose to neighbors with ID < node_id to prevent mutual proposals.
        This prevents deadlocks where both nodes end up in "accepting" stage simultaneously.
        In Itai-Israeli, only one direction of a potential pair should initiate.
        """
        best_neighbor = None
        best_weight = -1

        # Only consider neighbors with smaller IDs to prevent mutual proposals
        for neighbor in neighbors:
            if neighbor < node_id:  # KEY FIX: directional proposal
                weight = context.graph.get_edge_weight(node_id, neighbor)
                if weight > best_weight:
                    best_weight = weight
                    best_neighbor = neighbor

        # If no valid neighbor found (all neighbors have higher IDs), return empty proposal
        # This node will accept proposals from higher-ID neighbors instead
        if best_neighbor is None:
            return [Message(
                sender=node_id,
                recipient=None,  # Placeholder - signal that no valid proposal exists
                payload={
                    "type": "PROPOSE",
                    "weight": -1,
                    "proposer_id": node_id,
                },
                round_num=context.round_num,
            )]

        return [Message(
            sender=node_id,
            recipient=best_neighbor,
            payload={
                "type": "PROPOSE",
                "weight": best_weight,
                "proposer_id": node_id,
            },
            round_num=context.round_num,
        )]

    # ===== Private Helper Methods =====

    def _match_nodes(self, new_state, partner):
        """Finalize match between nodes."""
        new_state.set_matched_to(partner)
        new_state.set("status", "matched")
        new_state.set("active", False)
        new_state.set("negotiation_partner", None)
        new_state.set("negotiation_stage", None)
        new_state.set("stage_round", 0)
        return new_state

    def _start_negotiation(self, new_state, partner, stage):
        """Start negotiation with a new partner."""
        new_state.set("negotiation_partner", partner)
        new_state.set("negotiation_stage", stage)
        new_state.set("status", "proposing" if stage == "proposed" else "accepting")
        new_state.set("stage_round", 0)
        new_state.set("active", True)
        return new_state

    def _find_best_propose(self, proposes):
        """Find the best PROPOSE by weight and proposer_id."""
        return max(
            proposes,
            key=lambda m: (m.payload.get("weight", 0), m.payload.get("proposer_id", m.sender))
        )

    def check_termination(self, state_store: StateStore, round_num: RoundNumber, messages_sent: int) -> Tuple[bool, str | None]:
        """Check if algorithm has converged (uses common termination logic with aggressive round limit)."""
        max_rounds = self.metadata.properties.get("max_rounds", 30)
        return self.check_default_termination(state_store, round_num, messages_sent, max_rounds)
