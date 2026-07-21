"""Fully distributed node for autonomous algorithm execution and coordination."""

from typing import Dict, List, Tuple, Any
from src.state.node import NodeState
from src.communication.message import Message
from src.communication.message_queue import MessageQueue
from src.communication.node_communicator import NodeCommunicator
from src.graph.graph_manager import GraphManager
from src.metrics.metrics_collector import MetricsCollector
from src.meta.messages.edge_conflict_protocol import (
    EdgeProposalMessage,
    EdgeAcceptanceMessage,
)


class DistributedNode:
    """
    Autonomous node in a fully distributed system.

    Owns:
    - state: This node's algorithm state
    - inbox/outbox: This node's messages
    - round_number: Tracks its own execution rounds
    - local_metrics: Tracks own performance
    - coordination: Convergence voting

    Shared (read-only):
    - graph: Network topology (immutable)
    """

    def __init__(self, node_id: int, shared_graph: GraphManager):
        """Initialize a distributed node.

        Args:
            node_id: Unique node identifier
            shared_graph: Read-only reference to network topology
        """
        self.id = node_id
        self.graph = shared_graph

        # State (this node's algorithm state)
        self.state = NodeState(node_id)

        # Communication
        self.inbox = MessageQueue(shared_graph)  # Messages TO this node
        self.outbox = MessageQueue(shared_graph)  # Messages FROM this node
        self.communicator = NodeCommunicator(node_id, self.outbox, self.inbox)

        # Execution tracking
        self.round_number = 0
        self.finished = False

        # Local metrics (this node's performance)
        self.local_metrics = MetricsCollector()

        # Coordination state
        self.convergence_vote = None  # This node's convergence decision
        self.known_convergence_votes: Dict[int, bool] = {}  # Learned from neighbors
        self.convergence_threshold = 0.05  # Min improvement to continue
        self.quorum_threshold = 0.5  # Min fraction to stop
        self.last_matching_weight = 0.0  # For improvement tracking

        # Phase 1: Conflict resolution state
        self.pending_proposals: Dict[int, float] = {}  # proposer_id -> weight
        self.current_match: int | None = None
        self.best_weight = -1.0

        # Phase 2: Two-phase commit state
        self.tentative_match: int | None = None
        self.confirmed_match: int | None = None

        # Phase 4: Endpoint voting state for conflict resolution
        self.edge_votes: Dict[Tuple[int, int], List[bool]] = {}  # edge -> [votes]
        self.voting_quorum = 0.5  # Min fraction of endpoints that must agree

        # Algorithm reference (set by simulator)
        self.algorithm = None

        # Convergence detector (set by orchestrator)
        self.convergence_detector: Any | None = None

    def execute(self, algorithm) -> Tuple[bool, str]:
        """
        Execute one round of a single algorithm (for backward compatibility).

        Args:
            algorithm: MatchingAlgorithm instance to run

        Returns:
            (continue_running, status_message)
        """
        if self.finished:
            return False, "already_finished"

        self.algorithm = algorithm
        messages = self.inbox.get_messages(self.id)
        self._process_coordination_messages(messages)

        try:
            new_state, algorithm_messages = algorithm.node_behavior(
                self.id, self.state, messages, self._create_context()
            )
        except Exception as e:
            return False, f"algorithm_error: {str(e)}"

        self.state = new_state

        self.local_metrics.record_round(
            round_num=self.round_number,
            messages_sent=len(algorithm_messages),
            active_nodes=1 if self.state.get("active", True) else 0,
        )

        if algorithm_messages:
            self.outbox.send_batch(algorithm_messages)

        self._decide_convergence()
        self._gossip_convergence_vote()

        should_stop = self._should_stop_based_on_quorum()
        if should_stop:
            self.finished = True
            return False, "quorum_converged"

        self.round_number += 1
        return True, "continuing"

    def execute_distributed_round(self, canonical_vector) -> Tuple[bool, str]:
        """
        NEW PHASED EXECUTION (Protocol-Driven Merge).

        Execute one round with structured phases:
        PHASE 0: Process incoming messages/proposals
        PHASE 1: Each algorithm proposes to neighbors (LOCAL SCOPE ONLY)
        PHASE 2: Accumulate all proposals into pending_proposals
        PHASE 3: ALWAYS call conflict_solution() (Protocol Consistency)
        PHASE 4: Send confirmation messages (Phase 2 two-phase commit)
        PHASE 5: Check convergence

        Args:
            canonical_vector: CanonicalVector with parameters for all algorithms

        Returns:
            (continue_running, status_message)
        """
        if self.finished:
            return False, "already_finished"

        # PHASE 0: Process incoming messages
        messages = self.inbox.get_messages(self.id)
        self._process_coordination_messages(messages)

        # PHASE 1: Get proposals from each algorithm (LOCAL SCOPE ONLY - neighbors)
        from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer

        parameterizers = [
            UnifiedAlgorithmParameterizer("greedy"),
            UnifiedAlgorithmParameterizer("itai"),
            UnifiedAlgorithmParameterizer("luby"),
        ]
        neighbors = list(self.graph.neighbors(self.id))
        context = self._create_context()

        proposals_per_algorithm = {}
        for param in parameterizers:
            try:
                # Each algorithm proposes ONLY to neighbors (local scope)
                algo_name = param.name()
                proposals = param.propose_to_neighbors(self.id, neighbors, context)
                proposals_per_algorithm[algo_name] = proposals
            except Exception as e:
                # Log the error for debugging
                algo_name = param.name()
                proposals_per_algorithm[algo_name] = {}

        # PHASE 2: Accumulate proposals from all algorithms
        self.pending_proposals.clear()
        for algo_name, proposals in proposals_per_algorithm.items():
            if proposals:  # Only process non-empty proposals
                for neighbor_id, weight in proposals.items():
                    # Keep highest weight proposal per neighbor
                    if neighbor_id not in self.pending_proposals:
                        self.pending_proposals[neighbor_id] = weight
                    else:
                        self.pending_proposals[neighbor_id] = max(
                            self.pending_proposals[neighbor_id], weight
                        )

        # PHASE 3: ALWAYS call conflict_solution() - Protocol Consistency Guaranteed
        # Handles: 0 proposals (no-op), 1 proposal (select it), N proposals (full voting)
        self.conflict_solution()

        # PHASE 4: Handle confirmations (Phase 2 two-phase commit)
        self._process_phase2_confirmations(messages)

        # Track metrics
        total_proposals = sum(len(p) for p in proposals_per_algorithm.values())
        self.local_metrics.record_round(
            round_num=self.round_number,
            messages_sent=total_proposals,
            active_nodes=1 if self.confirmed_match is not None else 0,
        )

        # Send coordination messages to neighbors
        self._gossip_convergence_vote()

        # Decide convergence vote
        self._decide_convergence()

        # PHASE 5: Check convergence
        should_stop = self._should_stop_based_on_quorum()
        if should_stop:
            self.finished = True
            return False, "quorum_converged"

        self.round_number += 1
        return True, "continuing"

    def _local_conflict_resolution(self, matchings: List[Dict[int, int]]) -> Dict[int, int]:
        """Resolve conflicts via endpoint voting protocol."""
        from src.meta.distributed.edge_voting import collect_proposed_edges, apply_quorum_threshold

        proposed_edges = collect_proposed_edges(self.id, matchings, self.graph)
        return apply_quorum_threshold(proposed_edges, self.id, self.voting_quorum)

    def _create_context(self):
        """Create algorithm context for this node."""
        from src.simulation.algorithm_context import AlgorithmContext

        return AlgorithmContext(
            graph=self.graph, state_store=None, round_num=self.round_number
        )

    def _process_coordination_messages(self, messages: List[Message]) -> None:
        """Learn about other nodes' convergence decisions from messages.

        Args:
            messages: All messages received this round
        """
        from src.meta.messages.convergence_gossip import ConvergenceGossipMessage

        convergence_msgs = []

        for msg in messages:
            if msg.payload.get("type") == "CONVERGENCE_VOTE":
                sender_id = msg.sender
                vote = msg.payload.get("vote", False)
                self.known_convergence_votes[sender_id] = vote

                # Create ConvergenceGossipMessage for detector
                conv_msg = ConvergenceGossipMessage(
                    sender_node_id=sender_id,
                    should_stop=msg.payload.get("should_stop", vote),
                    round_num=msg.payload.get("round", self.round_number),
                    weight=msg.payload.get("weight", 0.0),
                )
                convergence_msgs.append(conv_msg)

        # Pass convergence messages to detector if available
        if convergence_msgs and self.convergence_detector is not None:
            self.convergence_detector.receive_convergence_votes(self.id, convergence_msgs)

    def _decide_convergence(self) -> None:
        """Decide this node's convergence vote based on local metrics.

        Uses detector if available, otherwise falls back to local logic.
        """
        if self.convergence_detector is None:
            # Fallback: simple convergence decision
            if self.round_number == 0:
                self.convergence_vote = False
                return

            # Get current matching weight from matched_edges
            matched_edges = self.state.get("matched_edges", [])
            current_weight = sum(edge.weight for edge in matched_edges) if matched_edges else 0.0

            # Compute improvement
            if self.last_matching_weight > 0:
                improvement = (current_weight - self.last_matching_weight) / self.last_matching_weight
            else:
                improvement = 1.0 if current_weight > 0 else 0.0

            self.last_matching_weight = current_weight
            self.convergence_vote = improvement < self.convergence_threshold
            return

        # Use detector for convergence decision
        matched_edges = self.state.get("matched_edges", [])
        current_weight = sum(edge.weight for edge in matched_edges) if matched_edges else 0.0

        self.convergence_detector.decide_convergence(
            self.id,
            self.last_matching_weight,
            current_weight
        )
        self.last_matching_weight = current_weight

    def _gossip_convergence_vote(self) -> None:
        """Send convergence vote to random neighbors via message."""
        # Check if detector says to gossip this round
        if self.convergence_detector is not None:
            if not self.convergence_detector.should_gossip(self.id, self.round_number):
                return

            # Get convergence message from detector
            conv_msg = self.convergence_detector.create_convergence_message(self.id, self.round_number)
            should_stop = conv_msg.should_stop
            weight = conv_msg.weight
        else:
            # Fallback: use simple convergence vote
            should_stop = self.convergence_vote if self.convergence_vote is not None else False
            weight = sum(edge.weight for edge in self.state.get("matched_edges", []))

        msg = Message(
            sender=self.id,
            recipient=-1,  # Will be set per-neighbor
            payload={
                "type": "CONVERGENCE_VOTE",
                "vote": should_stop,
                "should_stop": should_stop,
                "round": self.round_number,
                "weight": weight,
                "active": self.state.get("active", True),
                "matched": self.state.is_matched(),
            },
            round_num=self.round_number,
        )

        # Send to random neighbors (not all to avoid flooding)
        neighbors = list(self.graph.neighbors(self.id))
        if not neighbors:
            return

        # Sample up to 3 neighbors
        import random

        sample_size = min(3, len(neighbors))
        sampled_neighbors = random.sample(neighbors, sample_size)

        for neighbor in sampled_neighbors:
            neighbor_msg = Message(
                sender=self.id,
                recipient=neighbor,
                payload=msg.payload.copy(),
                round_num=self.round_number,
            )
            self.outbox.send(neighbor_msg)

    def _should_stop_based_on_quorum(self) -> bool:
        """Check if quorum of nodes voted to stop based on gossip.

        Uses detector if available, otherwise falls back to local logic.

        Returns:
            True if >quorum_threshold fraction of known nodes voted STOP
        """
        if self.convergence_detector is not None:
            # Use detector to check termination
            return self.convergence_detector.should_terminate(self.id)

        # Fallback: simple quorum check
        if not self.known_convergence_votes:
            return False

        stop_votes = sum(1 for v in self.known_convergence_votes.values() if v)
        total_known = len(self.known_convergence_votes)

        if total_known == 0:
            return False

        fraction_voting_stop = stop_votes / total_known
        return fraction_voting_stop > self.quorum_threshold

    def get_matching(self) -> Dict[int, int]:
        """Extract final matching from node state.

        Returns:
            Dict[node_id -> matched_to_id]
        """
        if self.state.is_matched():
            matched_to = self.state.get_matched_to()
            return {self.id: matched_to}
        return {}

    @property
    def metrics_summary(self) -> Dict[str, Any]:
        """Summary of node's metrics for gossip/observation.

        Returns:
            Dict with node's local metrics
        """
        return {
            "node_id": self.id,
            "round_number": self.round_number,
            "active": self.state.get("active", False),
            "matched": self.state.is_matched(),
            "convergence_vote": self.convergence_vote,
            "known_nodes": len(self.known_convergence_votes),
            "messages_sent": self.local_metrics.total_messages,
            "finished": self.finished,
        }

    def _send_edge_proposal(self, edge: Tuple[int, int], weight: float) -> None:
        """Send edge proposal to the other endpoint (Phase 4).

        Args:
            edge: Edge (u, v) tuple
            weight: Edge weight for conflict resolution
        """
        u, v = edge
        recipient = v if u == self.id else u

        msg = EdgeProposalMessage(
            sender=self.id,
            recipient=recipient,
            payload={"edge": edge, "weight": weight},
            round_num=self.round_number,
        )

        self.outbox.send(msg)

    def _receive_edge_votes(self, messages: List[Message]) -> Dict[Tuple[int, int], List[bool]]:
        """Extract edge acceptance votes from messages (Phase 4).

        Args:
            messages: Incoming messages

        Returns:
            Dict mapping edge -> list of votes from endpoints
        """
        votes: Dict[Tuple[int, int], List[bool]] = {}

        for msg in messages:
            if isinstance(msg, EdgeAcceptanceMessage):
                edge = msg.edge
                vote = msg.vote

                if edge not in votes:
                    votes[edge] = []
                votes[edge].append(vote)

        return votes

    # ============================================================================
    # PHASE 2: TWO-PHASE COMMIT (SYMMETRY GUARANTEE)
    # ============================================================================

    def handle_match_confirmation(self, sender: int, msg_dict: Dict) -> None:
        """Handle match confirmation from other endpoint (Phase 2).

        Args:
            sender: Node ID of the sender
            msg_dict: Message payload with status "CONFIRMED"
        """
        status = msg_dict.get("status", "")

        if status == "CONFIRMED":
            # Both sides confirmed the match
            if self.tentative_match == sender:
                self.confirmed_match = sender
                self.current_match = sender
                self.state.set_matched_to(sender)

    def handle_match_cancellation(self, sender: int, msg_dict: Dict) -> None:
        """Handle partner found better match (Phase 2).

        Args:
            sender: Node ID of the sender
            msg_dict: Message payload with new_partner
        """
        new_partner = msg_dict.get("new_partner", -1)

        # If we had match with sender, it's cancelled
        if self.confirmed_match == sender:
            self.confirmed_match = None
            self.current_match = None
            self.tentative_match = None
        elif self.tentative_match == sender:
            # Even tentative matches can be cancelled
            self.tentative_match = None

    def accept_tentative_match(self, neighbor_id: int, weight: float) -> None:
        """Accept match tentatively (Phase 2, Phase 1 of Two-Phase).

        Args:
            neighbor_id: Node we're tentatively matching to
            weight: Weight of edge
        """
        # If we have a better tentative match, cancel the old one
        if self.tentative_match is not None and self.best_weight > weight:
            # Keep current, reject new
            return

        # If this is better, update tentative
        if self.tentative_match is None or weight > self.best_weight:
            # Cancel old if existed
            if self.tentative_match is not None:
                self.send_match_cancellation(self.tentative_match, neighbor_id)

            self.tentative_match = neighbor_id
            self.best_weight = weight

            # Send confirmation back
            self.send_match_confirmation(neighbor_id, "CONFIRMED")

    def send_match_confirmation(self, recipient: int, status: str) -> None:
        """Send match confirmation to other endpoint (Phase 2).

        Args:
            recipient: Node to confirm match with
            status: "CONFIRMED" or "CANCELLED"
        """
        msg = Message(
            sender=self.id,
            recipient=recipient,
            payload={
                "type": "MATCH_CONFIRMATION",
                "status": status,
                "edge": (self.id, recipient)
            },
            round_num=self.round_number
        )

        self.outbox.send(msg)

    def send_match_cancellation(self, recipient: int, new_partner: int) -> None:
        """Send match cancellation (found better match) (Phase 2).

        Args:
            recipient: Node we're cancelling match with
            new_partner: Node we're switching to
        """
        msg = Message(
            sender=self.id,
            recipient=recipient,
            payload={
                "type": "MATCH_CANCELLATION",
                "edge": (self.id, recipient),
                "new_partner": new_partner
            },
            round_num=self.round_number
        )

        self.outbox.send(msg)

    # ============================================================================
    # PHASE 1: DISTRIBUTED CONFLICT RESOLUTION
    # ============================================================================

    def receive_edge_proposal(self, proposer_id: int, weight: float) -> None:
        """Receive proposal from another node (Phase 1).

        Args:
            proposer_id: Node ID making the proposal
            weight: Weight of the proposed edge
        """
        self.pending_proposals[proposer_id] = weight

    def resolve_conflicts(self) -> int | None:
        """Find best proposal locally (Phase 1).

        Returns:
            ID of best proposer, or None if no proposals
        """
        if not self.pending_proposals:
            return None

        # Sort by: (weight DESC, then node_id ASC for determinism)
        best_proposer = max(
            self.pending_proposals.items(),
            key=lambda item: (item[1], -item[0])
        )
        return best_proposer[0]

    def conflict_solution(self) -> None:
        """Resolve all pending proposals and send responses (Phase 1).

        Sends EdgeAcceptanceMessage to best proposer (accept)
        and to all others (reject).
        """
        best_proposer = self.resolve_conflicts()

        if best_proposer is None:
            # No proposals, nothing to do
            self.pending_proposals.clear()
            return

        # Send acceptance to best proposer
        self.send_edge_acceptance_message(
            recipient=best_proposer,
            status="ACCEPTED",
            reason="Best proposal"
        )

        # Send rejection to all others
        for proposer_id in self.pending_proposals.keys():
            if proposer_id != best_proposer:
                self.send_edge_acceptance_message(
                    recipient=proposer_id,
                    status="REJECTED",
                    reason="Better proposal chosen"
                )

        # Clear pending proposals for next round
        self.pending_proposals.clear()

    def send_edge_acceptance_message(
        self,
        recipient: int,
        status: str,
        reason: str
    ) -> None:
        """Send acceptance/rejection to proposer (Phase 1).

        Args:
            recipient: Node ID of proposer
            status: "ACCEPTED" or "REJECTED"
            reason: Human-readable reason
        """
        msg = Message(
            sender=self.id,
            recipient=recipient,
            payload={
                "type": "EDGE_ACCEPTANCE",
                "edge": (self.id, recipient),
                "vote": (status == "ACCEPTED"),
                "reason": reason,
                "status": status
            },
            round_num=self.round_number
        )

        self.outbox.send(msg)

    def handle_edge_acceptance(self, msg: EdgeAcceptanceMessage) -> None:
        """Handle response to our proposal (Phase 1).

        Args:
            msg: EdgeAcceptanceMessage
        """
        if msg.vote:  # msg.vote == True means ACCEPTED
            # Proposer accepted us
            self.current_match = msg.sender
            self.best_weight = self.graph.get_edge_weight(self.id, msg.sender)
        else:
            # Proposer rejected us - we can try next neighbor next round
            pass

    def _process_phase2_confirmations(self, messages: List[Message]) -> None:
        """Process Phase 1/2 acceptance and confirmation messages.

        Handles:
        - EDGE_ACCEPTANCE: Response to our proposal (Phase 1)
        - MATCH_CONFIRMATION: Two-phase commit confirmation (Phase 2)
        - MATCH_CANCELLATION: Partner found better match (Phase 2)

        Args:
            messages: All messages received this round
        """
        for msg in messages:
            payload = msg.payload
            msg_type = payload.get("type")
            sender = msg.sender

            if msg_type == "EDGE_ACCEPTANCE":
                # Phase 1: Response to our proposal
                status = payload.get("status", "")
                reason = payload.get("reason", "")
                edge_weight = self.graph.get_edge_weight(self.id, sender)

                if status == "ACCEPTED":
                    # Our proposal was accepted - move to tentative match
                    self.accept_tentative_match(sender, edge_weight)
                # If REJECTED, do nothing (we can try other neighbors)

            elif msg_type == "MATCH_CONFIRMATION":
                self.handle_match_confirmation(sender, payload)
            elif msg_type == "MATCH_CANCELLATION":
                self.handle_match_cancellation(sender, payload)

    def reset(self) -> None:
        """Reset node to initial state."""
        self.state = NodeState(self.id)
        self.inbox = MessageQueue(self.graph)
        self.outbox = MessageQueue(self.graph)
        self.round_number = 0
        self.finished = False
        self.local_metrics.reset()
        self.convergence_vote = None
        self.known_convergence_votes.clear()
        self.last_matching_weight = 0.0
        self.edge_votes.clear()
        self.pending_proposals.clear()
