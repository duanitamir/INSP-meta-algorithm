"""Fully distributed node for autonomous algorithm execution and coordination."""

from typing import Dict, List, Tuple, Any
from src.state.node_state import NodeState
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

        # Phase 4: Endpoint voting state for conflict resolution
        self.edge_votes: Dict[Tuple[int, int], List[bool]] = {}  # edge -> [votes]
        self.pending_proposals: Dict[Tuple[int, int], float] = {}  # edge -> weight
        self.voting_quorum = 0.5  # Min fraction of endpoints that must agree

        # Algorithm reference (set by simulator)
        self.algorithm = None

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
        Execute one round: run ALL 3 algorithms autonomously, merge locally.

        This is the truly distributed round - node decides which algorithms to run
        and how to handle their results. This is what happens on real network nodes.

        Args:
            canonical_vector: CanonicalVector with parameters for all algorithms

        Returns:
            (continue_running, status_message)
        """
        if self.finished:
            return False, "already_finished"

        # Get messages from inbox
        messages = self.inbox.get_messages(self.id)
        self._process_coordination_messages(messages)

        # Run ALL 3 algorithms autonomously
        from src.meta.parameterizers.factory import ParameterizerFactory

        parameterizers = ParameterizerFactory.create_default()

        # Run all 3 algorithms independently and collect results
        # Each algorithm uses StateStore with per-node locks for thread safety
        matchings = []
        for param in parameterizers:
            try:
                # Each algorithm executes on full graph, but its results should be valid
                algorithm_result = param.execute(self.graph, canonical_vector)
                # Filter to only edges involving this node
                filtered = (
                    {self.id: algorithm_result[self.id]} if self.id in algorithm_result else {}
                )
                matchings.append(filtered)
            except Exception:
                matchings.append({})

        # Node merges locally: take best edge by weight (endpoint voting simulation)
        merged_matching = self._local_conflict_resolution(matchings)

        # Update this node's state with merged result
        if self.id in merged_matching:
            self.state.set_matched_to(merged_matching[self.id])

        # Track metrics
        self.local_metrics.record_round(
            round_num=self.round_number,
            messages_sent=sum(len(m) for m in matchings),
            active_nodes=1 if len(merged_matching) > 0 else 0,
        )

        # Send coordination messages to neighbors
        self._gossip_convergence_vote()

        # Decide convergence vote
        self._decide_convergence()

        # Check if network converged (majority voting)
        should_stop = self._should_stop_based_on_quorum()
        if should_stop:
            self.finished = True
            return False, "quorum_converged"

        self.round_number += 1
        return True, "continuing"

    def _local_conflict_resolution(self, matchings: List[Dict[int, int]]) -> Dict[int, int]:
        """Resolve conflicts via endpoint voting protocol (Phase 4).

        For each proposed edge (u, v):
        1. Node u sends EdgeProposalMessage to node v
        2. Node v responds with EdgeAcceptanceMessage (vote)
        3. Edge included in final matching only if both endpoints vote YES

        Args:
            matchings: List of matching dicts from different algorithms

        Returns:
            Dict[node_id -> matched_to_id] for edges with quorum acceptance
        """
        # Step 1: Collect all unique edges proposed by any algorithm
        proposed_edges = {}  # edge -> weight
        for matching in matchings:
            if self.id in matching:
                matched_to = matching[self.id]

                # Skip self-matches and invalid edges
                if matched_to == self.id:
                    continue
                if not self.graph._graph.has_edge(self.id, matched_to):
                    continue

                weight = self.graph.get_edge_weight(self.id, matched_to)
                edge = (min(self.id, matched_to), max(self.id, matched_to))

                # Keep highest weight for each edge
                if edge not in proposed_edges or weight > proposed_edges[edge]:
                    proposed_edges[edge] = weight

        # Step 2: Send proposals to endpoints and collect votes
        final_matching = {}

        for edge, weight in proposed_edges.items():
            u, v = edge
            votes = []

            # This node is an endpoint: vote based on its preference
            # For now, this node always votes YES for proposed edges
            # (In real distributed system, node would vote based on constraints)
            this_node_vote = True
            votes.append(this_node_vote)

            # Simulate receiving vote from other endpoint
            # In real system, would send EdgeProposalMessage and wait for EdgeAcceptanceMessage
            # For simulation, we assume other node votes based on weight (high weight -> yes)
            other_node_vote = weight > 0  # Accept non-zero weight edges
            votes.append(other_node_vote)

            # Step 3: Apply quorum threshold
            # Edge included if >= 50% of endpoints vote YES (for 2 endpoints: both must agree)
            if len(votes) >= 2:
                yes_votes = sum(1 for v in votes if v)
                if yes_votes >= len(votes) * self.voting_quorum:
                    # Add to final matching (edge normalized to (this_node, other_node))
                    if u == self.id:
                        final_matching[self.id] = v
                    else:
                        final_matching[self.id] = u

        return final_matching

    def _create_context(self):
        """Create algorithm context for this node."""
        from src.simulation.algorithm_context import AlgorithmContext
        from src.state.node_state_store_adapter import NodeStateStoreAdapter

        # Create adapter so algorithms can work unchanged
        state_store_adapter = NodeStateStoreAdapter(self.state, self.id)

        return AlgorithmContext(
            graph=self.graph, state_store=state_store_adapter, round_num=self.round_number
        )

    def _process_coordination_messages(self, messages: List[Message]) -> None:
        """Learn about other nodes' convergence decisions from messages.

        Args:
            messages: All messages received this round
        """
        for msg in messages:
            if msg.payload.get("type") == "CONVERGENCE_VOTE":
                sender_id = msg.sender
                vote = msg.payload.get("vote", False)
                self.known_convergence_votes[sender_id] = vote

    def _decide_convergence(self) -> None:
        """Decide this node's convergence vote based on local metrics.

        Node votes STOP if no improvement in last round.
        """
        if self.round_number == 0:
            self.convergence_vote = False
            return

        # Get current matching weight from matched_edges
        matched_edges = self.state.get("matched_edges", [])
        current_weight = sum(edge.weight for edge in matched_edges) if matched_edges else 0

        # Compute improvement
        if self.last_matching_weight > 0:
            improvement = (current_weight - self.last_matching_weight) / self.last_matching_weight
        else:
            improvement = 1.0 if current_weight > 0 else 0.0

        self.last_matching_weight = current_weight

        # Vote to stop if improvement below threshold
        self.convergence_vote = improvement < self.convergence_threshold

    def _gossip_convergence_vote(self) -> None:
        """Send convergence vote to random neighbors via message."""
        msg = Message(
            sender=self.id,
            recipient=-1,  # Will be set per-neighbor
            payload={
                "type": "CONVERGENCE_VOTE",
                "vote": self.convergence_vote,
                "round": self.round_number,
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

        Returns:
            True if >quorum_threshold fraction of known nodes voted STOP
        """
        if not self.known_convergence_votes:
            # Don't know about other nodes yet
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
