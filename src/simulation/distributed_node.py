"""Fully distributed node for autonomous algorithm execution and coordination."""

from typing import Dict, List, Tuple, Any, Callable
from src.state.node_state import NodeState
from src.communication.message import Message
from src.communication.message_queue import MessageQueue
from src.graph.graph_manager import GraphManager
from src.metrics.metrics_collector import MetricsCollector


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

        # Algorithm reference (set by simulator)
        self.algorithm = None

    def execute(self, algorithm) -> Tuple[bool, str]:
        """
        Execute one round of the algorithm.

        This node:
        1. Processes incoming messages (including convergence votes)
        2. Runs algorithm with local state
        3. Tracks metrics locally
        4. Decides convergence vote
        5. Checks if quorum reached (should stop)

        Args:
            algorithm: MatchingAlgorithm instance to run

        Returns:
            (continue_running, status_message) - False if node should stop
        """
        if self.finished:
            return False, "already_finished"

        self.algorithm = algorithm

        # Get messages from inbox
        messages = self.inbox.get_messages(self.id)

        # Process coordination messages (convergence votes from neighbors)
        self._process_coordination_messages(messages)

        # Run algorithm for this round
        try:
            new_state, algorithm_messages = algorithm.node_behavior(
                self.id,
                self.state,
                messages,
                self._create_context()
            )
        except Exception as e:
            return False, f"algorithm_error: {str(e)}"

        # Update state
        self.state = new_state

        # Track metrics
        self.local_metrics.record_round(
            round_num=self.round_number,
            messages_sent=len(algorithm_messages),
            active_nodes=1 if self.state.get("active", True) else 0
        )

        # Send algorithm messages
        if algorithm_messages:
            self.outbox.send_batch(algorithm_messages)

        # Decide convergence vote
        self._decide_convergence()

        # Gossip convergence vote to neighbors
        self._gossip_convergence_vote()

        # Check if network converged (learned from neighbors)
        should_stop = self._should_stop_based_on_quorum()

        if should_stop:
            self.finished = True
            return False, "quorum_converged"

        self.round_number += 1
        return True, "continuing"

    def _create_context(self):
        """Create algorithm context for this node."""
        from src.simulation.algorithm_context import AlgorithmContext
        from src.state.node_state_store_adapter import NodeStateStoreAdapter

        # Create adapter so algorithms can work unchanged
        state_store_adapter = NodeStateStoreAdapter(self.state, self.id)

        return AlgorithmContext(
            graph=self.graph,
            state_store=state_store_adapter,
            round_num=self.round_number
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
                "matched": self.state.is_matched()
            },
            round_num=self.round_number
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
                round_num=self.round_number
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
            "finished": self.finished
        }

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
