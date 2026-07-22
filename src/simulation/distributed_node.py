"""Fully distributed node for autonomous algorithm execution and coordination."""

from typing import Dict, List, Tuple, Any
from src.state.node import NodeState
from src.communication.message import Message
from src.communication.message_queue import MessageQueue
from src.communication.node_communicator import NodeCommunicator
from src.graph.graph_manager import GraphManager
from src.graph.local_graph import LocalGraph
from src.metrics.metrics_collector import MetricsCollector
from src.config import DistributedAlgorithmConfig
from src.meta.messages.config_gossip_message import ConfigGossipMessage


class DistributedNode:
    """
    Autonomous node in a fully distributed system.

    **Owns (Local):**
    - state: This node's algorithm state
    - inbox/outbox: This node's messages
    - round_number: Tracks its own execution rounds
    - local_metrics: Tracks own performance
    - convergence_vote: Local convergence decision
    - pending_proposals: Proposals from neighbors to resolve

    **Shared (Read-Only):**
    - graph: Network topology (immutable)
    - algorithm_config: Algorithm configuration (immutable)
    - convergence_detector: Convergence detection logic (set by orchestrator)
    """

    def __init__(
        self,
        node_id: int,
        shared_graph: GraphManager,
        algorithm_config: DistributedAlgorithmConfig | None = None
    ):
        """Initialize a distributed node.

        Args:
            node_id: Unique node identifier
            shared_graph: Read-only reference to network topology
            algorithm_config: Algorithm configuration with convergence and algorithm parameters.
                             If None, creates default configuration.
        """
        self.id = node_id
        self.graph = LocalGraph(shared_graph, node_id)

        # Algorithm configuration (spreads via gossip protocol)
        self.algorithm_config = algorithm_config or DistributedAlgorithmConfig()

        # State (this node's algorithm state)
        self.state = NodeState(node_id)

        # Communication
        self.inbox = MessageQueue(shared_graph)
        self.outbox = MessageQueue(shared_graph)
        self.communicator = NodeCommunicator(node_id, self.outbox, self.inbox)

        # Execution tracking
        self.round_number = 0
        self.finished = False

        # Local metrics
        self.local_metrics = MetricsCollector()

        # Coordination state (convergence detection)
        self.convergence_vote = None
        self.known_convergence_votes: Dict[int, bool] = {}
        self.convergence_threshold = 0.05
        self.quorum_threshold = 0.5
        self.last_matching_weight = 0.0

        # Conflict resolution state (pending proposals from neighbors)
        self.pending_proposals: Dict[int, float] = {}

        # Convergence detector (set by orchestrator)
        self.convergence_detector: Any | None = None

    def _validate_available_algorithms(self) -> Tuple[bool, str]:
        """Validate that all configured algorithms are registered in this process.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        from src.meta.core.algorithm_registry import AlgorithmRegistry

        registry = AlgorithmRegistry.instance()
        for algo_name in self.algorithm_config.available_algorithms:
            if not registry.is_algorithm_registered(algo_name):
                return False, f"Algorithm '{algo_name}' not registered in this process"
        return True, ""

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

        # Validate that all configured algorithms are available
        is_valid, error = self._validate_available_algorithms()
        if not is_valid:
            return False, f"algorithm_validation_error: {error}"

        # PHASE 0: Process incoming messages
        messages = self.inbox.get_messages(self.id)
        self._process_coordination_messages(messages)

        # PHASE 1: Get proposals from each algorithm (LOCAL SCOPE ONLY - neighbors)
        from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer

        # FIXED (Task 8.3): Read from config instead of hardcoding
        parameterizers = [
            UnifiedAlgorithmParameterizer(algo_name)
            for algo_name in self.algorithm_config.available_algorithms
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
            active_nodes=1 if self.state.is_matched() else 0,
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


    def _create_context(self):
        """Create algorithm context for this node."""
        from src.simulation.algorithm_context import AlgorithmContext

        return AlgorithmContext(
            graph=self.graph, state_store=None, round_num=self.round_number
        )

    def _process_coordination_messages(self, messages: List[Message]) -> None:
        """Learn about other nodes' convergence decisions and configs from messages.

        Processes:
        - CONVERGENCE_VOTE: Other nodes' convergence decisions
        - CONFIG_GOSSIP: Algorithm configuration from neighbors

        Args:
            messages: All messages received this round
        """
        from src.meta.messages.convergence_gossip import ConvergenceGossipMessage

        convergence_msgs = []

        for msg in messages:
            # Handle config gossip messages
            if isinstance(msg, ConfigGossipMessage):
                self.receive_config_gossip(msg)
                continue

            # Handle convergence votes
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


    def conflict_solution(self) -> None:
        """Resolve all pending proposals via local endpoint voting.

        Selects the best proposal (highest weight) by deterministic criteria.
        Only matches if this node is not already matched (symmetry constraint).
        """
        # Only match if not already matched (matching constraint: each node matched at most once)
        if self.pending_proposals and not self.state.is_matched():
            # Find best proposer by highest weight (deterministic on node_id for tie-break)
            best_neighbor, best_weight = max(
                self.pending_proposals.items(),
                key=lambda item: (item[1], -item[0])
            )

            # Accept the best proposal
            self.state.set_matched_to(best_neighbor)

        # Clear proposals for next round
        self.pending_proposals.clear()

    def _process_phase2_confirmations(self, messages: List[Message]) -> None:
        """Process phase 2 confirmation messages.

        Args:
            messages: All messages received this round
        """
        # Phase 2/4 message handling moved to algorithm level
        # This method is a placeholder for future distributed consensus protocols
        pass

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
        self.pending_proposals.clear()

    # ============================================================================
    # CONFIG GOSSIP PROTOCOL (Distributed Configuration Spreading)
    # ============================================================================

    def gossip_config(self) -> None:
        """Send current algorithm configuration to random neighbors.

        Nodes spread their algorithm configuration via gossip protocol,
        including available algorithms list (Phase 8 - NEW).
        Neighbors will accept if version is higher than their current.
        """
        neighbors = list(self.graph.neighbors(self.id))
        if not neighbors:
            return

        # Create config message (includes algorithm list - Phase 8 - NEW)
        from src.meta.core.algorithm_registry import AlgorithmRegistry
        registry = AlgorithmRegistry.instance()
        available_algos = self.algorithm_config.available_algorithms or registry.all_algorithm_names()

        msg = ConfigGossipMessage(
            sender=self.id,
            recipient=-1,  # Will be set per-neighbor
            payload=self.algorithm_config.to_dict(),
            available_algorithms=available_algos,
            version=self.algorithm_config.version,
            algorithm_list_version=self.algorithm_config.algorithm_list_version,
            round_num=self.round_number
        )

        # Sample up to 3 neighbors to avoid flooding
        import random
        sample_size = min(3, len(neighbors))
        sampled_neighbors = random.sample(neighbors, sample_size)

        # Send to sampled neighbors
        for neighbor in sampled_neighbors:
            neighbor_msg = ConfigGossipMessage(
                sender=self.id,
                recipient=neighbor,
                payload=msg.payload.copy(),
                available_algorithms=msg.available_algorithms,
                version=msg.version,
                algorithm_list_version=msg.algorithm_list_version,
                round_num=self.round_number
            )
            self.outbox.send(neighbor_msg)

    def receive_config_gossip(self, msg: ConfigGossipMessage) -> None:
        """Receive and potentially adopt algorithm configuration from neighbor.

        Only accepts configuration if version > current version (version-based ordering).

        Args:
            msg: ConfigGossipMessage with algorithm configuration
        """
        if msg.version > self.algorithm_config.version:
            # Adopt this configuration
            self.algorithm_config = DistributedAlgorithmConfig.from_dict(msg.payload)
            self.algorithm_config.version = msg.version

            # Update algorithm list if neighbor has newer version
            if msg.algorithm_list_version > self.algorithm_config.algorithm_list_version:
                new_algos = msg.available_algorithms or []
                if new_algos != self.algorithm_config.available_algorithms:
                    self.algorithm_config.available_algorithms = new_algos.copy()
                    self.algorithm_config.algorithm_list_version = msg.algorithm_list_version
