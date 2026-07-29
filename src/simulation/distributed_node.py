"""Fully distributed node for autonomous algorithm execution and coordination."""

from typing import Dict, List, Tuple, Any
from src.state.node import NodeState
from src.communication.message import Message
from src.communication.node_communicator import NodeCommunicator
from src.communication.transport import InMemoryTransport
from src.graph.graph_manager import GraphManager
from src.graph.local_graph import LocalGraph
from src.metrics.metrics_collector import MetricsCollector
from src.config import DistributedAlgorithmConfig
from src.meta.messages.gossip_message import GossipMessage


class DistributedNode:
    """
    Autonomous node in a fully distributed system.

    **Owns (Local):**
    - state: This node's algorithm state
    - communicator: This node's addressed messages
    - round_number: Tracks its own execution rounds
    - local_metrics: Tracks own performance
    - convergence_vote: Local convergence decision
    - pending_proposals: Proposals from neighbors to resolve

    **Shared (Read-Only):**
    - graph: Network topology (immutable)
    - algorithm_config: Algorithm configuration (immutable)
    """

    def __init__(
        self,
        node_id: int,
        shared_graph: GraphManager,
        algorithm_config: DistributedAlgorithmConfig | None = None,
        transport: InMemoryTransport | None = None,
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

        # Communication is a recipient-bound facade over a black-box transport.
        # A standalone node receives a private transport for local unit tests;
        # a runtime supplies one shared transport to every node.
        self.transport = transport or InMemoryTransport(shared_graph.vertices())
        self.communicator = NodeCommunicator(node_id, self.transport)

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

        # Endpoint-owned matching protocol time.  This is deliberately local:
        # the executor may invoke a node, but does not advance its negotiations.
        self.local_time = 0
        self.proposal_timeout = 3

        # Track should_stop for autonomous loop (Phase 1)
        self.should_stop = False

    def run_autonomous(self, canonical_vector: CanonicalVector = None) -> None:
        """PHASE 1: Node's autonomous execution loop.

        This is the core autonomous agent behavior - the node runs its own complete
        algorithm loop without orchestrator control. Each node makes all its own
        decisions and only communicates via messages.

        Args:
            canonical_vector: Parameter vector for algorithm configuration (optional)
        """
        from src.meta.core.canonical_vector import CanonicalVector

        if canonical_vector is None:
            canonical_vector = CanonicalVector()

        max_iterations = int(canonical_vector.get("max_iterations") or 100)
        executed_iterations = 0

        # Run autonomous rounds until this node decides to stop or reaches its
        # node-owned safety limit.  A node without incoming quorum messages
        # must not spin forever.
        while (
            not self.should_stop
            and not self.finished
            and executed_iterations < max_iterations
        ):
            # Execute one round (PHASE 0-5 all happen in execute_distributed_round)
            continue_running, _ = self.execute_distributed_round(canonical_vector)
            executed_iterations += 1

            # Check if node should stop
            if not continue_running or self.finished:
                self.should_stop = True

        if executed_iterations >= max_iterations and self.is_active():
            self.should_stop = True

    def is_active(self) -> bool:
        """Check if node is still active (not finished and not stopped)."""
        return not self.finished and not self.should_stop

    def receive(self, message: Message) -> None:
        """Receive an addressed message from the transport."""
        if message.recipient != self.id:
            raise ValueError(f"Message for node {message.recipient} delivered to node {self.id}")
        self.transport.send(message.sender, self.id, message)

    def advance_local_time(self, elapsed: int = 1) -> None:
        """Advance only this node's logical clock."""
        if elapsed < 0:
            raise ValueError("Local time cannot move backwards")
        self.local_time += elapsed

    def tick(self) -> None:
        """Process protocol messages and local deadlines without central direction."""
        messages = self.communicator.receive_messages()
        self._process_coordination_messages(messages)
        self._process_protocol_messages(messages)
        self._expire_tentative_match_if_needed()
        self._decide_convergence()
        self._gossip_convergence_vote()
        if self._should_stop_based_on_quorum():
            self.finished = True
        self.advance_local_time()

    def start_proposal(self, neighbor_id: int, weight: float) -> None:
        """Start a match negotiation with one directly connected neighbor."""
        if neighbor_id not in self.graph.neighbors():
            raise ValueError(f"Node {neighbor_id} is not a neighbor of node {self.id}")
        if self.state.is_matched() or self.state.get(NodeState.TENTATIVE_PARTNER) is not None:
            return
        self.state.begin_tentative_match(
            neighbor_id,
            self.local_time + self.proposal_timeout,
            proposing=True,
        )
        self._send_protocol_message(neighbor_id, "PROPOSE", weight=weight)

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
        messages = self.communicator.receive_messages()
        self._process_coordination_messages(messages)
        self._process_protocol_messages(messages)
        self._expire_tentative_match_if_needed()

        # PHASE 1: Get proposals from each algorithm (LOCAL SCOPE ONLY - neighbors)
        from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer

        # FIXED (Task 8.3): Read from config instead of hardcoding
        parameterizers = [
            UnifiedAlgorithmParameterizer(algo_name)
            for algo_name in self.algorithm_config.available_algorithms
        ]
        neighbors = self.graph.neighbors()
        context = self._create_context()

        proposals_per_algorithm = {}
        for param in parameterizers:
            # Each algorithm proposes ONLY to neighbors (local scope). Unexpected
            # failures must surface rather than being mistaken for no proposal.
            algo_name = param.name()
            proposals = param.propose_to_neighbors(self.id, neighbors, context)
            proposals_per_algorithm[algo_name] = proposals

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

        # Track metrics
        total_proposals = sum(len(p) for p in proposals_per_algorithm.values())
        self.local_metrics.record_round(
            round_num=self.round_number,
            messages_sent=total_proposals,
            active_nodes=1 if self.state.is_matched() else 0,
        )

        # Decide convergence vote
        self._decide_convergence()

        # Share this tick's local decision only after computing it.
        self._gossip_convergence_vote()

        # PHASE 5: Check convergence
        should_stop = self._should_stop_based_on_quorum()
        if should_stop:
            self.finished = True
            return False, "quorum_converged"

        self.round_number += 1
        self.advance_local_time()
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
        - GossipMessage with subtype="config": Algorithm configuration from neighbors
        - Generic messages with type="CONVERGENCE_VOTE": Other nodes' convergence decisions

        Args:
            messages: All messages received this round
        """
        for msg in messages:
            # Handle generic gossip messages by subtype
            if isinstance(msg, GossipMessage):
                # Route by message subtype
                if msg.message_subtype == "config":
                    self.receive_config_gossip(msg)
                    continue
                elif msg.message_subtype == "convergence":
                    continue

            # Handle legacy generic messages with type payload
            if msg.payload.get("type") == "CONVERGENCE_VOTE":
                sender_id = msg.sender
                vote = msg.payload.get("vote", False)
                self.known_convergence_votes[sender_id] = vote


    def _decide_convergence(self) -> None:
        """Decide this node's convergence vote based on local metrics (FULLY DISTRIBUTED).

        PHASE 2: Computes local convergence vote based on improvement threshold.
        No centralized detector needed - pure distributed voting.
        """
        # First round: always vote CONTINUE (no improvement data yet)
        if self.round_number == 0:
            self.convergence_vote = False
            return

        # Get current matching weight from matched_edges
        matched_edges = self.state.get("matched_edges", [])
        current_weight = sum(edge.weight for edge in matched_edges) if matched_edges else 0.0

        # Compute improvement vs previous round
        if self.last_matching_weight > 0:
            improvement = (current_weight - self.last_matching_weight) / self.last_matching_weight
        else:
            improvement = 1.0 if current_weight > 0 else 0.0

        # Update weight for next round
        self.last_matching_weight = current_weight

        # Vote to STOP if improvement < threshold (distributed decision)
        self.convergence_vote = improvement < self.convergence_threshold

    def _gossip_convergence_vote(self) -> None:
        """Send convergence vote to random neighbors via message (FULLY DISTRIBUTED).

        PHASE 2: Each node broadcasts its convergence decision to neighbors.
        Pure distributed protocol - no centralized detector.
        """
        # Use node's local convergence vote (computed in _decide_convergence)
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
        neighbors = self.graph.neighbors()
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
            self.communicator.send_message(neighbor_msg)

    def _should_stop_based_on_quorum(self) -> bool:
        """Check if quorum of nodes voted to stop based on gossip (FULLY DISTRIBUTED).

        PHASE 2: Pure distributed quorum voting - no centralized detector.
        Checks if >50% of known neighbors voted to stop.

        Returns:
            True if >quorum_threshold fraction of known nodes voted STOP
        """
        # A node must finish an endpoint negotiation it already entered.  A
        # quorum can stop idle work, never discard a locally owned commit.
        if self.state.get(NodeState.TENTATIVE_PARTNER) is not None:
            return False

        # No votes collected yet
        if not self.known_convergence_votes:
            return False

        # Count votes for STOP
        stop_votes = sum(1 for v in self.known_convergence_votes.values() if v)
        total_known = len(self.known_convergence_votes)

        if total_known == 0:
            return False

        # Quorum: >50% of neighbors must vote to STOP
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


    def _send_protocol_message(self, recipient_id: int, message_type: str, **payload: Any) -> None:
        """Send one endpoint-protocol message through black-box transport."""
        self.communicator.send_message(
            Message(sender=self.id, recipient=recipient_id, payload={"type": message_type, **payload}, round_num=self.round_number)
        )

    def _process_protocol_messages(self, messages: List[Message]) -> None:
        """Advance only this endpoint's PROPOSE/ACCEPT/CONFIRM/CANCEL state."""
        proposals = []
        for message in messages:
            if message.sender not in self.graph.neighbors():
                continue
            payload = message.payload
            if not isinstance(payload, dict):
                continue
            message_type = payload.get("type")
            if message_type == "PROPOSE":
                proposals.append((message.sender, float(payload["weight"])))
            elif message_type == "ACCEPT":
                self._receive_accept(message.sender)
            elif message_type == "REJECT":
                self._receive_reject(message.sender)
            elif message_type == "CONFIRM":
                self._receive_confirm(message.sender, bool(payload.get("acknowledgement", False)))
            elif message_type == "CANCEL":
                self._receive_cancel(message.sender)

        if proposals:
            self._resolve_received_proposals(proposals)

    def _resolve_received_proposals(self, proposals: List[Tuple[int, float]]) -> None:
        """Choose one received proposal using local weight and deterministic tie-breaking."""
        if self.state.is_matched() or self.state.get(NodeState.TENTATIVE_PARTNER) is not None:
            for sender_id, _ in proposals:
                self._send_protocol_message(sender_id, "REJECT")
            return

        winner_id, winner_weight = max(proposals, key=lambda item: (item[1], -item[0]))
        self.state.begin_tentative_match(
            winner_id,
            self.local_time + self.proposal_timeout,
            proposing=False,
        )
        for sender_id, _ in proposals:
            if sender_id == winner_id:
                self._send_protocol_message(sender_id, "ACCEPT", weight=winner_weight)
            else:
                self._send_protocol_message(sender_id, "REJECT")

    def _receive_accept(self, sender_id: int) -> None:
        if (
            self.state.get(NodeState.TENTATIVE_PARTNER) != sender_id
            or not self.state.get(NodeState.PROPOSAL_PENDING)
        ):
            return
        self.state.begin_tentative_match(
            sender_id,
            self.local_time + self.proposal_timeout,
            proposing=False,
        )
        self._send_protocol_message(sender_id, "CONFIRM", acknowledgement=False)

    def _receive_reject(self, sender_id: int) -> None:
        if self.state.get(NodeState.TENTATIVE_PARTNER) == sender_id:
            self.state.clear_tentative_match()

    def _receive_confirm(self, sender_id: int, acknowledgement: bool) -> None:
        if self.state.get(NodeState.TENTATIVE_PARTNER) != sender_id:
            return
        if acknowledgement:
            self.state.set_matched_to(sender_id)
            self.state.clear_tentative_match()
            return

        self.state.set_matched_to(sender_id)
        self.state.clear_tentative_match()
        self._send_protocol_message(sender_id, "CONFIRM", acknowledgement=True)

    def _receive_cancel(self, sender_id: int) -> None:
        if self.state.get(NodeState.TENTATIVE_PARTNER) == sender_id:
            self.state.clear_tentative_match()

    def _expire_tentative_match_if_needed(self) -> None:
        partner_id = self.state.get(NodeState.TENTATIVE_PARTNER)
        deadline = self.state.get(NodeState.PROPOSAL_DEADLINE)
        if partner_id is None or deadline is None or self.local_time < deadline:
            return
        self.state.clear_tentative_match()
        self._send_protocol_message(partner_id, "CANCEL")

    def conflict_solution(self) -> None:
        """Start one locally selected negotiation; it never writes a final match."""
        if self.pending_proposals and not self.state.is_matched():
            best_neighbor, best_weight = max(
                self.pending_proposals.items(), key=lambda item: (item[1], -item[0])
            )
            self.start_proposal(best_neighbor, best_weight)
        self.pending_proposals.clear()

    def reset(self) -> None:
        """Reset node to initial state."""
        self.state = NodeState(self.id)
        self.round_number = 0
        self.finished = False
        self.local_metrics.reset()
        self.convergence_vote = None
        self.known_convergence_votes.clear()
        self.last_matching_weight = 0.0
        self.pending_proposals.clear()
        self.local_time = 0

    # ============================================================================
    # CONFIG GOSSIP PROTOCOL (Distributed Configuration Spreading)
    # ============================================================================

    def gossip_config(self) -> None:
        """Send current algorithm configuration to random neighbors.

        Nodes spread their algorithm configuration via gossip protocol,
        including available algorithms list (Phase 8 - NEW).
        Neighbors will accept if version is higher than their current.
        """
        neighbors = self.graph.neighbors()
        if not neighbors:
            return

        # Create config message payload with algorithm list metadata
        payload = self.algorithm_config.to_dict()
        payload["available_algorithms"] = self.algorithm_config.available_algorithms
        payload["algorithm_list_version"] = self.algorithm_config.algorithm_list_version

        # Create GossipMessage (for protocol abstraction)
        generic_msg = GossipMessage.config_gossip(
            sender_node_id=self.id,
            payload=payload,
            version=self.algorithm_config.version,
            round_num=self.round_number,
        )

        # Sample up to 3 neighbors to avoid flooding
        import random
        sample_size = min(3, len(neighbors))
        sampled_neighbors = random.sample(neighbors, sample_size)

        # Send to sampled neighbors through recipient-scoped transport.
        for neighbor in sampled_neighbors:
            msg = Message(
                sender=self.id,
                recipient=neighbor,
                payload=generic_msg.payload,  # Use GossipMessage payload
                round_num=self.round_number,
            )
            self.communicator.send_message(msg)

    def receive_config_gossip(self, msg: GossipMessage) -> None:
        """Receive and potentially adopt algorithm configuration from neighbor.

        Only accepts configuration if version > current version (version-based ordering).

        Args:
            msg: GossipMessage with subtype="config" and algorithm configuration
        """
        if msg.message_version > self.algorithm_config.version:
            # Adopt this configuration
            self.algorithm_config = DistributedAlgorithmConfig.from_dict(msg.payload)
            self.algorithm_config.version = msg.message_version

            # Update algorithm list if neighbor has newer list version (in payload)
            # Note: GossipMessage stores algorithm list version in payload if available
            new_list_version = msg.payload.get("algorithm_list_version", self.algorithm_config.algorithm_list_version)
            if new_list_version > self.algorithm_config.algorithm_list_version:
                new_algos = msg.payload.get("available_algorithms", [])
                if new_algos and new_algos != self.algorithm_config.available_algorithms:
                    self.algorithm_config.available_algorithms = new_algos.copy()
                    self.algorithm_config.algorithm_list_version = new_list_version
