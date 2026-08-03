"""Fully distributed node for autonomous algorithm execution and coordination."""

from typing import Any, Callable, Dict, List, Tuple
from src.state.node import NodeState
from src.communication.message import Message
from src.communication.node_communicator import NodeCommunicator
from src.communication.transport import InMemoryTransport
from src.graph.graph_manager import GraphManager
from src.graph.local_graph import LocalGraph
from src.metrics.metrics_collector import MetricsCollector
from src.algorithms.proposal_policy import build_local_proposal_policies
from src.config import DistributedAlgorithmConfig
from src.simulation.endpoint_protocol import EndpointProtocol
from src.simulation.local_convergence import LocalConvergence
from src.simulation.local_node_context import LocalNodeContext


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
        lifecycle_observer: Callable[[int, str, int], None] | None = None,
    ):
        """Initialize a distributed node.

        Args:
            node_id: Unique node identifier
            shared_graph: Read-only reference to network topology
            algorithm_config: Immutable startup snapshot for local policies and protocol.
                             If None, creates default configuration.
        """
        self.id = node_id
        self.graph = LocalGraph(shared_graph, node_id)

        # Algorithm configuration is an immutable-at-startup snapshot.
        self.algorithm_config = algorithm_config or DistributedAlgorithmConfig()
        self.proposal_policies = build_local_proposal_policies(self.algorithm_config)

        # State (this node's algorithm state)
        self.state = NodeState(node_id)

        # Communication is a recipient-bound facade over a black-box transport.
        # A standalone node receives a private transport for local unit tests;
        # a runtime supplies one shared transport to every node.
        self.transport = transport or InMemoryTransport(shared_graph.vertices())
        self.communicator = NodeCommunicator(node_id, self.transport)
        self._lifecycle_observer = lifecycle_observer

        # Execution tracking
        self.round_number = 0
        self.finished = False

        # Local metrics
        self.local_metrics = MetricsCollector()

        # Conflict resolution state (pending proposals from neighbors)
        self.pending_proposals: Dict[int, float] = {}

        # Endpoint-owned matching protocol time.  This is deliberately local:
        # the executor may invoke a node, but does not advance its negotiations.
        self.local_time = 0
        self.proposal_timeout = 3
        self.endpoint_protocol = EndpointProtocol(
            self.id,
            self.graph,
            self.state,
            self.communicator.send_message,
            self.proposal_timeout,
        )
        self.convergence = LocalConvergence(
            self.id, self.graph, self.state, self.communicator.send_message,
            self.algorithm_config.convergence_threshold, self.algorithm_config.quorum_threshold,
        )
        # Track should_stop for autonomous loop (Phase 1)
        self.should_stop = False

    @property
    def convergence_vote(self) -> bool | None:
        return self.convergence.vote

    @property
    def known_convergence_votes(self) -> Dict[int, bool]:
        return self.convergence.known_votes

    @property
    def convergence_threshold(self) -> float:
        return self.convergence.threshold

    @property
    def quorum_threshold(self) -> float:
        return self.convergence.quorum_threshold

    def run_autonomous(self) -> None:
        """PHASE 1: Node's autonomous execution loop.

        This is the core autonomous agent behavior - the node runs its own complete
        algorithm loop without orchestrator control. Each node makes all its own
        decisions and only communicates via messages.

        """
        executed_iterations = 0

        # Run autonomous rounds until this node decides to stop or reaches its
        # node-owned safety limit.
        while (
            not self.should_stop
            and not self.finished
            and executed_iterations < self.algorithm_config.max_iterations
        ):
            # Execute one round (PHASE 0-5 all happen in execute_distributed_round)
            continue_running, _ = self.execute_distributed_round()
            executed_iterations += 1

            # Check if node should stop
            if not continue_running or self.finished:
                self.should_stop = True

        if executed_iterations >= self.algorithm_config.max_iterations and self.is_active():
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
        self._finish_if_terminal()
        self.advance_local_time()

    def start_proposal(self, neighbor_id: int, weight: float) -> None:
        """Start a match negotiation with one directly connected neighbor."""
        self.endpoint_protocol.start_proposal(
            neighbor_id, weight, self.local_time, self.round_number
        )

    def execute_distributed_round(self) -> Tuple[bool, str]:
        """
        NEW PHASED EXECUTION (Protocol-Driven Merge).

        Execute one round with structured phases:
        PHASE 0: Process incoming messages/proposals
        PHASE 1: Each algorithm proposes to neighbors (LOCAL SCOPE ONLY)
        PHASE 2: Accumulate all proposals into pending_proposals
        PHASE 3: ALWAYS call conflict_solution() (Protocol Consistency)
        PHASE 4: Send confirmation messages (Phase 2 two-phase commit)

        Returns:
            (continue_running, status_message)
        """
        if self.finished:
            return False, "already_finished"

        # PHASE 0: Process incoming messages
        messages = self.communicator.receive_messages()
        self._process_coordination_messages(messages)
        self._process_protocol_messages(messages)
        self._expire_tentative_match_if_needed()

        # PHASE 1: Get proposals from each cached local policy.
        context = self._create_context(messages)
        proposals_per_algorithm = {
            policy.name: policy.propose(context) for policy in self.proposal_policies
        }

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

        self._decide_convergence()
        self._gossip_convergence_vote()

        if self._finish_if_terminal():
            return False, "matched" if self.state.is_matched() else "terminal_unmatched"

        self.round_number += 1
        self.advance_local_time()
        return True, "continuing"


    def _create_context(self, messages: List[Message]) -> LocalNodeContext:
        """Create the restricted context available to this endpoint's algorithms."""
        return LocalNodeContext(
            node_id=self.id,
            graph=self.graph,
            state=self.state,
            messages=tuple(messages),
            config=self.algorithm_config,
            round_number=self.round_number,
            logical_time=self.local_time,
        )

    def _process_coordination_messages(self, messages: List[Message]) -> None:
        self.convergence.process_messages(messages)

    def _decide_convergence(self) -> None:
        self.convergence.decide(self.round_number)

    def _gossip_convergence_vote(self) -> None:
        self.convergence.gossip(self.round_number)

    def _should_stop_based_on_quorum(self) -> bool:
        return self.convergence.should_stop()

    def _finish_if_terminal(self) -> bool:
        """Finish only when this endpoint has no remaining matching work."""
        reason = None
        if self.state.is_matched():
            reason = "matched"
        elif self.endpoint_protocol.can_quiesce():
            self.state.mark_terminal_unmatched()
            reason = "terminal_unmatched"
        if reason is not None and not self.finished:
            self.finished = True
            if self._lifecycle_observer is not None:
                self._lifecycle_observer(self.id, reason, self.local_time)
        return self.finished

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


    def _process_protocol_messages(self, messages: List[Message]) -> None:
        """Delegate endpoint messages to the local protocol collaborator."""
        self.endpoint_protocol.process_messages(messages, self.local_time, self.round_number)

    def _expire_tentative_match_if_needed(self) -> None:
        """Delegate local tentative-match expiry."""
        self.endpoint_protocol.expire_if_needed(self.local_time, self.round_number)

    def conflict_solution(self) -> None:
        """Start one locally selected negotiation; it never writes a final match."""
        self.endpoint_protocol.select_proposal(
            self.pending_proposals, self.local_time, self.round_number
        )
        self.pending_proposals.clear()

    def reset(self) -> None:
        """Reset node to initial state."""
        self.state = NodeState(self.id)
        self.round_number = 0
        self.finished = False
        self.local_metrics.reset()
        self.endpoint_protocol.reset(self.state)
        self.convergence.reset(self.state)
        self.pending_proposals.clear()
        self.local_time = 0
