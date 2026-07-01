"""Standalone node service for distributed algorithm execution.

Replaces both Scheduler (Phase 1 legacy) and DistributedSimulator.
Each node runs autonomously, managing its own round loop and termination.
"""

from typing import Dict, Any, Tuple
from src.simulation.distributed_node import DistributedNode
from src.communication.transport_driver import TransportDriver
from src.communication.drivers.in_memory_driver import InMemoryDriver
from src.graph.graph_manager import GraphManager


class NodeService:
    """Standalone service for autonomous node execution.

    Manages:
    - Round loop (no central Scheduler)
    - Message sending/receiving (via TransportDriver)
    - Termination decision (via local quorum voting)

    Each node process runs one NodeService instance.
    """

    def __init__(
        self,
        node_id: int,
        graph: GraphManager,
        transport_driver: TransportDriver = None,
        max_rounds: int = 1000,
    ):
        """Initialize node service.

        Args:
            node_id: Unique node identifier
            graph: Shared read-only graph
            transport_driver: Message transport (default: in-memory)
            max_rounds: Maximum execution rounds (safety limit)
        """
        self.node_id = node_id
        self.graph = graph
        self.max_rounds = max_rounds

        # Setup transport (default to in-memory for testing)
        self.transport = transport_driver or InMemoryDriver(graph)

        # Create autonomous node
        self.node = DistributedNode(node_id, graph)

        # Execution state
        self.rounds_executed = 0
        self.finished = False

    def execute_round(self, algorithm) -> Tuple[bool, str]:
        """Execute one round of the algorithm.

        Args:
            algorithm: MatchingAlgorithm to run

        Returns:
            (continue_running, status_message)
        """
        if self.finished or self.rounds_executed >= self.max_rounds:
            return False, "max_rounds_exceeded"

        # Execute node (includes: process messages, run algorithm, decide convergence)
        should_continue, status = self.node.execute(algorithm)

        # Send outgoing messages via transport
        if self.node.outbox:
            messages = self.node.outbox.get_messages(self.node_id)
            self.transport.send_batch(messages)

        # Receive incoming messages from transport
        received = self.transport.receive_messages(self.node_id)
        if received:
            for msg in received:
                self.node.inbox.send(msg)

        self.rounds_executed += 1

        if not should_continue:
            self.finished = True
            return False, status

        return True, "round_completed"

    def run(self, algorithm) -> Dict[str, Any]:
        """Run algorithm until convergence or max rounds.

        Args:
            algorithm: MatchingAlgorithm to run

        Returns:
            Execution results dict
        """
        status = "running"
        while not self.finished and self.rounds_executed < self.max_rounds:
            should_continue, status = self.execute_round(algorithm)
            if not should_continue:
                self.finished = True
                break

        # Mark as finished if we hit max rounds
        if self.rounds_executed >= self.max_rounds:
            self.finished = True
            status = "max_rounds_exceeded"

        return {
            "node_id": self.node_id,
            "rounds_executed": self.rounds_executed,
            "finished": self.finished,
            "final_status": status,
            "matching_weight": self.node.local_metrics.total_messages,  # Placeholder
        }

    @property
    def is_finished(self) -> bool:
        """Check if node has converged."""
        return self.finished

    @property
    def round_number(self) -> int:
        """Get current round number."""
        return self.rounds_executed
