"""Minimal orchestrator for fully distributed system (observation-only)."""

import os
import time
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.graph.graph_manager import GraphManager
from src.simulation.distributed_node import DistributedNode
from src.metrics.metrics_collector import MetricsCollector


class DistributedSimulator:
    """
    Minimal simulator for fully distributed algorithm execution.

    This is NOT a central orchestrator. Its role is limited to:
    - Creating and managing node objects
    - Triggering round execution for each node
    - Delivering messages between nodes
    - Observing global metrics (for analysis only)
    - Extracting final results

    All actual coordination (convergence, metrics, decisions)
    happens within nodes via gossip messages.
    """

    def __init__(self, graph: GraphManager, algorithm, config: Dict[str, Any] = None):
        """Initialize distributed simulator.

        Args:
            graph: Network topology (shared, read-only)
            algorithm: MatchingAlgorithm to run on all nodes
            config: Configuration dict with optional keys:
                - max_rounds: Maximum execution rounds (default 1000)
                - convergence_threshold: Min improvement (default 0.05)
                - quorum_threshold: Min fraction to stop (default 0.5)
                - num_workers: Number of parallel threads (default: CPU count)
                - use_parallel: Enable parallel execution (default: True)
        """
        self.graph = graph
        self.algorithm = algorithm
        self.config = config or {}

        # Parallelization settings
        self.num_workers = self.config.get("num_workers", os.cpu_count() or 1)
        self.use_parallel = self.config.get("use_parallel", True)

        # Create nodes
        self.nodes: Dict[int, DistributedNode] = {}
        for node_id in graph.vertices():
            node = DistributedNode(node_id, graph)
            # Apply config
            if "convergence_threshold" in self.config:
                node.convergence_threshold = self.config["convergence_threshold"]
            if "quorum_threshold" in self.config:
                node.quorum_threshold = self.config["quorum_threshold"]
            self.nodes[node_id] = node

        # Global observation (for analysis, not coordination!)
        self.global_metrics = MetricsCollector()
        self.round_count = 0
        self.max_rounds = self.config.get("max_rounds", 1000)

        # Execution timing
        self.execution_time = 0.0
        self.parallel_execution_time = 0.0
        self.message_delivery_time = 0.0

    def _execute_nodes_parallel(self) -> int:
        """Execute all nodes in parallel using ThreadPoolExecutor.

        Returns:
            Number of active nodes executed
        """
        active_nodes = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all node executions
            futures = {}
            for node in self.nodes.values():
                if not node.finished:
                    future = executor.submit(node.execute, self.algorithm)
                    futures[future] = node.id
                    active_nodes += 1

            # Collect results as they complete (but all must complete before returning)
            for future in as_completed(futures):
                try:
                    continue_running, status = future.result()
                except Exception as e:
                    raise RuntimeError(f"Node execution failed: {str(e)}") from e

        return active_nodes

    def _execute_nodes_sequential(self) -> int:
        """Execute all nodes sequentially (fallback/debugging mode).

        Returns:
            Number of active nodes executed
        """
        active_nodes = 0
        for node in self.nodes.values():
            if not node.finished:
                continue_running, status = node.execute(self.algorithm)
                active_nodes += 1

        return active_nodes

    def run(self) -> Dict[str, Any]:
        """Run simulation until convergence or max rounds.

        Each round:
        1. Execute each node (algorithm + coordination)
        2. Deliver messages between nodes
        3. Observe global metrics
        4. Check if all nodes finished

        Returns:
            Dict with execution results
        """
        start_time = time.time()

        # Initialize nodes
        from src.state.node_state_store_adapter import NodeStateStoreAdapter
        for node in self.nodes.values():
            # Create adapter for algorithm initialization
            adapter = NodeStateStoreAdapter(node.state, node.id)
            self.algorithm.initialize_state(adapter, self.graph)

        # Run until convergence or max rounds
        while self.round_count < self.max_rounds:
            # Execute each node (parallel or sequential) with timing
            exec_start = time.time()
            if self.use_parallel:
                active_nodes = self._execute_nodes_parallel()
            else:
                active_nodes = self._execute_nodes_sequential()
            self.parallel_execution_time += time.time() - exec_start

            # Deliver messages between nodes with timing
            deliver_start = time.time()
            self._deliver_messages()
            self.message_delivery_time += time.time() - deliver_start

            # Observe metrics (non-intrusive)
            self._collect_global_metrics()

            # Check if all nodes finished
            all_finished = all(node.finished for node in self.nodes.values())
            if all_finished:
                break

            self.round_count += 1

        self.execution_time = time.time() - start_time
        return self._extract_results()

    def _deliver_messages(self) -> None:
        """Move messages from node outboxes to node inboxes.

        Can be extended to simulate:
        - Network delays (don't deliver immediately)
        - Packet loss (drop some messages)
        - Duplicates (deliver same message multiple times)
        - Partitions (don't deliver to some nodes)
        """
        # Collect all messages from all nodes
        messages_to_deliver = []
        for sender in self.nodes.values():
            # Get all messages from this node's outbox for all recipients
            for recipient_id in self.nodes.keys():
                messages = sender.outbox.peek_messages(recipient_id)
                messages_to_deliver.extend(messages)
            # Clear outbox after collecting
            for recipient_id in self.nodes.keys():
                sender.outbox.get_messages(recipient_id)

        # Deliver all messages
        for msg in messages_to_deliver:
            if msg.recipient in self.nodes:
                recipient = self.nodes[msg.recipient]
                recipient.inbox.send(msg)

    def _collect_global_metrics(self) -> None:
        """Aggregate metrics from all nodes (observation only).

        This is NON-INTRUSIVE - nodes still make all decisions independently.
        We're just observing.
        """
        total_messages = sum(
            n.local_metrics.total_messages for n in self.nodes.values()
        )
        active_nodes = sum(
            1 for n in self.nodes.values()
            if n.state.get("active", False) and not n.finished
        )
        matched_nodes = sum(
            1 for n in self.nodes.values() if n.state.is_matched()
        )

        self.global_metrics.record_round(
            round_num=self.round_count,
            messages_sent=total_messages,
            active_nodes=active_nodes,
            matched_nodes=matched_nodes,
        )

    def _extract_results(self) -> Dict[str, Any]:
        """Extract final results from all nodes.

        Returns:
            Dict with final matching, metrics, convergence info, and timing
        """
        # Collect final matching from all nodes
        final_matching = {}
        for node in self.nodes.values():
            if node.state.is_matched():
                matched_to = node.state.get_matched_to()
                final_matching[node.id] = matched_to

        # Extract metrics
        return {
            "final_matching": final_matching,
            "total_rounds": self.round_count,
            "total_messages": sum(n.local_metrics.total_messages for n in self.nodes.values()),
            "convergence_votes": {
                n.id: n.convergence_vote for n in self.nodes.values()
            },
            "all_finished": all(n.finished for n in self.nodes.values()),
            "node_metrics": {
                n.id: n.metrics_summary for n in self.nodes.values()
            },
            # Timing information
            "execution_time_seconds": self.execution_time,
            "parallel_execution_time_seconds": self.parallel_execution_time,
            "message_delivery_time_seconds": self.message_delivery_time,
            "parallelization": {
                "enabled": self.use_parallel,
                "num_workers": self.num_workers,
                "num_nodes": len(self.nodes)
            }
        }

    def get_node(self, node_id: int) -> DistributedNode:
        """Get a specific node (for testing/inspection).

        Args:
            node_id: Node identifier

        Returns:
            DistributedNode instance
        """
        return self.nodes[node_id]

    def get_all_nodes(self) -> Dict[int, DistributedNode]:
        """Get all nodes.

        Returns:
            Dict mapping node_id to DistributedNode
        """
        return self.nodes.copy()

    def reset(self) -> None:
        """Reset simulator to initial state."""
        for node in self.nodes.values():
            node.reset()
        self.global_metrics.reset()
        self.round_count = 0
        self.execution_time = 0.0
        self.parallel_execution_time = 0.0
        self.message_delivery_time = 0.0
