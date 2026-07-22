"""Distributed orchestrator - supervises autonomous nodes instead of controlling them.

Phase 6: Parallel Node Execution
- Creates one DistributedNode per graph vertex (autonomous execution)
- Nodes execute in parallel via ParallelNodeExecutor (multi-core efficiency)
- Nodes coordinate via message passing
- Nodes run ALL 3 algorithms autonomously
- Nodes make own decisions
- Orchestrator supervises rounds, delivers messages, collects results

Only centralized: Graph (read-only), CanonicalVector (immutable), Transport (communication).
"""

from typing import Dict, List, Tuple
from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.simulation.distributed_node import DistributedNode
from src.simulation.parallel_node_executor import ParallelNodeExecutor
from src.meta.distributed.convergence_detector import DistributedConvergenceDetector
from src.config import ExperimentConfig, DistributedAlgorithmConfig


class DistributedOrchestrator:
    """Supervises autonomous nodes in truly distributed matching execution.

    Creates N DistributedNode instances (one per vertex) and coordinates their
    execution. Each node runs algorithms autonomously and coordinates via message
    passing. This is a TRUE distributed simulation.

    Centralized only: Graph, CanonicalVector, Transport layer.
    Distributed: State, algorithm execution, decision-making.
    """

    def __init__(self, max_workers: int = 4, use_convergence_detection: bool = False, min_iterations: int = 0) -> None:
        """Initialize orchestrator as supervisor.

        Args:
            max_workers: Number of concurrent worker threads for parallel node execution
            use_convergence_detection: If True, use convergence detector for early termination
                                      (good for autonomous networks, bad for GA optimization)
            min_iterations: Minimum iterations before allowing early termination (for synchronization)
        """
        self.executor = ParallelNodeExecutor(max_workers=max_workers)
        self.use_convergence_detection = use_convergence_detection
        self.min_iterations = min_iterations

    def execute(
        self,
        graph: GraphManager,
        canonical_vector: CanonicalVector,
        parameterizers: List = None,  # Ignored (nodes create their own)
        pre_matched_nodes: set = None,  # Nodes already matched in previous cascades
        algorithm_to_run: str = None,  # If specified, run ONLY this algorithm (for independent execution)
    ) -> Tuple[Dict[int, int], Dict]:
        """Execute matching using autonomous distributed nodes.

        Can run in two modes:
        1. algorithm_to_run=None: Run all algorithms simultaneously (default)
        2. algorithm_to_run=<name>: Run single algorithm independently (used for merge-based approach)

        Args:
            graph: Shared read-only graph
            canonical_vector: Shared immutable parameter chromosome
            parameterizers: Ignored (nodes create their own)
            pre_matched_nodes: Set of node IDs already matched in previous cascades (optional)
            algorithm_to_run: If specified, nodes only run this algorithm (for independent execution + merge)

        Returns:
            Tuple of:
            - Dict[int, int]: Final matching from nodes
            - dict: Metrics with keys:
              - iterations: int, number of rounds executed
              - final_weight: float, sum of matched edge weights
        """
        is_valid, error = canonical_vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid canonical vector: {error}")

        # Initialize convergence detector if enabled (disabled by default for GA)
        convergence_detector = None
        if self.use_convergence_detection:
            convergence_detector = DistributedConvergenceDetector(
                convergence_threshold=0.05,
                quorum_threshold=0.5,
                gossip_frequency=1,
                max_iterations=int(canonical_vector.get("max_iterations") or 100),
            )
            convergence_detector.initialize(graph.vertices())

        # Create algorithm config from canonical vector (distributed to all nodes)
        algorithm_config = DistributedAlgorithmConfig.from_canonical_vector(canonical_vector)

        # Create one autonomous node per graph vertex with shared algorithm config
        nodes: Dict[int, DistributedNode] = {}
        pre_matched_nodes = pre_matched_nodes or set()

        for node_id in graph.vertices():
            node = DistributedNode(node_id, graph, algorithm_config=algorithm_config)
            if convergence_detector is not None:
                node.convergence_detector = convergence_detector

            # For cascading: mark already-matched nodes as finished
            if node_id in pre_matched_nodes:
                node.finished = True

            nodes[node_id] = node

        max_iterations = int(canonical_vector.get("max_iterations") or 100)
        iteration = 0

        # Parallel node xecution + pure round scheduler loop
        # All decision logic is in DistributedNode
        while iteration < max_iterations:
            # Execute all nodes in parallel
            all_continue = self.executor.execute_all_nodes(nodes, canonical_vector)

            # Config gossip (spread algorithm config to neighbors every 5 rounds)
            if iteration % 5 == 0:
                for node in nodes.values():
                    node.gossip_config()

            # Deliver messages between nodes (includes config gossip messages)
            self._deliver_all_messages(nodes)

            # Check termination: only allow after min_iterations for synchronization
            if iteration >= self.min_iterations:
                # Quorum voting for convergence
                stop_votes = sum(1 for node in nodes.values() if node.convergence_vote is True)
                if len(nodes) > 0 and (stop_votes / len(nodes)) > 0.5:
                    break

                # All nodes inactive
                if not any(all_continue):
                    break

            iteration += 1

        # Collect results (nodes already handle symmetric matching via endpoint voting)
        matching, final_weight = self._collect_results(nodes, graph)

        return (
            matching,
            {
                "iterations": iteration,
                "final_weight": final_weight,
            },
        )

    def _deliver_all_messages(self, nodes: Dict[int, DistributedNode]) -> None:
        """Deliver all messages between nodes (simulates network transport).

        Args:
            nodes: Dict of node_id -> DistributedNode
        """
        # Collect all outgoing messages from all nodes' outboxes
        all_messages = []
        for node in nodes.values():
            # Get all pending messages from node's outbox
            outgoing = node.outbox.get_all_pending_messages()
            all_messages.extend(outgoing)

        # Deliver to recipients
        for msg in all_messages:
            if msg.recipient in nodes:
                nodes[msg.recipient].inbox.send(msg)

        # Clear outboxes after delivery
        for node in nodes.values():
            node.outbox.clear_all_messages()

    def _collect_results(
        self, nodes: Dict[int, DistributedNode], graph: GraphManager
    ) -> Tuple[Dict[int, int], float]:
        """Collect final matching from all nodes.

        Validates symmetric matching: only include edges where BOTH nodes
        agree they're matched to each other.

        Args:
            nodes: Dict of node_id -> DistributedNode
            graph: GraphManager for weight calculation

        Returns:
            Tuple of (matching dict in canonical form, final weight)
        """
        # Collect node->neighbor mappings
        node_matching = {}
        for node in nodes.values():
            node_id = node.id
            local_match = node.get_matching()
            if local_match and node_id in local_match:
                node_matching[node_id] = local_match[node_id]

        # Build final matching: only include edges both nodes agree on
        final_matching: Dict[int, int] = {}
        edges_added = set()

        for node_id, matched_to in node_matching.items():
            # Check if the other node also reports the same match
            if matched_to in node_matching and node_matching[matched_to] == node_id:
                # Both nodes agree on this edge - add in canonical form
                smaller, larger = min(node_id, matched_to), max(node_id, matched_to)
                pair = (smaller, larger)

                if pair not in edges_added:
                    edges_added.add(pair)
                    final_matching[smaller] = larger

        final_weight = graph.calculate_matching_weight(final_matching)
        return final_matching, final_weight

    def name(self) -> str:
        """Return orchestrator name."""
        return "DistributedOrchestrator"
