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


class DistributedOrchestrator:
    """Supervises autonomous nodes in truly distributed matching execution.

    Creates N DistributedNode instances (one per vertex) and coordinates their
    execution. Each node runs algorithms autonomously and coordinates via message
    passing. This is a TRUE distributed simulation.

    Centralized only: Graph, CanonicalVector, Transport layer.
    Distributed: State, algorithm execution, decision-making.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize orchestrator as supervisor.

        Args:
            max_workers: Number of concurrent worker threads for parallel node execution
        """
        self.executor = ParallelNodeExecutor(max_workers=max_workers)

    def execute(
        self,
        graph: GraphManager,
        canonical_vector: CanonicalVector,
        parameterizers: List = None,  # Nodes create their own parameterizers
    ) -> Tuple[Dict[int, int], Dict]:
        """Execute matching using autonomous distributed nodes (Phase 5 simplification).

        Pure round scheduler: Creates nodes, runs rounds, delivers messages, collects results.
        All decision logic moved to nodes. Orchestrator is stateless.

        Args:
            graph: Shared read-only graph
            canonical_vector: Shared immutable parameter chromosome
            parameterizers: Ignored (nodes create their own)

        Returns:
            Tuple of:
            - Dict[int, int]: Final matching from nodes (nodes handle symmetry)
            - dict: Metrics with keys:
              - iterations: int, number of rounds executed
              - final_weight: float, sum of matched edge weights
        """
        is_valid, error = canonical_vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid canonical vector: {error}")

        # Create one autonomous node per graph vertex
        nodes: Dict[int, DistributedNode] = {}
        for node_id in graph.vertices():
            nodes[node_id] = DistributedNode(node_id, graph)

        max_iterations = int(canonical_vector.max_iterations)
        iteration = 0

        # PHASE 6: Parallel node execution + pure round scheduler loop
        # All decision logic is in DistributedNode
        while iteration < max_iterations:
            # PHASE 6: Execute all nodes in parallel
            all_continue = self.executor.execute_all_nodes(nodes, canonical_vector)

            # Deliver messages between nodes
            self._deliver_all_messages(nodes)

            # Check termination: quorum voting or all nodes inactive
            stop_votes = sum(1 for node in nodes.values() if node.convergence_vote is True)
            if len(nodes) > 0 and (stop_votes / len(nodes)) > 0.5:
                break

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
        # Collect all outgoing messages from all nodes
        all_messages = []
        for node in nodes.values():
            outgoing = node.outbox.get_messages(node.id)
            all_messages.extend(outgoing)

        # Deliver to recipients
        for msg in all_messages:
            if msg.recipient in nodes:
                nodes[msg.recipient].inbox.send(msg)

    def _collect_results(
        self, nodes: Dict[int, DistributedNode], graph: GraphManager
    ) -> Tuple[Dict[int, int], float]:
        """Collect final matching from all nodes.

        Nodes handle symmetric matching via endpoint voting in Phase 4.
        Orchestrator just gathers results.

        Args:
            nodes: Dict of node_id -> DistributedNode
            graph: GraphManager for weight calculation

        Returns:
            Tuple of (matching dict, final weight)
        """
        final_matching: Dict[int, int] = {}
        for node in nodes.values():
            final_matching.update(node.get_matching())

        final_weight = graph.calculate_matching_weight(final_matching)
        return final_matching, final_weight

    def name(self) -> str:
        """Return orchestrator name."""
        return "DistributedOrchestrator"
