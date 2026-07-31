"""Bootstrap and observation boundary for distributed node execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from src.communication.transport import InMemoryTransport
from src.config import DistributedAlgorithmConfig
from src.graph.graph_manager import GraphManager
from src.simulation.distributed_node import DistributedNode
from src.simulation.parallel_node_executor import ParallelNodeExecutor, RuntimeOutcome

if TYPE_CHECKING:
    from src.meta.core.canonical_vector import CanonicalVector


class DistributedOrchestrator:
    """Build node endpoints, start the scheduler, and collect observations only."""

    def __init__(self, max_workers: int = 4) -> None:
        self.executor = ParallelNodeExecutor(max_workers=max_workers)
        self._nodes: Dict[int, DistributedNode] = {}
        self._config: DistributedAlgorithmConfig | None = None
        self._transport: InMemoryTransport | None = None

    def start(
        self, graph: GraphManager, canonical_vector: CanonicalVector, pre_matched_nodes: set | None = None
    ) -> None:
        """Create the opaque transport and endpoint-owned node runtime."""
        is_valid, error = canonical_vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid canonical vector: {error}")

        node_ids = graph.vertices()
        transport = InMemoryTransport(node_ids)
        config = DistributedAlgorithmConfig.from_canonical_vector(canonical_vector)
        pre_matched_nodes = pre_matched_nodes or set()
        self._nodes = {
            node_id: DistributedNode(node_id, graph, algorithm_config=config, transport=transport)
            for node_id in node_ids
        }
        for node_id in pre_matched_nodes:
            self._nodes[node_id].finished = True
        self._config = config
        self._transport = transport

    def run(self, max_ticks: int) -> RuntimeOutcome:
        """Run only the safety-bounded scheduler; it has no convergence logic."""
        if self._config is None:
            raise RuntimeError("start() must be called before run()")
        return self.executor.run_until_idle(
            self._nodes,
            max_ticks=max_ticks,
            tick=lambda node: node.execute_distributed_round(),
        )

    def execute(
        self,
        graph: GraphManager,
        canonical_vector: CanonicalVector,
        pre_matched_nodes: set | None = None,
    ) -> Tuple[Dict[int, int], Dict]:
        """Execute a safety-bounded simulator run and return passive observations."""
        self.start(graph, canonical_vector, pre_matched_nodes)
        if self._config is None:
            raise RuntimeError("start() must create a runtime configuration")
        outcome = self.run(max_ticks=self._config.max_iterations * max(1, len(self._nodes)))
        matching, final_weight = self._collect_results(graph)
        if self._transport is None:
            raise RuntimeError("start() must create a transport")
        quiescent = not outcome.watchdog_exhausted and all(
            node.state.is_terminal() for node in self._nodes.values()
        )
        return matching, {
            "outcome": "quiescent" if quiescent else "watchdog_exhausted",
            "scheduled_ticks": outcome.scheduled_ticks,
            "active_node_ids": outcome.active_node_ids,
            "terminal_node_count": sum(
                node.state.is_terminal() for node in self._nodes.values()
            ),
            "iterations": outcome.scheduled_ticks,
            "final_weight": final_weight,
            "message_count": self._transport.stats()["messages_sent"],
            "config_fingerprint": self._config.vector_fingerprint,
            "max_iterations": self._config.max_iterations,
            "algorithm_names": list(self._config.available_algorithms),
            "central_algorithmic_decisions": 0,
        }

    def _collect_results(self, graph: GraphManager) -> Tuple[Dict[int, int], float]:
        reported = {
            node.id: node.state.get_matched_to()
            for node in self._nodes.values()
            if node.state.is_matched()
        }
        matching: Dict[int, int] = {}
        for node_id, partner_id in reported.items():
            if reported.get(partner_id) == node_id and node_id < partner_id:
                matching[node_id] = partner_id
        return matching, graph.calculate_matching_weight(matching)

    def name(self) -> str:
        return "DistributedOrchestrator"
