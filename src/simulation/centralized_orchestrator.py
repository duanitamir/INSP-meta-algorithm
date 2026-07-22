"""Centralized orchestrator for running algorithms with centralized state.

Phase 1 algorithms (Greedy, Itai, Luby) require global state visibility to work correctly.
This orchestrator provides that via a centralized StateStore.

The orchestrator:
1. Creates centralized StateStore (for all nodes' state visibility)
2. Runs algorithms by manually looping through rounds
3. Merges results via conflict resolution
4. Repeats until convergence
"""

from typing import Any, Dict
from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.state.store import StateStore
from src.communication.message_queue import MessageQueue
from src.simulation.algorithm_context import AlgorithmContext
from src.config import ExperimentConfig


class CentralizedOrchestrator:
    """Runs Phase 1 algorithms on centralized state.

    This is used by fitness evaluator to run algorithms during GA optimization.
    Uses traditional centralized approach (StateStore + manual loop) for Phase 1 algorithms.
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.graph = None
        self.state_store = None
        self.experiment_config = None

    def setup(self, graph: GraphManager, config: ExperimentConfig | None = None) -> None:
        """Setup orchestrator with graph and state.

        Args:
            graph: Shared graph (centralized, read-only)
            config: Experiment configuration (optional, uses defaults if None)
        """
        self.graph = graph
        self.state_store = StateStore(graph)
        self.experiment_config = config or ExperimentConfig()

    def run_until_convergence(self, max_rounds: int = 100, vector: CanonicalVector | None = None) -> Dict[int, int]:
        """Run all registered algorithms with centralized state, merge results.

        Runs all algorithms from registry sequentially on independent StateStore copies.
        Merges results via conflict resolution (highest-weight edges win).

        Args:
            max_rounds: Maximum rounds to execute
            vector: CanonicalVector with algorithm parameters (uses defaults if None)

        Returns:
            Final matching dict {node_id -> matched_partner}
        """
        # Use provided vector or create default
        if vector is None:
            vector = CanonicalVector()

        # Get all algorithms from registry (100% agnostic - no hardcoding)
        from src.meta.core.algorithm_registry import AlgorithmRegistry
        from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder

        registry = AlgorithmRegistry.instance()
        available_algos = registry.all_algorithm_names()

        if not available_algos:
            raise RuntimeError("No algorithms registered")

        matchings = []

        # Run each algorithm on independent state (no interference between algorithms)
        for algo_name in available_algos:
            algo_class = AlgorithmRegistryBuilder.get_class(algo_name)
            if not algo_class:
                continue

            # Extract parameters for this algorithm from vector
            algo_params = registry.get_algorithm_parameters(algo_name)
            params = {}
            for param_name in (algo_params or {}).keys():
                full_name = f"{algo_name}_{param_name}"
                value = vector.get(full_name)
                if value is not None:
                    params[param_name] = value

            algorithm = algo_class(parameters=params if params else None)

            # Each algorithm runs on INDEPENDENT state copy (no interference)
            # This prevents earlier algorithms from consuming nodes needed by later ones
            independent_state = StateStore(self.graph)
            matching = self._run_single_algorithm_independent(algorithm, max_rounds, independent_state)
            matchings.append(matching)

        # Merge matchings via conflict resolution
        final_matching = self._merge_matchings(matchings)
        return final_matching

    def _run_single_algorithm_independent(self, algorithm: Any, max_rounds: int, state_store: StateStore) -> Dict[int, int]:
        """Run single algorithm on independent state copy.

        Args:
            algorithm: MatchingAlgorithm instance
            max_rounds: Maximum rounds
            state_store: Independent StateStore for this algorithm

        Returns:
            Matching dict
        """
        message_queue = MessageQueue(self.graph)
        round_num = 0

        for _ in range(max_rounds):
            all_continue = []

            # Each node executes
            for node_id in self.graph.vertices():
                node_state = state_store.get_node_state(node_id)
                messages = message_queue.get_messages(node_id)

                context = AlgorithmContext(
                    graph=self.graph,
                    state_store=state_store,
                    round_num=round_num
                )

                try:
                    new_state, out_messages = algorithm.node_behavior(
                        node_id,
                        node_state,
                        messages,
                        context
                    )
                    state_store.update_node_state(node_id, new_state)
                    if out_messages:
                        message_queue.send_batch(out_messages)
                    all_continue.append(True)
                except Exception:
                    all_continue.append(False)

            # Check convergence
            if not any(all_continue):
                break

            round_num += 1

        # Extract matching from independent state store
        matching = {}
        for node_id in self.graph.vertices():
            node_state = state_store.get_node_state(node_id)
            if node_state.is_matched():
                matched_to = node_state.get_matched_to()
                matching[node_id] = matched_to
        return matching

    def _run_single_algorithm(self, algorithm: Any, max_rounds: int) -> Dict[int, int]:
        """Run single algorithm manually through rounds.

        Args:
            algorithm: MatchingAlgorithm instance
            max_rounds: Maximum rounds

        Returns:
            Matching dict
        """
        message_queue = MessageQueue(self.graph)
        round_num = 0

        for _ in range(max_rounds):
            all_continue = []

            # Each node executes
            for node_id in self.graph.vertices():
                node_state = self.state_store.get_node_state(node_id)
                messages = message_queue.get_messages(node_id)

                context = AlgorithmContext(
                    graph=self.graph,
                    state_store=self.state_store,
                    round_num=round_num
                )

                try:
                    new_state, out_messages = algorithm.node_behavior(
                        node_id,
                        node_state,
                        messages,
                        context
                    )
                    self.state_store.update_node_state(node_id, new_state)
                    if out_messages:
                        message_queue.send_batch(out_messages)
                    all_continue.append(True)
                except Exception:
                    # If algorithm fails, stop
                    all_continue.append(False)

            # Check convergence
            if not any(all_continue):
                break

            round_num += 1

        return self._extract_matching()

    def _extract_matching(self) -> Dict[int, int]:
        """Extract matching from centralized state store.

        Returns:
            Matching dict {node_id -> matched_partner}
        """
        matching = {}
        for node_id in self.graph.vertices():
            node_state = self.state_store.get_node_state(node_id)
            if node_state.is_matched():
                matched_to = node_state.get_matched_to()
                matching[node_id] = matched_to
        return matching

    def _merge_matchings(self, matchings: list) -> Dict[int, int]:
        """Merge multiple matchings via agreement + weight ranking.

        Prioritizes edges agreed upon by multiple algorithms, breaks ties by weight.
        Ensures symmetric matching.

        Args:
            matchings: List of matching dicts

        Returns:
            Merged matching dict
        """
        if not matchings:
            return {}

        # Collect all edges with weights and proposal counts
        edge_proposals = {}
        for matching in matchings:
            for u, v in matching.items():
                if u is None or v is None:
                    continue
                edge = tuple(sorted([u, v]))
                if edge not in edge_proposals:
                    edge_proposals[edge] = {"weights": [], "count": 0}
                weight = self.graph.get_edge_weight(u, v)
                edge_proposals[edge]["weights"].append(weight)
                edge_proposals[edge]["count"] += 1

        # Sort by: (proposal_count DESC, max_weight DESC)
        # Prefer edges found by multiple algorithms, break ties by weight
        final_matching = {}
        used_nodes = set()
        for edge in sorted(
            edge_proposals.keys(),
            key=lambda e: (
                edge_proposals[e]["count"],  # Algorithms agreeing (desc)
                max(edge_proposals[e]["weights"])  # Highest weight (desc)
            ),
            reverse=True,
        ):
            u, v = edge
            if u not in used_nodes and v not in used_nodes:
                final_matching[u] = v
                final_matching[v] = u
                used_nodes.add(u)
                used_nodes.add(v)

        return final_matching
