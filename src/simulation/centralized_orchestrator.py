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
from src.state.state_store import StateStore
from src.communication.message_queue import MessageQueue
from src.simulation.algorithm_context import AlgorithmContext
from src.algorithms.implementations.greedy_matching import GreedyMatching
from src.algorithms.implementations.itai_israeli import ItaiIsraeliMaximalMatching
from src.algorithms.implementations.luby_randomized import LubyRandomizedMatching


class CentralizedOrchestrator:
    """Runs Phase 1 algorithms on centralized state.

    This is used by fitness evaluator to run algorithms during GA optimization.
    Uses traditional centralized approach (StateStore + manual loop) for Phase 1 algorithms.
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.graph = None
        self.state_store = None

    def setup(self, graph: GraphManager) -> None:
        """Setup orchestrator with graph and state.

        Args:
            graph: Shared graph (centralized, read-only)
        """
        self.graph = graph
        self.state_store = StateStore(graph)

    def run_until_convergence(self, max_rounds: int = 100, vector: CanonicalVector | None = None) -> Dict[int, int]:
        """Run three algorithms with centralized state, merge results.

        Runs Greedy, Itai-Israeli, and Luby sequentially on centralized StateStore.
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

        greedy = GreedyMatching()
        itai = ItaiIsraeliMaximalMatching(timeout_rounds=int(vector.itai_timeout_rounds))
        luby = LubyRandomizedMatching(activation_probability=vector.luby_base_probability)

        matchings = []

        for algorithm in [greedy, itai, luby]:
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
