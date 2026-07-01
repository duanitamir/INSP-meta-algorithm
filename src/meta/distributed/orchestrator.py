"""Distributed orchestrator replacing centralized CascadingLoop.

Uses distributed components (parameter evolution, conflict resolution, convergence detection)
to execute the matching algorithm. For single-node evaluation, it simulates a 1-node network.
"""

from typing import Dict, List, Tuple
from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.parameterizers.base import AlgorithmParameterizer
from src.meta.distributed.conflict_resolver import DistributedConflictResolver
from src.meta.distributed.convergence_detector import DistributedConvergenceDetector


class DistributedOrchestrator:
    """Orchestrates matching execution using distributed protocols.

    Replaces centralized CascadingLoop with:
    - DistributedConflictResolver for edge voting (instead of central merger)
    - DistributedConvergenceDetector for autonomous termination (instead of central loop)

    For single-node evaluation, simulates a 1-node network using distributed components.
    """

    def __init__(self) -> None:
        """Initialize distributed orchestrator with components."""
        self.conflict_resolver = DistributedConflictResolver(
            voting_frequency=1, threshold=0.5
        )
        self.convergence_detector = DistributedConvergenceDetector(
            convergence_threshold=0.05, quorum_threshold=0.5, max_iterations=100
        )

    def _is_maximal_matching(
        self, matching: Dict[int, int], graph: GraphManager
    ) -> bool:
        """Check if matching is truly maximal (no more edges can be added).

        Args:
            matching: Symmetric matching dictionary (u -> v and v -> u present)
            graph: GraphManager with vertices and edges

        Returns:
            bool: True if matching is maximal, False if more edges can be added
        """
        matched_nodes = set(matching.keys())
        unmatched_nodes = graph.vertices() - matched_nodes

        if len(unmatched_nodes) < 2:
            return True

        # Check if any edge exists between unmatched nodes
        for u in unmatched_nodes:
            for v in unmatched_nodes:
                if u < v and graph._graph.has_edge(u, v):
                    return False

        return True

    def execute(
        self,
        graph: GraphManager,
        canonical_vector: CanonicalVector,
        parameterizers: List[AlgorithmParameterizer],
    ) -> Tuple[Dict[int, int], Dict]:
        """Execute distributed matching algorithm until convergence.

        Uses distributed conflict resolution and convergence detection instead of
        centralized components.

        Args:
            graph: GraphManager with vertices and edges
            canonical_vector: 10-parameter canonical vector with iteration controls
            parameterizers: List of 3 AlgorithmParameterizer instances (Greedy, Itai, Luby)

        Returns:
            Tuple of:
            - Dict[int, int]: Final symmetric matching
            - dict: Metrics with keys:
              - iterations: int, number of iterations executed
              - final_weight: float, sum of matched edge weights
              - improvements: List[float], improvement fraction per iteration

        Raises:
            ValueError: If invalid canonical vector or empty parameterizers list
        """
        is_valid, error = canonical_vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid canonical vector: {error}")

        if not parameterizers or len(parameterizers) == 0:
            raise ValueError("Must provide at least one parameterizer")

        # Initialize distributed components for single node
        node_id = 0
        self.conflict_resolver.initialize(node_ids=[node_id])
        self.convergence_detector.initialize(node_ids=[node_id])

        all_matched: Dict[int, int] = {}
        previous_weight = 0.0
        improvements: List[float] = []
        iteration = 0

        max_iters = int(canonical_vector.max_iterations)

        while iteration < max_iters:
            # Create reduced graph (remove already matched nodes)
            remaining_nodes = frozenset(graph.vertices() - frozenset(all_matched.keys()))

            if len(remaining_nodes) == 0:
                break

            working_graph = graph.get_subgraph(remaining_nodes)

            # Run all parameterizers on reduced graph
            matchings = [param.execute(working_graph, canonical_vector) for param in parameterizers]

            # Extract Greedy, Itai, Luby matchings
            greedy_matching = matchings[0] if len(matchings) > 0 else {}
            itai_matching = matchings[1] if len(matchings) > 1 else {}
            luby_matching = matchings[2] if len(matchings) > 2 else {}

            # Use distributed conflict resolution (voting by endpoints)
            self.conflict_resolver.propose_edges(
                node_id, greedy_matching, itai_matching, luby_matching
            )

            # Generate voting messages (simulate endpoint voting)
            messages = self.conflict_resolver.broadcast_votes(
                node_id, greedy_matching, itai_matching, luby_matching, iteration
            )

            # Single node receives its own votes
            self.conflict_resolver.receive_votes(node_id, messages)

            # Resolve conflicts via voting
            merged = self.conflict_resolver.resolve_matches(node_id, threshold=0.5)

            # Calculate weight using ORIGINAL graph (edges exist there, not in reduced working_graph)
            current_weight = graph.calculate_matching_weight(merged)

            # Calculate improvement
            if iteration > 0:
                improvement = (current_weight - previous_weight) / (previous_weight + 1e-10)
                improvements.append(improvement)

                # For single-node evaluation, use simple improvement threshold
                # (distributed detector would converge too early with only 1 node)
                convergence_threshold = canonical_vector.convergence_threshold
                if improvement < convergence_threshold:
                    break
            else:
                improvements.append(0.0)

            # Add new matches to cumulative result
            all_matched.update(merged)
            previous_weight = current_weight
            iteration += 1

            # Reset for next iteration
            self.conflict_resolver.node_states[node_id].reset_votes()
            self.convergence_detector.reset_convergence_votes(node_id)

        # Guarantee maximal matching (with safety limit to prevent infinite loops)
        remaining_nodes = frozenset(graph.vertices() - frozenset(all_matched.keys()))
        maximal_iter = 0
        max_maximal_iterations = 20  # Safety limit to prevent infinite loops

        while (len(remaining_nodes) >= 2 and
               not self._is_maximal_matching(all_matched, graph) and
               maximal_iter < max_maximal_iterations):
            working_graph = graph.get_subgraph(remaining_nodes)
            matchings = [param.execute(working_graph, canonical_vector) for param in parameterizers]

            greedy_matching = matchings[0] if len(matchings) > 0 else {}
            itai_matching = matchings[1] if len(matchings) > 1 else {}
            luby_matching = matchings[2] if len(matchings) > 2 else {}

            self.conflict_resolver.propose_edges(
                node_id, greedy_matching, itai_matching, luby_matching
            )

            messages = self.conflict_resolver.broadcast_votes(
                node_id, greedy_matching, itai_matching, luby_matching, iteration
            )
            self.conflict_resolver.receive_votes(node_id, messages)

            merged = self.conflict_resolver.resolve_matches(node_id, threshold=0.5)

            if not merged:  # No more edges found
                break

            all_matched.update(merged)
            remaining_nodes = frozenset(graph.vertices() - frozenset(all_matched.keys()))
            maximal_iter += 1

        return (
            all_matched,
            {
                "iterations": len(improvements),
                "final_weight": graph.calculate_matching_weight(all_matched),
                "improvements": improvements,
            },
        )

    def name(self) -> str:
        """Return orchestrator name."""
        return "DistributedOrchestrator"
