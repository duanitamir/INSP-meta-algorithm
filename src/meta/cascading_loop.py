"""Orchestrator for cascading algorithm execution with convergence detection."""

from typing import Dict, List, Tuple
from src.graph.graph_manager import GraphManager
from src.meta.canonical_vector import CanonicalVector
from src.meta.algorithm_parameterizer import AlgorithmParameterizer
from src.meta.conflict_resolver import ConflictResolver


class CascadingLoop:
    """Orchestrates repeated rounds of algorithm execution with convergence detection.

    Runs parameterizers in sequence, merges results, and stops when convergence
    threshold is met or maximal matching is found.
    """

    def __init__(self, conflict_resolver: ConflictResolver) -> None:
        """Initialize cascading loop with conflict resolver.

        Args:
            conflict_resolver: ConflictResolver instance for merging matchings
        """
        self.conflict_resolver = conflict_resolver

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
        """Execute cascading loop until convergence or max iterations.

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

        all_matched: Dict[int, int] = {}
        previous_weight = 0.0
        improvements: List[float] = []

        max_iters = int(canonical_vector.max_iterations)
        convergence_thresh = canonical_vector.convergence_threshold

        for iteration in range(max_iters):
            # Create reduced graph (remove already matched nodes)
            remaining_nodes = frozenset(graph.vertices() - frozenset(all_matched.keys()))

            if len(remaining_nodes) == 0:
                break

            working_graph = graph.get_subgraph(remaining_nodes)

            # Run all parameterizers on reduced graph
            matchings = [param.execute(working_graph, canonical_vector) for param in parameterizers]

            # Merge with conflict resolution (first 3 are Greedy, Itai, Luby)
            # This check is just a formality and should always be true if parameterizers are provided correctly
            if len(matchings) >= 3:
                merged = self.conflict_resolver.resolve(
                    matchings[0], matchings[1], matchings[2], working_graph
                )
            else:
                merged = matchings[0] if matchings else {}

            # Calculate weight
            current_weight = working_graph.calculate_matching_weight(merged)

            # Calculate improvement
            if iteration > 0:
                improvement = (current_weight - previous_weight) / (previous_weight + 1e-10)
                improvements.append(improvement)

                # Check convergence
                if improvement < convergence_thresh:
                    break
            else:
                improvements.append(0.0)

            # Add new matches to cumulative result
            all_matched.update(merged)

            previous_weight = current_weight

        # Guarantee maximal matching: if not maximal, run recovery iteration
        # Also try additional iterations even if convergence threshold met, to find remaining edges
        remaining_nodes = frozenset(graph.vertices() - frozenset(all_matched.keys()))
        while len(remaining_nodes) >= 2 and not self._is_maximal_matching(all_matched, graph):
            working_graph = graph.get_subgraph(remaining_nodes)
            matchings = [param.execute(working_graph, canonical_vector) for param in parameterizers]

            if len(matchings) >= 3:
                merged = self.conflict_resolver.resolve(
                    matchings[0], matchings[1], matchings[2], working_graph
                )
            else:
                merged = matchings[0] if matchings else {}

            if not merged:  # No more edges found
                break

            all_matched.update(merged)
            remaining_nodes = frozenset(graph.vertices() - frozenset(all_matched.keys()))

        return (
            all_matched,
            {
                "iterations": len(improvements),
                "final_weight": graph.calculate_matching_weight(all_matched),
                "improvements": improvements,
            },
        )

    def name(self) -> str:
        """Return loop name."""
        return "CascadingLoop"
