"""Fitness evaluator for scoring CanonicalVectors in genetic algorithm.

Uses distributed orchestrator for all execution: each algorithm runs independently
on autonomous nodes, then results are merged for best quality.
"""

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector


class FitnessEvaluator:
    """Evaluates fitness of a CanonicalVector using centralized or distributed orchestration.

    Fitness is measured as the total weight of the matching produced.
    Can run in:
    - Centralized mode (default): Fast, uses CentralizedOrchestrator
    - Distributed mode: Fully autonomous nodes with message passing, convergence voting, etc.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize fitness evaluator.

        Args:
            max_workers: Number of worker threads for parallel execution
        """
        self.max_workers = max_workers

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate fitness of a vector on given graph.

        Args:
            graph: GraphManager instance
            vector: CanonicalVector to evaluate

        Returns:
            float: Fitness score (matching weight, higher is better)

        Raises:
            ValueError: If vector is invalid
        """
        is_valid, error = vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid vector: {error}")

        return self._evaluate_centralized(graph, vector)

    def _evaluate_centralized(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate using distributed orchestrator (fully distributed execution).

        Uses DistributedOrchestrator with all algorithms running autonomously on nodes.
        This maintains fully distributed architecture while achieving quality matching.

        Args:
            graph: GraphManager instance
            vector: CanonicalVector to evaluate

        Returns:
            float: Fitness score (matching weight)
        """
        from src.meta.distributed.orchestrator import DistributedOrchestrator

        # Run distributed orchestrator with all algorithms (standard mode)
        orchestrator = DistributedOrchestrator(
            max_workers=self.max_workers,
            use_convergence_detection=False,
            min_iterations=10
        )

        matching, _ = orchestrator.execute(graph, vector)

        # Calculate weight as fitness score
        if matching:
            weight = 0.0
            for u, v in matching.items():
                if u < v:  # Count each edge once
                    weight += graph.get_edge_weight(u, v)
            return weight

        return 0.0

    def _evaluate_distributed(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate using distributed orchestrator (fully autonomous nodes, message passing).

        Args:
            graph: GraphManager instance
            vector: CanonicalVector to evaluate

        Returns:
            float: Fitness score (matching weight)
        """
        from src.meta.distributed.orchestrator import DistributedOrchestrator

        # Disable convergence detection for GA (too aggressive threshold)
        # Convergence detector is useful for real distributed systems but breaks GA optimization
        orchestrator = DistributedOrchestrator(
            max_workers=self.max_workers,
            use_convergence_detection=False
        )
        matching, metrics = orchestrator.execute(graph, vector)

        # Calculate weight as fitness score
        if matching:
            weight = 0.0
            for u, v in matching.items():
                if u < v:  # Count each edge once
                    weight += graph.get_edge_weight(u, v)
            return weight

        return 0.0

    def name(self) -> str:
        """Return evaluator name."""
        return "FitnessEvaluator"
