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
        """Evaluate using centralized orchestrator (independent algorithm execution).

        Uses CentralizedOrchestrator which runs each algorithm on independent StateStore copies.
        This prevents algorithm interference while ensuring parameter differentiation.

        Args:
            graph: GraphManager instance
            vector: CanonicalVector to evaluate

        Returns:
            float: Fitness score (matching weight)
        """
        from src.simulation.centralized_orchestrator import CentralizedOrchestrator
        from src.config import ExperimentConfig

        # Run centralized orchestrator with all algorithms (each on independent state)
        # max_rounds=3 balances parameter differentiation (4.9%) with speed (0.21s)
        # Higher values (10+) cause convergence trap where all algorithms converge to same result
        orchestrator = CentralizedOrchestrator()
        orchestrator.setup(graph, ExperimentConfig(max_rounds=3))
        matching = orchestrator.run_until_convergence(max_rounds=3, vector=vector)

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
