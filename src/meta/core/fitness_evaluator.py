"""Fitness evaluator for scoring CanonicalVectors in genetic algorithm.

Supports both centralized and distributed execution modes.
"""

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.core.matching_merger import merge_matchings
from src.simulation.centralized_orchestrator import CentralizedOrchestrator


class FitnessEvaluator:
    """Evaluates fitness of a CanonicalVector using centralized or distributed orchestration.

    Fitness is measured as the total weight of the matching produced.
    Can run in:
    - Centralized mode (default): Fast, uses CentralizedOrchestrator
    - Distributed mode: Fully autonomous nodes with message passing, convergence voting, etc.
    """

    def __init__(self, use_distributed: bool = False, max_workers: int = 4) -> None:
        """Initialize fitness evaluator.

        Args:
            use_distributed: If True, use distributed orchestrator; if False, use centralized
            max_workers: Number of worker threads for distributed execution (ignored if centralized)
        """
        self.use_distributed = use_distributed
        self.max_workers = max_workers
        self.centralized_orchestrator = CentralizedOrchestrator()

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

        if self.use_distributed:
            return self._evaluate_distributed(graph, vector)
        else:
            return self._evaluate_centralized(graph, vector)

    def _evaluate_centralized(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Evaluate using centralized orchestrator (3 algorithms via parameterizers).

        Args:
            graph: GraphManager instance
            vector: CanonicalVector to evaluate

        Returns:
            float: Fitness score (matching weight)
        """
        from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer

        # Create parameterizers
        parameterizers = [
            UnifiedAlgorithmParameterizer("greedy"),
            UnifiedAlgorithmParameterizer("itai"),
            UnifiedAlgorithmParameterizer("luby"),
        ]

        # Run all 3 parameterizers to get their matchings
        matchings = []
        for parameterizer in parameterizers:
            try:
                matching = parameterizer.execute(graph, vector)
                matchings.append(matching)
            except Exception:
                matchings.append({})

        # Merge matchings via conflict resolution (keep highest-weight edges)
        final_matching = merge_matchings(matchings, graph)

        # Calculate weight as fitness score
        if final_matching:
            weight = 0.0
            for u, v in final_matching.items():
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

        orchestrator = DistributedOrchestrator(max_workers=self.max_workers)
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
        mode = "Distributed" if self.use_distributed else "Centralized"
        return f"FitnessEvaluator({mode})"
