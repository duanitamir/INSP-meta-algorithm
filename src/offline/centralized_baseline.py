"""Explicit centralized reference evaluator for offline comparison only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import ExperimentConfig
from src.graph.graph_manager import GraphManager
from src.meta.core.vector_evaluator import EvaluationResult
from src.simulation.centralized_orchestrator import CentralizedOrchestrator

if TYPE_CHECKING:
    from src.meta.core.canonical_vector import CanonicalVector


class CentralizedBaselineEvaluator:
    """Run the legacy centralized algorithm only as an explicit offline baseline."""

    mode = "centralized_baseline"

    def __init__(self, max_rounds: int = 3) -> None:
        self.max_rounds = max_rounds

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> EvaluationResult:
        """Evaluate one vector through centralized reference machinery."""
        is_valid, error = vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid vector: {error}")

        orchestrator = CentralizedOrchestrator()
        orchestrator.setup(graph, ExperimentConfig(max_rounds=self.max_rounds))
        matching = orchestrator.run_until_convergence(
            max_rounds=self.max_rounds,
            vector=vector,
        )
        final_weight = graph.calculate_matching_weight(matching)

        return EvaluationResult(
            score=float(final_weight),
            matching=matching,
            report={
                "final_weight": final_weight,
                "iterations": self.max_rounds,
                "central_algorithmic_decisions": "baseline_only",
            },
            mode=self.mode,
        )
