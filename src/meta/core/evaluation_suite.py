"""Graph-family evaluation for offline vector selection."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Sequence

from src.graph.graph_manager import GraphManager
from src.meta.core.vector_evaluator import VectorEvaluator

if TYPE_CHECKING:
    from src.meta.core.canonical_vector import CanonicalVector


@dataclass(frozen=True)
class EvaluationScenario:
    """One independently evaluated graph scenario."""

    graph: GraphManager
    label: str
    seed: int


@dataclass(frozen=True)
class EvaluationSuiteResult:
    """Aggregate outcome for all scenarios in an evaluation suite."""

    score: float
    mean_score: float
    standard_deviation: float
    worst_score: float
    mean_final_weight: float
    mean_scheduled_ticks: float
    mean_message_count: float


def aggregate_scores(scores: Sequence[float], instability_penalty: float) -> float:
    """Reward high mean score while penalizing variability across scenarios."""
    if not scores:
        raise ValueError("At least one score is required for aggregation")
    return mean(scores) - instability_penalty * pstdev(scores)


class EvaluationSuite:
    """Select vectors on graph families, then validate on unseen scenarios; never guarantees one graph."""

    def __init__(
        self,
        scenarios: Sequence[EvaluationScenario],
        instability_penalty: float = 0.0,
    ) -> None:
        if not scenarios:
            raise ValueError("EvaluationSuite requires at least one scenario")
        self.scenarios = tuple(scenarios)
        self.instability_penalty = instability_penalty

    def evaluate(
        self,
        vector: CanonicalVector,
        evaluator: VectorEvaluator,
    ) -> EvaluationSuiteResult:
        """Evaluate one vector independently on every configured scenario."""
        results = [evaluator.evaluate(scenario.graph, vector) for scenario in self.scenarios]
        scores = [result.score for result in results]

        return EvaluationSuiteResult(
            score=aggregate_scores(scores, self.instability_penalty),
            mean_score=mean(scores),
            standard_deviation=pstdev(scores),
            worst_score=min(scores),
            mean_final_weight=mean(float(result.report.get("final_weight", result.score)) for result in results),
            mean_scheduled_ticks=mean(float(result.report.get("scheduled_ticks", 0)) for result in results),
            mean_message_count=mean(float(result.report.get("message_count", 0)) for result in results),
        )
