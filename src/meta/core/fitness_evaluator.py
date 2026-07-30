"""Compatibility facade for offline vector evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.graph.graph_manager import GraphManager
from src.meta.core.vector_evaluator import (
    DistributedRuntimeEvaluator,
    EvaluationResult,
    VectorEvaluator,
)

if TYPE_CHECKING:
    from src.meta.core.canonical_vector import CanonicalVector


class FitnessEvaluator:
    """Score vectors with an explicit evaluator mode.

    The default evaluates the real distributed runtime.
    """

    def __init__(
        self,
        max_workers: int = 4,
        evaluator: VectorEvaluator | None = None,
    ) -> None:
        self.max_workers = max_workers
        self.evaluator = evaluator or DistributedRuntimeEvaluator(max_workers=max_workers)

    def evaluate_result(self, graph: GraphManager, vector: CanonicalVector) -> EvaluationResult:
        """Return the complete observer-only evaluation result."""
        is_valid, error = vector.validate()
        if not is_valid:
            raise ValueError(f"Invalid vector: {error}")
        return self.evaluator.evaluate(graph, vector)

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> float:
        """Return only the score for existing GA callers."""
        return self.evaluate_result(graph, vector).score

    def name(self) -> str:
        return "FitnessEvaluator"
