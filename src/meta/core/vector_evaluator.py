"""Explicit offline evaluators for candidate parameter vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from src.graph.graph_manager import GraphManager

if TYPE_CHECKING:
    from src.meta.core.canonical_vector import CanonicalVector


@dataclass(frozen=True)
class EvaluationResult:
    """Observer-only result of one offline vector evaluation."""

    score: float
    matching: Mapping[int, int]
    report: Mapping[str, Any]
    mode: str


class VectorEvaluator(Protocol):
    """Scores a vector without exposing evaluator internals to the GA.

    The explicit centralized reference evaluator lives in ``src.offline``;
    it is intentionally not part of the distributed runtime namespace.
    """

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> EvaluationResult:
        """Run one complete evaluation."""


class DistributedRuntimeEvaluator:
    """Evaluate a vector through the production-faithful distributed runtime."""

    mode = "distributed_runtime"

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def evaluate(self, graph: GraphManager, vector: CanonicalVector) -> EvaluationResult:
        from src.meta.distributed.orchestrator import DistributedOrchestrator

        matching, report = DistributedOrchestrator(max_workers=self.max_workers).execute(graph, vector)
        return EvaluationResult(
            score=float(report["final_weight"]),
            matching=matching,
            report=report,
            mode=self.mode,
        )
