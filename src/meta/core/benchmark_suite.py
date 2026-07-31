"""Reproducible observer-only benchmarks for distributed matching runs."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Sequence

import networkx as nx

from src.graph.graph_manager import GraphManager
from src.meta.core.graph_profile import GraphProfile
from src.meta.core.vector_evaluator import VectorEvaluator

if TYPE_CHECKING:
    from src.meta.core.canonical_vector import CanonicalVector


GraphFactory = Callable[[int], GraphManager]


@dataclass(frozen=True)
class BenchmarkScenario:
    """One named graph family and seed that can be rebuilt for every run."""

    family: str
    seed: int
    graph_factory: GraphFactory

    def build_graph(self) -> GraphManager:
        """Build a fresh deterministic graph for this scenario."""
        return self.graph_factory(self.seed)


@dataclass(frozen=True)
class BenchmarkMeasurement:
    """Observed quality and runtime metrics for one benchmark scenario."""

    family: str
    seed: int
    graph_profile: GraphProfile
    final_weight: float
    matching_cardinality: int
    message_count: int
    scheduled_ticks: int
    terminal_node_count: int
    outcome: str
    elapsed_seconds: float
    mode: str


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete reproducible benchmark output for one vector."""

    vector_fingerprint: str
    measurements: tuple[BenchmarkMeasurement, ...]


class ReproducibleBenchmarkSuite:
    """Run a vector against fixed graph families without affecting evaluation."""

    def __init__(self, scenarios: Sequence[BenchmarkScenario]) -> None:
        if not scenarios:
            raise ValueError("ReproducibleBenchmarkSuite requires at least one scenario")
        self.scenarios = tuple(scenarios)

    def run(self, vector: CanonicalVector, evaluator: VectorEvaluator) -> BenchmarkResult:
        """Measure each scenario independently using a fresh graph instance."""
        measurements = []
        for scenario in self.scenarios:
            graph = scenario.build_graph()
            started_at = perf_counter()
            result = evaluator.evaluate(graph, vector)
            elapsed_seconds = perf_counter() - started_at
            report = result.report
            outcome = str(report["outcome"])
            if outcome != "quiescent":
                raise RuntimeError(
                    f"Benchmark scenario {scenario.family!r} ended with {outcome}"
                )
            measurements.append(
                BenchmarkMeasurement(
                    family=scenario.family,
                    seed=scenario.seed,
                    graph_profile=GraphProfile.from_graph(graph),
                    final_weight=float(report["final_weight"]),
                    matching_cardinality=_matching_cardinality(result.matching),
                    message_count=int(report["message_count"]),
                    scheduled_ticks=int(report["scheduled_ticks"]),
                    terminal_node_count=int(report["terminal_node_count"]),
                    outcome=outcome,
                    elapsed_seconds=elapsed_seconds,
                    mode=result.mode,
                )
            )
        return BenchmarkResult(vector_fingerprint=vector.fingerprint(), measurements=tuple(measurements))


def standard_benchmark_scenarios() -> tuple[BenchmarkScenario, ...]:
    """Return the fixed graph families and seeds used for comparable local runs."""
    return (
        BenchmarkScenario("path", 101, _path_graph),
        BenchmarkScenario("erdos_renyi", 202, _erdos_renyi_graph),
        BenchmarkScenario("barabasi_albert", 303, _barabasi_albert_graph),
    )


def _matching_cardinality(matching: object) -> int:
    pairs = {
        tuple(sorted((node_id, partner_id)))
        for node_id, partner_id in dict(matching).items()
    }
    return len(pairs)


def _path_graph(seed: int) -> GraphManager:
    vertices = list(range(24))
    weights = Random(seed)
    return GraphManager.create_from_edges(
        vertices,
        [(node_id, node_id + 1, weights.uniform(0.5, 2.0)) for node_id in range(23)],
    )


def _erdos_renyi_graph(seed: int) -> GraphManager:
    return _from_networkx(nx.erdos_renyi_graph(24, 0.2, seed=seed), seed)


def _barabasi_albert_graph(seed: int) -> GraphManager:
    return _from_networkx(nx.barabasi_albert_graph(24, 2, seed=seed), seed)


def _from_networkx(graph: nx.Graph, seed: int) -> GraphManager:
    weights = Random(seed)
    return GraphManager.create_from_edges(
        sorted(graph.nodes()),
        [(node_id, neighbor_id, weights.uniform(0.5, 2.0)) for node_id, neighbor_id in sorted(graph.edges())],
    )
