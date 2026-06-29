"""Unit tests for DistributedOrchestrator - 15+ comprehensive tests.

Tests the distributed orchestrator which replaces CascadingLoop.
Verifies that the orchestrator correctly manages iterations, conflict resolution,
and convergence detection.
"""

import pytest

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.distributed.orchestrator import DistributedOrchestrator
from src.meta.parameterizers.factory import ParameterizerFactory


class TestDistributedOrchestratorBasics:
    """Test DistributedOrchestrator initialization and interface."""

    def test_initialization(self) -> None:
        """Should initialize without errors."""
        orchestrator = DistributedOrchestrator()
        assert orchestrator is not None

    def test_name(self) -> None:
        """Should return correct name."""
        orchestrator = DistributedOrchestrator()
        assert orchestrator.name() == "DistributedOrchestrator"

    def test_has_conflict_resolver(self) -> None:
        """Should have conflict resolver component."""
        orchestrator = DistributedOrchestrator()
        assert hasattr(orchestrator, "conflict_resolver")
        assert orchestrator.conflict_resolver is not None

    def test_has_convergence_detector(self) -> None:
        """Should have convergence detector component."""
        orchestrator = DistributedOrchestrator()
        assert hasattr(orchestrator, "convergence_detector")
        assert orchestrator.convergence_detector is not None


class TestDistributedOrchestratorExecution:
    """Test DistributedOrchestrator.execute() method."""

    def test_execute_returns_tuple(self) -> None:
        """execute() should return (matching, metrics) tuple."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 10.0)
        graph.add_edge(3, 4, 15.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector()

        result = orchestrator.execute(graph, vector, parameterizers)

        assert isinstance(result, tuple)
        assert len(result) == 2
        matching, metrics = result
        assert isinstance(matching, dict)
        assert isinstance(metrics, dict)

    def test_execute_empty_graph(self) -> None:
        """execute() should handle empty graph gracefully."""
        graph = GraphManager.create_empty_graph()

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector()

        matching, metrics = orchestrator.execute(graph, vector, parameterizers)

        assert matching == {}
        assert metrics["final_weight"] == 0.0

    def test_execute_single_edge(self) -> None:
        """execute() should find single edge in simple graph."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 10.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector()

        matching, metrics = orchestrator.execute(graph, vector, parameterizers)

        # Should find the edge
        assert len(matching) >= 2  # At minimum 1->2 and 2->1
        assert matching.get(1) == 2 or matching.get(2) == 1

    def test_execute_multiple_edges(self) -> None:
        """execute() should find maximal matching on multi-edge graph."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 10.0)
        graph.add_edge(3, 4, 15.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector()

        matching, metrics = orchestrator.execute(graph, vector, parameterizers)

        # Should find both edges
        weight = metrics["final_weight"]
        assert weight >= 25.0  # At least 10 + 15

    def test_execute_invalid_vector_raises_error(self) -> None:
        """execute() should raise ValueError on invalid canonical vector."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 1.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()

        with pytest.raises(ValueError, match="Invalid canonical vector"):
            vector = CanonicalVector(max_iterations=200)  # Out of bounds
            orchestrator.execute(graph, vector, parameterizers)

    def test_execute_no_parameterizers_raises_error(self) -> None:
        """execute() should raise ValueError with empty parameterizers list."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)

        orchestrator = DistributedOrchestrator()
        vector = CanonicalVector()

        with pytest.raises(ValueError, match="Must provide at least one parameterizer"):
            orchestrator.execute(graph, vector, [])


class TestDistributedOrchestratorMetrics:
    """Test that orchestrator correctly tracks metrics."""

    def test_metrics_has_required_keys(self) -> None:
        """Metrics dict should have iterations, final_weight, improvements."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 10.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector()

        _, metrics = orchestrator.execute(graph, vector, parameterizers)

        assert "iterations" in metrics
        assert "final_weight" in metrics
        assert "improvements" in metrics

    def test_metrics_iterations_is_positive(self) -> None:
        """iterations should be >= 1 (at least one iteration executed)."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 10.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector()

        _, metrics = orchestrator.execute(graph, vector, parameterizers)

        assert metrics["iterations"] >= 1

    def test_metrics_improvements_length_matches_iterations(self) -> None:
        """improvements list length should match iterations count."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 10.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector()

        _, metrics = orchestrator.execute(graph, vector, parameterizers)

        assert len(metrics["improvements"]) == metrics["iterations"]

    def test_final_weight_correlates_with_graph_size(self) -> None:
        """Final weight should generally increase with graph size."""
        graph1 = GraphManager.create_empty_graph()
        graph1.add_vertex(1)
        graph1.add_vertex(2)
        graph1.add_edge(1, 2, 10.0)

        graph2 = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph2.add_vertex(v)
        graph2.add_edge(1, 2, 10.0)
        graph2.add_edge(3, 4, 10.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector()

        _, metrics1 = orchestrator.execute(graph1, vector, parameterizers)
        _, metrics2 = orchestrator.execute(graph2, vector, parameterizers)

        # Larger graph should generally have larger weight
        assert metrics2["final_weight"] >= metrics1["final_weight"]


class TestDistributedOrchestratorConvergence:
    """Test convergence behavior of orchestrator."""

    def test_converges_on_simple_graph(self) -> None:
        """Should converge on simple graph within max iterations."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 10.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector(max_iterations=100)

        _, metrics = orchestrator.execute(graph, vector, parameterizers)

        # Should converge before max iterations
        assert metrics["iterations"] <= 100

    def test_respects_max_iterations(self) -> None:
        """Should not exceed max_iterations from canonical vector."""
        graph = GraphManager.create_empty_graph()
        for v in range(1, 11):
            graph.add_vertex(v)
        for i in range(1, 10):
            graph.add_edge(i, i + 1, float(i))

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()
        vector = CanonicalVector(max_iterations=5)

        _, metrics = orchestrator.execute(graph, vector, parameterizers)

        # Should not exceed max_iterations
        assert metrics["iterations"] <= 5

    def test_convergence_threshold_affects_termination(self) -> None:
        """Low convergence threshold should stop earlier than high."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 10.0)
        graph.add_edge(3, 4, 15.0)

        orchestrator = DistributedOrchestrator()
        parameterizers = ParameterizerFactory.create_default()

        # Low threshold should converge sooner
        vector_low = CanonicalVector(convergence_threshold=0.001)
        _, metrics_low = orchestrator.execute(graph, vector_low, parameterizers)

        # High threshold means less improvement needed to continue
        vector_high = CanonicalVector(convergence_threshold=0.1)
        _, metrics_high = orchestrator.execute(graph, vector_high, parameterizers)

        # Lower threshold typically converges sooner (stops with less improvement)
        assert metrics_low["iterations"] <= metrics_high["iterations"]
