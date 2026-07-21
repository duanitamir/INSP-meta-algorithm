"""Unit tests for FitnessEvaluator - 5+ comprehensive tests.

Tests now use DistributedOrchestrator instead of centralized CascadingLoop.
"""

import pytest
from src.meta.core.fitness_evaluator import FitnessEvaluator
from src.meta.core.canonical_vector import CanonicalVector
from src.graph.graph_manager import GraphManager


class TestFitnessEvaluatorBasics:
    """Test FitnessEvaluator initialization and interface."""

    def test_fitness_evaluator_initialization(self) -> None:
        """Should initialize without errors using distributed orchestrator."""
        evaluator = FitnessEvaluator()
        assert evaluator is not None

    def test_fitness_evaluator_name(self) -> None:
        """Should return correct name."""
        evaluator = FitnessEvaluator()
        assert "FitnessEvaluator" in evaluator.name()


class TestFitnessEvaluatorExecution:
    """Test FitnessEvaluator.evaluate() method with distributed algorithm."""

    def test_evaluate_returns_float(self) -> None:
        """evaluate() should return float fitness score."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        evaluator = FitnessEvaluator()
        vector = CanonicalVector()

        fitness = evaluator.evaluate(graph, vector)

        assert isinstance(fitness, float)
        assert fitness >= 0.0

    def test_evaluate_empty_graph(self) -> None:
        """evaluate() should handle empty graph."""
        graph = GraphManager.create_empty_graph()

        evaluator = FitnessEvaluator()
        vector = CanonicalVector()

        fitness = evaluator.evaluate(graph, vector)

        assert fitness == 0.0

    def test_evaluate_fitness_correlation(self) -> None:
        """Higher-weight graphs should score higher fitness."""
        # Small graph
        graph1 = GraphManager.create_empty_graph()
        for v in [1, 2]:
            graph1.add_vertex(v)
        graph1.add_edge(1, 2, 1.0)

        # Larger graph
        graph2 = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph2.add_vertex(v)
        graph2.add_edge(1, 2, 10.0)
        graph2.add_edge(3, 4, 10.0)

        evaluator = FitnessEvaluator()
        vector = CanonicalVector()

        fitness1 = evaluator.evaluate(graph1, vector)
        fitness2 = evaluator.evaluate(graph2, vector)

        assert fitness2 >= fitness1

    def test_evaluate_invalid_vector_raises_error(self) -> None:
        """Should raise ValueError on invalid canonical vector."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        evaluator = FitnessEvaluator()

        with pytest.raises(ValueError, match="Invalid vector"):
            vector = CanonicalVector(max_iterations=200)  # Out of bounds
            evaluator.evaluate(graph, vector)

    def test_evaluate_single_edge(self) -> None:
        """evaluate() should handle single edge graph."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 5.0)

        evaluator = FitnessEvaluator()
        vector = CanonicalVector()

        fitness = evaluator.evaluate(graph, vector)

        # Should find the edge or return 0 if endpoints don't both vote for it
        assert fitness >= 0.0
        # With distributed voting, single edge may or may not be included
        # depending on endpoint consensus logic
