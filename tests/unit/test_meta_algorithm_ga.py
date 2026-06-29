"""Unit tests for MetaAlgorithmGA - 8+ comprehensive tests."""

import pytest
from src.meta.core.meta_algorithm_ga import MetaAlgorithmGA
from src.meta.core.fitness_evaluator import FitnessEvaluator
from src.meta.core.canonical_vector import CanonicalVector
from src.graph.graph_manager import GraphManager


class TestMetaAlgorithmGABasics:
    """Test MetaAlgorithmGA initialization and interface."""

    def test_ga_initialization(self) -> None:
        """Should initialize without errors."""
        evaluator = FitnessEvaluator()
        ga = MetaAlgorithmGA(evaluator, population_size=10, generations=5)
        assert ga is not None

    def test_ga_name(self) -> None:
        """Should return correct name."""
        evaluator = FitnessEvaluator()
        ga = MetaAlgorithmGA(evaluator)
        assert ga.name() == "MetaAlgorithmGA"


class TestMetaAlgorithmGAEvolution:
    """Test MetaAlgorithmGA.evolve() method."""

    def test_evolve_returns_tuple(self) -> None:
        """evolve() should return (vector, fitness_history)."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)

        evaluator = FitnessEvaluator()
        ga = MetaAlgorithmGA(evaluator, population_size=5, generations=2)

        result = ga.evolve(graph)

        assert isinstance(result, tuple)
        assert len(result) == 2
        vector, history = result
        assert isinstance(vector, CanonicalVector)
        assert isinstance(history, list)

    def test_evolve_fitness_history_length(self) -> None:
        """Fitness history should have one entry per generation."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        evaluator = FitnessEvaluator()
        generations = 5
        ga = MetaAlgorithmGA(
            evaluator, population_size=5, generations=generations
        )

        _, history = ga.evolve(graph)

        assert len(history) == generations

    def test_evolve_fitness_improves(self) -> None:
        """Fitness should improve or stay same across generations."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4, 5, 6]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 5.0)
        graph.add_edge(3, 4, 4.0)
        graph.add_edge(5, 6, 3.0)

        evaluator = FitnessEvaluator()
        ga = MetaAlgorithmGA(
            evaluator, population_size=10, generations=5, mutation_rate=0.1
        )

        _, history = ga.evolve(graph)

        # Fitness should be non-decreasing
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1]

    def test_evolve_parameters_in_bounds(self) -> None:
        """Evolved vector should have parameters in valid bounds."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        evaluator = FitnessEvaluator()
        ga = MetaAlgorithmGA(
            evaluator, population_size=5, generations=3, mutation_rate=0.2
        )

        best_vector, _ = ga.evolve(graph)

        is_valid, error = best_vector.validate()
        assert is_valid, f"Vector out of bounds: {error}"

    def test_evolve_single_generation(self) -> None:
        """Should work with single generation."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)

        evaluator = FitnessEvaluator()
        ga = MetaAlgorithmGA(
            evaluator, population_size=3, generations=1
        )

        vector, history = ga.evolve(graph)

        assert isinstance(vector, CanonicalVector)
        assert len(history) == 1

    def test_evolve_large_population(self) -> None:
        """Should handle large population."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(3, 4, 1.0)

        evaluator = FitnessEvaluator()
        ga = MetaAlgorithmGA(
            evaluator, population_size=50, generations=2
        )

        vector, history = ga.evolve(graph)

        assert isinstance(vector, CanonicalVector)
        assert len(history) == 2

    def test_evolve_empty_graph(self) -> None:
        """Should handle empty graph."""
        graph = GraphManager.create_empty_graph()

        evaluator = FitnessEvaluator()
        ga = MetaAlgorithmGA(
            evaluator, population_size=5, generations=2
        )

        vector, history = ga.evolve(graph)

        assert isinstance(vector, CanonicalVector)
        assert all(f == 0.0 for f in history)

    def test_evolve_returns_best_vector(self) -> None:
        """Returned vector should have best fitness in history."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 10.0)
        graph.add_edge(3, 4, 10.0)

        evaluator = FitnessEvaluator()
        ga = MetaAlgorithmGA(
            evaluator, population_size=10, generations=3
        )

        best_vector, history = ga.evolve(graph)

        best_fitness = evaluator.evaluate(graph, best_vector)
        assert best_fitness >= max(history) - 0.01  # Allow floating point error
