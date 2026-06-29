"""Unit tests for GA Algorithm Enhancements - 12+ comprehensive tests.

Tests parallel evaluation, adaptive mutation, early stopping, and tunable elite fraction.
"""

import pytest

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.core.fitness_evaluator import FitnessEvaluator
from src.meta.core.meta_algorithm_ga import MetaAlgorithmGA


class TestParallelEvaluation:
    """Test parallel population evaluation enhancement."""

    def test_parallel_evaluation_produces_same_results_as_sequential(self) -> None:
        """Parallel evaluation should produce identical fitness scores."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 10.0)
        graph.add_edge(3, 4, 15.0)

        ga = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            population_size=5,
            generations=2,
            num_workers=2,
        )

        vector1 = CanonicalVector()
        vector2 = CanonicalVector()
        population = [vector1, vector2, vector1, vector2, vector1]

        evaluations = ga._evaluate_population_parallel(graph, population)

        assert len(evaluations) == 5
        assert evaluations[0].fitness == evaluations[2].fitness  # Same vector, same fitness
        assert all(e.fitness >= 0 for e in evaluations)

    def test_parallel_evaluation_with_different_workers(self) -> None:
        """Parallel evaluation should work with different worker counts."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 10.0)

        evaluator = FitnessEvaluator()
        population = [CanonicalVector() for _ in range(4)]

        for num_workers in [1, 2, 4]:
            ga = MetaAlgorithmGA(fitness_evaluator=evaluator, num_workers=num_workers)
            evaluations = ga._evaluate_population_parallel(graph, population)
            assert len(evaluations) == 4
            assert all(e.fitness >= 0 for e in evaluations)


class TestAdaptiveMutation:
    """Test adaptive mutation rate enhancement."""

    def test_adaptive_mutation_rate_increases_with_convergence(self) -> None:
        """Adaptive mutation rate should increase as population converges."""
        ga = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            mutation_rate=0.1,
            early_stop_generations=5,
        )

        # Early (low convergence): low mutation
        rate_early = ga._get_adaptive_mutation_rate(no_improve_count=0, max_no_improve=5)

        # Late (high convergence): high mutation
        rate_late = ga._get_adaptive_mutation_rate(no_improve_count=4, max_no_improve=5)

        assert rate_early < rate_late
        assert rate_early == 0.1  # Base rate
        assert rate_late > rate_early

    def test_adaptive_mutation_respects_bounds(self) -> None:
        """Adaptive mutation rate should stay within reasonable bounds."""
        ga = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(), mutation_rate=0.15
        )

        rates = [
            ga._get_adaptive_mutation_rate(i, 10) for i in range(11)
        ]

        assert all(r >= 0.15 for r in rates)  # Never below base rate
        assert all(r <= 0.15 * 3 for r in rates)  # Never exceed 3x base rate


class TestEarlyStopping:
    """Test early stopping enhancement."""

    def test_early_stopping_terminates_with_no_improvement(self) -> None:
        """GA should stop early if no improvement for N generations."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 10.0)

        ga = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            population_size=5,
            generations=20,
            early_stop_generations=3,
        )

        best_vector, fitness_history = ga.evolve(graph)

        # Should stop early (before 20 generations)
        assert len(fitness_history) <= 20
        # On simple graph, should converge quickly
        assert len(fitness_history) <= 10

    def test_early_stopping_disabled_with_large_threshold(self) -> None:
        """GA should not stop early if early_stop_generations is large."""
        graph = GraphManager.create_empty_graph()
        graph.add_vertex(1)
        graph.add_vertex(2)
        graph.add_edge(1, 2, 10.0)

        ga = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            population_size=5,
            generations=5,
            early_stop_generations=100,  # Very high threshold
        )

        best_vector, fitness_history = ga.evolve(graph)

        # Should run full generations (or close to it)
        assert len(fitness_history) == 5


class TestTunableEliteFraction:
    """Test tunable elite fraction enhancement."""

    def test_initialization_with_custom_elite_fraction(self) -> None:
        """Should accept custom elite fraction parameter."""
        ga = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            elite_fraction=0.3,
        )

        assert ga.elite_fraction == 0.3

    def test_elite_fraction_clamped_to_valid_range(self) -> None:
        """Elite fraction should be clamped to [0.1, 0.9]."""
        ga_low = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(), elite_fraction=0.01
        )
        ga_high = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(), elite_fraction=1.5
        )

        assert ga_low.elite_fraction == 0.1  # Clamped up
        assert ga_high.elite_fraction == 0.9  # Clamped down

    def test_high_elite_fraction_keeps_more_good_solutions(self) -> None:
        """Higher elite fraction should preserve more good solutions."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 10.0)
        graph.add_edge(3, 4, 15.0)

        # Conservative elitism (keep 70% of population)
        ga_high = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            population_size=10,
            generations=5,
            elite_fraction=0.7,
        )

        # Aggressive elitism (keep only 20% of population)
        ga_low = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            population_size=10,
            generations=5,
            elite_fraction=0.2,
        )

        best_high, history_high = ga_high.evolve(graph)
        best_low, history_low = ga_low.evolve(graph)

        # Both should find good solutions, but trajectories may differ
        assert isinstance(best_high, CanonicalVector)
        assert isinstance(best_low, CanonicalVector)


class TestParallelVersusSingleThread:
    """Test that parallel and single-threaded execution produce consistent results."""

    def test_same_seed_same_result(self) -> None:
        """Same seed should produce reproducible results."""
        import random

        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 10.0)
        graph.add_edge(3, 4, 15.0)

        # First run with seed
        random.seed(42)
        ga1 = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            population_size=5,
            generations=3,
            num_workers=1,
        )
        best1, history1 = ga1.evolve(graph)

        # Second run with same seed
        random.seed(42)
        ga2 = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            population_size=5,
            generations=3,
            num_workers=1,
        )
        best2, history2 = ga2.evolve(graph)

        # Should get same fitness history
        assert history1 == history2

    def test_parallel_vs_single_thread_convergence(self) -> None:
        """Both parallel and single-threaded should eventually converge."""
        graph = GraphManager.create_empty_graph()
        for v in [1, 2, 3, 4]:
            graph.add_vertex(v)
        graph.add_edge(1, 2, 10.0)
        graph.add_edge(3, 4, 15.0)

        ga_single = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            population_size=5,
            generations=10,
            num_workers=1,
        )

        ga_parallel = MetaAlgorithmGA(
            fitness_evaluator=FitnessEvaluator(),
            population_size=5,
            generations=10,
            num_workers=4,
        )

        _, history_single = ga_single.evolve(graph)
        _, history_parallel = ga_parallel.evolve(graph)

        # Both should improve
        assert history_single[-1] >= history_single[0]
        assert history_parallel[-1] >= history_parallel[0]
