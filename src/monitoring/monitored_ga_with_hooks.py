"""GA with real-time monitoring hooks integrated into evolution loop.

Patches MetaAlgorithmGA.evolve() to call monitoring callbacks at key points,
enabling real-time progress tracking with live dashboard updates.
Minimizes overhead by only tracking at generation level, not per-individual.
"""

from typing import Optional, List
import time
from src.meta.core.meta_algorithm_ga import MetaAlgorithmGA
from src.meta.core.canonical_vector import CanonicalVector
from src.graph.graph_manager import GraphManager
from src.monitoring.ga_progress_monitor import ExperimentProgressTracker


class MonitoredMetaAlgorithmGA(MetaAlgorithmGA):
    """MetaAlgorithmGA with real-time monitoring hooks (minimal overhead).

    Extends the GA to call monitoring callbacks during evolution,
    enabling live dashboard updates with minimal performance impact.
    Only tracks at generation level, not per-individual.
    """

    def __init__(self, *args, tracker: Optional[ExperimentProgressTracker] = None, **kwargs):
        """Initialize with optional monitoring tracker.

        Args:
            tracker: ExperimentProgressTracker for live monitoring
            *args, **kwargs: Passed to MetaAlgorithmGA.__init__
        """
        super().__init__(*args, **kwargs)
        self.tracker = tracker

    def evolve(self, graph: GraphManager) -> tuple[CanonicalVector, List[float]]:
        """Evolve population with real-time monitoring hooks (minimal overhead).

        Args:
            graph: GraphManager instance to optimize for

        Returns:
            Tuple of:
            - CanonicalVector: Best vector found
            - List[float]: Best fitness per generation
        """
        # Initialize population with random valid vectors
        population = [CanonicalVector() for _ in range(self.population_size)]

        best_vector = population[0]
        best_fitness = 0.0
        fitness_history = []
        no_improve_count = 0

        for generation in range(self.generations):
            # Start generation tracking (minimal overhead)
            if self.tracker:
                self.tracker.start_generation(generation + 1)

            # Use parent class parallel evaluation (no per-individual tracking)
            population_with_fitness = self._evaluate_population_parallel(graph, population)

            # Track best
            gen_best_fitness = max(pf.fitness for pf in population_with_fitness)
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_vector = next(
                    pf.vector for pf in population_with_fitness
                    if pf.fitness == gen_best_fitness
                )
                no_improve_count = 0
            else:
                no_improve_count += 1

            fitness_history.append(best_fitness)

            # Report generation complete (track metrics)
            if self.tracker:
                self.tracker.generation_complete(generation + 1, best_fitness)

            # Early stopping: if no improvement for N generations, stop
            if no_improve_count >= self.early_stop_generations:
                break

            # Select elite based on configurable fraction
            ranked = sorted(population_with_fitness, key=lambda x: x.fitness, reverse=True)
            elite_size = max(1, int(self.population_size * self.elite_fraction))
            elite = [pf.vector for pf in ranked[:elite_size]]

            # Adaptive mutation rate: increase as population converges
            current_mutation_rate = self._get_adaptive_mutation_rate(
                no_improve_count, self.early_stop_generations
            )

            # Generate offspring via crossover and mutation
            import random
            offspring = []
            while len(offspring) < self.population_size - elite_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, current_mutation_rate)
                offspring.append(child)

            # Replace population: keep elite + offspring
            population = elite + offspring[: self.population_size - elite_size]

        if self.tracker:
            self.tracker.finalize()

        return best_vector, fitness_history
