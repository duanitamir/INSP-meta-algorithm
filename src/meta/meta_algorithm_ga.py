"""Genetic algorithm for optimizing CanonicalVector parameters."""

import random
from typing import List, NamedTuple, Tuple

from src.graph.graph_manager import GraphManager
from src.meta.canonical_vector import CanonicalVector
from src.meta.fitness_evaluator import FitnessEvaluator


class PopulationEvaluation(NamedTuple):
    """Pairs a CanonicalVector with its fitness score."""

    vector: CanonicalVector
    fitness: float


class MetaAlgorithmGA:
    """Genetic algorithm for optimizing CanonicalVector parameters.

    Evolves a population of vectors across generations to maximize
    matching weight on a given graph.
    """

    def __init__(
        self,
        fitness_evaluator: FitnessEvaluator,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
    ) -> None:
        """Initialize genetic algorithm.

        Args:
            fitness_evaluator: FitnessEvaluator instance
            population_size: Number of vectors in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation per parameter
        """
        self.fitness_evaluator = fitness_evaluator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def evolve(self, graph: GraphManager) -> Tuple[CanonicalVector, List[float]]:
        """Evolve population to maximize fitness on graph.

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

        for generation in range(self.generations):
            # Evaluate fitness of all vectors
            population_with_fitness = [
                PopulationEvaluation(
                    vector=vector, fitness=self.fitness_evaluator.evaluate(graph, vector)
                )
                for vector in population
            ]

            # Track best
            gen_best_fitness = max(pf.fitness for pf in population_with_fitness)
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_vector = next(
                    pf.vector for pf in population_with_fitness if pf.fitness == gen_best_fitness
                )
            fitness_history.append(best_fitness)

            # Select top 50% as parents
            ranked = sorted(population_with_fitness, key=lambda x: x.fitness, reverse=True)
            elite_size = max(1, self.population_size // 2)
            elite = [pf.vector for pf in ranked[:elite_size]]

            # Generate offspring via crossover and mutation
            offspring = []
            while len(offspring) < self.population_size - elite_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                offspring.append(child)

            # Replace population: keep elite + offspring
            population = elite + offspring[: self.population_size - elite_size]

        return best_vector, fitness_history

    def _crossover(self, parent1: CanonicalVector, parent2: CanonicalVector) -> CanonicalVector:
        """Create offspring by blending two parents.

        Args:
            parent1: First parent vector
            parent2: Second parent vector

        Returns:
            CanonicalVector: Child vector with blended parameters
        """
        p1_values = parent1.to_list()
        p2_values = parent2.to_list()

        child_values = []
        for i in range(len(p1_values)):
            if random.random() < 0.5:
                child_values.append(p1_values[i])
            else:
                child_values.append(p2_values[i])

        return self._from_list(child_values)

    def _mutate(self, vector: CanonicalVector) -> CanonicalVector:
        """Mutate vector by perturbing parameters.

        Args:
            vector: Vector to mutate

        Returns:
            CanonicalVector: Mutated vector with parameters in valid bounds
        """
        values = vector.to_list()
        mutated_values = values.copy()

        # Parameter bounds (from CanonicalVector)
        bounds = [
            (0.0, 1.0),  # [0] luby_base_probability
            (-1.0, 1.0),  # [1-6] luby coefficients
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (1.0, 20.0),  # [7] itai_timeout_rounds
            (5.0, 100.0),  # [8] max_iterations
            (0.0, 0.1),  # [9] convergence_threshold
        ]

        for i in range(len(mutated_values)):
            if random.random() < self.mutation_rate:
                min_val, max_val = bounds[i]
                perturbation = random.uniform(-0.2, 0.2) * (max_val - min_val)
                new_val = mutated_values[i] + perturbation
                mutated_values[i] = max(min_val, min(max_val, new_val))

        return self._from_list(mutated_values)

    def _from_list(self, values: List) -> CanonicalVector:
        """Create CanonicalVector from list of parameter values.

        Args:
            values: List of 10 parameter values in order

        Returns:
            CanonicalVector: Constructed from parameter list
        """
        return CanonicalVector(
            luby_base_probability=float(values[0]),
            luby_coeff_degree=float(values[1]),
            luby_coeff_neighbors_unmatched=float(values[2]),
            luby_coeff_clustering=float(values[3]),
            luby_coeff_matched=float(values[4]),
            luby_coeff_round=float(values[5]),
            luby_coeff_weight=float(values[6]),
            itai_timeout_rounds=int(values[7]),
            max_iterations=int(values[8]),
            convergence_threshold=float(values[9]),
        )

    def name(self) -> str:
        """Return GA name."""
        return "MetaAlgorithmGA"
