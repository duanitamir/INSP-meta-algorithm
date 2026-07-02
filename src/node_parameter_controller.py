"""Node-local parameter controller with distributed GA and gossip learning.

Each node maintains its own population of CanonicalVectors and evolves them
based on local matching results. Nodes gossip elite vectors to learn from neighbors.
No central trainer required - parameters emerge through distributed evolution.
"""

from typing import List, Dict, Tuple
from src.meta.core.canonical_vector import CanonicalVector


class NodeParameterController:
    """Manages parameter evolution at a single node.

    Each node maintains:
    - Local population of CanonicalVectors
    - Local GA for evolution
    - Elite vectors received from neighbors
    - History of fitness values
    """

    def __init__(
        self,
        node_id: int,
        population_size: int = 10,
        generations_per_round: int = 2,
        mutation_rate: float = 0.1,
        elite_fraction: float = 0.5,
    ):
        """Initialize node parameter controller.

        Args:
            node_id: This node's ID
            population_size: Number of vectors in local population
            generations_per_round: GA generations per algorithm round
            mutation_rate: GA mutation rate
            elite_fraction: Fraction of population to keep as elite
        """
        self.node_id = node_id
        self.population_size = population_size
        self.generations_per_round = generations_per_round
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction

        # Initialize random population
        self.current_population: List[CanonicalVector] = [
            CanonicalVector() for _ in range(population_size)
        ]

        # Track elite vectors
        self.elite_vectors: List[CanonicalVector] = []
        self.best_vector = self.current_population[0]
        self.best_fitness = 0.0

        # Fitness history for convergence detection
        self.fitness_history: List[float] = []
        self.generation_counter = 0

    def receive_elite_vectors(self, vectors: List[CanonicalVector]) -> None:
        """Receive elite vectors from neighboring nodes via gossip.

        Integrates received vectors into local population to learn from
        neighbors' good solutions.

        Args:
            vectors: Elite vectors received from neighbors
        """
        if not vectors:
            return

        # Add received vectors to population (replace worst performers)
        num_to_add = min(len(vectors), self.population_size // 4)
        if num_to_add > 0:
            # Sort current population by fitness (worst first)
            sorted_pop = sorted(
                enumerate(self.current_population),
                key=lambda x: getattr(x[1], "_fitness", 0.0),
            )

            # Replace worst with received elite vectors
            for i in range(num_to_add):
                idx, _ = sorted_pop[i]
                self.current_population[idx] = vectors[i]

    def get_elite_vectors_to_share(self, count: int = 3) -> List[CanonicalVector]:
        """Get elite vectors to share with neighbors.

        Returns the best performing vectors from local population
        for other nodes to learn from.

        Args:
            count: Number of elite vectors to return

        Returns:
            List of elite CanonicalVectors
        """
        if not self.elite_vectors:
            # Return best vectors from current population
            sorted_pop = sorted(
                self.current_population,
                key=lambda x: getattr(x, "_fitness", 0.0),
                reverse=True,
            )
            return sorted_pop[:count]

        return self.elite_vectors[:count]

    def update_fitness(
        self, vector_idx: int, fitness: float
    ) -> None:
        """Update fitness score for a vector in population.

        Args:
            vector_idx: Index of vector in population
            fitness: New fitness score (matching weight)
        """
        if 0 <= vector_idx < len(self.current_population):
            self.current_population[vector_idx]._fitness = fitness

            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_vector = self.current_population[vector_idx]

            self.fitness_history.append(fitness)

    def evolve_one_generation(self) -> None:
        """Run one generation of local GA evolution.

        Performs selection, crossover, and mutation on local population.
        """
        # Sort population by fitness
        sorted_pop = sorted(
            self.current_population,
            key=lambda x: getattr(x, "_fitness", 0.0),
            reverse=True,
        )

        # Elite selection
        elite_size = max(1, int(self.population_size * 0.5))
        elite = sorted_pop[:elite_size]

        # Fill rest via crossover and mutation
        new_population = elite.copy()

        while len(new_population) < self.population_size:
            # Random parents from elite
            parent1 = elite[len(elite) % len(elite)]
            parent2 = elite[(len(elite) + 1) % len(elite)]

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            child = self._mutate(child)

            new_population.append(child)

        self.current_population = new_population[:self.population_size]
        self.generation_counter += 1

        # Update elite vectors
        self.elite_vectors = elite

    def _crossover(
        self, parent1: CanonicalVector, parent2: CanonicalVector
    ) -> CanonicalVector:
        """Create child via crossover of two parents.

        Args:
            parent1: First parent vector
            parent2: Second parent vector

        Returns:
            Child CanonicalVector
        """
        child = CanonicalVector()

        # Blend parameters
        child.luby_base_probability = (
            parent1.luby_base_probability + parent2.luby_base_probability
        ) / 2
        child.luby_coeff_degree = (
            parent1.luby_coeff_degree + parent2.luby_coeff_degree
        ) / 2
        child.luby_coeff_neighbors_unmatched = (
            parent1.luby_coeff_neighbors_unmatched
            + parent2.luby_coeff_neighbors_unmatched
        ) / 2
        child.luby_coeff_clustering = (
            parent1.luby_coeff_clustering + parent2.luby_coeff_clustering
        ) / 2
        child.luby_coeff_matched = (
            parent1.luby_coeff_matched + parent2.luby_coeff_matched
        ) / 2
        child.luby_coeff_round = (
            parent1.luby_coeff_round + parent2.luby_coeff_round
        ) / 2
        child.luby_coeff_weight = (
            parent1.luby_coeff_weight + parent2.luby_coeff_weight
        ) / 2
        child.itai_timeout_rounds = int(
            (parent1.itai_timeout_rounds + parent2.itai_timeout_rounds) / 2
        )
        child.max_iterations = int(
            (parent1.max_iterations + parent2.max_iterations) / 2
        )
        child.convergence_threshold = (
            parent1.convergence_threshold + parent2.convergence_threshold
        ) / 2

        return child

    def _mutate(self, vector: CanonicalVector) -> CanonicalVector:
        """Apply mutation to a vector.

        Args:
            vector: Vector to mutate

        Returns:
            Mutated CanonicalVector
        """
        import random

        mutated = CanonicalVector()

        # Copy parameters
        mutated.luby_base_probability = vector.luby_base_probability
        mutated.luby_coeff_degree = vector.luby_coeff_degree
        mutated.luby_coeff_neighbors_unmatched = (
            vector.luby_coeff_neighbors_unmatched
        )
        mutated.luby_coeff_clustering = vector.luby_coeff_clustering
        mutated.luby_coeff_matched = vector.luby_coeff_matched
        mutated.luby_coeff_round = vector.luby_coeff_round
        mutated.luby_coeff_weight = vector.luby_coeff_weight
        mutated.itai_timeout_rounds = vector.itai_timeout_rounds
        mutated.max_iterations = vector.max_iterations
        mutated.convergence_threshold = vector.convergence_threshold

        # Randomly mutate 2-3 parameters
        mutation_count = random.randint(2, 3)
        for _ in range(mutation_count):
            param_idx = random.randint(0, 9)

            if param_idx == 0:
                mutated.luby_base_probability = max(
                    0.0, min(1.0, mutated.luby_base_probability + random.uniform(-0.1, 0.1))
                )
            elif param_idx == 1:
                mutated.luby_coeff_degree = max(
                    -1.0, min(1.0, mutated.luby_coeff_degree + random.uniform(-0.1, 0.1))
                )
            elif param_idx == 2:
                mutated.luby_coeff_neighbors_unmatched = max(
                    -1.0,
                    min(
                        1.0,
                        mutated.luby_coeff_neighbors_unmatched
                        + random.uniform(-0.1, 0.1),
                    ),
                )
            elif param_idx == 3:
                mutated.luby_coeff_clustering = max(
                    -1.0,
                    min(1.0, mutated.luby_coeff_clustering + random.uniform(-0.1, 0.1)),
                )
            elif param_idx == 4:
                mutated.luby_coeff_matched = max(
                    -1.0, min(1.0, mutated.luby_coeff_matched + random.uniform(-0.1, 0.1))
                )
            elif param_idx == 5:
                mutated.luby_coeff_round = max(
                    -1.0, min(1.0, mutated.luby_coeff_round + random.uniform(-0.1, 0.1))
                )
            elif param_idx == 6:
                mutated.luby_coeff_weight = max(
                    -1.0, min(1.0, mutated.luby_coeff_weight + random.uniform(-0.1, 0.1))
                )
            elif param_idx == 7:
                mutated.itai_timeout_rounds = max(
                    1, min(20, mutated.itai_timeout_rounds + random.randint(-2, 2))
                )
            elif param_idx == 8:
                mutated.max_iterations = max(
                    5, min(100, mutated.max_iterations + random.randint(-5, 5))
                )
            elif param_idx == 9:
                mutated.convergence_threshold = max(
                    0.0,
                    min(0.1, mutated.convergence_threshold + random.uniform(-0.01, 0.01)),
                )

        return mutated

    def get_best_vector(self) -> CanonicalVector:
        """Get the best vector found so far.

        Returns:
            Best CanonicalVector
        """
        return self.best_vector

    def has_converged(self, window_size: int = 5) -> bool:
        """Check if population has converged.

        Considers converged if fitness improvement is minimal over recent
        generations.

        Args:
            window_size: Number of generations to check for improvement

        Returns:
            True if converged, False otherwise
        """
        if len(self.fitness_history) < window_size:
            return False

        recent = self.fitness_history[-window_size:]
        best_recent = max(recent)
        worst_recent = min(recent)

        # Converged if improvement is < 1%
        if best_recent > 0:
            improvement = (best_recent - worst_recent) / best_recent
            return improvement < 0.01

        return False

    def summary(self) -> Dict[str, float]:
        """Get summary of controller state.

        Returns:
            Dict with best_fitness, generation_counter, population_size
        """
        return {
            "node_id": self.node_id,
            "best_fitness": self.best_fitness,
            "generation_counter": self.generation_counter,
            "population_size": self.population_size,
            "has_converged": self.has_converged(),
        }
