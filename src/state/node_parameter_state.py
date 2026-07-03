"""Per-node parameter evolution state for distributed GA."""

from typing import List, Optional
from src.meta.core.canonical_vector import CanonicalVector


class NodeParameterState:
    """Tracks GA population and elite vectors for a single node."""

    def __init__(self, node_id: int):
        """Initialize parameter state for a node."""
        self.node_id = node_id
        self.population: List[CanonicalVector] = []
        self.elite_pool: List[CanonicalVector] = []
        self.fitness_values: List[float] = []
        self.best_fitness = 0.0
        self.best_vector: Optional[CanonicalVector] = None

    def initialize(self, population_size: int) -> None:
        """Initialize population with random vectors."""
        self.population = [CanonicalVector() for _ in range(population_size)]
        self.fitness_values = [0.0] * population_size

    def set_fitness(self, index: int, fitness: float) -> None:
        """Set fitness value for a vector."""
        if index < len(self.fitness_values):
            self.fitness_values[index] = fitness
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_vector = self.population[index]

    def get_best_vector(self) -> Optional[CanonicalVector]:
        """Get best vector found so far."""
        return self.best_vector

    def add_elite(self, vector: CanonicalVector, fitness: float) -> None:
        """Add vector to elite pool."""
        self.elite_pool.append(vector)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_vector = vector
