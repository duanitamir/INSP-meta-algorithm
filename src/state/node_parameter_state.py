"""Per-node state for distributed parameter evolution."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set
from src.meta.core.canonical_vector import CanonicalVector
from src.state.distributed_node_state import DistributedNodeState


@dataclass
class NodeParameterState(DistributedNodeState):
    """Tracks GA state for a single node in distributed parameter evolution.

    Each node maintains:
    - Local population of CanonicalVectors
    - Best vector found locally
    - Elite pool from gossip
    - Generation counter

    Attributes:
        node_id: Identifier of this node
        population: List of CanonicalVectors in local population
        fitness: Dict mapping vector ID to fitness score
        best_local: Best vector found by this node
        best_local_fitness: Fitness of best_local
        elite_pool: Set of elite vectors received via gossip
        generation: Current generation number
        gossip_round_count: Rounds since last gossip sent
    """

    node_id: int
    population: List[CanonicalVector] = field(default_factory=list)
    fitness: Dict[int, float] = field(default_factory=dict)
    best_local: CanonicalVector | None = None
    best_local_fitness: float = 0.0
    elite_pool: Set[CanonicalVector] = field(default_factory=set)
    generation: int = 0
    gossip_round_count: int = 0

    def initialize(self, initial_population_size: int = 10) -> None:
        """Initialize with random population.

        Args:
            initial_population_size: Number of random vectors to create
        """
        self.population = [CanonicalVector() for _ in range(initial_population_size)]
        self.fitness = {}
        self.best_local = None
        self.best_local_fitness = 0.0
        self.elite_pool = set()
        self.generation = 0
        self.gossip_round_count = 0

    def update_fitness(self, vector: CanonicalVector, fitness: float) -> None:
        """Record fitness for a vector.

        Args:
            vector: CanonicalVector to evaluate
            fitness: Fitness score (matching weight)
        """
        vector_id = id(vector)
        self.fitness[vector_id] = fitness

        # Update best local if this is better
        if fitness > self.best_local_fitness:
            self.best_local_fitness = fitness
            self.best_local = vector

    def get_elite_k(self, k: int = 3) -> List[CanonicalVector]:
        """Get top-K vectors from population by fitness.

        Args:
            k: Number of elite vectors to return

        Returns:
            List of top-K vectors (sorted by fitness descending)
        """
        # Score each vector
        scored = [
            (v, self.fitness.get(id(v), 0.0)) for v in self.population
        ]
        # Sort by fitness descending
        scored.sort(key=lambda x: x[1], reverse=True)
        # Return top-K vectors
        return [v for v, _ in scored[:k]]

    def add_elite_from_gossip(self, vectors: List[CanonicalVector]) -> None:
        """Integrate elite vectors from gossip into elite pool.

        Args:
            vectors: Elite vectors received via gossip
        """
        for v in vectors:
            self.elite_pool.add(v)

    def merge_elite_into_population(self, elite_pool_size: int = 3) -> None:
        """Merge elite pool into population, maintaining diversity.

        Args:
            elite_pool_size: How many elite vectors to keep from gossip
        """
        if not self.elite_pool:
            return

        # Score elite vectors (or use 0 if not yet evaluated)
        elite_scored = [
            (v, self.fitness.get(id(v), 0.0)) for v in self.elite_pool
        ]
        # Sort by fitness descending
        elite_scored.sort(key=lambda x: x[1], reverse=True)
        # Add top elite vectors to population
        for v, _ in elite_scored[:elite_pool_size]:
            if v not in self.population:
                self.population.append(v)

    def increment_generation(self) -> None:
        """Increment generation counter."""
        self.generation += 1

    def increment_gossip_round_count(self) -> None:
        """Increment rounds since last gossip."""
        self.gossip_round_count += 1

    def reset_gossip_round_count(self) -> None:
        """Reset gossip round counter after sending gossip."""
        self.gossip_round_count = 0

    def population_size(self) -> int:
        """Get current population size."""
        return len(self.population)

    def elite_pool_size(self) -> int:
        """Get current elite pool size."""
        return len(self.elite_pool)

    def reset(self) -> None:
        """Reset state for next round (clear population but keep elite pool)."""
        self.population.clear()
        self.fitness.clear()

    def get_state_dict(self) -> Dict[str, Any]:
        """Return state snapshot for serialization.

        Returns:
            Dict containing all parameter evolution state
        """
        return {
            "node_id": self.node_id,
            "population_size": self.population_size(),
            "best_local_fitness": self.best_local_fitness,
            "elite_pool_size": self.elite_pool_size(),
            "generation": self.generation,
            "gossip_round_count": self.gossip_round_count,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"NodeParameterState(node={self.node_id}, "
            f"pop_size={self.population_size()}, "
            f"best_fitness={self.best_local_fitness:.0f}, "
            f"generation={self.generation}, "
            f"elite_pool_size={self.elite_pool_size()})"
        )
