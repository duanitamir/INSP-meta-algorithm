"""Message payload for parameter gossip in distributed GA."""

from dataclasses import dataclass
from typing import List
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.messages.base import DistributedMessage


@dataclass(frozen=True)
class ParameterGossipMessage(DistributedMessage):
    """Payload for gossip-based parameter sharing between nodes.

    Contains elite vectors from a node's local GA population.
    Validates common distributed message fields.

    Attributes:
        sender_node_id: Node that is gossiping
        elite_vectors: Top-K CanonicalVectors to share (typically 3)
        fitness_values: Corresponding fitness scores for each vector
        generation: Generation number when this gossip was sent
        round_num: Algorithm round number
        weight: Associated weight value
    """

    sender_node_id: int
    elite_vectors: List[CanonicalVector]
    fitness_values: List[float]
    generation: int
    round_num: int = 0
    weight: float = 10.0

    def __post_init__(self) -> None:
        """Validate gossip message."""
        self.validate_base_fields(self.round_num, self.weight)

        if not self.elite_vectors:
            raise ValueError("Must include at least one elite vector")
        if len(self.elite_vectors) != len(self.fitness_values):
            raise ValueError(
                f"Vector count ({len(self.elite_vectors)}) must match "
                f"fitness count ({len(self.fitness_values)})"
            )
        if self.generation < 0:
            raise ValueError(f"Generation must be non-negative, got {self.generation}")

    def best_vector(self) -> CanonicalVector:
        """Get the highest-fitness vector in this gossip."""
        best_idx = self.fitness_values.index(max(self.fitness_values))
        return self.elite_vectors[best_idx]

    def best_fitness(self) -> float:
        """Get the highest fitness value in this gossip."""
        return max(self.fitness_values)

    def __repr__(self) -> str:
        """Return string representation."""
        best_fit = self.best_fitness()
        return (
            f"ParameterGossipMessage(sender={self.sender_node_id}, "
            f"vectors={len(self.elite_vectors)}, "
            f"best_fitness={best_fit:.0f}, "
            f"generation={self.generation})"
        )
