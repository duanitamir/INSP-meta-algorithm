"""Algorithm configuration that nodes carry and gossip to neighbors.

These parameters define how algorithms behave and are spread via gossip protocol.
Each node carries a local copy and learns updated configs from neighbors.

Parameters come from CanonicalVector during GA optimization.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class DistributedAlgorithmConfig:
    """Algorithm parameters distributed across nodes via gossip.

    Includes:
    - Convergence detection thresholds (when to stop)
    - Algorithm-specific parameters (tuned via GA)
    - Version for ordering gossip updates
    """

    # Convergence detection (Phase 3.3 uses these)
    convergence_threshold: float = 0.05
    quorum_threshold: float = 0.5
    max_iterations: int = 100

    # Greedy algorithm
    greedy_max_rounds: int = 100

    # Itai-Israeli algorithm
    itai_timeout_rounds: int = 5
    itai_max_rounds: int = 100

    # Luby randomized algorithm (all 7 Luby params + max_rounds)
    luby_base_probability: float = 0.5
    luby_coeff_degree: float = 0.1
    luby_coeff_neighbors_unmatched: float = 0.1
    luby_coeff_clustering: float = 0.1
    luby_coeff_matched: float = 0.1
    luby_coeff_round: float = 0.1
    luby_coeff_weight: float = 0.1
    luby_max_rounds: int = 100

    # Versioning for gossip protocol
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedAlgorithmConfig":
        """Create from dictionary (after deserialization)."""
        return cls(**data)

    @classmethod
    def from_canonical_vector(cls, vector) -> "DistributedAlgorithmConfig":
        """Create from CanonicalVector for GA optimization.

        Args:
            vector: CanonicalVector with all GA parameters

        Returns:
            DistributedAlgorithmConfig with parameters from vector
        """
        return cls(
            convergence_threshold=vector.convergence_threshold,
            quorum_threshold=vector.quorum_threshold,
            max_iterations=int(vector.max_iterations),
            greedy_max_rounds=100,
            itai_timeout_rounds=int(vector.itai_timeout_rounds),
            itai_max_rounds=100,
            luby_base_probability=vector.luby_base_probability,
            luby_coeff_degree=vector.luby_coeff_degree,
            luby_coeff_neighbors_unmatched=vector.luby_coeff_neighbors_unmatched,
            luby_coeff_clustering=vector.luby_coeff_clustering,
            luby_coeff_matched=vector.luby_coeff_matched,
            luby_coeff_round=vector.luby_coeff_round,
            luby_coeff_weight=vector.luby_coeff_weight,
            luby_max_rounds=100,
            version=1
        )
