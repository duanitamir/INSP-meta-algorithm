"""Configuration object for GA system consolidating all tunable parameters."""

from dataclasses import dataclass


@dataclass
class GAConfig:
    """Central configuration for entire GA system.

    Consolidates parameters from:
    - MetaAlgorithmGA: population_size, generations, mutation_rate, elite_fraction, early_stop_generations, num_workers
    - DistributedOrchestrator: max_iterations, convergence_threshold
    - DistributedConvergenceDetector: convergence_threshold (shared), quorum_threshold, voting_frequency
    - DistributedParameterEvolver: gossip_frequency

    This enables:
    - Single source of truth for all GA configuration
    - Easy persistence (save/load to JSON)
    - Experiment management (try different configs)
    - Cleaner function signatures
    """

    # GA Algorithm Parameters
    population_size: int = 20
    generations: int = 30
    mutation_rate: float = 0.1
    elite_fraction: float = 0.5
    early_stop_generations: int = 5
    num_workers: int = 4

    # Orchestration Parameters
    max_iterations: int = 100
    convergence_threshold: float = 0.05

    # Convergence Detection Parameters
    quorum_threshold: float = 0.5
    voting_frequency: int = 5

    # Distributed Parameter Evolution Parameters
    gossip_frequency: int = 5

    def validate(self) -> tuple[bool, str]:
        """Validate configuration parameters are in valid ranges.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.population_size < 2:
            return False, "population_size must be >= 2"

        if self.generations < 1:
            return False, "generations must be >= 1"

        if not (0.0 <= self.mutation_rate <= 1.0):
            return False, "mutation_rate must be in [0, 1]"

        if not (0.1 <= self.elite_fraction <= 0.9):
            return False, "elite_fraction must be in [0.1, 0.9]"

        if self.early_stop_generations < 1:
            return False, "early_stop_generations must be >= 1"

        if self.num_workers < 1:
            return False, "num_workers must be >= 1"

        if not (1 <= self.max_iterations <= 1000):
            return False, "max_iterations must be in [1, 1000]"

        if not (0.0 <= self.convergence_threshold <= 0.5):
            return False, "convergence_threshold must be in [0, 0.5]"

        if not (0.1 <= self.quorum_threshold <= 1.0):
            return False, "quorum_threshold must be in [0.1, 1.0]"

        if self.voting_frequency < 1:
            return False, "voting_frequency must be >= 1"

        if self.gossip_frequency < 1:
            return False, "gossip_frequency must be >= 1"

        return True, ""

    @staticmethod
    def small_graph() -> "GAConfig":
        """Configuration optimized for small graphs (10-100 nodes)."""
        return GAConfig(
            population_size=10,
            generations=10,
            mutation_rate=0.15,
            elite_fraction=0.5,
            early_stop_generations=3,
            num_workers=2,
            max_iterations=20,
            convergence_threshold=0.05,
        )

    @staticmethod
    def medium_graph() -> "GAConfig":
        """Configuration optimized for medium graphs (100-500 nodes)."""
        return GAConfig(
            population_size=15,
            generations=15,
            mutation_rate=0.12,
            elite_fraction=0.5,
            early_stop_generations=4,
            num_workers=4,
            max_iterations=50,
            convergence_threshold=0.05,
        )

    @staticmethod
    def large_graph() -> "GAConfig":
        """Configuration optimized for large graphs (500+ nodes)."""
        return GAConfig(
            population_size=20,
            generations=30,
            mutation_rate=0.10,
            elite_fraction=0.5,
            early_stop_generations=5,
            num_workers=4,
            max_iterations=100,
            convergence_threshold=0.05,
        )

    @staticmethod
    def aggressive_exploration() -> "GAConfig":
        """Configuration for aggressive exploration (lower elitism, higher mutation)."""
        return GAConfig(
            population_size=20,
            generations=30,
            mutation_rate=0.20,
            elite_fraction=0.3,
            early_stop_generations=5,
            num_workers=4,
            max_iterations=100,
            convergence_threshold=0.05,
        )

    @staticmethod
    def conservative_exploitation() -> "GAConfig":
        """Configuration for conservative exploitation (higher elitism, lower mutation)."""
        return GAConfig(
            population_size=20,
            generations=30,
            mutation_rate=0.08,
            elite_fraction=0.7,
            early_stop_generations=3,
            num_workers=4,
            max_iterations=100,
            convergence_threshold=0.05,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization.

        Returns:
            Dict with all configuration parameters
        """
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "elite_fraction": self.elite_fraction,
            "early_stop_generations": self.early_stop_generations,
            "num_workers": self.num_workers,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "quorum_threshold": self.quorum_threshold,
            "voting_frequency": self.voting_frequency,
            "gossip_frequency": self.gossip_frequency,
        }

    @staticmethod
    def from_dict(data: dict) -> "GAConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with configuration parameters

        Returns:
            GAConfig instance
        """
        return GAConfig(**data)
