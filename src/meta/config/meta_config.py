"""Central configuration for GA runs.

Provides:
- Algorithms enum: Easy selection of available algorithms (imported from register_all)
- GAConfig: Dataclass for GA parameters with sensible defaults
- MetaConfig: Combined configuration for algorithm selection + GA parameters
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Import Algorithms enum from where algorithms are registered (single source of truth)
from src.algorithms.implementations.register_all import Algorithms


@dataclass
class GAConfig:
    """Configuration for GA parameters with sensible defaults.

    Attributes:
        population_size: Number of vectors in population (default: 20)
        generations: Number of generations to evolve (default: 10)
        mutation_rate: Base mutation probability per parameter (default: 0.1)
        elite_fraction: Fraction of population to keep as elite (default: 0.5)
        early_stop_generations: Stop if no improvement for N generations (default: 10)
        num_workers: Number of parallel workers for evaluation (default: 4)
        use_cascading: If True, use cascading evaluator; else use standard (default: True)
    """
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    elite_fraction: float = 0.5
    early_stop_generations: int = 10
    num_workers: int = 4
    use_cascading: bool = True

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate GA parameters are within acceptable ranges.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        if self.population_size < 2:
            return False, "population_size must be >= 2"

        if self.generations < 1:
            return False, "generations must be >= 1"

        if not (0.0 <= self.mutation_rate <= 1.0):
            return False, "mutation_rate must be in [0.0, 1.0]"

        if not (0.1 <= self.elite_fraction <= 0.9):
            return False, "elite_fraction must be in [0.1, 0.9]"

        if self.early_stop_generations < 1:
            return False, "early_stop_generations must be >= 1"

        if self.num_workers < 1:
            return False, "num_workers must be >= 1"

        return True, None

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"GAConfig(\n"
            f"  population_size={self.population_size},\n"
            f"  generations={self.generations},\n"
            f"  mutation_rate={self.mutation_rate},\n"
            f"  elite_fraction={self.elite_fraction},\n"
            f"  early_stop_generations={self.early_stop_generations},\n"
            f"  num_workers={self.num_workers},\n"
            f"  use_cascading={self.use_cascading}\n"
            f")"
        )


@dataclass
class MetaConfig:
    """Central configuration for a GA run.

    Combines algorithm selection with GA parameters into one configuration object.
    This is the primary interface for configuring GA runs.

    Attributes:
        algorithms: List of Algorithms to use (if None, uses all available)
        ga_config: GAConfig with GA parameters (if None, uses defaults)

    Example:
        # Run GA with selected algorithms and custom parameters
        config = MetaConfig(
            algorithms=[a for a in Algorithms],  # Use all available algorithms
            ga_config=GAConfig(population_size=50, generations=20)
        )
        ga = MetaAlgorithmGA(config=config)
        best_vector, history = ga.evolve(graph)
    """
    algorithms: Optional[List[Algorithms]] = None
    ga_config: Optional[GAConfig] = None

    def __post_init__(self):
        """Initialize defaults if not provided."""
        if self.algorithms is None:
            # Use all available algorithms by default
            self.algorithms = [a for a in Algorithms]

        if self.ga_config is None:
            # Use default GA parameters
            self.ga_config = GAConfig()

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate entire configuration.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        # Check algorithms are selected
        if not self.algorithms:
            return False, "At least one algorithm must be selected"

        # Validate all algorithms are valid enum values
        for algo in self.algorithms:
            if not isinstance(algo, Algorithms):
                return False, f"Invalid algorithm: {algo}. Must be Algorithms enum value."

        # Validate GA config
        is_valid, error = self.ga_config.validate()
        if not is_valid:
            return False, error

        return True, None

    def get_algorithm_names(self) -> List[str]:
        """Get list of algorithm names (string values).

        Returns:
            List of algorithm names (e.g., ["greedy", "luby"])
        """
        return [a.value for a in self.algorithms]

    def __str__(self) -> str:
        """String representation for debugging."""
        algo_names = ", ".join(a.name for a in self.algorithms)
        return (
            f"MetaConfig(\n"
            f"  algorithms=[{algo_names}],\n"
            f"  ga_config={self.ga_config}\n"
            f")"
        )
