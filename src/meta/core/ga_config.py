"""Configuration object for GA system consolidating all tunable parameters."""

from dataclasses import dataclass
from typing import Union, Tuple


def _validate_parameter(
    value: Union[float, int],
    min_val: Union[float, int],
    max_val: Union[float, int],
    param_name: str,
) -> Tuple[bool, str | None]:
    """Validate a single parameter is within bounds.

    Args:
        value: The parameter value to check
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        param_name: Name of parameter (for error messages)

    Returns:
        (True, None) if valid, (False, error_msg) if invalid
    """
    if not (min_val <= value <= max_val):
        return False, f"{param_name} must be in [{min_val}, {max_val}], got {value}"
    return True, None


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
        validations = [
            (self.population_size, 2, float('inf'), "population_size"),
            (self.generations, 1, float('inf'), "generations"),
            (self.mutation_rate, 0.0, 1.0, "mutation_rate"),
            (self.elite_fraction, 0.1, 0.9, "elite_fraction"),
            (self.early_stop_generations, 1, float('inf'), "early_stop_generations"),
            (self.num_workers, 1, float('inf'), "num_workers"),
            (self.max_iterations, 1, 1000, "max_iterations"),
            (self.convergence_threshold, 0.0, 0.5, "convergence_threshold"),
            (self.quorum_threshold, 0.1, 1.0, "quorum_threshold"),
            (self.voting_frequency, 1, float('inf'), "voting_frequency"),
            (self.gossip_frequency, 1, float('inf'), "gossip_frequency"),
        ]

        for value, min_val, max_val, name in validations:
            is_valid, error = _validate_parameter(value, min_val, max_val, name)
            if not is_valid:
                return False, error or ""

        return True, ""

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
