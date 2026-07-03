"""Canonical parameter vector for genetic algorithm optimization.

This module defines the 10-parameter vector that represents algorithm
specialization configuration. The GA evolves these parameters to maximize
matching quality.

Parameters:
  [0]     Luby-Specific Base Activation
    - luby_base_probability: Base activation probability [0, 1]

  [1-6]   Luby Adaptive Activation Coefficients
    - luby_coeff_degree: Degree adjustment [-1, 1]
    - luby_coeff_neighbors_unmatched: Unmatched neighbors weight [-1, 1]
    - luby_coeff_clustering: Clustering coefficient weight [-1, 1]
    - luby_coeff_matched: Matched neighbors weight [-1, 1]
    - luby_coeff_round: Round number weight [-1, 1]
    - luby_coeff_weight: Edge weight adjustment [-1, 1]

  [7]     Itai-Israeli-Specific
    - itai_timeout_rounds: Timeout before reset [1, 20]

  [8-9]   Cascading Loop Control
    - max_iterations: Max cascading loop iterations [5, 100]
    - convergence_threshold: Min improvement to continue [0, 0.1]

Total: 10 parameters (node selection parameters removed)
"""

import random
from typing import Tuple, List, Union


# Parameter registry: name -> (min, max, random_gen_fn)
PARAMETER_REGISTRY = {
    "luby_base_probability": (0.0, 1.0, lambda: random.uniform(0.0, 1.0)),
    "luby_coeff_degree": (-1.0, 1.0, lambda: random.uniform(-1.0, 1.0)),
    "luby_coeff_neighbors_unmatched": (-1.0, 1.0, lambda: random.uniform(-1.0, 1.0)),
    "luby_coeff_clustering": (-1.0, 1.0, lambda: random.uniform(-1.0, 1.0)),
    "luby_coeff_matched": (-1.0, 1.0, lambda: random.uniform(-1.0, 1.0)),
    "luby_coeff_round": (-1.0, 1.0, lambda: random.uniform(-1.0, 1.0)),
    "luby_coeff_weight": (-1.0, 1.0, lambda: random.uniform(-1.0, 1.0)),
    "itai_timeout_rounds": (1, 20, lambda: random.randint(1, 20)),
    "max_iterations": (5, 100, lambda: random.randint(5, 100)),
    "convergence_threshold": (0.0, 0.1, lambda: random.uniform(0.0, 0.1)),
}

def _validate_parameter(value: Union[float, int], min_val: Union[float, int], max_val: Union[float, int], param_name: str) -> Tuple[bool, str | None]:
    """Validate a single parameter is within bounds."""
    if not (min_val <= value <= max_val):
        return False, f"{param_name} must be in [{min_val}, {max_val}], got {value}"
    return True, None


class CanonicalVector:
    """Container for 10 GA parameters governing algorithm specialization.

    This class manages the canonical parameter vector that controls:
    1. Algorithm-specific tuning (Luby activation, Itai timeout)
    2. Meta-algorithm control (iteration depth, convergence)

    All parameters are validated to ensure they meet constraints.
    Supports serialization to/from lists for GA operations.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize canonical parameter vector.

        Accepts any subset of the 10 parameters. Missing parameters are set to random values.
        """
        for param_name, (min_val, max_val, random_gen) in PARAMETER_REGISTRY.items():
            value = kwargs.get(param_name)
            setattr(self, param_name, value if value is not None else random_gen())

    def validate(self) -> Tuple[bool, str | None]:
        """Validate all parameter constraints."""
        for param_name, (min_val, max_val, _) in PARAMETER_REGISTRY.items():
            value = getattr(self, param_name)
            is_valid, error = _validate_parameter(value, min_val, max_val, param_name)
            if not is_valid:
                return False, error
        return True, None

    def to_list(self) -> List[float | int]:
        """Convert parameters to list for GA operations.

        Returns:
            List of 10 parameters in canonical order.
        """
        return [
            # Luby (8)
            self.luby_base_probability,
            self.luby_coeff_degree,
            self.luby_coeff_neighbors_unmatched,
            self.luby_coeff_clustering,
            self.luby_coeff_matched,
            self.luby_coeff_round,
            self.luby_coeff_weight,
            # Itai (1)
            self.itai_timeout_rounds,
            # Meta (2)
            self.max_iterations,
            self.convergence_threshold,
        ]

    @classmethod
    def from_list(cls, params: List[float | int]) -> "CanonicalVector":
        """Create vector from parameter list.

        Args:
            params: List of 10 parameters in canonical order.

        Returns:
            New CanonicalVector instance.

        Raises:
            ValueError: If parameter list has wrong length.
        """
        if len(params) != 10:
            raise ValueError(f"Expected 10 parameters, got {len(params)}")

        return cls(
            luby_base_probability=float(params[0]),
            luby_coeff_degree=float(params[1]),
            luby_coeff_neighbors_unmatched=float(params[2]),
            luby_coeff_clustering=float(params[3]),
            luby_coeff_matched=float(params[4]),
            luby_coeff_round=float(params[5]),
            luby_coeff_weight=float(params[6]),
            itai_timeout_rounds=int(params[7]),
            max_iterations=int(params[8]),
            convergence_threshold=float(params[9]),
        )

    @classmethod
    def random(cls) -> "CanonicalVector":
        """Generate random valid vector.

        Returns:
            New CanonicalVector with random parameters in valid ranges.
        """
        return cls(
            luby_base_probability=random.random(),
            luby_coeff_degree=random.uniform(-1.0, 1.0),
            luby_coeff_neighbors_unmatched=random.uniform(-1.0, 1.0),
            luby_coeff_clustering=random.uniform(-1.0, 1.0),
            luby_coeff_matched=random.uniform(-1.0, 1.0),
            luby_coeff_round=random.uniform(-1.0, 1.0),
            luby_coeff_weight=random.uniform(-1.0, 1.0),
            itai_timeout_rounds=random.randint(1, 20),
            max_iterations=random.randint(5, 20),  # Reduced from (5, 100)
            convergence_threshold=random.uniform(0.0, 0.1),
        )

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"CanonicalVector(\n"
            f"  luby_base={self.luby_base_probability:.2f},\n"
            f"  luby_coeff_degree={self.luby_coeff_degree:.2f},\n"
            f"  luby_coeff_unmatched={self.luby_coeff_neighbors_unmatched:.2f},\n"
            f"  luby_coeff_clustering={self.luby_coeff_clustering:.2f},\n"
            f"  luby_coeff_matched={self.luby_coeff_matched:.2f},\n"
            f"  luby_coeff_round={self.luby_coeff_round:.2f},\n"
            f"  luby_coeff_weight={self.luby_coeff_weight:.2f},\n"
            f"  itai_timeout={self.itai_timeout_rounds},\n"
            f"  max_iter={self.max_iterations},\n"
            f"  convergence_thresh={self.convergence_threshold:.3f}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Representation for debugging."""
        return self.__str__()
