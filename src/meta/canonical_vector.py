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


def _random_min_max() -> Tuple[float, float]:
    """Generate random min/max pair in [0, 1] with min <= max.

    Returns:
        Tuple of (min_value, max_value) where min_value <= max_value.
    """
    val1 = random.random()
    val2 = random.random()
    return (min(val1, val2), max(val1, val2))


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


def _validate_range_pair(
    min_val: Union[float, int], max_val: Union[float, int], name: str
) -> Tuple[bool, str | None]:
    """Validate that min_val <= max_val for a range parameter.

    Args:
        min_val: Minimum value
        max_val: Maximum value
        name: Base name for error message (e.g., "degree")

    Returns:
        (True, None) if valid, (False, error_msg) if invalid
    """
    if min_val > max_val:
        return False, f"{name}_min must be <= {name}_max"
    return True, None


class CanonicalVector:
    """Container for 10 GA parameters governing algorithm specialization.

    This class manages the canonical parameter vector that controls:
    1. Algorithm-specific tuning (Luby activation, Itai timeout)
    2. Meta-algorithm control (iteration depth, convergence)

    All parameters are validated to ensure they meet constraints.
    Supports serialization to/from lists for GA operations.
    """

    def __init__(
        self,
        # Luby Adaptive (8)
        luby_base_probability: float | None = None,
        luby_coeff_degree: float | None = None,
        luby_coeff_neighbors_unmatched: float | None = None,
        luby_coeff_clustering: float | None = None,
        luby_coeff_matched: float | None = None,
        luby_coeff_round: float | None = None,
        luby_coeff_weight: float | None = None,
        # Itai-Israeli (1)
        itai_timeout_rounds: int | None = None,
        # Meta-algorithm (2)
        max_iterations: int | None = None,
        convergence_threshold: float | None = None,
    ) -> None:
        """Initialize canonical parameter vector.

        Args:
            luby_base_probability: Base activation probability ([0, 1]), random if None
            luby_coeff_degree: Degree coefficient ([-1, 1]), random if None
            luby_coeff_neighbors_unmatched: Unmatched neighbors coeff ([-1, 1]), random if None
            luby_coeff_clustering: Clustering coefficient ([-1, 1]), random if None
            luby_coeff_matched: Matched neighbors coefficient ([-1, 1]), random if None
            luby_coeff_round: Round number coefficient ([-1, 1]), random if None
            luby_coeff_weight: Weight coefficient ([-1, 1]), random if None
            itai_timeout_rounds: Timeout rounds ([1, 20]), random if None
            max_iterations: Max iterations ([5, 100]), random if None
            convergence_threshold: Min improvement threshold ([0, 0.1]), random if None
        """

        # Luby Adaptive - use random if None
        self.luby_base_probability = luby_base_probability if luby_base_probability is not None else random.uniform(0.0, 1.0)
        self.luby_coeff_degree = luby_coeff_degree if luby_coeff_degree is not None else random.uniform(-1.0, 1.0)
        self.luby_coeff_neighbors_unmatched = luby_coeff_neighbors_unmatched if luby_coeff_neighbors_unmatched is not None else random.uniform(-1.0, 1.0)
        self.luby_coeff_clustering = luby_coeff_clustering if luby_coeff_clustering is not None else random.uniform(-1.0, 1.0)
        self.luby_coeff_matched = luby_coeff_matched if luby_coeff_matched is not None else random.uniform(-1.0, 1.0)
        self.luby_coeff_round = luby_coeff_round if luby_coeff_round is not None else random.uniform(-1.0, 1.0)
        self.luby_coeff_weight = luby_coeff_weight if luby_coeff_weight is not None else random.uniform(-1.0, 1.0)

        # Itai-Israeli - use random if None
        self.itai_timeout_rounds = itai_timeout_rounds if itai_timeout_rounds is not None else random.randint(1, 20)

        # Meta-algorithm - use random if None
        self.max_iterations = max_iterations if max_iterations is not None else random.randint(5, 100)
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else random.uniform(0.0, 0.1)

    def validate(self) -> Tuple[bool, str | None]:
        """Validate all parameter constraints.

        Checks:
        - All coefficients in [-1, 1]
        - Base probability in [0, 1]
        - timeout_rounds in [1, 20]
        - max_iterations in [5, 100]
        - convergence_threshold in [0, 0.1]

        Returns:
            Tuple of (is_valid, error_message).
            Returns (True, None) if valid.
            Returns (False, error_string) if invalid.
        """
        # Validate all individual parameters
        validations = [
            # Luby base probability [0, 1]
            (self.luby_base_probability, 0.0, 1.0, "luby_base_probability"),
            # Luby coefficients [-1, 1]
            (self.luby_coeff_degree, -1.0, 1.0, "luby_coeff_degree"),
            (self.luby_coeff_neighbors_unmatched, -1.0, 1.0, "luby_coeff_neighbors_unmatched"),
            (self.luby_coeff_clustering, -1.0, 1.0, "luby_coeff_clustering"),
            (self.luby_coeff_matched, -1.0, 1.0, "luby_coeff_matched"),
            (self.luby_coeff_round, -1.0, 1.0, "luby_coeff_round"),
            (self.luby_coeff_weight, -1.0, 1.0, "luby_coeff_weight"),
            # Itai timeout [1, 20]
            (self.itai_timeout_rounds, 1, 20, "itai_timeout_rounds"),
            # Meta parameters
            (self.max_iterations, 5, 100, "max_iterations"),
            (self.convergence_threshold, 0.0, 0.1, "convergence_threshold"),
        ]

        # Check all bounds
        for value, min_val, max_val, name in validations:
            is_valid, error = _validate_parameter(value, min_val, max_val, name)
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
            max_iterations=random.randint(5, 100),
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
