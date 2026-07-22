"""Canonical parameter vector for genetic algorithm optimization.

This module defines a generic parameter vector that discovers parameters
dynamically from available algorithms. The GA evolves these parameters to
maximize matching quality.

The vector is 100% agnostic to specific algorithms - it discovers and
evolves whatever parameters are available in the AlgorithmRegistry.
"""

import random
from typing import Tuple, List, Dict, Union, Any


class CanonicalVector:
    """Container for GA parameters governing algorithm specialization.

    This class manages a generic parameter vector that is 100% agnostic to
    specific algorithms. Parameters are discovered dynamically from
    AlgorithmRegistry, making it easy to add new algorithms without
    modifying this class.

    All parameters are validated to ensure they meet constraints.
    Supports serialization to/from dicts for GA operations.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize canonical parameter vector.

        Parameters are discovered from AlgorithmRegistry plus base parameters.
        For each parameter, if provided in kwargs, that value is used;
        otherwise a random value is generated within the parameter's bounds.

        Args:
            **kwargs: Optional specific parameter values
        """
        from src.meta.core.algorithm_registry import AlgorithmRegistry

        # Base parameters always available (generic, not algorithm-specific)
        base_params = {
            "max_iterations": (5, 100, lambda: __import__("random").randint(5, 100)),
            "convergence_threshold": (0.0, 0.1, lambda: __import__("random").uniform(0.0, 0.1)),
        }

        # Discover all available parameters from algorithm registry
        algo_params = AlgorithmRegistry.instance().all_parameter_definitions()

        # Merge base and algorithm-specific parameters
        self._parameter_definitions = {**base_params, **algo_params}

        # Initialize parameter storage
        self._parameters: Dict[str, Any] = {}

        # Generate initial values for each parameter
        for param_name, param_def in self._parameter_definitions.items():
            if param_name in kwargs:
                # Use provided value
                self._parameters[param_name] = kwargs[param_name]
            else:
                # Generate random value within bounds
                self._parameters[param_name] = self._random_value_for(param_def)

    def _random_value_for(self, param_def: Tuple[Any, Any, Any]) -> Any:
        """Generate random value for a parameter within its bounds.

        Args:
            param_def: Tuple of (min_val, max_val, random_gen_fn)

        Returns:
            Random value within parameter bounds
        """
        min_val, max_val, random_gen = param_def
        if random_gen is not None:
            return random_gen()
        elif isinstance(min_val, int) and isinstance(max_val, int):
            return random.randint(min_val, max_val)
        else:
            return random.uniform(float(min_val), float(max_val))

    def get(self, param_name: str) -> Any:
        """Get parameter value.

        Args:
            param_name: Name of parameter to retrieve

        Returns:
            Parameter value, or None if parameter doesn't exist
        """
        return self._parameters.get(param_name)

    def set(self, param_name: str, value: Any) -> None:
        """Set parameter value.

        Args:
            param_name: Name of parameter to set
            value: New value (must be valid for this parameter)
        """
        if param_name in self._parameter_definitions:
            self._parameters[param_name] = value

    def __getattr__(self, name: str) -> Any:
        """Support attribute-style access to parameters.

        Args:
            name: Parameter name

        Returns:
            Parameter value, or raise AttributeError if not found
        """
        # Avoid infinite recursion for private attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if it's a parameter
        if hasattr(self, "_parameters") and name in self._parameters:
            return self._parameters[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def validate(self) -> Tuple[bool, str | None]:
        """Validate all parameter constraints.

        Returns:
            Tuple of (is_valid, error_message)
        """
        for param_name, param_def in self._parameter_definitions.items():
            value = self._parameters.get(param_name)
            if value is None:
                return False, f"Parameter {param_name} is None"

            min_val, max_val, _ = param_def
            if not (min_val <= value <= max_val):
                return (
                    False,
                    f"{param_name} must be in [{min_val}, {max_val}], got {value}",
                )
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dict for serialization.

        Returns:
            Dict of all parameters
        """
        return self._parameters.copy()

    def to_list(self) -> List[float | int]:
        """Convert parameters to list for GA operations.

        Returns list in consistent order (sorted by parameter name for
        determinism across runs).

        Returns:
            List of parameter values
        """
        # Sort for deterministic ordering
        sorted_params = sorted(self._parameters.items())
        return [value for _, value in sorted_params]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalVector":
        """Create vector from dictionary.

        Args:
            data: Dict of parameter values

        Returns:
            New CanonicalVector with specified parameters
        """
        vector = cls()
        for param_name, value in data.items():
            vector.set(param_name, value)
        return vector

    @classmethod
    def from_list(cls, values: List[float | int]) -> "CanonicalVector":
        """Create vector from list.

        Parameters are matched to sorted parameter names for determinism.

        Args:
            values: List of parameter values in sorted order

        Returns:
            New CanonicalVector with specified parameters
        """
        vector = cls()
        sorted_params = sorted(vector._parameter_definitions.keys())

        if len(values) != len(sorted_params):
            raise ValueError(
                f"Expected {len(sorted_params)} values, got {len(values)}"
            )

        for param_name, value in zip(sorted_params, values):
            vector.set(param_name, value)

        return vector

    @classmethod
    def random(cls) -> "CanonicalVector":
        """Generate a random canonical vector.

        Returns:
            New CanonicalVector with all parameters randomly initialized
        """
        return cls()

    def crossover(self, other: "CanonicalVector") -> "CanonicalVector":
        """Perform crossover with another vector.

        Creates a new vector by randomly selecting parameters from this
        vector or the other vector.

        Args:
            other: Another CanonicalVector to crossover with

        Returns:
            New CanonicalVector from crossover
        """
        child_dict = {}
        for param_name in self._parameters.keys():
            # Randomly select from parent1 or parent2
            if random.random() < 0.5:
                child_dict[param_name] = self._parameters[param_name]
            else:
                child_dict[param_name] = other._parameters.get(
                    param_name, self._parameters[param_name]
                )

        return CanonicalVector.from_dict(child_dict)

    def mutate(self, mutation_rate: float = 0.1) -> "CanonicalVector":
        """Perform mutation on parameters.

        Each parameter has mutation_rate probability of being replaced with
        a random value within its bounds.

        Args:
            mutation_rate: Probability of mutating each parameter [0, 1]

        Returns:
            New mutated CanonicalVector
        """
        mutated_dict = self._parameters.copy()

        for param_name, param_def in self._parameter_definitions.items():
            if random.random() < mutation_rate:
                # Replace with random value
                mutated_dict[param_name] = self._random_value_for(param_def)

        return CanonicalVector.from_dict(mutated_dict)


    def __repr__(self) -> str:
        """String representation of vector."""
        params_str = ", ".join(
            f"{k}={round(v, 3) if isinstance(v, float) else v}"
            for k, v in sorted(self._parameters.items())
        )
        return f"CanonicalVector({params_str})"
