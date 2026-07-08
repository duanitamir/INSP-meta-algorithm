"""Dynamic parameter vector that auto-extends based on selected algorithms.

Unlike CanonicalVector which has fixed parameters, DynamicCanonicalVector:
- Accepts list of algorithm names to include
- Auto-creates attributes for each algorithm's parameters
- Dynamically serializes/deserializes based on selected subset
- Supports mutation and validation with dynamic parameters
"""

import random
from typing import List, Tuple, Optional, Any, Dict
from src.meta.core.algorithm_registry import AlgorithmRegistry


class DynamicCanonicalVector:
    """Parameter vector that auto-extends based on selected algorithms.

    Creates a parameter space that includes only the algorithms specified.
    Each parameter is dynamically created as an attribute on the instance.

    Parameters are named as: {algorithm_name}_{parameter_name}
    (e.g., "greedy_max_rounds", "luby_base_probability")

    Example:
        # Create vector for Greedy and Luby
        vector = DynamicCanonicalVector(["greedy", "luby"])

        # Dynamically created parameters
        print(vector.greedy_max_rounds)
        print(vector.luby_base_probability)

        # Serialization
        params_list = vector.to_list()
        vector2 = DynamicCanonicalVector.from_list(params_list, ["greedy", "luby"])
    """

    def __init__(
        self,
        selected_algorithms: List[str],
        registry: Optional[AlgorithmRegistry] = None,
    ):
        """Initialize dynamic vector for selected algorithms.

        Args:
            selected_algorithms: List of algorithm names (e.g., ["greedy", "luby"])
            registry: AlgorithmRegistry to read parameters from (default: singleton)

        Raises:
            ValueError: If no algorithms selected or invalid algorithm names
        """
        if not selected_algorithms:
            raise ValueError("At least one algorithm must be selected")

        self.registry = registry or AlgorithmRegistry.instance()
        self.selected_algorithms = selected_algorithms

        # Build parameter metadata for selected algorithms
        self._parameters: Dict[str, Tuple[Any, Any, type]] = {}  # name -> (min, max, type)
        self._parameter_order: List[str] = []  # Maintains consistent serialization order
        self._values: Dict[str, Any] = {}  # Stores actual parameter values

        # Add base parameters used by cascading evaluator and distributed orchestrator
        # These are not algorithm-specific but required for system-level control
        base_params = {
            "max_iterations": (5, 100, lambda: random.randint(5, 100)),
            "convergence_threshold": (0.0, 0.1, lambda: random.uniform(0.0, 0.1)),
        }
        for param_name, (min_val, max_val, random_gen) in base_params.items():
            param_type = type(min_val)
            self._parameters[param_name] = (min_val, max_val, param_type)
            self._parameter_order.append(param_name)
            self._values[param_name] = random_gen()

        # Build parameters from selected algorithms
        for algo_name in selected_algorithms:
            algo_def = self.registry.get(algo_name)
            if algo_def is None:
                raise ValueError(f"Algorithm '{algo_name}' not found in registry")

            # Extract parameters from algorithm definition
            for param_name, (min_val, max_val, random_gen) in algo_def["parameters"].items():
                full_param_name = f"{algo_name}_{param_name}"
                param_type = type(min_val)

                # Store parameter bounds and type
                self._parameters[full_param_name] = (min_val, max_val, param_type)
                self._parameter_order.append(full_param_name)

                # Initialize with random value
                self._values[full_param_name] = random_gen()

    def __getattr__(self, name: str) -> Any:
        """Dynamic attribute access for parameters.

        Args:
            name: Parameter name (e.g., "greedy_max_rounds")

        Returns:
            Parameter value

        Raises:
            AttributeError: If parameter not found
        """
        # Avoid infinite recursion for private/protected attributes
        if name.startswith("_"):
            return super().__getattribute__(name)

        # Check if this is a parameter
        if name in self._values:
            return self._values[name]

        raise AttributeError(f"No parameter '{name}' in vector")

    def get_algorithms(self) -> List[str]:
        """Get list of selected algorithms in this vector.

        Returns:
            List of algorithm names (e.g., ["greedy", "luby"])
        """
        return self.selected_algorithms

    def __setattr__(self, name: str, value: Any) -> None:
        """Dynamic attribute setting for parameters.

        Args:
            name: Attribute or parameter name
            value: Value to set
        """
        # Handle private/protected attributes normally
        if name.startswith("_") or name in ["registry", "selected_algorithms"]:
            super().__setattr__(name, value)
        # Set parameter value if it exists
        elif hasattr(self, "_values") and name in self._values:
            self._values[name] = value
        else:
            super().__setattr__(name, value)

    def to_list(self) -> List[Any]:
        """Serialize parameters to list for GA operations.

        Returns:
            List of parameter values in canonical order.
        """
        return [self._values[param] for param in self._parameter_order]

    @classmethod
    def from_list(
        cls,
        params: List[Any],
        selected_algorithms: List[str],
        registry: Optional[AlgorithmRegistry] = None,
    ) -> "DynamicCanonicalVector":
        """Deserialize parameters from list.

        Args:
            params: List of parameter values in canonical order
            selected_algorithms: List of algorithm names
            registry: AlgorithmRegistry (default: singleton)

        Returns:
            New DynamicCanonicalVector with loaded parameters

        Raises:
            ValueError: If parameter count doesn't match expected
        """
        vector = cls(selected_algorithms, registry)

        if len(params) != len(vector._parameter_order):
            raise ValueError(
                f"Expected {len(vector._parameter_order)} parameters, "
                f"got {len(params)} for algorithms {selected_algorithms}"
            )

        # Load parameters from list
        for param_name, value in zip(vector._parameter_order, params):
            vector._values[param_name] = value

        return vector

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate all parameters are within their bounds.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        for param_name, value in self._values.items():
            if param_name not in self._parameters:
                return False, f"Unknown parameter: {param_name}"

            min_val, max_val, param_type = self._parameters[param_name]

            # Check bounds
            if not (min_val <= value <= max_val):
                return (
                    False,
                    f"Parameter '{param_name}' value {value} out of range "
                    f"[{min_val}, {max_val}]",
                )

            # Check type
            if type(value) != param_type:
                return (
                    False,
                    f"Parameter '{param_name}' has wrong type: "
                    f"expected {param_type.__name__}, got {type(value).__name__}",
                )

        return True, None

    def mutate(self, mutation_rate: float) -> "DynamicCanonicalVector":
        """Create mutated copy with some parameters randomized.

        Args:
            mutation_rate: Probability of mutating each parameter [0, 1]

        Returns:
            New DynamicCanonicalVector with mutations applied

        Raises:
            ValueError: If mutation_rate not in [0, 1]
        """
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0.0, 1.0]")

        # Create new vector with same algorithms
        mutant = DynamicCanonicalVector(self.selected_algorithms, self.registry)

        # Apply mutations
        for param in self._parameter_order:
            if random.random() < mutation_rate:
                # Mutate: generate new random value
                min_val, max_val, param_type = self._parameters[param]
                if param_type == float:
                    mutant._values[param] = random.uniform(min_val, max_val)
                else:
                    mutant._values[param] = random.randint(int(min_val), int(max_val))
            else:
                # Keep existing value
                mutant._values[param] = self._values[param]

        return mutant

    def crossover(self, other: "DynamicCanonicalVector") -> "DynamicCanonicalVector":
        """Create offspring by blending two parent vectors.

        Parents must have same algorithms selected.

        Args:
            other: Other parent vector

        Returns:
            New DynamicCanonicalVector with blended parameters

        Raises:
            ValueError: If parents have different algorithms
        """
        if set(self.selected_algorithms) != set(other.selected_algorithms):
            raise ValueError(
                f"Cannot crossover vectors with different algorithms: "
                f"{self.selected_algorithms} vs {other.selected_algorithms}"
            )

        # Create child with same algorithms
        child = DynamicCanonicalVector(self.selected_algorithms, self.registry)

        # Blend parameters from both parents
        for param in self._parameter_order:
            if random.random() < 0.5:
                child._values[param] = self._values[param]
            else:
                child._values[param] = other._values[param]

        return child

    def get_algorithms(self) -> List[str]:
        """Get list of selected algorithms.

        Returns:
            List of algorithm names (e.g., ["greedy", "luby"])
        """
        return self.selected_algorithms.copy()

    def get_parameter_count(self) -> int:
        """Get total number of parameters in this vector.

        Returns:
            Number of parameters
        """
        return len(self._parameter_order)

    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names in canonical order.

        Returns:
            List of parameter names (e.g., ["greedy_max_rounds", "luby_base_probability"])
        """
        return self._parameter_order.copy()

    def get_parameter_values(self) -> Dict[str, Any]:
        """Get all parameter values as dictionary.

        Returns:
            Dict mapping parameter name -> value
        """
        return self._values.copy()

    def __str__(self) -> str:
        """String representation for debugging."""
        algo_str = ", ".join(self.selected_algorithms)
        param_count = len(self._parameter_order)
        return f"DynamicCanonicalVector(algorithms=[{algo_str}], params={param_count})"

    def __repr__(self) -> str:
        """Representation for debugging."""
        return self.__str__()
