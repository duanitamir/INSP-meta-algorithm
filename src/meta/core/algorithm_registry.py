"""Singleton registry for managing available algorithms.

Reads algorithm declarations from parameterizers and provides:
- Singleton access to all registered algorithms
- Thread-safe retrieval of algorithm definitions
- Selection of algorithm subsets for GA runs
"""

import threading
from typing import Dict, List, Optional, Any


class AlgorithmRegistry:
    """Singleton registry for all available algorithms.

    Manages algorithm metadata and provides access to algorithm definitions.
    Reads from ALGORITHM_DEFINITIONS in parameterizers.

    Thread-safe: Uses lock for concurrent access.

    Example:
        registry = AlgorithmRegistry.instance()
        all_algos = registry.get_all()
        algo_def = registry.get("greedy")
        selected = registry.get_selected(["greedy", "luby"])
    """

    _instance: Optional["AlgorithmRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize registry and load algorithm definitions."""
        self._algorithms: Dict[str, Dict[str, Any]] = {}
        self._registry_lock = threading.Lock()
        self._load_from_parameterizers()

    def _load_from_parameterizers(self) -> None:
        """Load algorithm definitions from AlgorithmRegistryBuilder.

        Algorithms self-register when their modules are imported via the
        centralized register_all module. This ensures all algorithms are
        registered and available in the registry.
        """
        # Import centralized registration module (triggers all algorithm registrations)
        import src.algorithms.implementations.register_all  # noqa

        # Read what was registered (algorithms registered themselves on import)
        from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder
        definitions = AlgorithmRegistryBuilder.get_all_definitions()

        with self._registry_lock:
            self._algorithms = definitions

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all registered algorithm definitions.

        Returns:
            List of algorithm definition dicts.
        """
        with self._registry_lock:
            return list(self._algorithms.values())

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific algorithm definition by name.

        Args:
            name: Algorithm name (e.g., "greedy", "itai", "luby")

        Returns:
            Algorithm definition dict, or None if not found.
        """
        with self._registry_lock:
            return self._algorithms.get(name)

    def get_selected(self, algorithm_names: List[str]) -> List[Dict[str, Any]]:
        """Get specific subset of algorithm definitions.

        Args:
            algorithm_names: List of algorithm names to retrieve.

        Returns:
            List of algorithm definition dicts for selected algorithms.
            Only includes algorithms that exist in the registry.
        """
        with self._registry_lock:
            return [
                self._algorithms[name]
                for name in algorithm_names
                if name in self._algorithms
            ]

    def all_algorithm_names(self) -> List[str]:
        """Get names of all registered algorithms.

        Returns:
            List of algorithm names (e.g., ["greedy", "itai", "luby"])
        """
        with self._registry_lock:
            return list(self._algorithms.keys())

    def is_algorithm_registered(self, name: str) -> bool:
        """Check if algorithm is registered.

        Args:
            name: Algorithm name.

        Returns:
            True if algorithm exists in registry, False otherwise.
        """
        with self._registry_lock:
            return name in self._algorithms

    def register(self, name: str, definition: Dict[str, Any]) -> None:
        """Register a new algorithm definition.

        Args:
            name: Algorithm name.
            definition: Algorithm definition dict with "name" and "parameters" keys.
        """
        with self._registry_lock:
            self._algorithms[name] = definition

    def unregister(self, name: str) -> bool:
        """Unregister an algorithm.

        Args:
            name: Algorithm name.

        Returns:
            True if algorithm was registered and removed, False otherwise.
        """
        with self._registry_lock:
            if name in self._algorithms:
                del self._algorithms[name]
                return True
            return False

    def reset(self) -> None:
        """Reset registry to empty state. Useful for testing."""
        with self._registry_lock:
            self._algorithms.clear()

    def reload(self) -> None:
        """Reload algorithm definitions from parameterizers.

        Useful if parameterizers have been updated.
        """
        self.reset()
        self._load_from_parameterizers()

    def all_parameter_definitions(self) -> Dict[str, tuple]:
        """Collect all parameter definitions from all algorithms.

        Returns a dict mapping parameter_name -> (min, max, random_gen_fn).
        Algorithm-specific parameter names are prefixed with algorithm name to avoid collisions.
        This is used by CanonicalVector to discover all available parameters.

        Returns:
            Dict mapping parameter names to their (min, max, random_gen) tuples.
        """
        all_params = {}

        with self._registry_lock:
            for algo_name, algo_def in self._algorithms.items():
                if "parameters" in algo_def:
                    params = algo_def["parameters"]
                    for param_name, param_spec in params.items():
                        # Prefix algorithm-specific parameters (except max_rounds which is generic)
                        if param_name != "max_iterations" and param_name != "convergence_threshold":
                            final_name = f"{algo_name}_{param_name}"
                        else:
                            final_name = param_name

                        all_params[final_name] = param_spec

        return all_params

    def get_algorithm_parameters(self, algorithm_name: str) -> Dict[str, tuple]:
        """Get parameter definitions for a specific algorithm (with algorithm prefix).

        Parameters are returned with algorithm name prefix for consistency with
        all_parameter_definitions().

        Args:
            algorithm_name: Algorithm name (e.g., "greedy", "itai", "luby")

        Returns:
            Dict mapping (algorithm-prefixed) parameter names to (min, max, random_gen) tuples,
            or empty dict if algorithm not found.
        """
        algo_def = self.get(algorithm_name)
        if not algo_def or "parameters" not in algo_def:
            return {}

        params = {}
        for param_name, param_spec in algo_def["parameters"].items():
            # Use same prefixing logic as all_parameter_definitions()
            if param_name not in ["max_iterations", "convergence_threshold"]:
                final_name = f"{algorithm_name}_{param_name}"
            else:
                final_name = param_name
            params[final_name] = param_spec

        return params

    def get_algorithm_parameters_unprefixed(self, algorithm_name: str) -> Dict[str, tuple]:
        """Get parameter definitions for a specific algorithm (without algorithm prefix).

        Parameters are returned without algorithm name prefix, useful for algorithm
        classes that want their own parameters.

        Args:
            algorithm_name: Algorithm name (e.g., "greedy", "itai", "luby")

        Returns:
            Dict mapping parameter names (unprefixed) to (min, max, random_gen) tuples,
            or empty dict if algorithm not found.
        """
        algo_def = self.get(algorithm_name)
        if not algo_def or "parameters" not in algo_def:
            return {}

        return algo_def["parameters"].copy()

    @classmethod
    def instance(cls) -> "AlgorithmRegistry":
        """Get singleton instance of algorithm registry.

        Thread-safe: Uses class-level lock for initialization.

        Returns:
            Singleton AlgorithmRegistry instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AlgorithmRegistry()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance. Useful for testing."""
        with cls._lock:
            cls._instance = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        names = self.all_algorithm_names()
        return f"AlgorithmRegistry(algorithms={names})"
