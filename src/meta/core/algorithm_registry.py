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
        """Load algorithm definitions from parameterizers.

        Checks UnifiedAlgorithmParameterizer.ALGORITHM_DEFINITIONS first (primary source).
        Falls back to individual algorithm PARAMETER_DEFINITION attributes if needed.
        """
        definitions = {}

        # Try to load from UnifiedAlgorithmParameterizer.ALGORITHM_DEFINITIONS (primary source)
        try:
            from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer

            if hasattr(UnifiedAlgorithmParameterizer, "ALGORITHM_DEFINITIONS"):
                definitions = UnifiedAlgorithmParameterizer.ALGORITHM_DEFINITIONS.copy()
        except (ImportError, AttributeError):
            pass

        # Fall back to loading from individual algorithm PARAMETER_DEFINITION attributes
        if not definitions:
            try:
                from src.algorithms.implementations.greedy_matching import GreedyMatching
                from src.algorithms.implementations.itai_israeli import ItaiIsraeliMaximalMatching
                from src.algorithms.implementations.luby_randomized import LubyRandomizedMatching

                # Auto-discover parameter definitions from algorithm classes
                algorithms_to_load = [
                    (GreedyMatching, "greedy"),
                    (ItaiIsraeliMaximalMatching, "itai"),
                    (LubyRandomizedMatching, "luby"),
                ]

                for algo_class, algo_name in algorithms_to_load:
                    if hasattr(algo_class, "PARAMETER_DEFINITION"):
                        definitions[algo_name] = algo_class.PARAMETER_DEFINITION
            except (ImportError, AttributeError):
                pass

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
