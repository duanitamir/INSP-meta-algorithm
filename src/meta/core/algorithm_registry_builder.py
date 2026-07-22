"""Algorithm self-registration via builder pattern.

Algorithms register themselves at module-level, enabling true plugin architecture.
Runtime code only looks up algorithms by name - zero hardcoded imports.
"""

from typing import Dict, Any, Type


class AlgorithmRegistryBuilder:
    """Singleton builder for algorithm self-registration.

    Algorithms call AlgorithmRegistryBuilder.register() at module level.
    Runtime code calls AlgorithmRegistryBuilder.get_class() to retrieve algorithms.

    This decouples runtime code from algorithm implementations.
    """

    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, param_def: Dict[str, Any], algo_class: Type) -> None:
        """Register an algorithm with its class and metadata.

        Called by algorithms at module level (e.g., end of greedy_matching.py).

        Args:
            name: Algorithm name string (e.g., "greedy", "itai", "luby")
            param_def: PARAMETER_DEFINITION dict with algorithm parameters and metadata
            algo_class: The algorithm class itself (e.g., GreedyMatching)
        """
        cls._registry[name] = {
            "param_def": param_def,
            "class": algo_class,
            "display_name": param_def.get("display_name", name),
        }

    @classmethod
    def get_class(cls, name: str) -> Type | None:
        """Get algorithm class by name.

        Args:
            name: Algorithm name string

        Returns:
            Algorithm class, or None if not registered
        """
        entry = cls._registry.get(name)
        return entry["class"] if entry else None

    @classmethod
    def get_display_name(cls, name: str) -> str:
        """Get algorithm display name (for UI).

        Args:
            name: Algorithm name string

        Returns:
            Display name (e.g., "Greedy Matching"), or the name itself if not registered
        """
        entry = cls._registry.get(name)
        return entry.get("display_name", name) if entry else name

    @classmethod
    def get_param_def(cls, name: str) -> Dict[str, Any] | None:
        """Get algorithm parameter definition by name.

        Args:
            name: Algorithm name

        Returns:
            PARAMETER_DEFINITION dict, or None if not registered
        """
        entry = cls._registry.get(name)
        return entry["param_def"] if entry else None

    @classmethod
    def get_all_definitions(cls) -> Dict[str, Any]:
        """Get all parameter definitions (for AlgorithmRegistry).

        Returns:
            Dict mapping algorithm name -> parameter definition
        """
        return {name: entry["param_def"] for name, entry in cls._registry.items()}

    @classmethod
    def clear(cls) -> None:
        """Clear registry (for testing).

        Returns:
            None
        """
        cls._registry.clear()
