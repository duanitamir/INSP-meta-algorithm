"""Centralized algorithm registration and enum definition.

This module:
1. Defines the Algorithms enum (single source of truth for algorithm names)
2. Imports all algorithm implementations
3. Registers them with AlgorithmRegistryBuilder in one place

Import this module to ensure all algorithms are available:
    from src.algorithms.implementations.register_all import *

The Algorithms enum lives here (not in meta.config) to ensure it's always in sync
with registered algorithms and to maintain 100% agnosticism in runtime code.
"""

from enum import Enum
from src.meta.core.algorithm_registry_builder import AlgorithmRegistryBuilder

# Import all algorithms
from src.algorithms.implementations.greedy_matching import GreedyMatching
from src.algorithms.implementations.itai_israeli import ItaiIsraeliMaximalMatching
from src.algorithms.implementations.luby_randomized import LubyRandomizedMatching


class Algorithms(Enum):
    """Algorithm selection enum - auto-discovered from registry.

    This is the single source of truth for algorithm names. Add new algorithms by:
    1. Creating algorithm class (e.g., AuctionMatching)
    2. Importing it above
    3. Adding enum entry below
    4. Registering it below with AlgorithmRegistryBuilder.register()
    5. That's it! All runtime code discovers it via registry.
    """
    GREEDY = "greedy"
    ITAI = "itai"
    LUBY = "luby"

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Algorithms.{self.name}"


# Register all algorithms in one place (centralized registration)
AlgorithmRegistryBuilder.register(
    Algorithms.GREEDY.value,  # Use .value to get string name
    GreedyMatching.PARAMETER_DEFINITION,
    GreedyMatching
)

AlgorithmRegistryBuilder.register(
    Algorithms.ITAI.value,
    ItaiIsraeliMaximalMatching.PARAMETER_DEFINITION,
    ItaiIsraeliMaximalMatching
)

AlgorithmRegistryBuilder.register(
    Algorithms.LUBY.value,
    LubyRandomizedMatching.PARAMETER_DEFINITION,
    LubyRandomizedMatching
)

__all__ = [
    "Algorithms",
    "GreedyMatching",
    "ItaiIsraeliMaximalMatching",
    "LubyRandomizedMatching",
]
