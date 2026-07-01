"""Factory for creating parameterizer sets."""

from typing import TYPE_CHECKING, List

from src.meta.parameterizers.greedy import GreedyParameterizer
from src.meta.parameterizers.itai import ItaiParameterizer
from src.meta.parameterizers.luby import LubyParameterizer

if TYPE_CHECKING:
    from src.meta.parameterizers.base import AlgorithmParameterizer


class ParameterizerFactory:
    """Factory for creating standard parameterizer combinations.

    Provides presets for common algorithm combinations and enables
    easy experimentation with different algorithm subsets.
    """

    @staticmethod
    def create_default() -> List[AlgorithmParameterizer]:
        """Create default 3-algorithm set (Greedy, Itai, Luby).

        Returns:
            List[AlgorithmParameterizer]: Standard parameterizers for meta-algorithm
        """
        return [
            GreedyParameterizer(),
            ItaiParameterizer(),
            LubyParameterizer(),
        ]

    @staticmethod
    def create_luby_only() -> List[AlgorithmParameterizer]:
        """Create Luby-only set (for debugging/baseline).

        Returns:
            List[AlgorithmParameterizer]: Single Luby parameterizer
        """
        return [LubyParameterizer()]

    @staticmethod
    def create_greedy_itai() -> List[AlgorithmParameterizer]:
        """Create Greedy + Itai set (deterministic algorithms).

        Returns:
            List[AlgorithmParameterizer]: Greedy and Itai parameterizers
        """
        return [
            GreedyParameterizer(),
            ItaiParameterizer(),
        ]
