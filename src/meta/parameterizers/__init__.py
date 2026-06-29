"""Algorithm parameterizers: wrappers for matching algorithms with parameter tuning.

Contains:
- AlgorithmParameterizer: Abstract base class (template method pattern)
- GreedyParameterizer: Greedy matching with canonical vector parameters
- ItaiParameterizer: Itai-Israeli matching with canonical vector parameters
- LubyParameterizer: Luby randomized with adaptive activation from parameters
- ParameterizerFactory: Factory for creating standard parameterizer sets
"""

from .base import AlgorithmParameterizer
from .greedy import GreedyParameterizer
from .itai import ItaiParameterizer
from .luby import LubyParameterizer
from .factory import ParameterizerFactory

__all__ = [
    "AlgorithmParameterizer",
    "GreedyParameterizer",
    "ItaiParameterizer",
    "LubyParameterizer",
    "ParameterizerFactory",
]
