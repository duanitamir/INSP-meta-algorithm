"""Phase 2 centralized meta-algorithm (reference implementation).

Contains:
- CanonicalVector: 10-parameter GA chromosome
- GAConfig: Configuration presets (small_graph, medium_graph, large_graph, etc.)
- FitnessEvaluator: Evaluate fitness of parameter vectors
- MetaAlgorithmGA: Centralized genetic algorithm for parameter optimization
"""

from .canonical_vector import CanonicalVector
from .ga_config import GAConfig
from .fitness_evaluator import FitnessEvaluator
from .meta_algorithm_ga import MetaAlgorithmGA, PopulationEvaluation

__all__ = [
    "CanonicalVector",
    "GAConfig",
    "FitnessEvaluator",
    "MetaAlgorithmGA",
    "PopulationEvaluation",
]
