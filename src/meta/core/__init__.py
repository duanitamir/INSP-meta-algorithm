"""Phase 2 centralized meta-algorithm (reference implementation).

Contains:
- CanonicalVector: 10-parameter GA chromosome
- FitnessEvaluator: Evaluate fitness of parameter vectors
- MetaAlgorithmGA: Centralized genetic algorithm for parameter optimization
"""

from .canonical_vector import CanonicalVector
from .fitness_evaluator import FitnessEvaluator
from .distributed_cascading_evaluator import DistributedCascadingEvaluator
from .meta_algorithm_ga import MetaAlgorithmGA, PopulationEvaluation

__all__ = [
    "CanonicalVector",
    "FitnessEvaluator",
    "DistributedCascadingEvaluator",
    "MetaAlgorithmGA",
    "PopulationEvaluation",
]
