"""Selection-driven metadata for reproducible notebook experiments."""

from collections.abc import Iterable

from src.meta.core.algorithm_registry import AlgorithmRegistry
from src.meta.core.canonical_vector import CanonicalVector


def selected_experiment_metadata(selected_algorithms: Iterable[object]) -> dict:
    """Describe exactly the algorithms and vector parameters selected for one run."""
    names = tuple(getattr(algorithm, "value", algorithm) for algorithm in selected_algorithms)
    registry = AlgorithmRegistry.instance()
    vector = CanonicalVector(algorithms=names)
    return {
        "algorithms": {
            name: {
                "parameters": list(registry.get_algorithm_parameters(name)),
            }
            for name in names
        },
        "vector_parameters": {
            name: {"minimum": minimum, "maximum": maximum}
            for name, (minimum, maximum, _) in vector.parameter_definitions.items()
        },
    }
