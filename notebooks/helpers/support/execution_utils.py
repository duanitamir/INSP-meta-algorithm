"""GA execution utilities."""

from src.graph.graph_manager import GraphManager
from src.meta.core.meta_algorithm_ga import MetaAlgorithmGA


def run_ga_evaluation(graph: GraphManager, config) -> tuple:
    """Run GA with configured algorithms.

    Uses the old API for backward compatibility with notebooks.

    Args:
        graph: GraphManager instance
        config: Configuration object with ga_config

    Returns:
        Tuple of (best_vector, fitness_history)
    """
    use_cascading = (
        config.ga_config.use_cascading
        if hasattr(config.ga_config, 'use_cascading')
        else False
    )
    ga = MetaAlgorithmGA(
        population_size=config.ga_config.population_size,
        generations=config.ga_config.generations,
        mutation_rate=config.ga_config.mutation_rate,
        use_cascading=use_cascading,
    )
    return ga.evolve(graph)
