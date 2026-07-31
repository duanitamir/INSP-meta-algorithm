"""Fitness computation and baseline utilities."""

import networkx as nx

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.core.fitness_evaluator import FitnessEvaluator
from src.meta.core.distributed_cascading_evaluator import DistributedCascadingEvaluator
from src.meta.core.vector_evaluator import DistributedRuntimeEvaluator


def get_optimal_weight(fixture_dict) -> float:
    """Compute optimal matching weight using NetworkX.

    Args:
        fixture_dict: Dictionary with 'vertices' and 'edges' lists

    Returns:
        Maximum matching weight
    """
    try:
        G = nx.Graph()
        for v in fixture_dict['vertices']:
            G.add_node(v)
        for u, v, w in fixture_dict['edges']:
            G.add_edge(u, v, weight=float(w))
        matching = nx.max_weight_matching(G, weight='weight', maxcardinality=False)
        return sum(G[u][v].get('weight', 1.0) for u, v in matching)
    except Exception:
        return 0.0


def get_baseline_fitness(graph: GraphManager, config) -> float:
    """Compute baseline fitness for selected algorithms.

    Args:
        graph: GraphManager instance
        config: Configuration object

    Returns:
        Baseline fitness score
    """
    try:
        evaluator = FitnessEvaluator()
        vector = CanonicalVector(algorithms=getattr(config, "algorithms", None))
        return evaluator.evaluate(graph, vector).score
    except Exception:
        return 0.0


def get_cascading_baseline(graph: GraphManager, config) -> float:
    """Compute cascading baseline using DistributedCascadingEvaluator.

    Args:
        graph: GraphManager instance
        config: Configuration object

    Returns:
        Cascading baseline fitness score
    """
    try:
        vector = CanonicalVector(algorithms=getattr(config, "algorithms", None))
        cascading = DistributedCascadingEvaluator()
        return cascading.evaluate(graph, vector).score
    except Exception:
        return 0.0


def get_individual_algorithm_weights(graph: GraphManager, selected_algorithms) -> dict:
    """Get weight for each SELECTED algorithm individually.

    Args:
        graph: GraphManager instance
        selected_algorithms: List of selected Algorithms enum values

    Returns:
        Dictionary mapping algorithm name to weight
    """
    algorithm_names = tuple(getattr(algorithm, "value", algorithm) for algorithm in selected_algorithms)
    selected_vector = CanonicalVector(algorithms=algorithm_names)
    evaluator = DistributedRuntimeEvaluator(max_workers=1)
    results = {}

    for algorithm_name in algorithm_names:
        vector = CanonicalVector.from_dict(selected_vector.to_dict(), algorithms=(algorithm_name,))
        results[algorithm_name] = evaluator.evaluate(graph, vector).score

    return results
