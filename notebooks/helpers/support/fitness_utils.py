"""Fitness computation and baseline utilities."""

import networkx as nx

from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.core.fitness_evaluator import FitnessEvaluator
from src.meta.core.distributed_cascading_evaluator import CascadingEvaluator
from src.meta.parameterizers.algorithm_parameterizer import UnifiedAlgorithmParameterizer


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
        vector = CanonicalVector()
        return evaluator.evaluate(graph, vector)
    except Exception:
        return 0.0


def get_cascading_baseline(graph: GraphManager, config) -> float:
    """Compute cascading baseline using CascadingEvaluator.

    Args:
        graph: GraphManager instance
        config: Configuration object

    Returns:
        Cascading baseline fitness score
    """
    try:
        vector = CanonicalVector()
        cascading = CascadingEvaluator()
        return cascading.evaluate(graph, vector)
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
    vector = CanonicalVector()
    results = {}

    # Only compute for selected algorithms
    algo_names_list = [a.value for a in selected_algorithms]

    for algo_type in algo_names_list:
        param = UnifiedAlgorithmParameterizer(algo_type)
        matching = param.execute(graph, vector)
        weight = sum(graph.get_edge_weight(u, v) for u, v in matching.items() if u < v)
        results[algo_type] = weight

    return results
