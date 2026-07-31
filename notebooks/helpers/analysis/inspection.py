"""Edge inspection and matching analysis utilities."""

from src.meta.core.canonical_vector import CanonicalVector
from src.meta.core.vector_evaluator import DistributedRuntimeEvaluator


def inspect_matched_edges(
    seed,
    graph,
    best_vector,
    selected_algorithms,
    nr_of_nodes,
):
    """Inspect and display matched edges from algorithms.

    Args:
        seed: Random seed identifier
        graph: GraphManager instance
        best_vector: Best CanonicalVector from GA
        selected_algorithms: List of selected algorithms
        nr_of_nodes: Number of nodes in graph

    Returns:
        Dictionary with individual and merged matching results
    """
    print('\n' + '='*100)
    print(f'SEED {seed}')
    print('='*100 + '\n')

    # Display best vector parameters
    print('Best GA Vector Parameters:')
    param_names = sorted(best_vector.to_dict())
    for param_name in param_names:
        if hasattr(best_vector, param_name):
            value = getattr(best_vector, param_name)
            if isinstance(value, float):
                print(f'  {param_name:.<40} {value:.4f}')
            else:
                print(f'  {param_name:.<40} {value}')

    print('\n' + '-'*100 + '\n')

    # Inspect individual distributed-runtime evaluations.
    algo_names_list = [getattr(algorithm, "value", algorithm) for algorithm in selected_algorithms]
    evaluator = DistributedRuntimeEvaluator(max_workers=1)
    individual_results = {}

    for algo_name in algo_names_list:
        print(f'{algo_name.upper()} Algorithm Matching')
        print('-'*100)

        vector = CanonicalVector.from_dict(best_vector.to_dict(), algorithms=(algo_name,))
        matching = dict(evaluator.evaluate(graph, vector).matching)

        total_weight = sum(
            graph.get_edge_weight(u, v) for u, v in matching.items() if u < v
        )

        print(f'  Total edges matched: {len(matching) // 2}  (dict size: {len(matching)})')
        print(f'  Total weight: {total_weight:.0f}')
        print('\n  Edge list (first 10):')

        edges = [
            (u, v, graph.get_edge_weight(u, v))
            for u, v in matching.items()
            if u < v
        ]
        edges.sort(key=lambda x: -x[2])

        for i, (u, v, w) in enumerate(edges[:10]):
            print(f'    {i+1:2d}. ({u:3d}, {v:3d}) weight: {w:>7.2f}')

        if len(edges) > 10:
            print(f'    ... and {len(edges) - 10} more edges')

        individual_results[algo_name] = {
            'matching': matching,
            'weight': total_weight,
            'edges': edges,
        }

        print()

    print('-'*100)
    print('COMBINED DISTRIBUTED RESULT - SELECTED Algorithms')
    print(f'Selected: {", ".join(algo_names_list)}')
    print('-'*100 + '\n')

    merged = dict(evaluator.evaluate(graph, best_vector).matching)
    total_weight = sum(graph.get_edge_weight(u, v) for u, v in merged.items() if u < v)

    print(f'Total edges in merged result: {len(merged) // 2}')
    print(f'Total weight: {total_weight:.0f}')
    print('\nEdges (sorted by weight, descending):')

    edges = [(u, v, graph.get_edge_weight(u, v)) for u, v in merged.items() if u < v]
    edges.sort(key=lambda x: -x[2])

    for i, (u, v, w) in enumerate(edges[:20]):
        print(f'  {i+1:2d}. ({u:3d}, {v:3d}) weight: {w:>7.2f}')

    if len(edges) > 20:
        print(f'  ... and {len(edges) - 20} more edges')

    return {
        'individual_results': individual_results,
        'merged_matching': merged,
        'merged_weight': total_weight,
        'merged_edges': edges,
    }
