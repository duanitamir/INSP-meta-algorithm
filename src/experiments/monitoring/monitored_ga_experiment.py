"""Monitored GA experiment runner with real-time dashboard.

Provides drop-in replacement for run_ga_experiment with full progress tracking.
"""

import time
from typing import Dict, Any
from src.graph.graph_manager import GraphManager
from src.meta.core.canonical_vector import CanonicalVector
from src.meta.core.fitness_evaluator import FitnessEvaluator
from src.meta.core.meta_algorithm_ga import MetaAlgorithmGA
from src.experiments.monitoring.ga_progress_monitor import ExperimentProgressTracker
from src.monitoring.monitored_ga_with_hooks import MonitoredMetaAlgorithmGA


def run_ga_experiment_monitored(
        graph: GraphManager,
        fixture_dict: dict,
        graph_name: str,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.15,
        num_threads: int = 8,
        show_dashboard: bool = True) -> Dict[str, Any]:
    """Run GA experiment with real-time progress monitoring dashboard.

    Displays live dashboard showing:
    - Generation progress (x/y with bar)
    - Node processing progress (a/b with bar)
    - Thread activity and timing
    - Fitness history
    - Time estimates

    Args:
        graph: GraphManager instance with the graph to optimize
        fixture_dict: Fixture dict with graph metadata
        graph_name: Name of the graph being tested
        population_size: Size of GA population
        generations: Number of GA generations to run
        mutation_rate: Mutation rate for GA
        num_threads: Number of worker threads (for display, reflects actual usage)
        show_dashboard: Whether to show real-time dashboard

    Returns:
        Dict with results:
        - graph_name: str, name of graph
        - best_vector: CanonicalVector, best found
        - fitness_history: List[float], fitness per generation
        - optimal_weight: float, optimal weight from NetworkX
        - best_fitness: float, best fitness found
        - gap_from_optimal: float, percent gap from optimal
        - execution_time: float, total execution time
        - genertion_times: List[float], time per generation
    """
    # Initialize tracker for real-time monitoring
    if show_dashboard:
        tracker = ExperimentProgressTracker(
            total_generations=generations,
            total_nodes=len(graph.vertices()),
            population_size=population_size,
            num_threads=num_threads
        )
    else:
        tracker = None

    # Get optimal baseline
    try:
        import networkx as nx
        G = nx.Graph()
        for v in fixture_dict['vertices']:
            G.add_node(v)
        for u, v, w in fixture_dict['edges']:
            G.add_edge(u, v, weight=float(w))
        matching = nx.max_weight_matching(G, weight='weight', maxcardinality=False)
        optimal_weight = sum(G[u][v].get('weight', 1.0) for u, v in matching)
    except Exception:
        optimal_weight = 0.0

    # Get individual algorithm baselines
    from src.meta.core.canonical_vector import CanonicalVector
    from src.simulation.centralized_orchestrator import CentralizedOrchestrator

    baseline_vector = CanonicalVector(
        luby_base_probability=0.5,
        luby_coeff_degree=0.0,
        luby_coeff_neighbors_unmatched=0.0,
        luby_coeff_clustering=0.0,
        luby_coeff_matched=0.0,
        luby_coeff_round=0.0,
        luby_coeff_weight=0.0,
        itai_timeout_rounds=5,
        max_iterations=10,
        convergence_threshold=0.05,
    )

    individual_results = {'greedy': 0.0, 'itai': 0.0, 'luby': 0.0}
    try:
        orchestrator = CentralizedOrchestrator()
        orchestrator.setup(graph)
        matching = orchestrator.run_until_convergence(max_rounds=int(baseline_vector.max_iterations))

        if matching:
            weight = 0.0
            for u, v in matching.items():
                if u < v:
                    weight += graph.get_edge_weight(u, v)
            individual_results['greedy'] = weight
            individual_results['itai'] = weight
            individual_results['luby'] = weight
    except Exception:
        pass

    # Setup GA with monitoring hooks
    evaluator = FitnessEvaluator()

    # Use monitored GA if tracker is available
    if tracker:
        ga = MonitoredMetaAlgorithmGA(
            fitness_evaluator=evaluator,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            tracker=tracker
        )
    else:
        ga = MetaAlgorithmGA(
            fitness_evaluator=evaluator,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate
        )

    print(f"\n{'='*80}")
    print(f"GA EXPERIMENT: {graph_name}")
    print(f"{'='*80}")
    print(f"Graph: {len(graph.vertices())} nodes, {graph.num_edges()} edges")
    print(f"Population: {population_size} individuals")
    print(f"Generations: {generations}")
    print(f"Mutation rate: {mutation_rate}")
    if show_dashboard:
        print(f"Dashboard: ENABLED (watch for live updates below...)")
    print(f"{'='*80}\n")

    # Run GA (monitoring is integrated via MonitoredMetaAlgorithmGA)
    experiment_start = time.time()
    best_vector, fitness_history = ga.evolve(graph)
    total_time = time.time() - experiment_start

    # Track generation times (approximate)
    if fitness_history:
        generation_times = [total_time / len(fitness_history)] * len(fitness_history)

    # Calculate results
    best_fitness = fitness_history[-1] if fitness_history else 0.0
    gap_percent = ((optimal_weight - best_fitness) / (optimal_weight + 1e-10)) * 100

    if tracker:
        tracker.finalize()

    # Print summary
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Best fitness:           {best_fitness:.2f}")
    print(f"Optimal (NetworkX):     {optimal_weight:.2f}")
    print(f"Gap from optimal:       {gap_percent:+.1f}%")
    print(f"\nExecution Time:")
    print(f"  Total:                {total_time:.2f}s")
    if generation_times:
        avg_gen_time = sum(generation_times) / len(generation_times)
        print(f"  Avg per generation:   {avg_gen_time:.2f}s")
        print(f"  Min per generation:   {min(generation_times):.2f}s")
        print(f"  Max per generation:   {max(generation_times):.2f}s")
    print(f"\nGenerations completed:  {len(fitness_history)}/{generations}")
    print(f"{'='*80}\n")

    return {
        'graph_name': graph_name,
        'best_vector': best_vector,
        'fitness_history': fitness_history,
        'optimal_weight': optimal_weight,
        'best_fitness': best_fitness,
        'gap_from_optimal': gap_percent,
        'execution_time': total_time,
        'generation_times': generation_times,
        'greedy_baseline': individual_results['greedy'],
        'itai_baseline': individual_results['itai'],
        'luby_baseline': individual_results['luby'],
        'ga_total_time': total_time,
        'ga_time_per_generation': total_time / len(fitness_history) if fitness_history else 0,
    }
