"""Metrics computation for GA analysis."""


def compute_metrics(all_results, seeds):
    """Compute summary metrics across all seeds.

    Args:
        all_results: Dictionary of results keyed by seed
        seeds: List of seed values

    Returns:
        Dictionary with improvement and gap metrics
    """
    improvements = []
    gaps = []

    for seed in seeds:
        r = all_results[seed]
        improvements.append(r['improvement'])
        gaps.append(r['gap'])

    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    return {
        'improvements': improvements,
        'gaps': gaps,
        'avg_improvement': avg_improvement,
        'avg_gap': avg_gap,
    }


def compute_local_metrics(baseline, optimal, best_fitness):
    """Compute metrics for a single seed.

    Args:
        baseline: Baseline fitness
        optimal: Optimal fitness (NetworkX)
        best_fitness: GA-found best fitness

    Returns:
        Dictionary with improvement and gap metrics
    """
    improvement = ((best_fitness - baseline) / (baseline + 1e-10)) * 100
    gap = ((optimal - best_fitness) / (optimal + 1e-10)) * 100

    return {
        'improvement': improvement,
        'gap': gap,
    }
