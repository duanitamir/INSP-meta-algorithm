"""Notebook helper utilities for GA optimization analysis.

This package provides reusable functions for:
- Data loading and transformation
- Graph operations and fitness computation
- GA execution and results analysis
- Visualizations
- Edge inspection and debugging
"""

from .support.graph_utils import fixture_to_graph, format_time
from .support.fitness_utils import (
    get_optimal_weight,
    get_baseline_fitness,
    get_cascading_baseline,
    get_individual_algorithm_weights,
)
from .support.execution_utils import run_ga_evaluation
from .analysis.metrics import compute_metrics
from .visualization.plots import (
    plot_fitness_progression,
    plot_baseline_comparison,
    plot_performance_metrics,
    plot_parameter_space,
)
from .analysis.inspection import inspect_matched_edges

__all__ = [
    # Graph utilities
    "fixture_to_graph",
    "format_time",
    # Fitness utilities
    "get_optimal_weight",
    "get_baseline_fitness",
    "get_cascading_baseline",
    "get_individual_algorithm_weights",
    # Execution utilities
    "run_ga_evaluation",
    # Metrics
    "compute_metrics",
    # Visualizations
    "plot_fitness_progression",
    "plot_baseline_comparison",
    "plot_performance_metrics",
    "plot_parameter_space",
    # Analysis
    "inspect_matched_edges",
]
