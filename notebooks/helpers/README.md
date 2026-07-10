# Notebook Helpers

Reusable utilities for GA optimization analysis notebooks.

## Structure

```
helpers/
├── support/          # Basic utilities
│   ├── graph_utils.py         # Graph loading and transformation
│   ├── fitness_utils.py       # Fitness computation and baselines
│   └── execution_utils.py     # GA execution
├── analysis/         # Analysis and metrics
│   ├── metrics.py             # Metrics computation
│   └── inspection.py          # Edge inspection and matching analysis
├── visualization/    # Plotting utilities
│   └── plots.py               # All visualization functions
└── __init__.py       # Package exports
```

## Usage

### Quick Start

```python
from notebooks.helpers import (
    # Graph utilities
    fixture_to_graph,
    format_time,
    # Fitness utilities
    get_optimal_weight,
    get_baseline_fitness,
    get_cascading_baseline,
    get_individual_algorithm_weights,
    # Execution
    run_ga_evaluation,
    # Metrics
    compute_metrics,
    # Visualization
    plot_fitness_progression,
    plot_baseline_comparison,
    plot_performance_metrics,
    plot_parameter_space,
    # Analysis
    inspect_matched_edges,
)

# Use in your notebook
graph = fixture_to_graph(fixture)
optimal = get_optimal_weight(fixture)
algo_weights = get_individual_algorithm_weights(graph, SELECTED_ALGORITHMS)
plot_fitness_progression(all_results, seeds, algo_names_str)
```

## Modules

### `support/graph_utils.py`
Graph loading and time formatting utilities.

**Functions:**
- `fixture_to_graph(fixture_dict)` - Convert fixture dict to GraphManager
- `format_time(seconds)` - Format duration as human-readable string

### `support/fitness_utils.py`
Fitness computation and baseline evaluation.

**Functions:**
- `get_optimal_weight(fixture_dict)` - Compute optimal matching (NetworkX)
- `get_baseline_fitness(graph, config)` - Standard algorithm baseline
- `get_cascading_baseline(graph, config)` - Cascading evaluator baseline
- `get_individual_algorithm_weights(graph, selected_algorithms)` - Per-algorithm weights

### `support/execution_utils.py`
GA execution utilities.

**Functions:**
- `run_ga_evaluation(graph, config)` - Run GA with configured parameters

### `analysis/metrics.py`
Metrics computation.

**Functions:**
- `compute_metrics(all_results, seeds)` - Summary metrics across seeds
- `compute_local_metrics(baseline, optimal, best_fitness)` - Per-seed metrics

### `analysis/inspection.py`
Edge inspection and matching analysis.

**Functions:**
- `inspect_matched_edges(seed, graph, best_vector, selected_algorithms, nr_of_nodes)` - Inspect matched edges and display results

### `visualization/plots.py`
Plotting utilities for analysis visualization.

**Functions:**
- `plot_fitness_progression(all_results, seeds, algo_names_str)` - Fitness progression per seed
- `plot_baseline_comparison(all_results, seeds, algo_names_str)` - Baseline comparison bar chart
- `plot_performance_metrics(all_results, seeds)` - Performance metrics plots
- `plot_parameter_space(all_results, seeds)` - Parameter space exploration

## Creating New Notebooks

To use these helpers in a new notebook:

1. Import the helpers:
```python
from notebooks.helpers import *
```

2. Use the provided functions to simplify your notebook:
```python
# Instead of writing visualization code, call:
plot_fitness_progression(all_results, seeds, algo_names_str)
```

3. Focus on your analysis logic, not boilerplate!

## Benefits

✅ **DRY Principle**: Reuse code across multiple notebooks  
✅ **Clean Notebooks**: Less clutter, more focus on analysis  
✅ **Maintainability**: Fix bugs in one place, fixes all notebooks  
✅ **Consistency**: All notebooks use the same analysis functions  
✅ **Extensibility**: Easy to add new utilities as needed  

## Adding New Functions

To add a new utility:

1. Decide which module it belongs to (support, analysis, visualization)
2. Add the function to the appropriate module
3. Update `__init__.py` to export it
4. Update this README

Example:

```python
# analysis/comparisons.py
def compare_ga_approaches(all_results, seeds):
    """Compare GA standard vs cascading approaches."""
    ...

# Update __init__.py
from .analysis.comparisons import compare_ga_approaches

# Now use in any notebook!
from notebooks.helpers import compare_ga_approaches
```
