"""Visualization utilities for GA analysis."""

import numpy as np
import matplotlib.pyplot as plt


def plot_fitness_progression(all_results, seeds, algo_names_str):
    """Plot full-graph distributed GA fitness progression for each seed."""
    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 5), squeeze=False)
    for axis, seed in zip(axes[0], seeds):
        result = all_results[seed]
        history = result['fitness_history']
        axis.plot(range(1, len(history) + 1), history, 'o-', label='Evolved vector')
        axis.axhline(result['baseline'], linestyle='--', label='Baseline')
        axis.axhline(result['optimal'], linestyle='-', label='NetworkX optimum')
        for name, weight in result['algo_weights'].items():
            axis.axhline(weight, linestyle=':', alpha=0.6, label=name)
        axis.set_title(f'Seed {seed}')
        axis.set_xlabel('Generation')
        axis.set_ylabel('Matching weight')
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle(f'Full-graph distributed evolution — {algo_names_str}')
    fig.tight_layout()
    plt.show()


def plot_baseline_comparison(all_results, seeds, algo_names_str):
    """Compare selected-policy baselines, evolved vectors, and optimum."""
    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 5), squeeze=False)
    for axis, seed in zip(axes[0], seeds):
        result = all_results[seed]
        labels = [*result['algo_weights'], 'Baseline', 'Evolved', 'Optimal']
        values = [*result['algo_weights'].values(), result['baseline'], result['best_score'], result['optimal']]
        axis.bar(labels, values)
        axis.set_title(f'Seed {seed}')
        axis.tick_params(axis='x', rotation=45)
        axis.set_ylabel('Matching weight')
    fig.suptitle(f'Full-graph evaluation — {algo_names_str}')
    fig.tight_layout()
    plt.show()


def plot_performance_metrics(all_results, seeds):
    """Plot improvement from the baseline and gap to the NetworkX optimum."""
    positions = np.arange(len(seeds))
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(positions - 0.2, [all_results[seed]['improvement'] for seed in seeds], 0.4, label='Improvement (%)')
    axis.bar(positions + 0.2, [all_results[seed]['gap'] for seed in seeds], 0.4, label='Gap to optimum (%)')
    axis.set_xticks(positions, [f'Seed {seed}' for seed in seeds])
    axis.legend()
    axis.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_parameter_space(all_results, seeds):
    """Plot parameter space exploration.

    Args:
        all_results: Dictionary of results keyed by seed
        seeds: List of seed values
    """
    best_vector = all_results[seeds[0]]['best_vector']
    param_bounds = {
        name: (minimum, maximum)
        for name, (minimum, maximum, _) in best_vector.parameter_definitions.items()
    }
    param_names = list(param_bounds)

    float_params = []
    int_params = []

    for param_name in param_names:
        min_val, max_val = param_bounds[param_name]
        if isinstance(min_val, float):
            float_params.append(param_name)
        else:
            int_params.append(param_name)

    # Plot float parameters
    if float_params:
        fig, ax = plt.subplots(figsize=(14, 6))
        x_pos = np.arange(len(float_params))

        for i, param_name in enumerate(float_params):
            min_val, max_val = param_bounds[param_name]

            all_values = []
            for seed in seeds:
                vector = all_results[seed]['best_vector']
                if hasattr(vector, param_name):
                    all_values.append(getattr(vector, param_name))

            if all_values:
                mean_val = np.mean(all_values)

                ax.plot([i, i], [min_val, max_val], 'k-', linewidth=3, alpha=0.3,
                        label='Search Range' if i == 0 else '')

                for val in all_values:
                    ax.scatter(i, val, s=150, marker='o', color='blue', alpha=0.6,
                               edgecolors='darkblue', linewidth=2)

                ax.scatter(i, mean_val, s=250, marker='*', color='red',
                          edgecolors='darkred', linewidth=2, label='Mean' if i == 0 else '',
                          zorder=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(float_params, rotation=45, ha='right', fontweight='bold')
        ax.set_ylabel('Parameter Value', fontsize=11, fontweight='bold')
        ax.set_title('Float Parameters: Search Range vs Evolved Values', fontsize=12,
                    fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    # Plot int parameters
    if int_params:
        fig, ax = plt.subplots(figsize=(14, 6))
        x_pos = np.arange(len(int_params))

        for i, param_name in enumerate(int_params):
            min_val, max_val = param_bounds[param_name]

            all_values = []
            for seed in seeds:
                vector = all_results[seed]['best_vector']
                if hasattr(vector, param_name):
                    all_values.append(int(getattr(vector, param_name)))

            if all_values:
                mean_val = np.mean(all_values)

                ax.plot([i, i], [min_val, max_val], 'k-', linewidth=3, alpha=0.3,
                        label='Search Range' if i == 0 else '')

                for val in all_values:
                    ax.scatter(i, val, s=150, marker='o', color='green', alpha=0.6,
                               edgecolors='darkgreen', linewidth=2)

                ax.scatter(i, mean_val, s=250, marker='*', color='red',
                          edgecolors='darkred', linewidth=2, label='Mean' if i == 0 else '',
                          zorder=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(int_params, rotation=45, ha='right', fontweight='bold')
        ax.set_ylabel('Parameter Value', fontsize=11, fontweight='bold')
        ax.set_title('Integer Parameters: Search Range vs Evolved Values', fontsize=12,
                    fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    print('✓ Parameter space visualization complete')
