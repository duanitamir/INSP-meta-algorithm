"""Visualization utilities for GA analysis."""

import numpy as np
import matplotlib.pyplot as plt


def plot_fitness_progression(all_results, seeds, algo_names_str):
    """Plot fitness progression for each seed.

    Args:
        all_results: Dictionary of results keyed by seed
        seeds: List of seed values
        algo_names_str: String of algorithm names
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, seed in enumerate(seeds):
        r = all_results[seed]
        fitness_history_std = r['fitness_history_standard']
        fitness_history_casc = r['fitness_history_cascading']
        baseline = r['baseline']
        baseline_cascading = r['baseline_cascading']
        optimal = r['optimal']
        algo_weights = r['algo_weights']

        ax = axes[idx]
        gens = range(1, len(fitness_history_std) + 1)

        # Plot GA fitness progression
        ax.plot(
            gens,
            fitness_history_std,
            'o-',
            linewidth=2.5,
            markersize=7,
            label='GA Standard',
            color='blue',
        )
        ax.plot(
            gens,
            fitness_history_casc,
            's-',
            linewidth=2.5,
            markersize=7,
            label='GA Cascading',
            color='green',
        )

        # Plot merged baselines
        ax.axhline(
            baseline,
            linestyle='--',
            color='orange',
            linewidth=2,
            label=f'Merged Baseline (Std): {baseline:.0f}',
        )
        ax.axhline(
            baseline_cascading,
            linestyle='--',
            color='purple',
            linewidth=2,
            label=f'Merged Baseline (Casc): {baseline_cascading:.0f}',
        )

        # Plot individual algorithm baselines
        colors_algo = {'greedy': '#d62728', 'itai': '#ff7f0e', 'luby': '#2ca02c'}
        for algo_name, weight in algo_weights.items():
            color = colors_algo.get(algo_name, '#1f77b4')
            ax.axhline(
                weight,
                linestyle=':',
                color=color,
                linewidth=1.5,
                alpha=0.7,
                label=f'{algo_name.upper()}: {weight:.0f}',
            )

        # Plot optimal
        ax.axhline(
            optimal,
            linestyle='-',
            color='red',
            linewidth=2.5,
            label=f'Optimal: {optimal:.0f}',
        )

        ax.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=11, fontweight='bold')
        ax.set_title(f'Seed {seed}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(
            min(fitness_history_std + list(algo_weights.values())) * 0.95,
            optimal * 1.05,
        )

    plt.suptitle(
        f'GA Fitness Progression with Algorithm Baselines - Algorithms: {algo_names_str}',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()
    plt.show()

    print('✓ Fitness progression plot with individual algorithm baselines complete')


def plot_baseline_comparison(all_results, seeds, algo_names_str):
    """Plot baseline comparison bar chart.

    Args:
        all_results: Dictionary of results keyed by seed
        seeds: List of seed values
        algo_names_str: String of algorithm names
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, seed in enumerate(seeds):
        ax = axes[idx]
        r = all_results[seed]

        algo_weights = r['algo_weights']
        baseline = r['baseline']
        baseline_cascading = r['baseline_cascading']
        optimal = r['optimal']
        best_standard = r['best_standard']
        best_cascading = r['best_cascading']

        labels = (
            list(algo_weights.keys())
            + ['Merged\n(Std)', 'Merged\n(Casc)', 'GA\n(Std)', 'GA\n(Casc)', 'Optimal']
        )
        values = (
            list(algo_weights.values())
            + [baseline, baseline_cascading, best_standard, best_cascading, optimal]
        )

        algo_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][: len(algo_weights)]
        merged_colors = ['#d62728', '#9467bd']
        ga_colors = ['#8c564b', '#e377c2']
        optimal_color = ['#bcbd22']
        colors = algo_colors + merged_colors + ga_colors + optimal_color

        bars = ax.bar(labels, values, color=colors, alpha=0.75, edgecolor='black', linewidth=1.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.0f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold',
            )

        ax.set_ylabel('Weight', fontsize=11, fontweight='bold')
        ax.set_title(f'Seed {seed}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, optimal * 1.15)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(
        f'Baseline Comparison: Individual Algorithms + Merged + GA - {algo_names_str}',
        fontsize=13,
        fontweight='bold',
    )
    plt.tight_layout()
    plt.show()

    print('✓ Baseline comparison plot complete')


def plot_performance_metrics(all_results, seeds):
    """Plot performance metrics comparison.

    Args:
        all_results: Dictionary of results keyed by seed
        seeds: List of seed values
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x_pos = np.arange(len(seeds))
    width = 0.25

    improvements_std = [all_results[seed]['improvement'] for seed in seeds]
    gaps = [all_results[seed]['gap'] for seed in seeds]

    bars1 = ax.bar(
        x_pos - width,
        improvements_std,
        width,
        label='GA Improvement (%)',
        color='blue',
        alpha=0.7,
        edgecolor='black',
    )
    bars2 = ax.bar(
        x_pos,
        gaps,
        width,
        label='Gap to Optimal (%)',
        color='red',
        alpha=0.7,
        edgecolor='black',
    )
    bars3 = ax.bar(
        x_pos + width,
        [1.0] * len(seeds),
        width,
        label='Target (1% gap)',
        color='green',
        alpha=0.3,
        edgecolor='black',
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
            )

    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('GA Improvement vs Gap to Optimal', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Seed {s}' for s in seeds], fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    baselines = [all_results[seed]['baseline'] for seed in seeds]
    gas_std = [all_results[seed]['best_standard'] for seed in seeds]
    gas_casc = [all_results[seed]['best_cascading'] for seed in seeds]
    optima = [all_results[seed]['optimal'] for seed in seeds]

    x_pos = np.arange(len(seeds))
    width = 0.2

    ax.bar(
        x_pos - 1.5 * width,
        baselines,
        width,
        label='Baseline',
        color='orange',
        alpha=0.7,
        edgecolor='black',
    )
    ax.bar(
        x_pos - 0.5 * width,
        gas_std,
        width,
        label='GA Standard',
        color='blue',
        alpha=0.7,
        edgecolor='black',
    )
    ax.bar(
        x_pos + 0.5 * width,
        gas_casc,
        width,
        label='GA Cascading',
        color='green',
        alpha=0.7,
        edgecolor='black',
    )
    ax.bar(
        x_pos + 1.5 * width,
        optima,
        width,
        label='Optimal',
        color='red',
        alpha=0.7,
        edgecolor='black',
    )

    ax.set_ylabel('Fitness (Weight)', fontsize=11, fontweight='bold')
    ax.set_title('Fitness Comparison Across All Approaches', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Seed {s}' for s in seeds], fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print('✓ Performance metrics plot complete')


def plot_parameter_space(all_results, seeds):
    """Plot parameter space exploration.

    Args:
        all_results: Dictionary of results keyed by seed
        seeds: List of seed values
    """
    param_bounds = {
        'luby_base_probability': (0.0, 1.0),
        'luby_coeff_degree': (-1.0, 1.0),
        'luby_coeff_neighbors_unmatched': (-1.0, 1.0),
        'luby_coeff_clustering': (-1.0, 1.0),
        'luby_coeff_matched': (-1.0, 1.0),
        'luby_coeff_round': (-1.0, 1.0),
        'luby_coeff_weight': (-1.0, 1.0),
        'itai_timeout_rounds': (1, 20),
        'max_iterations': (5, 100),
        'convergence_threshold': (0.0, 0.1),
    }

    best_vector = all_results[seeds[0]]['best_vector_standard']
    param_names = [p for p in param_bounds.keys() if hasattr(best_vector, p)]

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
                vector = all_results[seed]['best_vector_standard']
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
                vector = all_results[seed]['best_vector_standard']
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
