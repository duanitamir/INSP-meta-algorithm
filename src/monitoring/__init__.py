"""Monitoring and progress tracking for GA experiments (DEPRECATED).

Monitoring moved to src/experiments/monitoring for optional experimental features.
This module provides backward compatibility.
"""

try:
    from src.experiments.monitoring.ga_progress_monitor import (
        GAProgressMonitor,
        ExperimentProgressTracker,
        ThreadInfo,
        GenerationMetrics,
    )

    from src.experiments.monitoring.ga_integration import (
        GAMonitoringContext,
        MonitoredGAWrapper,
        set_monitoring_context,
        get_monitoring_context,
        monitored_node_processing,
    )

    from src.experiments.monitoring.monitored_ga_experiment import (
        run_ga_experiment_monitored,
    )
except ImportError:
    pass

__all__ = [
    # Progress monitoring
    'GAProgressMonitor',
    'ExperimentProgressTracker',
    'ThreadInfo',
    'GenerationMetrics',

    # Integration
    'GAMonitoringContext',
    'MonitoredGAWrapper',
    'ParameterizerMonitor',
    'set_monitoring_context',
    'get_monitoring_context',
    'monitored_node_processing',

    # Experiment runner
    'run_ga_experiment_monitored',
]
