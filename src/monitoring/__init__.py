"""Monitoring and progress tracking for GA experiments.

Provides real-time dashboard with:
- Generation progress bars
- Node processing status
- Thread activity tracking
- Fitness metrics
- Time estimates
"""

from src.monitoring.ga_progress_monitor import (
    GAProgressMonitor,
    ExperimentProgressTracker,
    ThreadInfo,
    GenerationMetrics,
)

from src.monitoring.ga_integration import (
    GAMonitoringContext,
    MonitoredGAWrapper,
    ParameterizerMonitor,
    set_monitoring_context,
    get_monitoring_context,
    monitored_node_processing,
)

from src.monitoring.monitored_ga_experiment import (
    run_ga_experiment_monitored,
)

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
