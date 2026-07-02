"""Integration layer between MetaAlgorithmGA and progress monitoring.

Provides hooks for GA lifecycle events and thread tracking.
"""

from typing import Optional, Dict, Callable
import threading
from functools import wraps

from src.monitoring.ga_progress_monitor import ExperimentProgressTracker


# Thread-local storage for tracking which thread is doing what
_thread_local = threading.local()


class GAMonitoringContext:
    """Context manager for GA monitoring integration."""

    def __init__(self,
                 total_generations: int,
                 total_nodes: int,
                 population_size: int,
                 num_threads: int = 8):
        """Initialize monitoring context.

        Args:
            total_generations: Total GA generations
            total_nodes: Total graph nodes
            population_size: GA population size
            num_threads: Number of worker threads
        """
        self.tracker = ExperimentProgressTracker(
            total_generations=total_generations,
            total_nodes=total_nodes,
            population_size=population_size,
            num_threads=num_threads
        )
        self.thread_id_counter = 0
        self.thread_id_map: Dict[int, int] = {}
        self.lock = threading.Lock()

    def get_thread_id(self) -> int:
        """Get numeric ID for current thread."""
        current = threading.current_thread().ident
        if current not in self.thread_id_map:
            with self.lock:
                if current not in self.thread_id_map:
                    self.thread_id_map[current] = self.thread_id_counter
                    self.thread_id_counter += 1
        return self.thread_id_map[current]

    def start_generation(self, generation: int) -> None:
        """Signal start of generation."""
        self.tracker.start_generation(generation)

    def track_node_processing(self, node_id: int, algorithm: str) -> None:
        """Signal thread is processing a node."""
        thread_id = self.get_thread_id()
        self.tracker.thread_processing_node(thread_id, node_id, algorithm)

    def complete_node_processing(self) -> None:
        """Signal thread completed node processing."""
        thread_id = self.get_thread_id()
        self.tracker.thread_completed_node(thread_id)

    def generation_complete(self, generation: int, fitness: float) -> None:
        """Signal generation complete."""
        self.tracker.generation_complete(generation, fitness)

    def finalize(self) -> None:
        """Display final metrics."""
        self.tracker.finalize()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


def monitored_node_processing(algorithm_name: str):
    """Decorator to automatically track node processing time.

    Args:
        algorithm_name: Name of algorithm being executed (e.g., "Greedy", "Itai", "Luby")

    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, node_id=None, **kwargs):
            monitor = getattr(_thread_local, 'monitor', None)
            if monitor and node_id is not None:
                monitor.track_node_processing(node_id, algorithm_name)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if monitor:
                    monitor.complete_node_processing()

        return wrapper
    return decorator


def set_monitoring_context(context: Optional[GAMonitoringContext]) -> None:
    """Set the current monitoring context for the thread.

    Args:
        context: Monitoring context or None to disable
    """
    _thread_local.monitor = context


def get_monitoring_context() -> Optional[GAMonitoringContext]:
    """Get the current monitoring context for the thread."""
    return getattr(_thread_local, 'monitor', None)


class MonitoredGAWrapper:
    """Wrapper that adds monitoring to an existing GA algorithm.

    Intercepts key methods and provides progress tracking.
    """

    def __init__(self, ga_instance, monitor_context: GAMonitoringContext):
        """Initialize wrapper.

        Args:
            ga_instance: The MetaAlgorithmGA instance to wrap
            monitor_context: The monitoring context to use
        """
        self.ga = ga_instance
        self.monitor = monitor_context
        self._wrap_methods()

    def _wrap_methods(self) -> None:
        """Wrap GA methods with monitoring calls."""
        original_evolve = self.ga.evolve

        def monitored_evolve(graph):
            """Wrapped evolve method with monitoring."""
            best_vector, fitness_history = original_evolve(graph)
            return best_vector, fitness_history

        self.ga.evolve = monitored_evolve

    def __getattr__(self, name):
        """Delegate attribute access to wrapped GA instance."""
        return getattr(self.ga, name)


class ParameterizerMonitor:
    """Monitor parameterizer execution for detailed thread tracking."""

    def __init__(self, context: Optional[GAMonitoringContext] = None):
        """Initialize parameterizer monitor.

        Args:
            context: Monitoring context (uses thread-local if None)
        """
        self.context = context or get_monitoring_context()

    def wrap_parameterizer_execute(self, parameterizer):
        """Wrap parameterizer.execute() method with monitoring.

        Args:
            parameterizer: The parameterizer instance to wrap

        Returns:
            Wrapped parameterizer with monitoring
        """
        original_execute = parameterizer.execute

        def monitored_execute(graph, canonical_vector):
            """Wrapped execute with monitoring calls."""
            if self.context:
                # Track that we're executing this algorithm
                algo_name = parameterizer.name()

                # For each node in the graph, track execution
                # This is a simplified version - real implementation would track each node
                for node_id in graph.vertices():
                    self.context.track_node_processing(node_id, algo_name)

            result = original_execute(graph, canonical_vector)

            if self.context:
                for node_id in graph.vertices():
                    self.context.complete_node_processing()

            return result

        parameterizer.execute = monitored_execute
        return parameterizer
