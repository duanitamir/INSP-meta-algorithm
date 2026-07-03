"""GA with real-time monitoring hooks integrated into evolution loop."""

from typing import Optional, List
from src.meta.core.meta_algorithm_ga import MetaAlgorithmGA
from src.meta.core.canonical_vector import CanonicalVector
from src.graph.graph_manager import GraphManager
from src.monitoring.ga_progress_monitor import ExperimentProgressTracker


class MonitoredMetaAlgorithmGA(MetaAlgorithmGA):
    """MetaAlgorithmGA with real-time monitoring hooks (minimal overhead)."""

    def __init__(self, *args, tracker: Optional[ExperimentProgressTracker] = None, **kwargs):
        """Initialize with optional monitoring tracker."""
        super().__init__(*args, **kwargs)
        self.tracker = tracker

    def evolve(self, graph: GraphManager) -> tuple[CanonicalVector, List[float]]:
        """Evolve population with real-time monitoring hooks."""
        if self.tracker:
            self.tracker.start_generation(0)
        best_vector, fitness_history = super().evolve(graph)
        if self.tracker:
            self.tracker.finalize()
        return best_vector, fitness_history
