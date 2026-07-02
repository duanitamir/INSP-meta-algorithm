"""Real-time progress monitoring for GA experiments with dashboard display.

Provides live tracking of:
- Generation progress with visual progress bar
- Node processing status and completion
- Thread activity and timing
- Fitness metrics and improvements
- Estimated time remaining
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from threading import Lock
import sys


@dataclass
class ThreadInfo:
    """Track info about a single worker thread."""
    thread_id: int
    current_node_id: Optional[int] = None
    algorithm: Optional[str] = None
    start_time: Optional[float] = None
    nodes_processed: int = 0
    total_time: float = 0.0

    def get_elapsed_time(self) -> float:
        """Get elapsed time for current task."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_avg_node_time(self) -> float:
        """Get average time per node."""
        if self.nodes_processed == 0:
            return 0.0
        return self.total_time / self.nodes_processed


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""
    generation: int
    start_time: float
    end_time: Optional[float] = None
    best_fitness: float = 0.0
    population_size: int = 0
    nodes_evaluated: int = 0
    improvement: float = 0.0

    def get_duration(self) -> float:
        """Get generation duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class GAProgressMonitor:
    """Real-time progress monitor for GA experiments with live dashboard.

    Tracks and displays:
    - Generation progress (x/y with progress bar)
    - Node processing progress (a/b with progress bar)
    - Thread activity and timing
    - Fitness history and improvements
    - Estimated time remaining
    """

    def __init__(self,
                 total_generations: int,
                 total_nodes: int,
                 population_size: int,
                 num_threads: int = 8,
                 refresh_interval: float = 2.0):
        """Initialize progress monitor.

        Args:
            total_generations: Total number of GA generations to run
            total_nodes: Total number of nodes in graph
            population_size: Size of GA population
            num_threads: Number of worker threads
            refresh_interval: Seconds between dashboard refreshes
        """
        self.total_generations = total_generations
        self.total_nodes = total_nodes
        self.population_size = population_size
        self.num_threads = num_threads
        self.refresh_interval = refresh_interval

        # Locks for thread-safe updates
        self.lock = Lock()

        # State tracking
        self.current_generation = 0
        self.nodes_processed_in_generation = 0
        self.start_time = time.time()
        self.experiment_start_time = self.start_time

        # Thread tracking
        self.threads: Dict[int, ThreadInfo] = {
            i: ThreadInfo(thread_id=i) for i in range(num_threads)
        }

        # Fitness tracking
        self.fitness_history: List[float] = []
        self.generation_metrics: List[GenerationMetrics] = []

        # Callbacks
        self.callbacks: List[Callable] = []

    def start_generation(self, generation: int) -> None:
        """Signal start of a new generation."""
        with self.lock:
            self.current_generation = generation
            self.nodes_processed_in_generation = 0
            self.start_time = time.time()

            metrics = GenerationMetrics(
                generation=generation,
                start_time=time.time(),
                population_size=self.population_size
            )
            self.generation_metrics.append(metrics)

    def update_thread_start(self, thread_id: int, node_id: int, algorithm: str) -> None:
        """Signal that a thread started processing a node."""
        with self.lock:
            if thread_id in self.threads:
                self.threads[thread_id].current_node_id = node_id
                self.threads[thread_id].algorithm = algorithm
                self.threads[thread_id].start_time = time.time()

    def update_thread_complete(self, thread_id: int) -> None:
        """Signal that a thread completed processing a node."""
        with self.lock:
            if thread_id in self.threads:
                elapsed = self.threads[thread_id].get_elapsed_time()
                self.threads[thread_id].nodes_processed += 1
                self.threads[thread_id].total_time += elapsed
                self.threads[thread_id].current_node_id = None
                self.threads[thread_id].algorithm = None
                self.threads[thread_id].start_time = None

                self.nodes_processed_in_generation += 1

    def update_fitness(self, generation: int, fitness: float) -> None:
        """Update fitness for a generation."""
        with self.lock:
            self.fitness_history.append(fitness)
            if self.generation_metrics and self.generation_metrics[-1].generation == generation:
                metric = self.generation_metrics[-1]
                metric.best_fitness = fitness
                if len(self.fitness_history) > 1:
                    metric.improvement = fitness - self.fitness_history[-2]
                metric.end_time = time.time()

    def register_callback(self, callback: Callable[['GAProgressMonitor'], None]) -> None:
        """Register callback to be called on dashboard refresh."""
        self.callbacks.append(callback)

    def get_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Generate a progress bar string.

        Args:
            current: Current progress
            total: Total items
            width: Width of progress bar

        Returns:
            Progress bar string like "████░░░░░░ 40%"
        """
        if total == 0:
            return "█" * width + " 0%"

        percent = current / total
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)
        pct = f"{percent * 100:.1f}%"
        return f"{bar} {pct}"

    def get_time_estimate_remaining(self) -> Optional[str]:
        """Estimate time remaining for experiment."""
        if self.current_generation < 1 or len(self.generation_metrics) < 1:
            return None

        avg_gen_time = sum(m.get_duration() for m in self.generation_metrics) / len(self.generation_metrics)
        remaining_generations = self.total_generations - self.current_generation
        estimated_seconds = avg_gen_time * remaining_generations

        return self._format_time(estimated_seconds)

    def get_elapsed_time(self) -> str:
        """Get elapsed time since experiment start."""
        elapsed = time.time() - self.experiment_start_time
        return self._format_time(elapsed)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)}h {int(minutes)}m"

    def refresh_dashboard(self) -> None:
        """Refresh and display the dashboard."""
        with self.lock:
            self._print_dashboard()

        # Call registered callbacks
        for callback in self.callbacks:
            try:
                callback(self)
            except Exception:
                pass

    def _print_dashboard(self) -> None:
        """Print the dashboard to terminal with in-place updates (no log spam)."""
        # Use ANSI codes to move cursor and clear lines (no full screen clear)
        # This prevents creating huge logs while keeping dashboard in place

        dashboard_lines = self._get_dashboard_lines()

        # Move to home position and clear downward
        print("\033[H\033[J", end="", flush=True)

        # Print all dashboard content at once
        for line in dashboard_lines:
            print(line, flush=True)

        sys.stdout.flush()

    def _get_dashboard_lines(self) -> list:
        """Generate dashboard lines (returns list instead of printing directly)."""
        lines = []

        lines.append("\n" + "=" * 80)
        lines.append("  GA EXPERIMENT REAL-TIME DASHBOARD")
        lines.append("=" * 80 + "\n")

        # Generation progress
        gen_bar = self.get_progress_bar(self.current_generation, self.total_generations, width=40)
        lines.append(f"📊 Generation Progress: {self.current_generation}/{self.total_generations}")
        lines.append(f"   {gen_bar}\n")

        # Node progress
        node_bar = self.get_progress_bar(self.nodes_processed_in_generation,
                                         self.total_nodes * self.population_size, width=40)
        lines.append(f"🔄 Node Processing: {self.nodes_processed_in_generation}/{self.total_nodes * self.population_size}")
        lines.append(f"   {node_bar}\n")

        # Fitness metrics
        if self.fitness_history:
            lines.append(f"📈 Fitness Metrics:")
            lines.append(f"   Current best:       {self.fitness_history[-1]:.2f}")
            if len(self.fitness_history) > 1:
                improvement = self.fitness_history[-1] - self.fitness_history[-2]
                lines.append(f"   Last improvement:   {improvement:+.2f}")
            lines.append("")

        # Thread activity
        lines.append(f"👨‍💻 Thread Activity ({self.num_threads} threads):")
        for thread_id, info in self.threads.items():
            if info.current_node_id is not None:
                elapsed = info.get_elapsed_time()
                status = f"🟢 Node {info.current_node_id} ({info.algorithm}) - {elapsed:.2f}s"
            else:
                status = f"⚪ Idle (processed {info.nodes_processed} nodes, avg {info.get_avg_node_time():.3f}s)"
            lines.append(f"   Thread {thread_id}: {status}")
        lines.append("")

        # Time estimates
        lines.append(f"⏱️  Timing:")
        lines.append(f"   Elapsed:           {self.get_elapsed_time()}")
        if self.generation_metrics:
            lines.append(f"   Avg gen time:      {self._format_time(sum(m.get_duration() for m in self.generation_metrics) / len(self.generation_metrics))}")
        est_remaining = self.get_time_estimate_remaining()
        if est_remaining:
            lines.append(f"   Estimated remaining: {est_remaining}")
        lines.append("")

        # Generation history
        if self.generation_metrics:
            lines.append(f"📊 Recent Generations:")
            for metric in self.generation_metrics[-5:]:  # Show last 5 generations
                lines.append(f"   Gen {metric.generation}: fitness={metric.best_fitness:.2f}, "
                      f"time={self._format_time(metric.get_duration())}, "
                      f"improvement={metric.improvement:+.2f}")
            lines.append("")

        lines.append("=" * 80 + "\n")

        return lines

        print("\n" + "=" * 80)
        print("  GA EXPERIMENT REAL-TIME DASHBOARD")
        print("=" * 80 + "\n")

        # Generation progress
        gen_bar = self.get_progress_bar(self.current_generation, self.total_generations, width=40)
        print(f"📊 Generation Progress: {self.current_generation}/{self.total_generations}")
        print(f"   {gen_bar}\n")

        # Node progress
        node_bar = self.get_progress_bar(self.nodes_processed_in_generation,
                                         self.total_nodes * self.population_size, width=40)
        print(f"🔄 Node Processing: {self.nodes_processed_in_generation}/{self.total_nodes * self.population_size}")
        print(f"   {node_bar}\n")

        # Fitness metrics
        if self.fitness_history:
            print(f"📈 Fitness Metrics:")
            print(f"   Current best:       {self.fitness_history[-1]:.2f}")
            print(f"   Generation best:    {self.fitness_history[-1]:.2f}")
            if len(self.fitness_history) > 1:
                improvement = self.fitness_history[-1] - self.fitness_history[-2]
                print(f"   Last improvement:   {improvement:+.2f}")
            print()

        # Thread activity
        print(f"👨‍💻 Thread Activity ({self.num_threads} threads):")
        for thread_id, info in self.threads.items():
            if info.current_node_id is not None:
                elapsed = info.get_elapsed_time()
                status = f"🟢 Node {info.current_node_id} ({info.algorithm}) - {elapsed:.2f}s"
            else:
                status = f"⚪ Idle (processed {info.nodes_processed} nodes, avg {info.get_avg_node_time():.3f}s)"
            print(f"   Thread {thread_id}: {status}")
        print()

        # Time estimates
        print(f"⏱️  Timing:")
        print(f"   Elapsed:           {self.get_elapsed_time()}")
        if self.generation_metrics:
            print(f"   Avg gen time:      {self._format_time(sum(m.get_duration() for m in self.generation_metrics) / len(self.generation_metrics))}")
        est_remaining = self.get_time_estimate_remaining()
        if est_remaining:
            print(f"   Estimated remaining: {est_remaining}")
        print()

        # Generation history
        if self.generation_metrics:
            print(f"📊 Recent Generations:")
            for metric in self.generation_metrics[-5:]:  # Show last 5 generations
                print(f"   Gen {metric.generation}: fitness={metric.best_fitness:.2f}, "
                      f"time={self._format_time(metric.get_duration())}, "
                      f"improvement={metric.improvement:+.2f}")
            print()

        print("=" * 80 + "\n")
        sys.stdout.flush()


class ExperimentProgressTracker:
    """High-level tracker for full GA experiment execution.

    Simplifies integration with GA algorithms by handling all monitoring calls.
    """

    def __init__(self,
                 total_generations: int,
                 total_nodes: int,
                 population_size: int,
                 num_threads: int = 8):
        """Initialize experiment tracker.

        Args:
            total_generations: Total GA generations
            total_nodes: Total graph nodes
            population_size: GA population size
            num_threads: Number of worker threads
        """
        self.monitor = GAProgressMonitor(
            total_generations=total_generations,
            total_nodes=total_nodes,
            population_size=population_size,
            num_threads=num_threads
        )
        self.last_refresh = time.time()

    def start_generation(self, generation: int) -> None:
        """Start tracking a new generation."""
        self.monitor.start_generation(generation)

    def thread_processing_node(self, thread_id: int, node_id: int, algorithm: str) -> None:
        """Signal thread started processing node."""
        self.monitor.update_thread_start(thread_id, node_id, algorithm)
        self._refresh_if_needed()

    def thread_completed_node(self, thread_id: int) -> None:
        """Signal thread completed processing node."""
        self.monitor.update_thread_complete(thread_id)
        self._refresh_if_needed()

    def generation_complete(self, generation: int, fitness: float) -> None:
        """Signal generation complete with fitness result."""
        self.monitor.update_fitness(generation, fitness)
        self._refresh_if_needed(force=True)

    def _refresh_if_needed(self, force: bool = False) -> None:
        """Refresh dashboard if enough time has passed."""
        now = time.time()
        if force or (now - self.last_refresh) >= self.monitor.refresh_interval:
            self.monitor.refresh_dashboard()
            self.last_refresh = now

    def finalize(self) -> None:
        """Display final dashboard."""
        self.monitor.refresh_dashboard()
        print("\n✅ Experiment Complete!\n")
