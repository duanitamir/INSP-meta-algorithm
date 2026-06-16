from typing import Dict
from src.graph.graph_manager import GraphManager
from src.state.state_store import StateStore
from src.communication.message_queue import MessageQueue
from src.metrics.metrics_collector import MetricsCollector
from src.utils.types import RoundNumber
from src.simulation.config import SimulationConfig
from src.simulation.algorithm_context import AlgorithmContext


class Scheduler:
    """Manages round-based synchronous execution."""

    def __init__(
        self,
        graph: GraphManager,
        algorithm_or_config=None,
        config: SimulationConfig | None = None,
    ):
        self.graph = graph

        if algorithm_or_config is None:
            self.algorithm = None
            self.config = config or SimulationConfig()
        elif isinstance(algorithm_or_config, SimulationConfig):
            # Old API: Scheduler(graph, config)
            self.algorithm = None
            self.config = algorithm_or_config
        else:
            # New API: Scheduler(graph, algorithm, config)
            self.algorithm = algorithm_or_config
            self.config = config or SimulationConfig()
        self.state_store = StateStore(graph)
        self.message_queue = MessageQueue(graph)
        self.metrics = MetricsCollector()
        self._current_round = RoundNumber(0)
        self._is_running = False
        self._terminated = False
        self._termination_reason: str | None = None
        self._final_matching: Dict[int, int] = {}

    @property
    def current_round(self) -> RoundNumber:
        """Get current round number."""
        return self._current_round

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running

    @property
    def is_terminated(self) -> bool:
        """Check if scheduler has terminated."""
        return self._terminated

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.state_store = StateStore(self.graph)
        self.message_queue = MessageQueue(self.graph)
        self.metrics = MetricsCollector()
        self._current_round = RoundNumber(0)
        self._is_running = False
        self._terminated = False
        self._termination_reason = None
        self._final_matching = {}
        if self.algorithm is not None:
            self.algorithm.initialize_state(self.state_store, self.graph)

    def initialize(self) -> None:
        """Initialize scheduler for execution."""
        self._is_running = True
        self._terminated = False
        self._termination_reason = None
        self._current_round = RoundNumber(0)
        if self.algorithm is not None:
            self.algorithm.initialize_state(self.state_store, self.graph)

    def execute_round(self) -> bool:
        """
        Execute one round.

        Returns:
            True if simulation should continue, False if terminated.
        """
        if not self._is_running:
            raise RuntimeError("Scheduler not running")

        if self._terminated:
            return False

        if self._current_round >= self.config.max_rounds:
            self._terminate("max_rounds_exceeded")
            return False

        # Execute algorithm for this round if provided
        if self.algorithm:
            self._execute_algorithm_round()

        messages_sent = self.message_queue.total_messages_sent()

        self.metrics.record_round(
            round_num=self._current_round,
            messages_sent=messages_sent,
        )

        self._current_round = RoundNumber(self._current_round + 1)

        if self.config.max_messages and messages_sent > self.config.max_messages:
            self._terminate("max_messages_exceeded")
            return False

        if self.config.collect_snapshots:
            self.state_store.create_snapshot(self._current_round)

        return True

    def _execute_algorithm_round(self) -> None:
        """Execute one round of the algorithm."""
        context = AlgorithmContext(self.graph, self.state_store, self._current_round)
        new_states = {}
        all_messages = []

        for node_id in self.graph.vertices():
            node_state = self.state_store.get_node_state(node_id)
            messages = self.message_queue.get_messages(node_id)

            new_state, out_messages = self.algorithm.node_behavior(
                node_id,
                node_state,
                messages,
                context,
            )

            new_states[node_id] = new_state
            all_messages.extend(out_messages)

        # Update all states atomically
        self.state_store.update_all_states(new_states)

        # Queue all messages for next round
        self.message_queue.send_batch(all_messages)

    def check_no_progress(self) -> bool:
        """Check if there's no progress (no messages sent)."""
        return self.message_queue.total_messages_sent() == 0

    def _terminate(self, reason: str) -> None:
        """Terminate execution."""
        self._terminated = True
        self._termination_reason = reason
        self._is_running = False

        # Extract final matching if algorithm provided
        if self.algorithm:
            self._final_matching = self.algorithm.extract_matching(
                self.state_store,
                self.graph,
            )

        meta_state = self.state_store.get_meta_state()
        updated_meta = meta_state.with_convergence(reason)
        updated_meta.final_matching = self._final_matching
        self.state_store.update_meta_state(updated_meta)

    def run_until_termination(
        self,
        termination_callback=None,
    ) -> int:
        """
        Run scheduler until termination or max rounds.

        Args:
            termination_callback: Optional callback to check for termination.
                Should return (should_terminate, reason) tuple.

        Returns:
            Number of rounds executed.
        """
        self.initialize()

        while self.execute_round():
            # Check algorithm termination if algorithm provided
            if self.algorithm:
                should_terminate, reason = self.algorithm.check_termination(
                    self.state_store,
                    self._current_round,
                    self.message_queue.total_messages_sent(),
                )
                if should_terminate:
                    self._terminate(reason or "algorithm_terminated")
                    break

            # Check custom termination if callback provided
            if termination_callback:
                should_terminate, reason = termination_callback(
                    self.state_store,
                    self._current_round,
                    self.message_queue.total_messages_sent(),
                )
                if should_terminate:
                    self._terminate(reason or "callback_terminated")
                    break

        return self._current_round

    @property
    def final_matching(self) -> Dict[int, int]:
        """Get final matching from last simulation."""
        return self._final_matching.copy()
